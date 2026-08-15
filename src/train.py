"""
Training script for the real-vs-AI-generated image classifier.

Designed to survive interruptions: saves a checkpoint after every epoch
(plus an optional mid-epoch save interval), and can resume from the last
checkpoint automatically with --resume.

Usage:
    python src/train.py --data-dir data --backbone resnet50 --epochs 15

Resume after an interruption:
    python src/train.py --data-dir data --backbone resnet50 --epochs 15 --resume
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import RealFakeImageDataset
from model import build_model, set_backbone_trainable


def get_device() -> torch.device:
    """cuda -> mps -> cpu, in order of preference. Same script works on the
    RTX PC or the M2 Mac without any changes."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_val_loss: float, history: list):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_val_loss": best_val_loss,
        "history": history,
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["best_val_loss"], ckpt.get("history", [])


def run_epoch(model, loader, criterion, optimizer, device, use_amp, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    # amp autocast is well-supported on CUDA; MPS support is newer/less
    # stable, so we only enable it on CUDA and run full precision elsewhere.
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="train" if train else "val", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=total_loss / total, acc=correct / total)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=2, help="epochs with backbone frozen before full fine-tuning")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--finetune-lr", type=float, default=3e-5, help="lower LR used once backbone is unfrozen")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoints/last.pt if it exists")
    parser.add_argument("--amp", action="store_true", default=True, help="mixed precision (CUDA only, ignored on MPS/CPU)")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt_path = ckpt_dir / "last.pt"
    best_ckpt_path = ckpt_dir / "best.pt"

    train_ds = RealFakeImageDataset(args.data_dir, "train", image_size=args.image_size)
    val_ds = RealFakeImageDataset(args.data_dir, "val", image_size=args.image_size)
    print(f"Train samples: {len(train_ds)} {train_ds.class_counts()}")
    print(f"Val samples:   {len(val_ds)} {val_ds.class_counts()}")

    # pin_memory only helps on CUDA
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)

    model = build_model(args.backbone, pretrained=True, freeze_backbone=(args.warmup_epochs > 0)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_loss = float("inf")
    history = []

    if args.resume and last_ckpt_path.exists():
        start_epoch, best_val_loss, history = load_checkpoint(last_ckpt_path, model, optimizer, scheduler, device)
        start_epoch += 1
        print(f"Resumed from checkpoint at epoch {start_epoch} (best_val_loss so far: {best_val_loss:.4f})")

    writer = SummaryWriter(log_dir="runs")

    for epoch in range(start_epoch, args.epochs):
        # Unfreeze the backbone after warmup, drop to a lower LR for fine-tuning.
        if epoch == args.warmup_epochs and args.warmup_epochs > 0:
            print(f"Epoch {epoch}: unfreezing backbone, switching to fine-tune LR {args.finetune_lr}")
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)

        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, args.amp, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, args.amp, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        # Always save "last" so a crash/power loss/interrupt never costs more
        # than the current epoch's progress.
        save_checkpoint(last_ckpt_path, model, optimizer, scheduler, epoch, best_val_loss, history)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, epoch, best_val_loss, history)
            print(f"  -> new best model saved (val_loss={val_loss:.4f})")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    writer.close()
    print("Training complete. Best model saved at", best_ckpt_path)


if __name__ == "__main__":
    main()
