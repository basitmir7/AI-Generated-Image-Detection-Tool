"""
Evaluate a trained checkpoint on a test set. Reports accuracy, precision,
recall, F1, and the confusion matrix, with special attention to false
positive rate (real photos wrongly flagged as AI) since that's the more
costly error for this project's use case.

Usage:
    python src/evaluate.py --checkpoint checkpoints/best.pt --data-dir data --split test

To test generalization on a specific generator's outputs only, point
--data-dir at a separate folder structured the same way (test/real, test/fake)
containing images from a generator NOT seen during training.
"""

import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from dataset import RealFakeImageDataset
from model import build_model
from train import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    ds = RealFakeImageDataset(args.data_dir, args.split, image_size=args.image_size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Evaluating on {len(ds)} samples: {ds.class_counts()}")

    model = build_model(args.backbone, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    print("\n--- Classification report ---")
    print(classification_report(all_labels, all_preds, target_names=["real", "fake"]))

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("--- Confusion matrix ---")
    print(f"                predicted_real  predicted_fake")
    print(f"actual_real     {tn:>14}  {fp:>14}")
    print(f"actual_fake     {fn:>14}  {tp:>14}")
    print(f"\nFalse positive rate (real flagged as fake): {false_positive_rate:.4f}")

    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"ROC-AUC: {auc:.4f}")
    except ValueError:
        pass  # only one class present in this split


if __name__ == "__main__":
    main()
