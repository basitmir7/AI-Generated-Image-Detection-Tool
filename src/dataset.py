"""
Dataset loader for real-vs-AI-generated image classification.

Expected folder layout (create this yourself after downloading a dataset
like CIFAKE or GenImage):

    data/
      train/
        real/   *.jpg / *.png
        fake/   *.jpg / *.png
      val/
        real/
        fake/
      test/
        real/
        fake/

Label convention: real = 0, fake = 1
"""

import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class RealFakeImageDataset(Dataset):
    """Binary classification dataset: 0 = real, 1 = fake (AI-generated)."""

    def __init__(self, root_dir: str, split: str, image_size: int = 224, augment: bool = None):
        """
        root_dir: path to the `data` folder described above
        split: "train", "val", or "test"
        image_size: resize target (square)
        augment: whether to apply training augmentations. Defaults to True
                 for "train" split, False otherwise.
        """
        self.root_dir = Path(root_dir) / split
        self.image_size = image_size
        self.augment = augment if augment is not None else (split == "train")

        self.samples = []  # list of (filepath, label)
        for label_name, label in [("real", 0), ("fake", 1)]:
            class_dir = self.root_dir / label_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Expected folder not found: {class_dir}. "
                    f"See dataset.py docstring for the expected layout."
                )
            for fp in class_dir.iterdir():
                if fp.suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append((str(fp), label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.root_dir}")

        self.transform = self._build_transform()

    def _build_transform(self) -> A.Compose:
        if self.augment:
            return A.Compose([
                A.RandomResizedCrop(size=(self.image_size, self.image_size), scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                # AI detectors are notoriously sensitive to compression /
                # resizing artifacts. Training with these makes the model
                # rely less on generator-specific noise fingerprints and
                # more on genuine visual signal.
                A.ImageCompression(quality_range=(60, 100), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.15),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if image is None:
            # Corrupt file safety net -- return a black image rather than
            # crashing a multi-hour training run over one bad file.
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image)
        return transformed["image"], label

    def class_counts(self) -> dict:
        counts = {"real": 0, "fake": 0}
        for _, label in self.samples:
            counts["fake" if label == 1 else "real"] += 1
        return counts
