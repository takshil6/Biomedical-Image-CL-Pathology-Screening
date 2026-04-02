"""
PyTorch Dataset and DataLoader factory for Kather 2016 histology tiles.

Pipeline:
  1. Scan folder structure → collect (image_path, original_class) pairs
  2. Map 8 tissue classes → 4 screening categories (Tumor, Stroma, Immune, Other)
  3. Stratified split into train / val / test (70 / 15 / 15)
  4. Training set gets heavy augmentation; val/test get deterministic transforms
  5. WeightedRandomSampler compensates for class imbalance in training

Usage:
    from src.dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
"""

import os
from collections import Counter
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import (
    BATCH_SIZE,
    CLASS_MAPPING,
    CLASS_NAMES,
    DATASET_DIR,
    NUM_WORKERS,
    SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    eval_transforms,
    train_transforms,
)


# ── Dataset ──────────────────────────────────────────────────────────────────

class PathologyDataset(Dataset):
    """
    Loads Kather histology tiles and maps 8 original classes → 4 categories.

    Folder layout expected:
        DATASET_DIR/
            01_TUMOR/    *.tif
            02_STROMA/   *.tif
            ...
            08_EMPTY/    *.tif

    Each image is 150x150 px, RGB, .tif format.
    """

    def __init__(self, root: str = DATASET_DIR, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []   # list of (path, mapped_label)
        self.labels = []    # just mapped labels (for split stratification)

        for folder_name, mapped_label in CLASS_MAPPING.items():
            folder_path = os.path.join(root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                    self.samples.append(
                        (os.path.join(folder_path, fname), mapped_label)
                    )
                    self.labels.append(mapped_label)

        self.labels = np.array(self.labels)

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found in {root}. Run `python -m src.download_data` first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Split helpers ────────────────────────────────────────────────────────────

def stratified_split(dataset: PathologyDataset):
    """
    Two-pass stratified split → train / val / test index arrays.

    First split:  train+val vs test  (85 / 15)
    Second split: train vs val       (70 / 15 of total → ~82.4 / 17.6 of trainval)
    """
    indices = np.arange(len(dataset))
    labels = dataset.labels

    # Split off test set
    trainval_idx, test_idx = train_test_split(
        indices, test_size=TEST_RATIO, stratify=labels, random_state=SEED
    )

    # Split trainval into train and val
    # val is 15% of total → 15/85 ≈ 17.6% of trainval
    val_fraction = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_fraction,
        stratify=labels[trainval_idx],
        random_state=SEED,
    )

    return train_idx, val_idx, test_idx


# ── Weighted sampler ─────────────────────────────────────────────────────────

def make_weighted_sampler(dataset: PathologyDataset, indices: np.ndarray) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so each mini-batch sees roughly equal
    representation from all 4 classes, regardless of dataset imbalance.
    """
    subset_labels = dataset.labels[indices]
    class_counts = Counter(subset_labels)
    # Weight per class = 1 / count → rarer classes get higher weight
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = np.array([class_weights[label] for label in subset_labels])
    sample_weights = torch.from_numpy(sample_weights).double()

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ── Transform-aware subset ───────────────────────────────────────────────────

class TransformSubset(Dataset):
    """
    Subset that applies its own transform instead of the parent dataset's.

    Note: we inherit from Dataset (not Subset) to avoid Subset.__getitems__
    bypassing our transform in newer PyTorch versions (batch-fetch optimization).
    """

    def __init__(self, dataset: PathologyDataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Public API ───────────────────────────────────────────────────────────────

def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders with all the bells and whistles:
      - Stratified splits
      - Augmentation on train only
      - WeightedRandomSampler on train
      - Deterministic eval transforms on val/test

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Load full dataset (no transform yet — TransformSubset handles it)
    full_dataset = PathologyDataset(root=DATASET_DIR, transform=None)

    # Stratified split
    train_idx, val_idx, test_idx = stratified_split(full_dataset)

    # Wrap with per-split transforms
    train_set = TransformSubset(full_dataset, train_idx, transform=train_transforms)
    val_set = TransformSubset(full_dataset, val_idx, transform=eval_transforms)
    test_set = TransformSubset(full_dataset, test_idx, transform=eval_transforms)

    # Weighted sampler for training
    sampler = make_weighted_sampler(full_dataset, train_idx)

    # Print split stats
    print(f"Dataset splits — Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    for i, name in enumerate(CLASS_NAMES):
        tr = np.sum(full_dataset.labels[train_idx] == i)
        va = np.sum(full_dataset.labels[val_idx] == i)
        te = np.sum(full_dataset.labels[test_idx] == i)
        print(f"  {name:>8s}: train={tr:4d}  val={va:4d}  test={te:4d}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,       # weighted sampling → no shuffle needed
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    train_dl, val_dl, test_dl = get_dataloaders()
    images, labels = next(iter(train_dl))
    print(f"\nBatch shape: {images.shape}")
    print(f"Label distribution in batch: {Counter(labels.numpy())}")
    print(f"Label->class: { {v: CLASS_NAMES[v] for v in sorted(set(labels.numpy()))} }")
