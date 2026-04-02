"""
Training loop for pathology image classifiers.

Supports:
  - Mixed-precision (torch.cuda.amp) for VRAM efficiency
  - Per-epoch train/val loss, accuracy, and macro-F1 logging
  - Best-model checkpointing on validation F1
  - Training curve plots saved to the experiment directory

Usage:
    python -m src.train --model baseline     # Phase 2
"""

import argparse
import json
import os
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.config import (
    BASELINE_EPOCHS,
    BASELINE_LR,
    CLASS_NAMES,
    DEVICE,
    EXPERIMENTS_DIR,
    NUM_CLASSES,
    SEED,
    WEIGHT_DECAY,
)
from src.dataset import get_dataloaders
from src.model import BaselineCNN


# ── Reproducibility ──────────────────────────────────────────────────────────

def seed_everything(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Class weights for loss function ──────────────────────────────────────────

def compute_class_weights(loader) -> torch.Tensor:
    """Inverse-frequency weights from training labels."""
    counts = Counter()
    for _, labels in loader:
        counts.update(labels.numpy())
    total = sum(counts.values())
    weights = torch.zeros(NUM_CLASSES)
    for cls, count in counts.items():
        weights[cls] = total / (NUM_CLASSES * count)
    return weights


# ── Single epoch helpers ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


# ── Plot training curves ─────────────────────────────────────────────────────

def save_training_curves(history: dict, save_dir: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1
    axes[2].plot(epochs, history["train_f1"], label="Train")
    axes[2].plot(epochs, history["val_f1"], label="Val")
    axes[2].set_title("Macro F1-Score")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


# ── Main training driver ─────────────────────────────────────────────────────

def train_baseline():
    seed_everything()
    save_dir = os.path.join(EXPERIMENTS_DIR, "baseline_cnn")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Experiment dir: {save_dir}\n")

    # Data
    train_loader, val_loader, _ = get_dataloaders()

    # Model
    model = BaselineCNN().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"BaselineCNN params: {total_params:,}\n")

    # Loss with class weights
    class_weights = compute_class_weights(train_loader).to(DEVICE)
    print(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=BASELINE_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # Training loop
    history = {k: [] for k in [
        "train_loss", "train_acc", "train_f1",
        "val_loss", "val_acc", "val_f1",
    ]}
    best_val_f1 = 0.0
    start_time = time.time()

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>10} {'Train F1':>10} | "
          f"{'Val Loss':>10} {'Val Acc':>10} {'Val F1':>10} | {'LR':>10}")
    print("-" * 95)

    for epoch in range(1, BASELINE_EPOCHS + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE
        )
        va_loss, va_acc, va_f1 = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(va_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)

        print(f"{epoch:>5} | {tr_loss:>10.4f} {tr_acc:>10.4f} {tr_f1:>10.4f} | "
              f"{va_loss:>10.4f} {va_acc:>10.4f} {va_f1:>10.4f} | {current_lr:>10.6f}")

        # Checkpoint on best val F1
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": va_f1,
                "val_acc": va_acc,
            }, os.path.join(save_dir, "best_model.pth"))

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} min")
    print(f"Best val F1: {best_val_f1:.4f}")

    # Save curves and metrics
    save_training_curves(history, save_dir)
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved: training_curves.png, history.json, best_model.pth -> {save_dir}")
    return history, best_val_f1


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pathology classifier")
    parser.add_argument(
        "--model", choices=["baseline"], default="baseline",
        help="Which model to train (default: baseline)",
    )
    args = parser.parse_args()

    if args.model == "baseline":
        train_baseline()
