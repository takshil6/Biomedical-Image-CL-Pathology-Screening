"""
Evaluation script for pathology image classifiers.

Generates on the held-out test set:
  - Classification report (console + CSV)
  - Confusion matrix heatmap
  - Multi-class AUC-ROC curves (one-vs-rest)
  - Precision-Recall curves per class
  - Baseline vs ResNet-50 comparison table (console + CSV)

Usage:
    python -m src.evaluate --model baseline
    python -m src.evaluate --model resnet
    python -m src.evaluate --model both      # runs both and prints comparison
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from src.config import CLASS_NAMES, DEVICE, EXPERIMENTS_DIR, NUM_CLASSES
from src.dataset import get_baseline_dataloaders, get_dataloaders
from src.model import BaselineCNN, BaselineCNNSimple, ResNet50Classifier


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """Return (labels, preds, probs) arrays over the full loader."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.append(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
    )


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, title, save_path):
    cm = confusion_matrix(labels, preds)
    # Normalize to percentages for readability
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, subtitle in [
        (axes[0], cm,      "d",     "Counts"),
        (axes[1], cm_norm, ".1f",   "Normalized (%)"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=0.5,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} — {subtitle}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(labels, probs, title, save_path):
    # Binarize labels for one-vs-rest
    from sklearn.preprocessing import label_binarize
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]

    aucs = []
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        auc = roc_auc_score(labels_bin[:, i], probs[:, i])
        aucs.append(auc)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})")

    macro_auc = roc_auc_score(labels_bin, probs, average="macro")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title} — ROC Curves (macro AUC = {macro_auc:.3f})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return macro_auc, aucs


def plot_pr_curves(labels, probs, title, save_path):
    from sklearn.preprocessing import label_binarize
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]

    aps = []
    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        precision, recall, _ = precision_recall_curve(labels_bin[:, i], probs[:, i])
        ap = average_precision_score(labels_bin[:, i], probs[:, i])
        aps.append(ap)
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{name} (AP = {ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    macro_ap = np.mean(aps)
    ax.set_title(f"{title} — Precision-Recall Curves (mAP = {macro_ap:.3f})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return macro_ap, aps


# ── Report helpers ────────────────────────────────────────────────────────────

def print_and_save_report(labels, preds, save_dir, prefix):
    """Print sklearn classification report and save as CSV."""
    report_str = classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=4
    )
    print(report_str)

    # Save text version
    with open(os.path.join(save_dir, f"{prefix}_classification_report.txt"), "w") as f:
        f.write(report_str)

    # Parse to CSV
    report_dict = classification_report(
        labels, preds, target_names=CLASS_NAMES, digits=4, output_dict=True
    )
    csv_path = os.path.join(save_dir, f"{prefix}_classification_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for cls in CLASS_NAMES + ["macro avg", "weighted avg"]:
            row = report_dict[cls]
            writer.writerow([cls, f"{row['precision']:.4f}", f"{row['recall']:.4f}",
                             f"{row['f1-score']:.4f}", int(row['support'])])
    return report_dict


# ── Per-model evaluation ──────────────────────────────────────────────────────

def evaluate_model(model_name: str, test_loader):
    """Load checkpoint, run test inference, generate all plots/reports."""
    if model_name == "baseline_simple":
        model = BaselineCNNSimple().to(DEVICE)
        ckpt_path = os.path.join(EXPERIMENTS_DIR, "baseline_cnn_simple", "best_model.pth")
        save_dir = os.path.join(EXPERIMENTS_DIR, "baseline_cnn_simple")
        display = "Baseline CNN (Simple)"
    elif model_name == "baseline":
        model = BaselineCNN().to(DEVICE)
        ckpt_path = os.path.join(EXPERIMENTS_DIR, "baseline_cnn", "best_model.pth")
        save_dir = os.path.join(EXPERIMENTS_DIR, "baseline_cnn")
        display = "Baseline CNN (Aug)"
    else:
        model = ResNet50Classifier().to(DEVICE)
        ckpt_path = os.path.join(EXPERIMENTS_DIR, "resnet50_finetuned", "best_model.pth")
        save_dir = os.path.join(EXPERIMENTS_DIR, "resnet50_finetuned")
        display = "ResNet-50"

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\n{'='*60}")
    print(f"  {display}  (best checkpoint: epoch {ckpt['epoch']})")
    print(f"{'='*60}\n")

    labels, preds, probs = run_inference(model, test_loader, DEVICE)

    print("Classification Report:")
    report_dict = print_and_save_report(labels, preds, save_dir, prefix="test")

    macro_auc, per_class_auc = plot_roc_curves(
        labels, probs, display,
        os.path.join(save_dir, "roc_curves.png")
    )
    macro_ap, per_class_ap = plot_pr_curves(
        labels, probs, display,
        os.path.join(save_dir, "pr_curves.png")
    )
    plot_confusion_matrix(
        labels, preds, display,
        os.path.join(save_dir, "confusion_matrix.png")
    )

    print(f"Macro AUC-ROC : {macro_auc:.4f}")
    print(f"Mean Avg Prec : {macro_ap:.4f}")
    print(f"Plots saved to {save_dir}/\n")

    return {
        "model": display,
        "accuracy": float(np.mean(labels == preds)),
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "macro_auc": macro_auc,
        "per_class_f1": {
            name: report_dict[name]["f1-score"] for name in CLASS_NAMES
        },
        "per_class_auc": dict(zip(CLASS_NAMES, per_class_auc)),
    }


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(results: list):
    col_w = 22
    n_cols = 1 + len(results)
    total_w = col_w * n_cols

    print("\n" + "=" * total_w)
    print("  MODEL COMPARISON (Test Set)")
    print("=" * total_w)
    header = f"{'Metric':<{col_w}}" + "".join(f"{r['model']:>{col_w}}" for r in results)
    print(header)
    print("-" * total_w)

    rows = [
        ("Accuracy",        lambda r: f"{r['accuracy']:.4f}"),
        ("Macro F1",        lambda r: f"{r['macro_f1']:.4f}"),
        ("Weighted F1",     lambda r: f"{r['weighted_f1']:.4f}"),
        ("Macro AUC-ROC",   lambda r: f"{r['macro_auc']:.4f}"),
        ("",                lambda r: ""),
    ] + [
        (f"  F1 {name}",   (lambda n: lambda r: f"{r['per_class_f1'][n]:.4f}")(name))
        for name in CLASS_NAMES
    ] + [
        (f"  AUC {name}",  (lambda n: lambda r: f"{r['per_class_auc'][n]:.4f}")(name))
        for name in CLASS_NAMES
    ]

    for label, fn in rows:
        if label == "":
            print()
            continue
        print(f"{label:<{col_w}}" + "".join(f"{fn(r):>{col_w}}" for r in results))

    # F1 improvement: always compare first result (simplest baseline) vs last (best model)
    if len(results) >= 2:
        simple_f1 = results[0]["macro_f1"]
        best_f1   = results[-1]["macro_f1"]
        delta_f1  = best_f1 - simple_f1
        delta_auc = results[-1]["macro_auc"] - results[0]["macro_auc"]
        pct       = ((best_f1 - simple_f1) / simple_f1) * 100
        print(f"\n  F1 improvement ({results[-1]['model']} vs {results[0]['model']}):")
        print(f"    +{delta_f1:.4f}  ({pct:+.1f}% relative)")
        print(f"  AUC improvement: +{delta_auc:.4f}")

    print("=" * total_w)

    # Save CSV
    csv_path = os.path.join(EXPERIMENTS_DIR, "comparison_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + [r["model"] for r in results])
        for label, fn in rows:
            if label:
                writer.writerow([label] + [fn(r) for r in results])
    print(f"\nSaved comparison to {csv_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["baseline_simple", "baseline", "resnet", "both", "all"],
        default="both",
    )
    args = parser.parse_args()

    if args.model == "baseline_simple":
        _, _, test_loader = get_baseline_dataloaders()
        evaluate_model("baseline_simple", test_loader)
    elif args.model == "baseline":
        _, _, test_loader = get_dataloaders()
        evaluate_model("baseline", test_loader)
    elif args.model == "resnet":
        _, _, test_loader = get_dataloaders()
        evaluate_model("resnet", test_loader)
    elif args.model == "both":
        _, _, test_loader = get_dataloaders()
        r_baseline = evaluate_model("baseline", test_loader)
        r_resnet   = evaluate_model("resnet",   test_loader)
        print_comparison([r_baseline, r_resnet])
    elif args.model == "all":
        _, _, test_simple = get_baseline_dataloaders()
        _, _, test_aug    = get_dataloaders()
        r_simple   = evaluate_model("baseline_simple", test_simple)
        r_baseline = evaluate_model("baseline",        test_aug)
        r_resnet   = evaluate_model("resnet",          test_aug)
        print_comparison([r_simple, r_baseline, r_resnet])
