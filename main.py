"""
Unified entry point: train then evaluate pathology classifiers.

Usage:
    python main.py --model baseline_simple  # train + eval, then 3-way comparison
    python main.py --model baseline         # train + eval augmented baseline
    python main.py --model resnet50         # train + eval ResNet-50
"""

import argparse

from src.dataset import get_baseline_dataloaders, get_dataloaders
from src.evaluate import evaluate_model, print_comparison
from src.train import train_baseline, train_baseline_simple, train_resnet


def main():
    parser = argparse.ArgumentParser(description="Train + evaluate pathology classifier")
    parser.add_argument(
        "--model",
        choices=["baseline_simple", "baseline", "resnet50"],
        required=True,
        help="Which model to train and evaluate",
    )
    args = parser.parse_args()

    if args.model == "baseline_simple":
        # Train simple baseline
        train_baseline_simple()

        # Evaluate on its own test loader (no augmentation transforms)
        _, _, test_simple = get_baseline_dataloaders()
        r_simple = evaluate_model("baseline_simple", test_simple)

        # Load existing augmented baseline + ResNet results for 3-way table
        _, _, test_aug = get_dataloaders()
        r_baseline = evaluate_model("baseline", test_aug)
        r_resnet   = evaluate_model("resnet",   test_aug)

        print_comparison([r_simple, r_baseline, r_resnet])

    elif args.model == "baseline":
        train_baseline()
        _, _, test_loader = get_dataloaders()
        r_baseline = evaluate_model("baseline", test_loader)
        print_comparison([r_baseline])

    elif args.model == "resnet50":
        train_resnet()
        _, _, test_loader = get_dataloaders()
        r_resnet = evaluate_model("resnet", test_loader)
        print_comparison([r_resnet])


if __name__ == "__main__":
    main()
