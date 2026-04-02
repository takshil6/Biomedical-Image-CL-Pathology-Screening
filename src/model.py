"""
Model definitions for pathology image classification.

Phase 2: BaselineCNN  — simple 3-layer conv net (benchmark)
Phase 3: ResNet-50    — transfer learning (added later)
"""

import torch
import torch.nn as nn

from src.config import NUM_CLASSES


class BaselineCNN(nn.Module):
    """
    Lightweight 3-layer CNN for establishing a baseline F1-score.

    Architecture:
        Conv2d(3,32)  -> ReLU -> MaxPool(2)
        Conv2d(32,64) -> ReLU -> MaxPool(2)
        Conv2d(64,128)-> ReLU -> AdaptiveAvgPool(1)
        Flatten -> FC(128, NUM_CLASSES)

    Input:  (B, 3, 224, 224)
    Output: (B, NUM_CLASSES) raw logits
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten (B, 128, 1, 1) -> (B, 128)
        return self.classifier(x)
