"""
Model definitions for pathology image classification.

Phase 2: BaselineCNN       — simple 3-layer conv net (benchmark)
Phase 3: ResNet50Classifier — transfer learning with two-phase fine-tuning
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from src.config import DROPOUT, NUM_CLASSES


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


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 with a custom classification head, trained in two phases:

    Phase A — head only (backbone frozen):
        Only backbone.fc is trained; backbone weights are fixed.
        Lets the new head converge without corrupting pretrained features.

    Phase B — layer4 + head (partial fine-tune):
        Call unfreeze_layer4() to unfreeze the last residual block.
        Use a lower lr (1e-4) to gently adapt high-level features.

    Input:  (B, 3, 224, 224)
    Output: (B, NUM_CLASSES) raw logits
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT):
        super().__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Freeze entire backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace the default 1000-class head
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def unfreeze_layer4(self):
        """Unfreeze layer4 for Phase B fine-tuning."""
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
