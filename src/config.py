"""
Centralized configuration for the Biomedical Image Classification project.

Kather et al. 2016 colorectal cancer histology dataset:
8 tissue classes → 4 clinically meaningful categories.
"""

import os
import torch
from torchvision import transforms

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# After extraction the images live one level deeper
DATASET_DIR = os.path.join(DATA_DIR, "Kather_texture_2016_image_tiles_5000")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

# ── Dataset ──────────────────────────────────────────────────────────────────
# Original 8 Kather classes → 4 screening categories
#   Tumor:  malignant epithelial cells
#   Stroma: tumor microenvironment (simple + complex stroma)
#   Immune: lymphocyte infiltrate (prognostic marker)
#   Other:  non-diagnostic tissue (debris, mucosa, adipose, empty)
CLASS_MAPPING = {
    "01_TUMOR":   0,  # Tumor
    "02_STROMA":  1,  # Stroma
    "03_COMPLEX": 1,  # Stroma (complex)
    "04_LYMPHO":  2,  # Immune
    "05_DEBRIS":  3,  # Other
    "06_MUCOSA":  3,  # Other
    "07_ADIPOSE": 3,  # Other
    "08_EMPTY":   3,  # Other
}

CLASS_NAMES = ["Tumor", "Stroma", "Immune", "Other"]
NUM_CLASSES = len(CLASS_NAMES)

# Train / Val / Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Hyperparameters ──────────────────────────────────────────────────────────
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0 if os.name == "nt" else 4  # Windows spawn doesn't play nice with DataLoader workers

# Baseline CNN (augmented, original)
BASELINE_EPOCHS = 20
BASELINE_LR = 1e-3

# Simple baseline (no augmentation, smaller model)
BASELINE_SIMPLE_EPOCHS = 15

# ResNet-50 transfer learning
RESNET_HEAD_EPOCHS = 10   # Phase A: frozen backbone, train head only
RESNET_FINETUNE_EPOCHS = 15  # Phase B: unfreeze layer4 + head
RESNET_HEAD_LR = 1e-3
RESNET_FINETUNE_LR = 1e-4

WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── ImageNet normalization stats ─────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Transforms ───────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Minimal transforms for the naive baseline — no augmentation, no crop tricks
baseline_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── Zenodo download ─────────────────────────────────────────────────────────
ZENODO_URL = (
    "https://zenodo.org/records/53169/files/"
    "Kather_texture_2016_image_tiles_5000.zip"
)
ZIP_FILENAME = "Kather_texture_2016_image_tiles_5000.zip"
