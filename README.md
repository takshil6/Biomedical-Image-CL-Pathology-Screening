# Biomedical Image Classification for Pathology Screening

Colorectal cancer tissue classification using the Kather 2016 histology dataset.  
Compares a custom 3-layer CNN baseline against ResNet-50 transfer learning.

---

## Project Overview

Digital pathology workflows require rapid, consistent screening of tissue slides.
This project trains classifiers that map 150×150 px histology tile images into
four clinically meaningful categories — **Tumor, Stroma, Immune, Other** — using
the publicly available Kather et al. 2016 dataset.

Two models are compared:

| Model | Strategy |
|---|---|
| **Baseline CNN** | 3-layer conv net trained from scratch |
| **ResNet-50** | Pretrained ImageNet backbone, two-phase fine-tuning |

---

## Dataset

**Kather Texture 2016 — Colorectal Cancer Histology**

- **5,000 images** (150×150 px, RGB, `.tif`)
- **8 original tissue classes** → mapped to **4 screening categories**
- Source: [Zenodo record 53169](https://zenodo.org/records/53169)

### Class Mapping (8 → 4)

| Original Class | Mapped Category | Clinical Meaning |
|---|---|---|
| `01_TUMOR` | **Tumor** | Malignant epithelial cells |
| `02_STROMA`, `03_COMPLEX` | **Stroma** | Tumor microenvironment |
| `04_LYMPHO` | **Immune** | Lymphocyte infiltrate (prognostic) |
| `05_DEBRIS`, `06_MUCOSA`, `07_ADIPOSE`, `08_EMPTY` | **Other** | Non-diagnostic tissue |

### Split

| Split | Images | Tumor | Stroma | Immune | Other |
|---|---|---|---|---|---|
| Train (70%) | 3,500 | 437 | 876 | 437 | 1,750 |
| Val (15%)   | 750   | 94  | 187 | 94  | 375   |
| Test (15%)  | 750   | 94  | 187 | 94  | 375   |

Splits are **stratified** by class; training uses **WeightedRandomSampler** to
compensate for the 4× imbalance of the "Other" super-class.

---

## Methodology

### Data Pipeline (`src/dataset.py`, `src/config.py`)

**Training augmentation:**
- `RandomResizedCrop(224, scale=(0.8, 1.0))`
- `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(90)`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`
- Normalize (ImageNet mean/std)

**Val / Test:** `Resize(256) → CenterCrop(224) → Normalize`

### Phase 2 — Baseline CNN (`src/model.py :: BaselineCNN`)

```
Conv2d(3,32)  → ReLU → MaxPool(2)
Conv2d(32,64) → ReLU → MaxPool(2)
Conv2d(64,128)→ ReLU → AdaptiveAvgPool(1)
Flatten → Linear(128, 4)
```

- **93,764 parameters**
- 20 epochs, Adam (lr=1e-3), ReduceLROnPlateau, CrossEntropyLoss with class weights
- Mixed precision (torch.amp) — ~9.5 min on RTX 4050

### Phase 3 — ResNet-50 Transfer Learning (`src/model.py :: ResNet50Classifier`)

Custom head replacing the default 1000-class FC:

```
Linear(2048, 512) → ReLU → Dropout(0.3) → Linear(512, 4)
```

Two-phase training strategy:

| Phase | Epochs | Trainable | LR |
|---|---|---|---|
| A — head only | 1–10 | 1,051,140 | 1e-3 |
| B — layer4 + head | 11–25 | 16,015,876 | 1e-4 |

- Adam optimizer, ReduceLROnPlateau, CrossEntropyLoss with class weights
- Mixed precision — ~12 min on RTX 4050

---

## Results

### Test Set Performance

| Metric | Baseline CNN | ResNet-50 |
|---|---|---|
| **Accuracy** | 84.3% | **96.1%** |
| **Macro F1** | 0.843 | **0.948** |
| Weighted F1 | 0.845 | 0.961 |
| Macro AUC-ROC | 0.974 | **0.998** |

**F1 improvement: +0.105 (+12.5% relative)**

### Per-Class F1

| Class | Baseline CNN | ResNet-50 |
|---|---|---|
| Tumor  | 0.867 | 0.944 |
| Stroma | 0.792 | 0.934 |
| Immune | 0.849 | 0.928 |
| Other  | 0.866 | **0.988** |

### Training Curves

| Baseline CNN | ResNet-50 |
|---|---|
| ![baseline curves](experiments/baseline_cnn/training_curves.png) | ![resnet curves](experiments/resnet50_finetuned/training_curves.png) |

### ResNet-50 — Confusion Matrix & ROC

| Confusion Matrix | ROC Curves |
|---|---|
| ![cm](experiments/resnet50_finetuned/confusion_matrix.png) | ![roc](experiments/resnet50_finetuned/roc_curves.png) |

---

## Reproducing Results

### 1. Setup

```bash
git clone https://github.com/takshil6/Biomedical-Image-CL-Pathology-Screening
cd Biomedical-Image-CL-Pathology-Screening
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python -m src.download_data
# Downloads ~1.2 GB from Zenodo and extracts to data/
```

### 3. Train

```bash
# Phase 2 — Baseline CNN (20 epochs, ~10 min)
python -m src.train --model baseline

# Phase 3 — ResNet-50 (25 epochs, ~12 min)
python -m src.train --model resnet
```

### 4. Evaluate

```bash
# Both models with full comparison table
python -m src.evaluate --model both

# Single model
python -m src.evaluate --model resnet
```

All plots are saved to `experiments/baseline_cnn/` and `experiments/resnet50_finetuned/`.

### Environment

Tested on Windows 11, Python 3.12, PyTorch 2.x, RTX 4050 (6 GB VRAM).  
Seed fixed to 42 for full reproducibility.

---

## Repository Structure

```
.
├── src/
│   ├── config.py          # Hyperparams, paths, class mapping, transforms
│   ├── dataset.py         # Dataset, stratified splits, WeightedRandomSampler
│   ├── model.py           # BaselineCNN + ResNet50Classifier
│   ├── train.py           # Training loops (baseline + resnet, mixed precision)
│   ├── evaluate.py        # Metrics, plots, comparison table
│   └── download_data.py   # Zenodo downloader
├── experiments/
│   ├── baseline_cnn/      # best_model.pth, history.json, all plots
│   ├── resnet50_finetuned/# best_model.pth, history.json, all plots
│   └── comparison_table.csv
├── requirements.txt
└── README.md
```

---

## Citation

Kather JN, Weis CA, Bianconi F, Melchers SM, Schad LR, Gaiser T, Marx A,
Zöllner FG. *Multi-class texture analysis in colorectal cancer histology.*
**Scientific Reports** 6, 27988 (2016).  
https://doi.org/10.1038/srep27988
