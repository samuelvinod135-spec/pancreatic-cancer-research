# Pancreatic Cancer Classification Research Pipeline

A full end-to-end deep learning pipeline for pancreatic cancer detection and staging
from 3D CT volumes using the **PanTSMini** dataset, Apple Silicon **MPS** acceleration,
and **Explainable AI** (Grad-CAM, SHAP, LIME).

---

## Project Structure

```
pancreatic_cancer_research/
├── dataset/
│   └── PanTSMini/          # Cloned from HuggingFace BodyMaps/PanTSMini
├── src/
│   ├── preprocess.py       # NIfTI → 2D PNG slice extraction
│   ├── train.py            # ResNet-50 training (binary + multi-class)
│   └── explain.py          # Grad-CAM / SHAP / LIME heatmaps
├── outputs/
│   ├── slices/             # Preprocessed 2D PNG slices
│   │   ├── Normal/
│   │   ├── Stage_1/
│   │   ├── Stage_2/
│   │   └── Stage_3/
│   ├── best_binary.pth     # Best binary model weights
│   ├── best_multi.pth      # Best multi-class model weights
│   ├── ta_va_binary.png    # Train/Val accuracy curves
│   ├── tl_vl_binary.png    # Train/Val loss curves
│   ├── ta_va_multi.png
│   ├── tl_vl_multi.png
│   ├── roc_binary.png      # ROC curve (binary)
│   ├── roc_multi.png       # ROC curve (multi-class OvR)
│   ├── cm_binary.png       # 2×2 Confusion Matrix
│   ├── cm_multi.png        # 3×3 Confusion Matrix
│   ├── xai_binary/         # XAI grid images (binary model)
│   └── xai_multi/          # XAI grid images (multi model)
└── run_pipeline.sh         # Master script: runs all 4 steps
```

---

## Setup

```bash
# 1. Install dependencies
pip install torch torchvision monai captum shap lime matplotlib \
            scikit-learn nibabel pandas tqdm Pillow

# 2. Clone dataset (requires git-lfs)
brew install git-lfs && git lfs install
git clone https://huggingface.co/datasets/BodyMaps/PanTSMini dataset/PanTSMini
```

---

## Running the Pipeline

### Option A — Full pipeline (recommended)
```bash
./run_pipeline.sh
```

### Option B — Step-by-step

```bash
# Step 1: Preprocess NIfTI volumes → 2D PNG slices
python3 src/preprocess.py --data_root dataset/PanTSMini \
                          --out_root outputs/slices \
                          --max_slices 20000 --workers 4

# Step 2: Train models (MPS-accelerated ResNet-50)
python3 src/train.py --slices_root outputs/slices \
                     --out_root outputs \
                     --epochs 15 --batch_size 32 --lr 1e-4

# Step 3: Generate XAI explanations (binary model)
python3 src/explain.py --slices_root outputs/slices \
                       --out_root outputs \
                       --n_samples 20 --model_type binary

# Step 4: Generate XAI explanations (multi-class model)
python3 src/explain.py --slices_root outputs/slices \
                       --out_root outputs \
                       --n_samples 20 --model_type multi
```

---

## Architecture

### Models
Both tasks use a **ResNet-50** backbone pretrained on ImageNet:
- Layers 1–3 are frozen; only **Layer 4 + custom FC head** are fine-tuned
- FC head: `Linear(2048→256) → ReLU → Dropout → Linear(256→N)`

### Dataset Splits
| Split | Purpose | Size |
|-------|---------|------|
| **DS1 – Training** | 5-fold cross-validation | 72% |
| **DS1 – Validation** | Held-out per-fold | 18% |
| **DS2 – Test** | Final evaluation (never seen during training) | 10% |

### Stage Assignment Heuristic
| Condition | Label |
|-----------|-------|
| No tumour mask voxels | Normal |
| Tumour volume ≤ 5 cm³ | Stage 1 |
| Tumour volume ≤ 20 cm³ | Stage 2 |
| Tumour volume > 20 cm³ | Stage 3 |

### Training
- Optimizer: **AdamW** (lr=1e-4, weight_decay=1e-4)
- Scheduler: **CosineAnnealingLR**
- Loss: **CrossEntropyLoss** with inverse-frequency class weights
- Augmentation: Horizontal flip, ±10° rotation, color jitter

### Hardware
- Primary backend: **MPS (Metal Performance Shaders)** – Apple Silicon GPU
- Fallback: CUDA → CPU

---

## Benchmark Results

### Task 1: Binary Classification (Normal vs Cancer)
* **5-Fold Cross-Validation Accuracy:** `[91.49%, 92.71%, 91.15%, 91.49%, 92.88%]` (Mean: **91.94%**)
* **Held-out Test Set (DS2) Accuracy:** **90.77%** (364 / 401 test slices correct)
* **Cancer Precision:** **0.98** | **Normal Recall:** **0.97** | **Macro F1:** **0.90**

```
              precision    recall  f1-score   support

      Normal       0.78      0.97      0.86       120
      Cancer       0.98      0.88      0.93       281

    accuracy                           0.91       401
   macro avg       0.88      0.92      0.90       401
weighted avg       0.92      0.91      0.91       401
```

### Task 2: Multi-Class Staging (Stage 1 vs Stage 2 vs Stage 3)
* **5-Fold Cross-Validation Accuracy:** `[75.00%, 74.69%, 71.22%, 74.19%, 76.67%]` (Mean: **74.35%**)
* **Held-out Test Set (DS2) Accuracy:** **67.26%**
* **Stage 3 Precision:** **0.94** | **Stage 1 Recall:** **0.78**

```
              precision    recall  f1-score   support

     Stage_1       0.47      0.78      0.59        60
     Stage_2       0.64      0.59      0.62       101
     Stage_3       0.94      0.68      0.79       120

    accuracy                           0.67       281
   macro avg       0.68      0.69      0.67       281
weighted avg       0.73      0.67      0.68       281
```

---

## Explainable AI (XAI) Methods

| Method | Library | Target / Formulation | Visualization |
|---|---|---|---|
| **Grad-CAM** | Captum (`LayerGradCam`) | Final Bottleneck (`resnet50.layer4[-1]`) | Jet Colormap Heatmap Overlay |
| **SHAP** | Captum (`GradientShap`) | Game-theoretic pixel attribution vs background distribution | Magma Colormap Saliency |
| **LIME** | Lime (`LimeImageExplainer`) | Perturbed superpixel surrogate model (300 samples) | Green Superpixel Mask & Boundaries |

Side-by-side 4-panel comparison grids are available in:
- `outputs/xai_binary/`: 20 comparison grids for Normal vs Cancer test scans.
- `outputs/xai_multi/`: 20 comparison grids for Stage 1 vs 2 vs 3 test scans.
- `outputs/ground_truth_segmentation/`: CT + Ground-Truth Tumor Delineation Overlays.

---

## Output Metrics & Artifacts

- **TA/VA curves** – `outputs/ta_va_binary.png`, `outputs/ta_va_multi.png`
- **TL/VL curves** – `outputs/tl_vl_binary.png`, `outputs/tl_vl_multi.png`
- **ROC curves** – `outputs/roc_binary.png`, `outputs/roc_multi.png`
- **Confusion Matrices** – `outputs/cm_binary.png` (2×2), `outputs/cm_multi.png` (3×3)
- **Model Checkpoints** – `outputs/best_binary.pth`, `outputs/best_multi.pth`

