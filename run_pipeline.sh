#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_pipeline.sh  –  Full Pancreatic Cancer Research Pipeline
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
BASE="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE"

echo "================================================================"
echo " Pancreatic Cancer Research Pipeline"
echo "================================================================"

# Step 1 — Preprocessing
echo ""
echo "[1/4] Preprocessing: Extracting 2D slices from NIfTI volumes ..."
python3 src/preprocess.py \
    --data_root dataset/PanTSMini \
    --out_root  outputs/slices \
    --max_slices 20000 \
    --workers 4

# Step 2 — Training
echo ""
echo "[2/4] Training: Binary + Multi-class ResNet-50 (MPS accelerated) ..."
python3 src/train.py \
    --slices_root outputs/slices \
    --out_root    outputs \
    --epochs      15 \
    --batch_size  32 \
    --lr          1e-4

# Step 3 — XAI (Binary)
echo ""
echo "[3/4] XAI: Generating Grad-CAM / SHAP / LIME overlays (Binary) ..."
python3 src/explain.py \
    --slices_root outputs/slices \
    --out_root    outputs \
    --n_samples   20 \
    --model_type  binary

# Step 4 — XAI (Multi-class)
echo ""
echo "[4/4] XAI: Generating Grad-CAM / SHAP / LIME overlays (Multi) ..."
python3 src/explain.py \
    --slices_root outputs/slices \
    --out_root    outputs \
    --n_samples   20 \
    --model_type  multi

echo ""
echo "================================================================"
echo " Pipeline complete! All outputs saved to: outputs/"
echo "================================================================"
ls -lh outputs/*.png 2>/dev/null || true
