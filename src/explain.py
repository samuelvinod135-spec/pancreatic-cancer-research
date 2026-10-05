"""
explain.py  –  Explainable AI for Pancreatic Cancer Models
════════════════════════════════════════════════════════════
Generates Grad-CAM, SHAP, and LIME explanation heatmaps for test samples
from DS2, saved as side-by-side image grids in outputs/.

Usage:
    python src/explain.py --slices_root outputs/slices --out_root outputs
                          --n_samples 20 --model_type binary
                          [--model_type multi]
"""

import argparse, os, random, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings("ignore")

# ── XAI libraries ─────────────────────────────────────────────────────────────
import captum.attr as captum_attr
import shap
import lime
import lime.lime_image
from skimage.segmentation import mark_boundaries

# ──────────────────────────────────────────────────────────────────────────────
# Device (MPS / CUDA / CPU)
# ──────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
print(f"[Device] {DEVICE}")

EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])


def denormalize(tensor):
    """Convert normalized tensor to [0,1] numpy for display."""
    img = tensor.clone().cpu()
    for c in range(3):
        img[c] = img[c] * STD[c] + MEAN[c]
    return img.permute(1, 2, 0).numpy().clip(0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────────────────────────────────────
def build_resnet50(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


def load_model(weights_path: Path, num_classes: int) -> nn.Module:
    model = build_resnet50(num_classes)
    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM (via Captum GradCAM)
# ──────────────────────────────────────────────────────────────────────────────
def compute_gradcam(model, img_tensor, target_class):
    """Returns H×W heatmap in [0,1]."""
    # Target layer: last conv block of ResNet50
    target_layers = [model.layer4[-1]]
    gcam = captum_attr.LayerGradCam(model, model.layer4[-1])
    attr = gcam.attribute(
        img_tensor.unsqueeze(0).to(DEVICE),
        target=target_class)
    # Upsample to 224×224
    attr_up = torch.nn.functional.interpolate(
        attr, size=(224, 224), mode="bilinear", align_corners=False)
    heatmap = attr_up[0, 0].detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


# ──────────────────────────────────────────────────────────────────────────────
# SHAP (GradientShap)
# ──────────────────────────────────────────────────────────────────────────────
def compute_shap(model, img_tensor, background_tensors, target_class=0):
    """Returns H×W SHAP attribution (absolute sum over channels)."""
    gs  = captum_attr.GradientShap(model)
    bg  = torch.stack(background_tensors[:5]).to(DEVICE)
    inp = img_tensor.unsqueeze(0).to(DEVICE)
    attr = gs.attribute(inp, baselines=bg, target=target_class)
    sv  = attr[0].abs().sum(dim=0).detach().cpu().numpy()
    if sv.max() > 0:
        sv = sv / sv.max()
    return sv


# ──────────────────────────────────────────────────────────────────────────────
# LIME
# ──────────────────────────────────────────────────────────────────────────────
def compute_lime(model, img_np_rgb):
    """
    img_np_rgb: H×W×3 uint8 numpy array.
    Returns H×W mask of positive-contribution superpixels.
    """
    def batch_predict(images):
        batch = []
        for img in images:
            pil = Image.fromarray(img.astype(np.uint8))
            t   = EVAL_TF(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            batch.append(probs)
        return np.array(batch)

    explainer = lime.lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np_rgb.astype(np.double),
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=300,
        random_seed=42)

    top_label   = explanation.top_labels[0]
    temp, mask  = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=8, hide_rest=False)
    return mask.astype(float)


# ──────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ──────────────────────────────────────────────────────────────────────────────
def overlay_heatmap(base_img_np, heatmap, alpha=0.5, colormap="jet"):
    """base_img_np: H×W×3 float [0,1]. Returns H×W×3 float [0,1]."""
    colored = plt.get_cmap(colormap)(heatmap)[:, :, :3]  # H×W×3
    overlaid = (1 - alpha) * base_img_np + alpha * colored
    return np.clip(overlaid, 0, 1)


def overlay_lime(base_img_np, lime_mask, alpha=0.6):
    green = np.zeros_like(base_img_np)
    green[:, :, 1] = 1.0
    out  = base_img_np.copy()
    out[lime_mask > 0] = ((1 - alpha) * base_img_np[lime_mask > 0]
                          + alpha * green[lime_mask > 0])
    return np.clip(out, 0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Grid saver
# ──────────────────────────────────────────────────────────────────────────────
def save_explanation_grid(orig, gradcam_img, shap_img, lime_img,
                          pred_label, true_label, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    titles = ["Original CT", "Grad-CAM", "SHAP", "LIME"]
    panels = [orig, gradcam_img, shap_img, lime_img]

    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel, cmap=None if panel.ndim == 3 else "gray")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"True: {true_label}   |   Predicted: {pred_label}",
                 fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slices_root", default="outputs/slices")
    parser.add_argument("--out_root",    default="outputs")
    parser.add_argument("--n_samples",   type=int, default=20)
    parser.add_argument("--model_type",  choices=["binary", "multi"], default="binary")
    args = parser.parse_args()

    out_root    = Path(args.out_root)
    xai_out     = out_root / f"xai_{args.model_type}"
    xai_out.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    full_ds = datasets.ImageFolder(str(args.slices_root), transform=EVAL_TF)

    if args.model_type == "binary":
        num_classes  = 2
        class_names  = ["Normal", "Cancer"]
        weights_path = out_root / "best_binary.pth"
        ds2_idx_path = out_root / "ds2_idx_binary.npy"
        binary_map   = {cls: (0 if cls == "Normal" else 1) for cls in full_ds.classes}
        dataset      = datasets.ImageFolder(str(args.slices_root), transform=EVAL_TF)
        dataset.samples = [(p, binary_map[Path(p).parent.name]) for p, _ in dataset.samples]
        dataset.targets = [s[1] for s in dataset.samples]
        def get_label(orig_idx):
            return dataset.targets[orig_idx]
    else:
        num_classes  = 3
        cancer_classes = [c for c in full_ds.classes if c != "Normal"]
        class_names  = cancer_classes
        weights_path = out_root / "best_multi.pth"
        ds2_idx_path = out_root / "ds2_idx_multi.npy"
        cancer_map   = {c: i for i, c in enumerate(cancer_classes)}
        dataset      = datasets.ImageFolder(str(args.slices_root), transform=EVAL_TF)
        dataset.samples = [(p, cancer_map[Path(p).parent.name])
                           for p, _ in dataset.samples
                           if Path(p).parent.name in cancer_map]
        dataset.targets = [s[1] for s in dataset.samples]
        def get_label(orig_idx):
            return dataset.targets[orig_idx]

    if not weights_path.exists():
        print(f"[ERROR] Model weights not found: {weights_path}")
        print("  → Run src/train.py first.")
        raise SystemExit(1)

    model = load_model(weights_path, num_classes)

    # ── Load DS2 indices ──────────────────────────────────────────────────────
    if ds2_idx_path.exists():
        ds2_idx = np.load(str(ds2_idx_path)).tolist()
    else:
        # fallback: random 10%
        ds2_idx = random.sample(range(len(dataset)),
                                max(1, len(dataset) // 10))

    sample_idx = random.sample(ds2_idx, min(args.n_samples, len(ds2_idx)))
    print(f"Processing {len(sample_idx)} samples from DS2 ...")

    # ── Background for SHAP (random 10 samples) ───────────────────────────────
    bg_idx  = random.sample(range(len(dataset)), min(10, len(dataset)))
    bg_tensors = [dataset[i][0] for i in bg_idx]

    # ── Process each sample ───────────────────────────────────────────────────
    for n, idx in enumerate(sample_idx):
        print(f"  [{n+1}/{len(sample_idx)}] index={idx}")
        img_tensor, _ = dataset[idx]
        orig_np       = denormalize(img_tensor)         # H×W×3 float [0,1]
        orig_uint8    = (orig_np * 255).astype(np.uint8)

        true_lbl_idx  = get_label(idx)
        true_name     = class_names[true_lbl_idx]

        with torch.no_grad():
            logits = model(img_tensor.unsqueeze(0).to(DEVICE))
            pred_idx = logits.argmax(1).item()
        pred_name = class_names[pred_idx]

        # Grad-CAM
        try:
            gcam = compute_gradcam(model, img_tensor, pred_idx)
            gradcam_overlay = overlay_heatmap(orig_np, gcam)
        except Exception as e:
            print(f"    [WARN] GradCAM failed: {e}")
            gradcam_overlay = orig_np

        # SHAP
        try:
            shap_map = compute_shap(model, img_tensor, bg_tensors, pred_idx)
            shap_overlay = overlay_heatmap(orig_np, shap_map, colormap="magma")
        except Exception as e:
            print(f"    [WARN] SHAP failed: {e}")
            shap_overlay = orig_np

        # LIME
        try:
            lime_mask    = compute_lime(model, orig_uint8)
            lime_overlay = overlay_lime(orig_np, lime_mask)
        except Exception as e:
            print(f"    [WARN] LIME failed: {e}")
            lime_overlay = orig_np

        out_path = xai_out / f"sample_{n+1:03d}_true{true_name}_pred{pred_name}.png"
        save_explanation_grid(orig_np, gradcam_overlay, shap_overlay, lime_overlay,
                              pred_name, true_name, out_path)
        print(f"    Saved: {out_path.name}")

    print(f"\n XAI complete. Grids saved to: {xai_out.resolve()}")


if __name__ == "__main__":
    main()
