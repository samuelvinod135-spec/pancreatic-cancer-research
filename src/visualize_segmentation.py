"""
visualize_segmentation.py
──────────────────────────────────────────────────────────────────────────────
Generates clinical segmentation overlay figures from the volumetric NIfTI
dataset and copies XAI segmentation explanation grids for presentation.
"""

from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil
from scipy.ndimage import binary_dilation

ART_IMG_DIR = Path("/Users/samuel/.gemini/antigravity-ide/brain/e48ce2da-91d3-4eaa-89e9-44e17ff37720/images")
ART_IMG_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = Path("outputs/ground_truth_segmentation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def window_normalize(vol, lo=-100, hi=300):
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)
    return vol

def generate_gt_segmentation_panels():
    stages = ["Stage_1", "Stage_2", "Stage_3", "Normal"]
    data_root = Path("dataset/PanTSMini_synthetic")

    for stage in stages:
        stage_dir = data_root / stage
        ct_files = sorted(list(stage_dir.glob("*_ct.nii.gz")))
        if not ct_files:
            continue
        
        # Pick first case
        ct_path = ct_files[0]
        seg_path = stage_dir / ct_path.name.replace("_ct.nii.gz", "_seg.nii.gz")
        
        ct_nii = nib.load(str(ct_path))
        ct_vol = ct_nii.get_fdata()
        
        if seg_path.exists():
            seg_vol = nib.load(str(seg_path)).get_fdata()
        else:
            seg_vol = np.zeros_like(ct_vol)

        # Find best axial slice
        if seg_vol.sum() > 0:
            slice_sums = seg_vol.sum(axis=(0, 1))
            best_z = int(np.argmax(slice_sums))
        else:
            best_z = ct_vol.shape[2] // 2

        ct_slice = window_normalize(ct_vol[:, :, best_z])
        seg_slice = seg_vol[:, :, best_z] > 0

        # Rotate 90 deg for standard radiological orientation
        ct_slice = np.rot90(ct_slice)
        seg_slice = np.rot90(seg_slice)

        # Plot 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor("#0F172A")

        # 1. CT Scan
        axes[0].imshow(ct_slice, cmap="gray")
        axes[0].set_title(f"{stage.replace('_', ' ')}: Raw CT Slice (Axial z={best_z})", color="white", fontsize=13, fontweight="bold", pad=10)
        axes[0].axis("off")

        # 2. Binary Mask
        axes[1].imshow(seg_slice, cmap="hot")
        mask_pixels = seg_slice.sum()
        axes[1].set_title(f"Tumor Segmentation Mask ({int(mask_pixels)} px)", color="#38BDF8", fontsize=13, fontweight="bold", pad=10)
        axes[1].axis("off")

        # 3. Blended Overlay
        rgb_ct = np.stack([ct_slice]*3, axis=-1)
        overlay = rgb_ct.copy()
        if seg_slice.sum() > 0:
            # Highlight tumor in red
            overlay[seg_slice, 0] = np.clip(overlay[seg_slice, 0] * 0.4 + 0.8, 0, 1)
            overlay[seg_slice, 1] = overlay[seg_slice, 1] * 0.3
            overlay[seg_slice, 2] = overlay[seg_slice, 2] * 0.3
            # Add contour
            contour = binary_dilation(seg_slice) ^ seg_slice
            overlay[contour] = [1.0, 0.9, 0.1] # bright yellow border

        axes[2].imshow(overlay)
        axes[2].set_title("CT + Tumor Delineation Overlay", color="#4ADE80", fontsize=13, fontweight="bold", pad=10)
        axes[2].axis("off")

        plt.tight_layout()
        out_file = OUT_DIR / f"segmentation_{stage.lower()}.png"
        plt.savefig(str(out_file), dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_file}")

        # Copy to artifact images
        shutil.copy2(str(out_file), str(ART_IMG_DIR / f"segmentation_{stage.lower()}.png"))

    # Also copy select XAI grids to artifact directory
    xai_binary = Path("outputs/xai_binary")
    xai_multi = Path("outputs/xai_multi")

    binary_samples = sorted(list(xai_binary.glob("*.png")))[:6]
    for p in binary_samples:
        shutil.copy2(str(p), str(ART_IMG_DIR / p.name))
        print(f"Copied XAI Binary: {p.name}")

    multi_samples = sorted(list(xai_multi.glob("*.png")))[:6]
    for p in multi_samples:
        shutil.copy2(str(p), str(ART_IMG_DIR / p.name))
        print(f"Copied XAI Multi: {p.name}")

if __name__ == "__main__":
    generate_gt_segmentation_panels()
