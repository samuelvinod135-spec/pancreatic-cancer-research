"""
generate_synthetic.py
──────────────────────────────────────────────────────────────────────────────
Generates a realistic synthetic dataset that mimics PanTSMini NIfTI volumes
when the real download is unavailable (files >300 GB).

Produces:
  outputs/slices/Normal/     – Gaussian abdominal CT background only
  outputs/slices/Stage_1/    – Small pancreatic tumour blob (vol ~2 cm³)
  outputs/slices/Stage_2/    – Medium tumour blob (vol ~10 cm³)
  outputs/slices/Stage_3/    – Large tumour blob (vol >20 cm³)

Each case is a 96×96×64 float32 volume saved as .nii.gz, then sliced and
saved as 224×224 RGB PNGs.  This enables end-to-end pipeline testing.

Usage:
    python3 src/generate_synthetic.py \
        --n_cases_per_class 80 \
        --out_nifti  dataset/PanTSMini_synthetic \
        --out_slices outputs/slices
"""

import argparse, random
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED)

VOL_SHAPE = (96, 96, 64)   # (X, Y, Z)  – lightweight
VOXEL_MM  = (1.5, 1.5, 2.5)  # mm
SLICES_PER_CASE = 12  # max axial slices per case


# ─── CT simulation helpers ────────────────────────────────────────────────────

def make_abdomen_background(shape):
    """HU values: soft tissue ~50, fat ~-80, noise ±30."""
    vol = np.random.normal(loc=40, scale=30, size=shape).astype(np.float32)
    # Add fat stripe
    y_mid = shape[1] // 2
    vol[:, y_mid - 10:y_mid + 10, :] += np.random.normal(-80, 20, (shape[0], 20, shape[2]))
    return np.clip(vol, -200, 400)


def add_pancreas_region(vol, centre, radius=12):
    """Simulate pancreatic parenchyma as a bright Gaussian blob."""
    cx, cy, cz = centre
    x, y, z = np.ogrid[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
    dist = np.sqrt(((x - cx) / radius)**2 + ((y - cy) / radius)**2 + ((z - cz / 1.5)**2))
    mask = dist < 1.0
    vol[mask] += np.random.normal(60, 10)
    return vol


def add_tumour_blob(vol, centre, equiv_radius_mm, voxel_mm):
    """Add a hypo-attenuating lesion blob at centre with given equivalent radius."""
    rx = equiv_radius_mm / voxel_mm[0]
    ry = equiv_radius_mm / voxel_mm[1]
    rz = equiv_radius_mm / voxel_mm[2]
    cx, cy, cz = centre
    x, y, z = np.ogrid[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
    dist = ((x - cx) / rx)**2 + ((y - cy) / ry)**2 + ((z - cz) / rz)**2
    mask = dist < 1.0
    vol[mask] = np.random.normal(-20, 15, mask.sum())  # hypo-dense
    return vol, mask


def window_normalize(vol, lo=-100, hi=300):
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)
    return (vol * 255).astype(np.uint8)


# ─── Stage → radius mapping ───────────────────────────────────────────────────
STAGE_CONFIGS = {
    "Normal":  {"tumour": False, "radius_range": (0, 0)},
    "Stage_1": {"tumour": True,  "radius_range": (6, 9)},    # vol ~3-4 cm³
    "Stage_2": {"tumour": True,  "radius_range": (11, 14)},  # vol ~6-12 cm³
    "Stage_3": {"tumour": True,  "radius_range": (18, 24)},  # vol >20 cm³
}


def make_case(stage: str, case_id: str,
              out_nifti: Path, out_slices: Path):
    """Generate one synthetic 3D case and extract axial slices."""
    config = STAGE_CONFIGS[stage]
    shape  = VOL_SHAPE

    vol = make_abdomen_background(shape)

    # Pancreas centre (slightly off-centre)
    panc_cx = shape[0] // 2 + random.randint(-8, 8)
    panc_cy = shape[1] // 2 + random.randint(-8, 8)
    panc_cz = shape[2] // 2 + random.randint(-6, 6)
    vol = add_pancreas_region(vol, (panc_cx, panc_cy, panc_cz))

    tumour_mask = np.zeros(shape, dtype=np.uint8)

    if config["tumour"]:
        r_mm = random.uniform(*config["radius_range"])
        # Small jitter around pancreas
        tx = panc_cx + random.randint(-4, 4)
        ty = panc_cy + random.randint(-4, 4)
        tz = panc_cz + random.randint(-3, 3)
        tx = np.clip(tx, 10, shape[0] - 10)
        ty = np.clip(ty, 10, shape[1] - 10)
        tz = np.clip(tz, 5,  shape[2] - 5)
        vol, mask = add_tumour_blob(vol, (tx, ty, tz), r_mm, VOXEL_MM)
        tumour_mask[mask] = 1

    # Save NIfTI
    nifti_dir = out_nifti / stage
    nifti_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag([*VOXEL_MM, 1])
    nib.save(nib.Nifti1Image(vol, affine),
             str(nifti_dir / f"{case_id}_ct.nii.gz"))
    nib.save(nib.Nifti1Image(tumour_mask.astype(np.float32), affine),
             str(nifti_dir / f"{case_id}_seg.nii.gz"))

    # Extract PNG slices
    img_norm = window_normalize(vol)
    slice_dir = out_slices / stage
    slice_dir.mkdir(parents=True, exist_ok=True)

    if tumour_mask.max() > 0:
        axial_sums  = tumour_mask.sum(axis=(0, 1))
        cand_slices = np.where(axial_sums > 3)[0].tolist()
    else:
        lo = int(shape[2] * 0.25)
        hi = int(shape[2] * 0.75)
        cand_slices = list(range(lo, hi))

    random.shuffle(cand_slices)
    saved = 0
    for sl_idx in cand_slices[:SLICES_PER_CASE]:
        sl  = img_norm[:, :, sl_idx]
        pil = Image.fromarray(sl, mode="L").convert("RGB").resize((224, 224))
        pil.save(str(slice_dir / f"{case_id}_ax{sl_idx:03d}.png"))
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cases_per_class", type=int, default=80)
    parser.add_argument("--out_nifti",  default="dataset/PanTSMini_synthetic")
    parser.add_argument("--out_slices", default="outputs/slices")
    args = parser.parse_args()

    out_nifti  = Path(args.out_nifti)
    out_slices = Path(args.out_slices)
    out_nifti.mkdir(parents=True, exist_ok=True)
    out_slices.mkdir(parents=True, exist_ok=True)

    total_slices = 0
    for stage in ["Normal", "Stage_1", "Stage_2", "Stage_3"]:
        print(f"Generating {args.n_cases_per_class} {stage} cases ...")
        for i in tqdm(range(args.n_cases_per_class), desc=stage):
            case_id = f"synthetic_{stage}_{i:04d}"
            n = make_case(stage, case_id, out_nifti, out_slices)
            total_slices += n

    print(f"\nDone!  {total_slices} PNG slices saved to: {out_slices.resolve()}")
    for stage in ["Normal", "Stage_1", "Stage_2", "Stage_3"]:
        d = out_slices / stage
        n = len(list(d.glob("*.png"))) if d.exists() else 0
        print(f"  {stage:10s}: {n} slices")


if __name__ == "__main__":
    main()
