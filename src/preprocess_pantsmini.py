"""
preprocess_pantsmini.py
──────────────────────────────────────────────────────────────────────────────
Processes REAL PanTSMini data after running src/download_pantsmini.sh.

Reads metadata.xlsx for tumour size → stage assignment:
  Normal  : tumor? == 0
  Stage_1 : tumour volume (from structured report) <= 5 cm³
  Stage_2 : tumour volume 5–20 cm³
  Stage_3 : tumour volume > 20 cm³ or T3/T4 mention

Usage:
    python3 src/preprocess_pantsmini.py \
        --data_root dataset/PanTSMini/data \
        --out_root  outputs/slices \
        --max_slices 20000 --workers 4
"""

import argparse, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from tqdm import tqdm

STAGE_DIRS = ["Normal", "Stage_1", "Stage_2", "Stage_3"]
MIN_MASK_VOXELS = 5
HU_LO, HU_HI = -100, 300


# ─── Stage from metadata ──────────────────────────────────────────────────────

def extract_volume_from_report(report: str) -> float:
    """Extract largest tumour volume in cc from structured report text."""
    if not isinstance(report, str):
        return 0.0
    matches = re.findall(r'Volume:\s*([\d.]+)\s*cc', report, re.IGNORECASE)
    if matches:
        return max(float(v) for v in matches)
    # Fallback: size in cm  (e.g. "2.4 x 1.4 cm")
    size_m = re.findall(r'([\d.]+)\s*x\s*([\d.]+)\s*cm', report, re.IGNORECASE)
    if size_m:
        a, b = float(size_m[0][0]), float(size_m[0][1])
        # Approximate volume of ellipsoid: 4/3 π (a/2)(b/2)(b/2)
        return (4/3) * np.pi * (a/2) * (b/2)**2
    return 0.0


def assign_stage_from_meta(row: dict) -> str:
    if row.get("tumor?", 0) == 0:
        return "Normal"
    report = row.get("structured report", "")
    vol    = extract_volume_from_report(report)
    if vol <= 0:
        return "Stage_1"   # unknown small
    if vol <= 5.0:
        return "Stage_1"
    if vol <= 20.0:
        return "Stage_2"
    return "Stage_3"


# ─── Worker ───────────────────────────────────────────────────────────────────

def window_normalize(vol, lo=HU_LO, hi=HU_HI):
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo) * 255).astype(np.uint8)


def process_case(args):
    img_path, mask_path, stage, out_root, max_slices = args
    results = []
    try:
        img_nib  = nib.load(str(img_path))
        img_can  = nib.as_closest_canonical(img_nib)
        img_data = np.asarray(img_can.dataobj, dtype=np.float32)

        if mask_path and Path(mask_path).exists():
            mask = np.asarray(
                nib.as_closest_canonical(nib.load(str(mask_path))).dataobj,
                dtype=np.uint8)
        else:
            mask = np.zeros(img_data.shape, dtype=np.uint8)

        n_z      = img_data.shape[2]
        img_norm = window_normalize(img_data)

        if mask.max() > 0:
            sums  = mask.sum(axis=(0, 1))
            cands = np.where(sums >= MIN_MASK_VOXELS)[0].tolist()
        else:
            lo = int(n_z * 0.20); hi = int(n_z * 0.80)
            cands = list(range(lo, hi))

        import random; random.shuffle(cands)
        cands = cands[:max_slices]

        stage_dir = Path(out_root) / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        case_id = Path(img_path).stem.replace(".nii", "")

        for sl_idx in cands:
            sl    = img_norm[:, :, sl_idx]
            fname = f"{case_id}_ax{sl_idx:04d}.png"
            dest  = stage_dir / fname
            if not dest.exists():
                pil = Image.fromarray(sl, mode="L").convert("RGB")
                pil.save(str(dest))
            results.append((stage, str(dest)))
    except Exception as e:
        print(f"[WARN] {img_path}: {e}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   default="dataset/PanTSMini/data")
    parser.add_argument("--out_root",    default="outputs/slices")
    parser.add_argument("--max_slices",  type=int, default=20_000)
    parser.add_argument("--workers",     type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta_path = data_root / "metadata.xlsx"
    meta      = pd.read_excel(str(meta_path))
    meta_dict = {str(row["PanTS ID"]).strip(): dict(row)
                 for _, row in meta.iterrows()}
    print(f"Loaded metadata for {len(meta_dict)} cases")

    # Discover image files
    img_dir  = data_root / "ImageTr"
    lbl_dir  = data_root / "LabelTr"
    img_files = sorted(img_dir.glob("*.nii.gz")) if img_dir.exists() else []
    print(f"Found {len(img_files)} training volumes")

    if not img_files:
        print(f"[ERROR] No .nii.gz files in {img_dir}")
        print("  → Run src/download_pantsmini.sh first")
        raise SystemExit(1)

    max_per_case = max(1, args.max_slices // len(img_files))

    worker_args = []
    for img_path in img_files:
        case_id   = img_path.stem.replace(".nii", "")
        row       = meta_dict.get(case_id, {"tumor?": 0})
        stage     = assign_stage_from_meta(row)
        mask_path = lbl_dir / img_path.name if lbl_dir.exists() else None
        worker_args.append((str(img_path), str(mask_path) if mask_path else None,
                            stage, str(out_root), max_per_case))

    all_results, total = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_case, wa): wa[0] for wa in worker_args}
        pbar    = tqdm(as_completed(futures), total=len(futures), desc="Cases")
        for fut in pbar:
            res    = fut.result()
            all_results.extend(res)
            total += len(res)
            pbar.set_postfix(slices=total)
            if total >= args.max_slices:
                pool.shutdown(wait=False, cancel_futures=True)
                break

    from collections import Counter
    cnt = Counter(stage for stage, _ in all_results)
    print("\n--- Preprocessing complete ---")
    for s in STAGE_DIRS:
        print(f"  {s:10s}: {cnt.get(s, 0):>6,} slices")
    print(f"  {'TOTAL':10s}: {sum(cnt.values()):>6,}")


if __name__ == "__main__":
    main()
