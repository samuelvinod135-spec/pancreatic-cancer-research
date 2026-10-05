"""
preprocess.py
─────────────────────────────────────────────────────────────────────────────
Parses PanTSMini NIfTI (.nii.gz) volumes, extracts annotated 2-D axial slices
and saves them as PNG files organised by stage label:
    outputs/slices/Normal/
    outputs/slices/Stage_1/
    outputs/slices/Stage_2/
    outputs/slices/Stage_3/

Stage assignment rules (heuristic – adapt if ground-truth metadata exists):
  - No tumour mask voxels   -> Normal
  - Tumour volume <= 5 cm3  -> Stage 1
  - Tumour volume <= 20 cm3 -> Stage 2
  - Tumour volume  > 20 cm3 -> Stage 3

Usage:
    python src/preprocess.py [--data_root dataset/PanTSMini] [--out_root outputs/slices]
                             [--max_slices 20000] [--workers 4]
"""

import argparse
import json
import os
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

# ─── Constants ────────────────────────────────────────────────────────────────
STAGE_DIRS = ["Normal", "Stage_1", "Stage_2", "Stage_3"]
MIN_MASK_VOXELS = 10
T1, T2 = 5.0, 20.0   # tumour volume thresholds in cm3


def load_metadata(data_root: Path) -> dict:
    """Return {case_id: stage_label} if a JSON sidecar exists, else {}."""
    for candidate in ["metadata.json", "labels.json", "cases.json"]:
        p = data_root / candidate
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return {}


def window_normalize(vol: np.ndarray, lo: int = -100, hi: int = 300) -> np.ndarray:
    """Clip to HU window and rescale to [0, 255] uint8."""
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)
    return (vol * 255).astype(np.uint8)


def voxel_volume_cm3(header) -> float:
    zooms = header.get_zooms()[:3]
    return float(np.prod(zooms)) / 1000.0


def assign_stage(tumour_vox: int, vox_cm3: float, case_id: str, metadata: dict) -> str:
    if case_id in metadata:
        return metadata[case_id]
    if tumour_vox == 0:
        return "Normal"
    vol = tumour_vox * vox_cm3
    if vol <= T1:
        return "Stage_1"
    if vol <= T2:
        return "Stage_2"
    return "Stage_3"


def save_slice(img_slice: np.ndarray, path: Path):
    pil = Image.fromarray(img_slice, mode="L").convert("RGB")
    pil.save(str(path), format="PNG")


def process_case(args):
    img_path, mask_path, case_id, out_root, metadata, max_per_case = args
    results = []
    try:
        img_nib  = nib.load(str(img_path))
        img_can  = nib.as_closest_canonical(img_nib)
        img_data = np.asarray(img_can.dataobj, dtype=np.float32)
        vox_cm3  = voxel_volume_cm3(img_can.header)

        if mask_path and Path(mask_path).exists():
            mask_nib  = nib.load(str(mask_path))
            mask_can  = nib.as_closest_canonical(mask_nib)
            mask_data = np.asarray(mask_can.dataobj, dtype=np.uint8)
        else:
            mask_data = np.zeros(img_data.shape, dtype=np.uint8)

        tumour_vox = int(mask_data.sum())
        stage      = assign_stage(tumour_vox, vox_cm3, case_id, metadata)

        n_axial  = img_data.shape[2]
        img_norm = window_normalize(img_data)

        if mask_data.max() > 0:
            axial_sums  = mask_data.sum(axis=(0, 1))
            cand_slices = np.where(axial_sums >= MIN_MASK_VOXELS)[0].tolist()
        else:
            lo = int(n_axial * 0.20)
            hi = int(n_axial * 0.80)
            cand_slices = list(range(lo, hi))

        random.shuffle(cand_slices)
        cand_slices = cand_slices[:max_per_case]

        stage_dir = Path(out_root) / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        for sl_idx in cand_slices:
            sl    = img_norm[:, :, sl_idx]
            fname = f"{case_id}_ax{sl_idx:04d}.png"
            dest  = stage_dir / fname
            if not dest.exists():
                save_slice(sl, dest)
            results.append((stage, str(dest)))

    except Exception as exc:
        print(f"[WARN] {case_id}: {exc}")
    return results


def discover_cases(data_root: Path):
    cases   = []
    img_dir = data_root / "images"
    lbl_dir = data_root / "labels"
    if img_dir.exists():
        for img_path in sorted(img_dir.glob("*.nii.gz")):
            case_id   = img_path.name.replace(".nii.gz", "")
            mask_path = lbl_dir / img_path.name if lbl_dir.exists() else None
            cases.append((img_path, mask_path, case_id))
        return cases

    for sub in sorted(data_root.iterdir()):
        if not sub.is_dir():
            continue
        ct_candidates = (list(sub.glob("ct*.nii.gz")) +
                         list(sub.glob("img*.nii.gz")) +
                         list(sub.glob("*.nii.gz")))
        if not ct_candidates:
            continue
        img_path      = ct_candidates[0]
        seg_candidates = [sub / "seg.nii.gz", sub / "label.nii.gz",
                          sub / "mask.nii.gz", sub / "segmentation.nii.gz"]
        mask_path = next((p for p in seg_candidates if p.exists()), None)
        cases.append((img_path, mask_path, sub.name))
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  default="dataset/PanTSMini")
    parser.add_argument("--out_root",   default="outputs/slices")
    parser.add_argument("--max_slices", type=int, default=20_000)
    parser.add_argument("--workers",    type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(data_root)
    cases    = discover_cases(data_root)
    if not cases:
        print(f"[ERROR] No NIfTI cases found under {data_root}")
        raise SystemExit(1)

    print(f"Found {len(cases)} cases.  Extracting up to {args.max_slices} slices ...")
    max_per_case = max(1, args.max_slices // len(cases))

    worker_args = [
        (str(img), str(mask) if mask else None, cid,
         str(out_root), metadata, max_per_case)
        for img, mask, cid in cases
    ]

    all_results, total = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_case, wa): wa[2] for wa in worker_args}
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
    for stage in STAGE_DIRS:
        print(f"  {stage:10s}: {cnt.get(stage, 0):>6,} slices")
    print(f"  {'TOTAL':10s}: {sum(cnt.values()):>6,} slices")
    print(f"Saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
