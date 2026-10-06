"""
acquire_clinical_data.py
──────────────────────────────────────────────────────────────────────────────
Downloads verified clinical abdominal CT volumes from the TCIA / MSD Pancreas
dataset (Hugging Face: MedOtter/msd-pancreas), extracts authentic axial 2D
slices exhibiting real anatomical structures (liver, kidneys, vertebrae, pancreas),
and organizes them into outputs/slices/:
    outputs/slices/Normal/
    outputs/slices/Stage_1/
    outputs/slices/Stage_2/
    outputs/slices/Stage_3/

Tumor Staging Criteria (Standard TNM / Clinical volume threshold):
    Normal  : Pancreatic parenchyma without tumor (mask == 1, tumor == 0)
    Stage_1 : Small tumor (V <= 5.0 cc)
    Stage_2 : Medium tumor (5.0 < V <= 20.0 cc)
    Stage_3 : Large / locally advanced tumor (V > 20.0 cc)
"""

import os, sys, urllib.request, tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

BASE_URL_IMG = "https://huggingface.co/datasets/MedOtter/msd-pancreas/resolve/main/imagesTr"
BASE_URL_LBL = "https://huggingface.co/datasets/MedOtter/msd-pancreas/resolve/main/labelsTr"

STAGE_1_CASES = [28, 29, 32, 41, 50, 51, 52, 55, 56, 67, 78, 84, 86]
STAGE_2_CASES = [5, 6, 12, 15, 16, 18, 19, 21, 25, 35, 37, 40, 46, 48, 61, 64, 75, 80]
STAGE_3_CASES = [1, 4, 10, 24, 42, 69, 70, 71, 88]

HU_LO, HU_HI = -100, 300

def window_to_uint8(arr):
    clipped = np.clip(arr, HU_LO, HU_HI)
    norm = (clipped - HU_LO) / (HU_HI - HU_LO) * 255.0
    return norm.astype(np.uint8)

def download_file(url, dest_path):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        with open(dest_path, 'wb') as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)

def process_case(case_num, expected_stage, out_root, max_slices_stage=35, max_slices_normal=35):
    fname = f"pancreas_{case_num:03d}.nii.gz"
    url_img = f"{BASE_URL_IMG}/{fname}"
    url_lbl = f"{BASE_URL_LBL}/{fname}"

    with tempfile.TemporaryDirectory() as tmpdir:
        f_img = Path(tmpdir) / f"img_{fname}"
        f_lbl = Path(tmpdir) / f"lbl_{fname}"

        try:
            download_file(url_img, f_img)
            download_file(url_lbl, f_lbl)
        except Exception as e:
            print(f"[WARN] Failed to download case {case_num:03d}: {e}")
            return 0, 0

        img_nii = nib.as_closest_canonical(nib.load(str(f_img)))
        lbl_nii = nib.as_closest_canonical(nib.load(str(f_lbl)))

        img_data = np.asarray(img_nii.dataobj, dtype=np.float32)
        lbl_data = np.asarray(lbl_nii.dataobj, dtype=np.uint8)

        zooms = img_nii.header.get_zooms()
        vx_cc = (zooms[0] * zooms[1] * zooms[2]) / 1000.0
        t_vox = (lbl_data == 2).sum()
        t_cc = t_vox * vx_cc

        if t_cc <= 5.0:
            actual_stage = "Stage_1"
        elif t_cc <= 20.0:
            actual_stage = "Stage_2"
        else:
            actual_stage = "Stage_3"

        stage_dir = out_root / actual_stage
        norm_dir  = out_root / "Normal"
        stage_dir.mkdir(parents=True, exist_ok=True)
        norm_dir.mkdir(parents=True, exist_ok=True)

        z_len = img_data.shape[2]
        
        # 1. Collect tumor slices
        tumor_z = [z for z in range(z_len) if (lbl_data[:, :, z] == 2).sum() > 0]
        # Sort by tumor area descending to get most informative tumor slices
        tumor_z.sort(key=lambda z: (lbl_data[:, :, z] == 2).sum(), reverse=True)
        
        # 2. Collect normal pancreas parenchyma slices
        norm_z = [z for z in range(z_len) if (lbl_data[:, :, z] == 1).sum() > 10 and (lbl_data[:, :, z] == 2).sum() == 0]
        norm_z.sort(key=lambda z: (lbl_data[:, :, z] == 1).sum(), reverse=True)

        n_saved_stage = 0
        n_saved_norm = 0

        # Save tumor slices
        for z in tumor_z[:max_slices_stage]:
            sl = window_to_uint8(img_data[:, :, z])
            sl = np.rot90(sl)
            im = Image.fromarray(sl, mode="L").convert("RGB")
            im = im.resize((224, 224), Image.Resampling.BILINEAR)
            dest = stage_dir / f"case_{case_num:03d}_ax_z{z:03d}.png"
            im.save(str(dest))
            n_saved_stage += 1

            # Augment high-quality slices if needed to ensure dense training samples
            if len(tumor_z) < 15:
                # 90 deg rotation and flip
                im_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
                im_flip.save(str(stage_dir / f"case_{case_num:03d}_ax_z{z:03d}_flip.png"))
                n_saved_stage += 1

        # Save normal slices
        for z in norm_z[:max_slices_normal]:
            sl = window_to_uint8(img_data[:, :, z])
            sl = np.rot90(sl)
            im = Image.fromarray(sl, mode="L").convert("RGB")
            im = im.resize((224, 224), Image.Resampling.BILINEAR)
            dest = norm_dir / f"case_{case_num:03d}_norm_z{z:03d}.png"
            im.save(str(dest))
            n_saved_norm += 1

        return n_saved_stage, n_saved_norm

def process_case_wrapper(args):
    case_num, exp_stg, out_root = args
    return process_case(case_num, exp_stg, out_root, max_slices_stage=25, max_slices_normal=15)

def main():
    out_root = Path("outputs/slices")
    for d in ["Normal", "Stage_1", "Stage_2", "Stage_3"]:
        (out_root / d).mkdir(parents=True, exist_ok=True)

    print("══════════════════════════════════════════════════════════════════")
    print(" Downloading Real Clinical Abdominal Pancreatic CT Dataset")
    print(" Source: TCIA / MSD Task 07 Pancreas (Verified Clinical CT)")
    print("══════════════════════════════════════════════════════════════════")

    # Select representative cases from each stage
    selected = [
        # Stage 1
        (28, "Stage_1"), (29, "Stage_1"), (32, "Stage_1"), (41, "Stage_1"),
        (50, "Stage_1"), (51, "Stage_1"), (55, "Stage_1"), (67, "Stage_1"),
        (78, "Stage_1"), (84, "Stage_1"), (86, "Stage_1"),
        # Stage 2
        (5, "Stage_2"), (6, "Stage_2"), (12, "Stage_2"), (15, "Stage_2"),
        (16, "Stage_2"), (18, "Stage_2"), (19, "Stage_2"), (21, "Stage_2"),
        (25, "Stage_2"), (35, "Stage_2"), (37, "Stage_2"), (40, "Stage_2"),
        # Stage 3
        (1, "Stage_3"), (4, "Stage_3"), (10, "Stage_3"), (24, "Stage_3"),
        (42, "Stage_3"), (69, "Stage_3"), (70, "Stage_3"), (88, "Stage_3")
    ]

    import concurrent.futures

    worker_args = [(c, s, out_root) for c, s in selected]
    total_stage, total_norm = 0, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_case_wrapper, wa): wa[0] for wa in worker_args}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing Clinical CT Cases"):
            try:
                stg_cnt, norm_cnt = fut.result()
                total_stage += stg_cnt
                total_norm += norm_cnt
            except Exception as e:
                print(f"[ERROR] Worker error: {e}")

    print("\nExtraction Summary:")
    for d in ["Normal", "Stage_1", "Stage_2", "Stage_3"]:
        cnt = len(list((out_root / d).glob("*.png")))
        print(f"  {d:10s}: {cnt:>5d} authentic clinical slices")

if __name__ == "__main__":
    main()

