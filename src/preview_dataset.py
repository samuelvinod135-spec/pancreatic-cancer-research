"""
preview_dataset.py
──────────────────────────────────────────────────────────────────────────────
Randomly selects 16 images from outputs/slices/ with a balanced mix of
Normal and Cancer stages (Stage 1, Stage 2, Stage 3). Plots them in a
4x4 grid and saves the resulting image to dataset_preview.png.
"""

import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def main():
    slices_dir = Path("outputs/slices")
    classes = ["Normal", "Stage_1", "Stage_2", "Stage_3"]
    
    # Select 4 images from each category for a balanced 16-image sample
    selected_samples = []
    random.seed(42)

    for cls in classes:
        cls_folder = slices_dir / cls
        all_imgs = sorted(list(cls_folder.glob("*.png")))
        sampled = random.sample(all_imgs, min(4, len(all_imgs)))
        for img_path in sampled:
            selected_samples.append((cls, img_path))

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.patch.set_facecolor("#0F172A")

    cls_colors = {
        "Normal": "#4ADE80",
        "Stage_1": "#38BDF8",
        "Stage_2": "#FBBF24",
        "Stage_3": "#F87171"
    }

    for idx, (cls, img_path) in enumerate(selected_samples):
        row, col = divmod(idx, 4)
        ax = axes[row, col]
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")
        
        title = f"{cls.replace('_', ' ')}\n{img_path.stem}"
        ax.set_title(title, color=cls_colors[cls], fontsize=10, fontweight="bold", pad=6)

    plt.suptitle("Clinical CT Dataset Preview (Authentic Abdominal Slices)", color="white", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_file = Path("dataset_preview.png")
    plt.savefig(str(out_file), dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Dataset preview grid saved to: {out_file.resolve()}")

if __name__ == "__main__":
    main()
