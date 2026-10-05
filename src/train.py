"""
train.py  –  Pancreatic Cancer Classification Training Pipeline
═══════════════════════════════════════════════════════════════
Two ResNet-50 models are trained:
  1. Binary  : Normal vs Cancer
  2. Multi   : Stage_1 / Stage_2 / Stage_3

Dataset splits
  DS1  →  80 % training  +  20 % validation  (5-fold CV on training set)
  DS2  →  100 % testing  (held-out, never seen during training)

Hardware: MPS (Apple Silicon) with CPU fallback.

Output (all saved to outputs/):
  best_binary.pth  /  best_multi.pth
  ta_va_binary.png /  tl_vl_binary.png
  ta_va_multi.png  /  tl_vl_multi.png
  roc_binary.png   /  roc_multi.png
  cm_binary.png    /  cm_multi.png

Usage:
    python src/train.py --slices_root outputs/slices --out_root outputs
                        --epochs 15 --batch_size 32
"""

import argparse, os, random, warnings, ssl
from pathlib import Path
from collections import defaultdict

# ── SSL fix for macOS Python 3.14 (certificate verify failed) ────────────────
try:
    import certifi
    os.environ['SSL_CERT_FILE']  = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except ImportError:
    pass
# Patch urllib globally as fallback
_orig_ssl_ctx = ssl.create_default_context
def _patched_ssl(*args, **kwargs):
    ctx = _orig_ssl_ctx(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    return ctx
import urllib.request as _urllib_req
_urllib_req.ssl = ssl  # keep reference

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, accuracy_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
SEED   = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────────
_RESNET50_CACHE = Path.home() / ".cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"

def build_resnet50(num_classes: int) -> nn.Module:
    # Try loading from local cache first (avoids SSL download)
    if _RESNET50_CACHE.exists():
        print(f"  [Weights] Loading ResNet50 from local cache: {_RESNET50_CACHE}")
        model = models.resnet50(weights=None)
        state = torch.load(str(_RESNET50_CACHE), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    else:
        print("  [Weights] Downloading ResNet50 pretrained weights ...")
        # Disable SSL verification as fallback for macOS cert issues
        import ssl as _ssl
        old_ctx = _ssl._create_default_https_context
        _ssl._create_default_https_context = _ssl._create_unverified_context
        try:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        finally:
            _ssl._create_default_https_context = old_ctx

    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────
def make_binary_targets(dataset):
    """Map Normal(0) → 0, everything else → 1"""
    mapping = {cls: (0 if cls == "Normal" else 1)
               for cls in dataset.classes}
    return [mapping[dataset.classes[t]] for t in dataset.targets]


def get_class_weights(targets, num_classes, device):
    counts = np.bincount(targets, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    w      = 1.0 / counts
    w      = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Training / evaluation loops
# ──────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct      += (out.argmax(1) == labels).sum().item()
        total        += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    running_loss, total = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out   = model(imgs)
        loss  = criterion(out, labels)
        probs = torch.softmax(out, dim=1)
        preds = probs.argmax(1)
        running_loss += loss.item() * imgs.size(0)
        total        += imgs.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return running_loss / total, acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ──────────────────────────────────────────────────────────────────────────────
# Plotting utilities
# ──────────────────────────────────────────────────────────────────────────────
COLORS = ["#6C63FF", "#FF6584", "#43D9AD", "#FFB347", "#64B5F6"]

def plot_curves(train_vals, val_vals, ylabel, title, save_path, color_t=COLORS[0], color_v=COLORS[1]):
    epochs = range(1, len(train_vals) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_vals, marker="o", color=color_t, linewidth=2.2, label=f"Train {ylabel}")
    ax.plot(epochs, val_vals,   marker="s", color=color_v, linewidth=2.2, linestyle="--", label=f"Val {ylabel}")
    ax.fill_between(epochs, train_vals, val_vals, alpha=0.12, color=color_t)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_facecolor("#F7F7FB")
    fig.patch.set_facecolor("#F7F7FB")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_binary(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    auc_val     = roc_auc_score(y_true, y_prob[:, 1])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=COLORS[0], lw=2.5, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS[0])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC – Binary (Normal vs Cancer)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_facecolor("#F7F7FB"); fig.patch.set_facecolor("#F7F7FB")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(save_path, dpi=180); plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_multi(y_true, y_prob, class_names, save_path):
    n_cls = len(class_names)
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=list(range(n_cls)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_val     = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], lw=2.2,
                label=f"{name}  AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC – Multi-class (Stage 1/2/3)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_facecolor("#F7F7FB"); fig.patch.set_facecolor("#F7F7FB")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(save_path, dpi=180); plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    n   = len(class_names)
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    fig, ax = plt.subplots(figsize=(4 + n, 3 + n))
    im  = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(n)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, fontsize=12, rotation=30, ha="right")
    ax.set_yticklabels(class_names, fontsize=12)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, dpi=180); plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Core training routine with 5-Fold CV
# ──────────────────────────────────────────────────────────────────────────────
def run_training(full_dataset, targets, num_classes, class_names,
                 label_name, args, out_root):
    """
    Splits:
      DS1 = 80% (train 5-fold CV)  +  20% (validation)
      DS2 = 10% held-out test
    """
    n = len(full_dataset)
    all_idx = list(range(n))

    # Reserve DS2 (10 % of total)
    ds1_idx, ds2_idx = train_test_split(
        all_idx, test_size=0.10, stratify=targets, random_state=SEED)

    # Within DS1: 80/20 train/val
    ds1_train_idx, ds1_val_idx = train_test_split(
        ds1_idx, test_size=0.20, stratify=[targets[i] for i in ds1_idx],
        random_state=SEED)

    print(f"\n[{label_name}] DS1 train={len(ds1_train_idx)}  val={len(ds1_val_idx)}  DS2 test={len(ds2_idx)}")

    # ── 5-Fold CV on DS1 training set ─────────────────────────────────────────
    kf          = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_accs   = []
    best_val_acc = -1.0
    best_state   = None

    # accumulate global train/val curves (averaged over folds per epoch)
    fold_train_losses, fold_val_losses   = [], []
    fold_train_accs, fold_val_accs       = [], []

    train_tgts = [targets[i] for i in ds1_train_idx]
    ds1_train_arr = np.array(ds1_train_idx)

    for fold, (tr_sub, vl_sub) in enumerate(kf.split(ds1_train_arr)):
        print(f"\n  ── Fold {fold+1}/5 ──────────────────────────────────────────")
        fold_tr_idx = ds1_train_arr[tr_sub].tolist()
        fold_vl_idx = ds1_train_arr[vl_sub].tolist()

        # Datasets with correct transforms
        tr_ds = Subset(full_dataset, fold_tr_idx)
        vl_ds = Subset(full_dataset, fold_vl_idx)

        fold_tgts      = [targets[i] for i in fold_tr_idx]
        class_weights  = get_class_weights(fold_tgts, num_classes, DEVICE)

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=0)
        vl_loader = DataLoader(vl_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=0)

        model     = build_resnet50(num_classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

        ep_tr_loss, ep_tr_acc, ep_vl_loss, ep_vl_acc = [], [], [], []

        for epoch in range(1, args.epochs + 1):
            tl, ta = train_one_epoch(model, tr_loader, optimizer, criterion)
            vl, va, _, _, _ = evaluate(model, vl_loader)
            scheduler.step()
            ep_tr_loss.append(tl); ep_tr_acc.append(ta)
            ep_vl_loss.append(vl); ep_vl_acc.append(va)
            print(f"    Epoch {epoch:02d}/{args.epochs}  "
                  f"TL={tl:.4f} TA={ta:.4f}  VL={vl:.4f} VA={va:.4f}")

            if va > best_val_acc:
                best_val_acc = va
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        fold_accs.append(max(ep_vl_acc))
        fold_train_losses.append(ep_tr_loss)
        fold_val_losses.append(ep_vl_loss)
        fold_train_accs.append(ep_tr_acc)
        fold_val_accs.append(ep_vl_acc)

    print(f"\n  5-Fold CV results ({label_name}): "
          f"{[f'{a:.4f}' for a in fold_accs]}  "
          f"mean={np.mean(fold_accs):.4f}")

    # ── Save best model ────────────────────────────────────────────────────────
    model_path = out_root / f"best_{label_name.lower()}.pth"
    torch.save(best_state, model_path)
    print(f"  Best model saved: {model_path}")

    # ── Average curves across folds ────────────────────────────────────────────
    avg_tl = np.mean(fold_train_losses, axis=0)
    avg_vl = np.mean(fold_val_losses,   axis=0)
    avg_ta = np.mean(fold_train_accs,   axis=0)
    avg_va = np.mean(fold_val_accs,     axis=0)

    plot_curves(avg_ta.tolist(), avg_va.tolist(), "Accuracy",
                f"TA / VA – {label_name} (5-Fold Avg)",
                out_root / f"ta_va_{label_name.lower()}.png")
    plot_curves(avg_tl.tolist(), avg_vl.tolist(), "Loss",
                f"TL / VL – {label_name} (5-Fold Avg)",
                out_root / f"tl_vl_{label_name.lower()}.png")

    # ── Evaluate on DS2 ────────────────────────────────────────────────────────
    print(f"\n  Evaluating on DS2 ({label_name}) ...")
    best_model = build_resnet50(num_classes)
    best_model.load_state_dict(best_state)

    ds2_ds    = Subset(full_dataset, ds2_idx)
    ds2_loader = DataLoader(ds2_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    _, ds2_acc, y_true, y_pred, y_prob = evaluate(best_model, ds2_loader)
    print(f"  DS2 Accuracy ({label_name}): {ds2_acc:.4f}")
    print(classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                target_names=class_names, zero_division=0))

    # ROC
    if num_classes == 2:
        plot_roc_binary(y_true, y_prob, out_root / "roc_binary.png")
        plot_confusion_matrix(y_true, y_pred, class_names,
                              "Confusion Matrix – Binary (2×2)",
                              out_root / "cm_binary.png")
    else:
        plot_roc_multi(y_true, y_prob, class_names, out_root / "roc_multi.png")
        plot_confusion_matrix(y_true, y_pred, class_names,
                              "Confusion Matrix – Multi-class (3×3)",
                              out_root / "cm_multi.png")

    # Save DS2 indices and model path for explain.py
    np.save(out_root / f"ds2_idx_{label_name.lower()}.npy", np.array(ds2_idx))
    return best_model, ds2_idx


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slices_root", default="outputs/slices")
    parser.add_argument("--out_root",    default="outputs")
    parser.add_argument("--epochs",      type=int,   default=15)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    args = parser.parse_args()

    slices_root = Path(args.slices_root)
    out_root    = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Binary Task ───────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  TASK 1: BINARY CLASSIFICATION (Normal vs Cancer)")
    print("═"*60)

    binary_ds = datasets.ImageFolder(str(slices_root), transform=TRAIN_TF)
    print(f"Full dataset: {len(binary_ds)} images  classes={binary_ds.classes}")

    binary_map = {cls: (0 if cls == "Normal" else 1) for cls in binary_ds.classes}
    binary_ds.samples = [(p, binary_map[Path(p).parent.name]) for p, _ in binary_ds.samples]
    binary_ds.targets = [s[1] for s in binary_ds.samples]

    binary_model, ds2_binary = run_training(
        binary_ds, binary_ds.targets, num_classes=2,
        class_names=["Normal", "Cancer"],
        label_name="Binary", args=args, out_root=out_root)

    # ── Multi-class Task ──────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  TASK 2: MULTI-CLASS CLASSIFICATION (Stage 1 / 2 / 3)")
    print("═"*60)

    # Filter: keep only cancer stages
    cancer_classes = [c for c in binary_ds.classes if c != "Normal"]
    cancer_class_to_idx = {c: i for i, c in enumerate(cancer_classes)}

    multi_ds = datasets.ImageFolder(str(slices_root), transform=TRAIN_TF)
    multi_ds.samples = [(p, cancer_class_to_idx[Path(p).parent.name])
                        for p, _ in multi_ds.samples
                        if Path(p).parent.name in cancer_class_to_idx]
    multi_ds.targets = [s[1] for s in multi_ds.samples]
    print(f"Cancer dataset: {len(multi_ds)} images  classes={cancer_classes}")

    run_training(
        multi_ds, multi_ds.targets, num_classes=3,
        class_names=cancer_classes,
        label_name="Multi", args=args, out_root=out_root)

    print("\n All done! Outputs saved to:", out_root.resolve())


if __name__ == "__main__":
    main()
