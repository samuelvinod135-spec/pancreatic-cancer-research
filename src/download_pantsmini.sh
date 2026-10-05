#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# download_pantsmini.sh  –  Downloads real PanTSMini data from HuggingFace
#
# Requirements: git-lfs, curl, ~320 GB free disk space
# The labels archive is ~2GB from JHU servers.
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
BASE="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$BASE/dataset/PanTSMini/data"
mkdir -p "$DATA/ImageTr" "$DATA/ImageTe"

echo "=== Downloading PanTSMini metadata ==="
curl -L --progress-bar \
  "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/metadata.xlsx?download=true" \
  -o "$DATA/metadata.xlsx"

echo "=== Downloading Training Images (9 batches × ~1000 cases each) ==="
for i in {1..9}; do
    start=$(printf "%08d" $(( (i - 1) * 1000 + 1 )))
    end=$(printf   "%08d" $(( i * 1000 )))
    file="PanTSMini_ImageTr_${start}_${end}.tar.gz"
    url="https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/${file}?download=true"
    echo "[${i}/9] $file"
    curl -L --progress-bar -o "$DATA/$file" "$url"
    tar -xzf "$DATA/$file" -C "$DATA/ImageTr"
    rm "$DATA/$file"
    echo "[${i}/9] Done"
done

echo "=== Downloading Test Images ==="
file="PanTSMini_ImageTe_00009001_00009901.tar.gz"
url="https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/$file?download=true"
curl -L --progress-bar -o "$DATA/$file" "$url"
tar -xzf "$DATA/$file" -C "$DATA/ImageTe"
rm "$DATA/$file"

echo "=== Downloading Labels from JHU ==="
curl -L --progress-bar \
  "http://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz" \
  -o "$DATA/PanTSMini_Label.tar.gz"
mkdir -p "$DATA/LabelAll" "$DATA/LabelTr" "$DATA/LabelTe"
tar -xzf "$DATA/PanTSMini_Label.tar.gz" -C "$DATA/LabelAll"
rm "$DATA/PanTSMini_Label.tar.gz"

# Organise into Tr/Te splits
cd "$DATA/LabelAll"
find . -maxdepth 1 -type d \( -name 'PanTS_0000[0-8]*' -o -name 'PanTS_00009000' \) \
  -print0 | xargs -0 -r mv -t ../LabelTr/
find . -maxdepth 1 -type d -name 'PanTS_00009*' \
  -print0 | xargs -0 -r mv -t ../LabelTe/
cd "$BASE"

echo "=== Download complete! ==="
echo "Run preprocessing with:"
echo "  python3 src/preprocess_pantsmini.py --data_root dataset/PanTSMini/data --out_root outputs/slices"
