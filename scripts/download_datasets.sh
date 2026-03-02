#!/usr/bin/env bash
# Download standard SISR benchmark datasets.
# Usage: bash scripts/download_datasets.sh [--data-root datasets]

set -euo pipefail

DATA_ROOT="datasets"

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$DATA_ROOT"

echo "=== Downloading DIV2K training set ==="
wget -q --show-progress -P "$DATA_ROOT/DIV2K" \
    "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip" || \
    echo "DIV2K train download failed (check URL or download manually)"

echo "=== Downloading DIV2K validation set ==="
wget -q --show-progress -P "$DATA_ROOT/DIV2K" \
    "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip" || \
    echo "DIV2K valid download failed"

echo "=== Extracting ==="
for zip in "$DATA_ROOT/DIV2K/"*.zip; do
    [ -f "$zip" ] && unzip -q "$zip" -d "$DATA_ROOT/DIV2K/" && rm "$zip"
done

echo ""
echo "NOTE: Set5, Set14, BSD100, and Urban100 must be downloaded manually."
echo "Place them under $DATA_ROOT/<DATASET_NAME>/HR/"
echo ""
echo "Done!"
