#!/usr/bin/env bash
# Train MEK + ResNet-18 on RAF-DB.
#   RAF_ROOT=/data/rafdb bash scripts/run_mek_resnet18_rafdb.sh
#   bash scripts/run_mek_resnet18_rafdb.sh --resume best.pth --eval-only
set -euo pipefail
cd "$(dirname "$0")/.."

RAF_ROOT="${RAF_ROOT:-/kaggle/input/raf-db-dataset/DATASET}"

python train_mek.py \
    --arch resnet18 \
    --dataset rafdb \
    --data-root "$RAF_ROOT" \
    "$@"
