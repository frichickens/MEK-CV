#!/usr/bin/env bash
# Train MEK + ResNet-34 on RAF-DB. See run_mek_resnet18_rafdb.sh for env-vars.
set -euo pipefail
cd "$(dirname "$0")/.."

RAF_ROOT="${RAF_ROOT:-/kaggle/input/raf-db-dataset/DATASET}"

python train_mek.py \
    --arch resnet34 \
    --dataset rafdb \
    --data-root "$RAF_ROOT" \
    "$@"
