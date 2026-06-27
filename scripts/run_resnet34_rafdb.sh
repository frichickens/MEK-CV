#!/usr/bin/env bash
# Train ResNet-34 on RAF-DB. See run_resnet18_rafdb.sh for env-var conventions.
set -euo pipefail
cd "$(dirname "$0")/.."

RAF_ROOT="${RAF_ROOT:-/kaggle/input/raf-db-dataset/DATASET}"

python train.py \
    --arch resnet34 \
    --dataset rafdb \
    --data-root "$RAF_ROOT" \
    "$@"
