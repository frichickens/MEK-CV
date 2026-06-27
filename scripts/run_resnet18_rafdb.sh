#!/usr/bin/env bash
# Train ResNet-18 on RAF-DB.
#
# Default RAF_ROOT matches the popular `shuvoalok/raf-db-dataset` Kaggle layout
# (<root>/DATASET/{train,test}/<class_idx>/*.jpg). Override for any other path:
#   RAF_ROOT=/data/rafdb bash scripts/run_resnet18_rafdb.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RAF_ROOT="${RAF_ROOT:-/kaggle/input/raf-db-dataset/DATASET}"

python train.py \
    --arch resnet18 \
    --dataset rafdb \
    --data-root "$RAF_ROOT" \
    "$@"
