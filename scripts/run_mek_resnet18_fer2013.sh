#!/usr/bin/env bash
# Train MEK + ResNet-18 on FER-2013.
#   FER_ROOT=/data/fer2013 bash scripts/run_mek_resnet18_fer2013.sh
#   bash scripts/run_mek_resnet18_fer2013.sh --epochs 60 --no-plot
#   bash scripts/run_mek_resnet18_fer2013.sh --resume mek_resnet18_fer2013_best.pth --eval-only
set -euo pipefail
cd "$(dirname "$0")/.."

FER_ROOT="${FER_ROOT:-/kaggle/input/datasets/msambare/fer2013}"

python train_mek.py \
    --arch resnet18 \
    --dataset fer2013 \
    --data-root "$FER_ROOT" \
    "$@"
