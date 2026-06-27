#!/usr/bin/env bash
# Train ResNet-18 on FER-2013.
#
# Override the dataset location with FER_ROOT, e.g.
#   FER_ROOT=/data/fer2013 bash scripts/run_resnet18_fer2013.sh
# Pass any additional `train.py` flags after the script name, e.g.
#   bash scripts/run_resnet18_fer2013.sh --epochs 60 --no-tta
set -euo pipefail
cd "$(dirname "$0")/.."

FER_ROOT="${FER_ROOT:-/kaggle/input/datasets/msambare/fer2013}"

python train.py \
    --arch resnet18 \
    --dataset fer2013 \
    --data-root "$FER_ROOT" \
    "$@"
