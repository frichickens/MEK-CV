#!/usr/bin/env bash
# Train ResNet-34 on FER-2013. See run_resnet18_fer2013.sh for env-var conventions.
set -euo pipefail
cd "$(dirname "$0")/.."

FER_ROOT="${FER_ROOT:-/kaggle/input/datasets/msambare/fer2013}"

python train.py \
    --arch resnet34 \
    --dataset fer2013 \
    --data-root "$FER_ROOT" \
    "$@"
