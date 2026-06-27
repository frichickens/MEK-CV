#!/usr/bin/env bash
# Run every (arch × dataset) combo sequentially and tee logs to logs/.
#
# Skips combos whose log already exists, so reruns can resume after a crash.
# Force a rerun with: FORCE=1 bash scripts/run_all.sh
#
# Override defaults:
#   FER_ROOT=/data/fer2013 RAF_ROOT=/data/rafdb bash scripts/run_all.sh
#   bash scripts/run_all.sh --no-tta            # appended to every run
set -euo pipefail
cd "$(dirname "$0")/.."

FER_ROOT="${FER_ROOT:-/kaggle/input/datasets/msambare/fer2013}"
RAF_ROOT="${RAF_ROOT:-/kaggle/input/raf-db-dataset/DATASET}"
FORCE="${FORCE:-0}"

mkdir -p logs

run() {
    local arch="$1" dataset="$2" root="$3"
    local tag="${arch}_${dataset}"
    local log="logs/${tag}.log"

    if [[ "$FORCE" != "1" && -s "$log" ]]; then
        echo "[skip] $tag — log already exists ($log). Set FORCE=1 to rerun."
        return 0
    fi

    echo "[run ] $tag → $log"
    python train.py \
        --arch "$arch" \
        --dataset "$dataset" \
        --data-root "$root" \
        --no-plot \
        "$@" \
        2>&1 | tee "$log"
}

# FER-2013 sweep
run resnet18 fer2013 "$FER_ROOT" "$@"
run resnet34 fer2013 "$FER_ROOT" "$@"

# RAF-DB sweep
run resnet18 rafdb   "$RAF_ROOT" "$@"
run resnet34 rafdb   "$RAF_ROOT" "$@"

echo
echo "All combos done. Per-run logs in logs/, checkpoints saved as <arch>_<dataset>_best.pth."
