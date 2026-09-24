#!/usr/bin/env bash
# Optional future-training recipe. Existing paper outputs are not regenerated.
set -euo pipefail
SNAP_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SNAP_PROJECT_DIR="$(cd -- "${SNAP_SCRIPT_DIR}/.." && pwd)"
cd "${SNAP_PROJECT_DIR}"
export PYTHONPATH="${SNAP_PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Example: bash training/train_SnapRegressor_with_scheduler.sh \
#              --grade_type C --photo_type C --device cuda:0
# Separate output root prevents this new recipe replacing the legacy outputs.
exec python -u main/SnapRegressor_main.py \
    --num_folds 10 --num_epochs 50 --batch_size 64 --lr 5e-4 --seed 42 \
    --scheduler warmup_cosine --warmup_epochs 5 --start_factor 0.1 \
    --T_0 15 --T_mult 2 --eta_min 1e-7 \
    --result_root file/SnapRegressor_warmup_cosine "$@"
