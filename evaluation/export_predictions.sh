#!/usr/bin/env bash

set -e

GRADE_TYPE=${1:-C}
NUM_FOLDS=${2:-10}
SEED=${3:-42}

mkdir -p log/export

INTERNAL_GPU=${INTERNAL_GPU:-cuda:5}
EXTERNAL_GPU=${EXTERNAL_GPU:-cuda:1}

COMMON_ARGS="
    --grade_type ${GRADE_TYPE}
    --num_folds ${NUM_FOLDS}
    --seed ${SEED}
    --model_family efficientnet
    --backbone_type b0
    --features_size 256
    --resize_size 224
    --crop_size 224
    --batch_size_single 64
    --batch_size_series 8
    --num_workers 4
    --num_heads 4
    --hidden_size 256
    --num_layers 2
    --batch_first
"

nohup python -u evaluation/export_main.py \
    --device ${INTERNAL_GPU} \
    --dataset_name internal \
    ${COMMON_ARGS} \
    > log/export/export_${GRADE_TYPE}_internal.txt 2>&1 &

nohup python -u evaluation/export_main.py \
    --device ${EXTERNAL_GPU} \
    --dataset_name external \
    ${COMMON_ARGS} \
    > log/export/export_${GRADE_TYPE}_external.txt 2>&1 &

echo "[*] Export jobs submitted."
echo "    internal log: log/export/export_${GRADE_TYPE}_internal.txt"
echo "    external log: log/export/export_${GRADE_TYPE}_external.txt"