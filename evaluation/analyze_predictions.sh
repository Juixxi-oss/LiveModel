#!/usr/bin/env bash
set -e

GRADE_TYPE=${1:-C}
N_BOOTSTRAP=${2:-5000}
SEED=${3:-42}

INPUT_ROOT=${INPUT_ROOT:-file/long_predictions}
OUTPUT_DIR=${OUTPUT_DIR:-file/stat_results}

mkdir -p log/analyze

python -u evaluation/analyze_main.py \
    --input-root "${INPUT_ROOT}" \
    --grade-type "${GRADE_TYPE}" \
    --datasets internal external \
    --dataset-groups internal S01 S09 \
    --tasks \
        single_time \
        evaluator_padding \
        predictor_length \
        predictor_tag \
        v9_models \
        single_vs_multi_evaluator \
        single_vs_multi_predictor \
        generalization \
    --n-bootstrap "${N_BOOTSTRAP}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_DIR}" \
    > "log/analyze/analyze_${GRADE_TYPE}.txt" 2>&1

echo "[*] Finished: ${GRADE_TYPE}"
echo "    log: log/analyze/analyze_${GRADE_TYPE}.txt"
echo "    output: ${OUTPUT_DIR}/${GRADE_TYPE}"



