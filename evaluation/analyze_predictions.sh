#!/usr/bin/env bash
set -e

GRADE_TYPES=${1:-"C N P BCVA MTF OSI SR"}
read -r -a GRADE_ARGS <<< "${GRADE_TYPES//,/ }"
GRADE_LABEL=$(IFS=_; echo "${GRADE_ARGS[*]}")
N_BOOTSTRAP=${2:-5000}
SEED=${3:-42}

INPUT_ROOT=${INPUT_ROOT:-file/long_predictions}
OUTPUT_DIR=${OUTPUT_DIR:-file/stat_results}

mkdir -p log/analyze

python -u evaluation/analyze_main.py \
    --input-root "${INPUT_ROOT}" \
    --grade-type "${GRADE_ARGS[@]}" \
    --datasets internal external \
    --dataset-groups internal external S01 S09 \
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
    > "log/analyze/analyze_${GRADE_LABEL}.txt" 2>&1

echo "[*] Finished: ${GRADE_LABEL}"
echo "    log: log/analyze/analyze_${GRADE_LABEL}.txt"
if (( ${#GRADE_ARGS[@]} == 1 )); then
    echo "    output: ${OUTPUT_DIR}/${GRADE_ARGS[0]}"
else
    echo "    output: ${OUTPUT_DIR}"
fi


