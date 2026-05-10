#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-cifar10}"
MODEL="${2:-vit_tiny}"
EPOCHS="${3:-100}"
BATCH_SIZE="${4:-256}"
OUT="${5:-results}"

for INIT in trunc_normal mimetic structured_imp3; do
  python code/train.py \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --init "${INIT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr 1e-3 \
    --weight-decay 0.01 \
    --amp \
    --output-dir "${OUT}" \
    --run-name "${DATASET}_${MODEL}_${INIT}_${EPOCHS}ep"
done

python code/plot_runs.py "${OUT}/${DATASET}_${MODEL}_"*"_${EPOCHS}ep" \
  --output "${OUT}/${DATASET}_${MODEL}_${EPOCHS}ep_comparison.png"
