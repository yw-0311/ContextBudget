#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-4}"
LOG_LEVEL="${LOG_LEVEL:-warning}"

CONTEXT_LEN="${CONTEXT_LEN:-32768}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"


echo "Starting SGLang server..."
echo "MODEL_PATH=${MODEL_PATH}"
echo "TP_SIZE=${TP_SIZE}"
echo "CONTEXT_LEN=${CONTEXT_LEN}"

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --log-level "${LOG_LEVEL}" \
  --context-length "${CONTEXT_LEN}" \
  --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \