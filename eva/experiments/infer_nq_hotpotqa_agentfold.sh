#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/wy517954/code/Elistic-Context-Fold-Verl/evaluation/eva"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export SWANLAB_API_KEY="jPvsjubltwXfluQst9wVs"

SGLANG_URL="http://127.0.0.1:30000"

# =========================
# Argument Parsing
# =========================
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --max_iteration) MAX_ITERATION="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --model_path PATH    Path to the model"
      echo "  --dataset NAME       Dataset name (nq_hotpotqa)"
      echo "  --target N           Target value (2, 8, 16, 32)"
      echo "  --max_iteration N    Max iterations (default: 6)"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# =========================
# Sanity checks
# =========================
if [ -z "${MODEL_PATH:-}" ]; then
  echo "[ERROR] MODEL_PATH not provided"
  exit 1
fi

if [ -z "${DATASET:-}" ]; then
  echo "[ERROR] DATASET not provided"
  exit 1
fi

if [ -z "${TARGET:-}" ]; then
  echo "[ERROR] TARGET not provided"
  exit 1
fi

# Set default max_iteration
MAX_ITERATION="${MAX_ITERATION:-6}"

# =========================
# Dataset Paths for nq_hotpotqa
# =========================
BASELINE_PATH_1="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_1/search_r1_processed/test.parquet"
BASELINE_PATH_2="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_2/search_r1_processed/test.parquet"
BASELINE_PATH_8="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_8/search_r1_processed/test.parquet"
BASELINE_PATH_16="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_16/search_r1_processed/test.parquet"
BASELINE_PATH_32="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_32/search_r1_processed/test.parquet"

# =========================
# Dataset selection
# =========================
case "${DATASET}" in
  nq_hotpotqa)
    if [[ "$TARGET" == "1" ]]; then
      PARQUET="$BASELINE_PATH_1"
    elif [[ "$TARGET" == "2" ]]; then
      PARQUET="$BASELINE_PATH_2"
    elif [[ "$TARGET" == "8" ]]; then
      PARQUET="$BASELINE_PATH_8"
    elif [[ "$TARGET" == "16" ]]; then
      PARQUET="$BASELINE_PATH_16"
    elif [[ "$TARGET" == "32" ]]; then
      PARQUET="$BASELINE_PATH_32"
    else
      echo "[ERROR] Unknown TARGET value: ${TARGET}"
      exit 1
    fi
    ;;
  *)
    echo "[ERROR] Unknown DATASET value: ${DATASET}"
    exit 1
    ;;
esac

# =========================
# Step name extraction
# =========================
STEP="step0"
if [[ "$MODEL_PATH" =~ global_step_([0-9]+) ]]; then
  STEP="step_${BASH_REMATCH[1]}"
fi

# Extract model name
MODEL_NAME=$(echo "$MODEL_PATH" | sed -nE 's#.*/experiments/[^/]+/([^/]+).*#\1#p')
if [[ -z "$MODEL_NAME" ]]; then
    CLEAN_PATH="${MODEL_PATH%/}"
    LAST_THREE=$(echo "$CLEAN_PATH" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
    MODEL_NAME="$LAST_THREE"
fi

echo "[INFO] Model name: ${MODEL_NAME}"
echo "[INFO] Using model step: ${STEP}"
echo "[INFO] Using dataset: ${DATASET}"
echo "[INFO] Using target: ${TARGET}"
echo "[INFO] Using max_iteration: ${MAX_ITERATION}"

# Output directory
OUT_DIR="/home/wy517954/code/outputs_agentfold"
mkdir -p "$OUT_DIR"

SUFFIX="${STEP}_iter${MAX_ITERATION}"
OUT_JSONL="${OUT_DIR}/${MODEL_NAME}_${DATASET}_agentfold_target${TARGET}_${SUFFIX}.jsonl"
LOG_FILE="${OUT_DIR}/${MODEL_NAME}_${DATASET}_agentfold_target${TARGET}_${SUFFIX}.log"

echo "[INFO] PARQUET= ${PARQUET}"
echo "[INFO] OUT_JSONL= ${OUT_JSONL}"
echo "[INFO] LOG_FILE= ${LOG_FILE}"

# Skip if output file exists
if [[ -f "$OUT_JSONL" && -s "$OUT_JSONL" ]]; then
    echo "[SKIP] Output file already exists: ${OUT_JSONL}"
    exit 0
fi

# Countdown
echo "[INFO] Cooling down..."
for i in {1..5}; do
    printf "\r⏳ Remaining: %3d seconds " "$((5-i))"
    sleep 1
done
printf "\n"

# Build command
CMD="python3 modules/infer_nq_hotpotqa_agentfold.py \
  --dataset \"${DATASET}\" \
  --parquet \"${PARQUET}\" \
  --limit 100000 \
  --out_jsonl \"${OUT_JSONL}\" \
  --sglang_url \"${SGLANG_URL}\" \
  --model_path \"${MODEL_PATH}\" \
  --max_iteration \"${MAX_ITERATION}\" \
  --temperature 0.0 \
  --target \"${TARGET}\" \
  --concurrency 32"

CMD="${CMD} > \"${LOG_FILE}\" 2>&1"

echo "[INFO] Executing command..."
eval $CMD

echo "[INFO] Done. Output saved to: ${OUT_JSONL}"
