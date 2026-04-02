#!/bin/bash
set -euo pipefail

# Set PYTHONPATH
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EVA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODULES_DIR="$(cd "$(dirname "$0")" && pwd)"
VERL_ROOT="$(cd "$(dirname "$0")/../../verl" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${EVA_ROOT}:${MODULES_DIR}:${VERL_ROOT}:${PYTHONPATH}"

SGLANG_URL="http://127.0.0.1:30000"

# =========================
# Argument Parsing
# =========================
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --agent_type) AGENT_TYPE="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --max_depth) MAX_DEPTH="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --output) OUTPUT_FILE="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --model_path PATH          Model path (required)"
      echo "  --agent_type TYPE         Agent Loop type (required)"
      echo "  --dataset NAME             Dataset name (required, bc)"
      echo "  --max_depth N              Max depth (default: 10)"
      echo "  --target N                 Target count (required, 1)"
      echo "  --max_model_len N          Max model length (default: 8192)"
      echo "  --out_dir PATH             Output directory (optional)"
      echo "  --output PATH              Output file path (optional, higher priority than --out_dir)"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# =========================
# Sanity checks
# =========================
if [ -z "$MODEL_PATH" ]; then
  echo "[ERROR] MODEL_PATH not provided"
  exit 1
fi

if [ -z "$AGENT_TYPE" ]; then
  echo "[ERROR] AGENT_TYPE not provided"
  exit 1
fi

if [ -z "$DATASET" ]; then
  echo "[ERROR] DATASET not provided"
  exit 1
fi

if [ -z "$TARGET" ]; then
  echo "[ERROR] TARGET not provided"
  exit 1
fi

# Default values
MAX_DEPTH="${MAX_DEPTH:-10}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
OUT_DIR="${OUT_DIR:-../outputs}"
OUTPUT_FILE="${OUTPUT_FILE:-}"

# =========================
# Dataset Paths for BC
# =========================
DATA_ROOT="$(cd "$(dirname "$0")/../data" && pwd)"
BASELINE_PATH_1="${DATA_ROOT}/processed_data_bc/search_r1_processed/test.parquet"
ELASTIC_PATH_1="${DATA_ROOT}/processed_data_bc/elastic_processed/test.parquet"

# =========================
# Dataset selection
# =========================
case "${DATASET}" in
  bc)
    if [[ "$TARGET" == "1" ]]; then
      PARQUET_BASELINE="$BASELINE_PATH_1"
      PARQUET_ELASTIC="$ELASTIC_PATH_1"
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
# Agent logic
# =========================
case "${AGENT_TYPE}" in
  baseline_loop|baseline_loop_wbudget)
    PARQUET="$PARQUET_BASELINE"
    TOOL_CFG="${VERL_ROOT}/train/configs/tools/search_tool_base_config.yaml"
    ;;
  branch_loop|branch_loop_wob)
    PARQUET="$PARQUET_ELASTIC"
    TOOL_CFG="${VERL_ROOT}/train/configs/tools/search_tool_mo_config.yaml"
    ;;
  summary_loop)
    PARQUET="$PARQUET_ELASTIC"
    TOOL_CFG="${VERL_ROOT}/train/configs/tools/search_tool_bacm_config.yaml"
    ;;
  *)
    echo "[ERROR] Unknown AGENT_TYPE='${AGENT_TYPE}'"
    exit 1
    ;;
esac

# =========================
# Step name
# =========================
STEP="step0"
if [[ "$MODEL_PATH" =~ global_step_([0-9]+) ]]; then
  STEP="step_${BASH_REMATCH[1]}"
fi

# Try to extract from experiments directory
MODEL_NAME=$(echo "$MODEL_PATH" | sed -nE 's#.*/experiments/[^/]+/([^/]+).*#\1#p')
# If not extracted
if [[ -z "$MODEL_NAME" ]]; then
    # Remove trailing /
    CLEAN_PATH="${MODEL_PATH%/}"
    # Take last 3 directory levels
    LAST_THREE=$(echo "$CLEAN_PATH" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
    MODEL_NAME="$LAST_THREE"
fi

echo "[INFO] ========================================="
echo "[INFO] Model name: ${MODEL_NAME}"
echo "[INFO] Model path: ${MODEL_PATH}"
echo "[INFO] Using model step: ${STEP}"
echo "[INFO] Dataset: ${DATASET}"
echo "[INFO] Target: ${TARGET}"
echo "[INFO] Agent type: ${AGENT_TYPE}"
echo "[INFO] Max depth: ${MAX_DEPTH}"
echo "[INFO] Max model len: ${MAX_MODEL_LEN}"
echo "[INFO] ========================================="

# If user specifies output file, use user-specified filename
if [ -n "$OUTPUT_FILE" ]; then
    OUT_JSONL="$OUTPUT_FILE"
    # Generate log filename
    LOG_FILE="${OUTPUT_FILE%.jsonl}.log"
    # Create output directory
    mkdir -p "$(dirname "$OUT_JSONL")"
else
    # Create output directory
    mkdir -p "$OUT_DIR"
    # Generate output filename
    SUFFIX="${STEP}_d${MAX_DEPTH}"
    OUT_JSONL="${OUT_DIR}/${MODEL_NAME}_${DATASET}_${AGENT_TYPE}_len${MAX_MODEL_LEN}_depth${MAX_DEPTH}_target${TARGET}_${SUFFIX}.jsonl"
    LOG_FILE="${OUT_DIR}/${MODEL_NAME}_${DATASET}_${AGENT_TYPE}_len${MAX_MODEL_LEN}_depth${MAX_DEPTH}_target${TARGET}_${SUFFIX}.log"
fi

echo "[INFO] Output JSONL: ${OUT_JSONL}"
echo "[INFO] Log file: ${LOG_FILE}"
echo "[INFO] Parquet: ${PARQUET}"

# If output file exists and size > 0, task completed, skip
if [[ -f "$OUT_JSONL" && -s "$OUT_JSONL" ]]; then
    echo "[SKIP] Output file already exists: ${OUT_JSONL}"
    exit 0
fi

# Build command
CMD="python3 \"${EVA_ROOT}/modules/infer_bc.py\" \
  --dataset \"${DATASET}\" \
  --parquet \"${PARQUET}\" \
  --agent_type \"${AGENT_TYPE}\" \
  --max_model_len \"${MAX_MODEL_LEN}\" \
  --limit 100000 \
  --out_jsonl \"${OUT_JSONL}\" \
  --sglang_url \"http://127.0.0.1:30000\" \
  --model_path \"${MODEL_PATH}\" \
  --tool_config_path \"${TOOL_CFG}\" \
  --tool_format \"hermes\" \
  --temperature 0.0 \
  --max_assistant_turns 100 \
  --max_parallel_calls 1 \
  --max_depth \"${MAX_DEPTH}\" \
  --max_tool_response_length 4096 \
  --tool_response_truncate_side left \
  --target \"${TARGET}\" \
  --concurrency 32"

# Enable budget if agent_type is not branch_loop_wob
if [ "${AGENT_TYPE}" != "branch_loop_wob" ]; then
  CMD="${CMD} --enable_budget"
fi

CMD="${CMD} > \"${LOG_FILE}\" 2>&1"

echo "[INFO] Executing command..."
echo "[INFO] ========================================="
eval $CMD

echo "[INFO] ========================================="
echo "[INFO] Done! Results saved to: ${OUT_JSONL}"
echo "[INFO] ========================================="