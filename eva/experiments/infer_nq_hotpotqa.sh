#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/wy517954/code/Elistic-Context-Fold-Verl/verl/"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export SWANLAB_API_KEY="jPvsjubltwXfluQst9wVs"
# export LOCAL=True

SGLANG_URL="http://127.0.0.1:30000"

# =========================
# Model length sweep
# =========================
MAX_MODEL_LENS=(8192 16384 14336 12288 10240 6144 4096)
# MAX_MODEL_LENS=(8192)

# MAX_MODEL_LENS=(4096)
# MAX_MODEL_LENS=(8192) 
# MAX_MODEL_LENS=(16384)
# MAX_MODEL_LENS=(4096)

# MAX_MODEL_LENS=(131072)
# MAX_MODEL_LENS=(130000)

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
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --model_path PATH"
      echo "  --agent_type TYPE"
      echo "  --dataset NAME (nq_hotpotqa)"
      echo "  --max_depth N (default: 20)"
      echo "  --target N (2, 8, 16)"
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

# =========================
# Dataset Paths for nq_hotpotqa
# =========================

# 数据集路径，针对不同目标选择不同的文件
BASELINE_PATH_1="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_1/search_r1_processed/test.parquet"
ELASTIC_PATH_1="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_1/elastic_processed/test.parquet"

BASELINE_PATH_2="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_2/search_r1_processed/test.parquet"
ELASTIC_PATH_2="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_2/elastic_processed/test.parquet"

BASELINE_PATH_8="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_8/search_r1_processed/test.parquet"
ELASTIC_PATH_8="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_8/elastic_processed/test.parquet"

BASELINE_PATH_16="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_16/search_r1_processed/test.parquet"
ELASTIC_PATH_16="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_16/elastic_processed/test.parquet"

BASELINE_PATH_32="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_32/search_r1_processed/test.parquet"
ELASTIC_PATH_32="/home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data/nq_hotpotqa_train_multi_32/elastic_processed/test.parquet"


# =========================
# Dataset selection
# =========================
case "${DATASET}" in
  nq_hotpotqa)
    if [[ "$TARGET" == "1" ]]; then
      PARQUET_BASELINE="$BASELINE_PATH_1"
      PARQUET_ELASTIC="$ELASTIC_PATH_1"
    elif [[ "$TARGET" == "2" ]]; then
      PARQUET_BASELINE="$BASELINE_PATH_2"
      PARQUET_ELASTIC="$ELASTIC_PATH_2"
    elif [[ "$TARGET" == "8" ]]; then
      PARQUET_BASELINE="$BASELINE_PATH_8"
      PARQUET_ELASTIC="$ELASTIC_PATH_8"
    elif [[ "$TARGET" == "16" ]]; then
      PARQUET_BASELINE="$BASELINE_PATH_16"
      PARQUET_ELASTIC="$ELASTIC_PATH_16"
    elif [[ "$TARGET" == "32" ]]; then
      PARQUET_BASELINE="$BASELINE_PATH_32"
      PARQUET_ELASTIC="$ELASTIC_PATH_32"
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
    TOOL_CFG="/home/wy517954/code/Elistic-Context-Fold-Verl/verl/submit/experiments/search_r1/configs/tools/search_tool_base_mo_config.yaml"
    ;;
  branch_loop|branch_loop_wob)
    PARQUET="$PARQUET_ELASTIC"
    TOOL_CFG="/home/wy517954/code/Elistic-Context-Fold-Verl/verl/submit/experiments/search_r1/configs/tools/search_tool_mo_config.yaml"
    ;;
  summary_loop)
    # PARQUET="$PARQUET_ELASTIC"
    PARQUET="$PARQUET_BASELINE"
    TOOL_CFG="/home/wy517954/code/Elistic-Context-Fold-Verl/verl/submit/experiments/search_r1/configs/tools/search_tool_base_mo_config.yaml"
    ;;
  eager_loop)
    PARQUET="$PARQUET_ELASTIC"
    TOOL_CFG="/home/wy517954/code/Elistic-Context-Fold-Verl/verl/submit/experiments/search_r1/configs/tools/search_tool_mo_config.yaml"
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

# MODEL_NAME=$(basename "$MODEL_PATH")
# # 如果 MODEL_PATH 指向 global_step_XXX 子目录，则取父目录名作为模型名
# if [[ "$MODEL_NAME" =~ ^global_step_[0-9]+$ ]]; then
#     MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
# fi

# 尝试从 experiments 目录提取
MODEL_NAME=$(echo "$MODEL_PATH" | sed -nE 's#.*/experiments/[^/]+/([^/]+).*#\1#p')
# 如果没提取到
if [[ -z "$MODEL_NAME" ]]; then
    # 去掉结尾可能的 /
    CLEAN_PATH="${MODEL_PATH%/}"
    
    # 取最后3级目录
    LAST_THREE=$(echo "$CLEAN_PATH" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
    
    MODEL_NAME="$LAST_THREE"
fi


echo "[INFO] Model name: ${MODEL_NAME}"
echo "[INFO] Using model step: ${STEP}"
echo "[INFO] Using dataset: ${DATASET}"
echo "[INFO] Using target: ${TARGET}"er 
echo "[INFO] Using max_depth: ${MAX_DEPTH}"


# OUT_DIR="/data/oss_bucket_0/shiyi/eva_log"
# OUT_DIR="/home/wy517954/code/outputs_new30b"
# OUT_DIR="/home/wy517954/code/outputs_random"
# OUT_DIR="/home/wy517954/code/outputs_summary"
# OUT_DIR="/home/wy517954/code/outputs_235_v2"
# OUT_DIR="/home/wy517954/code/outputs/outputs_7b_wobudget"
# OUT_DIR="/home/wy517954/code/outputs/outputs_7b_wobudget"
# OUT_DIR="/home/wy517954/code/outputs/7b_outputs_summary"
OUT_DIR="/home/wy517954/code/ourputs_eager_v6"


# Create output directory if not exists
mkdir -p "$OUT_DIR"


for MAX_MODEL_LEN in "${MAX_MODEL_LENS[@]}"; do
  SUFFIX="${STEP}_d${MAX_DEPTH}"
  # OUT_JSONL="${OUT_DIR}/${DATASET}_${AGENT_TYPE}_len${MAX_MODEL_LEN}_depth${MAX_DEPTH}_target${TARGET}_${SUFFIX}.jsonl"
  # LOG_FILE="${OUT_DIR}/${DATASET}_${AGENT_TYPE}_len${MAX_MODEL_LEN}_depth${MAX_DEPTH}_target${TARGET}_${SUFFIX}.log"
  OUT_JSONL="${OUT_DIR}/${MODEL_NAME}_${DATASET}_${AGENT_TYPE}_len${MAX_MODEL_LEN}_depth${MAX_DEPTH}_target${TARGET}_${SUFFIX}.jsonl"
  LOG_FILE="${OUT_DIR}/${MODEL_NAME}_${DATASET}_${AGENT_TYPE}_len${MAX_MODEL_LEN}_depth${MAX_DEPTH}_target${TARGET}_${SUFFIX}.log"

  echo "[INFO] MAX_MODEL_LEN=${MAX_MODEL_LEN}"
  echo "[INFO] PARQUET= ${PARQUET}"
  echo "[INFO] OUT_JSONL= ${OUT_JSONL}"
  echo "[INFO] LOG_FILE= ${LOG_FILE}"

  # 如果输出文件已存在且大小>0，认为任务已完成，跳过
  if [[ -f "$OUT_JSONL" && -s "$OUT_JSONL" ]]; then
      echo "[SKIP] 输出文件已存在: ${OUT_JSONL}"
      echo "[SKIP] 跳过 MAX_MODEL_LEN=${MAX_MODEL_LEN}"
      echo "----------------------------------------"
      continue
  fi
  
  # --- 倒计时部分 ---
  echo "[INFO] 冷却 0 分钟..."
  for i in {1..1}; do
      printf "\r⏳ 剩余等待时间：%3d 秒 " "$i"
      sleep 1
  done
  printf "\n" # 确保下一轮日志从新行开始

  CMD="python3 modules/infer_nq_hotpotqa.py \
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
  eval $CMD
done
