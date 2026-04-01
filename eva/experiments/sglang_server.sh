#!/usr/bin/env bash
set -euo pipefail

# MODEL_PATH="/home/wy517954/model/Qwen/Qwen2.5-7B-Instruct"

MODEL_PATH="/data/oss_bucket_0/shiyi/model/Qwen3-32B/Qwen3-32B"

# MODEL_PATH="/data/oss_bucket_0/shiyi/model/qwen-235-awq/"

# MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_220/actor/hf/"

# 7b ours dynamic
# MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_240/actor/hf"

# 30b ours dynamic
# MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_30B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_200/actor/hf"

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# MODEL_PATH="/mnt/nebula/cn-shanghai/juicefs/wy517954/model/Qwen3-VL-32B-Instruct"

# MODEL_PATH="/mnt/nebula/cn-shanghai/juicefs/wy517954/model/openai"

# MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_8K_MIS/global_step_200/actor/hf"

# 7b search-r1
# MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BASELINE_V3/BASELINE_7B_RL_F1_RW_BS128_8K_MIS/global_step_140/actor/hf"

# MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_240/actor/hf"




HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-4}"
LOG_LEVEL="${LOG_LEVEL:-warning}"

# CONTEXT_LEN="${CONTEXT_LEN:-131072}"
# MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
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