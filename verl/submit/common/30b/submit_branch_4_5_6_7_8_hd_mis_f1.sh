#!/bin/bash
set -e
trap 'echo "❌ 任务失败: $JOB_NAME"; exit 1' ERR

# 基本配置
#export QUEUE="${QUEUE:-industry_algo_llm_public_pool}"

CONFIG_FILE="${CONFIG_FILE:-$HOME/.config/config.sh}"
[[ -f "$CONFIG_FILE" ]] && source "$CONFIG_FILE" || { echo "❌ 缺少配置: $CONFIG_FILE"; exit 1; }

# EXPERIMENT_NAME="BranchRL_30B_RL_EM_Reward"
EXPERIMENT_NAME="BRANCH_30B_RL_F1_RW_BS128_4_5_6_7_8_MIS"
WORLD_SIZE=16
VERL_NODE_SIZE=16

GPU_PER_NODE=8
GPU_NUMS=$((WORLD_SIZE * GPU_PER_NODE))

DATE=$(date +"%Y%m%d%H%M")
VERSION="v0.01"
#VERSION="V0"
# MODEL_NAME="Qwen3_30B_A3B" 
MODEL_NAME="Qwen3_30B_A3B_TK" 
RM_NAME=$MODEL_NAME
#RM_NAME="None"

# JOB 名称
JOB_NAME="${EXPERIMENT_NAME}__${MODEL_NAME}__${GPU_NUMS}GPUS__${DATE}"

# 用户参数
USER_PARAMS=(
  --verl_node_size="$VERL_NODE_SIZE"
  --world_size="$WORLD_SIZE"
  --job_name="$JOB_NAME"
)

# QUEUE="item_intelligence_h20_T04"
QUEUE="industry_algo_llm_lingjun"
# MDL 参数
MDL_ARGS=(
  --engine=xdl
  --queue="$QUEUE"
  --entry="submit/common/entry.py"
  --worker_count="$WORLD_SIZE"
  --file.cluster_file="config/cluster.json"
  --oss_access_id="$OSS_ACCESS_ID"
  --oss_access_key="$OSS_ACCESS_KEY"
  --oss_bucket="$OSS_BUCKET"
  --oss_endpoint="$OSS_ENDPOINT"
  --job_name="$JOB_NAME"
  --algo_name=pytorch260
  --requirements_file_name="submit/common/requirements.txt"
  --oss_appendable=true
  --env=VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  --force
)

[[ -n "$OPENLM_TOKEN" ]] && MDL_ARGS+=(--env=OPENLM_TOKEN="$OPENLM_TOKEN")

# 启动任务
echo "🚀 启动 JOB: $JOB_NAME"
echo "🌐 队列: $QUEUE"
echo "🖥️ World size: $WORLD_SIZE, Verl nodes: $VERL_NODE_SIZE, 总 GPU: $GPU_NUMS"

nebulactl run mdl --user_params="${USER_PARAMS[*]}" "${MDL_ARGS[@]}"
