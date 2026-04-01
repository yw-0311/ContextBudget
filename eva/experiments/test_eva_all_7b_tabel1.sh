#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# change server infer
DATASET="nq_hotpotqa"  # 数据集名称改为 nq_hotpotqa
TARGETS=("2" "8" "16" "32")
# TARGETS=("32")


declare -A MODEL_MAP=(
  # ["/data/oss_bucket_0/shiyi/model/Qwen2.5-7B-Instruct/"]="baseline_loop branch_loop summary_loop"
  # ["/data/oss_bucket_0/shiyi/model/Qwen3-30B-A3B-Instruct-2507/"]="baseline_loop branch_loop summary_loop"
  ["/data/oss_bucket_0/shiyi/model/qwen-235-awq/"]="baseline_loop summary_loop"
  # ["/data/oss_bucket_0/shiyi/model/Qwen2.5-7B-Instruct/"]="baseline_loop summary_loop"
  ["/data/oss_bucket_0/shiyi/model/Qwen2.5-7B-Instruct/"]="summary_loop"
  ["/data/oss_bucket_0/shiyi/model/Qwen3-30B-A3B-Instruct-2507/"]="baseline_loop summary_loop"
  # ours dynamic 7B
  ["/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_240/actor/hf"]="eager_loop"
  # ours dynamic 30B
  ["/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_30B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_200/actor/hf"]="branch_loop"
  # ours dynamic 30B
  ["/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_30B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_320/actor/hf"]="branch_loop"
  ["/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_30B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_300/actor/hf"]="branch_loop"
  # Search-R1 7B
  ["/data/oss_bucket_0/shiyi/experiments/BASELINE_V3/BASELINE_7B_RL_F1_RW_BS128_8K_MIS/global_step_140/actor/hf"]="baseline_loop"
  # Search-R1 30B
  ["/data/oss_bucket_0/shiyi/experiments/BASELINE_V3/BASELINE_30B_RL_F1_RW_BS128_8K_MIS/global_step_380/actor/hf"]="baseline_loop baseline_loop_wbudget"
  # fix budget
  ["/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_200/actor/hf"]="branch_loop"
  ["/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_30B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_200/actor/hf"]="branch_loop"
)

# 保留模型路径的顺序
MODEL_PATHS=(
  # "/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_8_MIS/global_step_200/actor/hf/"
  # "/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_30B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_320/actor/hf"
  # "/data/oss_bucket_0/shiyi/model/qwen-235-awq/"
  # "/data/oss_bucket_0/shiyi/model/Qwen2.5-7B-Instruct/"
  # "/data/oss_bucket_0/shiyi/model/Qwen3-30B-A3B-Instruct-2507/"
  # # Search-R1 7B
  # "/data/oss_bucket_0/shiyi/experiments/BASELINE_V3/BASELINE_7B_RL_F1_RW_BS128_8K_MIS/global_step_140/actor/hf"
  # # Search-R1 30B
  # "/data/oss_bucket_0/shiyi/experiments/BASELINE_V3/BASELINE_30B_RL_F1_RW_BS128_8K_MIS/global_step_380/actor/hf"
  # # Ours_Dynamic 30B
  # "/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_30B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_200/actor/hf"
  # "/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_30B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_300/actor/hf"
  # ours dynamic 7B
  "/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_240/actor/hf"
  # Ours_fix_8k 7B
  # "/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_8K_MIS/global_step_300/actor/hf/"
  # "/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_200/actor/hf"
  # "/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_30B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_200/actor/hf"
)

# -------------------------------
# max_depth sweep 配置（只改这里就行）
# -------------------------------
MAX_DEPTH_START=10
MAX_DEPTH_END=10
MAX_DEPTH_STEP=10

build_depths() {
  local start="$1" end="$2" step="$3"
  local d
  for ((d=start; d<=end; d+=step)); do
    echo "$d"
  done
}

# 启动服务：直接运行 python3 custom_server_mo.py
start_server() {
  local model_path="$1"
  local log_file="${model_path}_server_${DATASET}.log"

  echo "Starting server with model: ${model_path}"
  echo "Log file: ${log_file}"

  python3 scripts/custom_server_mo.py "$model_path" "$log_file" > /dev/null 2>&1 &
}

# 执行评测：这里新增传 --max_depth
run_inference() {
  local model_path="$1"
  local agent_type="$2"
  local dataset="$3"
  local max_depth="$4"
  local target="$5"

  echo "Running inference for model: ${model_path} agent_type: ${agent_type} dataset: ${dataset} max_depth: ${max_depth} target: ${target}"
  ./scripts/infer_nq_hotpotqa.sh \
    --model_path "$model_path" \
    --agent_type "$agent_type" \
    --dataset "$dataset" \
    --max_depth "$max_depth" \
    --target "$target"
}

check_health() {
  local host="$1"
  local port="$2"
  local last_health_status=""

  echo "Checking health of the server at ${host}:${port}..."

  while true; do
    if curl --silent --fail "${host}:${port}/health"; then
      if [[ "$last_health_status" != "healthy" ]]; then
        echo "Server is healthy!"
        last_health_status="healthy"
      fi
      return 0
    else
      if [[ "$last_health_status" != "unhealthy" ]]; then
        echo "Server not healthy, retrying..."
        last_health_status="unhealthy"
      fi
      sleep 5
    fi
  done
}

graceful_shutdown() {
  local port="$1"
  echo "Initiating graceful shutdown for service on port ${port}..."
  pid=$(fuser ${port}/tcp 2>/dev/null || true)
  if [ -n "$pid" ]; then
    echo "Found process ${pid} on port ${port}. Terminating..."
    kill -SIGTERM "$pid"
    while kill -0 "$pid" 2>/dev/null; do
      sleep 20
    done
    echo "Service on port ${port} has been gracefully shut down."
  else
    echo "No service found on port ${port}."
  fi
}

# 循环遍历每个模型路径并执行
for model_path in "${MODEL_PATHS[@]}"; do
  agent_types="${MODEL_MAP[$model_path]}"

  echo "Starting service for model: $model_path"
  log_file="${model_path}_server_${DATASET}.log"
  start_server "$model_path"

  echo "Currently serving model: ${model_path}, logs are being written to: ${log_file}"
  check_health "0.0.0.0" "30000"

  # 对于当前模型，遍历其所有需要执行的 agent_type
  for agent_type in $agent_types; do
    echo "Currently evaluating model: ${model_path}, agent type: ${agent_type}"

    # baseline_loop 只跑 max_depth=20；其他 sweep
    if [[ "${agent_type}" == "baseline_loop" ]]; then
      for target in "${TARGETS[@]}"; do
        run_inference "$model_path" "$agent_type" "$DATASET" 20 "$target"
      done
    else
      for target in "${TARGETS[@]}"; do
        while read -r d; do
          run_inference "$model_path" "$agent_type" "$DATASET" "$d" "$target"
        done < <(build_depths "$MAX_DEPTH_START" "$MAX_DEPTH_END" "$MAX_DEPTH_STEP")
      done
    fi
  done

  echo "Waiting for server to finish and evaluation to complete..."
  graceful_shutdown "30000"
  echo "Finished evaluation for model: $model_path"
  echo "Server and evaluation for model $model_path are complete!"
done

echo "All models have been evaluated!"
