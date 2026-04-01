#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Dataset configuration
DATASET="nq_hotpotqa"
TARGETS=("2" "8" "16" "32")

# Model paths to evaluate
# Add your model paths here
MODEL_PATHS=(
  # Example paths - modify according to your needs
  # "/data/oss_bucket_0/shiyi/model/Qwen2.5-7B-Instruct/"
  # "/data/oss_bucket_0/shiyi/experiments/BASELINE_V3/BASELINE_7B_RL_F1_RW_BS128_8K_MIS/global_step_140/actor/hf"
)

# Max iterations to sweep
MAX_ITERATIONS=("6")

# -----------------------------
# Helper functions
# -----------------------------

start_server() {
  local model_path="$1"
  local log_file="${model_path}_server_${DATASET}.log"

  echo "Starting server with model: ${model_path}"
  echo "Log file: ${log_file}"

  python3 scripts/custom_server_mo.py "$model_path" "$log_file" > /dev/null 2>&1 &
}

run_inference() {
  local model_path="$1"
  local dataset="$2"
  local target="$3"
  local max_iteration="$4"

  echo "Running inference for model: ${model_path}"
  echo "  dataset: ${dataset}"
  echo "  target: ${target}"
  echo "  max_iteration: ${max_iteration}"

  ./scripts/infer_nq_hotpotqa_agentfold.sh \
    --model_path "$model_path" \
    --dataset "$dataset" \
    --target "$target" \
    --max_iteration "$max_iteration"
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

# -----------------------------
# Main evaluation loop
# -----------------------------

if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
  echo "[WARNING] No model paths specified. Please edit MODEL_PATHS in this script."
  exit 0
fi

for model_path in "${MODEL_PATHS[@]}"; do
  echo "=================================================="
  echo "Evaluating model: $model_path"
  echo "=================================================="

  # Start server
  log_file="${model_path}_server_${DATASET}.log"
  start_server "$model_path"

  echo "Currently serving model: ${model_path}"
  echo "Logs: ${log_file}"

  # Wait for server to be ready
  check_health "0.0.0.0" "30000"

  # Run inference for each target and max_iteration
  for target in "${TARGETS[@]}"; do
    for max_iter in "${MAX_ITERATIONS[@]}"; do
      run_inference "$model_path" "$DATASET" "$target" "$max_iter"
    done
  done

  # Shutdown server
  echo "Waiting for server to finish and evaluation to complete..."
  graceful_shutdown "30000"
  echo "Finished evaluation for model: $model_path"
  echo ""
done

echo "All models have been evaluated!"
