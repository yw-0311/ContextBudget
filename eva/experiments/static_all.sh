#!/bin/bash
set -e
set -x

PROJECT_ROOT="/home/wy517954/code/Elistic-Context-Fold-Verl/verl/"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ENABLE_LLM_JUDGE=1 SGLANG_URL=http://localhost:30000 ./static_all.sh

export TOKENIZER_PATH_7B=/data/oss_bucket_0/shiyi/model/Qwen2.5-7B-Instruct/
export TOKENIZER_PATH_30B=/data/oss_bucket_0/shiyi/model/Qwen3-30B-A3B-Instruct-2507/

# LLM Judge configuration (optional)
# Set ENABLE_LLM_JUDGE=1 to enable LLM-as-judge evaluation
# Set SGLANG_URL to your sglang server URL (default: http://localhost:30000)
# Set JUDGE_MODEL_PATH to the model path for LLM judge (default: Qwen2.5-7B-Instruct)

ENABLE_LLM_JUDGE="0"
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/data/oss_bucket_0/shiyi/model/Qwen2.5-7B-Instruct/}"

OUTPUTS_DIR="/home/wy517954/code/ourputs_eager_loop_q10"

# Build command arguments
BASE_ARGS="--by-datasource"

if [ "$ENABLE_LLM_JUDGE" = "1" ]; then
    echo "LLM Judge enabled - using server: $SGLANG_URL"
    BASE_ARGS="$BASE_ARGS --enable-llm-judge --sglang-url $SGLANG_URL --judge-model-path $JUDGE_MODEL_PATH"
fi

# Iterate through each subdirectory in outputs
for subdir in "$OUTPUTS_DIR"/*; do
    if [ -d "$subdir" ]; then
        echo "Processing directory: $subdir"
        # Check if evaluation_results.csv already exists
        if [ -f "$subdir/evaluation_results.csv" ]; then
            echo "Skipping directory (CSV already exists): $subdir"
            continue
        fi
        python modules/eval_dir_to_csv.py "$subdir" $BASE_ARGS
    fi
done
