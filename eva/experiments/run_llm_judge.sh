#!/bin/bash
set -euo pipefail

# SGLang Server URL
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"

# Model path for LLM Judge
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/home/wy517954/model/Qwen2.5-7B-Instruct/}"

# Input directory containing JSONL files to be judged
INPUT_DIR="${INPUT_DIR:-../outputs}"

# Output directory for judged JSONL files
OUTPUT_DIR="${OUTPUT_DIR:-../outputs/judged}"

# Concurrency level
CONCURRENCY="${CONCURRENCY:-128}"

echo "========================================="
echo "LLM Judge Evaluation for BC Dataset"
echo "========================================="
echo "SGLang URL:      ${SGLANG_URL}"
echo "Judge Model:     ${JUDGE_MODEL_PATH}"
echo "Input Directory: ${INPUT_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Concurrency:     ${CONCURRENCY}"
echo "========================================="

# Check if input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "[ERROR] Input directory not found: ${INPUT_DIR}"
    exit 1
fi

# Check if there are any JSONL files
if [ -z "$(ls -A ${INPUT_DIR}/*.jsonl 2>/dev/null)" ]; then
    echo "[WARNING] No JSONL files found in ${INPUT_DIR}"
    exit 0
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run LLM Judge
python3 ../modules/evaluate_bc.py \
    --base-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --sglang-url "${SGLANG_URL}" \
    --model-path "${JUDGE_MODEL_PATH}" \
    --concurrency "${CONCURRENCY}"

echo ""
echo "========================================="
echo "LLM Judge Evaluation Complete!"
echo "========================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================="