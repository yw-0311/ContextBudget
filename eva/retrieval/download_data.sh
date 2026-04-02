export HF_ENDPOINT=https://hf-mirror.com

# Set data directory relative to script location
DATA_DIR="$(cd "$(dirname "$0")/../data" && pwd)"

# Test dataset
huggingface-cli download \
    --repo-type dataset \
    --local-dir "${DATA_DIR}/test" \
    --local-dir-use-symlinks False \
    Tevatron/browsecomp-plus

# browsecomp-plus-corpus
huggingface-cli download \
    --repo-type dataset \
    --local-dir "${DATA_DIR}" \
    --local-dir-use-symlinks False \
    Tevatron/browsecomp-plus-corpus

# Qwen3-Embedding-8B indexes
huggingface-cli download Tevatron/browsecomp-plus-indexes --repo-type=dataset --include="qwen3-embedding-8b/*" --local-dir "${DATA_DIR}"

# Qwen3-Embedding-8B
huggingface-cli download \
    --repo-type model \
    --local-dir "${DATA_DIR}/model" \
    --local-dir-use-symlinks False \
    Qwen/Qwen3-Embedding-8B