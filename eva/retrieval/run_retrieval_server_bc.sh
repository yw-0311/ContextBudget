export CUDA_VISIBLE_DEVICES=6,7

# Set data directory relative to script location
DATA_DIR="$(cd "$(dirname "$0")/../data" && pwd)"

index_file="${DATA_DIR}/qwen3-embedding-8b"
corpus_dataset_path="${DATA_DIR}"
port=8000

retriever_name=qwen3-faiss
retriever_model_path="${DATA_DIR}/model"

python retrieval_server.py \
  --retriever_name qwen3-faiss \
  --index_path "$index_file" \
  --dataset_name "$corpus_dataset_path" \
  --topk 3 \
  --retriever_model "$retriever_model_path" \
  --port "$port" \
  --faiss_gpu