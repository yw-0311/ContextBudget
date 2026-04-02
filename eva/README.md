# Context Budget Evaluation

## 1. Start SGLang Server

```bash
# Start server
export MODEL_PATH="/path/to/your/model" ./experiments/sglang_server.sh
```

**Check status:** `ps aux | grep sglang | grep -v grep`  
**Test API:** `curl http://127.0.0.1:30000/v1/models`

## 2. Setup Retriever Environment (Optional)

Both Multi-objective QA and BrowseComp-Plus require a retrieval server. Create a separate conda environment for the retriever:

```bash
conda create -n retriever python=3.10
conda activate retriever

# Install PyTorch with CUDA support
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install transformers datasets pyserini

# Install GPU version of FAISS for efficient retrieval
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# Install API server dependencies
pip install uvicorn fastapi
```

For more details, refer to: https://github.com/petergriffinjin/search-r1

## 3. Start Retrieval Server

### For Multi-objective QA

```bash
# Start retrieval server for Multi-objective QA
./retrieval/run_retrieval_server_wiki.sh
```

### For BrowseComp-Plus

```bash
# Start retrieval server for BrowseComp-Plus
./retrieval/run_retrieval_server_bc.sh
```

**Check status:** `ps aux | grep retrieval_server | grep -v grep`  
**Test API:** `curl http://127.0.0.1:8000/health`

## 4. Run Evaluation

### Search-R1 (Baseline)
```bash
MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BASELINE_V3/BASELINE_7B_RL_F1_RW_BS128_8K_MIS/global_step_140/actor/hf"

# Multi-objective QA (Target 2/8/16/32)
./experiments/infer_nq_hotpotqa_single.sh --model_path "$MODEL_PATH" --agent_type baseline_loop --dataset nq_hotpotqa --max_depth 10 --target 2 --output ./outputs/mq_sr1_t2.jsonl
./experiments/infer_nq_hotpotqa_single.sh --model_path "$MODEL_PATH" --agent_type baseline_loop --dataset nq_hotpotqa --max_depth 10 --target 8 --output ./outputs/mq_sr1_t8.jsonl
./experiments/infer_nq_hotpotqa_single.sh --model_path "$MODEL_PATH" --agent_type baseline_loop --dataset nq_hotpotqa --max_depth 10 --target 16 --output ./outputs/mq_sr1_t16.jsonl
./experiments/infer_nq_hotpotqa_single.sh --model_path "$MODEL_PATH" --agent_type baseline_loop --dataset nq_hotpotqa --max_depth 10 --target 32 --output ./outputs/mq_sr1_t32.jsonl

# BrowseComp-Plus
./experiments/infer_bc_single.sh --model_path "$MODEL_PATH" --agent_type baseline_loop --dataset bc --max_depth 10 --target 1 --output ./outputs/bcp_sr1.jsonl
```

### Ours
```bash
MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_240/actor/hf"

# Multi-objective QA
./experiments/infer_nq_hotpotqa_single.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset nq_hotpotqa --max_depth 10 --target 2 --output ./outputs/mq_ours_t2.jsonl
./experiments/infer_nq_hotpotqa_single.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset nq_hotpotqa --max_depth 10 --target 8 --output ./outputs/mq_ours_t8.jsonl
./experiments/infer_nq_hotpotqa_single.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset nq_hotpotqa --max_depth 10 --target 16 --output ./outputs/mq_ours_t16.jsonl
./experiments/infer_nq_hotpotqa_single.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset nq_hotpotqa --max_depth 10 --target 32 --output ./outputs/mq_ours_t32.jsonl

# BrowseComp-Plus
./experiments/infer_bc_single.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset bc --max_depth 10 --target 1 --output ./outputs/bcp_ours.jsonl
```

## 5. Evaluation

```bash
# Multi-objective QA
python3 ./modules/evaluate_mq.py --input_jsonl ./outputs/mq_ours_t2.jsonl --tokenizer_path "$MODEL_PATH" --target 2

# BrowseComp-Plus (start Judge Server first)
export MODEL_PATH="/path/to/qwen3-32B" ./experiments/sglang_server.sh
./experiments/run_llm_judge.sh --input-dir ./outputs --output-dir ./outputs/judged
```

## Scripts

- `evaluate_mq.py` - Evaluate Multi-objective QA (F1, EM, Token Cost, Branch Depth)
- `evaluate_bc.py` - Evaluate BrowseComp-Plus (LLM Judge)
- `infer_nq_hotpotqa_single.sh` - Run inference on Multi-objective QA
- `infer_bc_single.sh` - Run inference on BrowseComp-Plus
- `run_llm_judge.sh` - Run LLM Judge evaluation for BrowseComp-Plus
- `sglang_server.sh` - Start SGLang inference server
- `run_retrieval_server_bc.sh` - Start Retrieval server for BrowseComp-Plus