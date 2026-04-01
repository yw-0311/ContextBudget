# Model convert
BASE_DIR="/data/oss_bucket_0/shiyi/experiments/BRANCH_V2/BRANCH_30B_RL_LAJ_RW_BS128_8K_HD_MIS/global_step_60/actor"
python scripts/legacy_model_merger.py merge \
       --backend fsdp \
       --local_dir "$BASE_DIR" \
       --hf_model_path "$BASE_DIR/huggingface" \
       --target_dir "$BASE_DIR/hf"

# MOE 
## Baseline 
MODEL_PATH="/home/wy517954/model/Qwen/Qwen3-30B-A3B-Instruct-2507"

MODEL_PATH="/home/wy517954/model/Qwen/Qwen3-30B-A3B-Instruct-2507"
./scripts/infer.sh --model_path "$MODEL_PATH" --agent_type baseline_loop



## Ours wo budget
./scripts/infer.sh --model_path "$MODEL_PATH" --agent_type branch_loop_wob

## Ours 
./scripts/infer.sh --model_path "$MODEL_PATH" --agent_type branch_loop 

# Baseline RL
MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BASELINE_V2/BASELINE_30B_RL_LAJ_RW_BS128_8K_HD_MIS/global_step_300/actor/hf"
./scripts/infer.sh --model_path "$MODEL_PATH" --agent_type baseline_loop 

# Ours RL
## v2 60 step
MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V2/BRANCH_30B_RL_LAJ_RW_BS128_8K_HD_MIS/global_step_60/actor/hf"
./scripts/infer.sh --model_path "$MODEL_PATH" --agent_type branch_loop 

## v3 dynamic 4k 8k 480 step
MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V2/BRANCH_30B_RL_LAJ_RW_BS128_8K_HD_MIS_DYNAMICLEN/global_step_480/actor/hf"
./scripts/infer.sh --model_path "$MODEL_PATH" --agent_type branch_loop 

# Dense 4B
MODEL_PATH="/home/wy517954/model/Qwen/Qwen3-4B-Instruct-2507"
## baseline
./scripts/infer.sh --model_path "$MODEL_PATH" --agent_type baseline_loop
## Ours wo budget
./scripts/infer.sh --model_path "$MODEL_PATH" --agent_type branch_loop_wob
## Ours 
./scripts/infer.sh --model_path "$MODEL_PATH" --agent_type branch_loop


## baseline RL
MODEL_PATH=""
./scripts/infer.sh --model_path "$MODEL_PATH" --agent_type baseline_loop
## Ours wo budget RL
MODEL_PATH=""
./scripts/infer.sh --model_path "$MODEL_PATH" --agent_type branch_loop_wob
## Ours RL
MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V2/BRANCH_4B_RL_LAJ_RW_BS128_8K_HD_MIS/global_step_140/actor/hf"
./scripts/infer.sh --model_path "$MODEL_PATH" --agent_type branch_loop 


# BC Dataset
MODEL_PATH="/home/wy517954/model/Qwen/Qwen3-30B-A3B-Instruct-2507"

MODEL_PATH="/home/wy517954/model/Qwen/Qwen2.5-7B-Instruct"
./scripts/infer_bc.sh --model_path "$MODEL_PATH" --agent_type baseline_loop --dataset bc

# 7B Instrcut

# 7B ReACT 
## 2o 0.3+
MODEL_PATH="/data/oss_bucket_0/shiyi/model/Qwen2.5-7B-Instruct"
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type baseline_loop --dataset nq_hotpotqa --max_depth 10 --target 2
## 8o 0.73
MODEL_PATH="/data/oss_bucket_0/shiyi/model/Qwen2.5-7B-Instruct"
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type baseline_loop --dataset nq_hotpotqa --max_depth 10 --target 8


# 7B Branch v1 max_depth 10 
## 2 object 0.4833 
MODEL_PATH="/data/oss_bucket_0/shiyi/model/Qwen2.5-7B-Instruct"
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset nq_hotpotqa --max_depth 10 --target 2

## 8 object 0.7x
MODEL_PATH="/data/oss_bucket_0/shiyi/model/Qwen2.5-7B-Instruct"
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset nq_hotpotqa --max_depth 10 --target 8

# 7B Branch RL step100 
## 2 object  0.7+
MODEL_PATH="/data/oss_bucket_0/shiyi/model/ours_v2_8k_depth_high/hf/"


./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset nq_hotpotqa --max_depth 10 --target 2

## 8 object 2.337 
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset nq_hotpotqa --max_depth 10 --target 8

## 16 object 2.386 
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset nq_hotpotqa --max_depth 10 --target 16

## 32 object 1.056 
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset nq_hotpotqa --max_depth 10 --target 32


# Dynamic
MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_180/actor/hf/"

MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_220/actor/hf/"

MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_240/actor/hf"


MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_240/actor/hf"
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset nq_hotpotqa --max_depth 10 --target 32


# 7B Branch RL v3 step300 
## 2 object   0.9472
MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_8K_MIS/global_step_300/actor/hf"

./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset nq_hotpotqa --max_depth 10 --target 2
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset nq_hotpotqa --max_depth 10 --target 8



# MEM1 + Budget-aware 0.3+
MODEL_PATH="/home/wy517954/model/Mem-Lab"
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset nq_hotpotqa --max_depth 10 --target 2

# Search-R1 
MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BASELINE_V3/BASELINE_7B_RL_F1_RW_BS128_8K_MIS/global_step_140/actor/hf"

./scripts/infer_bc.sh --model_path "$MODEL_PATH" --agent_type baseline_loop --dataset bc --max_depth 10 --target 1

./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type baseline_loop --dataset nq_hotpotqa --max_depth 10 --target 2
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type baseline_loop --dataset nq_hotpotqa --max_depth 10 --target 8
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type baseline_loop --dataset nq_hotpotqa --max_depth 10 --target 16
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type baseline_loop --dataset nq_hotpotqa --max_depth 10 --target 32

MODEL_PATH="/mnt/nebula/cn-shanghai/juicefs/wy517954/model/openai"
./scripts/infer_gpt_oss.sh \
     --model_path $MODEL_PATH \
     --dataset nq_hotpotqa \
     --target 2 \
     --max_iterations 10 \
     --sglang_url http://127.0.0.1:30000


# Search-R1 wbudget
MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BASELINE_V3/BASELINE_7B_RL_F1_RW_BS128_8K_MIS/global_step_80/actor/hf"
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type baseline_loop_wbudget --dataset nq_hotpotqa --max_depth 10 --target 2
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type baseline_loop_wbudget --dataset nq_hotpotqa --max_depth 10 --target 8
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type baseline_loop_wbudget --dataset nq_hotpotqa --max_depth 10 --target 16
./scripts/infer_nq_hotpotqa.sh --model_path "$MODEL_PATH" --agent_type baseline_loop_wbudget --dataset nq_hotpotqa --max_depth 10 --target 32

# Tabel 3
# SR1
MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BASELINE_V3/BASELINE_7B_RL_F1_RW_BS128_8K_MIS/global_step_140/actor/hf"
./scripts/infer_bc.sh --model_path "$MODEL_PATH" --agent_type baseline_loop --dataset bc --max_depth 10 --target 1

# Ours
MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_240/actor/hf"
./scripts/infer_bc.sh --model_path "$MODEL_PATH" --agent_type branch_loop --dataset bc --max_depth 10 --target 1
