MODEL_PATH="/data/oss_bucket_0/shiyi/experiments/BRANCH_V3/BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS/global_step_220/actor/hf/"
# JSONL_PATH="/home/wy517954/code/outputs/nq_hotpotqa_branch_loop_len8192_depth10_target16_step_100_d10.jsonl"
JSONL_PATH="/home/wy517954/code/outputs/BASELINE_7B_RL_F1_RW_BS128_8K_MIS_nq_hotpotqa_baseline_loop_len4096_depth20_target2_step_140_d20.jsonl"

PROJECT_ROOT="/home/wy517954/code/Elistic-Context-Fold-Verl/verl/"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python modules/eval_f1_offline.py $JSONL_PATH  \
    --tokenizer_path $MODEL_PATH \
    --target 2