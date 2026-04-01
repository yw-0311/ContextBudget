import os
import argparse
import time
import sys, re
from typing import Tuple, Any
from openlm_hub import repo_download

# JOB_NAME="${EXPERIMENT_NAME}_${MODEL_NAME}_${GPU_NUMS}GPUS_${DATE}"

def parse_job_name(job_name: str) -> tuple[str | Any, ...]:
    """
    解析 JOB_NAME，返回：
    (EXPERIMENT_NAME, MODEL_NAME, GPU_NUMS, DATE)
    """
    # 正则表达式，贪婪匹配前面三个字段（可能含有 _）
    pattern = r"^(.*?)__(.*?)__(.*?)GPUS_(.*?)$"
    match = re.match(pattern, job_name)
    if not match:
        raise ValueError(f"无法解析 JOB_NAME: {job_name}")
    return match.groups()


def get_script_path() -> str:
    # 7B Branch
    if EXPERIMENT_NAME == "BRANCH_7B_RL_F1_RW_BS128_8K_MIS" :
        script = f"submit/experiments/branch_rl/7b/branch_train_8k_hd_mis_f1.sh"
        print(f"使用 branch_train_8k_hd_mis_f1 脚本: {script}")
        return script
    if EXPERIMENT_NAME == "BRANCH_7B_RL_F1_RW_BS128_4_5_6_7_8_MIS" :
        script = f"submit/experiments/branch_rl/7b/branch_train_4_5_6_7_8_hd_mis_f1.sh"
        print(f"使用 branch_train_4_5_6_7_8_hd_mis_f1 脚本: {script}")
        return script
    # 7B Baseline 
    if EXPERIMENT_NAME == "BASELINE_7B_RL_F1_RW_BS128_8K_MIS" :
        script = f"submit/experiments/search_r1/7b/baseline_train_8k_hd_mis_f1.sh"
        print(f"使用 baseline_train_8k_hd_mis_f1 脚本: {script}")
        return script
    # 30B Branch
    if EXPERIMENT_NAME == "BRANCH_30B_RL_F1_RW_BS128_4_5_6_7_8_MIS" :
        script = f"submit/experiments/branch_rl/30b/branch_train_4_5_6_7_8_hd_mis_f1.sh"
        print(f"使用 branch_train_4_5_6_7_8_hd_mis_f1 脚本: {script}")
        return script
    if EXPERIMENT_NAME == "BRANCH_30B_RL_F1_RW_BS128_16_4_MIS":
        script = f"submit/experiments/branch_rl/30b/branch_train_16_4_hd_mis_f1.sh"
        print(f"使用 branch_train_16_4_hd_mis_f1 脚本: {script}")
        return script
    # 30B Baseline
    if EXPERIMENT_NAME == "BASELINE_30B_RL_F1_RW_BS128_8K_MIS" :
        script = f"submit/experiments/search_r1/30b/baseline_train_8k_hd_mis_f1.sh"
        print(f"使用 baseline_train_8k_hd_mis_f1 脚本: {script}")
        return script
        
    return script

# def get_script_path() -> str:
#     # TEST
#     if EXPERIMENT_NAME == "baseline_test_laj" :
#         script = f"submit/experiments/test/baseline_test_laj.sh"
#         print(f"使用 baseline_test_laj 脚本: {script}")
#         return script
#     # baseline    
#     if EXPERIMENT_NAME == "BASELINE_30B_RL_EM_RW_GSPO_BS128" :
#         script = f"submit/experiments/search_r1/baseline_train.sh"
#         print(f"使用 baseline_train 脚本: {script}")
#         return script
#     if EXPERIMENT_NAME == "BASELINE_30B_RL_EM_RW_GSPO_BS128_8K" :
#         script = f"submit/experiments/search_r1/baseline_train_8k.sh"
#         print(f"使用 baseline_train 脚本: {script}")
#         return script
#     if EXPERIMENT_NAME == "BRANCH_30B_RL_EM_RW_BS128_8K_HD_MIS_LAJ":
#         script = f"submit/experiments/search_r1/baseline_train_8k_hd_mis_laj.sh"
#         print(f"使用 branch_eva 脚本: {script}")
#         return script
#     if EXPERIMENT_NAME == "BASELINE_30B_EM_RW_EVA" :
#         script = f"submit/experiments/search_r1/baseline_eva.sh"
#         print(f"使用 baseline_eva 脚本: {script}")
#         return script
#     # branch method
#     if EXPERIMENT_NAME == "BRANCH_30B_RL_EM_RW_GSPO_BS128" :
#         script = f"submit/experiments/branch_rl/branch_train.sh"
#         print(f"使用 branch_train 脚本: {script}")
#         return script
#     if EXPERIMENT_NAME == "BRANCH_30B_RL_EM_RW_GSPO_BS128_8K" :
#         script = f"submit/experiments/branch_rl/branch_train_8k.sh"
#         print(f"使用 branch_train 脚本: {script}")
#         return script
#     if EXPERIMENT_NAME == "BRANCH_30B_RL_EM_RW_GSPO_BS128_8K_CLIP_NORM" :
#         script = f"submit/experiments/branch_rl/branch_train_8k_clip_norm.sh"
#         print(f"使用 branch_train 脚本: {script}")
#         return script
#     if EXPERIMENT_NAME == "BRANCH_30B_RL_EM_RW_GSPO_BS128_8K_CLIP_NORM_HD" :
#         script = f"submit/experiments/branch_rl/branch_train_8k_clip_norm_hd.sh"
#         print(f"使用 branch_train 脚本: {script}")
#         return script
#     if EXPERIMENT_NAME == "BRANCH_30B_RL_EM_RW_GSPO_BS128_8K_CLIP_HD" :
#         script = f"submit/experiments/branch_rl/branch_train_8k_clip_hd.sh"
#         print(f"使用 branch_train 脚本: {script}")
#         return script
#     if EXPERIMENT_NAME == "BRANCH_30B_RL_EM_RW_GSPO_BS128_8K_CLIP_HD_MD30" :
#         script = f"submit/experiments/branch_rl/branch_train_8k_clip_hd_md30.sh"
#         print(f"使用 branch_train 脚本: {script}")
#         return script
#     if EXPERIMENT_NAME == "BRANCH_30B_RL_EM_RW_GSPO_BS128_8K_CLIP_HD_MIS" :
#         script = f"submit/experiments/branch_rl/branch_train_8k_clip_hd_mis.sh"
#         print(f"使用 branch_train 脚本: {script}")
#         return script
#     if EXPERIMENT_NAME == "BRANCH_30B_RL_EM_RW_GSPO_BS128_8K_CLIP_HD_MIS_LAJ" :
#         script = f"submit/experiments/branch_rl/branch_train_8k_clip_hd_mis_laj.sh"
#         print(f"使用 branch_train 脚本: {script}")
#         return script
#     if EXPERIMENT_NAME == "BRANCH_30B_RL_EM_RW_GSPO_BS128_8K_CLIP_0.0003_0.0004_NORM_WOB" :
#         script = f"submit/experiments/branch_rl/branch_train_8k_clip_0.0003_0.0004_norm_wob.sh"
#         print(f"使用 branch_train 脚本: {script}")
#         return script
#     if EXPERIMENT_NAME == "BASELINE_30B_RL_EM_RW_GSPO_BS128_8K_CLIP" :
#         script = f"submit/experiments/search_r1/baseline_train_8k_clip.sh"
#         print(f"使用 branch_train 脚本: {script}")
#         return script
#     if EXPERIMENT_NAME == "BRANCH_30B_EM_RW_EVA" :
#         script = f"submit/experiments/branch_rl/branch_eva.sh"
#         print(f"使用 branch_eva 脚本: {script}")
#         return script
#     return script



def download_train_model_if_needed(model_name: str, rank: int, verl_node_size: int) -> str:
    # Judge Node 不需要下载
    if rank >= verl_node_size:
        print(f"✅ [RANK {rank}] Skip train model download (verl_node_size={verl_node_size})")
        return ""

    from openlm_hub import repo_download
    model_map = {
        "Qwen2_5_7B": "Qwen/Qwen2.5-7B-Instruct",
        "Qwen3-4B": "Qwen/Qwen3-4B-Instruct-2507",
        # "Qwen3_30B_A3B": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen3_30B_A3B_TK": "Qwen/Qwen3-30B-A3B-Thinking-2507"
    }

    for key, repo in model_map.items():
        if key in model_name:
            print(f"🚀 [RANK {rank}] Downloading {key} ...")
            model_path = repo_download(repo)
            return model_path

    print(f"⚠️ [RANK {rank}] 未匹配到需要下载的模型，跳过下载")
    return ""

def download_judge_model_if_needed(rank: int, verl_node_size: int) -> str:
    # actor Node 不需要下载
    if rank < verl_node_size:
        print(f"✅ [RANK {rank}] Skip judge model download (verl_node_size={verl_node_size})")
        return ""

    model_path = repo_download("Qwen/Qwen3-30B-A3B-Instruct-2507",max_workers=64)
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl_node_size", type=int, required=True)
    parser.add_argument("--search_node_size", type=int, required=False)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--job_name", type=str, required=True)
    args = parser.parse_args()


    EXPERIMENT_NAME, MODEL_NAME, GPU_NUMS, DATE = parse_job_name(args.job_name)

    script_path = get_script_path()

    # 获取环境变量 RANK，默认为 0
    rank = int(os.environ.get("RANK", 0))

    # 获取模型路径
    print(f"[DEBUG] downlad MODEL_NAME{MODEL_NAME}")
    actor_model_path = download_train_model_if_needed(MODEL_NAME, rank, args.verl_node_size)
    judge_model_name_path=""
    # judge_model_name_path = download_judge_model_if_needed(rank, args.verl_node_size)

    # print("judge_model_name_path",judge_model_name_path)
    print("actor_model_path",actor_model_path)
    
    # 拼接 worker 启动命令
    script_base = "submit/common/worker.sh"

    cmd_parts = [
        "bash", script_base,
        f"--script_path={script_path}",
        f"--verl_node_size={args.verl_node_size}",
        f"--world_size={args.world_size}",
        f"--experiment_name={EXPERIMENT_NAME}",
        f"--actor_model_path={actor_model_path}",
        f"--judge_model_name_path={judge_model_name_path}"
    ]

    cmd = " ".join(cmd_parts)
    print(f"📦 启动指令：\n{cmd}")
    os.system(cmd)