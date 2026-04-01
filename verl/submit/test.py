import os
import argparse
import time
import sys, re
from typing import Tuple, Any
from openlm_hub import repo_download

name = "BRANCH_30B_RL_F1_RW_BS128_4_5_6_7_8_MIS__Qwen3_30B_A3B_TK__128GPUS__202603080620"

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

EXPERIMENT_NAME, MODEL_NAME, GPU_NUMS, DATE = parse_job_name(name)

print(MODEL_NAME)