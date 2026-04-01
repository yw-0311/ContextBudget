#!/usr/bin/env python3
"""
评估 JSONL，修正 Token Cost 统计：
- Dependent Cost: 所有 assistant 输出内容的 token 累计（原逻辑）
- Peak Tokens: 单轮完整上下文的 token 累计（所有 role 内容），取所有轮中的最大值
- Last Branch Depth: 轨迹最终分支深度的平均值
- F1 标准误差：计算 F1 分数的标准误差
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple
import math

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers not installed")
    sys.exit(1)

try:
    from submit.experiments.search_r1.ferret.reward_score.search_r1_format import (
        compute_score_multi_answer,
    )
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def safe_get_ground_truth(item: dict) -> dict:
    """安全获取 ground_truth"""
    gt = item.get("ground_truth")
    if gt and isinstance(gt, dict) and "target" in gt:
        return gt
    
    rm = item.get("reward_model", {})
    if isinstance(rm, dict):
        gt = rm.get("ground_truth", {})
        if gt and isinstance(gt, dict) and "target" in gt:
            return gt
    
    extra = item.get("extra_info", {})
    if isinstance(extra, dict):
        gt = extra.get("ground_truth", {})
        if gt and isinstance(gt, dict) and "target" in gt:
            return gt
    
    return {}


def count_trajectory_tokens(final_messages: List[List[Dict]], tokenizer) -> Tuple[int, int]:
    """
    统计轨迹的 token 消耗。
    
    Args:
        final_messages: List of rounds, each round is List[Dict] with keys 'role', 'content'
        tokenizer: HuggingFace tokenizer
    
    Returns:
        (dependent_cost, peak_tokens)
        - dependent_cost: 所有轮次中 assistant 消息 token 数的累计值
        - peak_tokens: 所有轮次中，单轮完整上下文（所有 role 内容）token 数的最大值
    """
    if not final_messages or not isinstance(final_messages, list):
        return 0, 0
    
    dependent_cost = 0  # 所有 assistant 输出累计
    peak_tokens = 0     # 单轮最大上下文（所有 role）
    
    for round_messages in final_messages:
        if not isinstance(round_messages, list):
            continue
        
        # 计算该轮完整上下文的 token 数（所有 role）
        round_total_tokens = 0
        
        for msg in round_messages:
            if not isinstance(msg, dict):
                continue
            
            content = msg.get("content", "") or ""
            if not content:
                continue
                
            # 计算 token 数
            try:
                tokens = tokenizer.encode(content, add_special_tokens=False)
                token_count = len(tokens)
            except Exception:
                # fallback: 粗略估算（英文约4字符/token，中文约1.5字符/token，这里保守按3算）
                token_count = len(content) // 3
            
            # 累加到该轮总 token 数（所有 role）
            round_total_tokens += token_count
            
            # 如果是 assistant，累加到 dependent_cost
            if msg.get("role") == "assistant":
                dependent_cost += token_count
        
        # 更新峰值：该轮完整上下文长度 vs 历史最大
        peak_tokens = max(peak_tokens, round_total_tokens)
    
    return dependent_cost, peak_tokens


def compute_standard_error(values: List[float]) -> float:
    """计算标准误差"""
    if len(values) < 2:
        return 0.0  # 样本数少于2时，标准误差无法计算
    mean_val = sum(values) / len(values)  # 计算均值
    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)  # 计算方差
    return math.sqrt(variance) / math.sqrt(len(values))  # 计算标准误差


def evaluate_jsonl(
    input_path: str,
    tokenizer_path: str,
    target: int = 8,
    val_type: str = "f1",
    require_same_len: bool = False,
):
    """评估 JSONL，包含 F1、Token Cost 和 Last Branch Depth 统计"""
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    # 加载 tokenizer
    print(f"Loading tokenizer from: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
    
    total_items = 0
    success_items = 0
    
    # F1 统计
    total_score = 0.0
    total_tar_mean = 0.0
    has_answer_count = 0
    f1_scores = []  # 收集所有的 F1 分数
    
    # Token Cost 统计
    total_dependent_cost = 0  # 所有 item 的 assistant 累计 token 总和
    total_peak_tokens = 0     # 所有 item 的单轮峰值总和
    
    # Branch Depth 统计 (新增)
    total_branch_depth = 0.0
    
    token_stats_by_source = defaultdict(lambda: {
        "count": 0,
        "dependent_cost_sum": 0,
        "peak_tokens_sum": 0,
    })
    
    stats_by_source = defaultdict(lambda: {
        "count": 0,
        "score_sum": 0.0,
        "tar_mean_sum": 0.0,
        "has_answer_count": 0,
        "branch_depth_sum": 0.0,  # 新增
    })
    
    print(f"Processing: {input_path}")
    print(f"Config -> target: {target}, val_type: {val_type}, require_same_len: {require_same_len}\n")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                total_items += 1
                
                # ========== F1 计算 ==========
                prediction = item.get("prediction", "") or ""
                ground_truth = safe_get_ground_truth(item)
                data_source = item.get("data_source", "unknown")
                extra_info = item.get("extra_info", {})
                
                reward_extra = compute_score_multi_answer(
                    solution_str=prediction,
                    ground_truth=ground_truth,
                    data_source=data_source,
                    extra_info=extra_info,
                    val_type=val_type,
                    cot=False,
                    require_same_len=require_same_len,
                    target=target,
                )
                
                score = reward_extra.get("score", 0.0)
                tar_mean = reward_extra.get("tar_mean_score", 0.0)
                has_answer = reward_extra.get("has_answer", 0.0)
                
                total_score += score
                total_tar_mean += tar_mean
                f1_scores.append(score)  # 将 F1 分数添加到列表
                if has_answer > 0:
                    has_answer_count += 1
                
                # ========== Token Cost 统计（修正版） ==========
                final_messages = item.get("final_messages", [])
                dep_cost, peak_tok = count_trajectory_tokens(final_messages, tokenizer)
                
                total_dependent_cost += dep_cost
                total_peak_tokens += peak_tok
                
                # ========== Branch Depth 统计 (新增) ==========
                branch_depth = item.get("last_branch_depth", 0)
                if not isinstance(branch_depth, (int, float)):
                    branch_depth = 0
                total_branch_depth += branch_depth
                
                # 按 source 统计
                token_stats_by_source[data_source]["count"] += 1
                token_stats_by_source[data_source]["dependent_cost_sum"] += dep_cost
                token_stats_by_source[data_source]["peak_tokens_sum"] += peak_tok
                
                stats_by_source[data_source]["count"] += 1
                stats_by_source[data_source]["score_sum"] += score
                stats_by_source[data_source]["tar_mean_sum"] += tar_mean
                stats_by_source[data_source]["branch_depth_sum"] += branch_depth  # 新增
                if has_answer > 0:
                    stats_by_source[data_source]["has_answer_count"] += 1
                
                success_items += 1
                
                if success_items % 100 == 0:
                    print(f"Processed {success_items} items...", end='\r')
                    
            except Exception as e:
                print(f"\nError at line {line_num}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if success_items == 0:
        print("No valid items processed!")
        return
    
    # 计算全局平均
    mean_score = total_score / success_items
    mean_tar_mean = total_tar_mean / success_items
    mean_dependent_cost = total_dependent_cost / success_items
    mean_peak_tokens = total_peak_tokens / success_items
    mean_branch_depth = total_branch_depth / success_items
    
    # 计算 F1 分数的标准误差
    f1_se = compute_standard_error(f1_scores)
    
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Total Items:           {success_items}")
    print(f"Has Answer Rate:       {has_answer_count/success_items:.4f}")
    print(f"-" * 70)
    print(f"F1 Metrics:")
    print(f"  Mean F1 Sum:         {mean_score:.4f} ± {f1_se:.4f}")  # 添加标准误差
    print(f"  Mean F1/Target:      {mean_tar_mean:.4f}")
    print(f"-" * 70)
    print(f"Token Cost Metrics:")
    print(f"  Mean Dependent Cost: {mean_dependent_cost:.2f} (assistant输出累计)")
    print(f"  Mean Peak Tokens:    {mean_peak_tokens:.2f} (单轮完整上下文峰值)")
    print(f"-" * 70)
    print(f"Trajectory Metrics:")  # 新增区块
    print(f"  Mean Branch Depth:   {mean_branch_depth:.2f} (平均最终分支深度)")
    print(f"{'='*70}")
    
    # 按 Data Source 细分
    if stats_by_source:
        print(f"\nBreakdown by Data Source:")
        print(f"{'-'*90}")
        print(f"{'Source':<20s} {'Count':>6s} {'F1/Target':>10s} {'Dep.Cost':>12s} {'PeakRound':>12s} {'BranchDepth':>12s}")
        print(f"{'-'*90}")
        
        for src in sorted(stats_by_source.keys()):
            st = stats_by_source[src]
            tok_st = token_stats_by_source[src]
            cnt = st["count"]
            
            if cnt > 0:
                f1_target = st["tar_mean_sum"] / cnt
                dep_cost_avg = tok_st["dependent_cost_sum"] / cnt
                peak_avg = tok_st["peak_tokens_sum"] / cnt
                branch_depth_avg = st["branch_depth_sum"] / cnt
                
                print(f"{src:<20s} {cnt:>6d} {f1_target:>10.4f} {dep_cost_avg:>12.2f} {peak_avg:>12.2f} {branch_depth_avg:>12.2f}")
    
    return {
        "total_items": success_items,
        "mean_score": mean_score,
        "mean_tar_mean_score": mean_tar_mean,
        "has_answer_rate": has_answer_count / success_items,
        "mean_dependent_cost": mean_dependent_cost,
        "mean_peak_tokens": mean_peak_tokens,
        "mean_branch_depth": mean_branch_depth,
        "f1_se": f1_se,  # 返回 F1 的标准误差
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate JSONL with F1, Token Cost metrics and Branch Depth (修正版)."
    )
    parser.add_argument("input_jsonl", type=str, help="Input JSONL file path")
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        required=True,
        help="Path to tokenizer (e.g., /path/to/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument("--target", type=int, default=8, help="Target number of questions")
    parser.add_argument(
        "--val_type", 
        type=str, 
        default="f1", 
        choices=["f1", "em"],
        help="Validation type"
    )
    parser.add_argument(
        "--require_same_len", 
        action="store_true",
        help="Require slot count to match GT (default: False)"
    )
    
    args = parser.parse_args()
    
    evaluate_jsonl(
        input_path=args.input_jsonl,
        tokenizer_path=args.tokenizer_path,
        target=args.target,
        val_type=args.val_type,
        require_same_len=args.require_same_len,
    )


if __name__ == "__main__":
    main()
