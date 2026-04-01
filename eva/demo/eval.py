#!/usr/bin/env python3
"""
EVA 评估脚本 - 计算F1分数和EM分数

基本用法：
    python eval.py --input_jsonl results.jsonl --tokenizer_path /path/to/tokenizer
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

from transformers import AutoTokenizer


def normalize_answer(s: str) -> str:
    """标准化答案文本"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """计算F1分数"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    
    overlap = 0
    for token in truth_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    
    if overlap == 0:
        return 0.0
    
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction: str, ground_truth: str) -> bool:
    """计算精确匹配"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def count_tokens(text: str, tokenizer) -> int:
    """计算token数量"""
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return len(text) // 3


def count_trajectory_tokens(final_messages: List[List[Dict]], tokenizer):
    """统计轨迹的token消耗"""
    if not final_messages or not isinstance(final_messages, list):
        return 0, 0
    
    dependent_cost = 0
    peak_tokens = 0
    
    for round_messages in final_messages:
        if not isinstance(round_messages, list):
            continue
        
        round_total_tokens = 0
        
        for msg in round_messages:
            if not isinstance(msg, dict):
                continue
            
            content = msg.get("content", "") or ""
            if not content:
                continue
            
            token_count = count_tokens(content, tokenizer)
            round_total_tokens += token_count
            
            if msg.get("role") == "assistant":
                dependent_cost += token_count
        
        peak_tokens = max(peak_tokens, round_total_tokens)
    
    return dependent_cost, peak_tokens


def main():
    parser = argparse.ArgumentParser(description="EVA 评估脚本")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    
    args = parser.parse_args()
    
    print(f"[EVA] 输入: {args.input_jsonl}")
    print(f"[EVA] Tokenizer: {args.tokenizer_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        print(f"[Error] 文件不存在: {args.input_jsonl}")
        sys.exit(1)
    
    total_items = 0
    success_items = 0
    f1_scores = []
    em_correct = 0
    total_dependent_cost = 0
    total_peak_tokens = 0
    total_branch_depth = 0
    
    print(f"[EVA] 开始评估...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                total_items += 1
                
                prediction = item.get("prediction", "") or ""
                answer = item.get("answer", "") or prediction
                ground_truth = item.get("ground_truth", {})
                
                if isinstance(ground_truth, dict):
                    gt_answer = ground_truth.get("target", ground_truth.get("answer", ""))
                elif isinstance(ground_truth, list) and ground_truth:
                    gt_answer = ground_truth[0]
                else:
                    gt_answer = str(ground_truth) if ground_truth else ""
                
                f1 = compute_f1(answer, gt_answer)
                f1_scores.append(f1)
                
                em = compute_em(answer, gt_answer)
                if em:
                    em_correct += 1
                
                final_messages = item.get("final_messages", [])
                dep_cost, peak_tok = count_trajectory_tokens(final_messages, tokenizer)
                total_dependent_cost += dep_cost
                total_peak_tokens += peak_tok
                
                branch_depth = item.get("num_turns", 0)
                total_branch_depth += branch_depth
                
                success_items += 1
                
                if success_items % 100 == 0:
                    print(f"[EVA] 已处理 {success_items} 个样本...", end='\r')
                    
            except Exception as e:
                print(f"\n[Error] 处理出错: {e}")
                continue
    
    if success_items == 0:
        print(f"[EVA] 没有成功处理的样本")
        return
    
    mean_f1 = sum(f1_scores) / success_items
    mean_em = em_correct / success_items
    mean_dependent_cost = total_dependent_cost / success_items
    mean_peak_tokens = total_peak_tokens / success_items
    mean_branch_depth = total_branch_depth / success_items
    
    import math
    f1_mean = sum(f1_scores) / success_items
    f1_variance = sum((x - f1_mean) ** 2 for x in f1_scores) / (success_items - 1) if success_items > 1 else 0
    f1_se = math.sqrt(f1_variance) / math.sqrt(success_items) if success_items > 1 else 0.0
    
    print(f"\n{'='*70}")
    print(f"[EVA] 评估结果")
    print(f"{'='*70}")
    print(f"样本数:              {success_items}")
    print(f"{'-'*70}")
    print(f"F1:                  {mean_f1:.4f} ± {f1_se:.4f}")
    print(f"EM:                  {mean_em:.4f}")
    print(f"{'-'*70}")
    print(f"Dependent Cost:      {mean_dependent_cost:.2f}")
    print(f"Peak Tokens:         {mean_peak_tokens:.2f}")
    print(f"平均轮次:            {mean_branch_depth:.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()