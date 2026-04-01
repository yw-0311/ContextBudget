#!/usr/bin/env python3
"""
EVA 推理脚本 - 支持MOQA和BC数据集

基本用法：
    python inference.py --parquet data.parquet --out_jsonl results.jsonl --sglang_url http://localhost:30000
"""

import argparse
import asyncio
import json
from typing import Any, Dict, List

import pandas as pd
from transformers import AutoTokenizer

import sys
sys.path.append('/home/wy517954/code/Elistic-Context-Fold-Verl/verl')

from sglang_server_manager import SGLangServerManager
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop


class Config:
    def __init__(self, model_path: str, tool_config_path: str, max_model_len: int):
        self.model_path = model_path
        self.tool_config_path = tool_config_path
        self.max_model_len = max_model_len
        
        class RolloutConfig:
            def __init__(self):
                self.max_model_len = max_model_len
                self.max_assistant_turns = 10
                self.max_depth = 10
                self.max_parallel_calls = 1
                self.max_tool_response_length = 4096
                self.tool_response_truncate_side = "left"
        
        self.rollout = RolloutConfig()


class TrainerConfig:
    def __init__(self, config: Config):
        self.config = config


async def run_inference(
    agent_loop: ToolAgentLoop,
    messages: List[Dict[str, str]],
    sampling_params: Dict[str, Any],
) -> Dict[str, Any]:
    """运行单次推理"""
    outs = await agent_loop.run(
        sampling_params=sampling_params,
        raw_prompt=messages,
        image_data=None,
        tools_kwargs={},
    )
    
    out = outs[-1]
    final_messages = out.extra_fields.get("final_messages", [])
    
    prediction = ""
    for msg in reversed(final_messages):
        if msg.get("role") == "assistant":
            prediction = msg.get("content", "")
            break
    
    return {
        "prediction": prediction,
        "num_turns": out.num_turns,
        "final_messages": final_messages,
    }


def extract_answer(prediction: str) -> str:
    """从预测中提取答案"""
    import re
    match = re.search(r'<answer>(.*?)</answer>', prediction, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    sentences = prediction.split('.')
    if sentences:
        return sentences[-1].strip()
    
    return prediction.strip()


async def main():
    parser = argparse.ArgumentParser(description="EVA 推理脚本")
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--parquet", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    parser.add_argument("--sglang_url", type=str, default="http://localhost:30000")
    parser.add_argument("--tool_config_path", type=str, default="")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=4)
    
    args = parser.parse_args()
    
    print(f"[EVA] 加载数据: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"[EVA] 样本数: {len(df)}")
    
    print(f"[EVA] 加载tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    config = Config(args.model_path, args.tool_config_path, args.max_model_len)
    trainer_config = TrainerConfig(config)
    server_manager = SGLangServerManager(args.sglang_url)
    
    agent_loop = ToolAgentLoop(
        trainer_config=trainer_config,
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
    )
    agent_loop.loop = asyncio.get_running_loop()
    
    sampling_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": args.max_model_len,
    }
    
    sem = asyncio.Semaphore(args.concurrency)
    
    async def process_row(idx: int, row: Dict[str, Any]):
        async with sem:
            messages = row.get("prompt", [])
            
            try:
                result = await run_inference(agent_loop, messages, sampling_params)
                answer = extract_answer(result["prediction"])
                
                return {
                    "idx": idx,
                    "question": messages[-1].get("content", "") if messages else "",
                    "prediction": result["prediction"],
                    "answer": answer,
                    "num_turns": result["num_turns"],
                    "ground_truth": row.get("ground_truth", {}),
                }
            except Exception as e:
                print(f"[EVA] Error {idx}: {e}")
                return {"idx": idx, "error": str(e)}
    
    tasks = [process_row(i, row.to_dict()) for i, row in enumerate(df.itertuples(index=False))]
    
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            if (i + 1) % 10 == 0:
                print(f"[EVA] 进度: {i + 1}/{len(df)}")
    
    print(f"[EVA] 完成! 结果保存到: {args.out_jsonl}")


if __name__ == "__main__":
    asyncio.run(main())
