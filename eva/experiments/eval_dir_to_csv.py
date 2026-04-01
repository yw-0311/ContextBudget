#!/usr/bin/env python3
"""
Batch evaluate all JSONL files in a directory and save results to CSV.

- Automatically select tokenizer:
    If filename or model_name contains "7B" → use TOKENIZER_PATH_7B
    If contains "30B" → use TOKENIZER_PATH_30B

Env required:
    export TOKENIZER_PATH_7B=...
    export TOKENIZER_PATH_30B=...
"""

import os
import re
import csv
import json
import sys
import math
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers not installed")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: tqdm not installed. Install with: pip install tqdm")
    sys.exit(1)

try:
    from submit.experiments.search_r1.ferret.reward_score.search_r1_format import (
        compute_score_multi_answer,
    )
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


# =============================================================================
# LLM Judge Configuration
# =============================================================================
GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. 

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.""".strip()


class LLMJudgeClient:
    """LLM-as-Judge client that reuses rollout server logic."""
    
    def __init__(
        self,
        sglang_url: str,
        model_path: str,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        """
        Initialize LLM Judge client.
        
        Args:
            sglang_url: URL of the sglang server to use for judgment
            model_path: Path to the model to use for judgment
            max_retries: Maximum number of retry attempts
            timeout: Timeout for each request in seconds
        """
        self.sglang_url = sglang_url
        self.model_path = model_path
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Lazy load dependencies only when LLM judge is enabled
        self._aiohttp = None
        self._client_session = None
        
    def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self._client_session is None:
            try:
                import aiohttp
                self._aiohttp = aiohttp
                self._client_session = aiohttp.ClientSession()
            except ImportError:
                raise ImportError("aiohttp is required for LLM judge. Install with: pip install aiohttp")
    
    async def close(self):
        """Close the aiohttp session if it exists."""
        if self._client_session is not None:
            await self._client_session.close()
            self._client_session = None
    
    async def judge(
        self,
        question: str,
        response: str,
        correct_answer: str,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Judge whether the response is correct using LLM server.
        
        Returns:
            Tuple of (is_correct, confidence, judge_result)
            judge_result contains extracted_final_answer, reasoning, correct, confidence
        """
        self._ensure_session()
        
        prompt = GRADER_TEMPLATE.format(
            question=question,
            response=response,
            correct_answer=correct_answer,
        )

        for attempt in range(self.max_retries):
            try:
                result = await self._call_sglang_server(prompt)
                
                # Parse the judge result
                is_correct = result.get("correct", "no").lower() == "yes"
                confidence = self._parse_confidence(result.get("confidence", "100"))
                
                return is_correct, confidence, result
                
            except Exception as e:
                print(f"[LLM Judge] Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt == self.max_retries - 1:
                    # Return default values on final failure
                    return False, 0.0, {
                        "error": str(e),
                        "extracted_final_answer": "",
                        "reasoning": f"Judge API failed after {self.max_retries} attempts",
                        "correct": "no",
                        "confidence": 0,
                    }
                # Wait before retry (exponential backoff)
                await asyncio.sleep(2 ** attempt)
    
    async def _call_sglang_server(self, prompt: str) -> Dict[str, Any]:
        """Call the sglang server to get judgment."""
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model_path,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": 2000,
            "stop": [],
        }

        async with self._client_session.post(
            f"{self.sglang_url}/v1/completions",
            headers=headers,
            json=payload,
            timeout=self._aiohttp.ClientTimeout(total=self.timeout),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Server request failed with status {resp.status}: {error_text}")
            
            data = await resp.json()
            content = data["choices"][0]["text"]
            return self._parse_judge_response(content)
    
    def _parse_judge_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM judge response into structured format."""
        result = {
            "extracted_final_answer": "",
            "correct_answer": "",
            "reasoning": "",
            "correct": "no",
            "confidence": 100,
        }

        # Try to extract each field using regex
        patterns = {
            "extracted_final_answer": r"extracted_final_answer:\s*(.+?)(?=\n\[correct_answer\]:|\nreasoning:|\ncorrect:|\nconfidence:|$)",
            "correct_answer": r"\[correct_answer\]:\s*(.+?)(?=\nreasoning:|\ncorrect:|\nconfidence:|$)",
            "reasoning": r"reasoning:\s*(.+?)(?=\ncorrect:|\nconfidence:|$)",
            "correct": r"correct:\s*(yes|no)",
            "confidence": r"confidence:\s*(\d+)",
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if field == "confidence":
                    try:
                        value = int(value)
                    except ValueError:
                        value = 100
                elif field == "correct":
                    value = value.lower()
                result[field] = value

        return result
    
    def _parse_confidence(self, confidence_str: str) -> float:
        """Parse confidence string to float."""
        try:
            if isinstance(confidence_str, (int, float)):
                return float(confidence_str)
            # Extract number from string like "90%" or "90"
            match = re.search(r"(\d+)", str(confidence_str))
            if match:
                return float(match.group(1))
        except Exception:
            pass
        return 100.0


def safe_get_ground_truth(item: dict) -> dict:
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

            try:
                token_count = len(tokenizer.encode(content, add_special_tokens=False))
            except Exception:
                token_count = max(0, len(content) // 3)

            round_total_tokens += token_count
            if msg.get("role") == "assistant":
                dependent_cost += token_count

        peak_tokens = max(peak_tokens, round_total_tokens)

    return dependent_cost, peak_tokens


def compute_standard_error(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def parse_from_path(path_str: str, dataset_name: str = "nq_hotpotqa") -> Dict[str, Any]:
    base = Path(path_str).name
    if base.endswith(".jsonl"):
        base = base[:-6]

    def _m_int(pat: str):
        m = re.search(pat, base)
        return int(m.group(1)) if m else None

    max_model_len = _m_int(r"_len(\d+)")
    max_depth = _m_int(r"_depth(\d+)")
    target = _m_int(r"_target(\d+)")

    # 1) cut everything after _lenXXXX
    m = re.search(r"_len\d+", base)
    prefix = base[: m.start()] if m else base

    # 2) split by dataset marker inside prefix
    marker = f"_{dataset_name}_"
    model_name = None
    agent_type = None
    dataset = None

    if marker in prefix:
        model_name, agent_type = prefix.split(marker, 1)
        dataset = dataset_name

    return {
        "model_name": model_name,
        "dataset": dataset,
        "agent_type": agent_type,
        "max_model_len": max_model_len,
        "max_depth": max_depth,
        "target": target,
        "filename_base": base,
        "prefix_before_len": prefix,
    }


class DataSourceStats:
    """Accumulator for statistics per data_source."""
    def __init__(self, enable_llm_judge: bool = False):
        self.total_items = 0
        self.success_items = 0
        self.total_score = 0.0
        self.total_tar_mean = 0.0
        self.has_answer_count = 0
        self.f1_scores: List[float] = []
        self.total_dependent_cost = 0
        self.total_peak_tokens = 0
        self.total_branch_depth = 0.0
        
        # LLM Judge metrics
        self.enable_llm_judge = enable_llm_judge
        self.llm_judge_correct_count = 0
        self.llm_judge_total_count = 0
        self.llm_judge_confidence_sum = 0.0
        self.llm_judge_scores: List[float] = []

    def add(self, score: float, tar_mean: float, has_answer: float, 
            dep_cost: int, peak_tok: int, branch_depth: float,
            llm_judge_result: Optional[Dict[str, Any]] = None):
        self.total_items += 1
        self.total_score += score
        self.total_tar_mean += tar_mean
        self.f1_scores.append(score)
        if has_answer > 0:
            self.has_answer_count += 1
        self.total_dependent_cost += dep_cost
        self.total_peak_tokens += peak_tok
        self.total_branch_depth += branch_depth
        self.success_items += 1
        
        # Add LLM judge metrics if enabled
        if self.enable_llm_judge and llm_judge_result:
            self.llm_judge_total_count += 1
            if llm_judge_result.get("is_correct", False):
                self.llm_judge_correct_count += 1
            confidence = llm_judge_result.get("confidence", 100.0)
            self.llm_judge_confidence_sum += float(confidence)
            self.llm_judge_scores.append(1.0 if llm_judge_result.get("is_correct", False) else 0.0)

    def finalize(self) -> Dict[str, Any]:
        if self.success_items == 0:
            base_result = {
                "total_items": 0,
                "has_answer_rate": 0.0,
                "mean_score": 0.0,
                "mean_tar_mean_score": 0.0,
                "f1_se": 0.0,
                "mean_dependent_cost": 0.0,
                "mean_peak_tokens": 0.0,
                "mean_branch_depth": 0.0,
            }
        else:
            base_result = {
                "total_items": self.success_items,
                "has_answer_rate": self.has_answer_count / self.success_items,
                "mean_score": self.total_score / self.success_items,
                "mean_tar_mean_score": self.total_tar_mean / self.success_items,
                "f1_se": compute_standard_error(self.f1_scores),
                "mean_dependent_cost": self.total_dependent_cost / self.success_items,
                "mean_peak_tokens": self.total_peak_tokens / self.success_items,
                "mean_branch_depth": self.total_branch_depth / self.success_items,
            }
        
        # Add LLM judge metrics if enabled
        if self.enable_llm_judge:
            if self.llm_judge_total_count > 0:
                base_result.update({
                    "llm_judge_accuracy": self.llm_judge_correct_count / self.llm_judge_total_count,
                    "llm_judge_mean_confidence": self.llm_judge_confidence_sum / self.llm_judge_total_count,
                    "llm_judge_se": compute_standard_error(self.llm_judge_scores),
                    "llm_judge_total_count": self.llm_judge_total_count,
                    "llm_judge_correct_count": self.llm_judge_correct_count,
                })
            else:
                base_result.update({
                    "llm_judge_accuracy": 0.0,
                    "llm_judge_mean_confidence": 0.0,
                    "llm_judge_se": 0.0,
                    "llm_judge_total_count": 0,
                    "llm_judge_correct_count": 0,
                })
        
        return base_result


def evaluate_one_jsonl(
    input_path: Path,
    tokenizer,
    target: int,
    val_type: str = "f1",
    require_same_len: bool = False,
    by_datasource: bool = False,
    llm_judge_client: Optional[LLMJudgeClient] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Dict[str, Any]]]]:
    """
    Returns:
        overall_stats: Dict with overall statistics
        datasource_stats: Dict mapping data_source -> stats (only if by_datasource=True)
    """
    
    enable_llm_judge = llm_judge_client is not None
    
    # Overall accumulators
    total_items = 0
    success_items = 0
    total_score = 0.0
    total_tar_mean = 0.0
    has_answer_count = 0
    f1_scores: List[float] = []

    total_dependent_cost = 0
    total_peak_tokens = 0
    total_branch_depth = 0.0

    # LLM Judge accumulators
    llm_judge_correct_count = 0
    llm_judge_total_count = 0
    llm_judge_confidence_sum = 0.0
    llm_judge_scores: List[float] = []

    # Per-datasource accumulators
    ds_stats: Dict[str, DataSourceStats] = defaultdict(lambda: DataSourceStats(enable_llm_judge))

    # First pass: count total lines for progress bar
    with input_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Process lines
    with input_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc=f"  Processing {input_path.name}", unit="lines", leave=False):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                total_items += 1

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

                score = float(reward_extra.get("score", 0.0))
                tar_mean = float(reward_extra.get("tar_mean_score", 0.0))
                has_answer = float(reward_extra.get("has_answer", 0.0))

                # Overall stats
                total_score += score
                total_tar_mean += tar_mean
                f1_scores.append(score)
                if has_answer > 0:
                    has_answer_count += 1

                final_messages = item.get("final_messages", [])
                dep_cost, peak_tok = count_trajectory_tokens(final_messages, tokenizer)
                total_dependent_cost += dep_cost
                total_peak_tokens += peak_tok

                branch_depth = item.get("last_branch_depth", 0)
                if not isinstance(branch_depth, (int, float)):
                    branch_depth = 0
                total_branch_depth += float(branch_depth)

                success_items += 1

                # LLM Judge evaluation
                llm_judge_result = None
                if enable_llm_judge:
                    try:
                        # Extract question from final_messages
                        question = ""
                        for msg in final_messages:
                            if isinstance(msg, dict) and msg.get("role") == "user":
                                question = msg.get("content", "")
                                break
                        
                        # Get correct answer from ground_truth
                        correct_answer = str(ground_truth.get("target", ground_truth.get("answer", "")))
                        
                        # Run LLM judge
                        is_correct, confidence, judge_result = asyncio.run(
                            llm_judge_client.judge(
                                question=question,
                                response=prediction,
                                correct_answer=correct_answer,
                            )
                        )
                        
                        llm_judge_result = {
                            "is_correct": is_correct,
                            "confidence": confidence,
                            "judge_result": judge_result,
                        }
                        
                        # Update LLM judge accumulators
                        llm_judge_total_count += 1
                        if is_correct:
                            llm_judge_correct_count += 1
                        llm_judge_confidence_sum += float(confidence)
                        llm_judge_scores.append(1.0 if is_correct else 0.0)
                        
                    except Exception as e:
                        print(f"[LLM Judge] Error evaluating item: {e}")
                        llm_judge_result = {
                            "is_correct": False,
                            "confidence": 0.0,
                            "judge_result": {"error": str(e)},
                        }

                # Per-datasource stats
                if by_datasource:
                    ds_stats[data_source].add(score, tar_mean, has_answer, 
                                              dep_cost, peak_tok, float(branch_depth),
                                              llm_judge_result)

            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    # Build overall stats
    if success_items == 0:
        overall = {
            "total_items": 0,
            "has_answer_rate": 0.0,
            "mean_score": 0.0,
            "mean_tar_mean_score": 0.0,
            "f1_se": 0.0,
            "mean_dependent_cost": 0.0,
            "mean_peak_tokens": 0.0,
            "mean_branch_depth": 0.0,
        }
    else:
        overall = {
            "total_items": success_items,
            "has_answer_rate": has_answer_count / success_items,
            "mean_score": total_score / success_items,
            "mean_tar_mean_score": total_tar_mean / success_items,
            "f1_se": compute_standard_error(f1_scores),
            "mean_dependent_cost": total_dependent_cost / success_items,
            "mean_peak_tokens": total_peak_tokens / success_items,
            "mean_branch_depth": total_branch_depth / success_items,
        }
    
    # Add LLM judge metrics to overall stats if enabled
    if enable_llm_judge:
        if llm_judge_total_count > 0:
            overall.update({
                "llm_judge_accuracy": llm_judge_correct_count / llm_judge_total_count,
                "llm_judge_mean_confidence": llm_judge_confidence_sum / llm_judge_total_count,
                "llm_judge_se": compute_standard_error(llm_judge_scores),
                "llm_judge_total_count": llm_judge_total_count,
                "llm_judge_correct_count": llm_judge_correct_count,
            })
        else:
            overall.update({
                "llm_judge_accuracy": 0.0,
                "llm_judge_mean_confidence": 0.0,
                "llm_judge_se": 0.0,
                "llm_judge_total_count": 0,
                "llm_judge_correct_count": 0,
            })

    # Build per-datasource stats
    datasource_result = None
    if by_datasource:
        datasource_result = {
            ds: stats.finalize() for ds, stats in ds_stats.items()
        }

    return overall, datasource_result


def format_row(
    jsonl_file: str,
    model_name: str,
    dataset: str,
    agent_type: str,
    max_model_len: Optional[int],
    target: int,
    stats: Dict[str, Any],
    data_source: Optional[str] = None,
    enable_llm_judge: bool = False,
) -> Dict[str, Any]:
    """Format a statistics dict into a CSV row."""
    row = {
        "jsonl_file": jsonl_file,
        "model_name": model_name,
        "dataset": dataset,
        "agent_type": agent_type,
        "max_model_len": max_model_len if max_model_len is not None else "",
        "target": target,
        "total_items": stats["total_items"],
        "has_answer_rate": f"{stats['has_answer_rate']:.6f}",
        "mean_f1_sum": f"{stats['mean_score']:.6f}",
        "f1_se": f"{stats['f1_se']:.6f}",
        "mean_f1_target": f"{stats['mean_tar_mean_score']:.6f}",
        "mean_dependent_cost": f"{stats['mean_dependent_cost']:.6f}",
        "mean_peak_tokens": f"{stats['mean_peak_tokens']:.6f}",
        "mean_branch_depth": f"{stats['mean_branch_depth']:.6f}",
    }
    if data_source is not None:
        row["data_source"] = data_source
    
    # Add LLM judge metrics if enabled
    if enable_llm_judge:
        row.update({
            "llm_judge_accuracy": f"{stats.get('llm_judge_accuracy', 0.0):.6f}",
            "llm_judge_mean_confidence": f"{stats.get('llm_judge_mean_confidence', 0.0):.2f}",
            "llm_judge_se": f"{stats.get('llm_judge_se', 0.0):.6f}",
            "llm_judge_total_count": stats.get('llm_judge_total_count', 0),
            "llm_judge_correct_count": stats.get('llm_judge_correct_count', 0),
        })
    
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate JSONL files and compute statistics with optional LLM-as-judge"
    )
    parser.add_argument("input_dir", type=str, help="Directory containing JSONL files to evaluate")
    parser.add_argument(
        "--by-datasource",
        action="store_true",
        help="Also output per-data_source statistics (adds 'data_source' column)",
    )
    
    # LLM Judge arguments
    parser.add_argument(
        "--enable-llm-judge",
        action="store_true",
        help="Enable LLM-as-judge evaluation using sglang server",
    )
    parser.add_argument(
        "--sglang-url",
        type=str,
        default="http://localhost:30000",
        help="URL of the sglang server for LLM judge (default: http://localhost:30000)",
    )
    parser.add_argument(
        "--judge-model-path",
        type=str,
        default="/data/oss_bucket_0/shiyi/model/Qwen2.5-7B-Instruct/",
        help="Path to the model to use for LLM judge (default: /data/oss_bucket_0/shiyi/model/Qwen2.5-7B-Instruct/)",
    )
    parser.add_argument(
        "--judge-max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for LLM judge requests (default: 3)",
    )
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for LLM judge requests (default: 60.0)",
    )
    
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}")
        sys.exit(1)

    tokenizer_7b_path = os.environ.get("TOKENIZER_PATH_7B", "").strip()
    tokenizer_30b_path = os.environ.get("TOKENIZER_PATH_30B", "").strip()

    if not tokenizer_7b_path or not tokenizer_30b_path:
        print("Error: TOKENIZER_PATH_7B and TOKENIZER_PATH_30B must be set.")
        sys.exit(1)

    val_type = os.environ.get("VAL_TYPE", "f1").strip()
    require_same_len = os.environ.get("REQUIRE_SAME_LEN", "0").strip() in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )

    tokenizer_cache = {}

    # Initialize LLM Judge client if enabled
    llm_judge_client = None
    if args.enable_llm_judge:
        print(f"[LLM Judge] Enabled - using server: {args.sglang_url}")
        print(f"[LLM Judge] Model path: {args.judge_model_path}")
        llm_judge_client = LLMJudgeClient(
            sglang_url=args.sglang_url,
            model_path=args.judge_model_path,
            max_retries=args.judge_max_retries,
            timeout=args.judge_timeout,
        )

    out_csv = input_dir / "evaluation_results.csv"
    
    # Check if CSV already exists, skip if it does
    if out_csv.exists():
        print(f"CSV already exists: {out_csv}")
        print("Skipping evaluation.")
        sys.exit(0)
    fieldnames = [
        "jsonl_file",
        "model_name",
        "dataset",
        "agent_type",
        "max_model_len",
        "target",
        "total_items",
        "has_answer_rate",
        "mean_f1_sum",
        "f1_se",
        "mean_f1_target",
        "mean_dependent_cost",
        "mean_peak_tokens",
        "mean_branch_depth",
    ]
    
    # Add data_source column if needed
    if args.by_datasource:
        fieldnames.append("data_source")
    
    # Add LLM judge columns if enabled
    if args.enable_llm_judge:
        fieldnames.extend([
            "llm_judge_accuracy",
            "llm_judge_mean_confidence",
            "llm_judge_se",
            "llm_judge_total_count",
            "llm_judge_correct_count",
        ])

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("No .jsonl files found.")
        sys.exit(0)

    try:
        with out_csv.open("w", newline="", encoding="utf-8") as wf:
            writer = csv.DictWriter(wf, fieldnames=fieldnames)
            writer.writeheader()

            for p in tqdm(jsonl_files, desc="Processing files", unit="file"):
                meta = parse_from_path(str(p))
                model_name = meta["model_name"] or ""
                target = meta["target"] or 8

                name_str = f"{model_name}_{p.name}"

                if re.search(r"30[bB]", name_str):
                    tokenizer_key = "30B"
                    tokenizer_path = tokenizer_30b_path
                else:
                    tokenizer_key = "7B"
                    tokenizer_path = tokenizer_7b_path

                if tokenizer_key not in tokenizer_cache:
                    print(f"Loading {tokenizer_key} tokenizer from: {tokenizer_path}")
                    tokenizer_cache[tokenizer_key] = AutoTokenizer.from_pretrained(
                        tokenizer_path,
                        trust_remote_code=True,
                    )

                tokenizer = tokenizer_cache[tokenizer_key]

                tqdm.write(f"Evaluating: {p.name} (tokenizer={tokenizer_key})")

                overall_stats, ds_stats = evaluate_one_jsonl(
                    input_path=p,
                    tokenizer=tokenizer,
                    target=int(target),
                    val_type=val_type,
                    require_same_len=require_same_len,
                    by_datasource=args.by_datasource,
                    llm_judge_client=llm_judge_client,
                )

                # Write overall row (data_source column will be empty if by_datasource is True)
                row = format_row(
                    jsonl_file=p.name,
                    model_name=meta["model_name"] or "",
                    dataset=meta["dataset"] or "",
                    agent_type=meta["agent_type"] or "",
                    max_model_len=meta["max_model_len"],
                    target=target,
                    stats=overall_stats,
                    data_source="" if args.by_datasource else None,
                    enable_llm_judge=args.enable_llm_judge,
                )
                writer.writerow(row)

                # Write per-datasource rows if requested
                if args.by_datasource and ds_stats:
                    for ds_name in sorted(ds_stats.keys()):
                        ds_row = format_row(
                            jsonl_file=p.name,
                            model_name=meta["model_name"] or "",
                            dataset=meta["dataset"] or "",
                            agent_type=meta["agent_type"] or "",
                            max_model_len=meta["max_model_len"],
                            target=target,
                            stats=ds_stats[ds_name],
                            data_source=ds_name,
                            enable_llm_judge=args.enable_llm_judge,
                        )
                        writer.writerow(ds_row)

    finally:
        # Clean up LLM judge client if initialized
        if llm_judge_client is not None:
            asyncio.run(llm_judge_client.close())
            print("[LLM Judge] Client closed")

    tqdm.write(f"\nDone. CSV saved to: {out_csv}")


if __name__ == "__main__":
    main()