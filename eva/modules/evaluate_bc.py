#!/usr/bin/env python3
"""
Batch LLM Judge Evaluation Script with JSONL Output
Save merged complete data to JSONL, skip existing files

Basic usage:
    python llm_judge.py --base-dir /path/to/jsonl/files --output-dir /path/to/output
"""

import asyncio
import json
import re
import httpx
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse
import sys

# Configuration
SGLANG_URL = "http://localhost:30000/v1"
BASE_DIR = Path("./outputs_bc")
DEFAULT_OUTPUT_DIR = Path("./outputs_bc_judged")


# ==================== LLM Judge Prompt ====================

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator. Your task is to judge whether the model's answer correctly answers the question, compared to the ground truth answer.

Question: {question}

Model's Response: {response}

Correct Answer: {correct_answer}

Please analyze the model's response and determine if it correctly answers the question.

Instructions:
1. Extract the final answer from the model's response (look for <answer> tags or the final conclusion)
2. Compare it with the correct answer
3. Consider semantic equivalence (not just exact string match)
4. Provide your reasoning

Output your evaluation in the following JSON format:
{{
    "extracted_final_answer": "the answer extracted from model response",
    "correct_answer": "the correct answer provided",
    "reasoning": "your detailed reasoning for the judgment",
    "correct": "YES" or "NO",
    "confidence": "HIGH" or "MEDIUM" or "LOW"
}}

JSON Output:"""


# ==================== Helper Functions ====================

def safe_get_ground_truth(item: Dict[str, Any]) -> Dict[str, Any]:
    """Safely extract ground truth information"""
    ground_truth = item.get("ground_truth", {})

    if isinstance(ground_truth, (str, list)):
        if isinstance(ground_truth, list):
            return {"target": ground_truth}
        else:
            return {"target": [ground_truth]}

    if isinstance(ground_truth, dict):
        if "target" in ground_truth:
            return ground_truth
        if "answer" in ground_truth and "target" not in ground_truth:
            answer = ground_truth["answer"]
            if isinstance(answer, list):
                return {"target": answer}
            else:
                return {"target": [answer]}

    for key in ["answer", "target", "gold_answer", "label"]:
        if key in item:
            val = item[key]
            if isinstance(val, list):
                return {"target": val}
            else:
                return {"target": [val]}

    return {"target": []}


def format_correct_answer(ground_truth: Dict[str, Any]) -> str:
    """Format ground truth to string"""
    target = ground_truth.get("target", [])

    if not target:
        return ""

    if isinstance(target, list):
        if len(target) == 1:
            return str(target[0])
        else:
            return "; ".join(str(t) for t in target)
    else:
        return str(target)


# ==================== LLM Judge Client ====================

class LLMJudgeClient:
    """LLM Judge client that directly requests SGLang OpenAI-compatible API"""

    def __init__(
        self,
        sglang_url: str = SGLANG_URL,
        model_path: str = None,
        max_retries: int = 3,
        timeout: float = 120.0,
        api_key: str = "EMPTY"
    ):
        self.sglang_url = sglang_url.rstrip("/")
        if not self.sglang_url.endswith("/v1"):
            self.sglang_url += "/v1"

        self.model_path = model_path or "default"
        self.max_retries = max_retries
        self.timeout = timeout
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        await self.client.aclose()

    async def _call_api(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """Call SGLang OpenAI-compatible API"""
        url = f"{self.sglang_url}/chat/completions"

        payload = {
            "model": self.model_path,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
                else:
                    raise ValueError(f"Unexpected response format: {data}")

            except httpx.HTTPStatusError as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"    [Retry {attempt+1}/{self.max_retries}] HTTP error: {e.response.status_code}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"    [Retry {attempt+1}/{self.max_retries}] Error: {str(e)}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        return ""

    def _parse_judge_result(self, response_text: str) -> Dict[str, Any]:
        """Parse Judge's JSON output"""
        json_match = re.search(r"\{.*?\}", response_text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return {
                    "extracted_final_answer": result.get("extracted_final_answer", ""),
                    "correct_answer": result.get("correct_answer", ""),
                    "reasoning": result.get("reasoning", ""),
                    "correct": result.get("correct", ""),
                    "confidence": result.get("confidence", "")
                }
            except json.JSONDecodeError:
                pass

        result = {
            "extracted_final_answer": "",
            "correct_answer": "",
            "reasoning": response_text[:500],
            "correct": "",
            "confidence": ""
        }

        if "correct\": \"YES\"" in response_text or "correct: YES" in response_text.upper():
            result["correct"] = "YES"
        elif "correct\": \"NO\"" in response_text or "correct: NO" in response_text.upper():
            result["correct"] = "NO"

        for level in ["HIGH", "MEDIUM", "LOW"]:
            if f"confidence\": \"{level}\"" in response_text or f"confidence: {level}" in response_text.upper():
                result["confidence"] = level
                break

        return result

    async def judge(
            self,
            question: str,
            response: str,
            correct_answer: str
        ) -> tuple[bool, str, Dict[str, Any]]:
        """Execute LLM Judge evaluation"""
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            response=response,
            correct_answer=correct_answer
        )

        messages = [
            {"role": "system", "content": "You are an expert evaluator. Respond only in the requested JSON format."},
            {"role": "user", "content": prompt}
        ]

        api_response = await self._call_api(messages, temperature=0.0, max_tokens=16384)
        judge_result = self._parse_judge_result(api_response)

        is_correct = judge_result.get("correct", "").upper() == "YES"
        confidence = judge_result.get("confidence", "MEDIUM")

        return is_correct, confidence, judge_result


# ==================== Batch Processing ====================

async def process_single_item(
    item: Dict[str, Any],
    llm_judge_client: LLMJudgeClient,
    file_name: str
) -> Optional[Dict[str, Any]]:
    """
    Process single data item
    Returns: merged_item_for_jsonl or None if error
    """
    idx = item.get("idx", -1)

    # Judge response uses answer_extracted
    # answer_extracted = item.get("answer_extracted", "no answer")
    answer_extracted = item.get("prediction", "no answer")
    # answer_extracted = answer_extracted[-500:]

    # Question directly from extra_info.question
    extra_info = item.get("extra_info", {})
    question = extra_info.get("question", "")

    # Extract ground truth
    ground_truth = safe_get_ground_truth(item)
    correct_answer = format_correct_answer(ground_truth)

    try:
        # Call LLM Judge
        is_correct, confidence, judge_result = await llm_judge_client.judge(
            question=question,
            response=answer_extracted,
            correct_answer=correct_answer,
        )

        # Build merged complete item (for saving to JSONL)
        merged_item = dict(item)
        merged_item["judge_result"] = {
            "question": question,
            "answer_evaluated": answer_extracted,
            "correct_answer": correct_answer,
            "reasoning": judge_result.get("reasoning", ""),
            "correct": judge_result.get("correct", ""),
            "confidence": judge_result.get("confidence", ""),
            "is_correct": is_correct,
            "judge_timestamp": datetime.now().isoformat(),
            "status": "success"
        }

        return merged_item

    except Exception as e:
        error_msg = str(e)
        print(f"  [Error] Failed to judge item {idx} in {file_name}: {error_msg}")

        merged_item = dict(item)
        merged_item["judge_result"] = {
            "question": question,
            "answer_evaluated": answer_extracted,
            "correct_answer": correct_answer,
            "extracted_final_answer": "",
            "reasoning": error_msg,
            "correct": "NO",
            "confidence": "LOW",
            "is_correct": False,
            "judge_timestamp": datetime.now().isoformat(),
            "status": f"error: {error_msg}"
        }

        return merged_item


async def process_jsonl_file(
    file_path: Path,
    llm_judge_client: LLMJudgeClient,
    output_dir: Path,
    max_items: Optional[int] = None,
    concurrency: int = 10
) -> Optional[Path]:
    """
    Process single jsonl file (concurrent processing)
    Returns: output_jsonl_path or None
    """
    file_name = file_path.name
    output_jsonl_path = output_dir / file_name

    # Check if output file exists, skip if exists
    if output_jsonl_path.exists():
        print(f"\n{'='*80}")
        print(f"Skipping: {file_name} (already exists at {output_jsonl_path})")
        print(f"{'='*80}")
        return output_jsonl_path

    print(f"\n{'='*80}")
    print(f"Processing: {file_name}")
    print(f"{'='*80}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[Error] Failed to read {file_name}: {e}")
        return None

    total = len(lines)
    if max_items:
        total = min(total, max_items)
        lines = lines[:max_items]

    # Parse all valid JSON lines
    items_to_process = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            items_to_process.append((item, i))
        except json.JSONDecodeError as e:
            print(f"  [Error] JSON parse error at line {i+1}: {e}")
            continue

    total_items = len(items_to_process)
    if total_items == 0:
        print(f"  [Warning] No valid items found in {file_name}")
        return None

    print(f"  Total items to process: {total_items}")
    print(f"  Concurrency: {concurrency}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use async lock to ensure safe concurrent write
    write_lock = asyncio.Lock()

    # Open file for streaming write
    jsonl_file = open(output_jsonl_path, 'w', encoding='utf-8')

    try:
        # Create semaphore to control concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def process_with_semaphore(item_with_idx):
            """Processing function with semaphore control"""
            async with semaphore:
                item, idx = item_with_idx
                return await process_single_item(item, llm_judge_client, file_name)

        # Concurrently process all items - use as_completed for real-time progress
        print(f"  Starting concurrent processing...")

        # Create all tasks
        tasks = [process_with_semaphore(item_with_idx) for item_with_idx in items_to_process]

        success_count = 0
        error_count = 0
        processed_count = 0

        # Use as_completed to process by completion order for real-time progress
        for completed_task in asyncio.as_completed(tasks):
            merged_item = await completed_task

            if isinstance(merged_item, Exception):
                print(f"  [Error] Processing exception: {merged_item}")
                error_count += 1
                processed_count += 1
                continue

            if merged_item:
                # Streaming write to JSONL - write immediately after each item processed
                async with write_lock:
                    json_line = json.dumps(merged_item, ensure_ascii=False)
                    jsonl_file.write(json_line)
                    jsonl_file.write("\n")
                    jsonl_file.flush()

                status = merged_item.get("judge_result", {}).get("status", "")
                if status == "success":
                    success_count += 1
                else:
                    error_count += 1

            processed_count += 1

            # Real-time progress print - print after each item
            print(f"  Progress: {processed_count}/{total_items} | Success: {success_count} | Error: {error_count}")

        print(f"\n[Summary] {file_name}:")
        print(f"  Total items: {processed_count}")
        print(f"  Success: {success_count} | Errors: {error_count}")
        print(f"  [Saved] Judged JSONL: {output_jsonl_path}")

        return output_jsonl_path

    finally:
        # Ensure file closed properly
        jsonl_file.close()


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Batch LLM Judge with JSONL Output (Skip existing)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for judged JSONL files")
    parser.add_argument("--sglang-url", type=str, default=SGLANG_URL, help="SGLang server URL")
    parser.add_argument("--model-path", type=str, default=None, help="Model path/name")
    parser.add_argument("--base-dir", type=str, default=str(BASE_DIR), help="Input directory containing jsonl files")
    parser.add_argument("--pattern", type=str, default="*.jsonl", help="File pattern to match")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--max-items", type=int, default=None, help="Maximum items per file")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests (default: 10)")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)

    if not base_dir.exists():
        print(f"[Error] Input directory not found: {base_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Judged JSONL files will be saved to: {output_dir}")
    print(f"[Info] Existing files will be skipped")

    # Find all jsonl files
    jsonl_files = list(base_dir.glob(args.pattern))
    if not jsonl_files:
        print(f"[Warning] No {args.pattern} files found in {base_dir}")
        sys.exit(0)

    jsonl_files.sort()
    if args.max_files:
        jsonl_files = jsonl_files[:args.max_files]

    print(f"\n{'#'*80}")
    print(f"Batch LLM Judge Evaluation with JSONL Output")
    print(f"{'#'*80}")
    print(f"Input directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {len(jsonl_files)}")
    print(f"SGLang URL: {args.sglang_url}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize LLM Judge Client
    print(f"\n[Init] Initializing LLM Judge client...")
    llm_judge_client = LLMJudgeClient(
        sglang_url=args.sglang_url,
        model_path=args.model_path,
        max_retries=3,
        timeout=60.0,
    )

    processed_files = []
    skipped_files = []

    try:
        # Process files one by one
        for file_path in jsonl_files:
            output_path = await process_jsonl_file(
                file_path, llm_judge_client, output_dir, args.max_items, args.concurrency
            )
            if output_path:
                if output_path.exists() and output_path.stat().st_size > 0:
                    # Check if newly generated or skipped
                    # Since process_jsonl_file prints skip info internally, handle simply
                    processed_files.append(output_path)

    except KeyboardInterrupt:
        print("\n\n[Interrupted] Processing interrupted by user")
    finally:
        await llm_judge_client.close()
        print("\n[Cleanup] LLM Judge client closed")

        # Statistics
        print(f"\n{'#'*80}")
        print(f"Final Summary")
        print(f"{'#'*80}")
        print(f"Total files processed/skipped: {len(processed_files)}")
        print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
