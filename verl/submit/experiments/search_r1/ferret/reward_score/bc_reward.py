import json
import re
import time
import random
import threading
from time import sleep
import requests

import os

# ============================================================
# 1) vLLM endpoint health (沿用你第一个 reward 的模式)
# ============================================================
IP_FILE = os.environ["IP_FILE"]
MODEL_NAME = os.environ.get("RM_MODEL_NAME", "q")

MAX_RETRIES = 3
BASE_DELAY = 4
COOLDOWN_SECONDS = 240
TIMEOUT_SECONDS = 60

_lock = threading.Lock()
_BAD_ENDPOINTS = {}  # endpoint -> last_fail_time


def load_vllm_endpoints():
    with open(IP_FILE, "r") as f:
        eps = [f"http://{line.strip()}" for line in f if line.strip()]
    if not eps:
        raise RuntimeError("No vLLM endpoints found.")
    return eps


_VLLM_ENDPOINTS = load_vllm_endpoints()


def mark_endpoint_bad(endpoint: str):
    with _lock:
        _BAD_ENDPOINTS[endpoint] = time.time()


def mark_endpoint_good(endpoint: str):
    with _lock:
        _BAD_ENDPOINTS.pop(endpoint, None)


def get_alive_endpoint() -> str:
    now = time.time()
    with _lock:
        healthy = [
            e for e in _VLLM_ENDPOINTS
            if (e not in _BAD_ENDPOINTS) or (now - _BAD_ENDPOINTS[e] > COOLDOWN_SECONDS)
        ]
    if not healthy:
        raise RuntimeError("No healthy vLLM endpoints available.")
    return random.choice(healthy)


def vllm_chat(messages, max_retries=MAX_RETRIES) -> str:
    last_err = None
    for attempt in range(max_retries):
        base_url = get_alive_endpoint()
        chat_url = f"{base_url}/v1/chat/completions"
        try:
            resp = requests.post(
                chat_url,
                headers={"Content-Type": "application/json"},
                json={"model": MODEL_NAME, "messages": messages},
                timeout=TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            mark_endpoint_good(base_url)
            return content
        except Exception as e:
            last_err = e
            mark_endpoint_bad(base_url)
            if attempt < max_retries - 1:
                sleep(BASE_DELAY * (2 ** attempt))
    raise ConnectionRefusedError(f"All retries failed: {repr(last_err)}")


# ============================================================
# 2) Judge prompt (按你 GRADER_TEMPLATE 思路：先提取再判断)
# ============================================================

EM_LLM_GRADER_SYSTEM = """
You are a strict grader.

You MUST:
1) extract the final answer from the response (or from the provided extracted_answer if present).
2) compare it to the correct_answer with robust equivalence:
   - ignore case, punctuation, quotes, diacritics
   - allow minor wording/order changes
   - allow standard abbreviations (Inc., Ltd., FC etc.)
   - allow small numerical tolerance if obviously numeric
Return ONLY valid JSON with:
{
  "extracted_final_answer": string|null,
  "correct": true|false,
  "answer_score": number,   // 0..1, can be partial
  "retrieval_correct": true|false,
  "confidence": number,     // 0..100
  "reason": string
}
No extra text.
""".strip()

EM_LLM_GRADER_USER = """
[correct_answer]:
{correct_answer}

[targets_json]:
{targets_json}

[extracted_answer_from_tags]:
{extracted_answer}

[retrieval_blocks]:
{retrieval_blocks_json}

[full_solution_str]:
{solution_str}
""".strip()


def _extract_json_loose(text: str):
    """优先找 ```json```，否则捞最后一个 {...}"""
    if not text:
        return None
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    # try last json object
    cands = re.findall(r"\{.*\}", text, flags=re.S)
    for cand in reversed(cands):
        try:
            return json.loads(cand)
        except Exception:
            continue
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def _clamp(x, lo=0.0, hi=1.0):
    try:
        x = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, x))


# ============================================================
# 3) compute_score_em_llm (参考 compute_score_em 的格式)
# 依赖你已有的函数：
# - is_valid_sequence
# - extract_solution
# - extract_information_blocks
# - is_retrieval_correct
# - em_check
# ============================================================

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score
    

def is_valid_sequence(text):
    # Remove role tags (user/assistant) that may appear between structured tags
    content = text

    # First pass: Simply remove lines that contain only role tags
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Only skip lines that are exactly "user" or "assistant"
        if stripped not in ['user', 'assistant']:
            cleaned_lines.append(line)

    # Rejoin the lines
    content = '\n'.join(cleaned_lines)


    # Check for balanced tags - now using think, tool_call, tool_response, answer
    tags_to_check = ["think", "tool_call", "tool_response", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"

    # Now check for proper sequence pattern and no extraneous content

    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|tool_call|tool_response|answer)>)"
    parts = re.split(split_pattern, content)

    # 2. Keep track of the current position in the expected sequence
    # start -> [<think> -> <tool_call> -> <tool_response>]* -> <think> -> <answer> -> end
    state = "start"

    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue

        # Check if this is a tag
        if re.match(r"</?(?:think|tool_call|tool_response|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "after_tool_response"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<tool_call>" and state == "after_think":
                state = "in_tool_call"
            elif part == "</tool_call>" and state == "in_tool_call":
                state = "after_tool_call"
            elif part == "<tool_response>" and state == "after_tool_call":
                state = "in_tool_response"
            elif part == "</tool_response>" and state == "in_tool_response":
                state = "after_tool_response"
            elif part == "<answer>" and state == "after_think":
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_tool_call", "in_tool_response", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_tool_call", "after_tool_response"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            else:
                return False, f"Unexpected content in state {state}"

    # Check final state
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"

    return True, "Valid sequence format"


def extract_solution(solution_str):
    """Extract the answer from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are no matches, return None
    if len(matches) == 0:
        return None

    # Return the last answer tag content
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> list[str]:
    """Extract information from <tool_response> tags."""
    pattern = r"<tool_response>(.*?)</tool_response>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def is_retrieval_correct(text: str, golden_answers: list[str]) -> list[str]:
    seqs = extract_information_blocks(text)
    for seq in seqs:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(seq):
                return True
    return False


def compute_score_em_llm(
    solution_str,
    ground_truth,
    data_source=None,
    extra_info=None,
    structure_format_score=0.2,
    final_format_score=0.1,
    retrieval_score=0.1,
    score=1.0,
    llm_weight=0.7,  # LLM reward 占比（保留规则保底）
    *args, **kwargs
):
    # --------- 1) 结构 & 提取（先提取！）---------
    is_valid_format, _ = is_valid_sequence(solution_str)

    retrieval_blocks = extract_information_blocks(solution_str)
    num_tool_calls = float(len(retrieval_blocks))

    extracted_answer = extract_solution(solution_str=solution_str)  # 取最后一个 <answer>...</answer>
    has_answer = extracted_answer is not None

    targets = ground_truth.get("target", [])
    if isinstance(targets, str):
        targets = [targets]
    correct_answer = targets[0] if targets else ""  # judge 用主要 label（也把 targets 全传进去）

    # --------- 2) 规则强判：命中直接满分（像你 judge() 一样省 LLM）---------
    answer_correct_rule = False
    if has_answer:
        try:
            answer_correct_rule = bool(em_check(extracted_answer, targets))
        except Exception:
            answer_correct_rule = False

    retrieval_correct_rule = False
    if is_valid_format:
        try:
            retrieval_correct_rule = bool(is_retrieval_correct(solution_str, targets))
        except Exception:
            retrieval_correct_rule = False

    if answer_correct_rule:
        final_reward = float(score if is_valid_format else (score - structure_format_score))
        return {
            "score": final_reward,
            "format_valid": 1.0 if is_valid_format else 0.0,
            "has_answer": 1.0 if has_answer else 0.0,
            "acc": 1.0,
            "num_tool_calls": num_tool_calls,
            "reward_final": final_reward,
            "retrieval_correct": 1.0 if retrieval_correct_rule else 0.0,
            "judge_used": 0.0,
            "judge_reason": "rule_em_hit",
            "answer_score": 1.0,
        }

    # --------- 3) LLM judge：规则不确定才调用（提取 + 判断）---------
    judge_prompt = EM_LLM_GRADER_USER.format(
        correct_answer=correct_answer,
        targets_json=json.dumps(targets, ensure_ascii=False),
        extracted_answer=(extracted_answer if extracted_answer is not None else "None"),
        retrieval_blocks_json=json.dumps(retrieval_blocks, ensure_ascii=False),
        solution_str=solution_str,
    )

    messages = [
        {"role": "system", "content": EM_LLM_GRADER_SYSTEM},
        {"role": "user", "content": judge_prompt},
    ]

    judge_obj = None
    judge_text = ""
    for attempt in range(MAX_RETRIES):
        try:
            judge_text = vllm_chat(messages)
            judge_obj = _extract_json_loose(judge_text)
            if isinstance(judge_obj, dict) and ("correct" in judge_obj):
                break
        except Exception:
            judge_obj = None
        sleep(BASE_DELAY * (2 ** attempt))

    # --------- 4) fallback：LLM 失败则退回原 rule 框架 ---------
    if not isinstance(judge_obj, dict):
        # 和你 compute_score_em 的 fallback 结构一致
        if not has_answer:
            if is_valid_format:
                final_reward = structure_format_score + (retrieval_score if retrieval_correct_rule else 0.0)
            else:
                final_reward = 0.0
        else:
            if is_valid_format:
                final_reward = structure_format_score + (retrieval_score if retrieval_correct_rule else 0.0)
            else:
                final_reward = final_format_score

        return {
            "score": float(final_reward),
            "format_valid": 1.0 if is_valid_format else 0.0,
            "has_answer": 1.0 if has_answer else 0.0,
            "acc": 0.0,
            "num_tool_calls": num_tool_calls,
            "reward_final": float(final_reward),
            "retrieval_correct": 1.0 if retrieval_correct_rule else 0.0,
            "judge_used": 0.0,
            "judge_reason": "llm_parse_failed_fallback",
            "answer_score": 0.0,
        }

    # --------- 5) 融合：保留 format/retrieval 保底 + LLM 部分正确能力 ---------
    llm_correct = bool(judge_obj.get("correct", False))
    llm_answer_score = _clamp(judge_obj.get("answer_score", 1.0 if llm_correct else 0.0))
    llm_retrieval_correct = bool(judge_obj.get("retrieval_correct", False))
    reason = str(judge_obj.get("reason", ""))[:500]

    # 规则保底
    base = 0.0
    if not has_answer:
        base = (structure_format_score if is_valid_format else 0.0) + (
            retrieval_score if (retrieval_correct_rule or llm_retrieval_correct) else 0.0
        )
        # 没 answer 的情况下，LLM 不应把分抬太高
        llm_part = 0.35 * llm_answer_score if is_valid_format else 0.15 * llm_answer_score
        final_reward = max(base, llm_part)
    else:
        # 有 answer：允许部分正确；完全正确仍接近满分
        format_part = (structure_format_score if is_valid_format else final_format_score)
        retrieval_part = retrieval_score if (retrieval_correct_rule or llm_retrieval_correct) else 0.0

        # LLM 主导部分：answer_score -> [0,1]
        llm_part = llm_answer_score
        if llm_correct and llm_answer_score < 0.95:
            llm_part = 1.0  # correct=true 但打分偏低，做保护

        fused = (1.0 - llm_weight) * (format_part + retrieval_part) + llm_weight * llm_part
        # 至少不低于 format/retrieval 的保底
        final_reward = max(format_part + retrieval_part, fused)
        final_reward = float(_clamp(final_reward))

        # 若 LLM 判断完全正确：对齐你原始逻辑（format invalid 扣分）
        if llm_correct:
            final_reward = float(score if is_valid_format else (score - structure_format_score))

    return {
        "score": float(final_reward),
        "format_valid": 1.0 if is_valid_format else 0.0,
        "has_answer": 1.0 if has_answer else 0.0,
        "acc": 1.0 if llm_correct else 0.0,
        "num_tool_calls": num_tool_calls,
        "reward_final": float(final_reward),
        "retrieval_correct": 1.0 if (retrieval_correct_rule or llm_retrieval_correct) else 0.0,
        "judge_used": 1.0,
        "judge_reason": reason,
        "answer_score": float(llm_answer_score),
        "extracted_final_answer_llm": judge_obj.get("extracted_final_answer", None),
    }
