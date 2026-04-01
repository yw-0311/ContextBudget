import os
import re
import time
import string
import random
import requests
from typing import Optional, Dict, List, Union


# ==========================================================
# 1) 原规则部分：normalize + EM + extract + tool block
# ==========================================================

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers: Union[str, List[str]]) -> int:
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


def extract_solution(solution_str: str) -> Optional[str]:
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if len(matches) == 0:
        return None
    return matches[-1].group(1).strip()


def extract_information_blocks(text: str) -> List[str]:
    """Extract information from <tool_response> tags."""
    pattern = r"<tool_response>(.*?)</tool_response>"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


# ==========================================================
# 2) vLLM Judge reward model 部分
# ==========================================================

# export IP_FILE="/data/oss_bucket_0/shiyi/log/TEST/train/vllm_ip_0.txt"
IP_FILE = os.environ.get("IP_FILE", "")
print(f"[DEBUG] {IP_FILE}")
MODEL_NAME = "q"
MAX_RETRIES = 3
BASE_DELAY = 2
TIMEOUT = 60

if not IP_FILE:
    print("[WARN] IP_FILE is not set. Judge fallback will fail unless you set it.")


GRADER_TEMPLATE = """
Judge whether the following [response] is correct or not based on the precise and unambiguous [correct_answer] below.

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. 

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|%| and 100|% from [response]. Put 100 if there is no confidence score available.
""".strip()


def load_vllm_endpoints(ip_file: str) -> List[str]:
    if not ip_file or not os.path.exists(ip_file):
        raise RuntimeError(f"Cannot find vLLM IP file: {ip_file}")
    with open(ip_file, "r") as f:
        eps = [f"http://{line.strip()}" for line in f if line.strip()]
    if not eps:
        raise RuntimeError("No vLLM endpoints found.")
    return eps


def pick_endpoint(endpoints: List[str]) -> str:
    return random.choice(endpoints)


def vllm_chat(endpoint: str, messages, temperature=0.0, timeout=TIMEOUT) -> str:
    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def build_judge_prompt(pred: str, gold: str, question: str = "N/A") -> str:
    return GRADER_TEMPLATE.format(
        question=question,
        response=pred,
        correct_answer=gold,
    )


def parse_judge_correct(text: str) -> Optional[bool]:
    if not text:
        return None
    m = re.search(r"correct:\s*(yes|no)", text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower() == "yes"


def parse_judge_confidence(text: str) -> float:
    if not text:
        return 100.0
    m = re.search(r"confidence:\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not m:
        return 100.0
    val = float(m.group(1))
    return min(val, 100.0)


def semantic_judge_reward(
    pred: str,
    gold: str,
    question: str = "N/A",
    reward_correct: float = 1.0,
    reward_incorrect: float = 0.0,
    min_conf: float = 0.0,
    do_print: int = 0,
) -> Dict:
    endpoints = load_vllm_endpoints(IP_FILE)
    prompt = build_judge_prompt(pred, gold, question)
    messages = [{"role": "user", "content": prompt}]

    judge_text = ""
    correct_flag = None
    confidence = 100.0

    for attempt in range(MAX_RETRIES):
        try:
            endpoint = pick_endpoint(endpoints)
            judge_text = vllm_chat(endpoint, messages, temperature=0.0)
            correct_flag = parse_judge_correct(judge_text)
            confidence = parse_judge_confidence(judge_text)

            if correct_flag is not None:
                break
        except Exception as e:
            if do_print:
                print(f"[Judge attempt {attempt+1}] error: {repr(e)}")
            time.sleep(BASE_DELAY * (2 ** attempt))

    if correct_flag is None:
        return {
            "reward": 0.0,
            "correct": False,
            "confidence": 0.0,
            "judge_text": judge_text[:500],
            "error": "judge_parse_failed",
        }

    ok = correct_flag and confidence >= min_conf
    reward = reward_correct if ok else reward_incorrect

    print("====== Judge Result ======")
    print("correct =", correct_flag)
    print("confidence =", confidence)
    print("reward =", reward)
    print("judge_text =", judge_text[:300])

    return {
        "reward": float(reward),
        "correct": bool(correct_flag),
        "confidence": float(confidence),
        "judge_text": judge_text[:500],
    }


# ==========================================================
# 3) ✅ 融合 reward：规则错 -> 调用 judge
# ==========================================================

def compute_score_em_with_judge(
    solution_str,
    ground_truth,
    data_source=None,
    extra_info=None,
    structure_format_score=0,
    final_format_score=0,
    retrieval_score=0,
    format_score=0,
    score=1.0,
    use_judge_when_em_wrong=True,
    judge_reward_correct=1.0,
    judge_reward_incorrect=0.0,
    judge_min_conf=0.0,
    do_print=0,
    *args,
    **kwargs
) -> Dict:
    """
    规则 EM 优先，如果规则判错 -> 调用 vLLM judge 判断语义等价
    """

    # 1) 提取 answer
    answer = extract_solution(solution_str=solution_str)
    answer_len = len(answer) if answer is not None else 0

    # 2) 规则 EM 判断
    answer_correct = False
    if answer is not None:
        answer_correct = em_check(answer, ground_truth["target"])

    # 3) tool calls 数量统计
    num_tool_calls = len(extract_information_blocks(solution_str))

    # 4) 规则 reward
    if answer is None:
        final_reward = 0.0
        rule_correct = False
    else:
        rule_correct = bool(answer_correct)
        final_reward = float(score) if rule_correct else 0.0

    # 5) 如果规则错 -> fallback judge
    judge_used = 0.0
    judge_correct = 0.0
    judge_confidence = 0.0
    judge_text = ""
    judge_error = ""

    if use_judge_when_em_wrong and answer and final_reward == 0.0:
        judge_used = 1.0

        pred_for_judge = answer if answer is not None else solution_str

        gold_targets = ground_truth["target"]
        if isinstance(gold_targets, list):
            gold_for_judge = gold_targets[0] if len(gold_targets) > 0 else ""
        else:
            gold_for_judge = str(gold_targets)

        # question 用 extra_info 里的
        question = "N/A"
        if isinstance(extra_info, dict):
            question = extra_info.get("question", "N/A")

        try:
            ret = semantic_judge_reward(
                pred=pred_for_judge,
                gold=gold_for_judge,
                question=question,
                reward_correct=judge_reward_correct,
                reward_incorrect=judge_reward_incorrect,
                min_conf=judge_min_conf,
                do_print=do_print,
            )

            final_reward = float(ret["reward"])
            judge_correct = 1.0 if ret.get("correct", False) else 0.0
            judge_confidence = float(ret.get("confidence", 0.0))
            judge_text = ret.get("judge_text", "")

        except Exception as e:
            judge_error = repr(e)
            final_reward = 0.0

    # 6) 返回兼容 RewardManager 的 dict
    return {
        "score": float(final_reward),

        # base metrics
        "has_answer": 1.0 if answer is not None else 0.0,
        "answer_len": float(answer_len), 
        "acc_rule": 1.0 if rule_correct else 0.0,
        "num_tool_calls": float(num_tool_calls),
        "reward_final": float(final_reward),

        # judge metrics
        "judge_used": float(judge_used),
        "acc_judge": float(judge_correct),
        "judge_confidence": float(judge_confidence),
        "judge_error": judge_error,
        "judge_text": judge_text[:500],
    }


# ==========================================================
# 4) Quick test
# ==========================================================
if __name__ == "__main__":
    # 模拟数据
    solution_str = "something B"
    ground_truth = {"target": ["something B"]}
    extra_info = {"question": "dummy question"}

    ret = compute_score_em_with_judge(
        solution_str=solution_str,
        ground_truth=ground_truth,
        data_source=None,
        extra_info=extra_info,
        score=1.0,
        use_judge_when_em_wrong=True,
        do_print=1,
    )
    print("==== FINAL RET ====")
    print(ret)
