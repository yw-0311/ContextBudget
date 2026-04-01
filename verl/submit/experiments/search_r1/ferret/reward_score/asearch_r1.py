import re
import string
import random
import numpy as np

def normalize_answer(s):
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


def extract_solution(solution_str):
    """Extract the answer from the solution string, assuming it's in the last part."""
    search_window = solution_str[-500:] if len(solution_str) > 500 else solution_str
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, search_window, re.DOTALL))
    
    if not matches:
        return None
    
    return matches[-1].group(1).strip()


def f1_score(answer_content, gt):
    answer_content = normalize_answer(answer_content)
    gt = normalize_answer(gt)

    pred_tokens = set(answer_content.split())
    gt_tokens = set(gt.split())
    
    if not gt_tokens or not pred_tokens:
        return 0
    
    common_tokens = pred_tokens & gt_tokens
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
    recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
    
    f1 = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def compute_score_f1(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    # If ground_truth is a numpy array, convert it to a list
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()
    
    if isinstance(ground_truth, list):
        results = [compute_score_f1(solution_str, g) for g in ground_truth]
        return max(results, key=lambda x: x["score"])

    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        return {
            "score": 0,
            "has_answer": 0.0,
            "acc": 0.0,
            "num_tool_calls": 0.0,
            "reward_final": 0.0,
        }
    else:
        ret_score = f1_score(answer, ground_truth)
        
        # Calculate whether the answer is correct (accuracy)
        is_correct = 1.0 if ret_score > 0 else 0.0
        
        # Assume tool calls are extracted (dummy value for this example)
        num_tool_calls = 0.0  # You should calculate this based on your task
        
        return {
            "score": ret_score,  # F1 score as the reward
            "has_answer": 1.0 if answer is not None else 0.0,
            "acc": is_correct,  # Accuracy based on F1 score
            "num_tool_calls": num_tool_calls,  # Tool calls, which you can calculate
            "reward_final": ret_score,  # Final reward is based on F1 score
        }


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    if isinstance(ground_truth, list):
        answer = extract_solution(solution_str=solution_str)
        return answer, max([compute_score_em(solution_str, g)[1] for g in ground_truth])

    answer = extract_solution(solution_str=solution_str)

    if answer is None:
        return None, 0
    else:
        if em_check(answer, ground_truth):
            return answer, score
        else:
            return answer, format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def normalize_text(text: str) -> str:
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text
