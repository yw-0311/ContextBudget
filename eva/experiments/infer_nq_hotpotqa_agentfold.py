# infer_nq_hotpotqa_agentfold.py
# AgentFold version for NQ-HotpotQA evaluation
# Decoupled from verl directory
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer
from openai import OpenAI

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentfold.agentfold_search_loop_nq_hotpotqa import AgentFoldSearchLoop

# Import scoring functions - need to adjust path for verl imports
sys.path.append('/home/wy517954/code/Elistic-Context-Fold-Verl/verl')
from submit.experiments.search_r1.ferret.reward_score.search_r1_format import (
    compute_score_multi_answer,
    extract_solution
)

import swanlab


# =============================================================================
# SGLang Server Manager compatible with AgentFold
# =============================================================================
class SGLangServerManager:
    """Simple wrapper around OpenAI client for SGLang server."""

    def __init__(self, base_url: str, timeout_s: float = 1000.0):
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
            timeout=timeout_s
        )

    def completions(self):
        """Return completions API."""
        return self.client.completions


# =============================================================================
# JSON-safe conversion
# =============================================================================
def maybe_json_load(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    t = s.strip()
    if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
        try:
            return json.loads(t)
        except Exception:
            return s
    return s


def to_python_scalar(x: Any) -> Any:
    x = maybe_json_load(x)

    # numpy
    try:
        import numpy as np  # type: ignore

        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
    except Exception:
        pass

    # torch
    try:
        import torch  # type: ignore

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
    except Exception:
        pass

    return x


def to_jsonable(x: Any) -> Any:
    x = to_python_scalar(x)

    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [to_jsonable(v) for v in x]
    return str(x)


# =============================================================================
# Row parsing helpers
# =============================================================================
def as_dict(x: Any) -> Dict[str, Any]:
    x = to_python_scalar(x)
    return x if isinstance(x, dict) else {}


def parse_messages(row: Dict[str, Any], prompt_col: str) -> List[Dict[str, str]]:
    raw = row.get(prompt_col, None)
    raw = to_python_scalar(raw)

    if not isinstance(raw, list):
        raise TypeError(
            f"prompt_col '{prompt_col}' expected list, got {type(raw)}; preview={str(raw)[:200]}"
        )

    msgs: List[Dict[str, str]] = []
    for j, m in enumerate(raw):
        m = to_python_scalar(m)
        if not isinstance(m, dict):
            raise TypeError(f"prompt[{j}] expected dict, got {type(m)}; preview={str(m)[:200]}")
        role = m.get("role", None)
        content = m.get("content", None)
        if role is None or content is None:
            raise ValueError(f"prompt[{j}] missing role/content keys: keys={list(m.keys())}")
        msgs.append({"role": str(role), "content": str(content)})
    return msgs


def parse_extra_info(row: Dict[str, Any]) -> Dict[str, Any]:
    return as_dict(row.get("extra_info", None))


def parse_tools_kwargs(extra_info: Dict[str, Any]) -> Dict[str, Any]:
    need = extra_info.get("need_tools_kwargs", False)
    if isinstance(need, str):
        need = need.lower() in ("1", "true", "yes")
    if not need:
        return {}
    return as_dict(extra_info.get("tools_kwargs", None))


def parse_ground_truth(row: Dict[str, Any], extra_info: Dict[str, Any]) -> Dict[str, Any]:
    rm = as_dict(row.get("reward_model", None))
    gt = as_dict(rm.get("ground_truth", None))
    if gt:
        return gt
    gt2 = as_dict(extra_info.get("ground_truth", None))
    return gt2 if gt2 else {"target": None}


def extract_last_assistant(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "assistant":
            return m.get("content", "") or ""
    return ""


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


# =============================================================================
# Inference
# =============================================================================
async def infer_one(
    agent_loop: AgentFoldSearchLoop,
    messages: List[Dict[str, Any]],
    sampling_params: Dict[str, Any],
    tools_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    outs = await agent_loop.run(
        sampling_params=sampling_params,
        raw_prompt=messages,
        image_data=None,
        tools_kwargs=tools_kwargs or {},
    )

    out = outs[-1]
    final_msgs = out.extra_fields.get("final_messages", [])

    pred = extract_last_assistant(final_msgs)

    result = {
        "prediction": pred,
        "num_turns": out.num_turns,
        "last_branch_depth": out.branch_depth,
        "prompt_ids_len": len(out.prompt_ids) if hasattr(out, 'prompt_ids') else 0,
        "response_ids_len": len(out.response_ids) if hasattr(out, 'response_ids') else 0,
        "final_messages": final_msgs,
    }

    return result


# =============================================================================
# Reward
# =============================================================================
async def compute_reward_async(
    prediction: str,
    ground_truth: Dict[str, Any],
    data_source: str,
    extra_info: Dict[str, Any],
    *,
    val_type: str,
    cot: bool,
    require_same_len: bool,
    target: int,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Use unified multi-answer scoring function.
    """
    reward_extra = compute_score_multi_answer(
        solution_str=prediction,
        ground_truth=ground_truth,
        data_source=data_source,
        extra_info=extra_info,
        val_type=val_type,
        cot=cot,
        require_same_len=require_same_len,
        target=target
    )

    score = reward_extra["score"]
    return float(score), reward_extra


# =============================================================================
# Concurrent worker
# =============================================================================
async def process_row(
    idx: int,
    row: Dict[str, Any],
    *,
    agent_loop: AgentFoldSearchLoop,
    sampling_params: Dict[str, Any],
    prompt_col: str,
    semaphore: asyncio.Semaphore,
    val_type: str,
    cot: bool,
    require_same_len: bool,
    target: int,
) -> Tuple[bool, Dict[str, Any], Optional[float], float, Optional[float]]:
    async with semaphore:
        data_source = str(to_python_scalar(row.get("data_source", "unknown")))
        extra_info = parse_extra_info(row)
        tools_kwargs = parse_tools_kwargs(extra_info)
        ground_truth = parse_ground_truth(row, extra_info)
        messages = parse_messages(row, prompt_col)

        one = await infer_one(agent_loop, messages, sampling_params, tools_kwargs)

        pred = one["prediction"]
        ans = extract_solution(pred)
        last_depth = _safe_float_or_none(one.get("last_branch_depth", None))

        score, reward_extra = await compute_reward_async(
            pred,
            ground_truth,
            data_source,
            extra_info,
            val_type=val_type,
            cot=cot,
            require_same_len=require_same_len,
            target=target
        )

        has_answer = _safe_float(
            reward_extra.get("has_answer", None),
            default=1.0 if ans is not None else 0.0,
        )

        out_row = {
            "idx": idx,
            "data_source": data_source,
            "agent_name": row.get("agent_name", None),
            "ability": row.get("ability", None),
            "prompt": messages,
            "reward_model": row.get("reward_model", None),
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            "tools_kwargs": tools_kwargs,
            "prediction": pred,
            "answer_extracted": ans,
            "score": score,
            "reward_extra": reward_extra,
            "has_answer": has_answer,
            "num_turns": one["num_turns"],
            "prompt_ids_len": one["prompt_ids_len"],
            "response_ids_len": one["response_ids_len"],
            "final_messages": one["final_messages"],
            "last_branch_depth": last_depth,
        }

        return True, to_jsonable(out_row), score, has_answer, last_depth


# =============================================================================
# Main
# =============================================================================
async def main() -> None:
    ap = argparse.ArgumentParser()

    # DataSet
    ap.add_argument("--dataset", type=str, default="nq_hotpotqa")

    # IO
    ap.add_argument("--parquet", type=str, required=True)
    ap.add_argument("--prompt_col", type=str, default="prompt")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--out_jsonl", type=str, required=True)

    # sglang + model
    ap.add_argument("--sglang_url", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)

    # AgentFold specific settings
    ap.add_argument("--max_iteration", type=int, default=6, help="Max iterations for AgentFold loop")

    # sampling
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--timeout_s", type=float, default=1000.0)

    # runtime
    ap.add_argument("--flush_every", type=int, default=50)
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4)

    # multi-answer scoring config
    ap.add_argument("--val_type", type=str, default="f1", choices=["f1", "em", "mbe"])
    ap.add_argument("--cot", action="store_true", help="If True, append </answer> before parsing")
    ap.add_argument(
        "--require_same_len",
        action="store_true",
        help="If True, pred slot count must equal gt slot count else score=0",
    )

    # Target argument
    ap.add_argument("--target", type=int, required=True, help="The target for dataset (e.g., 2, 8, 16)")

    args = ap.parse_args()

    # ==================== SwanLab Initialization ====================
    model_short = args.model_path.split("/")[-1] if args.model_path else "unknown"
    project_name = f"{args.dataset}_eva"
    experiment_name = (
        f"agentfold"
        f"_{model_short}"
        f"_temp{args.temperature}"
        f"_{args.val_type}"
        f"_same{int(args.require_same_len)}"
        f"_target{args.target}"
        f"_iter{args.max_iteration}"
    )

    swanlab_enabled = False
    try:
        swanlab.init(project=project_name, experiment_name=experiment_name)
        swanlab.config.update(
            {
                "agent_type": "agentfold",
                "model_path": args.model_path,
                "model_short": model_short,
                "val_type": args.val_type,
                "require_same_len": int(args.require_same_len),
                "target": args.target,
                "max_iteration": args.max_iteration,
            }
        )
        swanlab_enabled = True
        print(f"[swanlab] Initialized run: {experiment_name}")
    except Exception as e:
        print(f"[swanlab] Warning: Failed to initialize SwanLab: {e}")
    # ================================================================

    df = pd.read_parquet(args.parquet)
    if args.prompt_col not in df.columns:
        raise ValueError(f"Column '{args.prompt_col}' not found. Columns={list(df.columns)}")
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    n_total = len(df)
    print(f"[info] loaded rows={n_total}, columns={list(df.columns)}")
    print(f"[score] val_type={args.val_type} cot={int(args.cot)} require_same_len={int(args.require_same_len)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    server_manager = SGLangServerManager(args.sglang_url, timeout_s=args.timeout_s)

    agent_loop = AgentFoldSearchLoop(
        server_manager=server_manager,
        tokenizer=tokenizer,
        max_iteration=args.max_iteration,
    )
    agent_loop.loop = asyncio.get_running_loop()

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }

    rows: List[Dict[str, Any]] = []
    for tup in df.itertuples(index=False):
        rows.append(tup._asdict() if hasattr(tup, "_asdict") else dict(tup))

    try:
        from tqdm import tqdm

        pbar = tqdm(total=n_total, desc="infer", dynamic_ncols=True)
    except Exception:
        pbar = None

    sem = asyncio.Semaphore(max(1, args.concurrency))

    # Running stats
    n_ok = 0
    n_err = 0
    reward_sum = 0.0
    reward_cnt = 0
    has_answer_sum = 0.0
    has_answer_cnt = 0
    stats_by_source: Dict[str, Dict[str, Any]] = {}

    def get_src_stat(src: str) -> Dict[str, Any]:
        if src not in stats_by_source:
            stats_by_source[src] = {
                "ok": 0,
                "err": 0,
                "reward_sum": 0.0,
                "reward_cnt": 0,
                "has_answer_sum": 0.0,
                "has_answer_cnt": 0,
            }
        return stats_by_source[src]

    t0 = time.time()

    tasks = [
        asyncio.create_task(
            process_row(
                i,
                rows[i],
                agent_loop=agent_loop,
                sampling_params=sampling_params,
                prompt_col=args.prompt_col,
                semaphore=sem,
                val_type=args.val_type,
                cot=args.cot,
                require_same_len=args.require_same_len,
                target=args.target
            )
        )
        for i in range(n_total)
    ]

    def update_progress() -> None:
        dt = time.time() - t0
        avg_s = dt / max(1, (n_ok + n_err))
        mean_reward = (reward_sum / reward_cnt) if reward_cnt else 0.0
        has_answer_rate = (has_answer_sum / has_answer_cnt) if has_answer_cnt else 0.0

        if pbar is not None:
            pbar.set_postfix(
                ok=n_ok,
                err=n_err,
                avg_s=f"{avg_s:.3f}",
                mean_reward=f"{mean_reward:.4f}",
                has_ans=f"{has_answer_rate:.4f}",
                sources=len(stats_by_source),
            )
            pbar.update(1)

    try:
        with open(args.out_jsonl, "w", encoding="utf-8") as out_f:
            for fut in asyncio.as_completed(tasks):
                try:
                    ok, safe_row, score, has_answer, last_depth = await fut
                    out_f.write(json.dumps(safe_row, ensure_ascii=False) + "\n")

                    ds = str(safe_row.get("data_source", "unknown"))
                    st = get_src_stat(ds)

                    if ok:
                        n_ok += 1
                        st["ok"] += 1

                        if score is not None:
                            reward_sum += float(score)
                            reward_cnt += 1
                            st["reward_sum"] += float(score)
                            st["reward_cnt"] += 1

                        has_answer_sum += float(has_answer)
                        has_answer_cnt += 1
                        st["has_answer_sum"] += float(has_answer)
                        st["has_answer_cnt"] += 1
                    else:
                        n_err += 1
                        st["err"] += 1

                except Exception as e:
                    n_err += 1
                    st = get_src_stat("__task_crash__")
                    st["err"] += 1
                    err_row = to_jsonable({"error": repr(e)})
                    out_f.write(json.dumps(err_row, ensure_ascii=False) + "\n")

                    if not args.continue_on_error:
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                        raise

                if args.flush_every > 0 and (n_ok + n_err) % args.flush_every == 0:
                    out_f.flush()
                update_progress()

    finally:
        if pbar is not None:
            pbar.close()

        dt = time.time() - t0
        mean_reward = (reward_sum / reward_cnt) if reward_cnt else 0.0
        has_answer_rate = (has_answer_sum / has_answer_cnt) if has_answer_cnt else 0.0

        print(
            f"[done] wrote {n_total} rows to {args.out_jsonl}, ok={n_ok} err={n_err}, "
            f"avg={dt/max(1,n_total):.3f}s/item mean_reward={mean_reward:.4f} "
            f"has_answer_rate={has_answer_rate:.4f}"
        )

        def total_seen(v: Dict[str, Any]) -> int:
            return int(v.get("ok", 0)) + int(v.get("err", 0))

        swanlab_payload: Dict[str, Any] = {}

        swanlab_payload["has_answer_rate"] = has_answer_rate
        swanlab_payload["mean_reward"] = mean_reward
        swanlab_payload["mean_reward_target"] = mean_reward/args.target if args.target > 0 else 0

        print("[per_data_source]")
        sorted_sources = sorted(
            stats_by_source.items(), key=lambda kv: total_seen(kv[1]), reverse=True
        )
        for src, st in sorted_sources:
            ok_s = int(st.get("ok", 0))
            err_s = int(st.get("err", 0))
            total_s = ok_s + err_s

            cnt_s = int(st.get("reward_cnt", 0))
            sum_s = float(st.get("reward_sum", 0.0))
            mean_s = (sum_s / cnt_s) if cnt_s else 0.0

            ha_cnt = int(st.get("has_answer_cnt", 0))
            ha_sum = float(st.get("has_answer_sum", 0.0))
            ha_rate = (ha_sum / ha_cnt) if ha_cnt else 0.0

            print(
                f"  - {src}: seen={total_s} ok={ok_s} err={err_s} "
                f"reward_cnt={cnt_s} reward_sum={sum_s:.6f} mean_reward={mean_s:.6f} "
                f"has_answer_cnt={ha_cnt} has_answer_rate={ha_rate:.6f}"
            )

            prefix = f"src/{src}/"
            swanlab_payload.update(
                {
                    f"{prefix}seen": total_s,
                    f"{prefix}ok": ok_s,
                    f"{prefix}err": err_s,
                    f"{prefix}reward_cnt": cnt_s,
                    f"{prefix}reward_sum": sum_s,
                    f"{prefix}mean_reward": mean_s,
                    f"{prefix}has_answer_cnt": ha_cnt,
                    f"{prefix}has_answer_rate": ha_rate,
                }
            )

        if swanlab_enabled:
            try:
                swanlab.log(swanlab_payload)
                swanlab.finish()
                print(f"[swanlab] Successfully uploaded metrics for {len(sorted_sources)} sources")
            except Exception as e:
                print(f"[swanlab] Failed to upload metrics: {e}")

        print(
            f"[overall] total_samples={n_total} success_rate={n_ok/max(1,n_total):.4f} "
            f"mean_reward={mean_reward:.4f} "
            f"has_answer_rate={has_answer_rate:.4f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
