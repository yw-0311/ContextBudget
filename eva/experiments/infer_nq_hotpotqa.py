# infer_concurrent_multi_f1.py
from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer

from sglang_server_manager import SGLangServerManager
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop as baseline_loop
from verl.experimental.agent_loop.my_tool_agent_loop_flat_commit import ToolAgentLoop as branch_loop
from verl.experimental.agent_loop.tool_agent_loop_budget import ToolAgentLoop as baseline_loop_wbudget
from verl.experimental.agent_loop.mem1_agent_loop import Mem1AgentLoop
from verl.experimental.agent_loop.tool_agent_loop_eager import ToolAgentLoopEager as eager_loop
from verl.experimental.agent_loop.summary_agent_loop import ToolAgentLoop as summary_loop
from submit.experiments.search_r1.ferret.reward_score.search_r1_format import (
    compute_score_multi_answer,
    extract_solution
)

import re
import string

import swanlab


# =============================================================================
# Config shim
# =============================================================================
class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def build_verl_like_config(
    *,
    model_path: str,
    tool_config_path: str,
    tool_format: str,
    max_model_len: int,
    enable_budget: bool,
    max_assistant_turns: int,
    max_parallel_calls: int,
    max_depth: int,
    max_tool_response_length: int,
    tool_response_truncate_side: str,
    apply_chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    multi_turn = Obj(
        format=tool_format,
        enable=True,
        max_assistant_turns=max_assistant_turns,
        max_tool_response_length=max_tool_response_length,
        max_user_turns=None,
        max_depth=max_depth,
        max_parallel_calls=max_parallel_calls,
        tool_response_truncate_side=tool_response_truncate_side,
        tool_config_path=tool_config_path,
        enable_budget=enable_budget,
        tokenization_sanity_check_mode="disable",
    )
    rollout = Obj(
        n=1,
        name="sglang",
        mode="async",
        max_model_len=max_model_len,
        multi_turn=multi_turn,
        agent=Obj(num_workers=1, default_agent_loop="tool_agent", agent_loop_config_path=None),
        calculate_log_probs=False,
        log_prob_micro_batch_size_per_gpu=128,
        tensor_model_parallel_size=1,
        gpu_memory_utilization=0.3,
    )
    actor_rollout_ref = Obj(
        model=Obj(
            path=model_path,
            use_remove_padding=True,
            enable_gradient_checkpointing=True,
            custom_chat_template=None,
        ),
        rollout=rollout,
    )
    cfg = Obj(
        actor_rollout_ref=actor_rollout_ref,
        reward_model=Obj(enable=False, use_reward_loop=False),
        trainer=Obj(project_name="infer", experiment_name="infer", n_gpus_per_node=1, nnodes=1),
        data={"apply_chat_template_kwargs": (apply_chat_template_kwargs or {})},
        custom_reward_function=Obj(path=None, name=None),
    )
    return cfg


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
    agent_loop: "ToolAgentLoop",
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
    output_msgs = [ o.extra_fields.get("final_messages", []) for o in outs ]
    pred = extract_last_assistant(final_msgs)
    return {
        "prediction": pred,
        "num_turns": out.num_turns,
        "last_branch_depth": out.branch_depth,
        "prompt_ids_len": len(out.prompt_ids),
        "response_ids_len": len(out.response_ids),
        "final_messages": output_msgs,
    }

# =============================================================================
# Reward (NO LLM JUDGE)
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
    使用统一格式的多答案评分函数。
    """
    # 直接调用新函数，它返回的就是标准格式的 reward_extra
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
    agent_loop: "ToolAgentLoop",
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
    ap.add_argument("--dataset", type=str, default="mh")

    # IO
    ap.add_argument("--parquet", type=str, required=True)
    ap.add_argument("--prompt_col", type=str, default="prompt")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--out_jsonl", type=str, required=True)

    # sglang + model
    ap.add_argument("--sglang_url", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--tool_config_path", type=str, required=True)
    ap.add_argument("--tool_format", type=str, required=True)

    # lengths + multi-turn
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--enable_budget", action="store_true", help="Enable context budget management")
    ap.add_argument("--max_assistant_turns", type=int, default=10)
    ap.add_argument("--max_parallel_calls", type=int, default=1)
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--max_tool_response_length", type=int, default=8192)
    ap.add_argument("--tool_response_truncate_side", type=str, default="middle")

    # sampling
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--timeout_s", type=float, default=1000.0)

    # runtime
    ap.add_argument("--flush_every", type=int, default=50)
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--agent_type", type=str, required=True)

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
        f"{args.agent_type}"
        f"_{model_short}"
        f"_len{args.max_model_len}"
        f"_temp{args.temperature}"
        f"_budget{int(args.enable_budget)}"
        f"_{args.val_type}"
        f"_same{int(args.require_same_len)}"
        f"_target{args.target}"  # Add target to the experiment name
    )

    swanlab_enabled = False
    try:
        swanlab.init(project=project_name, experiment_name=experiment_name)
        swanlab.config.update(
            {
                "agent_type": args.agent_type,
                "model_path": args.model_path,
                "model_short": model_short,
                "max_model_len": args.max_model_len,
                "val_type": args.val_type,
                "require_same_len": int(args.require_same_len),
                "max_depth": args.max_depth,
                "target": args.target,  # Add target to the config
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

    cfg = build_verl_like_config(
        model_path=args.model_path,
        tool_config_path=args.tool_config_path,
        tool_format=args.tool_format,
        max_model_len=args.max_model_len,
        enable_budget=args.enable_budget,
        max_assistant_turns=args.max_assistant_turns,
        max_parallel_calls=args.max_parallel_calls,
        max_depth=args.max_depth,
        max_tool_response_length=args.max_tool_response_length,
        tool_response_truncate_side=args.tool_response_truncate_side,
        apply_chat_template_kwargs={"enable_thinking": False},
    )
    trainer_config = type("DummyTrainerConfig", (), {"config": cfg})()

    server_manager = SGLangServerManager(args.sglang_url, timeout_s=args.timeout_s)

    if args.agent_type == "baseline_loop":
        agent_loop = baseline_loop(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=None,
        )
    elif args.agent_type in ("branch_loop", "branch_loop_wob"):
        agent_loop = branch_loop(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=None,
        )
    elif args.agent_type in ("baseline_loop_wbudget"):
        agent_loop = baseline_loop_wbudget(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=None,
        )
    elif args.agent_type in ("mem1_agent", "mem1_agent_normal", "mem1_agent_amem", "mem1_agent_mem1"):
        # Determine inference_type from agent_type
        inference_type = "mem1"
        if args.agent_type == "mem1_agent_normal":
            inference_type = "normal"
        elif args.agent_type == "mem1_agent_amem":
            inference_type = "amem"
        elif args.agent_type == "mem1_agent_mem1":
            inference_type = "mem1"
        
        agent_loop = Mem1AgentLoop(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=None,
            inference_type=inference_type,
            max_iteration=6,
        )
    elif args.agent_type == "summary_loop":
        agent_loop = summary_loop(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=None,
        )
    elif args.agent_type == "agentfold_search":
        agent_loop = AgentFoldSearchLoop(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=None,
        )
    elif args.agent_type == "eager_loop":
        agent_loop = eager_loop(
            trainer_config=trainer_config,
            server_manager=server_manager,
            tokenizer=tokenizer,
            processor=None,
        )
    else:
        raise ValueError(f"Unknown agent_type={args.agent_type}")

    agent_loop.loop = asyncio.get_running_loop()

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_model_len,
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
    depth_sum = 0.0
    depth_cnt = 0
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
                "depth_sum": 0.0,
                "depth_cnt": 0,
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
        mean_depth = (depth_sum / depth_cnt) if depth_cnt else 0.0

        if pbar is not None:
            pbar.set_postfix(
                ok=n_ok,
                err=n_err,
                avg_s=f"{avg_s:.3f}",
                mean_reward=f"{mean_reward:.4f}",
                mean_depth=f"{mean_depth:.4f}",
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

                        if last_depth is not None:
                            depth_sum += float(last_depth)
                            depth_cnt += 1
                            st["depth_sum"] += float(last_depth)
                            st["depth_cnt"] += 1
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
        mean_depth = (depth_sum / depth_cnt) if depth_cnt else 0.0

        print(
            f"[done] wrote {n_total} rows to {args.out_jsonl}, ok={n_ok} err={n_err}, "
            f"avg={dt/max(1,n_total):.3f}s/item mean_reward={mean_reward:.4f} "
            f"mean_depth={mean_depth:.4f} has_answer_rate={has_answer_rate:.4f}"
        )

        def total_seen(v: Dict[str, Any]) -> int:
            return int(v.get("ok", 0)) + int(v.get("err", 0))

        swanlab_payload: Dict[str, Any] = {}

        # Update swanlab_payload with the weighted mean reward
        swanlab_payload["has_answer_rate"] = has_answer_rate
        swanlab_payload["mean_reward"] = mean_reward
        swanlab_payload["mean_depth"] = mean_depth
        swanlab_payload["mean_reward_target"] = mean_reward/args.target

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

            d_cnt = int(st.get("depth_cnt", 0))
            d_sum = float(st.get("depth_sum", 0.0))
            d_mean = (d_sum / d_cnt) if d_cnt else 0.0

            print(
                f"  - {src}: seen={total_s} ok={ok_s} err={err_s} "
                f"reward_cnt={cnt_s} reward_sum={sum_s:.6f} mean_reward={mean_s:.6f} "
                f"depth_cnt={d_cnt} depth_sum={d_sum:.6f} mean_depth={d_mean:.6f} "
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
                    f"{prefix}depth_cnt": d_cnt,
                    f"{prefix}depth_sum": d_sum,
                    f"{prefix}mean_depth": d_mean,
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
            f"mean_reward={mean_reward:.4f} mean_depth={mean_depth:.4f} "
            f"has_answer_rate={has_answer_rate:.4f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
