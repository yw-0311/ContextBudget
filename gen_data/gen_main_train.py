#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import datasets

## python gen_main.py --sample_frac 0.1 --seed 42
## python gen_main_train.py --sample_n 256 --seed 42
## python gen_main_train.py
## cp -r /home/wy517954/code/Elistic-Context-Fold-Verl/gen_data/processed_data_train/nq_hotpotqa_train_multi_1 /data/oss_bucket_0/shiyi/data/train_data/

# ========= Prompt Templates (两种版本) =========

SEARCH_R1_SYSTEM = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant."""

SEARCH_R1_USER_TMPL = (
    "You will answer multiple complex questions"
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    '<tool_call>{{"name": "search", "arguments": {{"query_list": ["query"]}}}}</tool_call> '
    "and it will return the top searched results between <tool_response> and </tool_response>. "
    "You can search as many times as your want. "
    "If all questions are answered, you can directly provide the answer inside <answer> and </answer>,without detailed illustrations."
    "For example, <answer> answer1; answer2; ... </answer>."
    "Question: {query} "
)

# ELASTIC_SYSTEM = """
# You are a research agent for long-running, multi-step investigations across sessions.
# Your goal is to gather evidence, verify claims, and progressively reduce uncertainty while preserving continuity.
# Research behavior:
# - Treat each turn as part of an ongoing research process, not a final answer.
# - Actively identify unknowns and resolve them through search and verification.
# - Use search frequently when facts, comparisons, or examples are needed.
# - Narrow queries step by step rather than issuing broad searches.
# - Cross-check important claims using multiple sources.
# - Keep reasoning concise and evidence-driven.
# Memory and context:
# - All tool observations are stored as <context_commit ...> blocks and act as durable working memory.
# - Do not repeat raw tool output in free text when it already exists in a context_commit.
# Folding (context management):
# - Folding is used to free context space so research can continue.
# - When prompted, call the summarize tool once to choose how to fold memory:
#   - NONE: keep all commits.
#   - PARTIAL: fold older or low-value commits.
#   - ALL: fold all commits into a compact continuation state.
# - Prefer folding when more searches or heavy analysis are expected.
# When folding (PARTIAL or ALL), merged_commit should be a compact compressed version of the folded commits, preserving as much problem-solving–relevant information from the original commit blocks as possible.
# Correctness and continuity:
# - Never allow context overflow.
# - Over-folding may lose detail; under-folding may block further research.
# - Use the lightest folding that provides enough space for continued searching and analysis.
# """

# ELASTIC_SYSTEM = """You are a research agent for long-running investigations.
# === CORE PRINCIPLES ===
# 1. VERIFY FIRST
#    - Cross-check claims with 2+ sources before accepting
#    - Check dates/numbers specifically. Note conflicts explicitly.
#    - Tag confidence: High/Medium/Low
# 2. COMMIT SYSTEM (Working Memory)
#    Format: `&lt;context_commit id="c0001"&gt;finding|source|confidence&lt;/context_commit&gt;`
#    SAVE: Store only critical facts (with verification status), active hypotheses, unsolved questions, source reliability, conflicts.
#    RETRIEVE: Before any action, scan existing `&lt;context_commit&gt;` blocks, synthesize what you know, only seek missing info.
# 3. RESEARCH LOOP
#    ASSESS (review commits→identify gaps) → SEARCH (targeted) → VERIFY (cross-check) → COMMIT (store) → REPEAT
# === OUTPUT RULES ===
# - Cite commit IDs: "Per c0001..."
# - State confidence levels
# - Use `&lt;answer&gt;` for final responses
# """


# ELASTIC_USER_TMPL = (
#     "You will answer multiple complex questions"
#     "You must conduct reasoning inside <think> and </think> first every time you get new information. "
#     "After reasoning, if you find you lack some knowledge, you can call a search engine by "
#     '<tool_call>{{"name": "search", "arguments": {{"query_list": ["query"]}}}}</tool_call> '
#     'Before obtaining the information, you need to determine the compression behavior.'
#     "and it will return the top searched results between <tool_response> and </tool_response>."
#     "You can search as many times as your want. "
#     "If all questions are answered, you can directly provide the answer inside <answer> and </answer>,without detailed illustrations."
#     "For example, <answer> answer1; answer2; ... </answer>."
#     "Question: {query} "
# )

ELASTIC_SYSTEM = """You are a research agent for long-running investigations.
=== CORE PRINCIPLES ===
1. VERIFY FIRST
   - Cross-check claims with 2+ sources before accepting
   - Check dates/numbers specifically. Note conflicts explicitly.
   - Tag confidence: High/Medium/Low
2. COMMIT SYSTEM (Working Memory)
   Format: `&lt;context_commit id="c0001"&gt;finding|source|confidence&lt;/context_commit&gt;`
   SAVE: Store only critical facts (with verification status), active hypotheses, unsolved questions, source reliability, conflicts.
   RETRIEVE: Before any action, scan existing `&lt;context_commit&gt;` blocks, synthesize what you know, only seek missing info.
3. RESEARCH LOOP
   ASSESS (review commits→identify gaps) → SEARCH (targeted) → VERIFY (cross-check) → COMMIT (store) → REPEAT
=== OUTPUT RULES ===
- Cite commit IDs: "Per c0001..."
- State confidence levels
- Answer must follow question sequence strictly
- Use `<answer>` for final responses
"""


ELASTIC_USER_TMPL = (
    "You will answer multiple complex questions"
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "-If you find you lack some knowledge, you can call a search engine by "
    '<tool_call>{{"name": "search", "arguments": {{"query_list": ["query"]}}}}</tool_call>'
    'before obtaining the information, you need to determine the compression behavior'
    "and it will return the top searched results between <tool_response> and </tool_response>."
    "-If all questions are answered, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations."
    "IMPORTANT:" 
    "-You can only make ONE tool call per turn. Do not search multiple queries or questions simultaneously."
    "-Any misalignment between answer position and question order will be considered incorrect."
    "Question: {query} "
)

# ========= 公共逻辑 =========

def get_query(example: dict) -> str:
    # 兼容 extra_info 可能不存在 / 不是 dict 的情况
    extra = example.get("extra_info", None)
    if isinstance(extra, dict):
        q = extra.get("question", "") or ""
    else:
        q = ""
    return q or example.get("question", "") or ""


def make_prompt(system_content: str, user_template: str, query: str):
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_template.format(query=query)},
    ]


def update_prompt_factory(system_content: str, user_template: str):
    def _fn(example):
        q = get_query(example)
        return {"prompt": make_prompt(system_content, user_template, q)}
    return _fn


def update_agent_factory(agent_name: str):
    def _fn(_example):
        return {"agent_name": agent_name}
    return _fn


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sample_dataset(ds: datasets.Dataset, seed: int, sample_frac: float = None, sample_n: int = None):
    n_total = len(ds)
    if sample_n is not None:
        n = min(sample_n, n_total)
        if n <= 0:
            return ds.select([])
        print(f"[sample] total={n_total}, sample_n={n}, seed={seed}")
        return ds.shuffle(seed=seed).select(range(n))

    if sample_frac is not None and sample_frac < 1.0:
        n = int(n_total * sample_frac)
        n = max(1, n) if n_total > 0 else 0
        print(f"[sample] total={n_total}, sample_frac={sample_frac}, sample_n={n}, seed={seed}")
        return ds.shuffle(seed=seed).select(range(n))

    print(f"[sample] total={n_total}, no sampling")
    return ds


def process_variant_from_base(base_parquet_path: str, out_parquet_path: str,
                              system_content: str, user_template: str, agent_name: str,
                              num_proc: int = None):
    print(f"[variant] base={base_parquet_path}")
    ds = datasets.Dataset.from_parquet(base_parquet_path)

    ds = ds.map(update_prompt_factory(system_content, user_template), num_proc=num_proc)
    ds = ds.map(update_agent_factory(agent_name), num_proc=num_proc)

    ensure_dir(os.path.dirname(out_parquet_path))
    ds.to_parquet(out_parquet_path)
    print(f"[save] {out_parquet_path} (rows={len(ds)})")


def process_one_dir(dir_path: str, seed: int, sample_frac: float, sample_n: int,
                    out_root: str, num_proc: int = None):
    """
    对单个 nq_hotpotqa_train_multi_* 目录：
    1) 读取 test.parquet
    2) 确定性采样并保存 sampled_base/test.parquet
    3) 基于 sampled_base 生成两种版本：
       - search_r1_processed/test.parquet
       - elastic_processed/test.parquet
    """
    in_test = os.path.join(dir_path, "train.parquet")
    if not os.path.exists(in_test):
        print(f"[skip] not found: {in_test}")
        return

    dir_name = os.path.basename(dir_path.rstrip("/"))
    print(f"\n=== Processing {dir_name} ===")
    print(f"[input] {in_test}")

    ds = datasets.Dataset.from_parquet(in_test)
    print(f"[load] rows={len(ds)}")

    # 先采样，得到“共同基数据”
    ds_base = sample_dataset(ds, seed=seed, sample_frac=sample_frac, sample_n=sample_n)

    # 保存采样后的基数据（不改字段）
    base_out = os.path.join(out_root, dir_name, "sampled_base", "train.parquet")
    ensure_dir(os.path.dirname(base_out))
    ds_base.to_parquet(base_out)
    print(f"[save] sampled base -> {base_out} (rows={len(ds_base)})")

    # 用同一份 base 生成两个版本
    search_out = os.path.join(out_root, dir_name, "search_r1_processed", "train.parquet")
    elastic_out = os.path.join(out_root, dir_name, "elastic_processed", "train.parquet")

    process_variant_from_base(
        base_parquet_path=base_out,
        out_parquet_path=search_out,
        system_content=SEARCH_R1_SYSTEM,
        user_template=SEARCH_R1_USER_TMPL,
        agent_name="tool_agent",
        num_proc=num_proc,
    )

    process_variant_from_base(
        base_parquet_path=base_out,
        out_parquet_path=elastic_out,
        system_content=ELASTIC_SYSTEM,
        user_template=ELASTIC_USER_TMPL,
        agent_name="my_tool_agent",
        num_proc=num_proc,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/wy517954/code/MEM1/Mem1/train/data_all_raw_train",
        help="包含 nq_hotpotqa_train_multi_* 的目录",
    )
    parser.add_argument(
        "--dirs",
        type=str,
        nargs="*",
        default=[
            "nq_hotpotqa_train_multi_1",
            # "nq_hotpotqa_train_multi_2",
            # "nq_hotpotqa_train_multi_3",
            # "nq_hotpotqa_train_multi_4",
            # "nq_hotpotqa_train_multi_5",
            # "nq_hotpotqa_train_multi_6",
            # "nq_hotpotqa_train_multi_7",
            # "nq_hotpotqa_train_multi_8",
            # "nq_hotpotqa_train_multi_9",
            # "nq_hotpotqa_train_multi_16",
        ],
        help="要处理的子目录名列表",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=os.path.join(os.getcwd(), "processed_data_train"),
        help="输出根目录（默认：脚本当前目录/processed_data",
    )
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子（决定性采样）")
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="按比例采样（<1.0 才生效）。例如 0.1 表示采样 10%",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=None,
        help="按固定条数采样（优先级高于 sample_frac）",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="datasets.map 并行进程数（可选）",
    )
    args = parser.parse_args()

    ensure_dir(args.out_root)
    print(f"[config] data_root={args.data_root}")
    print(f"[config] out_root={args.out_root}")
    print(f"[config] seed={args.seed}, sample_frac={args.sample_frac}, sample_n={args.sample_n}")

    for d in args.dirs:
        dir_path = os.path.join(args.data_root, d)
        if not os.path.isdir(dir_path):
            print(f"[skip] not a dir: {dir_path}")
            continue
        process_one_dir(
            dir_path=dir_path,
            seed=args.seed,
            sample_frac=args.sample_frac,
            sample_n=args.sample_n,
            out_root=args.out_root,
            num_proc=args.num_proc,
        )


if __name__ == "__main__":
    main()
