#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import datasets

## python gen_main.py --sample_frac 0.1 --seed 42
## python gen_main.py --sample_n 256 --seed 42

## python gen_main_test.py --seed 42

# ========= Prompt Templates =========

SEARCH_R1_SYSTEM = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant."""

SEARCH_R1_USER_TMPL = (
    "You will answer multiple complex questions"
    "You must conduct reasoning inside <information> and </information> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by "
    '{{"name": "search", "arguments": {{"query_list": ["query"]}}}} '
    "and it will return the top searched results between <observe> and </observe>. "
    "You can search as many times as your want. "
    "If all questions are answered, you can directly provide the answer inside <answer> and </answer>,without detailed illustrations."
    "For example, <answer> answer1; answer2; ... </answer>."
    "Question: {query} "
)

# ========= Mem1 (FlashRAG-style) Template =========

def make_mem1_prefix(questions_str: str) -> str:
    """FlashRAG base template: <search>/<answer>/<information> format"""
    if questions_str and not questions_str.endswith('?'):
        questions_str += '?'
    
    return f"""You will answer multiple complex questions using iterative reasoning, summarization, and web search.
At each step, you will see the questions, a cumulative summary of relevant information, the current search query, and search results (except in the first step, where only the questions are provided). Your task is to:
1. Perform reasoning and update a cumulative, concise summary within <information> ... </information>. This acts as persistent memory and must include all essential information from previous <information> and <observe> tags.
2. Then choose one of the following actions:
   - If any question remains unanswered, issue a single query for one question inside <search> ... </search>. The query should consist of keywords or a short phrase. Only search one question at a time.
   - If all questions are answered, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations.
Important:
- Always follow this structure after <observe> or the initial questions: <information> ... </information><search> ... </search> or <information> ... </information><answer> ... </answer>.
- Do not search multiple queries or questions simultaneously.
Answer the following questions: {questions_str}\n"""


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
    "You must conduct reasoning inside <information> and </information> first every time you get new information. "
    "-If you find you lack some knowledge, you can call a search engine by "
    '{{"name": "search", "arguments": {{"query_list": ["query"]}}}}'
    'before obtaining the information, you need to determine the compression behavior'
    "and it will return the top searched results between <observe> and </observe>."
    "-If all questions are answered, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations."
    "IMPORTANT:" 
    "-You can only make ONE tool call per turn. Do not search multiple queries or questions simultaneously."
    "-Any misalignment between answer position and question order will be considered incorrect."
    "Question: {query} "
)

ELASTIC_SYSTEM_PRE = """You are a research agent for long-running investigations.
=== CORE PRINCIPLES ===
1. VERIFY FIRST
   - Cross-check claims with 2+ sources before accepting
   - Check dates/numbers specifically. Note conflicts explicitly.
   - Tag confidence: High/Medium/Low
2. COMMIT SYSTEM (Working Memory)
   Format: `&lt;context_commit id="c0001"&gt;finding|source|confidence&lt;/context_commit&gt;`
   SAVE: Store only critical facts (with verification status), active hypotheses, unsolved questions, source reliability, conflicts.
   RETRIEVE: Before any action, scan existing `&lt;context_commit&gt;` blocks, synthesize what you know, only seek missing info.
=== OUTPUT RULES ===
- Cite commit IDs: "Per c0001..."
- State confidence levels
- Answer must follow question sequence strictly
- Use `<answer>` for final responses
"""


ELASTIC_USER_TMPL_PRE = (
    "You will answer multiple complex questions"
    "You must conduct reasoning inside <information> and </information> first every time you get new information. "
    "-If you find you lack some knowledge, you can call a search engine by "
    '{{"name": "search", "arguments": {{"query_list": ["query"]}}}}'
    "and it will return the top searched results between <observe> and </observe>."
    "-If all questions are answered, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations."
    "IMPORTANT:" 
    "-You can only make ONE tool call per turn. Do not search multiple queries or questions simultaneously."
    "-Any misalignment between answer position and question order will be considered incorrect."
    "Question: {query} "
)

# ========= Common Utilities =========

def get_query(example: dict) -> str:
    # Handle case where extra_info may not exist or is not a dict
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

def process_mem1_variant(base_parquet_path: str, out_parquet_path: str, num_proc: int = None):
    """
    Process Mem1 (FlashRAG-style) format:
    - Only user prompt, no system prompt
    - Use <search>/<answer>/<information> tag format
    - agent_name is fixed to 'mem1'
    """
    print(f"[variant] base={base_parquet_path} (Mem1 format)")
    ds = datasets.Dataset.from_parquet(base_parquet_path)

    def _update_mem1(example):
        q = get_query(example)
        content = make_mem1_prefix(q)
        return {
            "prompt": [{"role": "user", "content": content}],
            "agent_name": "mem1"
        }

    ds = ds.map(_update_mem1, num_proc=num_proc)
    
    ensure_dir(os.path.dirname(out_parquet_path))
    ds.to_parquet(out_parquet_path)
    print(f"[save] {out_parquet_path} (rows={len(ds)}, agent=mem1)")

def process_one_dir(dir_path: str, seed: int, sample_frac: float, sample_n: int,
                    out_root: str, num_proc: int = None):
    """
    Process a single nq_hotpotqa_train_multi_* directory:
    1) Read test.parquet
    2) Deterministic sampling and save to sampled_base/test.parquet
    3) Generate two versions based on sampled_base:
       - search_r1_processed/test.parquet
       - elastic_processed/test.parquet
    """
    in_test = os.path.join(dir_path, "test.parquet")
    if not os.path.exists(in_test):
        print(f"[skip] not found: {in_test}")
        return

    dir_name = os.path.basename(dir_path.rstrip("/"))
    print(f"\n=== Processing {dir_name} ===")
    print(f"[input] {in_test}")

    ds = datasets.Dataset.from_parquet(in_test)
    print(f"[load] rows={len(ds)}")

    # First sample to get 'shared base data'
    ds_base = sample_dataset(ds, seed=seed, sample_frac=sample_frac, sample_n=sample_n)

    # Save sampled base data (fields unchanged)
    base_out = os.path.join(out_root, dir_name, "sampled_base", "test.parquet")
    ensure_dir(os.path.dirname(base_out))
    ds_base.to_parquet(base_out)
    print(f"[save] sampled base -> {base_out} (rows={len(ds_base)})")

    # Generate two versions using the same base
    search_out = os.path.join(out_root, dir_name, "search_r1_processed", "test.parquet")
    elastic_out = os.path.join(out_root, dir_name, "elastic_processed", "test.parquet")
    mem1_out = os.path.join(out_root, dir_name, "mem1_processed", "test.parquet")

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

    process_mem1_variant(
        base_parquet_path=base_out,
        out_parquet_path=mem1_out,
        num_proc=num_proc,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join(os.getcwd(), "data_all_raw"),
        help="Directory containing nq_hotpotqa_train_multi_*",
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
            # "nq_hotpotqa_train_multi_32",
        ],
        help="List of subdirectories to process",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=os.path.join(os.getcwd(), "processed_data"),
        help="Output root directory (default: current working directory/processed_data)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling random seed (deterministic sampling)")
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=1.0,
        help="Sample by fraction (<1.0 to take effect). E.g., 0.1 means sample 10%",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=None,
        help="Sample by fixed number (higher priority than sample_frac)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of parallel processes for datasets.map (optional)",
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