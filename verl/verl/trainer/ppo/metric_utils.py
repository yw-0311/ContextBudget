# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated

from typing import Any, Dict, List


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["response_mask"]
    # response_mask = batch.batch["response_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    
    # DEBUG: 检查 response_length 的计算细节
    print(f"[DEBUG _compute_response_info] response_shape: {batch.batch['responses'].shape}")
    print(f"[DEBUG _compute_response_info] attention_mask_shape: {batch.batch['attention_mask'].shape}")
    print(f"[DEBUG _compute_response_info] response_length_tensor: {response_length}")
    print(f"[DEBUG _compute_response_info] zero_response_count: {(response_length == 0).sum().item()}")
    
    if (response_length == 0).any():
        zero_indices = torch.where(response_length == 0)[0]
        print(f"[DEBUG _compute_response_info] zero_response_indices: {zero_indices.tolist()}")
        for idx in zero_indices[:3]:  # 显示前3个zero样本的详细信息
            print(f"[DEBUG _compute_response_info] Sample {idx}:")
            print(f"  - response_mask_sum: {response_mask[idx].sum().item()}")
            print(f"  - response_mask: {response_mask[idx]}")
            print(f"  - attention_mask: {batch.batch['attention_mask'][idx]}")
            if 'responses' in batch.batch:
                print(f"  - responses: {batch.batch['responses'][idx]}")

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def _extract_custom_metrics(batch: DataProto) -> dict[str, Any]:
    """
    Automatically extract custom metrics from batch.non_tensor_batch.

    This function detects any numeric metrics added by the reward function
    and computes statistics for them.

    Args:
        batch: A DataProto object containing batch data.

    Returns:
        A dictionary of custom metrics with statistics.
    """
    custom_metrics = {}

    if not hasattr(batch, 'non_tensor_batch'):
        return custom_metrics

    # Known standard keys to skip (these are not metrics)
    skip_keys = {
        'data_source', 'reward_model', 'extra_info', 'uid',
        'request_id', '__num_turns__', 'tool_call_counts'
    }

    for key in batch.non_tensor_batch.keys():
        # Skip known non-metric keys
        if key in skip_keys:
            continue

        values = batch.non_tensor_batch[key]

        if key == "branch_depth" and isinstance(values, np.ndarray) and values.dtype == object:
            flat = values.reshape(-1)
            cleaned = []
            for x in flat:
                try:
                    if x is None:
                        cleaned.append(float("nan"))
                    else:
                        cleaned.append(float(x))
                except Exception:
                    cleaned.append(float("nan"))
            values = torch.tensor(cleaned, dtype=torch.float32)
    
        # Try to convert to numeric tensor
        try:
            if isinstance(values, (list, np.ndarray)):
                values = torch.tensor(values, dtype=torch.float32)
            elif isinstance(values, torch.Tensor):
                values = values.float()
            else:
                # Skip non-numeric values
                continue

            # Check if it's numeric data
            if values.dtype not in [torch.float16, torch.float32, torch.float64,
                                   torch.int8, torch.int16, torch.int32, torch.int64]:
                continue

            # Compute statistics
            if values.numel() > 0:
                custom_metrics[f"custom/{key}"] = values.mean().item()

            print(f"[DEBUG] {key} OK")

        except (ValueError, TypeError, RuntimeError):
            # Skip values that can't be converted to numeric tensors
            continue

    return custom_metrics


def select_max_depth_indices(ids: List[Any], depths: List[float]) -> List[int]:
    """
    给定 ids 和 depths（等长），返回每个 id 对应 depth 最大的那条样本 index。
    """
    if len(ids) != len(depths):
        raise ValueError("ids 和 depths 长度不一致")

    best_idx = {}
    best_depth = {}

    for i, (bid, d) in enumerate(zip(ids, depths)):
        if bid not in best_idx or d > best_depth[bid]:
            best_idx[bid] = i
            best_depth[bid] = d

    # 保持原 batch 的顺序（只过滤不重排）
    keep_indices = sorted(best_idx.values())
    return keep_indices


def build_pruned_dataproto_by_branch_max_depth(
    dp,  # DataProto
    branch_id_key: str = "branch_id",
    branch_depth_key: str = "branch_depth",
):
    """
    按 dp.non_tensor_batch[branch_id_key] 分组，
    每组保留 dp.non_tensor_batch[branch_depth_key] 最大的那条样本（并列取最先出现），
    然后用 dp.select_idxs(...) 同步裁剪 batch 和 non_tensor_batch。
    """

    ids = dp.non_tensor_batch[branch_id_key]
    depths = dp.non_tensor_batch[branch_depth_key]

    if isinstance(ids, np.ndarray) and ids.ndim == 2 and ids.shape[1] == 1:
        ids = ids.squeeze(1)
    if isinstance(depths, np.ndarray) and depths.ndim == 2 and depths.shape[1] == 1:
        depths = depths.squeeze(1)

    ids_list = ids.tolist()
    depths_list = [float(x) for x in depths.tolist()]

    keep_indices = select_max_depth_indices(ids_list, depths_list)
    return dp.select_idxs(keep_indices)

# ---------------------------
# 用法示例：
# new_batch = build_pruned_batch_by_branch_max_depth(batch)
# print("old N =", batch["branch_id"].shape[0], "new N =", new_batch["branch_id"].shape[0])


def _describe_value(v):
    """只描述维度/长度/类型，不打印具体内容"""
    if isinstance(v, torch.Tensor):
        return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device})"
    if isinstance(v, np.ndarray):
        return f"ndarray(shape={v.shape}, dtype={v.dtype})"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}(len={len(v)})"
    if isinstance(v, str):
        return f"str(len={len(v)})"
    if isinstance(v, (int, float, bool)):
        return f"{type(v).__name__}({v})"
    # 其他类型简单打印类型名
    return f"{type(v).__name__}"

import pprint
def show_dataproto(dp: DataProto, title: str = "DataProto"):
    """统一的 DataProto 结构打印：字段名 + 维度/类型"""
    print(f"\n========== [{title}] ==========")

    # Tensor 部分
    print("\n--- batch (tensor fields) ---")
    for k, v in dp.batch.items():
        print(f"{k:25s}: {_describe_value(v)}")

    # non_tensor 部分
    print("\n--- non_tensor_batch (meta / ids / extra info) ---")
    for k, v in dp.non_tensor_batch.items():
        print(f"{k:25s}: {_describe_value(v)}")

    # meta_info
    print("\n--- meta_info ---")
    pprint.pprint(dp.meta_info, indent=2)
    print("=====================================")


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    ENHANCED: This version also automatically detects and includes custom metrics from
    batch.non_tensor_batch that were added by the reward function.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
            - custom/{key}/mean, max, min, rate, std: Custom metrics from reward function
    """

    # if True:
    #     _dump_args(
    #         "batch",
    #         {
    #             "batch": batch
    #         },
    #     )
    #     print("[PVM] Captured args to .debug_cache/batch.pkl")
    # ======================================

    if False:
        cached = _load_args("batch")
        batch = cached["batch"]
        print("[PVM] Replayed args from .debug_cache/batch.pkl")

    # print(f"[DEBUG] {batch}")
    # show_dataproto(batch, "batch (original)")
    batch = build_pruned_dataproto_by_branch_max_depth(batch)
        
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    score_mean = torch.mean(non_aborted_sequence_score).detach().item()
    score_max = torch.max(non_aborted_sequence_score).detach().item()
    score_min = torch.min(non_aborted_sequence_score).detach().item()

    reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()
    reward_max = torch.max(non_aborted_sequence_reward).detach().item()
    reward_min = torch.min(non_aborted_sequence_reward).detach().item()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float()).detach().item()
        )
    else:
        raise ValueError("All samples are aborted, this should not happen.")

    metrics = {
        # score
        "critic/score/mean": score_mean,
        "critic/score/max": score_max,
        "critic/score/min": score_min,
        # reward
        "critic/rewards/mean": reward_mean,
        "critic/rewards/max": reward_max,
        "critic/rewards/min": reward_min,
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # branch_depth = batch.non_tensor_batch["branch_depth"]
    # branch_depth_np = np.asarray(branch_depth, dtype=np.int32)
    # metrics["branch_depth/mean"] = float(branch_depth_np.mean())
    # metrics["branch_depth/max"] = float(branch_depth_np.max())
    # metrics["branch_depth/min"] = float(branch_depth_np.min())

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    # ===== AUTOMATICALLY EXTRACT AND ADD CUSTOM METRICS =====
    custom_metrics = _extract_custom_metrics(batch)
    metrics.update(custom_metrics)
    # =========================================================

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], n_gpus: int) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


from verl.trainer.ppo.debug_metric_utils import _load_args,_dump_args

CAPTURE_PVM = True
REPLAY_PVM = False

from typing import Any
from collections import defaultdict

from collections import defaultdict, Counter

def debug_duplicate_uids(
    data_sources: list[str],
    sample_uids: list[str],
    infos_dict: dict[str, list[Any]],
    *,
    branch_key: str = "branch_id",
    depth_key: str = "branch_depth",
    uid_key: str = "uid",
    max_uids: int = 5,
    max_rows_per_uid: int = 20,
):
    """
    打印导致 uid 重复的根因：
    - 同一个 uid 是否出现多个 branch_id
    - 每个 uid 对应哪些行（row index）、depth、branch_id、data_source
    - sample_uids vs infos_dict['uid'] 是否对齐
    """
    n = len(infos_dict.get(uid_key, []))
    if n == 0:
        print("[DEBUG] infos_dict has no uid or empty")
        return

    # 0) 基本长度一致性检查
    problems = []
    if len(sample_uids) != n:
        problems.append(f"len(sample_uids)={len(sample_uids)} != len(infos_dict['{uid_key}'])={n}")
    if len(data_sources) != n:
        problems.append(f"len(data_sources)={len(data_sources)} != len(infos_dict['{uid_key}'])={n}")
    for k, v in infos_dict.items():
        if len(v) != n:
            problems.append(f"len(infos_dict['{k}'])={len(v)} != {n}")
    if problems:
        print("[DEBUG][LENGTH MISMATCH]")
        for p in problems:
            print("  -", p)
        # 长度不一致时，后续 zip/索引一定会错位，优先修这里
        return

    # 1) 对齐检查：sample_uids[i] 是否等于 infos_dict['uid'][i]
    mismatch_cnt = 0
    for i in range(n):
        if sample_uids[i] != infos_dict[uid_key][i]:
            mismatch_cnt += 1
            if mismatch_cnt <= 5:
                print(f"[DEBUG][UID MISMATCH] row={i} sample_uids={repr(sample_uids[i])} infos_uid={repr(infos_dict[uid_key][i])}")
    print(f"[DEBUG] uid mismatch count between sample_uids and infos_dict['{uid_key}']: {mismatch_cnt}/{n}")

    # 2) 统计重复 uid（按 infos_dict['uid']，因为你后面重建 sample_uids 用的是它）
    uids = infos_dict[uid_key]
    uid_cnt = Counter(uids)
    dup_uids = [u for u, c in uid_cnt.items() if c > 1]
    print(f"[DEBUG] #duplicate uids in infos_dict['{uid_key}'] = {len(dup_uids)} / unique={len(uid_cnt)} total={n}")

    if not dup_uids:
        return

    # 3) 建 uid -> rows / branches
    uid2rows = defaultdict(list)
    uid2branches = defaultdict(set)
    for i, u in enumerate(uids):
        uid2rows[u].append(i)
        uid2branches[u].add(infos_dict[branch_key][i])

    # 4) 打印前 max_uids 个“重复 uid”的明细
    for u in dup_uids[:max_uids]:
        rows = uid2rows[u]
        branches = uid2branches[u]
        print("\n" + "-" * 80)
        print(f"[DEBUG][DUP UID] uid={repr(u)} count={len(rows)} branches={list(branches)}")

        # 打印该 uid 的若干行
        for i in rows[:max_rows_per_uid]:
            b = infos_dict[branch_key][i]
     

def _filter_keep_max_depth_per_branch(
    data_sources: list[str],
    sample_uids: list[str],
    infos_dict: dict[str, list[Any]],
    *,
    branch_key: str = "branch_id",
    depth_key: str = "branch_depth",
    uid_key_in_infos: str = "uid",
) -> tuple[list[str], list[str], dict[str, list[Any]]]:
    """
    Filter samples so that for each branch_id, only the record with the maximum depth is kept.
    Also realign sample_uids using infos_dict[uid_key_in_infos].
    """

    if branch_key not in infos_dict:
        raise KeyError(f"infos_dict missing key: {branch_key}")
    if depth_key not in infos_dict:
        raise KeyError(f"infos_dict missing key: {depth_key}")
    if uid_key_in_infos not in infos_dict:
        raise KeyError(f"infos_dict missing key: {uid_key_in_infos}")

    n = len(infos_dict[branch_key])
    # 基本一致性检查：infos_dict 里的每个字段长度应一致
    for k, v in infos_dict.items():
        if len(v) != n:
            raise ValueError(f"infos_dict[{k}] length {len(v)} != {n}")

    # data_sources 通常也是按 sample 对齐的（你原代码是这么用的）
    if len(data_sources) != n:
        raise ValueError(f"len(data_sources) {len(data_sources)} != len(infos_dict[{branch_key}]) {n}")

    # 1) 找到每个 branch_id 对应 depth 最大的样本 idx
    best_idx_by_branch: dict[Any, int] = {}
    best_depth_by_branch: dict[Any, Any] = {}

    for i, (b, d) in enumerate(zip(infos_dict[branch_key], infos_dict[depth_key], strict=True)):
        # depth 若可能是字符串，建议转一下；按你实际数据决定
        # d_val = float(d) if isinstance(d, str) else d
        d_val = d
        if b not in best_idx_by_branch or d_val > best_depth_by_branch[b]:
            best_idx_by_branch[b] = i
            best_depth_by_branch[b] = d_val

    kept_indices = sorted(best_idx_by_branch.values())

    # 2) 同步裁剪 data_sources / infos_dict 的所有字段
    data_sources_f = [data_sources[i] for i in kept_indices]
    infos_dict_f = {k: [v[i] for i in kept_indices] for k, v in infos_dict.items()}

    # 3) 用 infos_dict 里的 uid 重建 sample_uids（确保对齐）
    sample_uids_f = [infos_dict_f[uid_key_in_infos][i] for i in range(len(kept_indices))]

    return data_sources_f, sample_uids_f, infos_dict_f


def process_validation_metrics(
    data_sources: list[str], sample_uids: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    if False:
        _dump_args(
            "process_validation_metrics_args",
            {
                "data_sources": data_sources,
                "sample_uids": sample_uids,
                "infos_dict": infos_dict,
                "seed": seed,
            },
        )
        print("[PVM] Captured args to .debug_cache/process_validation_metrics_args.pkl")
    # ======================================

    if False:
        cached = _load_args("process_validation_metrics_args")
        data_sources = cached["data_sources"]
        sample_uids   = cached["sample_uids"]
        infos_dict    = cached["infos_dict"]
        seed          = cached["seed"]
        print("[PVM] Replayed args from .debug_cache/process_validation_metrics_args.pkl")


    # Group metrics by data source, prompt and variable
    print(f"len of data_sources {len(data_sources)}")
    print(f"len of sample_uids {len(sample_uids)}")
    # print(f"infos_dict {infos_dict}")
    
    # debug_duplicate_uids
    debug_duplicate_uids(data_sources, sample_uids, infos_dict)
    # process branch reward
    data_sources, sample_uids , infos_dict = _filter_keep_max_depth_per_branch(data_sources,sample_uids,infos_dict)

    print(f"len of data_sources after _filter_keep_max_depth_per_branch {len(data_sources)}")
    print(f"len of sample_uids after _filter_keep_max_depth_per_branch {sample_uids}")

    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2uid2var2metric[data_source][uid][var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)

    return data_src2var2metric2val

if __name__ == "__main__":
    process_validation_metrics()