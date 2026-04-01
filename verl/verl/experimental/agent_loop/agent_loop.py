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

import asyncio
import heapq
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
import torch.nn.functional as F
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.reward import RewardManagerWorker
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# =========================
# Padding helpers
# =========================

def _pad_left_2d(x: torch.Tensor, target_len: int, pad_value: int):
    """
    x: [B, L] left-pad to target_len (truncate keeping rightmost if too long)
    """
    L = x.size(1)
    if L >= target_len:
        return x[:, -target_len:]
    return F.pad(x, (target_len - L, 0), value=pad_value)


def _pad_right_2d(x: torch.Tensor, target_len: int, pad_value):
    """
    x: [B, L] right-pad to target_len (truncate keeping leftmost if too long)
    """
    L = x.size(1)
    if L >= target_len:
        return x[:, :target_len]
    return F.pad(x, (0, target_len - L), value=pad_value)


def _pad_position_ids(position_ids: torch.Tensor, dp_left: int, dr_right: int, total_len: int):
    """
    position_ids:
      - text:   [B, T]
      - qwen2vl:[B, 3, T]
    We align prompt by left-pad dp_left and response by right-pad dr_right, then crop to total_len.
    """
    if position_ids.dim() == 2:
        position_ids = F.pad(position_ids, (dp_left, dr_right), value=0)
        return position_ids[:, :total_len]
    else:
        position_ids = F.pad(position_ids, (dp_left, dr_right), value=0)
        return position_ids[:, :, :total_len]


def _pad_tensordict_rollout(td: TensorDict, max_prompt: int, max_resp: int, pad_id: int):
    """
    Cross-worker padding for a worker output batch TensorDict.
    Important:
      - prompts: left-pad
      - responses: right-pad
      - response_mask/log_probs/rm_scores: right-pad
      - attention_mask: split into prompt/resp and pad separately, then concat
      - input_ids: rebuild from padded prompts+responses
      - position_ids: pad on both sides consistently
    """
    prompts = td["prompts"]      # [B, cur_p]
    responses = td["responses"]  # [B, cur_r]
    cur_p = prompts.size(1)
    cur_r = responses.size(1)

    dp_left = max(max_prompt - cur_p, 0)
    dr_right = max(max_resp - cur_r, 0)

    prompts_new = _pad_left_2d(prompts, max_prompt, pad_id)
    responses_new = _pad_right_2d(responses, max_resp, pad_id)

    if "response_mask" in td.keys():
        td["response_mask"] = _pad_right_2d(td["response_mask"], max_resp, 0)

    if "rollout_log_probs" in td.keys():
        td["rollout_log_probs"] = _pad_right_2d(td["rollout_log_probs"], max_resp, 0.0)

    if "rm_scores" in td.keys():
        td["rm_scores"] = _pad_right_2d(td["rm_scores"], max_resp, 0.0)

    # attention_mask: pad prompt and resp separately
    att = td["attention_mask"]  # [B, cur_p + cur_r]
    att_p = att[:, :cur_p]
    att_r = att[:, cur_p:cur_p + cur_r]
    att_p_new = _pad_left_2d(att_p, max_prompt, 0)
    att_r_new = _pad_right_2d(att_r, max_resp, 0)
    attention_mask_new = torch.cat([att_p_new, att_r_new], dim=1)

    input_ids_new = torch.cat([prompts_new, responses_new], dim=1)

    total_len = max_prompt + max_resp
    position_ids_new = _pad_position_ids(td["position_ids"], dp_left, dr_right, total_len)

    td["prompts"] = prompts_new
    td["responses"] = responses_new
    td["attention_mask"] = attention_mask_new
    td["input_ids"] = input_ids_new
    td["position_ids"] = position_ids_new
    return td

def _format_debug_attention_samples(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor, prompt_length: int, max_samples: int = 3) -> list:
    """Return list[str] of debug strings for first `max_samples` samples.
    Each string contains tokens (with PAD shown), attention mask and response_mask.
    """
    debug_list = []
    max_samples = 1
    for i in range(max_samples):
        ids = input_ids[i].tolist()
        att = attention_mask[i].tolist()
        # response_mask may be shorter/None; handle safely
        try:
            resp_m = response_mask[i].tolist()
        except Exception:
            resp_m = [0] * (len(ids) - prompt_length)
        try:
            toks = tokenizer.convert_ids_to_tokens(ids)
        except Exception:
            toks = [str(x) for x in ids]

        toks_display = []
        for idx_tok, tok in enumerate(toks):
            if att[idx_tok]:
                toks_display.append(f"{tok}")
            else:
                toks_display.append("<PAD>")

        sample_str = (
            f"sample={i} prompt_len={prompt_length} total_len={len(ids)}\n"
            f"TOKENS: {' '.join(toks_display)}\n"
            f"ATT:    {' '.join(str(x) for x in att)}\n"
            f"RESP_M: {' '.join(str(x) for x in resp_m)}\n"
        )
        debug_list.append(sample_str)
    return debug_list


# =========================
# LLM server manager
# =========================

class AsyncLLMServerManager:
    """
    Manage multiple OpenAI compatible LLM servers:
      - Load balance: least requests
      - Sticky session: same request_id -> same server
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
        )
        return output


# =========================
# Agent loop outputs
# =========================

class AgentLoopMetrics(BaseModel):
    generate_sequences: float = 0.0
    tool_calls: float = 0.0


class AgentLoopOutput(BaseModel):
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: Optional[list[float]] = None
    multi_modal_data: Optional[dict[str, Any]] = None
    reward_score: Optional[float] = None
    num_turns: int = 0
    metrics: AgentLoopMetrics
    extra_fields: dict[str, Any] = {}
    uid: Optional[str] = ""
    branch_id: Optional[str] = ""
    branch_depth: Optional[int] = 0


class _InternalAgentLoopOutput(AgentLoopOutput):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    response_mask: torch.Tensor
    attention_mask: torch.Tensor
    response_logprobs: Optional[torch.Tensor] = None
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    extra_fields: dict[str, Any] = {}
    uid: Optional[str] = ""
    branch_id: Optional[str] = ""
    branch_depth: Optional[int] = 0


class _DummyConfig:
    def __init__(self, config: DictConfig) -> None:
        self.config = config


class AgentLoopBase(ABC):
    _class_initialized = False

    def __init__(
        self,
        trainer_config: _DummyConfig,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        self.init_class(config=trainer_config.config, tokenizer=tokenizer, processor=processor, **kwargs)
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, processor: AutoProcessor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        raise NotImplementedError


_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass
    return decorator


async def get_trajectory_info(step, index, validate):
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


# =========================
# Worker
# =========================

class AgentLoopWorkerBase:
    """
    Ray actor worker:
      - For each incoming batch chunk: run per-sample async agent loop
      - Within this worker chunk: pad to a consistent length (for stacking)
      - Return DataProto (per worker)
    """

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
    ):
        self.config = config

        if not hasattr(self, "server_manager"):
            self.server_manager = AsyncLLMServerManager(config, server_handles)

        self.reward_router_address = reward_router_address

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config

        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        self.reward_manager_worker = RewardManagerWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, self.reward_router_address)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """
        NOTE: This is async (allowed inside ray actor).
        It returns a per-worker DataProto that is internally padded (within worker).
        Cross-worker padding is handled in AgentLoopManager.generate_sequences (sync).
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        # Get selected max_model_len from meta_info (set by ray_trainer)
        selected_max_len = batch.meta_info.get("selected_max_model_len")

        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            # Support per-batch max_model_len if enabled
            config = self.config.actor_rollout_ref.rollout
            if selected_max_len is not None:
                # Use the selected value from ray_trainer - same for all samples in batch
                kwargs["max_model_len"] = selected_max_len
            tasks.append(asyncio.create_task(self._run_agent_loop_branch(sampling_params, trajectory_info[i], **kwargs)))
        outputs = await asyncio.gather(*tasks)

        # flatten outputs (each task returns list[_InternalAgentLoopOutput])
        flat_outputs: list[_InternalAgentLoopOutput] = []
        for item in outputs:
            if isinstance(item, list):
                flat_outputs.extend(item)
            else:
                flat_outputs.append(item)

        # -------- Within-worker padding to a consistent length (so _postprocess torch.cat works) --------
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        max_prompt_len_batch = max(x.prompt_ids.size(1) for x in flat_outputs)
        max_resp_len_batch = max(x.response_ids.size(1) for x in flat_outputs)
        total_len = max_prompt_len_batch + max_resp_len_batch

        for x in flat_outputs:
            # prompts: left pad
            x.prompt_ids = _pad_left_2d(x.prompt_ids, max_prompt_len_batch, pad_id)
            # responses: right pad
            x.response_ids = _pad_right_2d(x.response_ids, max_resp_len_batch, pad_id)
            # response_mask: right pad with 0
            x.response_mask = _pad_right_2d(x.response_mask, max_resp_len_batch, 0)

            # response_logprobs (optional): right pad with 0.0
            if x.response_logprobs is not None:
                x.response_logprobs = _pad_right_2d(x.response_logprobs, max_resp_len_batch, 0.0)

            # attention_mask / input_ids: align to total_len
            x.attention_mask = _pad_right_2d(x.attention_mask, total_len, 0)
            x.input_ids = _pad_right_2d(x.input_ids, total_len, pad_id)

            # position_ids: [1,total] or [1,3,total]
            if x.position_ids.dim() == 2:
                x.position_ids = _pad_right_2d(x.position_ids, total_len, 0)
            else:
                # pad on last dim
                cur = x.position_ids.size(2)
                if cur < total_len:
                    x.position_ids = F.pad(x.position_ids, (0, total_len - cur), value=0)
                else:
                    x.position_ids = x.position_ids[:, :, :total_len]
        # ---------------------------------------------------------------------------------------------

        return self._postprocess(flat_outputs)

    async def _run_agent_loop_branch(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> list[_InternalAgentLoopOutput]:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
            )

            raw_outputs = await agent_loop.run(sampling_params, **kwargs)
            if isinstance(raw_outputs, list):
                agent_outputs = raw_outputs
            else:
                agent_outputs = [raw_outputs]

            internal_outputs: list[_InternalAgentLoopOutput] = []
            max_prompt_len = max(len(o.prompt_ids) for o in agent_outputs)
            max_resp_len = max(len(o.response_ids) for o in agent_outputs)

            for output in agent_outputs:
                # prompt padding (left)
                self.tokenizer.padding_side = "left"
                prompt_output = self.tokenizer.pad(
                    {"input_ids": output.prompt_ids},
                    padding="max_length",
                    max_length=max_prompt_len,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                if prompt_output["input_ids"].dim() == 1:
                    prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                    prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

                # response padding (right)
                self.tokenizer.padding_side = "right"
                response_output = self.tokenizer.pad(
                    {"input_ids": output.response_ids},
                    padding="max_length",
                    max_length=max_resp_len,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                if response_output["input_ids"].dim() == 1:
                    response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                    response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

                # response_mask padding (MUST pad with 0)
                rm = torch.tensor(output.response_mask, dtype=torch.long)
                if rm.numel() < max_resp_len:
                    rm = torch.cat([rm, torch.zeros(max_resp_len - rm.numel(), dtype=torch.long)], dim=0)
                else:
                    rm = rm[:max_resp_len]
                response_mask = rm.unsqueeze(0)  # [1, max_resp_len]
                response_mask = response_mask * response_output["attention_mask"]

                # response_logprobs padding
                response_logprobs = None
                if output.response_logprobs is not None:
                    lp = torch.tensor(output.response_logprobs, dtype=torch.float32)
                    if lp.numel() < max_resp_len:
                        lp = torch.cat([lp, torch.zeros(max_resp_len - lp.numel(), dtype=torch.float32)], dim=0)
                    else:
                        lp = lp[:max_resp_len]
                    response_logprobs = lp.unsqueeze(0)

                attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
                input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

                # position_ids
                if (
                    self.processor is not None
                    and getattr(self.processor, "image_processor", None) is not None
                    and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
                ):
                    from verl.models.transformers.qwen2_vl import get_rope_index

                    images = None
                    if output.multi_modal_data is not None:
                        images = output.multi_modal_data.get("image", None)

                    current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
                    multi_modal_inputs = self.processor(text=[current_text], images=images, return_tensors="pt")
                    multi_modal_inputs.pop("input_ids", None)
                    multi_modal_inputs.pop("attention_mask", None)
                    multi_modal_inputs = dict(multi_modal_inputs)

                    vision_position_ids = get_rope_index(
                        self.processor,
                        input_ids=input_ids.squeeze(0),
                        image_grid_thw=multi_modal_inputs.get("image_grid_thw"),
                        video_grid_thw=multi_modal_inputs.get("video_grid_thw"),
                        second_per_grid_ts=multi_modal_inputs.get("second_per_grid_ts"),
                        attention_mask=attention_mask.squeeze(0),
                    ).unsqueeze(0)

                    valid_mask = attention_mask[0].bool()
                    text_position_ids = torch.ones((1, input_ids.size(1)), dtype=torch.long)
                    text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                    text_position_ids = text_position_ids.unsqueeze(0)  # [1,1,T]

                    position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)
                else:
                    multi_modal_inputs = None
                    position_ids = compute_position_id_with_mask(attention_mask)

                # reward (optional async)
                enable_async_reward = (
                    (self.reward_router_address is not None and self.config.reward_model.enable_resource_pool)
                    or not self.config.reward_model.enable
                )

                if output.reward_score is None and enable_async_reward:
                    batch_td = TensorDict(
                        {
                            "prompts": prompt_output["input_ids"],
                            "responses": response_output["input_ids"],
                            "attention_mask": attention_mask,
                            "input_ids": input_ids,
                            "position_ids": position_ids,
                        },
                        batch_size=1,
                    )

                    non_tensor_batch = {
                        **{k: np.array([v]) for k, v in kwargs.items()},
                        "__num_turns__": np.array([output.num_turns]),
                    }
                    for k, v in output.extra_fields.items():
                        non_tensor_batch[k] = np.array([v], dtype=object)

                    data = DataProto(batch=batch_td, non_tensor_batch=non_tensor_batch)
                    result = await self.reward_manager_worker.compute_score.remote(data)
                    output.reward_score = result["reward_score"]
                    output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

                internal_outputs.append(
                    _InternalAgentLoopOutput(
                        prompt_ids=prompt_output["input_ids"],
                        response_ids=response_output["input_ids"],
                        input_ids=input_ids,
                        position_ids=position_ids,
                        response_mask=response_mask,
                        attention_mask=attention_mask,
                        response_logprobs=response_logprobs,
                        multi_modal_inputs=multi_modal_inputs,
                        multi_modal_data=output.multi_modal_data,
                        reward_score=output.reward_score,
                        num_turns=output.num_turns,
                        metrics=output.metrics,
                        extra_fields=output.extra_fields,
                        uid=output.uid,
                        branch_id=output.branch_id,
                        branch_depth=output.branch_depth,
                    )
                )

            return internal_outputs

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        prompt_ids = torch.cat([x.prompt_ids for x in inputs], dim=0)
        response_ids = torch.cat([x.response_ids for x in inputs], dim=0)
        response_mask = torch.cat([x.response_mask for x in inputs], dim=0)
        attention_mask = torch.cat([x.attention_mask for x in inputs], dim=0)
        input_ids = torch.cat([x.input_ids for x in inputs], dim=0)
        position_ids = torch.cat([x.position_ids for x in inputs], dim=0)

        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat(
                [x.response_logprobs for x in inputs], dim=0
            )

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [x.reward_score for x in inputs]
        if all(s is not None for s in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {"__num_turns__": np.array([x.num_turns for x in inputs], dtype=np.int32)}
        non_tensor_batch["uid"] = np.array([getattr(x, "uid", "") for x in inputs], dtype=object)
        non_tensor_batch["branch_id"] = np.array([getattr(x, "branch_id", "") for x in inputs], dtype=object)
        non_tensor_batch["branch_depth"] = np.array([getattr(x, "branch_depth", 0) for x in inputs], dtype=object)

        # reward extra info keys
        reward_extra_infos = [x.extra_fields.get("reward_extra_info", {}) for x in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys()) if len(reward_extra_infos) > 0 else []
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info.get(key) for info in reward_extra_infos])

        # multi_modal_inputs
        multi_modal_inputs_list = [x.multi_modal_inputs for x in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        # extra fields union
        extra_fields = {}
        all_keys = set(k for x in inputs for k in x.extra_fields.keys())
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [x.extra_fields.get(key) for x in inputs]
            extra_fields[key] = temp_arr
        non_tensor_batch.update(extra_fields)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        # [DEBUG]:attention mask and response mask
        prompt_length = prompt_ids.size(1)
        debug_list = _format_debug_attention_samples(self.tokenizer, input_ids, attention_mask, response_mask, prompt_length, max_samples=3)
        # for sample_str in debug_list:
        #     print("[DEBUG debug_attention] %s", sample_str)

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"metrics": [x.metrics.model_dump() for x in inputs], "reward_extra_keys": reward_extra_keys, "pad_id": pad_id},
        )

    def create_transferqueue_client(self, controller_infos, storage_infos, role):
        from verl.single_controller.ray.base import get_random_string
        from verl.utils.transferqueue_utils import create_transferqueue_client

        client_name = get_random_string(length=6)
        create_transferqueue_client(
            client_id=f"{role}_worker_{client_name}",
            controller_infos=controller_infos,
            storage_infos=storage_infos,
        )


@ray.remote
class AgentLoopWorker(AgentLoopWorkerBase):
    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], reward_router_address: str = None):
        super().__init__(config, server_handles, reward_router_address)


# =========================
# Manager
# =========================

class AgentLoopManager:
    """
    Manager (NOT async):
      - owns rollout replicas / server handles
      - owns agent_loop_workers (ray actors)
      - generate_sequences() must be SYNC (trainer expects DataProto, not coroutine)
      - cross-worker padding is done here BEFORE DataProto.concat(outputs)
    """

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup = None, rm_wg: RayWorkerGroup = None):
        self.config = config
        self.worker_group = worker_group

        self.reward_model_manager = None
        self.reward_router_address = None
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward import RewardModelManager
            self.reward_model_manager = RewardModelManager(config.reward_model, rm_wg)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = AgentLoopWorker

        self._initialize_llm_servers()
        self._init_agent_loop_workers()

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]
        if self.worker_group:
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.reward_router_address)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        SYNC function (trainer expects immediate DataProto, not coroutine).
        Steps:
          1) split prompts -> workers
          2) ray.get worker outputs
          3) CROSS-WORKER pad
          4) DataProto.concat
        """
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.wake_up()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.wake_up()

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )

        # -------- ✅ Cross-worker padding (fix 1410 vs 1709) --------
        max_prompt = max(o.batch["prompts"].shape[1] for o in outputs)
        max_resp = max(o.batch["responses"].shape[1] for o in outputs)

        pad_id = outputs[0].meta_info.get("pad_id", None)
        if pad_id is None:
            pad_id = 0

        for o in outputs:
            o.batch = _pad_tensordict_rollout(o.batch, max_prompt, max_resp, pad_id)

        output = DataProto.concat(outputs)
        # ----------------------------------------------------------

        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            self.sleep()
        if self.reward_model_manager and self.config.reward_model.rollout.free_cache_engine:
            self.reward_model_manager.sleep()

        metrics = [out.meta_info.pop("metrics") for out in outputs]
        timing = self._performance_metrics(metrics, output)
        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([m["generate_sequences"] for chunk in metrics for m in chunk])
        t_tool_calls = np.array([m["tool_calls"] for chunk in metrics for m in chunk])

        timing["agent_loop/generate_sequences/min"] = float(t_generate_sequences.min()) if t_generate_sequences.size else 0.0
        timing["agent_loop/generate_sequences/max"] = float(t_generate_sequences.max()) if t_generate_sequences.size else 0.0
        timing["agent_loop/generate_sequences/mean"] = float(t_generate_sequences.mean()) if t_generate_sequences.size else 0.0
        timing["agent_loop/tool_calls/min"] = float(t_tool_calls.min()) if t_tool_calls.size else 0.0
        timing["agent_loop/tool_calls/max"] = float(t_tool_calls.max()) if t_tool_calls.size else 0.0
        timing["agent_loop/tool_calls/mean"] = float(t_tool_calls.mean()) if t_tool_calls.size else 0.0

        if t_generate_sequences.size and t_tool_calls.size:
            slowest = int(np.argmax(t_generate_sequences + t_tool_calls))
            attention_mask = output.batch["attention_mask"][slowest]
            prompt_length = output.batch["prompts"].shape[1]
            timing["agent_loop/slowest/generate_sequences"] = float(t_generate_sequences[slowest])
            timing["agent_loop/slowest/tool_calls"] = float(t_tool_calls[slowest])
            timing["agent_loop/slowest/prompt_length"] = float(attention_mask[:prompt_length].sum().item())
            timing["agent_loop/slowest/response_length"] = float(attention_mask[prompt_length:].sum().item())
        else:
            timing["agent_loop/slowest/generate_sequences"] = 0.0
            timing["agent_loop/slowest/tool_calls"] = 0.0
            timing["agent_loop/slowest/prompt_length"] = 0.0
            timing["agent_loop/slowest/response_length"] = 0.0

        return timing

    def wake_up(self):
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)
        asyncio.run(run_all())
