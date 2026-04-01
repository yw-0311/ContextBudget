# Copyright 2025 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

import asyncio
import copy
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, List, Tuple
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

MAX_SUMMARY_DEPTH = 10

# =========================
# Agent State
# =========================
class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


# =========================
# Regex for context_commit
# =========================
_CONTEXT_COMMIT_RE = re.compile(r"<context_commit\b([^>]*)>(.*?)</context_commit>", re.DOTALL | re.IGNORECASE)


# =========================
# ENGLISH POLICY / BUDGET PROMPT
# =========================

# FOLD_POLICY_EN = (
#     "Please choose the most appropriate folding strategy based on the remaining available budget. "
#     "If budget permits, perform as many searches as needed to verify the information and deliver the correct answer. Use None if no further Fold search is required."
#     "The options include: None / Partial (e.g., c0001,c0002) / ALL. "
#     "'None' means no folding. "
#     "'c0001,c0002' means folding those commits and replacing them with merged_commit information. "
#     "'ALL' means folding all commits and replacing them with merged_commit. "
# )

# FOLD_POLICY_EN = (
#     "Please choose the most appropriate folding strategy based on the remaining available budget. After Use fold summarize(None,ALL or Partial),you will get the before search result "
#     "If budget permits, perform as many searches as needed to verify the information and deliver the correct answer. Use None if no further Fold is required."
#     "The options include: None / Partial (e.g., c0001,c0002) / ALL. "
#     "'None' means no folding. "
#     "'c0001,c0002' means folding those commits and replacing them with merged_commit information. "
#     "'ALL' means folding all commits and replacing them with merged_commit. "
#     "You must decide the folding option and what to write in merged_commit based on the budget. "
# )

# FOLD_POLICY_EN = (
# "Before receiving the response from your previous tool call, please select a context management strategy based on the remaining context budget. You cannot choose any other tools—only the 'summarize' tool is available—but you may specify particular fold IDs."
# "The options include: None / Partial (e.g., c0001,c0002) / ALL."
# "'None' means no folding."
# "'c0001,c0002' means folding those specific commits and replacing them with merged_commit information. "
# "'ALL' means folding all commits and replacing them with a single merged_commit. "
# "merged_commit should minimize information loss from the previously compressed blocks as much as possible."
# "You must decide both the folding option and the content of the merged_commit based on the available budget. Excessive folding may lead to information loss, while retaining redundant context may cause failure in your subsequent tasks. After executing the 'summarize' tool, you will receive the folded context along with the response from your previous tool call, enabling your next reasoning step."
# "Before outputting the summarize tool, you are prohibited from using any other tools such as search (even if they are available). Note that even if you do not need to fold anything, you must still use the summarize tool and set fold_commit_ids to None."
# "If you can provide the final answer based on the existing information, you may choose to output the final result directly in accordance with the required format."
# )

FOLD_POLICY_EN = (
"Before receiving the response from your previous tool call, decide whether context folding is necessary based on the remaining context budget. The goal of folding is to preserve context for future reasoning, not to compress information by default."
"Available options: None / Partial (e.g., c0001,c0002) / ALL."
"'None' SHOULD be chosen when the remaining context budget is sufficient. This is the preferred option when no immediate context pressure exists."
"IMPORTANT: Choosing 'None' does NOT mean skipping the summarize tool. It means calling the summarize tool with fold_commit_ids set to None."
"'Partial' means folding only the specified commits and replacing them with a merged_commit summary."
"'ALL' means folding all commits into a single merged_commit and should be used only when context is critically limited."
"When folding is applied, merged_commit should preserve all information necessary for subsequent reasoning."
"Do not fold unless it is necessary. Excessive or unnecessary folding may degrade task performance."
"Your search tools are now banned. You cannot use any search tools before selecting the fold policy, even if they are available. Give your summarize strategy:"
"OUTPUT FORMAT (STRICT): Your response MUST be a single JSON object with exact structure: <tool_call>\n{\"name\": \"summarize\", \"arguments\": {\"fold_commit_ids\": None | [\"c0001\",\"c0002\",...] | \"ALL\", \"merged_commit\": None | \"summary text\"}}\n</tool_call>"
)



WO_BUDGET_TEMPLATE_EN = """\
    "Please choose the most appropriate folding strategy"
    "The options include:  Partial (e.g., c0001,c0002) / ALL. "
    "'c0001,c0002' means folding those commits and replacing them with merged_commit information. "
    "'ALL' means folding all commits and replacing them with merged_commit. "
"""

BUDGET_TEMPLATE_EN = """\
[Remaining context capacity prior to receiving tool call results]
Please decide whether to summarize or fold the upcoming tool response to stay within limits.
- Current prompt length: {current_ctx_len} tokens
- Estimated tool response length: {tool_response_len} tokens
- Remaining tokens for next turn: {remaining_budget} tokens ({remaining_pct:.1f}% of usable context)
- Usable context limit (max length minus 1,000 safety margin): {usable_limit} tokens
{fold_policy}
"""

# BUDGET_TEMPLATE_EN = """\
# [CONTEXT BUDGET Info]- Remaining tokens for next turn: {remaining_budget} tokens ({remaining_pct:.1f}% context left)
# Please decide whether to summarize or fold the upcoming tool response to stay within limits.
# {fold_policy}
# """


# =========================
# Commit Registry (encapsulated)
# =========================
class CommitRegistry:
    """
    CommitRegistry is the ONLY truth source for commit contents.
    It supports add/remove/fold/renumber and can render commit messages.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._commits: List[Dict[str, Any]] = []  # item: {id, kind, content, tokens, digest}

    # -------- basic access --------
    def items(self) -> List[Dict[str, Any]]:
        return self._commits

    def is_empty(self) -> bool:
        return len(self._commits) == 0

    def ids(self) -> List[str]:
        return [c.get("id", "") for c in self._commits]

    def clear(self) -> None:
        self._commits.clear()

    def extend(self, commits: List[Dict[str, Any]]) -> None:
        self._commits.extend(commits)

    # -------- commit ops --------
    def add_commit(self, kind: str, content: str, digest: str = "", cid: Optional[str] = None) -> str:
        tok = len(self._tokenizer.encode(content or "", add_special_tokens=False))
        if cid is None:
            # temporary id; will be renumbered by renumber()
            cid = "TEMP"
        self._commits.append(
            {"id": cid, "kind": kind, "content": content or "", "tokens": tok, "digest": digest or ""}
        )
        return cid

    def remove_by_ids(self, idset: set) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """Remove commits whose id in idset; return kept commits and first removed index in kept-space."""
        kept: List[Dict[str, Any]] = []
        first_fold_index: Optional[int] = None
        for c in self._commits:
            cid = c.get("id")
            if cid in idset:
                if first_fold_index is None:
                    first_fold_index = len(kept)
                continue
            kept.append(c)
        return kept, first_fold_index

    def renumber(self) -> None:
        """Renumber commits as c0001.. in order (in-place)."""
        for i, c in enumerate(self._commits, start=1):
            c["id"] = f"c{i:04d}"

    # -------- rendering --------
    @staticmethod
    def strip_all_context_commits(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove ALL <context_commit> blocks from messages."""
        new_msgs: List[Dict[str, Any]] = []
        for m in messages:
            t = m.get("content", "") or ""
            if not t:
                new_msgs.append(m)
                continue
            nt = _CONTEXT_COMMIT_RE.sub("", t).strip()
            if nt:
                nm = dict(m)
                nm["content"] = nt
                new_msgs.append(nm)
        return new_msgs

    @staticmethod
    def make_context_commit_block(cid: str, content: str, tokens: int, kind: str, digest: str = "") -> str:
        digest_attr = f' digest="{digest}"' if digest else ""
        return (
            f'<context_commit id="{cid}" kind="{kind}" tokens="{tokens}"{digest_attr}>\n'
            f"{content}\n"
            f"</context_commit>"
        )

    def render_tool_messages(self) -> List[Dict[str, Any]]:
        """Render commits into tool role messages."""
        tool_msgs: List[Dict[str, Any]] = []
        for c in self._commits:
            tool_msgs.append(
                {
                    "role": "tool",
                    "content": self.make_context_commit_block(
                        cid=c.get("id", ""),
                        content=c.get("content", ""),
                        tokens=int(c.get("tokens", 0)),
                        kind=c.get("kind", "tool_obs"),
                        digest=c.get("digest", ""),
                    ),
                }
            )
        return tool_msgs

    # -------- fold protocol --------
    def apply_fold(
        self,
        fold_commit_ids: str,
        merged_commit: str,
        merged_digest: str,
        pending_tool_payload_text: str,
    ) -> List[Dict[str, Any]]:
        """
        Fold only mutates registry (truth source). It also appends pending tool payload as tool_obs commit.
        Always renumbers and returns a DELTA messages list (summary + rendered commits).
        """
        raw = [x.strip() for x in (fold_commit_ids or "NONE").split(",") if x.strip()]
        current_ids = self.ids()

        if any(x.upper() == "ALL" for x in raw):
            fold_ids = set(current_ids)
            print(f"[DEBUG] ALL Fold id: {fold_ids}")
        # elif any(x.upper() == "NONE" for x in raw) or len(raw) == 0:
        #     fold_ids = set()
        else:
            fold_ids = set(raw)
            print(f"[DEBUG] Fold id {fold_ids}")

        # 1) remove selected
        kept, first_fold_index = self.remove_by_ids(fold_ids)
        print(f"[DEBUG] len(kept):{len(kept)}")

        # 2) insert merged fold commit if needed
        if fold_ids:
            merged_text = merged_commit or ""
            merged_tok = len(self._tokenizer.encode(merged_text, add_special_tokens=False))
            fold_entry = {
                "id": "TEMP",
                "kind": "fold",
                "content": merged_text,
                "tokens": merged_tok,
                "digest": merged_digest or "",
            }
            insert_at = first_fold_index if first_fold_index is not None else len(kept)
            kept.insert(insert_at, fold_entry)

        # 3) append pending tool payload as tool_obs commit (if any)
        if pending_tool_payload_text:
            pending_text = pending_tool_payload_text
            pending_tok = len(self._tokenizer.encode(pending_text, add_special_tokens=False))
            kept.append(
                {
                    "id": "TEMP",
                    "kind": "tool_obs",
                    "content": pending_text,
                    "tokens": pending_tok,
                    "digest": "",
                }
            )

        # 4) replace registry in-place + renumber in-place
        self._commits.clear()
        print(f"[DEBUG] before self._commits.extend(kept) {len(kept)}")
        self._commits.extend(kept)
        self.renumber()

        # 5) return delta messages (optional summary + commits)
        # 返回一个列表，将commit转换为了 tool 的message
        delta: List[Dict[str, Any]] = []
        delta.extend(self.render_tool_messages())
        return delta


# =========================
# Agent Data
# =========================
class AgentData:
    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        commit_registry: CommitRegistry,
        branch_depth: int = 0,
        uid: Optional[str] = "null",
        branch_id: Optional[str] = "null",
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs

        # Tokens and masks
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []

        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        self.tool_calls: list[FunctionCall] = []

        self.uid = uid
        self.branch_id = branch_id
        self.branch_depth = branch_depth

        # Pending tool payload (deferred until summarize/fold)
        self.pending_tool_payload_text: str = ""

        # Shared branch-global registry (truth source)
        self.commit_registry: CommitRegistry = commit_registry


# =========================
# Base Tool Agent Loop
# =========================
class ToolAgentLoopBase(AgentLoopBase, ABC):
    """
    Base class for tool agent loops with shared functionality.
    Subclasses should implement the abstract method _handle_processing_tools_state
    to customize tool processing behavior.
    """

    # =========================
    # init
    # =========================
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        cls.tokenizer = tokenizer
        cls.processor = processor

        rollout = config.actor_rollout_ref.rollout
        multi_turn = rollout.multi_turn

        cls.max_user_turns = multi_turn.max_user_turns
        cls.max_assistant_turns = multi_turn.max_assistant_turns
        cls.max_parallel_calls = multi_turn.max_parallel_calls
        cls.max_tool_response_length = multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = multi_turn.tool_response_truncate_side
        cls.max_model_len = rollout.max_model_len
        cls.max_depth = multi_turn.max_depth

        # Budget Info
        cls.enable_budget = multi_turn.enable_budget
        print(f"[DEBUG] enable_budget {cls.enable_budget}")

        tool_list = initialize_tools_from_config(multi_turn.tool_config_path) if multi_turn.tool_config_path else []
        cls.tools = {t.name: t for t in tool_list}
        cls.tool_schemas = [t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in tool_list]

        cls.tool_parser = ToolParser.get_tool_parser(multi_turn.format, tokenizer)
        cls.tool_parser_name = multi_turn.format

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        # cls.prompt_length = rollout.prompt_length
        # cls.response_length = rollout.response_length

        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

        print("[INIT] ToolAgentLoop init_class done.", flush=True)
        print(f"[INIT] tools={list(cls.tools.keys())}", flush=True)
        print(f"[INIT] tool_parser_name={cls.tool_parser_name}", flush=True)

    # =========================
    # debug helper
    # =========================
    def _preview(self, s: str, n: int = 200) -> str:
        if not s:
            return ""
        s = s.replace("\n", "\\n")
        return s[:n] + ("..." if len(s) > n else "")

    # =========================
    # Append messages and encode tail
    # =========================
    async def _append_messages_and_encode_tail(self, agent_data: AgentData, add_messages: list[dict[str, Any]]):
        if not add_messages:
            return

        depth = agent_data.branch_depth
        before_msg_len = len(agent_data.messages)
        before_prompt_len = len(agent_data.prompt_ids)

        # semantic append
        agent_data.messages.extend(add_messages)

        # encode ONLY tail messages
        tail_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                add_messages,
                tools=None,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )

        # strip system prompt prefix if included
        if tail_ids[: len(self.system_prompt)] == self.system_prompt:
            tail_ids = tail_ids[len(self.system_prompt) :]

        agent_data.prompt_ids += tail_ids
        agent_data.response_mask += [0] * len(tail_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(tail_ids)

        agent_data.user_turns += len(add_messages)

        after_msg_len = len(agent_data.messages)
        after_prompt_len = len(agent_data.prompt_ids)
        print(
            f"[D{agent_data.branch_id}-{depth}][APPEND_TAIL] messages {before_msg_len}->{after_msg_len}, "
            f"prompt_ids {before_prompt_len}->{after_prompt_len}, tail={len(tail_ids)}",
            flush=True,
        )

    # =========================
    # DFS entry (raw_prompt + delta; branch-global commit registry)
    # =========================
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> List[AgentLoopOutput]:
        # Support dynamic max_model_len from ray_trainer
        if "max_model_len" in kwargs:
            self.max_model_len = kwargs.pop("max_model_len")
            print(f"[DEBUG] dynamic max_model_len {self.max_model_len }")

        outputs: List[AgentLoopOutput] = []
        raw_prompt = list(kwargs["raw_prompt"])

        self._branch_registries: Dict[str, CommitRegistry] = {}

        def merge_delta_into_user(prompt: list, delta: list) -> list:
            if not delta:
                return prompt

            summary = "\n".join(
                m["content"].strip()
                for m in delta
                if isinstance(m, dict)
                and isinstance(m.get("content"), str)
                and m["content"].strip()
            )
            if not summary:
                return prompt

            prompt = list(prompt)  # shallow copy
            prompt[1] = dict(prompt[1])  # copy user msg
            prompt[1]["content"] = (prompt[1].get("content") or "") + f"\n\nBelow is a summarized working memory from previous steps.It may be incomplete or lossy due to context limitations. <summary>{summary}<summary>. Based on this summary: Decide whether the information is sufficient to answer the task. If sufficient, produce the final answer. If not, identify what is missing and specify the next search or tool call needed:\n"
            return prompt

        async def dfs(prompt_messages, delta_messages, image_data, depth, branch_id, registry):
            if depth >= self.max_depth:
                print(f"[D{depth}][DFS] MAX_DEPTH_REACHED, STOP", flush=True)
                return

            # if branch_id not in self._branch_registries:
            #     self._branch_registries[branch_id] = CommitRegistry(self.tokenizer)

            uid = kwargs.get("uid", "")

            prompt_messages = copy.deepcopy(raw_prompt)
            prompt_messages_delta = merge_delta_into_user(prompt_messages, delta_messages)

            reg = registry
            print(
                f"[D{branch_id}-{depth}][DFS_ENTER] commits={len(reg.items())} ids={reg.ids()[:5]} ...",
                flush=True,
            )
            out, next_delta = await self._run_single_branch(
                prompt_messages_delta,
                image_data,
                sampling_params,
                depth,
                uid,
                branch_id,
                commit_registry=registry,
            )
            outputs.append(out)

            if next_delta is not None:
                print(f"[D{depth}][DFS] RECURSE -> next_delta_len={len(next_delta)}", flush=True)
                await dfs(raw_prompt, next_delta, image_data, depth + 1, branch_id,registry)

        await dfs(
            prompt_messages=raw_prompt,
            delta_messages=[],
            image_data=copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image")),
            depth=0,
            branch_id=uuid4().hex,
            registry = CommitRegistry(self.tokenizer)
        )
        return outputs

    async def _run_single_branch(
        self,
        messages,
        image_data,
        sampling_params,
        depth: int,
        uid,
        branch_id,
        commit_registry: CommitRegistry,
    ):
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics={},
            request_id=uuid4().hex,
            tools_kwargs={},
            branch_depth=depth,
            uid=uid,
            branch_id=branch_id,
            commit_registry=commit_registry,
        )

        state = AgentState.PENDING
        summarize_delta_messages: Optional[list] = None

        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state, summarize_delta_messages = await self._handle_processing_tools_state(agent_data)
            else:
                state = AgentState.TERMINATED

        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        
        # output = AgentLoopOutput(
        #     prompt_ids=prompt_ids,
        #     response_ids=response_ids[: self.response_length],
        #     response_mask=agent_data.response_mask[: self.response_length],
        #     multi_modal_data={"image": agent_data.image_data} if agent_data.image_data else {},
        #     response_logprobs=agent_data.response_logprobs[: self.response_length] if agent_data.response_logprobs else None,
        #     num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
        #     metrics=agent_data.metrics,
        #     extra_fields={},
        #     uid=uid,
        #     branch_id=branch_id,
        #     branch_depth=depth,
        # )

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=agent_data.response_mask,
            multi_modal_data={"image": agent_data.image_data} if agent_data.image_data else {},
            response_logprobs=agent_data.response_logprobs,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
            uid=uid,
            branch_id=branch_id,
            branch_depth=depth,
        )

        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        output.extra_fields["final_messages"] = agent_data.messages
        # output.extra_fields["commit_registry"] = agent_data.commit_registry.items()
        output.extra_fields["summarize_child_messages"] = summarize_delta_messages

        return output, summarize_delta_messages

    # =========================
    # State handlers
    # =========================
    async def _handle_pending_state(self, agent_data: AgentData) -> AgentState:
        depth = agent_data.branch_depth
        print(f"[D{agent_data.branch_id}-{depth}][PENDING] encode initial prompt from messages={len(agent_data.messages)}", flush=True)

        if self.processor is not None:
            raw = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

        print(f"[D{agent_data.branch_id}-{depth}][PENDING] prompt_ids_len={len(agent_data.prompt_ids)}", flush=True)
        return AgentState.GENERATING

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        depth = agent_data.branch_depth

        usable_limit = self.max_model_len - 1000
        current_len = len(agent_data.prompt_ids)
        used_pct = (current_len / usable_limit) * 100 if usable_limit > 0 else 100.0
        remaining_usable = usable_limit - current_len
        print(
            f"[D{agent_data.branch_id}-{depth}][CTX] prompt_len={current_len}"
            f"({used_pct:.1f}% used)",
            flush=True,
        )

        if self.max_model_len is not None:
            remaining_total = self.max_model_len - len(agent_data.prompt_ids)
            if remaining_total <= 0:
                print(
                    f"[D{agent_data.branch_id}-{depth}][GEN] TERMINATE: context full, prompt_len={len(agent_data.prompt_ids)}, "
                    f"max_len={self.max_model_len}",
                    flush=True,
                )
                return AgentState.TERMINATED

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        
        # DEBUG: 监控 response_mask 构建
        new_response_len = len(agent_data.response_ids)
        if new_response_len == 0:
            print(f"[DEBUG][GEN_EMPTY] depth={depth}, assistant_turns={agent_data.assistant_turns}")
            print(f"[DEBUG][GEN_EMPTY] Generated empty response! token_ids={output.token_ids}")
        
        # if self.max_model_len is not None:
        #     remaining_total = self.max_model_len - len(agent_data.prompt_ids)
        #     if remaining_total <= 0:
        #         print(
        #             f"[D{agent_data.branch_id}-{depth}][GEN] TERMINATE: context full, prompt_len={len(agent_data.prompt_ids)}, "
        #             f"max_len={self.max_model_len}",
        #             flush=True,
        #         )
        #         return AgentState.TERMINATED

        agent_data.response_mask += [1] * new_response_len
        # if output.log_probs:
        if output.log_probs is not None:
            agent_data.response_logprobs += output.log_probs

        assistant_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        tool_names = [tc.name for tc in agent_data.tool_calls] if agent_data.tool_calls else []

        agent_data.messages.append({"role": "assistant", "content": assistant_text})

        print(f"[D{agent_data.branch_id}-{depth}][GEN] out_tokens={len(agent_data.response_ids)} tool_calls={tool_names}", flush=True)
        print(f"[D{agent_data.branch_id}-{depth}][GEN] preview='{self._preview(assistant_text, 3000)}'", flush=True)

        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            print(f"[D{agent_data.branch_id}-{depth}][GEN] TERMINATE: reached max_assistant_turns {self.max_assistant_turns}", flush=True)
            return AgentState.TERMINATED

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS

        print(f"[D{agent_data.branch_id}-{depth}][GEN] TERMINATE: no tool call", flush=True)
        return AgentState.TERMINATED

    # =========================
    # Abstract method for tool processing
    # =========================
    @abstractmethod
    async def _handle_processing_tools_state(self, agent_data: AgentData) -> tuple[AgentState, Optional[list]]:
        """
        Handle tool processing state. Subclasses must implement this method.
        
        Returns:
            tuple[AgentState, Optional[list]]: Next state and optional delta messages
        """
        pass

    # =========================
    # Tool call
    # =========================
    async def _call_tool(self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]) -> tuple[ToolResponse, float, dict]:
        tool, instance_id = None, None
        try:
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)
        except Exception as e:
            return (ToolResponse(text=f"Error when executing tool: {e}"), 0.0, {})
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        return ToolResponse(text=tool_response_text), tool_reward, res