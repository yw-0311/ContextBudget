# Copyright 2025 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

from typing import Any, Optional, List

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop_base import (
    ToolAgentLoopBase,
    AgentState,
    AgentData,
    CommitRegistry,
    BUDGET_TEMPLATE_EN,
    BUDGET_TEMPLATE_EN_BEFORE,
    WO_BUDGET_TEMPLATE_EN,
    FOLD_POLICY_EN,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer

import asyncio
import json
import re


@register("tool_agent_eager")
class ToolAgentLoopEager(ToolAgentLoopBase):
    """
    Tool agent loop with eager (pre-inject) tool response handling.

    Protocol:
    1) search/open result arrives
    2) inject tool_obs commit immediately
    3) inject budget prompt immediately
    4) after budget injection, further search/open is forbidden until summarize happens
    """

    def _next_cid(self, reg: CommitRegistry) -> str:
        """Generate next stable cid based on current registry ids."""
        max_n = 0
        for cid in reg.ids():
            m = re.match(r"c(\d{4,})$", cid or "")
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f"c{max_n + 1:04d}"

    def _ensure_runtime_flags(self, agent_data: AgentData) -> None:
        """
        为 eager 版本补充运行时 flag。
        不依赖 AgentData 静态定义，避免改动上游 dataclass。
        """
        if not hasattr(agent_data, "budget_injected_waiting_summarize"):
            agent_data.budget_injected_waiting_summarize = False

    async def _inject_tool_obs_commit(
        self,
        agent_data: AgentData,
        content: str,
        *,
        cid: Optional[str] = None,
        digest: str = "",
        kind: str = "tool_obs",
    ) -> str:
        """
        统一封装：把 tool response 写入 CommitRegistry + 以 <context_commit> tool message 注入上下文。
        返回最终使用的 cid。
        """
        if not content:
            return ""

        if cid is None:
            cid = self._next_cid(agent_data.commit_registry)

        tok = len(self.tokenizer.encode(content, add_special_tokens=False))

        # 1) truth source
        agent_data.commit_registry.add_commit(
            kind=kind,
            content=content,
            digest=digest,
            cid=cid,
        )

        # 2) inject into model context
        tool_msg = {
            "role": "tool",
            "content": CommitRegistry.make_context_commit_block(
                cid=cid,
                content=content,
                tokens=tok,
                kind=kind,
                digest=digest,
            ),
        }
        await self._append_messages_and_encode_tail(agent_data, [tool_msg])
        return cid

    async def _inject_budget_prompt(self, agent_data: AgentData) -> None:
        """
        注入 budget prompt，基于当前的上下文长度计算。
        在 eager 版本中，tool response 已经注入，所以计算 remaining budget。
        """
        self._ensure_runtime_flags(agent_data)

        depth = agent_data.branch_depth
        current_ctx_len = len(agent_data.prompt_ids)

        usable_limit = self.max_model_len
        remaining_budget = usable_limit - current_ctx_len
        remaining_pct = (
            max(0.0, remaining_budget / usable_limit * 100.0)
            if usable_limit > 0
            else 0.0
        )

        if self.enable_budget:
            budget_text = BUDGET_TEMPLATE_EN_BEFORE.format(
                current_ctx_len=current_ctx_len,
                remaining_budget=remaining_budget,
                remaining_pct=remaining_pct,
                usable_limit=usable_limit,
                fold_policy=FOLD_POLICY_EN,
            )
        else:
            budget_text = WO_BUDGET_TEMPLATE_EN

        budget_msg = {"role": "tool", "content": budget_text}

        print(
            f"[D{agent_data.branch_id}-{depth}][TOOLS] search/open -> inject budget, "
            f"remaining={remaining_budget} "
            f"budget_info {budget_text}",
            flush=True,
        )

        await self._append_messages_and_encode_tail(agent_data, [budget_msg])

        # 关键 gate:
        # 只要注入过 budget，在 summarize 前禁止再次 search/open
        agent_data.budget_injected_waiting_summarize = True

    def _has_search_or_open(self, tool_calls: List[FunctionCall]) -> bool:
        for tc in tool_calls:
            if tc.name in ("search", "open"):
                return True
        return False

    async def _handle_processing_tools_state(
        self, agent_data: AgentData
    ) -> tuple[AgentState, Optional[list]]:
        self._ensure_runtime_flags(agent_data)

        depth = agent_data.branch_depth

        tool_calls_all = agent_data.tool_calls[: self.max_parallel_calls]
        summarize_call = next((tc for tc in tool_calls_all if tc.name == "summarize"), None)
        exec_calls = [tc for tc in tool_calls_all if tc.name != "summarize"]

        # ------------------------------------------------------------
        # Guard:
        # 如果上一次 search/open 后已经注入了 budget，但还没 summarize，
        # 此时又出现新的 search/open，则直接打断。
        # ------------------------------------------------------------
        if agent_data.budget_injected_waiting_summarize and self._has_search_or_open(exec_calls):
            print(
                f"[D{agent_data.branch_id}-{depth}][GUARD] "
                f"search/open emitted again after budget injection before summarize -> TERMINATED",
                flush=True,
            )
            return AgentState.TERMINATED, None

        responses = []
        if exec_calls:
            tasks = [self._call_tool(tc, agent_data.tools_kwargs) for tc in exec_calls]
            with simple_timer("tool_calls", agent_data.metrics):
                responses = await asyncio.gather(*tasks)

        # collect search/open payloads (support multiple)
        search_open_payloads: List[str] = []

        for tc, (tool_response, tool_reward, _) in zip(exec_calls, responses):
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

            if not tool_response or not tool_response.text:
                continue

            if tc.name in ("search", "open"):
                search_open_payloads.append(tool_response.text)
            else:
                # Optional: keep other tool outputs
                # await self._append_messages_and_encode_tail(
                #     agent_data,
                #     [{"role": "tool", "content": tool_response.text}]
                # )
                pass

        # ------------------------------------------------------------
        # A) search/open eager protocol:
        # 立即注入 tool response，然后注入 budget
        # ------------------------------------------------------------
        if search_open_payloads:
            combined = "\n\n".join(search_open_payloads)

            # 1) 立即注入 tool response
            cid = await self._inject_tool_obs_commit(agent_data, combined)
            print(
                f"[D{agent_data.branch_id}-{depth}][TOOLS] "
                f"search/open -> inject tool_obs commit={cid}",
                flush=True,
            )

            # 2) 立即注入 budget prompt，并开启 guard
            await self._inject_budget_prompt(agent_data)

            return AgentState.GENERATING, None

        # ------------------------------------------------------------
        # B) summarize => fold registry + clear budget guard
        # ------------------------------------------------------------
        if summarize_call is not None:
            try:
                args = json.loads(summarize_call.arguments)
                if not isinstance(args, dict):
                    raise ValueError

                fold_commit_ids = args.get("fold_commit_ids", "NONE")
                if not isinstance(fold_commit_ids, str):
                    raise ValueError

                merged_commit = args.get("merged_commit", "")
                merged_digest = args.get("merged_digest", "")
            except Exception:
                return AgentState.TERMINATED, None

            # summarize 一旦发生，说明 budget 已被“消费”，解除限制
            agent_data.budget_injected_waiting_summarize = False

            if fold_commit_ids.upper() == "NONE":
                print(
                    f"[D{agent_data.branch_id}-{depth}][SUM] "
                    f"fold=NONE -> continue generating",
                    flush=True,
                )
                return AgentState.GENERATING, None

            next_delta = agent_data.commit_registry.apply_fold(
                fold_commit_ids=fold_commit_ids,
                merged_commit=merged_commit,
                merged_digest=merged_digest,
                pending_tool_payload_text="",  # eager 版本没有 pending
            )

            print(
                f"[D{agent_data.branch_id}-{depth}][SUM] "
                f"fold_commit_ids={fold_commit_ids} -> next_delta_len={len(next_delta)}",
                flush=True,
            )
            print(
                f"[DEBUG][AFTER_FOLD] commits_after={len(agent_data.commit_registry.items())}",
                flush=True,
            )
            return AgentState.TERMINATED, next_delta

        # 没有 search/open，也没有 summarize，继续生成
        return AgentState.GENERATING, None
