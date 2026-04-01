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
    WO_BUDGET_TEMPLATE_EN,
    FOLD_POLICY_EN,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer

import asyncio
import json

import re

@register("my_tool_agent")
class ToolAgentLoop(ToolAgentLoopBase):
    """
    Tool agent loop with flat commit handling.
    Uses deferred tool response protocol with commit registry.
    """

    def _next_cid(self, reg: CommitRegistry) -> str:
            """Generate next stable cid based on current registry ids."""
            max_n = 0
            for cid in reg.ids():
                m = re.match(r"c(\d{4,})$", cid or "")
                if m:
                    max_n = max(max_n, int(m.group(1)))
            return f"c{max_n + 1:04d}"

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
        agent_data.commit_registry.add_commit(kind=kind, content=content, digest=digest, cid=cid)

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

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> tuple[AgentState, Optional[list]]:
        depth = agent_data.branch_depth

        tool_calls_all = agent_data.tool_calls[: self.max_parallel_calls]
        summarize_call = next((tc for tc in tool_calls_all if tc.name == "summarize"), None)
        exec_calls = [tc for tc in tool_calls_all if tc.name != "summarize"]

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
                # 添加直接失败，在search前必须完成上一次search tool response的commit
                if agent_data.pending_tool_payload_text:
                    return AgentState.TERMINATED, None
                search_open_payloads.append(tool_response.text)
            else:
                # Optional: keep other tool outputs
                # await self._append_messages_and_encode_tail(agent_data, [{"role": "tool", "content": tool_response.text}])
                pass

        # A) search/open deferred protocol
        if search_open_payloads:
            combined = "\n\n".join(search_open_payloads)
            agent_data.pending_tool_payload_text = combined

            # FIRST tool result: directly inject as c0001 and also store in registry
            if agent_data.commit_registry.is_empty():
                agent_data.pending_tool_payload_text = ""
                cid = await self._inject_tool_obs_commit(agent_data, combined, cid="c0001")
                print(f"[D{agent_data.branch_id}-{depth}][TOOLS] search/open FIRST -> inject tool_obs commit={cid}", flush=True)
                return AgentState.GENERATING, None

            # LATER tool result: inject budget message only; keep pending for summarize
            current_ctx_len = len(agent_data.prompt_ids)
            tool_response_len = len(self.tokenizer.encode(agent_data.pending_tool_payload_text, add_special_tokens=False))
            projected_ctx_len = current_ctx_len + tool_response_len

            usable_limit = self.max_model_len - 1000
            remaining_budget = usable_limit - projected_ctx_len
            remaining_pct = max(0.0, remaining_budget / usable_limit * 100.0) if usable_limit > 0 else 0.0

            budget_text = ""
            if self.enable_budget:
                budget_text = BUDGET_TEMPLATE_EN.format(
                    current_ctx_len=current_ctx_len,
                    tool_response_len=tool_response_len,
                    remaining_budget=remaining_budget,
                    remaining_pct=remaining_pct,
                    usable_limit=usable_limit,
                    fold_policy=FOLD_POLICY_EN,
                )
            else:
                budget_text = WO_BUDGET_TEMPLATE_EN

            budget_msg = {"role": "tool", "content": budget_text}
            print(
                f"[D{agent_data.branch_id}-{depth}][TOOLS] search/open LATER -> inject budget, "
                f"pending_tool_response_len={tool_response_len}, remaining={remaining_budget}",
                f"budget_info {budget_text}",
                flush=True,
            )
            await self._append_messages_and_encode_tail(agent_data, [budget_msg])
            return AgentState.GENERATING, None

        # B) summarize => fold registry + return NEXT DELTA MESSAGES
        if summarize_call is not None:
            try:
                args = json.loads(summarize_call.arguments)
                if not isinstance(args, dict): raise ValueError
                fold_commit_ids = args.get("fold_commit_ids", "NONE")
                if not isinstance(fold_commit_ids, str): raise ValueError
                merged_commit = args.get("merged_commit", "")
                merged_digest = args.get("merged_digest", "")
            except Exception:
                    return AgentState.TERMINATED, None
            
            if fold_commit_ids.upper() == "NONE":
                if agent_data.pending_tool_payload_text:
                    combined = agent_data.pending_tool_payload_text
                    agent_data.pending_tool_payload_text = ""
                    cid = await self._inject_tool_obs_commit(agent_data, combined)  # 自动 next cid
                    print(f"[D{agent_data.branch_id}-{depth}][SUM] fold=NONE -> inject tool_obs commit={cid}", flush=True)
                return AgentState.GENERATING, None

            next_delta = agent_data.commit_registry.apply_fold(
                fold_commit_ids=fold_commit_ids,
                merged_commit=merged_commit,
                merged_digest=merged_digest,
                pending_tool_payload_text=agent_data.pending_tool_payload_text,
            )
            agent_data.pending_tool_payload_text = ""  # consumed by fold
            
            print(f"[D{agent_data.branch_id}-{depth}][SUM] fold_commit_ids={fold_commit_ids} -> next_delta_len={len(next_delta)}", flush=True)
            print(f"[DEBUG][AFTER_FOLD] commits_after={len(agent_data.commit_registry.items())}", flush=True)
            return AgentState.TERMINATED, next_delta

        return AgentState.GENERATING, None