# Copyright 2025 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

import asyncio
import json
import logging
from typing import Any, Optional, List

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop_base import (
    ToolAgentLoopBase,
    AgentState,
    AgentData,
    CommitRegistry,
    BUDGET_TEMPLATE_EN,
    FOLD_POLICY_EN,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)


@register("tool_agent_wbudget")
class ToolAgentLoop(ToolAgentLoopBase):
    """
    Tool agent loop with standard and flat commit handling.
    Supports both standard tool response processing and deferred protocol.
    """

    def _make_budget_context(self, agent_data: AgentData, tool_text: str) -> str:
        # 允许不开 budget 时也有一个固定提示（和 my_tool_agent 对齐）
        if not tool_text:
            return ""

        # 当前上下文长度（注意：此时还没把本轮 tool message 编进 prompt_ids）
        current_ctx_len = len(agent_data.prompt_ids)

        # 估算将要注入的 tool 文本 token 数（这里建议把“前缀+正文”一起算；但为了生成前缀，先算正文）
        tool_response_len = len(self.tokenizer.encode(tool_text, add_special_tokens=False))

        projected_ctx_len = current_ctx_len + tool_response_len
        usable_limit = self.max_model_len - 1000

        remaining_budget = usable_limit - projected_ctx_len
        remaining_pct = max(0.0, remaining_budget / usable_limit * 100.0) if usable_limit > 0 else 0.0

        budget_text=""
        if getattr(self, "enable_budget", True):
            budget_text = BUDGET_TEMPLATE_EN.format(
                current_ctx_len=current_ctx_len,
                tool_response_len=tool_response_len,
                remaining_budget=remaining_budget,
                remaining_pct=remaining_pct,
                usable_limit=usable_limit,
                fold_policy=""
            )

        # 作为“前缀”加到 tool content 前面。建议加个分隔，避免和 tool 内容粘连
        return "[HINT:" + budget_text.rstrip() + "]\n\n"


    async def _handle_processing_tools_state(self, agent_data: AgentData) -> tuple[AgentState, Optional[list]]:
        logger.info(f"[TOOL_AGENT][STANDARD] Processing tool responses in standard mode")
        
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))
            tool_call_names.append(tool_call.name)

        logger.info(f"[TOOL_AGENT][STANDARD] Executing {len(tasks)} tool calls: {tool_call_names}")

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        logger.info(f"[TOOL_AGENT][STANDARD] Tool execution completed, processing responses")

        # Process tool responses and update multi_modal_data
        for idx, (tool_response, tool_reward, _) in enumerate(responses):
            logger.info(f"[TOOL_AGENT][STANDARD] Processing response for tool {tool_call_names[idx]}, "
                       f"has_image={bool(tool_response.image)}, has_video={bool(tool_response.video)}, "
                       f"text_len={len(tool_response.text) if tool_response.text else 0}")
            
            budget_text = self._make_budget_context(agent_data,tool_response.text)
            message = {"role": "tool", "content": budget_text + tool_response.text or ""}
            add_messages.append(message)
            agent_data.messages.extend([message])

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        # Update prompt with tool responses
        logger.info(f"[TOOL_AGENT][STANDARD] Encoding {len(add_messages)} tool response messages")
        
        response_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
        )
        response_ids = response_ids[len(self.system_prompt) :]
        logger.info(f"[TOOL_AGENT][STANDARD] Tool response encoded: {len(response_ids)} tokens")

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        
        logger.info(f"[TOOL_AGENT][STANDARD] Tool processing complete, continuing to GENERATING")
        return AgentState.GENERATING, None