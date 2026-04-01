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


@register("tool_agent")
class ToolAgentLoop(ToolAgentLoopBase):
    """
    Tool agent loop with standard and flat commit handling.
    Supports both standard tool response processing and deferred protocol.
    """

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
            
            message = {"role": "tool", "content": tool_response.text or ""}
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