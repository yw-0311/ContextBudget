# Copyright 2025 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

import asyncio
import copy
import json
import logging
import re
from typing import Any, Optional, List, Literal

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop_base import (
    ToolAgentLoopBase,
    AgentState,
    AgentData,
    CommitRegistry,
)
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)

# Regex patterns for Mem1 tags
_INTERNAL_STATE_RE = re.compile(r'<internal_state>(.*?)</internal_state>', re.DOTALL)


@register("mem1_agent")
class Mem1AgentLoop(ToolAgentLoopBase):
    """
    Mem1 agent loop implementation with separated memory.
    
    Key features:
    1. Separates initial prompt from conversation history
    2. Compresses context by removing previous turns before each generation
    3. Only keeps latest assistant response in cur_obs
    4. Uses basic tool call processing (reuses parent logic)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_type: Literal["normal", "amem", "mem1"] = kwargs.get("inference_type", "mem1")
        self.max_iteration = kwargs.get("max_iteration", 6)
        
        # Mem1-specific state
        self.cur_obs = ""  # Current observation (only stores responses)
        self.initial_prompt = ""  # Store initial prompt separately
        self.memory_system = kwargs.get("memory_system", None)
        self.turn_count = 0
        
        logger.info(f"[MEM1_INIT] inference_type={self.inference_type}, max_iteration={self.max_iteration}")

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> tuple[AgentState, Optional[list]]:
        """
        Handle tool processing using basic standard mode (same as tool_agent_loop).
        """
        logger.info(f"[MEM1_AGENT] Processing tool responses")
        
        add_messages: list[dict[str, Any]] = []

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))
            tool_call_names.append(tool_call.name)

        logger.info(f"[MEM1_AGENT] Executing {len(tasks)} tool calls: {tool_call_names}")

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        logger.info(f"[MEM1_AGENT] Tool execution completed, processing responses")

        # Process tool responses
        for idx, (tool_response, tool_reward, _) in enumerate(responses):
            logger.info(f"[MEM1_AGENT] Processing response for tool {tool_call_names[idx]}")
            
            message = {"role": "tool", "content": tool_response.text or ""}
            add_messages.append(message)
            agent_data.messages.extend([message])

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        # Update prompt with tool responses
        logger.info(f"[MEM1_AGENT] Encoding {len(add_messages)} tool response messages")
        
        response_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
        )
        response_ids = response_ids[len(self.system_prompt) :]
        logger.info(f"[MEM1_AGENT] Tool response encoded: {len(response_ids)} tokens")

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        
        # Update cur_obs with the latest assistant response
        if agent_data.messages:
            for msg in reversed(agent_data.messages):
                if msg.get("role") == "assistant":
                    self.cur_obs = msg.get("content", "")
                    break
        
        self.turn_count += 1
        logger.info(f"[MEM1_AGENT] Turn count: {self.turn_count}/{self.max_iteration}")
        
        # Check max iteration
        if self.turn_count >= self.max_iteration:
            logger.info(f"[MEM1_AGENT] Max iteration reached, terminating")
            return AgentState.TERMINATED, None
        
        logger.info(f"[MEM1_AGENT] Tool processing complete, continuing to GENERATING")
        return AgentState.GENERATING, None

    async def _handle_generating_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """
        Override generating state to implement Mem1's context compression.
        
        For Mem1 mode:
        1. Extract initial prompt from first user message
        2. Clear previous context from prompt_ids
        3. Rebuild prompt with only: [initial_prompt, cur_obs]
        """
        depth = agent_data.branch_depth
        
        # Store initial prompt if not already stored
        if not self.initial_prompt and len(agent_data.messages) >= 1:
            # Extract the initial user question from messages
            for msg in agent_data.messages:
                if msg.get("role") == "user":
                    self.initial_prompt = msg.get("content", "")
                    break
            logger.info(f"[MEM1_AGENT] Initial prompt stored: {self._preview(self.initial_prompt, 200)}")
        
        # For mem1 mode, clear previous context and rebuild with separated structure
        if self.inference_type == "mem1" and self.initial_prompt:
            logger.info(f"[MEM1_AGENT] Rebuilding prompt with separated structure")
            logger.info(f"[MEM1_AGENT] cur_obs length: {len(self.cur_obs)}")
            
            # Build messages with separated structure
            # Only keep: [user: initial_prompt, assistant: cur_obs]
            mem1_messages = [
                {"role": "user", "content": self.initial_prompt},
            ]
            
            # Add assistant response if cur_obs exists
            if self.cur_obs:
                mem1_messages.append({"role": "assistant", "content": self.cur_obs})
            
            # Rebuild prompt_ids from scratch (clears previous context)
            new_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    mem1_messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            
            # Replace prompt_ids with mem1-style encoding (clears previous context)
            agent_data.prompt_ids = new_prompt_ids
            logger.info(f"[MEM1_AGENT] Rebuilt prompt_ids: {len(agent_data.prompt_ids)} tokens (context compressed)")
        elif self.inference_type == "amem":
            # For amem mode, we can use memory compression
            logger.info(f"[MEM1_AGENT] Using amem mode with memory compression")
        
        # Call parent implementation for actual generation
        return await super()._handle_generating_state(agent_data, sampling_params)

    def _extract_internal_state(self, response: str) -> Optional[str]:
        """Extract internal state from response."""
        match = _INTERNAL_STATE_RE.search(response)
        if match:
            state = match.group(1).strip()
            return f"<internal_state>{state}</internal_state>"
        return None

    def _preview(self, s: str, n: int = 200) -> str:
        """Preview string with truncation."""
        if not s:
            return ""
        s = s.replace("\n", "\\n")
        return s[:n] + ("..." if len(s) > n else "")
