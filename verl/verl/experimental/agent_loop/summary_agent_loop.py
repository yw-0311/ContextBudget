# Copyright 2025 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

from typing import Any, Optional, List
from copy import deepcopy
import asyncio
import json
import re

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop_base import (
    ToolAgentLoopBase,
    AgentState,
    AgentData,
)
from verl.utils.profiler import simple_timer


@register("summary_tool_agent")
class ToolAgentLoop(ToolAgentLoopBase):
    SUMMARY_TRIGGER_RATIO = 0.70
    SUMMARY_MAX_TOKENS = 2096

    SUMMARY_PROMPT_TEMPLATE = """You are an expert at analyzing conversation history and extracting relevant information. Your task is
to thoroughly evaluate the conversation history and current question to provide a comprehensive
summary that will help answer the question.

Task Guidelines:
1. Information Analysis
• Carefully analyze the conversation history to identify truly useful information.
• Focus on information that directly contributes to answering the question.
• Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated in
the conversation.
• If information is missing or unclear, do NOT include it in your summary.

2. Summary Requirements
• Extract only the most relevant information that is explicitly present in the conversation.
• Synthesize information from multiple exchanges when relevant.
• Only include information that is certain and clearly stated in the conversation.
• Do NOT output or mention any information that is uncertain, insufficient, or cannot be
confirmed from the conversation.

3. Output Format
Your response should be structured as follows:
<summary>
• Essential Information: [Organize the relevant and certain information from the conversation history that helps address the question.]
</summary>

Strictly avoid fabricating, inferring, or exaggerating any information not present in the conversation.
Only output information that is certain and explicitly stated.

Question {Question}

Conversation {Conversation_History}

Please generate a comprehensive and useful summary"""

    def _usable_limit(self) -> int:
        return max(1, self.max_model_len - 1000)

    def _context_usage_ratio(self, agent_data: AgentData) -> tuple[float, int, int]:
        current_ctx_len = len(agent_data.prompt_ids)
        usable_limit = self._usable_limit()
        ratio = current_ctx_len / usable_limit
        return ratio, current_ctx_len, usable_limit

    def _normalize_content_for_summary(self, content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type in ("text", "input_text", "output_text"):
                        text = item.get("text") or item.get("content") or ""
                        if text:
                            parts.append(str(text))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "\n".join([p for p in parts if p])

        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)

        return str(content)

    def _extract_latest_user_question(self, agent_data: AgentData) -> str:
        messages = getattr(agent_data, "messages", None) or []
        for msg in reversed(messages):
            if msg.get("role") == "user":
                text = self._normalize_content_for_summary(msg.get("content", ""))
                if text.strip():
                    return text.strip()
        return ""

    def _build_conversation_history_for_summary(self, agent_data: AgentData) -> str:
        messages = getattr(agent_data, "messages", None) or []
        parts: List[str] = []

        for msg in messages:
            role = msg.get("role", "unknown")
            if role == "system":
                continue

            content = self._normalize_content_for_summary(msg.get("content", ""))
            if not content.strip():
                continue

            parts.append(f"[{role}]\n{content}")

        return "\n\n".join(parts).strip()

    def _build_summary_prompt(self, agent_data: AgentData) -> str:
        question = self._extract_latest_user_question(agent_data)
        conversation_history = self._build_conversation_history_for_summary(agent_data)

        return self.SUMMARY_PROMPT_TEMPLATE.format(
            Question=question,
            Conversation_History=conversation_history,
        )

    def _extract_summary_block(self, text: str) -> str:
        if not text:
            return ""

        m = re.search(r"<summary>(.*?)</summary>", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return f"<summary>{m.group(1).strip()}</summary>"

        return text.strip()

    def _build_summary_sampling_params(self, agent_data: AgentData) -> dict[str, Any]:
        base_params = deepcopy(getattr(agent_data, "_latest_sampling_params", {}) or {})

        if "max_tokens" in base_params:
            base_params["max_tokens"] = min(int(base_params["max_tokens"]), self.SUMMARY_MAX_TOKENS)
        elif "max_new_tokens" in base_params:
            base_params["max_new_tokens"] = min(int(base_params["max_new_tokens"]), self.SUMMARY_MAX_TOKENS)
        else:
            base_params["max_tokens"] = self.SUMMARY_MAX_TOKENS

        if "temperature" in base_params:
            base_params["temperature"] = 0.0
        if "top_p" in base_params:
            base_params["top_p"] = 1.0

        return base_params

    async def _generate_summary_text(self, agent_data: AgentData, prompt: str) -> str:
        summary_messages = [{"role": "user", "content": prompt}]

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                summary_messages,
                tools=None,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )

        summary_sampling_params = self._build_summary_sampling_params(agent_data)

        output = await self.server_manager.generate(
            request_id=f"{agent_data.request_id}_summary",
            prompt_ids=prompt_ids,
            sampling_params=summary_sampling_params,
            image_data=None,
        )

        summary_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        return self._extract_summary_block(summary_text)

    def _build_next_delta_messages(self, summary_text: str) -> list[dict]:
        # 只返回摘要内容本身即可。
        # 基类 run()->dfs()->merge_delta_into_user() 只会读取 content 并拼回下一轮 user message。
        return [
            {
                "role": "tool",
                "content": summary_text,
            }
        ]

    async def _self_summary_and_restart(self, agent_data: AgentData) -> list[dict]:
        summary_prompt = self._build_summary_prompt(agent_data)
        summary_text = (await self._generate_summary_text(agent_data, summary_prompt)).strip()

        if not summary_text:
            summary_text = "<summary>\n• Essential Information: (empty summary returned by model)\n</summary>"

        next_delta = self._build_next_delta_messages(summary_text)

        print(
            f"[D{agent_data.branch_id}-{agent_data.branch_depth}][SUM] "
            f"self-summary triggered -> next_delta_len={len(next_delta)}",
            flush=True,
        )
        print(
            f"[D{agent_data.branch_id}-{agent_data.branch_depth}][SUM] "
            f"summary_preview={summary_text[:500]}",
            flush=True,
        )
        return next_delta

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> AgentState:
        # 这里只做一件事：缓存当前 sampling_params，供 tool 后触发 summary 时复用
        agent_data._latest_sampling_params = deepcopy(sampling_params)
        return await super()._handle_generating_state(agent_data, sampling_params)

    async def _handle_processing_tools_state(
        self, agent_data: AgentData
    ) -> tuple[AgentState, Optional[list]]:
        depth = agent_data.branch_depth
        tool_calls_all = agent_data.tool_calls[: self.max_parallel_calls]

        responses = []
        if tool_calls_all:
            tasks = [self._call_tool(tc, agent_data.tools_kwargs) for tc in tool_calls_all]
            with simple_timer("tool_calls", agent_data.metrics):
                responses = await asyncio.gather(*tasks)

        tool_messages: List[dict] = []

        for tc, (tool_response, tool_reward, _) in zip(tool_calls_all, responses):
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

            if not tool_response or not getattr(tool_response, "text", None):
                continue

            tool_messages.append(
                {
                    "role": "tool",
                    "content": tool_response.text,
                }
            )

            print(
                f"[D{agent_data.branch_id}-{depth}][TOOLS] "
                f"{tc.name} DIRECT -> append tool message",
                flush=True,
            )

        if not tool_messages:
            return AgentState.GENERATING, None

        # 1) 先把 tool response 正常放入当前轮上下文
        await self._append_messages_and_encode_tail(agent_data, tool_messages)

        # 2) append 完之后检查上下文占用
        ratio, current_ctx_len, usable_limit = self._context_usage_ratio(agent_data)
        print(
            f"[D{agent_data.branch_id}-{depth}][POST_TOOLS_CTX] "
            f"ratio={ratio:.2%}, current_ctx_len={current_ctx_len}, usable_limit={usable_limit}",
            flush=True,
        )

        # 3) 如果达到阈值，则在这里直接 summary 并结束当前轮
        #    这样 _run_single_branch 会把 next_delta 传给 dfs，触发新一轮，depth++
        if ratio >= self.SUMMARY_TRIGGER_RATIO:
            next_delta = await self._self_summary_and_restart(agent_data)
            print(
                f"[D{agent_data.branch_id}-{depth}][POST_TOOLS_CTX] "
                f"trigger summary and recurse next round",
                flush=True,
            )
            return AgentState.TERMINATED, next_delta

        # 4) 否则继续当前轮生成
        return AgentState.GENERATING, None
