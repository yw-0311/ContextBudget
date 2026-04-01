# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import os
import logging
from typing import Any, Optional
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SummaryTool(BaseTool):
    """A no-op summary tool.

    This tool is intentionally a shell: it does NOT call any external service.
    It's mainly for framework compatibility (instance lifecycle, tracing, reward hooks).

    If `execute` is called, it can optionally echo back provided text (e.g., parameters["text"])
    so downstream parsers can still extract fields from a well-formed tool response.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

        # Optional behavior knobs
        # - echo_param_keys: try these keys in order to find text to echo back
        self.echo_param_keys = config.get("echo_param_keys", ["text", "summary", "content", "output"])
        # - default_text: returned when no echo-able text is provided
        self.default_text = config.get("default_text", "")

        logger.info(f"Initialized SummaryTool (no-op) with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "outputs": [],
            "reward": [],
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        """No-op execute.

        - If parameters contain a text field, echo it back.
        - Otherwise return default_text (usually empty).
        """
        if instance_id not in self._instance_dict:
            # Be defensive: create implicit state if caller forgot create()
            self._instance_dict[instance_id] = {"outputs": [], "reward": []}

        text = None
        if isinstance(parameters, dict):
            for k in self.echo_param_keys:
                v = parameters.get(k)
                if isinstance(v, str) and v.strip() != "":
                    text = v
                    break

        if text is None:
            text = self.default_text

        self._instance_dict[instance_id]["outputs"].append(text)
        # If you want to use this tool to contribute to "reward history", keep it mirrored
        self._instance_dict[instance_id]["reward"].append(text)

        metrics = {
            "status": "noop",
            "echoed": text != self.default_text,
            "text_len": len(text),
        }
        return ToolResponse(text=text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> Any:
        # Keep the same pattern as your SearchTool: return stored reward list
        return self._instance_dict.get(instance_id, {}).get("reward", [])

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
