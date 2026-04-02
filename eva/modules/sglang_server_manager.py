from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import requests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import requests.exceptions


@dataclass
class TokenOutput:
    """
    Minimal structure compatible with verl server_manager.generate() return usage:
      output.token_ids
      output.log_probs (optional)
    """
    token_ids: List[int]
    log_probs: Optional[List[float]] = None
    routed_experts: Any = None


class SGLangServerManager:
    """
    Adapter used by verl AgentLoop to call SGLang Runtime:
      POST {sglang_url}/generate
    Payload:
      { "input_ids": [...], "sampling_params": {...}, "rid": "...", "stream": false }
    """

    def __init__(self, sglang_url: str, timeout_s: float = 120.0, max_retries: int = 5, retry_delay: int = 20):
        self.base = sglang_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5), retry=retry_if_exception_type(requests.exceptions.Timeout))
    async def generate(
        self,
        request_id: str,
        prompt_ids: List[int],
        sampling_params: Dict[str, Any],
        image_data: Any = None,
    ) -> TokenOutput:
        url = f"{self.base}/generate"
        payload = {
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "rid": request_id,
            "stream": False,
        }

        # NOTE: image_data ignored in this minimal adapter.

        def _post():
            r = requests.post(url, json=payload, timeout=self.timeout_s)
            r.raise_for_status()  # Raise an exception for HTTP errors
            return r.json()

        try:
            j = await asyncio.get_running_loop().run_in_executor(None, _post)
        except requests.exceptions.RequestException as e:
            # Capture request exception and print, but don't throw
            print(f"Request failed: {e}")
            return TokenOutput(token_ids=[], log_probs=None)

        out_ids = list(j.get("output_ids", []) or [])

        # If SGLang runtime returns logprobs, can uncomment and use
        # meta = j.get("meta_info", {}) or {}
        # log_probs = meta.get("token_logprobs", None)
        log_probs = None

        return TokenOutput(token_ids=out_ids, log_probs=log_probs)
