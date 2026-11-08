import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from app.adapters.base import (
    ChatMessage,
    Completion,
    LLMAdapter,
    StreamChunk,
    TokenUsage,
    build_json_instruction,
)
from app.config.model_config import ModelConfig
from app.config.settings import SYSTEM_PROMPT, get_api_key

ANTHROPIC_VERSION = "2023-06-01"
MAX_TOKENS = 4096
TIMEOUT_SECONDS = 60


class AnthropicAdapter(LLMAdapter):
    """Anthropic Messages 协议适配器：x-api-key 鉴权、system 顶层字段、content 块返回。"""

    def __init__(self, model_id: str, config: ModelConfig):
        self._provider_model = config.provider_model  # 上游实际模型名称
        self._base_url = config.base_url.rstrip("/")  # 去除尾部斜杠的上游接口地址
        self._api_key = get_api_key(model_id)  # 上游鉴权 Key

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        stream: bool,
        response_format: dict[str, Any] | None = None,
    ) -> dict:
        system_parts = [
            SYSTEM_PROMPT,
            *[m.content for m in messages if m.role == "system"],
        ]
        # Anthropic 无原生 response_format，用系统提示词把 JSON 约束传递给模型
        if response_format is not None:
            system_parts.append(build_json_instruction(response_format))
        system = "\n\n".join(system_parts)
        return {
            "model": self._provider_model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "stream": stream,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.role != "system"
            ],
        }

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> Completion:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{self._base_url}/v1/messages",
                headers=self._headers(),
                json=self._payload(
                    messages, stream=False, response_format=response_format
                ),
            )
            response.raise_for_status()
            data = response.json()
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        return Completion(
            text=text, usage=self._usage_from_payload(data.get("usage"))
        )

    async def stream(
        self, messages: Sequence[ChatMessage]
    ) -> AsyncIterator[StreamChunk]:
        # 1. 记录 message_start 返回的输入侧消耗，等 message_delta 补全输出侧
        prompt_tokens = 0
        cached_tokens = 0
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/messages",
                headers=self._headers(),
                json=self._payload(messages, stream=True),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    event = json.loads(data)
                    event_type = event.get("type")
                    if event_type == "message_start":
                        usage = event.get("message", {}).get("usage") or {}
                        prompt_tokens = int(usage.get("input_tokens") or 0)
                        cached_tokens = int(
                            usage.get("cache_read_input_tokens") or 0
                        )
                    elif event_type == "content_block_delta":
                        text = event.get("delta", {}).get("text")
                        if text:
                            yield StreamChunk(text=text)
                    elif event_type == "message_delta":
                        # 2. 结束事件携带累计输出 Token，补齐完整 usage
                        output_tokens = int(
                            (event.get("usage") or {}).get("output_tokens") or 0
                        )
                        yield StreamChunk(
                            usage=TokenUsage(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=output_tokens,
                                total_tokens=prompt_tokens + output_tokens,
                                cached_tokens=cached_tokens,
                            )
                        )

    @staticmethod
    def _usage_from_payload(usage: dict[str, Any] | None) -> TokenUsage | None:
        """把 Anthropic 的 usage 字段转成统一的分类统计。"""
        if not usage:
            return None
        prompt = int(usage.get("input_tokens") or 0)
        completion = int(usage.get("output_tokens") or 0)
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            cached_tokens=int(usage.get("cache_read_input_tokens") or 0),
        )
