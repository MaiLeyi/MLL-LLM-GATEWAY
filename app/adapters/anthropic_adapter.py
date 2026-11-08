import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from app.adapters.base import ChatMessage, LLMAdapter
from app.config.settings import SYSTEM_PROMPT, ModelConfig, get_api_key

ANTHROPIC_VERSION = "2023-06-01"
MAX_TOKENS = 4096
TIMEOUT_SECONDS = 60


class AnthropicAdapter(LLMAdapter):
    """Anthropic Messages 协议适配器：x-api-key 鉴权、system 顶层字段、content 块返回。"""

    def __init__(self, model_id: str, config: ModelConfig):
        self._model = config.model
        self._base_url = config.base_url.rstrip("/")
        self._api_key = get_api_key(model_id)

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
            system_parts.append(self._json_instruction(response_format))
        system = "\n\n".join(system_parts)
        return {
            "model": self._model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "stream": stream,
            "messages": [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.role != "system"
            ],
        }

    @staticmethod
    def _json_instruction(response_format: dict[str, Any]) -> str:
        """把 OpenAI 的 response_format 转成自然语言约束。"""
        if response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema", {})
            return (
                "请只返回符合以下 JSON Schema 的合法 JSON，"
                "不要输出多余文字或 Markdown 代码块：\n"
                + json.dumps(schema, ensure_ascii=False)
            )
        return "请只返回合法 JSON，不要输出多余文字或 Markdown 代码块。"

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> str:
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
        return "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
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
                    if event.get("type") == "content_block_delta":
                        text = event.get("delta", {}).get("text")
                        if text:
                            yield text
