from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.adapters.base import (
    ChatMessage,
    Completion,
    LLMAdapter,
    StreamChunk,
    TokenUsage,
)
from app.config.settings import SYSTEM_PROMPT, ModelConfig, get_api_key


class OpenAIAdapter(LLMAdapter):
    """OpenAI Chat Completions 协议适配器，适用于 DeepSeek、MiniMax 等兼容实现。"""

    def __init__(self, model_id: str, config: ModelConfig):
        self._model = config.model  # 上游实际模型名称
        self._api_key = get_api_key(model_id)  # 上游鉴权 Key
        self._base_url = config.base_url  # 上游接口地址
        # 默认（text）场景复用同一个 Agent，避免每次请求重复构建
        self._agent = self._build_agent()  # 复用的默认 Agent 实例

    def _build_agent(self, response_format: dict[str, Any] | None = None):
        model = ChatOpenAI(
            model=self._model,
            api_key=self._api_key,
            base_url=self._base_url,
            # 流式调用也要求上游返回 usage，才能统计 Token 消耗
            stream_usage=True,
            # response_format 属于 OpenAI 请求体字段，通过 extra_body 原样下传
            extra_body=(
                {"response_format": response_format} if response_format else None
            ),
        )
        return create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
        )

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> Completion:
        # 结构化约束是请求级参数，需要按本次请求重建 Agent
        agent = self._agent if response_format is None else self._build_agent(
            response_format
        )
        response = await agent.ainvoke({"messages": self._to_pairs(messages)})
        message = response["messages"][-1]
        return Completion(
            text=message.content,
            usage=self._usage_from_metadata(getattr(message, "usage_metadata", None)),
        )

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamChunk]:
        async for chunk, _metadata in self._agent.astream(
            {"messages": self._to_pairs(messages)},
            stream_mode="messages",
        ):
            content = getattr(chunk, "content", "")
            text = content if isinstance(content, str) else ""
            usage = self._usage_from_metadata(getattr(chunk, "usage_metadata", None))
            # 仅含 usage、无文本的结束分片也要产出，供上层统计
            if text or usage is not None:
                yield StreamChunk(text=text, usage=usage)

    @staticmethod
    def _usage_from_metadata(metadata: dict | None) -> TokenUsage | None:
        """把 LangChain 的 usage_metadata 转成统一的分类统计。"""
        if not metadata:
            return None
        prompt = int(metadata.get("input_tokens") or 0)
        completion = int(metadata.get("output_tokens") or 0)
        total = int(metadata.get("total_tokens") or (prompt + completion))
        input_details = metadata.get("input_token_details") or {}
        output_details = metadata.get("output_token_details") or {}
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cached_tokens=int(input_details.get("cache_read") or 0),
            reasoning_tokens=int(output_details.get("reasoning") or 0),
        )

    @staticmethod
    def _to_pairs(messages: Sequence[ChatMessage]) -> list[tuple[str, str]]:
        # 系统提示词由 Agent 统一注入，这里只转换对话消息
        return [(m.role, m.content) for m in messages if m.role != "system"]
