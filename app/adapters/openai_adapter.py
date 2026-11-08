from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.adapters.base import ChatMessage, LLMAdapter
from app.config.settings import SYSTEM_PROMPT, ModelConfig, get_api_key


class OpenAIAdapter(LLMAdapter):
    """OpenAI Chat Completions 协议适配器，适用于 DeepSeek、MiniMax 等兼容实现。"""

    def __init__(self, model_id: str, config: ModelConfig):
        self._model = config.model
        self._api_key = get_api_key(model_id)
        self._base_url = config.base_url
        # 默认（text）场景复用同一个 Agent，避免每次请求重复构建
        self._agent = self._build_agent()

    def _build_agent(self, response_format: dict[str, Any] | None = None):
        model = ChatOpenAI(
            model=self._model,
            api_key=self._api_key,
            base_url=self._base_url,
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
    ) -> str:
        # 结构化约束是请求级参数，需要按本次请求重建 Agent
        agent = self._agent if response_format is None else self._build_agent(
            response_format
        )
        response = await agent.ainvoke({"messages": self._to_pairs(messages)})
        return response["messages"][-1].content

    async def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        async for chunk, _metadata in self._agent.astream(
            {"messages": self._to_pairs(messages)},
            stream_mode="messages",
        ):
            content = getattr(chunk, "content", "")
            if isinstance(content, str) and content:
                yield content

    @staticmethod
    def _to_pairs(messages: Sequence[ChatMessage]) -> list[tuple[str, str]]:
        # 系统提示词由 Agent 统一注入，这里只转换对话消息
        return [(m.role, m.content) for m in messages if m.role != "system"]
