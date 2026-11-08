from collections.abc import AsyncIterator, Sequence

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.adapters.base import ChatMessage, LLMAdapter
from app.config.settings import SYSTEM_PROMPT, ModelConfig, get_api_key


class OpenAIAdapter(LLMAdapter):
    """OpenAI Chat Completions 协议适配器，适用于 DeepSeek、MiniMax 等兼容实现。"""

    def __init__(self, model_id: str, config: ModelConfig):
        model = ChatOpenAI(
            model=config.model,
            api_key=get_api_key(model_id),
            base_url=config.base_url,
        )
        self._agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
        )

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        response = await self._agent.ainvoke(
            {"messages": self._to_pairs(messages)}
        )
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
