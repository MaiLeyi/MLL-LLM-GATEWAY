from collections.abc import AsyncIterator

from app.adapters.base import ChatMessage
from app.adapters.llm import get_adapter


async def complete(message: str, model_id: str) -> str:
    """一次性返回完整回答"""
    return await get_adapter(model_id).complete(
        [ChatMessage(role="user", content=message)]
    )


async def stream_reply(message: str, model_id: str) -> AsyncIterator[str]:
    """流式产出回答文本片段"""
    async for content in get_adapter(model_id).stream(
        [ChatMessage(role="user", content=message)]
    ):
        yield content
