from collections.abc import AsyncIterator

from app.adapters.llm import agent, model


async def complete(message: str) -> str:
    """一次性返回完整回答"""
    response = await agent.ainvoke({"messages": [("user", message)]})
    return response["messages"][-1].content


async def stream_reply(message: str) -> AsyncIterator[str]:
    """流式产出回答文本片段"""
    async for chunk in model.astream([{"role": "user", "content": message}]):
        content = getattr(chunk, "content", "")
        if content:
            yield content
