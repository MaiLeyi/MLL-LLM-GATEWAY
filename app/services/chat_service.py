from collections.abc import AsyncIterator

from app.adapters.llm import get_agent


async def complete(message: str, model_id: str) -> str:
    """一次性返回完整回答"""
    response = await get_agent(model_id).ainvoke({"messages": [("user", message)]})
    return response["messages"][-1].content


async def stream_reply(message: str, model_id: str) -> AsyncIterator[str]:
    """流式产出回答文本片段"""
    agent = get_agent(model_id)
    async for chunk, _metadata in agent.astream(
        {"messages": [("user", message)]},
        stream_mode="messages",
    ):
        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content:
            yield content
