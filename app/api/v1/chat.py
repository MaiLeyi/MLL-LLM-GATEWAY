import json

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.services.chat_service import complete, stream_reply

router = APIRouter(tags=["chat"])


@router.get("/chat")
async def chat(message: str = Query(..., description="用户输入的消息")):
    """一次性返回完整回答"""
    return {"reply": await complete(message)}


@router.get("/stream")
async def stream_chat(message: str = Query(..., description="用户输入的消息")):
    """流式返回 Agent 回答（Server-Sent Events）"""

    async def event_generator():
        async for content in stream_reply(message):
            yield f"data: {json.dumps({'content': content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(), media_type="text/event-stream; charset=utf-8"
    )
