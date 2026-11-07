import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.config.settings import DEFAULT_MODEL, MODEL_CONFIGS
from app.services.chat_service import complete, stream_reply

router = APIRouter(tags=["chat"])


def resolve_model(model: str) -> str:
    if model not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的模型: {model}，可选: {sorted(MODEL_CONFIGS)}",
        )
    return model


@router.get("/chat")
async def chat(
    message: str = Query(..., description="用户输入的消息"),
    model: str = Query(DEFAULT_MODEL, description="模型 id"),
):
    """一次性返回完整回答"""
    return {"reply": await complete(message, resolve_model(model))}


@router.get("/stream")
async def stream_chat(
    message: str = Query(..., description="用户输入的消息"),
    model: str = Query(DEFAULT_MODEL, description="模型 id"),
):
    """流式返回 Agent 回答（Server-Sent Events）"""
    model_id = resolve_model(model)

    async def event_generator():
        async for content in stream_reply(message, model_id):
            yield f"data: {json.dumps({'content': content})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(), media_type="text/event-stream; charset=utf-8"
    )
