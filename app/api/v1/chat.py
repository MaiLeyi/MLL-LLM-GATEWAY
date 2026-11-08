import json

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.application.chat_app_service import chat_app_service
from app.config.settings import DEFAULT_MODEL
from app.core.exceptions import AppException

router = APIRouter(tags=["chat"])


@router.get("/chat")
async def chat(
    message: str = Query(..., description="用户输入的消息"),
    model: str = Query(DEFAULT_MODEL, description="模型 id"),
):
    """一次性返回完整回答"""
    # 1. 交给应用服务完成校验、限流、重试与上游调用
    reply = await chat_app_service.complete(message, model)
    # 2. 接口层只负责组装 HTTP 响应
    return {"reply": reply}


@router.get("/stream")
async def stream_chat(
    message: str = Query(..., description="用户输入的消息"),
    model: str = Query(DEFAULT_MODEL, description="模型 id"),
):
    """流式返回 Agent 回答（Server-Sent Events）"""
    # 1. 先完成校验与限流并拿到带重试的流（超限在此直接返回 429）
    reply_stream = await chat_app_service.stream_reply(message, model)

    async def event_generator():
        try:
            # 2. 逐段输出 SSE 内容
            async for content in reply_stream:
                yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
        except AppException as exc:
            # 3. 流建立后的异常转成 error 事件
            payload = {"error": exc.to_dict()}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        # 4. 发送结束标记
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(), media_type="text/event-stream; charset=utf-8"
    )
