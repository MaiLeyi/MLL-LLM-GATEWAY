import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.application.chat_app_service import chat_app_service
from app.core.exceptions import AppException
from app.schemas.chat import ChatRequest, ChatResponse, StreamRequest

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    """一次性返回完整回答，支持 response_format 结构化输出与 Prompt 版本引用"""
    # 1. 交给应用服务完成校验、限流、重试与上游调用
    result = await chat_app_service.complete(
        payload.message, payload.model, payload.response_format, payload.prompt
    )
    # 2. 接口层只负责组装结构化响应
    return ChatResponse(reply=result.reply, data=result.data)


@router.post("/stream")
async def stream_chat(payload: StreamRequest):
    """流式返回 Agent 回答（Server-Sent Events），支持 Prompt 版本引用"""
    # 1. 先完成校验与限流并拿到带重试的流（超限在此直接返回 429）
    reply_stream = await chat_app_service.stream_reply(
        payload.message, payload.model, payload.prompt
    )

    async def event_generator():
        try:
            # 2. 逐段输出 SSE 内容
            async for content in reply_stream:
                yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
        except AppException as exc:
            # 3. 流建立后的异常转成 error 事件
            error_payload = {"error": exc.to_dict()}
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
        # 4. 发送结束标记
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(), media_type="text/event-stream; charset=utf-8"
    )
