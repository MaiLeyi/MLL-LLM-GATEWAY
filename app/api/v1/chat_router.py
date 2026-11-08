import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.application.chat_app_service import chat_app_service
from app.core.exceptions import AppException
from app.schemas.chat import ChatRequest, ChatResponse, StreamRequest, Usage

router = APIRouter(tags=["chat"])


def _encode_sse(event: dict[str, Any]) -> str:
    """把事件编码为 SSE 帧；结尾两个换行是协议规定的事件分隔符。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    """一次性返回完整回答，支持多轮消息、结构化输出与 Prompt 版本引用"""
    # 1. 交给应用服务完成校验、限流、重试、主备切换与上游调用
    result = await chat_app_service.complete(
        payload.to_chat_messages(),
        payload.model,
        payload.response_format,
        payload.prompt,
        payload.timeout_seconds,
    )
    # 2. 接口层只负责组装结构化响应
    return ChatResponse(
        reply=result.reply,
        data=result.data,
        request_id=result.request_id,
        model=result.model,
        usage=Usage(**result.usage.to_dict()) if result.usage else None,
        latency_ms=result.latency_ms,
        attempts=result.attempts,
    )


@router.post("/stream")
async def stream_chat(payload: StreamRequest):
    """流式返回 Agent 回答：标准 SSE 事件（response.created / content.delta / response.completed / response.failed）"""
    # 1. 先完成校验与限流并拿到带主备切换与重试能力的事件流（超限在此直接返回 429）
    request_id, event_stream = await chat_app_service.stream_reply(
        payload.to_chat_messages(), payload.model, payload.prompt, payload.timeout_seconds
    )

    async def event_generator():
        # 2. 先告知客户端请求 ID 与请求模型，便于在 /metrics 明细中串联本次调用
        yield _encode_sse(
            {"type": "response.created", "request_id": request_id, "model": payload.model}
        )
        try:
            # 3. 逐段输出增量文本与结束汇总事件
            async for event in event_stream:
                yield _encode_sse(event.to_dict())
        except AppException as exc:
            # 4. 流建立后的异常转成 response.failed 事件
            yield _encode_sse({"type": "response.failed", "error": exc.to_dict()})

    return StreamingResponse(
        event_generator(), media_type="text/event-stream; charset=utf-8"
    )
