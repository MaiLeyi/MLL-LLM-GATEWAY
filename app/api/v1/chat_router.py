import json
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.adapters.base import TokenUsage
from app.application.chat_app_service import chat_app_service
from app.config.tenancy import DEFAULT_WORKSPACE_ID
from app.core.exceptions import AppException
from app.schemas.chat import ChatRequest, ChatResponse, StreamRequest, Usage
from app.services.conversation_service import conversation_service
from app.services.run_service import run_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _encode_sse(event: dict[str, Any]) -> str:
    """把事件编码为 SSE 帧；结尾两个换行是协议规定的事件分隔符。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _prompt_snapshot(payload: ChatRequest | StreamRequest) -> dict[str, Any] | None:
    """记录本次执行引用的模板与变量，保证历史可追溯。"""
    if payload.prompt is None:
        return None
    return {"prompt": payload.prompt.model_dump()}


def _finish_run_failed(run_id: str, error_code: str, error_message: str) -> None:
    """结束失败的 Run；持久化异常只记日志，不影响对外的原始错误。"""
    try:
        run_service.finish(
            run_id,
            status="failed",
            error_code=error_code,
            error_message=error_message,
        )
    except Exception:  # noqa: BLE001 - 收尾失败不能覆盖原始错误
        logger.exception("结束 Run 失败: %s", run_id)


def _finish_run_cancelled(run_id: str) -> None:
    try:
        run_service.finish(
            run_id,
            status="cancelled",
            error_code="CANCELLED",
            error_message="客户端断开或流未正常结束",
        )
    except Exception:  # noqa: BLE001 - 收尾失败只记日志
        logger.exception("取消 Run 失败: %s", run_id)


def _persist_exchange(
    payload: ChatRequest | StreamRequest,
    *,
    conversation_id: str,
    run_id: str,
    reply: str,
    model: str,
    request_id: str,
    usage: TokenUsage | None,
    latency_ms: int | None,
) -> None:
    """把一次成功问答写入历史，并结束 Run；失败只降级为审计日志。"""
    last = payload.to_chat_messages()[-1]
    try:
        conversation_service.append_exchange(
            conversation_id,
            user_role=last.role,
            user_content=last.content,
            assistant_content=reply,
            assistant_model=model,
            request_id=request_id,
            usage=usage.to_dict() if usage else None,
            latency_ms=latency_ms,
            prompt=payload.prompt,
            model=payload.model,
            run_id=run_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
        )
        run_service.finish(
            run_id,
            status="succeeded",
            model=model,
            usage=usage.to_dict() if usage else None,
            latency_ms=latency_ms,
        )
    except Exception:  # noqa: BLE001 - 历史写入失败不推翻已成功的模型调用
        logger.exception("写入历史对话失败: %s", conversation_id)
        _finish_run_failed(run_id, "HISTORY_WRITE_FAILED", "写入历史对话失败")


def _start_run(
    payload: ChatRequest | StreamRequest, request_id: str
) -> str:
    """有 conversation_id 时先校验会话，再创建 Run；无会话的调用不落 Run。"""
    if not payload.conversation_id:
        return ""
    conversation_service.require(
        payload.conversation_id, workspace_id=DEFAULT_WORKSPACE_ID
    )
    return run_service.start(
        request_id=request_id,
        workspace_id=DEFAULT_WORKSPACE_ID,
        conversation_id=payload.conversation_id,
        model=payload.model,
        prompt_snapshot=_prompt_snapshot(payload),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    """一次性返回完整回答，支持多轮消息、结构化输出、Prompt 版本与历史对话写入"""
    # 1. 先校验会话并创建 Run（有历史对话时）
    request_id = str(uuid4())
    run_id = _start_run(payload, request_id)

    # 2. 交给应用服务完成校验、限流、重试、主备切换与上游调用
    try:
        result = await chat_app_service.complete(
            payload.to_chat_messages(),
            payload.model,
            payload.response_format,
            payload.prompt,
            payload.timeout_seconds,
            conversation_id=payload.conversation_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            run_id=run_id or None,
            request_id=request_id,
        )
    except AppException as exc:
        if run_id:
            _finish_run_failed(run_id, exc.code.code, exc.message)
        raise
    except Exception as exc:  # noqa: BLE001 - 统一结束 Run 后继续抛出
        if run_id:
            _finish_run_failed(run_id, "INTERNAL_ERROR", str(exc))
        raise

    # 3. 成功问答写入历史并结束 Run
    if payload.conversation_id and run_id:
        _persist_exchange(
            payload,
            conversation_id=payload.conversation_id,
            run_id=run_id,
            reply=result.reply,
            model=result.model,
            request_id=result.request_id,
            usage=result.usage,
            latency_ms=result.latency_ms,
        )

    return ChatResponse(
        reply=result.reply,
        data=result.data,
        request_id=result.request_id,
        model=result.model,
        usage=Usage(**result.usage.to_dict()) if result.usage else None,
        latency_ms=result.latency_ms,
        attempts=result.attempts,
        conversation_id=payload.conversation_id,
    )


@router.post("/stream")
async def stream_chat(payload: StreamRequest):
    """流式返回 Agent 回答：标准 SSE 事件，并在结束时写入历史对话与 Run"""
    # 1. 先完成会话校验、Run 创建、限流与事件流准备（超限/不存在在此直接返回 HTTP 错误）
    request_id = str(uuid4())
    run_id = _start_run(payload, request_id)

    try:
        request_id, event_stream = await chat_app_service.stream_reply(
            payload.to_chat_messages(),
            payload.model,
            payload.prompt,
            payload.timeout_seconds,
            conversation_id=payload.conversation_id,
            workspace_id=DEFAULT_WORKSPACE_ID,
            run_id=run_id or None,
            request_id=request_id,
        )
    except AppException as exc:
        if run_id:
            _finish_run_failed(run_id, exc.code.code, exc.message)
        raise
    except Exception as exc:  # noqa: BLE001 - 统一结束 Run 后继续抛出
        if run_id:
            _finish_run_failed(run_id, "INTERNAL_ERROR", str(exc))
        raise

    async def event_generator():
        # 2. 先告知客户端请求 ID 与请求模型，便于在 /metrics 明细中串联本次调用
        yield _encode_sse(
            {
                "type": "response.created",
                "request_id": request_id,
                "model": payload.model,
                "conversation_id": payload.conversation_id,
            }
        )
        reply_parts: list[str] = []
        finished = False
        try:
            # 3. 逐段输出增量文本；completed 事件前先落库，保证客户端刷新即可看到历史
            async for event in event_stream:
                if event.type == "content.delta":
                    reply_parts.append(event.delta)
                elif event.type == "response.completed":
                    if payload.conversation_id and run_id:
                        _persist_exchange(
                            payload,
                            conversation_id=payload.conversation_id,
                            run_id=run_id,
                            reply="".join(reply_parts),
                            model=event.model,
                            request_id=request_id,
                            usage=event.usage,
                            latency_ms=event.latency_ms,
                        )
                    finished = True
                yield _encode_sse(event.to_dict())
        except AppException as exc:
            # 4. 流建立后的异常转成 response.failed 事件，并结束 Run
            if run_id and not finished:
                _finish_run_failed(run_id, exc.code.code, exc.message)
                finished = True
            yield _encode_sse({"type": "response.failed", "error": exc.to_dict()})
        finally:
            # 5. 客户端断开或流未正常结束：标记 Run 为 cancelled，避免长期 running
            if run_id and not finished:
                _finish_run_cancelled(run_id)

    return StreamingResponse(
        event_generator(), media_type="text/event-stream; charset=utf-8"
    )
