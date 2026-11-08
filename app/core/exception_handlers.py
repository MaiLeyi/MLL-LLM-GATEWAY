import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.error_codes import ErrorCode
from app.core.exceptions import AppException

logger = logging.getLogger(__name__)


def _error_response(
    error_code: ErrorCode,
    message: str | None = None,
    *,
    status_code: int | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """按统一结构输出：未指定消息/状态时使用 ErrorCode 中的默认配置。"""
    return JSONResponse(
        status_code=status_code if status_code is not None else error_code.http_status,
        content={"code": error_code.code, "message": message or error_code.message},
        headers=headers,
    )


def _code_for_status(status_code: int) -> ErrorCode:
    """把普通 HTTP 状态码归一到统一错误码。"""
    if status_code == 429:
        return ErrorCode.RATE_LIMITED
    if status_code == 404:
        return ErrorCode.NOT_FOUND
    if 400 <= status_code < 500:
        return ErrorCode.INVALID_REQUEST
    return ErrorCode.INTERNAL_ERROR


async def _handle_app_exception(request: Request, exc: AppException) -> JSONResponse:
    """1. 业务异常直接按异常携带的错误码与消息输出。"""
    return _error_response(exc.error_code, exc.message, headers=exc.headers)


async def _handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    # 1. 取第一个校验错误
    errors = exc.errors()
    first = errors[0] if errors else {}
    # 2. 拼接出错参数位置（如 message）
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = first.get("msg", "请求参数不合法")
    if location:
        message = f"{location}: {message}"
    # 3. 统一为 422 + INVALID_REQUEST
    return _error_response(ErrorCode.INVALID_REQUEST, message, status_code=422)


async def _handle_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """1. 普通 HTTP 异常按状态码归类后统一输出。"""
    return _error_response(
        _code_for_status(exc.status_code),
        str(exc.detail),
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
    )


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """1. 兜底异常：记录日志，对外只暴露统一错误结构。"""
    logger.exception("未处理异常: %s", exc)
    return _error_response(ErrorCode.INTERNAL_ERROR)


def register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器，保证所有错误都输出统一的 {code, message} 结构。"""
    app.add_exception_handler(AppException, _handle_app_exception)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(Exception, _handle_unexpected_error)
