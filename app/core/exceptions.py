from app.core.error_codes import ErrorCode


class AppException(Exception):
    """所有业务异常的父类：错误码、HTTP 状态和默认消息统一取自 ErrorCode。"""

    error_code: ErrorCode

    def __init__(
        self,
        message: str | None = None,
        *,
        headers: dict[str, str] | None = None,
    ):
        self.code = self.error_code
        self.message = message or self.error_code.message
        self.status_code = self.error_code.http_status
        self.headers = headers or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.code, "message": self.message}


class InvalidRequestException(AppException):
    error_code = ErrorCode.INVALID_REQUEST


class NotFoundException(AppException):
    error_code = ErrorCode.NOT_FOUND


class UnsupportedModelException(AppException):
    error_code = ErrorCode.UNSUPPORTED_MODEL


class RateLimitedException(AppException):
    error_code = ErrorCode.RATE_LIMITED


class UpstreamTimeoutException(AppException):
    error_code = ErrorCode.UPSTREAM_TIMEOUT


class UpstreamException(AppException):
    error_code = ErrorCode.UPSTREAM_ERROR


class InvalidModelOutputException(AppException):
    error_code = ErrorCode.INVALID_MODEL_OUTPUT


class InternalException(AppException):
    error_code = ErrorCode.INTERNAL_ERROR
