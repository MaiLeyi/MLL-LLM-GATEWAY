from enum import Enum


class ErrorCode(Enum):
    """统一错误码配置：错误编码、HTTP 状态、默认提示消息集中在这里维护。"""

    INVALID_REQUEST = ("INVALID_REQUEST", 400, "请求参数不合法")
    NOT_FOUND = ("NOT_FOUND", 404, "资源不存在")
    UNSUPPORTED_MODEL = ("UNSUPPORTED_MODEL", 400, "不支持的模型")
    RATE_LIMITED = ("RATE_LIMITED", 429, "请求过于频繁")
    UPSTREAM_TIMEOUT = ("UPSTREAM_TIMEOUT", 504, "上游模型调用超时")
    UPSTREAM_ERROR = ("UPSTREAM_ERROR", 502, "上游模型调用失败")
    INVALID_MODEL_OUTPUT = ("INVALID_MODEL_OUTPUT", 502, "模型返回内容不符合输出格式约束")
    INTERNAL_ERROR = ("INTERNAL_ERROR", 500, "服务内部错误")

    def __init__(self, code: str, http_status: int, message: str):
        self.code = code
        self.http_status = http_status
        self.message = message
