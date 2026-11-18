import time

import httpx

from app.adapters.base import ChatMessage
from app.adapters.llm import get_adapter
from app.core.error_codes import ErrorCode
from app.core.exceptions import AppException
from app.services.model_registry import model_registry
from app.schemas.model import ModelTestResult

TEST_TIMEOUT_SECONDS = 20.0
TEST_SYSTEM_PROMPT = "你是连通性测试助手，只按用户要求简短回复。"
TEST_USER_PROMPT = "只回复两个字：可用"


class ModelTestService:
    """模型可用性测试：用最小请求真实打通上游，返回延迟与错误信息。"""

    async def test(self, model_id: str) -> ModelTestResult:
        # 先校验模型是否存在，不存在时抛出统一的 400
        model_registry.require(model_id)
        started = time.perf_counter()

        try:
            adapter = get_adapter(model_id)
            completion = await adapter.complete(
                [ChatMessage(role="user", content=TEST_USER_PROMPT)],
                system_prompt=TEST_SYSTEM_PROMPT,
                timeout_seconds=TEST_TIMEOUT_SECONDS,
            )
        except AppException as exc:
            return ModelTestResult(
                ok=False,
                model=model_id,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error_code=exc.code.code,
                error_message=exc.message,
            )
        except Exception as exc:  # noqa: BLE001 - 测试接口需要把任意上游异常转成结果
            name = type(exc).__name__.lower()
            if isinstance(exc, (httpx.TimeoutException, TimeoutError)) or "timeout" in name:
                code = ErrorCode.UPSTREAM_TIMEOUT.code
                message = f"模型 {model_id} 调用超时"
            else:
                code = ErrorCode.UPSTREAM_ERROR.code
                message = f"模型 {model_id} 调用失败: {exc}"
            return ModelTestResult(
                ok=False,
                model=model_id,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error_code=code,
                error_message=message,
            )

        return ModelTestResult(
            ok=True,
            model=model_id,
            latency_ms=int((time.perf_counter() - started) * 1000),
            reply=completion.text.strip()[:200],
            usage=completion.usage.to_dict() if completion.usage else None,
        )


model_test_service = ModelTestService()
