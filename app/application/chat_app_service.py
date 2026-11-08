from collections.abc import AsyncIterator, Sequence

import httpx

from app.adapters.base import ChatMessage, LLMAdapter
from app.adapters.llm import get_adapter
from app.config.settings import MODEL_CONFIGS
from app.core.exceptions import (
    AppException,
    UnsupportedModelException,
    UpstreamException,
    UpstreamTimeoutException,
)
from app.services.rate_limit_service import rate_limit_service
from app.services.retry_service import retry_service


class ChatAppService:
    """应用服务层：编排模型校验、限流、适配器调用、重试与错误映射。"""

    async def complete(self, message: str, model_id: str) -> str:
        """一次性返回完整回答。"""
        # 1. 前置校验
        # 1.1 校验模型是否存在
        self._validate_model(model_id)
        # 1.2 校验模型是否还有配额（超限抛 429）
        await rate_limit_service.check(model_id)

        # 2. 准备调用：获取适配器、组装对话消息
        adapter = get_adapter(model_id)
        messages = self._messages(message)

        try:
            # 3. 交给重试服务执行上游调用
            return await retry_service.run(lambda: adapter.complete(messages))
        except Exception as exc:
            # 4. 重试耗尽或不可重试：统一映射为对外错误码
            raise self._upstream_error(model_id, exc) from exc

    async def stream_reply(self, message: str, model_id: str) -> AsyncIterator[str]:
        """返回流式回答；校验与限流在返回前完成，超限才能正常响应 429。"""
        # 1. 前置校验（必须发生在返回流之前）
        # 1.1 校验模型是否存在
        self._validate_model(model_id)
        # 1.2 校验模型是否还有配额
        await rate_limit_service.check(model_id)

        # 2. 准备调用：获取适配器、组装对话消息
        adapter = get_adapter(model_id)
        messages = self._messages(message)

        # 3. 返回带重试能力的流，异常在流出阶段统一映射
        return self._stream(model_id, adapter, messages)

    async def _stream(
        self,
        model_id: str,
        adapter: LLMAdapter,
        messages: Sequence[ChatMessage],
    ) -> AsyncIterator[str]:
        try:
            # 1. 交给重试服务流式执行（仅首包前可重试）
            async for content in retry_service.iterate(
                lambda: adapter.stream(messages)
            ):
                yield content
        except Exception as exc:
            # 2. 统一映射为对外错误码
            raise self._upstream_error(model_id, exc) from exc

    @staticmethod
    def _validate_model(model_id: str) -> None:
        """校验模型是否在配置中存在，不存在抛 400。"""
        if model_id not in MODEL_CONFIGS:
            raise UnsupportedModelException(
                f"不支持的模型: {model_id}，可选: {sorted(MODEL_CONFIGS)}"
            )

    @staticmethod
    def _messages(message: str) -> list[ChatMessage]:
        """把用户消息包装成统一的对话消息结构。"""
        return [ChatMessage(role="user", content=message)]

    @staticmethod
    def _upstream_error(model_id: str, exc: Exception) -> AppException:
        """把上游异常归类为统一错误码。"""
        # 1. 已经是业务异常，直接透传
        if isinstance(exc, AppException):
            return exc

        # 2. 超时类异常 -> 504
        name = type(exc).__name__.lower()
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)) or "timeout" in name:
            return UpstreamTimeoutException(f"模型 {model_id} 调用超时，请稍后重试")

        # 3. 其余上游异常 -> 502
        return UpstreamException(f"模型 {model_id} 调用失败: {exc}")


chat_app_service = ChatAppService()
