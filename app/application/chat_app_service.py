import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from jsonschema import ValidationError, validate

from app.adapters.base import ChatMessage, LLMAdapter, TokenUsage
from app.adapters.llm import get_adapter
from app.config.settings import MODEL_CONFIGS
from app.core.exceptions import (
    AppException,
    InvalidModelOutputException,
    UnsupportedModelException,
    UpstreamException,
    UpstreamTimeoutException,
)
from app.schemas.chat import ResponseFormat
from app.schemas.prompt import PromptSelection
from app.services.metrics_service import CallMetrics, metrics_service
from app.services.prompt_service import prompt_service
from app.services.rate_limit_service import rate_limit_service
from app.services.retry_service import retry_service


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """一次性调用的结构化结果：原始回答 + 按约束解析出的数据。"""

    reply: str  # 模型返回的原始回答文本
    data: Any | None = None  # response_format 要求 JSON 时解析出的数据


class ChatAppService:
    """应用服务层：编排模型校验、限流、适配器调用、重试与错误映射。"""

    async def complete(
        self,
        message: str,
        model_id: str,
        response_format: ResponseFormat | None = None,
        prompt: PromptSelection | None = None,
    ) -> CompletionResult:
        """一次性返回完整回答，并按 response_format 校验输出。"""
        response_format = response_format or ResponseFormat()
        # 1. 前置校验
        # 1.1 校验模型是否存在
        self._validate_model(model_id)
        # 1.2 按版本引用渲染系统提示词（模板不存在/变量缺失在此返回 4xx）
        system_prompt = prompt_service.render(prompt) if prompt else None
        # 1.3 校验模型是否还有配额（超限抛 429）
        await rate_limit_service.check(model_id)

        # 2. 准备调用：获取适配器、组装对话消息与上游格式约束
        adapter = get_adapter(model_id)
        messages = self._messages(message)
        upstream_format = response_format.to_upstream()

        # 3. 交给重试服务执行上游调用，并记录本次调用指标
        started = time.perf_counter()
        retries = 0

        def _on_retry(attempt: int) -> None:
            nonlocal retries
            retries = attempt

        try:
            completion = await retry_service.run(
                lambda: adapter.complete(
                    messages,
                    response_format=upstream_format,
                    system_prompt=system_prompt,
                ),
                on_retry=_on_retry,
            )
        except Exception as exc:
            # 4. 重试耗尽或不可重试：统一映射为对外错误码
            error = self._upstream_error(model_id, exc)
            self._record_metrics(
                model_id,
                stream=False,
                started=started,
                success=False,
                retries=retries,
                error_code=error.code.code,
            )
            raise error from exc

        # 5. 结构化输出：要求 JSON 时必须能解析成合法 JSON，声明 schema 时再做校验
        try:
            data = (
                self._parse_structured(completion.text, response_format)
                if response_format.requires_json
                else None
            )
        except AppException as exc:
            self._record_metrics(
                model_id,
                stream=False,
                started=started,
                success=False,
                usage=completion.usage,
                retries=retries,
                error_code=exc.code.code,
            )
            raise

        self._record_metrics(
            model_id,
            stream=False,
            started=started,
            success=True,
            usage=completion.usage,
            retries=retries,
        )
        return CompletionResult(reply=completion.text, data=data)

    async def stream_reply(
        self,
        message: str,
        model_id: str,
        prompt: PromptSelection | None = None,
    ) -> AsyncIterator[str]:
        """返回流式回答；校验与限流在返回前完成，超限才能正常响应 429。"""
        # 1. 前置校验（必须发生在返回流之前）
        # 1.1 校验模型是否存在
        self._validate_model(model_id)
        # 1.2 按版本引用渲染系统提示词
        system_prompt = prompt_service.render(prompt) if prompt else None
        # 1.3 校验模型是否还有配额
        await rate_limit_service.check(model_id)

        # 2. 准备调用：获取适配器、组装对话消息
        adapter = get_adapter(model_id)
        messages = self._messages(message)

        # 3. 返回带重试能力的流，异常在流出阶段统一映射
        return self._stream(model_id, adapter, messages, system_prompt)

    async def _stream(
        self,
        model_id: str,
        adapter: LLMAdapter,
        messages: Sequence[ChatMessage],
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        # 1. 记录起始时间、首 Token 延迟与上游返回的 usage
        started = time.perf_counter()
        first_token_ms: float | None = None
        usage: TokenUsage | None = None
        retries = 0

        def _on_retry(attempt: int) -> None:
            nonlocal retries
            retries = attempt

        try:
            # 2. 交给重试服务流式执行（仅首包前可重试）
            async for chunk in retry_service.iterate(
                lambda: adapter.stream(messages, system_prompt=system_prompt),
                on_retry=_on_retry,
            ):
                # 2.1 结束分片携带 usage，覆盖为完整统计
                if chunk.usage is not None:
                    usage = chunk.usage
                if chunk.text:
                    # 2.2 首个有效文本片段即首 Token 延迟
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    yield chunk.text
        except Exception as exc:
            # 3. 统一映射为对外错误码，并记录失败指标
            error = self._upstream_error(model_id, exc)
            self._record_metrics(
                model_id,
                stream=True,
                started=started,
                success=False,
                usage=usage,
                first_token_ms=first_token_ms,
                retries=retries,
                error_code=error.code.code,
            )
            raise error from exc

        # 4. 正常结束：记录本次流式调用的完整指标
        self._record_metrics(
            model_id,
            stream=True,
            started=started,
            success=True,
            usage=usage,
            first_token_ms=first_token_ms,
            retries=retries,
        )

    @staticmethod
    def _record_metrics(
        model_id: str,
        *,
        stream: bool,
        started: float,
        success: bool,
        usage: TokenUsage | None = None,
        first_token_ms: float | None = None,
        retries: int = 0,
        error_code: str | None = None,
    ) -> None:
        metrics_service.record(
            CallMetrics(
                model=model_id,
                stream=stream,
                success=success,
                latency_ms=(time.perf_counter() - started) * 1000,
                first_token_ms=first_token_ms,
                usage=usage,
                retries=retries,
                error_code=error_code,
            )
        )

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
    def _parse_structured(reply: str, response_format: ResponseFormat) -> Any:
        """解析并校验结构化输出：先保证是合法 JSON，再按声明的 schema 校验。"""
        data = ChatAppService._parse_json(reply)
        # json_schema 模式要求返回结构匹配，schema 不匹配同样属于模型输出违约
        if response_format.type == "json_schema" and response_format.json_schema:
            try:
                validate(
                    instance=data,
                    schema=response_format.json_schema.schema_,
                )
            except ValidationError as exc:
                raise InvalidModelOutputException(
                    f"模型返回的 JSON 不符合 schema: {exc.message}"
                ) from exc
        return data

    @staticmethod
    def _parse_json(reply: str) -> Any:
        """把模型输出解析为 JSON；不合法时抛 502，保证对外契约。"""
        try:
            return json.loads(reply)
        except json.JSONDecodeError as exc:
            raise InvalidModelOutputException(
                f"模型未返回合法 JSON: {exc.msg}"
            ) from exc

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
