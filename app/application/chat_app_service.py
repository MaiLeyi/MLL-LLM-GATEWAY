import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

import httpx
from jsonschema import ValidationError, validate

from app.adapters.base import ChatMessage, LLMAdapter, StreamChunk, TokenUsage
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
    """一次性调用的结构化结果：原始回答 + 按约束解析出的数据 + 调用元信息。"""

    reply: str  # 模型返回的原始回答文本
    data: Any | None = None  # response_format 要求 JSON 时解析出的数据
    model: str = ""  # 实际应答的模型 id；发生主备切换时与请求模型不同
    request_id: str = ""  # 本次请求 ID，用于在 /metrics 明细中串联调用
    usage: TokenUsage | None = None  # 本次调用的 Token 分类消耗，上游未返回时为 None
    latency_ms: int = 0  # 请求总耗时（毫秒），含重试与主备切换
    attempts: int = 0  # 上游调用总次数（重试与主备切换都会计入）


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """流式输出的标准 SSE 事件：content.delta 增量文本，response.completed 结束汇总。"""

    type: Literal["content.delta", "response.completed"]
    delta: str = ""  # content.delta 的文本增量
    model: str = ""  # response.completed 的实际应答模型
    request_id: str = ""  # response.completed 的请求 ID
    usage: TokenUsage | None = None  # response.completed 的 Token 分类消耗
    latency_ms: int = 0  # response.completed 的请求总耗时

    def to_dict(self) -> dict[str, Any]:
        if self.type == "content.delta":
            return {"type": self.type, "delta": self.delta}
        return {
            "type": self.type,
            "model": self.model,
            "request_id": self.request_id,
            "usage": self.usage.to_dict() if self.usage else None,
            "latency_ms": self.latency_ms,
        }


class ChatAppService:
    """应用服务层：编排模型校验、限流、适配器调用、重试、主备切换与错误映射。"""

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        model_id: str,
        response_format: ResponseFormat | None = None,
        prompt: PromptSelection | None = None,
        timeout_seconds: float | None = None,
    ) -> CompletionResult:
        """一次性返回完整回答，并按 response_format 校验输出。"""
        response_format = response_format or ResponseFormat()
        # 1. 前置校验（请求级，只做一次）
        # 1.1 校验模型是否存在
        self._validate_model(model_id)
        # 1.2 按版本引用渲染系统提示词（模板不存在/变量缺失在此返回 4xx），
        #     并记录 latest 解析后的具体版本，供指标归因
        system_prompt = None
        prompt_name = None
        prompt_version = None
        if prompt is not None:
            system_prompt, template = prompt_service.render_resolved(prompt)
            prompt_name, prompt_version = template.name, template.version

        # 2. 准备调用：上游格式约束与请求级计时/计数
        upstream_format = response_format.to_upstream()
        request_id = str(uuid4())
        request_started = time.perf_counter()
        attempts = 0

        # 3. 沿主备链依次尝试：同一候选内的瞬时故障由重试服务兜底，
        #    重试耗尽后切下一个候选模型
        last_error: AppException | None = None
        for candidate in self._fallback_chain(model_id):
            adapter = get_adapter(candidate)
            # 每个候选模型独立限流；429 属于本网关的配额治理决策，直接抛出、不触发切换
            await rate_limit_service.check(candidate)
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
                        timeout_seconds=timeout_seconds,
                    ),
                    on_retry=_on_retry,
                )
            except Exception as exc:
                # 3.1 重试耗尽或不可重试：统一映射为对外错误码并记录指标
                attempts += 1 + retries
                error = self._upstream_error(candidate, exc)
                self._record_metrics(
                    request_id,
                    candidate,
                    requested_model=model_id,
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    stream=False,
                    started=started,
                    success=False,
                    retries=retries,
                    error_code=error.code.code,
                )
                # 仅上游可用性问题才切换备用模型；输出契约错误等直接抛出
                if isinstance(error, (UpstreamException, UpstreamTimeoutException)):
                    last_error = error
                    continue
                raise error from exc
            attempts += 1 + retries

            # 3.2 结构化输出：要求 JSON 时必须能解析成合法 JSON，声明 schema 时再做校验
            try:
                data = (
                    self._parse_structured(completion.text, response_format)
                    if response_format.requires_json
                    else None
                )
            except AppException as exc:
                self._record_metrics(
                    request_id,
                    candidate,
                    requested_model=model_id,
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    stream=False,
                    started=started,
                    success=False,
                    usage=completion.usage,
                    retries=retries,
                    error_code=exc.code.code,
                )
                raise

            self._record_metrics(
                request_id,
                candidate,
                requested_model=model_id,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                stream=False,
                started=started,
                success=True,
                usage=completion.usage,
                retries=retries,
            )
            return CompletionResult(
                reply=completion.text,
                data=data,
                model=candidate,
                request_id=request_id,
                usage=completion.usage,
                latency_ms=int((time.perf_counter() - request_started) * 1000),
                attempts=attempts,
            )

        # 4. 主备全部不可用：抛出最后一次上游错误
        assert last_error is not None
        raise last_error

    async def stream_reply(
        self,
        messages: Sequence[ChatMessage],
        model_id: str,
        prompt: PromptSelection | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[str, AsyncIterator[StreamEvent]]:
        """返回 (request_id, 标准 SSE 事件流)；校验与首选模型限流在返回前完成，超限才能正常响应 429。"""
        # 1. 前置校验（必须发生在返回流之前）
        # 1.1 校验模型是否存在
        self._validate_model(model_id)
        # 1.2 按版本引用渲染系统提示词，并记录 latest 解析后的具体版本
        system_prompt = None
        prompt_name = None
        prompt_version = None
        if prompt is not None:
            system_prompt, template = prompt_service.render_resolved(prompt)
            prompt_name, prompt_version = template.name, template.version
        # 1.3 校验首选模型是否还有配额
        await rate_limit_service.check(model_id)

        # 2. 生成请求 ID，返回带主备切换与重试能力的事件流，异常在流出阶段统一映射
        request_id = str(uuid4())
        return request_id, self._stream_with_fallback(
            request_id,
            model_id,
            messages,
            system_prompt,
            prompt_name,
            prompt_version,
            timeout_seconds,
        )

    async def _stream_with_fallback(
        self,
        request_id: str,
        requested_model: str,
        messages: Sequence[ChatMessage],
        system_prompt: str | None,
        prompt_name: str | None,
        prompt_version: str | None,
        timeout_seconds: float | None,
    ) -> AsyncIterator[StreamEvent]:
        """沿主备链依次建流；已产出内容后不再切换，避免客户端看到重复文本。"""
        started = time.perf_counter()
        emitted = False
        last_error: AppException | None = None
        for index, candidate in enumerate(self._fallback_chain(requested_model)):
            if index > 0:
                # 备用模型在真正切换时才占用配额（首选模型已在返回流之前完成限流校验）
                await rate_limit_service.check(candidate)
            adapter = get_adapter(candidate)
            usage: TokenUsage | None = None
            try:
                async for chunk in self._stream(
                    request_id,
                    requested_model,
                    candidate,
                    adapter,
                    messages,
                    system_prompt,
                    prompt_name,
                    prompt_version,
                    timeout_seconds,
                ):
                    # 结束分片携带 usage，留给 completed 事件汇总
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if chunk.text:
                        emitted = True
                        yield StreamEvent(type="content.delta", delta=chunk.text)
                yield StreamEvent(
                    type="response.completed",
                    model=candidate,
                    request_id=request_id,
                    usage=usage,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
                return
            except Exception as exc:
                error = self._upstream_error(candidate, exc)
                # 已输出内容 或 非上游可用性问题：不再切换，直接向外抛
                if emitted or not isinstance(
                    error, (UpstreamException, UpstreamTimeoutException)
                ):
                    raise error from exc
                last_error = error
        assert last_error is not None
        raise last_error

    async def _stream(
        self,
        request_id: str,
        requested_model: str,
        model_id: str,
        adapter: LLMAdapter,
        messages: Sequence[ChatMessage],
        system_prompt: str | None,
        prompt_name: str | None,
        prompt_version: str | None,
        timeout_seconds: float | None,
    ) -> AsyncIterator[StreamChunk]:
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
                lambda: adapter.stream(
                    messages,
                    system_prompt=system_prompt,
                    timeout_seconds=timeout_seconds,
                ),
                on_retry=_on_retry,
            ):
                # 2.1 结束分片携带 usage，覆盖为完整统计
                if chunk.usage is not None:
                    usage = chunk.usage
                if chunk.text:
                    # 2.2 首个有效文本片段即首 Token 延迟
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                # 2.3 文本分片与仅含 usage 的结束分片都向上传递，供 completed 事件汇总
                if chunk.text or chunk.usage is not None:
                    yield chunk
        except Exception as exc:
            # 3. 统一映射为对外错误码，并记录失败指标
            error = self._upstream_error(model_id, exc)
            self._record_metrics(
                request_id,
                model_id,
                requested_model=requested_model,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
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
            request_id,
            model_id,
            requested_model=requested_model,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            stream=True,
            started=started,
            success=True,
            usage=usage,
            first_token_ms=first_token_ms,
            retries=retries,
        )

    @staticmethod
    def _record_metrics(
        request_id: str,
        model_id: str,
        *,
        requested_model: str | None = None,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
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
                request_id=request_id,
                requested_model=requested_model,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                first_token_ms=first_token_ms,
                usage=usage,
                retries=retries,
                error_code=error_code,
            )
        )

    @staticmethod
    def _fallback_chain(model_id: str) -> list[str]:
        """从请求模型出发沿 fallback_model 链去重，得到依次尝试的候选列表。"""
        chain: list[str] = []
        current: str | None = model_id
        while current is not None and current not in chain:
            chain.append(current)
            current = MODEL_CONFIGS[current].fallback_model
        return chain

    @staticmethod
    def _validate_model(model_id: str) -> None:
        """校验模型是否在配置中存在，不存在抛 400。"""
        if model_id not in MODEL_CONFIGS:
            raise UnsupportedModelException(
                f"不支持的模型: {model_id}，可选: {sorted(MODEL_CONFIGS)}"
            )

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
