import json
import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.adapters.base import TokenUsage
from app.config.tenancy import DEFAULT_WORKSPACE_ID
from app.repositories import repositories
from app.repositories.base import LlmCall, MetricsRepository
from app.services.model_registry import model_registry

logger = logging.getLogger(__name__)

# 内存中保留的最近调用明细条数
RECENT_CALL_LIMIT = 100


def calculate_cost(model_id: str, usage: TokenUsage | None) -> float | None:
    """按模型定价与 Token 消耗折算本次调用成本（美元）；无 usage 时返回 None。"""
    if usage is None:
        return None
    config = model_registry.get(model_id)
    if config is None:
        return None
    return (
        usage.prompt_tokens * config.input_price_per_million
        + usage.completion_tokens * config.output_price_per_million
    ) / 1_000_000


@dataclass(frozen=True, slots=True)
class CallMetrics:
    """单次调用指标：Token 分类消耗、总延迟与首 Token 延迟。"""

    model: str  # 实际应答的模型 id
    stream: bool  # 是否为流式调用
    success: bool  # 调用是否成功
    latency_ms: float  # 本次调用总耗时（毫秒）
    request_id: str = ""  # 本次请求的 ID，同一请求的主备多次尝试共享
    requested_model: str | None = None  # 调用方请求的模型 id，与 model 不同即发生了主备切换
    prompt_name: str | None = None  # 本次调用引用的 Prompt 模板名，便于按模板版本归因
    prompt_version: str | None = None  # 引用的模板版本（含 latest 解析后的具体版本）
    first_token_ms: float | None = None  # 首 Token 延迟（毫秒），非流式为 None
    usage: TokenUsage | None = None  # Token 分类消耗，上游未返回时为 None
    retries: int = 0  # 本次调用因瞬时故障重试的次数
    error_code: str | None = None  # 失败时的错误码
    conversation_id: str | None = None  # 关联的历史对话 id
    workspace_id: str = DEFAULT_WORKSPACE_ID  # 所属工作区
    run_id: str | None = None  # 关联的 Run id

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": self.request_id,
            "requested_model": self.requested_model,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "stream": self.stream,
            "success": self.success,
            "latency_ms": round(self.latency_ms, 2),
            "first_token_ms": (
                None if self.first_token_ms is None else round(self.first_token_ms, 2)
            ),
            "tokens": self.usage.to_dict() if self.usage else None,
            "retries": self.retries,
            "error_code": self.error_code,
            "workspace_id": self.workspace_id,
            "conversation_id": self.conversation_id,
            "run_id": self.run_id,
        }


@dataclass
class _Aggregate:
    """累计指标，按调用方式与 Token 类型分类。"""

    calls: int = 0  # 调用总次数
    successes: int = 0  # 成功次数
    failures: int = 0  # 失败次数
    stream_calls: int = 0  # 流式调用次数
    complete_calls: int = 0  # 非流式调用次数
    usage_calls: int = 0  # 上游返回了 usage 的调用次数
    prompt_tokens: int = 0  # 输入 Token 累计
    completion_tokens: int = 0  # 输出 Token 累计
    total_tokens: int = 0  # 总 Token 累计
    cached_tokens: int = 0  # 缓存命中 Token 累计
    reasoning_tokens: int = 0  # 推理 Token 累计
    latency_ms_sum: float = 0.0  # 总延迟求和，用于计算均值
    latency_ms_max: float = 0.0  # 最大总延迟
    first_token_ms_sum: float = 0.0  # 首 Token 延迟求和，用于计算均值
    first_token_ms_max: float = 0.0  # 最大首 Token 延迟
    first_token_calls: int = 0  # 有首 Token 延迟统计的调用次数
    retries: int = 0  # 重试次数累计
    fallbacks: int = 0  # 主备切换次数：实际应答模型与请求模型不同的调用数
    cost_usd: float = 0.0  # 累计成本（美元），按模型价格表折算


class MetricsService:
    """进程内指标记录：每次调用写结构化日志，并维护全局与按模型的聚合统计。"""

    def __init__(
        self,
        recent_limit: int = RECENT_CALL_LIMIT,
        repository: MetricsRepository | None = None,
    ):
        self._repository = repository or repositories.metrics
        self._total = _Aggregate()  # 全局累计指标
        self._by_model: dict[str, _Aggregate] = {}  # 按模型分组的累计指标
        self._recent: deque[dict[str, Any]] = deque(maxlen=recent_limit)  # 最近调用明细
        self._lock = threading.Lock()  # 保护累计指标与明细的并发访问
        # 启动时从仓储恢复历史调用，重启后 /metrics 不清零
        self._restore()

    def record(self, call: CallMetrics) -> None:
        """记录一次调用；日志便于外部采集，聚合数据供 /metrics 查询。"""
        cost_usd = calculate_cost(call.model, call.usage)
        payload = call.to_dict()
        payload["cost_usd"] = cost_usd
        logger.info("llm_call %s", json.dumps(payload, ensure_ascii=False))
        with self._lock:
            self._add(self._total, call, cost_usd)
            self._add(
                self._by_model.setdefault(call.model, _Aggregate()), call, cost_usd
            )
            self._recent.append(payload)
        self._persist(call, cost_usd, payload["timestamp"])

    def _persist(
        self, call: CallMetrics, cost_usd: float, timestamp: str
    ) -> None:
        """调用明细落库；写入失败只告警，不影响在线请求与内存指标。"""
        usage = call.usage
        try:
            self._repository.insert_call(
                LlmCall(
                    workspace_id=call.workspace_id,
                    conversation_id=call.conversation_id,
                    run_id=call.run_id,
                    timestamp=timestamp,
                    request_id=call.request_id,
                    requested_model=call.requested_model,
                    model=call.model,
                    stream=call.stream,
                    success=call.success,
                    latency_ms=call.latency_ms,
                    first_token_ms=call.first_token_ms,
                    has_usage=usage is not None,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    cached_tokens=usage.cached_tokens if usage else 0,
                    reasoning_tokens=usage.reasoning_tokens if usage else 0,
                    retries=call.retries,
                    error_code=call.error_code,
                    cost_usd=cost_usd,
                    prompt_name=call.prompt_name,
                    prompt_version=call.prompt_version,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 指标持久化失败不阻断调用
            logger.warning("写入调用指标失败: %s", exc)

    def _restore(self) -> None:
        """从仓储恢复聚合指标与最近明细。"""
        try:
            rows = self._repository.list_calls()
        except Exception as exc:  # noqa: BLE001 - 恢复失败时从零开始
            logger.warning("恢复调用指标失败: %s", exc)
            return
        for row in rows:
            usage = (
                TokenUsage(
                    prompt_tokens=row["prompt_tokens"],
                    completion_tokens=row["completion_tokens"],
                    total_tokens=row["total_tokens"],
                    cached_tokens=row["cached_tokens"],
                    reasoning_tokens=row["reasoning_tokens"],
                )
                if row["has_usage"]
                else None
            )
            call = CallMetrics(
                model=row["model"],
                stream=bool(row["stream"]),
                success=bool(row["success"]),
                latency_ms=row["latency_ms"],
                request_id=row["request_id"],
                requested_model=row["requested_model"],
                prompt_name=row["prompt_name"],
                prompt_version=row["prompt_version"],
                first_token_ms=row["first_token_ms"],
                usage=usage,
                retries=row["retries"],
                error_code=row["error_code"],
                conversation_id=row["conversation_id"],
                workspace_id=row["workspace_id"],
                run_id=row["run_id"],
            )
            cost_usd = row["cost_usd"]
            self._add(self._total, call, cost_usd)
            self._add(self._by_model.setdefault(row["model"], _Aggregate()), call, cost_usd)
            self._recent.append(self._row_payload(row))

    @staticmethod
    def _row_payload(row: Any) -> dict[str, Any]:
        """把数据库行还原为与 CallMetrics.to_dict 一致的明细结构。"""
        tokens = None
        if row["has_usage"]:
            tokens = {
                "prompt": row["prompt_tokens"],
                "completion": row["completion_tokens"],
                "total": row["total_tokens"],
                "cached": row["cached_tokens"],
                "reasoning": row["reasoning_tokens"],
            }
        return {
            "timestamp": row["timestamp"],
            "request_id": row["request_id"],
            "requested_model": row["requested_model"],
            "prompt_name": row["prompt_name"],
            "prompt_version": row["prompt_version"],
            "model": row["model"],
            "stream": bool(row["stream"]),
            "success": bool(row["success"]),
            "latency_ms": round(row["latency_ms"], 2),
            "first_token_ms": (
                None if row["first_token_ms"] is None else round(row["first_token_ms"], 2)
            ),
            "tokens": tokens,
            "retries": row["retries"],
            "error_code": row["error_code"],
            "workspace_id": row["workspace_id"],
            "conversation_id": row["conversation_id"],
            "run_id": row["run_id"],
            "cost_usd": row["cost_usd"],
        }

    def snapshot(self) -> dict[str, Any]:
        """返回全局、按模型以及最近调用的指标快照。"""
        with self._lock:
            return {
                "total": self._dump(self._total),
                "by_model": {
                    model: self._dump(aggregate)
                    for model, aggregate in self._by_model.items()
                },
                "recent": list(self._recent),
            }

    @staticmethod
    def _add(aggregate: _Aggregate, call: CallMetrics, cost_usd: float | None) -> None:
        aggregate.calls += 1
        if call.success:
            aggregate.successes += 1
        else:
            aggregate.failures += 1
        if call.stream:
            aggregate.stream_calls += 1
        else:
            aggregate.complete_calls += 1

        # 实际应答模型与请求模型不同，说明发生了主备切换
        if call.requested_model is not None and call.requested_model != call.model:
            aggregate.fallbacks += 1
        if cost_usd is not None:
            aggregate.cost_usd += cost_usd

        if call.usage is not None:
            aggregate.usage_calls += 1
            aggregate.prompt_tokens += call.usage.prompt_tokens
            aggregate.completion_tokens += call.usage.completion_tokens
            aggregate.total_tokens += call.usage.total_tokens
            aggregate.cached_tokens += call.usage.cached_tokens
            aggregate.reasoning_tokens += call.usage.reasoning_tokens

        aggregate.retries += call.retries
        aggregate.latency_ms_sum += call.latency_ms
        aggregate.latency_ms_max = max(aggregate.latency_ms_max, call.latency_ms)
        if call.first_token_ms is not None:
            aggregate.first_token_calls += 1
            aggregate.first_token_ms_sum += call.first_token_ms
            aggregate.first_token_ms_max = max(
                aggregate.first_token_ms_max, call.first_token_ms
            )

    @staticmethod
    def _dump(aggregate: _Aggregate) -> dict[str, Any]:
        return {
            "calls": {
                "total": aggregate.calls,
                "success": aggregate.successes,
                "failure": aggregate.failures,
                "stream": aggregate.stream_calls,
                "complete": aggregate.complete_calls,
                "retries": aggregate.retries,
                "fallbacks": aggregate.fallbacks,
            },
            "tokens": {
                "calls_with_usage": aggregate.usage_calls,
                "prompt": aggregate.prompt_tokens,
                "completion": aggregate.completion_tokens,
                "total": aggregate.total_tokens,
                "cached": aggregate.cached_tokens,
                "reasoning": aggregate.reasoning_tokens,
            },
            "cost_usd": round(aggregate.cost_usd, 6),
            "latency_ms": {
                "avg": (
                    round(aggregate.latency_ms_sum / aggregate.calls, 2)
                    if aggregate.calls
                    else 0.0
                ),
                "max": round(aggregate.latency_ms_max, 2),
            },
            "first_token_ms": {
                "avg": (
                    round(
                        aggregate.first_token_ms_sum / aggregate.first_token_calls, 2
                    )
                    if aggregate.first_token_calls
                    else None
                ),
                "max": (
                    round(aggregate.first_token_ms_max, 2)
                    if aggregate.first_token_calls
                    else None
                ),
            },
        }


metrics_service = MetricsService()
