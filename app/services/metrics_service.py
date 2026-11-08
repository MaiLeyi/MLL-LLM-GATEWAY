import json
import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.adapters.base import TokenUsage

logger = logging.getLogger(__name__)

# 内存中保留的最近调用明细条数
RECENT_CALL_LIMIT = 100


@dataclass(frozen=True, slots=True)
class CallMetrics:
    """单次调用指标：Token 分类消耗、总延迟与首 Token 延迟。"""

    model: str  # 调用使用的模型 id
    stream: bool  # 是否为流式调用
    success: bool  # 调用是否成功
    latency_ms: float  # 本次调用总耗时（毫秒）
    first_token_ms: float | None = None  # 首 Token 延迟（毫秒），非流式为 None
    usage: TokenUsage | None = None  # Token 分类消耗，上游未返回时为 None
    retries: int = 0  # 本次调用因瞬时故障重试的次数
    error_code: str | None = None  # 失败时的错误码

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
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


class MetricsService:
    """进程内指标记录：每次调用写结构化日志，并维护全局与按模型的聚合统计。"""

    def __init__(self, recent_limit: int = RECENT_CALL_LIMIT):
        self._total = _Aggregate()  # 全局累计指标
        self._by_model: dict[str, _Aggregate] = {}  # 按模型分组的累计指标
        self._recent: deque[dict[str, Any]] = deque(maxlen=recent_limit)  # 最近调用明细
        self._lock = threading.Lock()  # 保护累计指标与明细的并发访问

    def record(self, call: CallMetrics) -> None:
        """记录一次调用；日志便于外部采集，聚合数据供 /metrics 查询。"""
        payload = call.to_dict()
        logger.info("llm_call %s", json.dumps(payload, ensure_ascii=False))
        with self._lock:
            self._add(self._total, call)
            self._add(self._by_model.setdefault(call.model, _Aggregate()), call)
            self._recent.append(payload)

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
    def _add(aggregate: _Aggregate, call: CallMetrics) -> None:
        aggregate.calls += 1
        if call.success:
            aggregate.successes += 1
        else:
            aggregate.failures += 1
        if call.stream:
            aggregate.stream_calls += 1
        else:
            aggregate.complete_calls += 1

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
            },
            "tokens": {
                "calls_with_usage": aggregate.usage_calls,
                "prompt": aggregate.prompt_tokens,
                "completion": aggregate.completion_tokens,
                "total": aggregate.total_tokens,
                "cached": aggregate.cached_tokens,
                "reasoning": aggregate.reasoning_tokens,
            },
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
