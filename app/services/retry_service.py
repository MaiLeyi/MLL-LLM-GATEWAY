import asyncio
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeVar

import httpx

from app.config.settings import MAX_RETRIES, RETRY_BASE_DELAY, RETRY_MAX_DELAY
from app.core.exceptions import (
    AppException,
    UpstreamException,
    UpstreamTimeoutException,
)

T = TypeVar("T")

# 这些上游状态码属于瞬时故障，适合退避重试；4xx 业务错误不重试
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class RetryService:
    """指数退避重试服务：只重试瞬时故障，最多重试 max_retries 次。"""

    def __init__(
        self,
        max_retries: int = MAX_RETRIES,
        base_delay: float = RETRY_BASE_DELAY,
        max_delay: float = RETRY_MAX_DELAY,
    ):
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

    def is_retryable(self, exc: BaseException) -> bool:
        """判断异常是否属于可重试的瞬时故障。"""
        # 1. 业务异常：仅上游瞬时故障可重试
        if isinstance(exc, AppException):
            return isinstance(exc, (UpstreamException, UpstreamTimeoutException))
        # 2. 网络层异常
        # 2.1 超时
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            return True
        # 2.2 连接失败
        if isinstance(exc, (httpx.TransportError, ConnectionError)):
            return True
        # 3. 携带 HTTP 状态码的异常：仅 408/429/5xx 可重试
        status = self._http_status(exc)
        if status is not None:
            return status in RETRYABLE_STATUS
        # 4. 兜底：按异常名判断超时/连接/限流类
        name = type(exc).__name__.lower()
        return any(
            key in name for key in ("timeout", "connection", "ratelimit", "rate_limit")
        )

    def backoff_delay(self, attempt: int) -> float:
        """第 attempt 次重试前的等待时间（attempt 从 1 开始）：指数增长 + 抖动。"""
        # 1. 指数增长：base * 2^(attempt-1)
        delay = self._base_delay * (2 ** (attempt - 1))
        # 2. 封顶，避免无限增长
        delay = min(delay, self._max_delay)
        # 3. 叠加 0~20% 抖动，避免并发请求同时重试
        return delay + random.uniform(0, delay * 0.2)

    async def run(self, fn: Callable[[], Awaitable[T]]) -> T:
        """执行一次性调用，失败时按指数退避重试。"""
        # 1. 记录已重试次数
        attempt = 0
        # 2. 循环执行，直到成功或达到重试上限
        while True:
            try:
                return await fn()
            except Exception as exc:
                # 2.1 不可重试或已达上限：直接抛出
                if attempt >= self._max_retries or not self.is_retryable(exc):
                    raise
                # 2.2 递增次数，指数退避后重新执行
                attempt += 1
                await asyncio.sleep(self.backoff_delay(attempt))

    async def iterate(
        self, factory: Callable[[], AsyncIterator[T]]
    ) -> AsyncIterator[T]:
        """执行流式调用；仅在产出首个片段前重试，避免已输出内容重复。"""
        # 1. 记录已重试次数
        attempt = 0
        # 2. 循环建立流，直到正常结束或达到重试上限
        while True:
            # 2.1 标记本轮是否已产出片段
            emitted = False
            try:
                async for item in factory():
                    # 2.2 已产出片段后不再允许重试，避免内容重复
                    emitted = True
                    yield item
                return
            except Exception as exc:
                # 2.3 已产出 / 不可重试 / 达到上限：直接抛出
                if emitted or attempt >= self._max_retries or not self.is_retryable(exc):
                    raise
                # 2.4 递增次数，指数退避后重建流
                attempt += 1
                await asyncio.sleep(self.backoff_delay(attempt))

    @staticmethod
    def _http_status(exc: BaseException) -> int | None:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status if isinstance(status, int) else None


retry_service = RetryService()
