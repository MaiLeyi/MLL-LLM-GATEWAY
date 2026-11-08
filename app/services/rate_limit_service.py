import asyncio
import math
import time
from collections import defaultdict, deque

from app.config.settings import MODEL_CONFIGS, RATE_LIMIT_WINDOW_SECONDS
from app.core.exceptions import RateLimitedException


class RateLimitService:
    """按模型独立限流：进程内滑动窗口，超限抛出 429。"""

    def __init__(self, window_seconds: float = RATE_LIMIT_WINDOW_SECONDS):
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, model_id: str) -> None:
        """校验该模型是否还有配额，超限时抛出 Retry-After 提示的 429。"""
        # 1. 读取该模型的限流配置
        config = MODEL_CONFIGS[model_id]
        if config.rate_limit_per_minute <= 0:
            # 1.1 限额 <= 0 表示不限制，直接放行
            return

        # 2. 在滑动窗口内申请一次配额
        retry_after = await self._acquire(model_id, config.rate_limit_per_minute)

        # 3. 超限：转换为 429，并带上 Retry-After
        if retry_after > 0:
            seconds = max(1, math.ceil(retry_after))
            raise RateLimitedException(
                f"模型 {model_id} 请求过于频繁，限额 "
                f"{config.rate_limit_per_minute} 次/分钟，请 {seconds} 秒后重试",
                headers={"Retry-After": str(seconds)},
            )

    async def _acquire(self, key: str, limit: int) -> float:
        """放行返回 0；超限返回建议的 Retry-After 秒数。"""
        # 1. 取当前时间并加锁，保证并发下计数准确
        now = time.monotonic()
        async with self._lock:
            hits = self._hits[key]
            # 2. 淘汰滑动窗口之外的历史请求
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            # 3. 已达上限：返回需要等待的秒数
            if len(hits) >= limit:
                return self._window - (now - hits[0])
            # 4. 未达上限：记录本次请求并放行
            hits.append(now)
            return 0.0


rate_limit_service = RateLimitService()
