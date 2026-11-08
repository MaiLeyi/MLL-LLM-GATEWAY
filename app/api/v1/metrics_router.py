from typing import Any

from fastapi import APIRouter

from app.services.metrics_service import metrics_service

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    """查询进程内累计指标：Token 分类消耗、延迟、首 Token 延迟与最近调用明细。"""
    return metrics_service.snapshot()
