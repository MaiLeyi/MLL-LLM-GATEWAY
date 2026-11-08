from fastapi import APIRouter

from app.config.settings import DEFAULT_MODEL, MODEL_CONFIGS

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models() -> dict:
    """返回可用模型列表与默认模型，供前端页面渲染模型选择器。"""
    return {"models": list(MODEL_CONFIGS.keys()), "default": DEFAULT_MODEL}
