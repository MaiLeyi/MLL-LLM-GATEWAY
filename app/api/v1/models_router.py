from fastapi import APIRouter

from app.application.model_test_service import model_test_service
from app.schemas.model import (
    ModelCard,
    ModelCreate,
    ModelListResponse,
    ModelTestResult,
    ModelUpsert,
)
from app.services.model_registry import ModelDraft, model_registry

router = APIRouter(tags=["models"])


def _to_draft(payload: ModelUpsert) -> ModelDraft:
    return ModelDraft(
        display_name=payload.display_name,
        provider_model=payload.provider_model,
        base_url=payload.base_url,
        protocol=payload.protocol,
        structured_output_mode=payload.structured_output_mode,
        rate_limit_per_minute=payload.rate_limit_per_minute,
        fallback_model=payload.fallback_model,
        input_price_per_million=payload.input_price_per_million,
        output_price_per_million=payload.output_price_per_million,
        api_key=payload.api_key,
    )


@router.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """返回模型 id 列表、默认模型与卡片详情，供 Playground 与 Models 页面使用。"""
    model_ids = model_registry.list_ids()
    return ModelListResponse(
        models=model_ids,
        default=model_registry.default_model,
        items=[ModelCard(**model_registry.describe(model_id)) for model_id in model_ids],
    )


@router.post("/models", response_model=ModelCard, status_code=201)
async def create_model(payload: ModelCreate) -> ModelCard:
    """新增模型配置；API Key 写入系统钥匙串，不写入配置文件。"""
    card = model_registry.create(payload.id, _to_draft(payload))
    return ModelCard(**card)


@router.put("/models/{model_id}", response_model=ModelCard)
async def update_model(model_id: str, payload: ModelUpsert) -> ModelCard:
    """更新模型配置；api_key 留空表示保留原 Key。"""
    card = model_registry.update(model_id, _to_draft(payload))
    return ModelCard(**card)


@router.delete("/models/{model_id}")
async def delete_model(model_id: str) -> dict[str, str]:
    """删除模型；默认模型或被引用为备用模型时拒绝。"""
    model_registry.delete(model_id)
    return {"deleted": model_id}


@router.post("/models/{model_id}/default", response_model=ModelCard)
async def set_default_model(model_id: str) -> ModelCard:
    """切换默认模型，Playground 未显式指定模型时使用它。"""
    return ModelCard(**model_registry.set_default(model_id))


@router.post("/models/{model_id}/test", response_model=ModelTestResult)
async def test_model(model_id: str) -> ModelTestResult:
    """真实调用上游做一次最小请求，验证地址、Key 与模型名是否可用。"""
    return await model_test_service.test(model_id)
