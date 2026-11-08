from fastapi import APIRouter, Query

from app.config.prompt_config import LATEST_VERSION
from app.schemas.prompt import PromptTemplateCreate, PromptTemplateResponse
from app.services.prompt_service import prompt_service

router = APIRouter(prefix="/prompts", tags=["prompts"])


@router.post("", response_model=PromptTemplateResponse, status_code=201)
async def store_prompt(payload: PromptTemplateCreate) -> PromptTemplateResponse:
    """发布一个模板版本；同名称下版本不可覆盖，便于稳定做版本引用。"""
    return prompt_service.store(payload.name, payload.version, payload.template)


@router.get("", response_model=list[PromptTemplateResponse])
async def list_prompts(
    name: str | None = Query(None, description="按模板名称过滤，不传则返回全部"),
) -> list[PromptTemplateResponse]:
    """列出已存储的模板版本，并标注每个模板的最新版本。"""
    return prompt_service.list_templates(name)


@router.get("/{name}/{version}", response_model=PromptTemplateResponse)
async def get_prompt(
    name: str,
    version: str = LATEST_VERSION,
) -> PromptTemplateResponse:
    """按名称与版本查询模板详情；version 传 latest 时返回最新发布版本。"""
    return prompt_service.get_template(name, version)
