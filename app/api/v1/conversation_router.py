from fastapi import APIRouter, Query

from app.schemas.conversation import (
    ConversationCreate,
    ConversationDetail,
    ConversationListResponse,
    ConversationRename,
    ConversationSummary,
)
from app.services.conversation_service import conversation_service

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = Query(50, ge=1, le=200, description="分页条数"),
    offset: int = Query(0, ge=0, description="偏移量"),
) -> ConversationListResponse:
    """按最近更新时间倒序返回历史对话摘要。"""
    return ConversationListResponse(**conversation_service.list(limit=limit, offset=offset))


@router.post("", response_model=ConversationSummary, status_code=201)
async def create_conversation(payload: ConversationCreate) -> ConversationSummary:
    """创建历史对话；后续 /chat 与 /stream 传入 conversation_id 即可写入。"""
    return ConversationSummary(
        **conversation_service.create(
            title=payload.title, model=payload.model, prompt=payload.prompt
        )
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str) -> ConversationDetail:
    """读取会话详情与完整消息，用于 History 页查看或继续对话。"""
    return ConversationDetail(**conversation_service.get(conversation_id))


@router.patch("/{conversation_id}", response_model=ConversationSummary)
async def rename_conversation(
    conversation_id: str, payload: ConversationRename
) -> ConversationSummary:
    """重命名历史对话。"""
    return ConversationSummary(
        **conversation_service.rename(conversation_id, payload.title)
    )


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict[str, str]:
    """删除历史对话及其全部消息（消息随会话级联删除）。"""
    conversation_service.delete(conversation_id)
    return {"deleted": conversation_id}
