import json
from typing import Any
from uuid import uuid4

from app.config.tenancy import DEFAULT_USER_ID, DEFAULT_WORKSPACE_ID
from app.core.exceptions import InvalidRequestException, NotFoundException
from app.core.time import utc_now_iso
from app.repositories import repositories
from app.repositories.base import ConversationExchange, ConversationRepository, NewConversation
from app.schemas.prompt import PromptSelection
from app.services.audit_service import audit_service
from app.services.model_registry import model_registry

DEFAULT_TITLE = "新对话"
TITLE_MAX_LENGTH = 60


def _normalize_title(title: str | None) -> str:
    """标题统一去空白并截断，空标题回退为默认值。"""
    text = " ".join((title or "").split())
    return text[:TITLE_MAX_LENGTH] or DEFAULT_TITLE


def _title_from_message(content: str) -> str:
    """用首条用户消息生成会话标题。"""
    return _normalize_title(content)


class ConversationService:
    """历史对话服务：只负责业务规则，数据访问全部交给 Repository。"""

    def __init__(self, repository: ConversationRepository | None = None) -> None:
        self._repository = repository or repositories.conversations

    @staticmethod
    def _variables(raw: str | None) -> dict[str, str]:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def _summary(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "model": row["model"],
            "prompt_name": row["prompt_name"],
            "prompt_version": row["prompt_version"],
            "prompt_variables": self._variables(row["prompt_variables"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": int(row["message_count"]),
            "last_message": row["last_message"],
        }

    def create(
        self,
        *,
        title: str | None = None,
        model: str | None = None,
        prompt: PromptSelection | None = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        created_by: str = DEFAULT_USER_ID,
    ) -> dict[str, Any]:
        """创建空会话；标题、模型与模板引用会随会话保存，便于恢复现场。"""
        if model:
            model_registry.require(model)
        conversation_id = str(uuid4())
        now = utc_now_iso()
        self._repository.create(
            NewConversation(
                id=conversation_id,
                workspace_id=workspace_id,
                created_by=created_by,
                title=_normalize_title(title),
                model=model,
                prompt_name=prompt.name if prompt else None,
                prompt_version=prompt.version if prompt else None,
                prompt_variables=(
                    json.dumps(prompt.variables, ensure_ascii=False) if prompt else None
                ),
                created_at=now,
                updated_at=now,
            )
        )
        audit_service.record(
            "conversation.create",
            "conversation",
            resource_id=conversation_id,
            workspace_id=workspace_id,
            actor_id=created_by,
            detail={"model": model, "title": _normalize_title(title)},
        )
        return self.require(conversation_id, workspace_id=workspace_id)

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> dict[str, Any]:
        """按最近更新时间倒序分页返回工作区内的会话摘要。"""
        rows, total = self._repository.list(
            workspace_id=workspace_id, limit=limit, offset=offset
        )
        return {
            "items": [self._summary(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def require(
        self, conversation_id: str, *, workspace_id: str = DEFAULT_WORKSPACE_ID
    ) -> dict[str, Any]:
        """取会话摘要，不存在或不属于当前工作区时按统一错误码返回 404。"""
        row = self._repository.get(conversation_id, workspace_id=workspace_id)
        if row is None:
            raise NotFoundException(f"历史对话不存在: {conversation_id}")
        return self._summary(row)

    def get(
        self, conversation_id: str, *, workspace_id: str = DEFAULT_WORKSPACE_ID
    ) -> dict[str, Any]:
        """会话详情：摘要字段 + 按写入顺序排列的完整消息。"""
        summary = self.require(conversation_id, workspace_id=workspace_id)
        messages = []
        for row in self._repository.list_messages(conversation_id):
            usage = None
            if row["usage_json"]:
                try:
                    usage = json.loads(row["usage_json"])
                except json.JSONDecodeError:
                    usage = None
            messages.append(
                {
                    "id": row["id"],
                    "run_id": row["run_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "content_type": row["content_type"],
                    "model": row["model"],
                    "request_id": row["request_id"],
                    "usage": usage,
                    "latency_ms": row["latency_ms"],
                    "created_at": row["created_at"],
                }
            )
        return {**summary, "messages": messages}

    def rename(
        self,
        conversation_id: str,
        title: str,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> dict[str, Any]:
        if not title.strip():
            raise InvalidRequestException("对话标题不能为空")
        normalized = _normalize_title(title)
        if not self._repository.rename(
            conversation_id,
            workspace_id=workspace_id,
            title=normalized,
            updated_at=utc_now_iso(),
        ):
            raise NotFoundException(f"历史对话不存在: {conversation_id}")
        audit_service.record(
            "conversation.rename",
            "conversation",
            resource_id=conversation_id,
            workspace_id=workspace_id,
            detail={"title": normalized},
        )
        return self.require(conversation_id, workspace_id=workspace_id)

    def delete(
        self, conversation_id: str, *, workspace_id: str = DEFAULT_WORKSPACE_ID
    ) -> None:
        if not self._repository.delete(conversation_id, workspace_id=workspace_id):
            raise NotFoundException(f"历史对话不存在: {conversation_id}")
        audit_service.record(
            "conversation.delete",
            "conversation",
            resource_id=conversation_id,
            workspace_id=workspace_id,
        )

    def append_exchange(
        self,
        conversation_id: str,
        *,
        user_role: str,
        user_content: str,
        assistant_content: str,
        assistant_model: str,
        request_id: str,
        usage: dict[str, int] | None = None,
        latency_ms: int | None = None,
        prompt: PromptSelection | None = None,
        model: str | None = None,
        run_id: str | None = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> None:
        """把一次成功的问答原子写入会话：用户消息 + 助手回复 + 会话元信息更新。"""
        auto_title = (
            _title_from_message(user_content) if user_role == "user" else None
        )
        if not self._repository.append_exchange(
            ConversationExchange(
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                run_id=run_id,
                user_role=user_role,
                user_content=user_content,
                assistant_content=assistant_content,
                assistant_model=assistant_model,
                request_id=request_id,
                usage_json=(
                    json.dumps(usage, ensure_ascii=False) if usage else None
                ),
                latency_ms=latency_ms,
                auto_title=auto_title,
                model=model,
                prompt_name=prompt.name if prompt else None,
                prompt_version=prompt.version if prompt else None,
                prompt_variables=(
                    json.dumps(prompt.variables, ensure_ascii=False) if prompt else None
                ),
                updated_at=utc_now_iso(),
            )
        ):
            raise NotFoundException(f"历史对话不存在: {conversation_id}")


conversation_service = ConversationService()
