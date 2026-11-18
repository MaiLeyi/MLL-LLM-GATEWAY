from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.chat import Usage
from app.schemas.prompt import PromptSelection


class ConversationCreate(BaseModel):
    """新建历史对话：标题可留空，首条消息会自动补标题。"""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="", max_length=60, description="会话标题")
    model: str | None = Field(
        default=None, max_length=50, description="上次使用的模型 id"
    )
    prompt: PromptSelection | None = Field(
        default=None, description="上次引用的 Prompt 模板版本与变量"
    )


class ConversationRename(BaseModel):
    """重命名历史对话。"""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=60, description="新标题")


class ConversationMessage(BaseModel):
    """历史对话中的单条消息；助手消息附带模型、请求 ID 与 Token 用量。"""

    id: int
    run_id: str | None = None
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    content_type: str = "text"
    model: str | None = None
    request_id: str | None = None
    usage: Usage | None = None
    latency_ms: int | None = None
    created_at: str


class ConversationSummary(BaseModel):
    """会话摘要：用于 History 列表，不含完整消息。"""

    id: str
    title: str
    model: str | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    prompt_variables: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    message_count: int = 0
    last_message: str | None = None


class ConversationDetail(ConversationSummary):
    """会话详情：摘要 + 按时间顺序的完整消息。"""

    messages: list[ConversationMessage] = Field(default_factory=list)


class ConversationListResponse(BaseModel):
    """会话分页列表。"""

    items: list[ConversationSummary]
    total: int
    limit: int
    offset: int
