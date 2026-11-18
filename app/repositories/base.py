from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class DuplicateRecordError(Exception):
    """唯一键冲突；由 Repository 抛出，Service 转换成业务错误。"""


@dataclass(frozen=True, slots=True)
class NewConversation:
    id: str
    workspace_id: str
    created_by: str
    title: str
    model: str | None
    prompt_name: str | None
    prompt_version: str | None
    prompt_variables: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class ConversationExchange:
    """一次成功问答的落库参数；auto_title 仅在首条用户消息时非空。"""

    conversation_id: str
    workspace_id: str
    run_id: str | None
    user_role: str
    user_content: str
    assistant_content: str
    assistant_model: str
    request_id: str
    usage_json: str | None
    latency_ms: int | None
    auto_title: str | None
    model: str | None
    prompt_name: str | None
    prompt_version: str | None
    prompt_variables: str | None
    updated_at: str


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    workspace_id: str
    conversation_id: str | None
    created_by: str
    request_id: str
    model: str | None
    prompt_snapshot: str | None
    started_at: str


@dataclass(frozen=True, slots=True)
class RunFinish:
    run_id: str
    status: str
    model: str | None
    usage_json: str | None
    latency_ms: int | None
    error_code: str | None
    error_message: str | None
    finished_at: str


@dataclass(frozen=True, slots=True)
class RunEventRecord:
    run_id: str
    type: str
    payload: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class PromptVersion:
    workspace_id: str
    name: str
    version: str
    template: str
    created_at: str


@dataclass(frozen=True, slots=True)
class LlmCall:
    workspace_id: str
    conversation_id: str | None
    run_id: str | None
    timestamp: str
    request_id: str
    requested_model: str | None
    model: str
    stream: bool
    success: bool
    latency_ms: float
    first_token_ms: float | None
    has_usage: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    retries: int
    error_code: str | None
    cost_usd: float | None
    prompt_name: str | None
    prompt_version: str | None


@dataclass(frozen=True, slots=True)
class AuditEvent:
    workspace_id: str | None
    actor_id: str | None
    timestamp: str
    action: str
    resource_type: str
    resource_id: str | None
    request_id: str | None
    detail: str | None


class ConversationRepository(Protocol):
    """会话聚合：会话元信息 + 消息。"""

    def create(self, record: NewConversation) -> None: ...

    def list(
        self, *, workspace_id: str, limit: int, offset: int
    ) -> tuple[list[dict[str, Any]], int]:
        """按 updated_at 倒序返回当前页，第二项为该工作区会话总数。"""
        ...

    def get(
        self, conversation_id: str, *, workspace_id: str
    ) -> dict[str, Any] | None:
        """返回会话摘要行（含 message_count / last_message），不存在返回 None。"""
        ...

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        """按写入顺序返回会话内全部消息。"""
        ...

    def rename(
        self,
        conversation_id: str,
        *,
        workspace_id: str,
        title: str,
        updated_at: str,
    ) -> bool: ...

    def delete(self, conversation_id: str, *, workspace_id: str) -> bool: ...

    def append_exchange(self, exchange: ConversationExchange) -> bool:
        """原子写入 user + assistant 两条消息，并更新会话元信息。"""
        ...


class RunRepository(Protocol):
    """Run 与追加式事件流。"""

    def create(self, record: RunRecord) -> None: ...

    def finish(self, record: RunFinish) -> None: ...

    def append_event(self, event: RunEventRecord) -> None:
        """追加事件；实现负责为同一 run_id 生成严格递增的 seq。"""
        ...

    def get(self, run_id: str) -> dict[str, Any] | None: ...

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        """按 seq 升序返回事件流。"""
        ...


class PromptRepository(Protocol):
    """模板版本仓库。"""

    def seed(self, records: list[PromptVersion]) -> None: ...

    def list(self, *, workspace_id: str) -> list[dict[str, Any]]:
        """按 name、created_at、version 升序返回；同名最后一条即 latest。"""
        ...

    def get(
        self, *, workspace_id: str, name: str, version: str
    ) -> dict[str, Any] | None: ...

    def latest(
        self, *, workspace_id: str, name: str
    ) -> dict[str, Any] | None:
        """返回该名称下 created_at 最新的版本。"""
        ...

    def insert(self, record: PromptVersion) -> None: ...


class MetricsRepository(Protocol):
    """上游调用明细仓库。"""

    def insert_call(self, call: LlmCall) -> None: ...

    def list_calls(self) -> list[dict[str, Any]]:
        """按写入顺序返回全部调用明细，用于启动时重建聚合指标。"""
        ...


class AuditRepository(Protocol):
    """追加式审计仓库。"""

    def insert(self, event: AuditEvent) -> None: ...
