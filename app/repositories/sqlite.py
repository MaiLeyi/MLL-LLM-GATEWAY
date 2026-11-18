from __future__ import annotations

import sqlite3
from typing import Any

from app.repositories.base import (
    AuditEvent,
    ConversationExchange,
    DuplicateRecordError,
    LlmCall,
    NewConversation,
    PromptVersion,
    RunEventRecord,
    RunFinish,
    RunRecord,
)
from app.services.database import database


class SqliteConversationRepository:
    """会话聚合的 SQLite 实现；SQL 只出现在 Repository 层。"""

    _SUMMARY_SQL = """
        SELECT c.*,
               (SELECT COUNT(*) FROM messages m
                 WHERE m.conversation_id = c.id) AS message_count,
               (SELECT substr(m.content, 1, 120) FROM messages m
                 WHERE m.conversation_id = c.id
                 ORDER BY m.id DESC LIMIT 1) AS last_message
          FROM conversations c
    """

    def create(self, record: NewConversation) -> None:
        database.execute(
            """
            INSERT INTO conversations
                (id, workspace_id, created_by, title, mode, model,
                 prompt_name, prompt_version, prompt_variables,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, 'chat', ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.workspace_id,
                record.created_by,
                record.title,
                record.model,
                record.prompt_name,
                record.prompt_version,
                record.prompt_variables,
                record.created_at,
                record.updated_at,
            ),
        )

    def list(
        self, *, workspace_id: str, limit: int, offset: int
    ) -> tuple[list[dict[str, Any]], int]:
        rows = database.query(
            self._SUMMARY_SQL
            + " WHERE c.workspace_id = ?"
            " ORDER BY c.updated_at DESC, c.id DESC LIMIT ? OFFSET ?",
            (workspace_id, limit, offset),
        )
        total_row = database.query_one(
            "SELECT COUNT(*) AS total FROM conversations WHERE workspace_id = ?",
            (workspace_id,),
        )
        return [dict(row) for row in rows], int(total_row["total"]) if total_row else 0

    def get(
        self, conversation_id: str, *, workspace_id: str
    ) -> dict[str, Any] | None:
        row = database.query_one(
            self._SUMMARY_SQL + " WHERE c.id = ? AND c.workspace_id = ?",
            (conversation_id, workspace_id),
        )
        return dict(row) if row else None

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = database.query(
            """
            SELECT id, run_id, role, content, content_type, model, request_id,
                   usage_json, latency_ms, parent_id, metadata, created_at
              FROM messages
             WHERE conversation_id = ?
             ORDER BY id
            """,
            (conversation_id,),
        )
        return [dict(row) for row in rows]

    def rename(
        self,
        conversation_id: str,
        *,
        workspace_id: str,
        title: str,
        updated_at: str,
    ) -> bool:
        cursor = database.execute(
            "UPDATE conversations SET title = ?, updated_at = ?"
            " WHERE id = ? AND workspace_id = ?",
            (title, updated_at, conversation_id, workspace_id),
        )
        return cursor.rowcount > 0

    def delete(self, conversation_id: str, *, workspace_id: str) -> bool:
        cursor = database.execute(
            "DELETE FROM conversations WHERE id = ? AND workspace_id = ?",
            (conversation_id, workspace_id),
        )
        return cursor.rowcount > 0

    def append_exchange(self, exchange: ConversationExchange) -> bool:
        with database.transaction() as cursor:
            row = cursor.execute(
                "SELECT id FROM conversations WHERE id = ? AND workspace_id = ?",
                (exchange.conversation_id, exchange.workspace_id),
            ).fetchone()
            if row is None:
                return False

            cursor.execute(
                """
                INSERT INTO messages
                    (conversation_id, run_id, role, content, content_type,
                     model, request_id, usage_json, latency_ms, created_at)
                VALUES (?, ?, ?, ?, 'text', ?, ?, ?, ?, ?)
                """,
                (
                    exchange.conversation_id,
                    exchange.run_id,
                    exchange.user_role,
                    exchange.user_content,
                    None,
                    exchange.request_id,
                    None,
                    None,
                    exchange.updated_at,
                ),
            )
            cursor.execute(
                """
                INSERT INTO messages
                    (conversation_id, run_id, role, content, content_type,
                     model, request_id, usage_json, latency_ms, created_at)
                VALUES (?, ?, 'assistant', ?, 'text', ?, ?, ?, ?, ?)
                """,
                (
                    exchange.conversation_id,
                    exchange.run_id,
                    exchange.assistant_content,
                    exchange.assistant_model,
                    exchange.request_id,
                    exchange.usage_json,
                    exchange.latency_ms,
                    exchange.updated_at,
                ),
            )
            cursor.execute(
                """
                UPDATE conversations
                   SET title = CASE
                                 WHEN ? IS NOT NULL AND title IN ('新对话', '')
                                 THEN ? ELSE title
                               END,
                       model = COALESCE(?, model),
                       prompt_name = COALESCE(?, prompt_name),
                       prompt_version = COALESCE(?, prompt_version),
                       prompt_variables = COALESCE(?, prompt_variables),
                       updated_at = ?
                 WHERE id = ? AND workspace_id = ?
                """,
                (
                    exchange.auto_title,
                    exchange.auto_title,
                    exchange.model,
                    exchange.prompt_name,
                    exchange.prompt_version,
                    exchange.prompt_variables,
                    exchange.updated_at,
                    exchange.conversation_id,
                    exchange.workspace_id,
                ),
            )
        return True


class SqliteRunRepository:
    """Run 与事件流的 SQLite 实现。"""

    def create(self, record: RunRecord) -> None:
        database.execute(
            """
            INSERT INTO runs
                (id, workspace_id, conversation_id, created_by, request_id,
                 status, model, prompt_snapshot, started_at)
            VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)
            """,
            (
                record.id,
                record.workspace_id,
                record.conversation_id,
                record.created_by,
                record.request_id,
                record.model,
                record.prompt_snapshot,
                record.started_at,
            ),
        )

    def finish(self, record: RunFinish) -> None:
        database.execute(
            """
            UPDATE runs
               SET status = ?, model = COALESCE(?, model),
                   usage_json = ?, latency_ms = ?, error_code = ?,
                   error_message = ?, finished_at = ?
             WHERE id = ?
            """,
            (
                record.status,
                record.model,
                record.usage_json,
                record.latency_ms,
                record.error_code,
                record.error_message,
                record.finished_at,
                record.run_id,
            ),
        )

    def append_event(self, event: RunEventRecord) -> None:
        with database.transaction() as cursor:
            row = cursor.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq"
                " FROM run_events WHERE run_id = ?",
                (event.run_id,),
            ).fetchone()
            cursor.execute(
                """
                INSERT INTO run_events (run_id, seq, type, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.run_id,
                    int(row["next_seq"]),
                    event.type,
                    event.payload,
                    event.created_at,
                ),
            )

    def get(self, run_id: str) -> dict[str, Any] | None:
        row = database.query_one("SELECT * FROM runs WHERE id = ?", (run_id,))
        return dict(row) if row else None

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        rows = database.query(
            "SELECT seq, type, payload, created_at FROM run_events"
            " WHERE run_id = ? ORDER BY seq",
            (run_id,),
        )
        return [dict(row) for row in rows]


class SqlitePromptRepository:
    """模板版本的 SQLite 实现。"""

    def seed(self, records: list[PromptVersion]) -> None:
        for record in records:
            database.execute(
                """
                INSERT OR IGNORE INTO prompts
                    (workspace_id, name, version, template, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.workspace_id,
                    record.name,
                    record.version,
                    record.template,
                    record.created_at,
                ),
            )

    def list(self, *, workspace_id: str) -> list[dict[str, Any]]:
        rows = database.query(
            "SELECT workspace_id, name, version, template, created_at"
            " FROM prompts WHERE workspace_id = ?"
            " ORDER BY name, created_at, version",
            (workspace_id,),
        )
        return [dict(row) for row in rows]

    def get(
        self, *, workspace_id: str, name: str, version: str
    ) -> dict[str, Any] | None:
        row = database.query_one(
            "SELECT workspace_id, name, version, template, created_at"
            " FROM prompts WHERE workspace_id = ? AND name = ? AND version = ?",
            (workspace_id, name, version),
        )
        return dict(row) if row else None

    def latest(
        self, *, workspace_id: str, name: str
    ) -> dict[str, Any] | None:
        row = database.query_one(
            "SELECT workspace_id, name, version, template, created_at"
            " FROM prompts WHERE workspace_id = ? AND name = ?"
            " ORDER BY created_at DESC, version DESC LIMIT 1",
            (workspace_id, name),
        )
        return dict(row) if row else None

    def insert(self, record: PromptVersion) -> None:
        try:
            database.execute(
                """
                INSERT INTO prompts
                    (workspace_id, name, version, template, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.workspace_id,
                    record.name,
                    record.version,
                    record.template,
                    record.created_at,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise DuplicateRecordError(
                f"模板版本已存在: {record.name}@{record.version}"
            ) from exc


class SqliteMetricsRepository:
    """调用明细的 SQLite 实现。"""

    def insert_call(self, call: LlmCall) -> None:
        database.execute(
            """
            INSERT INTO llm_calls
                (workspace_id, conversation_id, run_id, timestamp, request_id,
                 requested_model, model, stream, success, latency_ms,
                 first_token_ms, has_usage, prompt_tokens, completion_tokens,
                 total_tokens, cached_tokens, reasoning_tokens, retries,
                 error_code, cost_usd, prompt_name, prompt_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                call.workspace_id,
                call.conversation_id,
                call.run_id,
                call.timestamp,
                call.request_id,
                call.requested_model,
                call.model,
                int(call.stream),
                int(call.success),
                call.latency_ms,
                call.first_token_ms,
                int(call.has_usage),
                call.prompt_tokens,
                call.completion_tokens,
                call.total_tokens,
                call.cached_tokens,
                call.reasoning_tokens,
                call.retries,
                call.error_code,
                call.cost_usd,
                call.prompt_name,
                call.prompt_version,
            ),
        )

    def list_calls(self) -> list[dict[str, Any]]:
        rows = database.query("SELECT * FROM llm_calls ORDER BY id")
        return [dict(row) for row in rows]


class SqliteAuditRepository:
    """审计日志的 SQLite 实现。"""

    def insert(self, event: AuditEvent) -> None:
        database.execute(
            """
            INSERT INTO audit_events
                (workspace_id, actor_id, timestamp, action, resource_type,
                 resource_id, request_id, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.workspace_id,
                event.actor_id,
                event.timestamp,
                event.action,
                event.resource_type,
                event.resource_id,
                event.request_id,
                event.detail,
            ),
        )
