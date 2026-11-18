import os
from dataclasses import dataclass

from app.repositories.base import (
    AuditRepository,
    ConversationRepository,
    MetricsRepository,
    PromptRepository,
    RunRepository,
)


@dataclass(frozen=True, slots=True)
class Repositories:
    """业务层依赖的仓储集合；切换数据库时只替换这里的实现。"""

    conversations: ConversationRepository
    runs: RunRepository
    prompts: PromptRepository
    metrics: MetricsRepository
    audit: AuditRepository


def build_repositories(backend: str | None = None) -> Repositories:
    """按 MLL_DB_BACKEND 构建仓储实现；默认 SQLite。"""
    resolved = (backend or os.getenv("MLL_DB_BACKEND", "sqlite")).strip().lower()
    if resolved == "sqlite":
        from app.repositories.sqlite import (
            SqliteAuditRepository,
            SqliteConversationRepository,
            SqliteMetricsRepository,
            SqlitePromptRepository,
            SqliteRunRepository,
        )

        return Repositories(
            conversations=SqliteConversationRepository(),
            runs=SqliteRunRepository(),
            prompts=SqlitePromptRepository(),
            metrics=SqliteMetricsRepository(),
            audit=SqliteAuditRepository(),
        )
    if resolved in {"postgres", "postgresql"}:
        raise RuntimeError(
            "PostgreSQL Repository 尚未实现：新增 app/repositories/postgres.py "
            "并在 build_repositories 中注册后，将 MLL_DB_BACKEND 设为 postgres"
        )
    raise RuntimeError(f"未知数据库后端: {resolved}")


repositories = build_repositories()
