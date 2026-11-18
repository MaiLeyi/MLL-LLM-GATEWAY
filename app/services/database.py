import logging
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config.paths import database_path
from app.config.tenancy import DEFAULT_USER_ID, DEFAULT_WORKSPACE_ID
from app.core.time import utc_now_iso

logger = logging.getLogger(__name__)

# 当前 schema 版本，后续结构变更通过 user_version 做增量迁移
SCHEMA_VERSION = 1

# 多用户能力接入前，所有数据先归属默认用户与默认工作区
DEFAULT_USER_ID = "local-user"
DEFAULT_WORKSPACE_ID = "default"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    email         TEXT UNIQUE,
    display_name  TEXT,
    status        TEXT NOT NULL DEFAULT 'active',
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workspaces (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    owner_id    TEXT,
    settings    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS workspace_members (
    workspace_id  TEXT NOT NULL,
    user_id       TEXT NOT NULL,
    role          TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (workspace_id, user_id),
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS conversations (
    id                TEXT PRIMARY KEY,
    workspace_id      TEXT NOT NULL,
    created_by        TEXT,
    title             TEXT NOT NULL,
    mode              TEXT NOT NULL DEFAULT 'chat',
    model             TEXT,
    prompt_name       TEXT,
    prompt_version    TEXT,
    prompt_variables  TEXT,
    status            TEXT NOT NULL DEFAULT 'active',
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_conversations_workspace_updated
    ON conversations(workspace_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS runs (
    id               TEXT PRIMARY KEY,
    workspace_id     TEXT NOT NULL,
    conversation_id  TEXT,
    created_by       TEXT,
    request_id       TEXT NOT NULL,
    status           TEXT NOT NULL,
    model            TEXT,
    prompt_snapshot  TEXT,
    usage_json       TEXT,
    latency_ms       INTEGER,
    error_code       TEXT,
    error_message    TEXT,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE SET NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_request_id ON runs(request_id);
CREATE INDEX IF NOT EXISTS idx_runs_conversation
    ON runs(conversation_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_workspace_status
    ON runs(workspace_id, status);

CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL,
    run_id           TEXT,
    role             TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content          TEXT NOT NULL,
    content_type     TEXT NOT NULL DEFAULT 'text',
    model            TEXT,
    request_id       TEXT,
    usage_json       TEXT,
    latency_ms       INTEGER,
    parent_id        TEXT,
    metadata         TEXT,
    created_at       TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_messages_run ON messages(run_id);

CREATE TABLE IF NOT EXISTS tool_calls (
    id               TEXT PRIMARY KEY,
    workspace_id     TEXT NOT NULL,
    conversation_id  TEXT,
    run_id           TEXT NOT NULL,
    tool_name        TEXT NOT NULL,
    arguments_json   TEXT,
    result_json      TEXT,
    artifact_id      TEXT,
    status           TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    error            TEXT,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE,
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id);

CREATE TABLE IF NOT EXISTS artifacts (
    id               TEXT PRIMARY KEY,
    workspace_id     TEXT NOT NULL,
    conversation_id  TEXT,
    run_id           TEXT,
    kind             TEXT NOT NULL,
    name             TEXT NOT NULL,
    mime             TEXT,
    storage_key      TEXT NOT NULL,
    size_bytes       INTEGER NOT NULL DEFAULT 0,
    sha256           TEXT,
    created_by       TEXT,
    created_at       TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS prompts (
    workspace_id  TEXT NOT NULL,
    name          TEXT NOT NULL,
    version       TEXT NOT NULL,
    template      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (workspace_id, name, version),
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id       TEXT NOT NULL,
    conversation_id    TEXT,
    run_id             TEXT,
    timestamp          TEXT NOT NULL,
    request_id         TEXT NOT NULL,
    requested_model    TEXT,
    model              TEXT NOT NULL,
    stream             INTEGER NOT NULL,
    success            INTEGER NOT NULL,
    latency_ms         REAL NOT NULL,
    first_token_ms     REAL,
    has_usage          INTEGER NOT NULL DEFAULT 0,
    prompt_tokens      INTEGER NOT NULL DEFAULT 0,
    completion_tokens  INTEGER NOT NULL DEFAULT 0,
    total_tokens       INTEGER NOT NULL DEFAULT 0,
    cached_tokens      INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens   INTEGER NOT NULL DEFAULT 0,
    retries            INTEGER NOT NULL DEFAULT 0,
    error_code         TEXT,
    cost_usd           REAL,
    prompt_name        TEXT,
    prompt_version     TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_timestamp
    ON llm_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_calls_workspace
    ON llm_calls(workspace_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id);

CREATE TABLE IF NOT EXISTS run_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    type        TEXT NOT NULL,
    payload     TEXT,
    created_at  TEXT NOT NULL,
    UNIQUE (run_id, seq),
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events(run_id, seq);

CREATE TABLE IF NOT EXISTS audit_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id   TEXT,
    actor_id       TEXT,
    timestamp      TEXT NOT NULL,
    action         TEXT NOT NULL,
    resource_type  TEXT NOT NULL,
    resource_id    TEXT,
    request_id     TEXT,
    detail         TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_workspace
    ON audit_events(workspace_id, timestamp DESC);
"""


class Database:
    """单文件 SQLite 连接：WAL 模式 + 进程内锁，仅作为 Repository 的 SQLite 基础设施。

    FastAPI 的异步端点会直接调用同步方法，写入都很短；统一用一把可重入锁
    串行化访问，避免多线程共享连接时的竞争。业务层只依赖 Repository 协议，
    切 PostgreSQL 时新增 app/repositories/postgres.py 即可，不需要改 Service。
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = str(path) if path is not None else str(database_path())
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._seed_defaults()
            self._conn.commit()
        if self._path != ":memory:":
            try:
                os.chmod(self._path, 0o600)
            except OSError:
                logger.debug("设置数据库文件权限失败: %s", self._path)

    @property
    def path(self) -> str:
        return self._path

    def _seed_defaults(self) -> None:
        """种下默认用户与默认工作区，让多用户接入前的单用户控制台可直接使用。"""
        now = utc_now_iso()
        self._conn.execute(
            "INSERT OR IGNORE INTO users (id, email, display_name, status, created_at)"
            " VALUES (?, ?, ?, 'active', ?)",
            (DEFAULT_USER_ID, None, "本地用户", now),
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO workspaces"
            " (id, name, owner_id, settings, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (DEFAULT_WORKSPACE_ID, "默认工作区", DEFAULT_USER_ID, None, now, now),
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO workspace_members"
            " (workspace_id, user_id, role, created_at) VALUES (?, ?, 'owner', ?)",
            (DEFAULT_WORKSPACE_ID, DEFAULT_USER_ID, now),
        )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """多条写入的原子事务；异常时回滚。"""
        with self._lock:
            cursor = self._conn.cursor()
            try:
                yield cursor
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def execute(
        self, sql: str, params: tuple | list = ()
    ) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            return cursor

    def query(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def query_one(
        self, sql: str, params: tuple | list = ()
    ) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


database = Database()
