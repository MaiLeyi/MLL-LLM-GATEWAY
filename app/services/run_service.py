import json
from typing import Any
from uuid import uuid4

from app.config.tenancy import DEFAULT_USER_ID, DEFAULT_WORKSPACE_ID
from app.core.time import utc_now_iso
from app.repositories import repositories
from app.repositories.base import RunEventRecord, RunFinish, RunRecord, RunRepository

RUN_STATUSES = {"running", "succeeded", "failed", "cancelled"}


class RunService:
    """Run 生命周期：一次问答/Agent 执行对应一个 Run，并记录追加式事件流。"""

    def __init__(self, repository: RunRepository | None = None) -> None:
        self._repository = repository or repositories.runs

    def start(
        self,
        *,
        request_id: str,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        conversation_id: str | None = None,
        created_by: str = DEFAULT_USER_ID,
        model: str | None = None,
        prompt_snapshot: dict[str, Any] | None = None,
    ) -> str:
        """创建 running 状态的 Run；同一 request_id 只对应一个 Run。"""
        run_id = str(uuid4())
        now = utc_now_iso()
        self._repository.create(
            RunRecord(
                id=run_id,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                created_by=created_by,
                request_id=request_id,
                model=model,
                prompt_snapshot=(
                    json.dumps(prompt_snapshot, ensure_ascii=False)
                    if prompt_snapshot
                    else None
                ),
                started_at=now,
            )
        )
        self.append_event(run_id, "run.started", {"model": model})
        return run_id

    def finish(
        self,
        run_id: str,
        *,
        status: str,
        model: str | None = None,
        usage: dict[str, int] | None = None,
        latency_ms: int | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """结束 Run：成功/失败/取消都统一落状态与汇总信息。"""
        if status not in RUN_STATUSES:
            raise ValueError(f"非法的 Run 状态: {status}")
        self._repository.finish(
            RunFinish(
                run_id=run_id,
                status=status,
                model=model,
                usage_json=(
                    json.dumps(usage, ensure_ascii=False) if usage else None
                ),
                latency_ms=latency_ms,
                error_code=error_code,
                error_message=error_message,
                finished_at=utc_now_iso(),
            )
        )
        event_type = "run.completed" if status == "succeeded" else "run.failed"
        self.append_event(
            run_id,
            event_type,
            {
                "status": status,
                "model": model,
                "usage": usage,
                "latency_ms": latency_ms,
                "error_code": error_code,
                "error_message": error_message,
            },
        )

    def append_event(
        self, run_id: str, event_type: str, payload: dict[str, Any] | None = None
    ) -> None:
        """追加一条 Run 事件；seq 由 Repository 在同一 Run 内严格递增。"""
        self._repository.append_event(
            RunEventRecord(
                run_id=run_id,
                type=event_type,
                payload=json.dumps(payload, ensure_ascii=False) if payload else None,
                created_at=utc_now_iso(),
            )
        )

    def get(self, run_id: str) -> dict[str, Any] | None:
        """读取 Run 详情与事件流，供后续 Run 回放/排障使用。"""
        row = self._repository.get(run_id)
        if row is None:
            return None
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "request_id": row["request_id"],
            "status": row["status"],
            "model": row["model"],
            "prompt_snapshot": (
                json.loads(row["prompt_snapshot"]) if row["prompt_snapshot"] else None
            ),
            "usage": json.loads(row["usage_json"]) if row["usage_json"] else None,
            "latency_ms": row["latency_ms"],
            "error_code": row["error_code"],
            "error_message": row["error_message"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "events": [
                {
                    "seq": event["seq"],
                    "type": event["type"],
                    "payload": json.loads(event["payload"])
                    if event["payload"]
                    else None,
                    "created_at": event["created_at"],
                }
                for event in self._repository.list_events(run_id)
            ],
        }


run_service = RunService()
