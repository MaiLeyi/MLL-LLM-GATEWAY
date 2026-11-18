import json
import logging
from typing import Any

from app.config.tenancy import DEFAULT_USER_ID, DEFAULT_WORKSPACE_ID
from app.core.time import utc_now_iso
from app.repositories import repositories
from app.repositories.base import AuditEvent, AuditRepository

logger = logging.getLogger(__name__)


class AuditService:
    """追加式审计日志：记录配置与数据的关键变更，供后续审计页面或外部采集。"""

    def __init__(self, repository: AuditRepository | None = None) -> None:
        self._repository = repository or repositories.audit

    def record(
        self,
        action: str,
        resource_type: str,
        *,
        resource_id: str | None = None,
        request_id: str | None = None,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        actor_id: str = DEFAULT_USER_ID,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """写入一条审计事件；审计失败不阻断业务流程，只记录警告。"""
        try:
            self._repository.insert(
                AuditEvent(
                    workspace_id=workspace_id,
                    actor_id=actor_id,
                    timestamp=utc_now_iso(),
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    request_id=request_id,
                    detail=json.dumps(detail, ensure_ascii=False)
                    if detail is not None
                    else None,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 审计不能影响主流程
            logger.warning("写入审计日志失败: %s", exc)


audit_service = AuditService()
