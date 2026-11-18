from dataclasses import dataclass
from datetime import datetime
from string import Template

from app.config.prompt_config import DEFAULT_PROMPT_TEMPLATES, LATEST_VERSION
from app.config.tenancy import DEFAULT_WORKSPACE_ID
from app.core.exceptions import InvalidRequestException, NotFoundException
from app.core.time import utc_now_iso
from app.repositories import repositories
from app.repositories.base import DuplicateRecordError, PromptRepository, PromptVersion
from app.schemas.prompt import PromptSelection, PromptTemplateResponse
from app.services.audit_service import audit_service


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """单个模板版本的不可变快照：正文发布后不再允许修改。"""

    workspace_id: str  # 所属工作区
    name: str  # 模板名称
    version: str  # 模板版本
    template: str  # 模板正文，包含 ${变量名} 占位符
    created_at: datetime  # 版本发布时间（UTC）


class PromptService:
    """模板版本服务：版本不可覆盖，latest 指向最新发布版本。"""

    def __init__(self, repository: PromptRepository | None = None) -> None:
        self._repository = repository or repositories.prompts
        self._seed_defaults()

    def store(
        self,
        name: str,
        version: str,
        template: str,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> PromptTemplateResponse:
        """发布一个模板版本；同一名称下的版本不可覆盖。"""
        # 1. latest 是引用别名，不能作为真实版本号发布
        if version == LATEST_VERSION:
            raise InvalidRequestException(
                f"版本号 {LATEST_VERSION} 是引用别名，请使用具体版本号"
            )
        # 2. 提前校验模板语法，避免发布后渲染时才暴露问题
        try:
            Template(template).get_identifiers()
        except ValueError as exc:
            raise InvalidRequestException(f"模板语法不合法: {exc}") from exc
        # 3. 写入版本仓库；重复版本由 Repository 抛出唯一键冲突
        try:
            self._repository.insert(
                PromptVersion(
                    workspace_id=workspace_id,
                    name=name,
                    version=version,
                    template=template,
                    created_at=utc_now_iso(),
                )
            )
        except DuplicateRecordError as exc:
            raise InvalidRequestException(
                f"模板版本已存在: {name}@{version}"
            ) from exc
        audit_service.record(
            "prompt.publish",
            "prompt",
            resource_id=f"{name}@{version}",
            workspace_id=workspace_id,
            detail={"name": name, "version": version},
        )
        return self.get_template(name, version, workspace_id=workspace_id)

    def get(
        self,
        name: str,
        version: str = LATEST_VERSION,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> PromptTemplate:
        """按名称与版本取模板；latest 解析为该模板最新发布版本。"""
        row = (
            self._repository.latest(workspace_id=workspace_id, name=name)
            if version == LATEST_VERSION
            else self._repository.get(
                workspace_id=workspace_id, name=name, version=version
            )
        )
        if row is None:
            raise NotFoundException(f"Prompt 模板不存在: {name}@{version}")
        return PromptTemplate(
            workspace_id=row["workspace_id"],
            name=row["name"],
            version=row["version"],
            template=row["template"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get_template(
        self,
        name: str,
        version: str = LATEST_VERSION,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> PromptTemplateResponse:
        """按名称与版本查询模板详情，供管理接口返回。"""
        template = self.get(name, version, workspace_id=workspace_id)
        latest = self._repository.latest(workspace_id=workspace_id, name=name)
        return self._describe(template, latest["version"] if latest else None)

    def list_templates(
        self,
        name: str | None = None,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> list[PromptTemplateResponse]:
        """列出工作区内的模板版本；传 name 时只返回该模板的版本列表。"""
        rows = self._repository.list(workspace_id=workspace_id)
        latest: dict[str, str] = {}
        # Repository 按 name / created_at / version 升序返回，最后一条即 latest
        for row in rows:
            latest[row["name"]] = row["version"]
        return [
            self._describe(
                PromptTemplate(
                    workspace_id=row["workspace_id"],
                    name=row["name"],
                    version=row["version"],
                    template=row["template"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                ),
                latest.get(row["name"]),
            )
            for row in rows
            if name is None or row["name"] == name
        ]

    def render(
        self,
        selection: PromptSelection,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> str:
        """按引用渲染系统提示词，变量缺失或格式错误都返回 400。"""
        return self.render_resolved(selection, workspace_id=workspace_id)[0]

    def render_resolved(
        self,
        selection: PromptSelection,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> tuple[str, PromptTemplate]:
        """渲染并返回 (系统提示词, 解析后的模板)，供指标记录 latest 对应的具体版本。"""
        # 1. 先解析版本引用，拿到具体版本的正文
        template = self.get(selection.name, selection.version, workspace_id=workspace_id)
        # 2. 用调用方变量替换占位符；缺失变量属于调用方错误
        try:
            text = Template(template.template).substitute(selection.variables)
        except KeyError as exc:
            raise InvalidRequestException(
                f"缺少 Prompt 变量: {exc.args[0]}"
            ) from exc
        except ValueError as exc:
            raise InvalidRequestException(f"Prompt 模板渲染失败: {exc}") from exc
        return text, template

    def _seed_defaults(self) -> None:
        """把内置模板写入默认工作区，已存在时保持不变。"""
        now = utc_now_iso()
        self._repository.seed(
            [
                PromptVersion(
                    workspace_id=DEFAULT_WORKSPACE_ID,
                    name=name,
                    version=version,
                    template=template,
                    created_at=now,
                )
                for (name, version), template in DEFAULT_PROMPT_TEMPLATES.items()
            ]
        )

    @staticmethod
    def _describe(
        template: PromptTemplate, latest_version: str | None
    ) -> PromptTemplateResponse:
        """把内部快照转换成对外的版本详情，并标注是否为最新版本。"""
        return PromptTemplateResponse(
            name=template.name,
            version=template.version,
            template=template.template,
            created_at=template.created_at,
            is_latest=latest_version == template.version,
        )


prompt_service = PromptService()
