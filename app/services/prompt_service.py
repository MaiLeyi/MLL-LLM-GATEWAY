import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from string import Template

from app.config.prompt_config import DEFAULT_PROMPT_TEMPLATES, LATEST_VERSION
from app.core.exceptions import InvalidRequestException, NotFoundException
from app.schemas.prompt import PromptSelection, PromptTemplateResponse


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """单个模板版本的不可变快照：正文发布后不再允许修改。"""

    name: str  # 模板名称
    version: str  # 模板版本
    template: str  # 模板正文，包含 ${变量名} 占位符
    created_at: datetime  # 版本发布时间（UTC）


class PromptService:
    """进程内模板版本仓库：按 (名称, 版本) 存储，latest 指向最新发布版本。"""

    def __init__(self) -> None:
        self._templates: dict[tuple[str, str], PromptTemplate] = {}  # 已存储的模板版本
        self._latest: dict[str, str] = {}  # 模板名称 -> 最新发布版本
        self._lock = threading.Lock()  # 保护版本仓库与 latest 指针的并发访问
        # 1. 载入内置模板，保证默认模板可直接被引用
        for (name, version), template in DEFAULT_PROMPT_TEMPLATES.items():
            self._put(name, version, template)

    def store(self, name: str, version: str, template: str) -> PromptTemplateResponse:
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
        # 3. 写入仓库并更新 latest 指针
        stored = self._put(name, version, template)
        return self._describe(stored)

    def get(self, name: str, version: str = LATEST_VERSION) -> PromptTemplate:
        """按名称与版本取模板；latest 解析为该模板最新发布版本。"""
        with self._lock:
            return self._resolve(name, version)

    def get_template(
        self, name: str, version: str = LATEST_VERSION
    ) -> PromptTemplateResponse:
        """按名称与版本查询模板详情，供管理接口返回。"""
        with self._lock:
            return self._describe(self._resolve(name, version))

    def list_templates(self, name: str | None = None) -> list[PromptTemplateResponse]:
        """列出模板版本；传 name 时只返回该模板的版本列表。"""
        with self._lock:
            items = [
                self._describe(template)
                for template in self._templates.values()
                if name is None or template.name == name
            ]
        return sorted(items, key=lambda item: (item.name, item.created_at))

    def render(self, selection: PromptSelection) -> str:
        """按引用渲染系统提示词，变量缺失或格式错误都返回 400。"""
        # 1. 先解析版本引用，拿到具体版本的正文
        template = self.get(selection.name, selection.version)
        # 2. 用调用方变量替换占位符；缺失变量属于调用方错误
        try:
            return Template(template.template).substitute(selection.variables)
        except KeyError as exc:
            raise InvalidRequestException(
                f"缺少 Prompt 变量: {exc.args[0]}"
            ) from exc
        except ValueError as exc:
            raise InvalidRequestException(f"Prompt 模板渲染失败: {exc}") from exc

    def _put(self, name: str, version: str, template: str) -> PromptTemplate:
        """在锁内写入一个版本；重复版本直接拒绝，保证版本可追溯。"""
        stored = PromptTemplate(
            name=name,
            version=version,
            template=template,
            created_at=datetime.now(timezone.utc),
        )
        with self._lock:
            if (name, version) in self._templates:
                raise InvalidRequestException(f"模板版本已存在: {name}@{version}")
            self._templates[(name, version)] = stored
            self._latest[name] = version
        return stored

    def _resolve(self, name: str, version: str) -> PromptTemplate:
        """把 latest 别名解析为具体版本；不存在时抛 404。"""
        resolved_version = self._latest.get(name, version) if version == LATEST_VERSION else version
        template = self._templates.get((name, resolved_version))
        if template is None:
            raise NotFoundException(f"Prompt 模板不存在: {name}@{version}")
        return template

    def _describe(self, template: PromptTemplate) -> PromptTemplateResponse:
        """把内部快照转换成对外的版本详情，并标注是否为最新版本。"""
        return PromptTemplateResponse(
            name=template.name,
            version=template.version,
            template=template.template,
            created_at=template.created_at,
            is_latest=self._latest.get(template.name) == template.version,
        )


prompt_service = PromptService()
