from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.config.prompt_config import LATEST_VERSION


class PromptSelection(BaseModel):
    """调用方对已发布模板的引用：只提交模板名、版本与变量，不能覆盖模板正文。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=100, description="模板名称")
    version: str = Field(
        default=LATEST_VERSION,
        min_length=1,
        max_length=50,
        description="模板版本；latest 表示最新发布版本",
    )
    variables: dict[str, str] = Field(
        default_factory=dict, description="模板变量，用于替换正文中的 ${变量名}"
    )


class PromptTemplateCreate(BaseModel):
    """模板存储入参：一次发布一个不可变版本。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=100, description="模板名称")
    version: str = Field(
        ..., min_length=1, max_length=50, description="模板版本，不能使用 latest"
    )
    template: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="模板正文，使用 ${变量名} 作为占位符",
    )


class PromptTemplateResponse(BaseModel):
    """模板版本详情：供管理端查看已存储的正文与当前发布状态。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="模板名称")
    version: str = Field(..., description="模板版本")
    template: str = Field(..., description="模板正文")
    created_at: datetime = Field(..., description="版本发布的时间（UTC）")
    is_latest: bool = Field(..., description="是否为该模板最新发布的版本")
