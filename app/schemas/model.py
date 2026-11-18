from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Protocol = Literal["openai", "anthropic"]
StructuredOutputMode = Literal["json_schema", "json_object"]


class ModelUpsert(BaseModel):
    """管理页面提交的模型配置；api_key 只写不读，不落 JSON。"""

    model_config = ConfigDict(extra="forbid")

    display_name: str = Field(default="", max_length=60, description="展示名称")
    provider_model: str = Field(
        ..., min_length=1, max_length=200, description="上游实际模型名"
    )
    base_url: str = Field(..., min_length=1, max_length=500, description="上游接口地址")
    protocol: Protocol = Field(default="openai", description="上游协议类型")
    structured_output_mode: StructuredOutputMode | None = Field(
        default=None, description="结构化输出能力；null 表示不支持"
    )
    rate_limit_per_minute: int = Field(
        default=60, ge=0, le=100_000, description="每分钟请求上限，0 表示不限"
    )
    fallback_model: str | None = Field(
        default=None, max_length=50, description="备用模型 id"
    )
    input_price_per_million: float = Field(
        default=0.0, ge=0, description="输入价格（美元/百万 Token）"
    )
    output_price_per_million: float = Field(
        default=0.0, ge=0, description="输出价格（美元/百万 Token）"
    )
    api_key: str | None = Field(
        default=None,
        max_length=1000,
        description="API Key；仅写入系统钥匙串，编辑时留空表示不修改",
    )

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url 必须以 http:// 或 https:// 开头")
        return value


class ModelCreate(ModelUpsert):
    """新增模型入参：id 保存后不可修改。"""

    id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,49}$",
        description="模型 id",
    )


class ModelCard(BaseModel):
    """模型卡片：只包含非敏感信息与 Key 掩码。"""

    id: str
    display_name: str
    provider_model: str
    base_url: str
    protocol: str
    structured_output_mode: str | None
    rate_limit_per_minute: int
    fallback_model: str | None
    input_price_per_million: float
    output_price_per_million: float
    api_key_env: str
    has_api_key: bool
    api_key_masked: str | None
    api_key_source: Literal["keychain", "env"] | None
    is_default: bool


class ModelListResponse(BaseModel):
    """模型列表：保留 models/default 字段兼容 Playground，items 供卡片页渲染。"""

    models: list[str]
    default: str
    items: list[ModelCard]


class ModelTestResult(BaseModel):
    """模型可用性测试结果；失败也返回 200，由 ok 字段区分。"""

    ok: bool
    model: str
    latency_ms: int
    reply: str | None = None
    usage: dict | None = None
    error_code: str | None = None
    error_message: str | None = None
