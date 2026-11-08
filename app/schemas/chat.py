from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config.settings import DEFAULT_MODEL
from app.schemas.prompt import PromptSelection


class JsonSchemaSpec(BaseModel):
    """json_schema 模式的结构描述，字段与 OpenAI 协议保持一致。"""

    name: str = Field(default="response", description="结构名称")  # 上游结构标识名
    schema_: dict[str, Any] = Field(
        ..., alias="schema", description="JSON Schema 定义"
    )  # 期望的 JSON 结构定义
    strict: bool = Field(default=True, description="是否严格按 Schema 约束")  # 严格模式开关


class ResponseFormat(BaseModel):
    """输出结构化约束：text 自由文本 / json_object 合法 JSON / json_schema 指定结构。"""

    type: Literal["text", "json_object", "json_schema"] = Field(
        default="text", description="输出格式类型"
    )  # 输出格式类型：text / json_object / json_schema
    json_schema: JsonSchemaSpec | None = Field(
        default=None, description="type=json_schema 时的结构定义"
    )  # json_schema 模式下的结构定义

    @model_validator(mode="after")
    def _check_json_schema(self) -> "ResponseFormat":
        # type=json_schema 必须携带结构定义，提前在入参阶段拦截
        if self.type == "json_schema" and self.json_schema is None:
            raise ValueError("type=json_schema 时必须提供 json_schema")
        return self

    @property
    def requires_json(self) -> bool:
        return self.type != "text"

    def to_upstream(self) -> dict[str, Any] | None:
        """转换为 OpenAI 协议的 response_format 请求体；text 模式不传。"""
        if not self.requires_json:
            return None
        if self.type == "json_schema":
            assert self.json_schema is not None
            return {
                "type": "json_schema",
                "json_schema": self.json_schema.model_dump(by_alias=True),
            }
        return {"type": "json_object"}


class ChatInput(BaseModel):
    """对话输入公共字段：消息、模型与 Prompt 版本引用。"""

    model_config = ConfigDict(extra="forbid")  # 禁止未声明字段，流式接口可据此拒绝 response_format

    message: str = Field(
        ..., min_length=1, max_length=8192, description="用户输入的消息"
    )  # 用户输入的消息
    model: str = Field(default=DEFAULT_MODEL, description="模型 id")  # 目标模型 id
    prompt: PromptSelection | None = Field(
        default=None, description="引用的 Prompt 模板版本；不传则使用默认系统提示词"
    )  # 模板版本引用与变量


class ChatRequest(ChatInput):
    """结构化输入：在公共字段上增加输出格式约束。"""

    response_format: ResponseFormat = Field(
        default_factory=ResponseFormat, description="输出结构化约束"
    )  # 输出结构化约束配置


class StreamRequest(ChatInput):
    """流式输入：保持纯文本流式输出，不支持 response_format。"""


class ChatResponse(BaseModel):
    """结构化输出：保留原始回答，JSON 模式下额外返回解析后的数据。"""

    reply: str = Field(..., description="模型返回的原始回答")  # 模型返回的原始回答
    data: Any | None = Field(
        default=None, description="response_format 要求 JSON 时解析出的结果"
    )  # JSON 模式下解析出的结果
