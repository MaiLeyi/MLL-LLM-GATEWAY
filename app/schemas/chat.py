from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.adapters.base import ChatMessage
from app.config.settings import DEFAULT_MODEL
from app.schemas.prompt import PromptSelection


class Message(BaseModel):
    """跨模型通用的单条对话消息，支持 system / user / assistant 三种角色。"""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"] = Field(..., description="消息角色")
    content: str = Field(..., min_length=1, max_length=20_000, description="消息内容")


class Usage(BaseModel):
    """Token 分类消耗，口径与 /metrics 明细一致。"""

    prompt: int = 0
    completion: int = 0
    total: int = 0
    cached: int = 0
    reasoning: int = 0


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
    """对话输入公共字段：单条 message 或多轮 messages（二选一）、模型与 Prompt 版本引用。"""

    model_config = ConfigDict(extra="forbid")  # 禁止未声明字段，流式接口可据此拒绝 response_format

    message: str | None = Field(
        default=None, min_length=1, max_length=8192, description="单条用户消息"
    )  # 单条用户消息，与 messages 二选一
    messages: list[Message] | None = Field(
        default=None, min_length=1, max_length=100, description="多轮对话消息"
    )  # 多轮对话消息（含 system/assistant 角色），与 message 二选一
    model: str = Field(default=DEFAULT_MODEL, description="模型 id")  # 目标模型 id
    prompt: PromptSelection | None = Field(
        default=None, description="引用的 Prompt 模板版本；不传则使用默认系统提示词"
    )  # 模板版本引用与变量
    timeout_seconds: float = Field(
        default=30, gt=0, le=120, description="上游调用超时（秒），逐请求生效"
    )  # 请求级超时

    @model_validator(mode="after")
    def _check_message_input(self) -> "ChatInput":
        # message 与 messages 必须且只能提供一个
        if (self.message is None) == (self.messages is None):
            raise ValueError("message 与 messages 必须且只能提供一个")
        return self

    def to_chat_messages(self) -> list[ChatMessage]:
        """把两种输入统一成完整的对话消息列表，角色原样保留。"""
        if self.messages is not None:
            return [ChatMessage(role=m.role, content=m.content) for m in self.messages]
        assert self.message is not None
        return [ChatMessage(role="user", content=self.message)]


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
    request_id: str = Field(
        ..., description="本次请求 ID，可在 /metrics 的 recent 明细中串联调用"
    )  # 请求 ID
    model: str = Field(
        ..., description="实际应答的模型 id；发生主备切换时与请求模型不同"
    )  # 实际应答模型
    usage: Usage | None = Field(
        default=None, description="本次调用的 Token 分类消耗，上游未返回时为 null"
    )  # Token 分类消耗
    latency_ms: int = Field(
        ..., ge=0, description="请求总耗时（毫秒），含重试与主备切换"
    )  # 请求总耗时
    attempts: int = Field(
        ..., ge=1, description="上游调用总次数（重试与主备切换都会计入）"
    )  # 上游调用总次数
