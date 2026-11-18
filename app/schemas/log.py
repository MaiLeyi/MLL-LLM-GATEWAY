"""Agent 日志与网关调用记录的数据契约。

本模块只放纯类型定义，不包含业务逻辑。
- 会话日志侧：Session / Event（判别联合）/ Usage / Checkpoint
- 网关侧：Call / Attempt（一次请求的多次主备/重试展开）
"""

from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated


# --------------------------------------------------------------------------- #
# 通用：Token 消耗
# --------------------------------------------------------------------------- #
class UsageRecord(BaseModel):
    """Token 分类消耗，挂在 Attempt 或 assistant Message 上。"""

    model_config = ConfigDict(extra="forbid")  # 禁止未声明字段

    usage_id: str = Field(..., description="消耗记录 ID")  # 消耗记录唯一 ID
    attempt_id: str | None = Field(
        default=None, description="所属 Attempt ID；会话日志内联时为空"
    )  # 关联的网关尝试；纯会话日志可空
    prompt_tokens: int = Field(default=0, ge=0, description="输入 Token 数")  # 输入 Token
    completion_tokens: int = Field(default=0, ge=0, description="输出 Token 数")  # 输出 Token
    total_tokens: int = Field(default=0, ge=0, description="输入 + 输出总 Token 数")  # 总 Token
    cached_tokens: int = Field(default=0, ge=0, description="缓存命中 Token 数")  # 缓存命中
    reasoning_tokens: int = Field(default=0, ge=0, description="推理过程 Token 数")  # 推理 Token
    cost_usd: float = Field(default=0.0, ge=0, description="折算成本（美元）")  # 按模型定价折算


# --------------------------------------------------------------------------- #
# 会话：Session
# --------------------------------------------------------------------------- #
class SessionStatus(str, Enum):
    """会话状态。"""

    ACTIVE = "active"  # 进行中
    COMPLETED = "completed"  # 正常结束
    INTERRUPTED = "interrupted"  # 被中断


class SessionRecord(BaseModel):
    """会话头，对应 JSONL 文件第一行；不参与事件树。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(..., description="会话唯一 ID")  # 会话 ID，即文件名
    version: int = Field(default=1, description="日志格式版本号")  # 格式版本，向前兼容用
    cwd: str = Field(default="", description="工作目录")  # agent 运行时的工作目录
    model: str = Field(default="", description="默认模型 ID")  # 默认模型，实际模型记在 Message 上
    status: SessionStatus = Field(
        default=SessionStatus.ACTIVE, description="会话状态"
    )  # active / completed / interrupted
    parent_session_id: str | None = Field(
        default=None, description="fork 来源会话 ID"
    )  # fork 自哪个会话，支持分支
    created_at: str = Field(..., description="创建时间 ISO8601")  # 创建时间
    updated_at: str = Field(..., description="更新时间 ISO8601")  # 最后更新时间


# --------------------------------------------------------------------------- #
# 事件：Event（判别联合，type 字段区分种类）
# --------------------------------------------------------------------------- #
class BaseEvent(BaseModel):
    """所有事件的公共字段；物理上对应 JSONL 的一行。"""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(..., description="事件唯一 ID")  # 事件 ID
    session_id: str = Field(..., description="所属会话 ID")  # 所属 Session
    parent_event_id: str | None = Field(
        default=None, description="父事件 ID，组成树结构"
    )  # 上一条事件，支持分支/fork
    timestamp: str = Field(..., description="ISO8601 时间戳")  # 事件发生时间
    file_id: str | None = Field(
        default=None, description="物理文件 ID"
    )  # 所在文件标识，用于跨文件定位
    offset: int | None = Field(
        default=None, ge=0, description="文件内字节偏移"
    )  # 文件内起始字节偏移，给断点续读用


class MessageEvent(BaseEvent):
    """对话消息：user / assistant / system / toolResult。"""

    type: Literal["message"] = "message"  # 判别字段：message
    role: Literal["system", "user", "assistant", "toolResult"] = Field(
        ..., description="消息角色"
    )  # 消息角色
    content: str | list[dict[str, Any]] = Field(
        ..., description="文本或 content block 数组（含 thinking/tool_use）"
    )  # 文本或内容块数组
    model: str | None = Field(
        default=None, description="实际应答模型；主备切换后与 session.model 可能不同"
    )  # 实际应答模型
    stop_reason: str | None = Field(
        default=None, description="结束原因：end_turn / tool_use / max_tokens 等"
    )  # 结束原因
    usage: UsageRecord | None = Field(
        default=None, description="内联 Token 消耗；非 assistant 消息为空"
    )  # 内联的 Token 消耗


class ToolCallEvent(BaseEvent):
    """工具调用入参，由 assistant 消息中的 tool_use block 产生。"""

    type: Literal["tool_call"] = "tool_call"  # 判别字段：tool_call
    tool_call_id: str = Field(..., description="工具调用 ID")  # 上游返回的调用 ID
    tool_name: str = Field(..., description="工具名")  # 工具名
    input: dict[str, Any] = Field(
        default_factory=dict, description="工具入参"
    )  # 工具入参


class ToolResultEvent(BaseEvent):
    """工具执行结果，通过 tool_call_id 与 ToolCallEvent 配对。"""

    type: Literal["tool_result"] = "tool_result"  # 判别字段：tool_result
    tool_call_id: str = Field(..., description="对应的工具调用 ID")  # 配对的调用 ID
    content: str = Field(default="", description="工具输出文本")  # 工具输出
    is_error: bool = Field(default=False, description="执行是否出错")  # 是否出错


class CompactionEvent(BaseEvent):
    """上下文压缩：summary 替代 first_kept_event_id 之前的所有事件。"""

    type: Literal["compaction"] = "compaction"  # 判别字段：compaction
    summary: str = Field(..., description="压缩摘要文本")  # 压缩摘要
    first_kept_event_id: str = Field(
        ..., description="保留的起始事件 ID"
    )  # 重放时从此事件开始保留
    tokens_before: int = Field(
        default=0, ge=0, description="压缩前 Token 数"
    )  # 压缩前的上下文 Token 数


class ModelChangeEvent(BaseEvent):
    """会话中途切换模型。"""

    type: Literal["model_change"] = "model_change"  # 判别字段：model_change
    provider: str = Field(..., description="厂商")  # 厂商：anthropic / openai ...
    model_id: str = Field(..., description="新模型 ID")  # 切换后的模型


class SystemEvent(BaseEvent):
    """其它系统事件，作为兜底类型。"""

    type: Literal["system"] = "system"  # 判别字段：system
    content: str = Field(default="", description="系统事件内容")  # 事件内容


# 判别联合：通过 type 字段自动选择具体事件类型；未知类型由 reader 跳过
Event = Annotated[
    Union[
        MessageEvent,
        ToolCallEvent,
        ToolResultEvent,
        CompactionEvent,
        ModelChangeEvent,
        SystemEvent,
    ],
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------- #
# 网关：Call / Attempt
# --------------------------------------------------------------------------- #
class CallStatus(str, Enum):
    """网关请求最终状态。"""

    SUCCESS = "success"  # 成功
    FAILURE = "failure"  # 失败


class AttemptRecord(BaseModel):
    """一次上游尝试；主备切换、重试各产生一条。"""

    model_config = ConfigDict(extra="forbid")

    attempt_id: str = Field(..., description="尝试 ID")  # 尝试唯一 ID
    request_id: str = Field(..., description="所属 Call 的 request_id")  # 所属请求
    seq: int = Field(..., ge=0, description="尝试序号，从 0 开始")  # 第几次尝试
    provider: str = Field(..., description="上游厂商")  # 厂商
    model: str = Field(..., description="实际应答模型")  # 实际模型
    latency_ms: int = Field(..., ge=0, description="本次尝试耗时（毫秒）")  # 耗时
    first_token_ms: int | None = Field(
        default=None, ge=0, description="首 token 延迟（毫秒），非流式为空"
    )  # 首 token 延迟
    http_status: int | None = Field(
        default=None, ge=100, le=599, description="上游 HTTP 状态码"
    )  # 上游状态码
    error_code: str | None = Field(
        default=None, description="失败时的错误码，如 TIMEOUT / INVALID_MODEL_OUTPUT"
    )  # 错误码
    success: bool = Field(default=False, description="本次尝试是否成功")  # 是否成功


class CallRecord(BaseModel):
    """一次网关请求；多次 Attempt 共享同一个 request_id。"""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., description="请求 ID")  # 请求唯一 ID
    session_id: str | None = Field(
        default=None, description="关联的会话 ID；纯补全请求为空"
    )  # 关联会话，可空
    tenant: str | None = Field(default=None, description="租户/调用方标识")  # 租户
    requested_model: str = Field(..., description="调用方请求的模型 ID")  # 请求模型
    stream: bool = Field(default=False, description="是否流式调用")  # 是否流式
    prompt_name: str | None = Field(
        default=None, description="引用的 Prompt 模板名"
    )  # Prompt 模板名
    prompt_version: str | None = Field(
        default=None, description="引用的 Prompt 模板版本"
    )  # Prompt 模板版本
    status: CallStatus = Field(
        default=CallStatus.FAILURE, description="最终状态"
    )  # 最终 success / failure
    total_latency_ms: int = Field(
        default=0, ge=0, description="总耗时（毫秒），含重试与主备切换"
    )  # 总耗时
    total_cost_usd: float = Field(
        default=0.0, ge=0, description="所有 Attempt 成本之和（美元）"
    )  # 总成本
    fallback_count: int = Field(
        default=0, ge=0, description="主备切换次数"
    )  # 实际模型 ≠ 请求模型的次数
    created_at: str = Field(..., description="请求时间 ISO8601")  # 请求时间


# --------------------------------------------------------------------------- #
# 消费侧：Checkpoint
# --------------------------------------------------------------------------- #
class CheckpointRecord(BaseModel):
    """消费者检查点：记录每个消费者读到了哪个文件的哪个字节偏移。"""

    model_config = ConfigDict(extra="forbid")

    consumer_name: str = Field(..., description="消费者标识")  # 消费者名（UI/审计/采集）
    file_id: str = Field(..., description="当前读取的文件 ID")  # 当前文件
    offset: int = Field(default=0, ge=0, description="字节偏移")  # 文件内字节偏移
    updated_at: str = Field(..., description="最后提交时间 ISO8601")  # 提交时间
