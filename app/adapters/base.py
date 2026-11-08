import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role  # 消息角色：system / user / assistant
    content: str  # 消息文本内容


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """单次调用的 Token 分类消耗。"""

    prompt_tokens: int = 0  # 输入（提示词）Token 数
    completion_tokens: int = 0  # 输出（回答）Token 数
    total_tokens: int = 0  # 输入 + 输出总 Token 数
    cached_tokens: int = 0  # 命中上游缓存的 Token 数
    reasoning_tokens: int = 0  # 推理过程消耗的 Token 数

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt": self.prompt_tokens,
            "completion": self.completion_tokens,
            "total": self.total_tokens,
            "cached": self.cached_tokens,
            "reasoning": self.reasoning_tokens,
        }


@dataclass(frozen=True, slots=True)
class Completion:
    """非流式结果：回答文本 + Token 消耗（上游未返回时为 None）。"""

    text: str  # 完整回答文本
    usage: TokenUsage | None = None  # 本次调用的 Token 分类消耗


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """流式分片：文本片段；usage 通常只在结束分片返回一次。"""

    text: str = ""  # 本片段输出的文本，结束分片可能为空
    usage: TokenUsage | None = None  # 本次流式调用的 Token 分类消耗


def build_json_instruction(response_format: dict[str, Any]) -> str:
    """把 OpenAI 的 response_format 转成提示词约束，供不支持原生结构化输出的模型兜底。"""
    if response_format.get("type") == "json_schema":
        schema = response_format.get("json_schema", {}).get("schema", {})
        return (
            "请只返回符合以下 JSON Schema 的合法 JSON，"
            "不要输出多余文字或 Markdown 代码块：\n"
            + json.dumps(schema, ensure_ascii=False)
        )
    return "请只返回合法 JSON，不要输出多余文字或 Markdown 代码块。"


class LLMAdapter(ABC):
    """统一调用接口，屏蔽底层鉴权、请求体结构和返回格式的差异。"""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> Completion:
        """非流式调用，返回完整回答文本与 Token 消耗；response_format 为 OpenAI 协议的格式约束。"""

    @abstractmethod
    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[StreamChunk]:
        """流式调用，按片段产出回答文本与 Token 消耗。"""
