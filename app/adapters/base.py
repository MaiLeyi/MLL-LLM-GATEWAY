from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str


class LLMAdapter(ABC):
    """统一调用接口，屏蔽底层鉴权、请求体结构和返回格式的差异。"""

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """非流式调用，返回完整回答文本；response_format 为 OpenAI 协议的格式约束。"""

    @abstractmethod
    def stream(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        """流式调用，按片段产出回答文本。"""
