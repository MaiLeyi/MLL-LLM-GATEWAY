import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.adapters.base import (
    ChatMessage,
    Completion,
    LLMAdapter,
    StreamChunk,
    TokenUsage,
    build_json_instruction,
)
from app.config.model_config import ModelConfig
from app.config.settings import SYSTEM_PROMPT, get_api_key

# 部分 OpenAI 兼容推理模型（如 MiniMax）会把推理过程包在 <think> 块里，
# 统一在这里剥离，避免污染结构化输出解析和流式正文。
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """剥离完整的 <think>...</think> 块；未成对时原样返回。"""
    cleaned = _THINK_RE.sub("", text)
    return cleaned.strip() if cleaned != text else text


class _ThinkFilter:
    """流式场景下跨分片剥离推理块：分片可能把标签切成两半。"""

    def __init__(self) -> None:
        self._in_think = False  # 当前是否位于推理块内部
        self._pending = ""  # 暂存可能是标签前缀的尾部文本

    def feed(self, text: str) -> str:
        """喂入一个分片，返回过滤后应对外输出文本。"""
        out: list[str] = []
        buf = self._pending + text
        self._pending = ""
        while buf:
            if self._in_think:
                # 1. 推理块内：丢弃直到闭合标签出现
                idx = buf.find(_THINK_CLOSE)
                if idx == -1:
                    self._pending = buf[-(len(_THINK_CLOSE) - 1) :]
                    break
                buf = buf[idx + len(_THINK_CLOSE) :]
                self._in_think = False
                continue
            # 2. 正常正文：查找开标签，并保留可能被切断的标签前缀
            idx = buf.find(_THINK_OPEN)
            if idx == -1:
                keep = len(_THINK_OPEN) - 1
                if len(buf) > keep:
                    out.append(buf[:-keep])
                    self._pending = buf[-keep:]
                else:
                    self._pending = buf
                break
            if idx > 0:
                out.append(buf[:idx])
            buf = buf[idx + len(_THINK_OPEN) :]
            self._in_think = True
        return "".join(out)

    def flush(self) -> str:
        """流结束时吐出被暂存的尾部文本。"""
        rest = self._pending
        self._pending = ""
        return "" if self._in_think else rest


class OpenAIAdapter(LLMAdapter):
    """OpenAI Chat Completions 协议适配器，适用于 DeepSeek、MiniMax 等兼容实现。"""

    def __init__(self, model_id: str, config: ModelConfig):
        self._provider_model = config.provider_model  # 上游实际模型名称
        self._api_key = get_api_key(model_id)  # 上游鉴权 Key
        self._base_url = config.base_url  # 上游接口地址
        self._structured_output_mode = (
            config.structured_output_mode
        )  # 上游原生结构化输出能力
        # 默认（text）场景复用同一个 Agent，避免每次请求重复构建
        self._agent = self._build_agent()  # 复用的默认 Agent 实例

    def _build_agent(
        self,
        system_prompt: str | None = None,
        response_format: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ):
        # 1. 按模型能力决定走原生 response_format 还是提示词兜底
        native_format, instruction = self._resolve_response_format(response_format)
        base_prompt = system_prompt or SYSTEM_PROMPT
        system_prompt = (
            base_prompt if instruction is None else f"{base_prompt}\n\n{instruction}"
        )
        model = ChatOpenAI(
            model=self._provider_model,
            api_key=self._api_key,
            base_url=self._base_url,
            # 请求级超时，透传给上游 SDK；None 时使用 SDK 默认
            timeout=timeout_seconds,
            # 流式调用也要求上游返回 usage，才能统计 Token 消耗
            stream_usage=True,
            # response_format 属于 OpenAI 请求体字段，通过 extra_body 原样下传
            extra_body=(
                {"response_format": native_format} if native_format else None
            ),
        )
        return create_agent(
            model=model,
            tools=[],
            system_prompt=system_prompt,
        )

    def _resolve_response_format(
        self, response_format: dict[str, Any] | None
    ) -> tuple[dict[str, Any] | None, str | None]:
        """按上游能力解析格式：无原生能力走提示词，能力不足时降级。"""
        if response_format is None:
            return None, None
        if self._structured_output_mode == "json_schema":
            # 原生支持 json_schema：完整下传 schema 定义
            return dict(response_format), None
        # 其余情况一律使用 json_object，并注入格式说明：
        # 1) 上游不支持原生结构化输出时，只能靠提示词约束；
        # 2) 降级请求也要说明期望结构，否则 schema 会被静默丢弃；
        # 3) 多数 OpenAI 兼容实现要求提示词中出现 "json" 字样，否则直接报错。
        return {"type": "json_object"}, build_json_instruction(response_format)

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Completion:
        # 结构化约束、自定义提示词、自定义超时都是请求级参数，需要按本次请求重建 Agent
        agent = (
            self._agent
            if response_format is None
            and system_prompt is None
            and timeout_seconds is None
            else self._build_agent(system_prompt, response_format, timeout_seconds)
        )
        response = await agent.ainvoke({"messages": self._to_pairs(messages)})
        message = response["messages"][-1]
        content = message.content if isinstance(message.content, str) else ""
        return Completion(
            text=strip_think(content),
            usage=self._usage_from_metadata(getattr(message, "usage_metadata", None)),
        )

    async def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        system_prompt: str | None = None,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        agent = (
            self._agent
            if system_prompt is None and timeout_seconds is None
            else self._build_agent(system_prompt, timeout_seconds=timeout_seconds)
        )
        think_filter = _ThinkFilter()
        async for chunk, _metadata in agent.astream(
            {"messages": self._to_pairs(messages)},
            stream_mode="messages",
        ):
            content = getattr(chunk, "content", "")
            raw = content if isinstance(content, str) else ""
            # 跨分片剥离 <think> 推理块，避免推理过程被当成正文下发
            text = think_filter.feed(raw) if raw else ""
            usage = self._usage_from_metadata(getattr(chunk, "usage_metadata", None))
            # 仅含 usage、无文本的结束分片也要产出，供上层统计
            if text or usage is not None:
                yield StreamChunk(text=text, usage=usage)
        # 流结束后吐出被暂存的尾部正文
        tail = think_filter.flush()
        if tail:
            yield StreamChunk(text=tail)

    @staticmethod
    def _usage_from_metadata(metadata: dict | None) -> TokenUsage | None:
        """把 LangChain 的 usage_metadata 转成统一的分类统计。"""
        if not metadata:
            return None
        prompt = int(metadata.get("input_tokens") or 0)
        completion = int(metadata.get("output_tokens") or 0)
        total = int(metadata.get("total_tokens") or (prompt + completion))
        input_details = metadata.get("input_token_details") or {}
        output_details = metadata.get("output_token_details") or {}
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cached_tokens=int(input_details.get("cache_read") or 0),
            reasoning_tokens=int(output_details.get("reasoning") or 0),
        )

    @staticmethod
    def _to_pairs(messages: Sequence[ChatMessage]) -> list[tuple[str, str]]:
        # 角色原样透传（含 system / assistant），保证多轮对话上下文完整
        return [(m.role, m.content) for m in messages]
