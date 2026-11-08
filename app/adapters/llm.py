from functools import lru_cache

from app.adapters.anthropic_adapter import AnthropicAdapter
from app.adapters.base import LLMAdapter
from app.adapters.openai_adapter import OpenAIAdapter
from app.config.settings import MODEL_CONFIGS

# 协议名 -> 适配器实现，新增协议时只需在这里注册
ADAPTER_REGISTRY: dict[str, type[LLMAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
}


@lru_cache(maxsize=None)
def get_adapter(model_id: str) -> LLMAdapter:
    """根据请求中的 model 找到配置，并按其 protocol 动态路由到对应适配器。"""
    config = MODEL_CONFIGS[model_id]
    adapter_cls = ADAPTER_REGISTRY.get(config.protocol)
    if adapter_cls is None:
        raise ValueError(
            f"模型 {model_id} 使用了不支持的协议: {config.protocol}，"
            f"可选: {sorted(ADAPTER_REGISTRY)}"
        )
    return adapter_cls(model_id, config)
