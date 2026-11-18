from functools import lru_cache

from app.adapters.anthropic_adapter import AnthropicAdapter
from app.adapters.base import LLMAdapter
from app.adapters.openai_adapter import OpenAIAdapter
from app.services.model_registry import model_registry

# 协议名 -> 适配器实现，新增协议时只需在这里注册
ADAPTER_REGISTRY: dict[str, type[LLMAdapter]] = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
}


@lru_cache(maxsize=64)
def _build_adapter(model_id: str, revision: int, api_key: str) -> LLMAdapter:
    """按模型配置构建适配器；revision 只参与缓存键，配置变更后自动重建。"""
    config = model_registry.require(model_id)
    adapter_cls = ADAPTER_REGISTRY.get(config.protocol)
    if adapter_cls is None:
        raise ValueError(
            f"模型 {model_id} 使用了不支持的协议: {config.protocol}，"
            f"可选: {sorted(ADAPTER_REGISTRY)}"
        )
    return adapter_cls(config, api_key)


def get_adapter(model_id: str) -> LLMAdapter:
    """根据注册表中的最新配置与 Key 动态路由到对应适配器。"""
    return _build_adapter(
        model_id, model_registry.revision, model_registry.get_api_key(model_id)
    )
