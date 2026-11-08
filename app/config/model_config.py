from dataclasses import dataclass
from typing import Literal

Protocol = Literal["openai", "anthropic"]
StructuredOutputMode = Literal["json_schema", "json_object"]


@dataclass(frozen=True)
class ModelConfig:
    """单个模型的上游接入配置：模型名、地址、鉴权来源、协议与能力声明。"""

    provider_model: str  # 上游实际模型名称
    base_url: str  # 上游接口地址
    api_key_env: str  # 读取 API Key 的环境变量名
    protocol: Protocol  # 上游协议类型，用于选择适配器
    # 上游原生结构化输出能力：None 表示不支持，需要网关用提示词兜底
    structured_output_mode: StructuredOutputMode | None = None
    # 单模型独立限流：每分钟允许的请求数，<=0 表示不限制
    rate_limit_per_minute: int = 60
    # 主备切换：本模型上游不可用时切换的备用模型 id，None 表示无备用
    fallback_model: str | None = None
    # 计价单价：美元 / 百万 Token，成本核算按实际用量折算
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0
