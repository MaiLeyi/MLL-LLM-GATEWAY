import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str
    api_key_env: str
    protocol: str


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "deepseek": ModelConfig(
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        protocol="openai",
    ),
    "minimax": ModelConfig(
        model="MiniMax-M3",
        base_url="https://api.minimax.cn/v1",
        api_key_env="MINI_MAX_API_KEY",
        protocol="openai",
    ),
}

SYSTEM_PROMPT = "你是一个专家"

DEFAULT_MODEL = "deepseek"

HOST = "0.0.0.0"
PORT = 8000


def get_api_key(model_id: str) -> str:
    api_key = os.getenv(MODEL_CONFIGS[model_id].api_key_env)
    if not api_key:
        raise RuntimeError(f"{MODEL_CONFIGS[model_id].api_key_env} NOT EXISTS")
    return api_key


# 默认模型的 Key 在启动时校验，避免服务起来却无法应答
get_api_key(DEFAULT_MODEL)
