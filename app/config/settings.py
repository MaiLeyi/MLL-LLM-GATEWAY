import os

from dotenv import load_dotenv

from app.config.model_config import ModelConfig

load_dotenv()


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "deepseek": ModelConfig(
        provider_model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        protocol="openai",
        # DeepSeek 原生仅支持 json_object，json_schema 请求会降级
        structured_output_mode="json_object",
        rate_limit_per_minute=60,
    ),
    "minimax": ModelConfig(
        provider_model="MiniMax-M3",
        base_url="https://api.minimax.cn/v1",
        api_key_env="MINI_MAX_API_KEY",
        protocol="openai",
        # 保守声明为 json_object，避免上游忽略 json_schema 时静默失效
        structured_output_mode="json_object",
        rate_limit_per_minute=30,
    ),
}

# 指数退避重试：失败后最多重试 3 次，延迟 0.5s -> 1s -> 2s（带抖动）
MAX_RETRIES = 3
RETRY_BASE_DELAY = 0.5
RETRY_MAX_DELAY = 8.0

# 限流统计窗口
RATE_LIMIT_WINDOW_SECONDS = 60

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
