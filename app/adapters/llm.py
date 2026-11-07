from functools import lru_cache

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.config.settings import MODEL_CONFIGS, get_api_key

SYSTEM_PROMPT = "你是一个专家"


@lru_cache(maxsize=None)
def get_agent(model_id: str):
    config = MODEL_CONFIGS[model_id]
    model = ChatOpenAI(
        model=config.model,
        api_key=get_api_key(model_id),
        base_url=config.base_url,
    )
    return create_agent(model=model, tools=[], system_prompt=SYSTEM_PROMPT)
