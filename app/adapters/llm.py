from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.config.settings import DEEPSEEK_API_KEY, MODEL_BASE_URL, MODEL_NAME

model = ChatOpenAI(
    model=MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=MODEL_BASE_URL,
)

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个专家",
)
