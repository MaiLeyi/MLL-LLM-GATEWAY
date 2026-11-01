import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

load_dotenv()

# ---------- 1. 初始化 Agent ----------
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError(
        "缺少 DEEPSEEK_API_KEY。请先设置环境变量后再启动，例如：\n"
        "  export DEEPSEEK_API_KEY=sk-xxxxxxxx\n"
        "或在项目根目录创建 .env 文件写入 DEEPSEEK_API_KEY=sk-xxxxxxxx"
    )

model = ChatOpenAI(
    model="deepseek-flash",
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个乐于助人的助手"
)

# ---------- 2. 创建 FastAPI 应用 ----------
app = FastAPI(title="LangChain Agent API", description="浏览器测试 Agent 的一次性回答和流式输出")


from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "usage": "GET /chat?message=你的问题 → 一次性返回",
        "stream": "GET /stream?message=你的问题 → 流式返回（SSE）",
        "example": "http://localhost:8000/chat?message=你好"
    }

@app.get("/chat")
async def chat(message: str = Query(..., description="用户输入的消息")):
    """一次性返回完整回答"""
    response = await agent.ainvoke({"messages": [("user", message)]})
    reply = response["messages"][-1].content
    return {"reply": reply}

@app.get("/stream")
async def stream_chat(message: str = Query(..., description="用户输入的消息")):
    """流式返回 Agent 回答（Server-Sent Events）"""
    async def event_generator():
        # 直接使用 model.astream 而不是 agent.astream
        async for chunk in model.astream([{"role": "user", "content": message}]):
            if hasattr(chunk, "content") and chunk.content:
                yield f"data: {json.dumps({'content': chunk.content})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream; charset=utf-8")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)