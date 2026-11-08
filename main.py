from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.v1.chat_router import router as chat_router
from app.api.v1.metrics_router import router as metrics_router
from app.api.v1.models_router import router as models_router
from app.api.v1.prompt_router import router as prompt_router
from app.config.settings import HOST, PORT
from app.core.exception_handlers import register_exception_handlers

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="MLL Agent API", description="我的第一个练习AGENT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(chat_router)
app.include_router(metrics_router)
app.include_router(prompt_router)
app.include_router(models_router)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """返回控制台单页应用（全部资源内联，无需挂载静态目录）。"""
    return FileResponse(BASE_DIR / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
