from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.chat_router import router as chat_router
from app.api.v1.metrics_router import router as metrics_router
from app.config.settings import HOST, PORT
from app.core.exception_handlers import register_exception_handlers

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


@app.get("/")
async def root():
    return {"HELLO WORLD"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
