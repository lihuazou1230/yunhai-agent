"""FastAPI 应用入口。

启动命令见 README：`uvicorn app.main:app --reload --port 8000`。
CORS 开发期只放行 localhost 与 Tauri WebView 的 origin（规划 10.1）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import ask, documents, health, sessions
from app.errors import AgentError
from app.runtime import VERSION, get_runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = get_runtime()
    if runtime.degraded_reason:
        print(f"[yunhai-agent] 警告：{runtime.degraded_reason}")
    yield
    # 进程退出：把入库线程池收干净，避免 reload 时残留线程卡住端口
    runtime.jobs.shutdown()


def create_app() -> FastAPI:
    settings = get_runtime().settings
    app = FastAPI(
        title="云海工作台 · Agent 后端",
        description="RAG 知识库服务（第十阶段）：文档入库、检索、流式问答、会话历史",
        version=VERSION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(AgentError)
    async def agent_error_handler(_: Request, exc: AgentError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
        )

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(ask.router)
    app.include_router(sessions.router)
    return app


app = create_app()
