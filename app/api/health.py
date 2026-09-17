"""自检接口：前端知识库页第一件事就是打它。

一次返回"后端活着吗、Key 配了吗、模型是什么、库里有多少内容"——
前端据此决定显示正常界面还是「去配置」引导，不需要靠试错。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.errors import AgentError
from app.runtime import get_runtime

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health():
    return get_runtime().health()


@router.get("/health/embedder")
async def warmup_embedder():
    """给「模型到底能不能用」一个明确的按钮：跑一次真实编码。

    bge 首次加载要几秒，放在这里让用户主动触发，
    好过第一句提问时莫名等 10 秒还不知道发生了什么。
    """
    runtime = get_runtime()
    try:
        vectors = runtime.embedder.encode(["云海工作台知识库自检"], is_query=True)
    except Exception as exc:  # noqa: BLE001
        raise AgentError(f"向量模型加载失败：{exc}", code="embedder_error", status_code=503) from exc
    return {
        "status": "ok",
        "embedder": runtime.embedder.name,
        "dim": len(vectors[0]) if vectors else 0,
    }
