"""流式问答接口（规划 10.1 / 10.3）。

一个细节值得说明：**接口本身不检查 Key**。未配 Key 时照样返回 200 + SSE，
把 `error` 事件发在流里——因为前端只写了一套流解析逻辑，
如果这里改用 4xx，"流中错误"和"请求错误"就变成两条代码路径了。
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.runtime import get_runtime
from app.schemas import AskRequest

router = APIRouter(prefix="/api", tags=["ask"])

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # 关掉 Nginx/IIS 的响应缓冲，否则流式会攒成一坨再吐（规划里的 SSE 坑）
    "X-Accel-Buffering": "no",
}


@router.post("/ask")
async def ask(payload: AskRequest):
    runtime = get_runtime()
    stream = runtime.ask.stream(
        payload.question.strip(),
        session_id=payload.session_id,
        mode=payload.mode,
        fallback_mode=payload.fallback_mode,
        top_k=payload.top_k,
        threshold=payload.threshold,
    )
    return StreamingResponse(stream, media_type="text/event-stream", headers=SSE_HEADERS)
