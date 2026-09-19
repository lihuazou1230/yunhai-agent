"""流式问答接口（规划 10.1 / 10.3 / 11.1）。

两条策略共存，由请求里的 `strategy` 选：
- `agent`（默认，第十一阶段）：ReAct 循环 + 工具；知识库只是它可选的一个工具；
- `rag`（第十阶段）：直接检索 → 阈值 → 约束生成，`eval_answers.py` 仍走这条路，
  这样第十阶段的验收数据在第十一阶段之后依然可复现（换了架构不等于旧数据失效）。

两个细节：
1. **接口本身不检查 Key**：未配 Key 时照样返回 200 + SSE，把 `error` 事件发在流里——
   前端只写了一套流解析逻辑，若这里改 4xx，"流中错误"与"请求错误"就变成两条代码路径；
2. `/api/ask/resume`：client 工具（task_crud / get_weather）在前端执行完回来续跑，
   与 `/api/ask` 返回同一种流，前端复用同一个解析器。
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.runtime import get_runtime
from app.schemas import AskRequest, ResumeRequest

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
    if payload.strategy == "agent":
        stream = runtime.agent.stream(
            payload.question.strip(),
            session_id=payload.session_id,
            mode=payload.mode,
            fallback_mode=payload.fallback_mode,
            tools_enabled=payload.tools_enabled,
        )
    else:
        stream = runtime.ask.stream(
            payload.question.strip(),
            session_id=payload.session_id,
            mode=payload.mode,
            fallback_mode=payload.fallback_mode,
            top_k=payload.top_k,
            threshold=payload.threshold,
        )
    return StreamingResponse(stream, media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/ask/resume")
async def resume(payload: ResumeRequest):
    """前端执行完 client 工具后回来续跑同一轮（同一个 SSE 协议）。

    先在流**之前**校验 run_id：状态过期是"请求级"错误，直接给 404 JSON 比
    流里报错更好排查；流一旦开始就只能用 error 事件收口了。
    """
    runtime = get_runtime()
    runtime.runs.load(payload.run_id)
    stream = runtime.agent.resume(
        payload.run_id,
        [item.model_dump() for item in payload.results],
    )
    return StreamingResponse(stream, media_type="text/event-stream", headers=SSE_HEADERS)
