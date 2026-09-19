"""SSE 事件协议（规划 10.1 的「全局硬性约定」）。

凡产生内容的接口都走同一条事件流，前端因此只需要**一套**解析逻辑：

| 事件 | 内容 | 前端消费方 |
|------|------|-----------|
| token | 增量文本 | BaseChatBubble 逐字渲染 |
| tool_call / tool_result | 工具调用开始/结果 | BaseToolTag（第十一阶段） |
| citation | 引用来源块 | BaseCitationChip |
| proposal | 文件变更提案 | BaseDiffCard（第十三阶段） |
| done / error | 结束 / 异常 | 状态收口 |

第十阶段只发 token / citation / done / error，但协议一次定死——
第十一阶段的工具调用不需要再改前端解析器。
"""

from __future__ import annotations

import json
from typing import Any

TOKEN = "token"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
CITATION = "citation"
PROPOSAL = "proposal"
DONE = "done"
ERROR = "error"

ALL_EVENT_TYPES = (TOKEN, TOOL_CALL, TOOL_RESULT, CITATION, PROPOSAL, DONE, ERROR)


def sse(event: str, data: dict[str, Any]) -> str:
    """序列化成一帧 SSE。

    `ensure_ascii=False`：中文逐字流式时不要变成 \\uXXXX，省一半带宽也方便人肉调试。
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def token_event(text: str) -> str:
    return sse(TOKEN, {"text": text})


def citation_event(citation: dict[str, Any]) -> str:
    return sse(CITATION, citation)


def tool_call_event(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    *,
    executor: str = "server",
    status: str = "running",
) -> str:
    """工具调用开始 / 需要客户端执行。

    `status` 取值：`running`（服务端正在执行）、`awaiting_client`（等前端在工作台数据上执行）。
    """
    return sse(
        TOOL_CALL,
        {"id": call_id, "name": name, "arguments": arguments, "executor": executor, "status": status},
    )


def tool_result_event(
    call_id: str,
    name: str,
    ok: bool,
    *,
    summary: str = "",
    error: str = "",
    citations: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    return sse(
        TOOL_RESULT,
        {
            "id": call_id,
            "name": name,
            "ok": ok,
            "summary": summary,
            "error": error,
            "citations": citations or [],
            "meta": meta or {},
        },
    )


def proposal_event(proposal: dict[str, Any]) -> str:
    return sse(PROPOSAL, proposal)


def done_event(**payload: Any) -> str:
    return sse(DONE, payload)


def error_event(message: str, code: str = "internal_error", **extra: Any) -> str:
    return sse(ERROR, {"code": code, "message": message, **extra})
