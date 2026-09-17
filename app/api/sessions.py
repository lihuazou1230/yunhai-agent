"""会话历史接口（规划 10.1 的 `GET /api/sessions`）。"""

from __future__ import annotations

from fastapi import APIRouter

from app.runtime import get_runtime
from app.schemas import SessionDetail, SessionList

router = APIRouter(prefix="/api", tags=["sessions"])


@router.get("/sessions", response_model=SessionList)
async def list_sessions(limit: int = 50):
    runtime = get_runtime()
    return SessionList(sessions=runtime.sessions.list_sessions(limit=max(1, min(limit, 200))))


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str):
    return SessionDetail(**get_runtime().sessions.get_session(session_id))


@router.delete("/sessions/{session_id}", response_model=dict)
async def delete_session(session_id: str):
    runtime = get_runtime()
    runtime.sessions.delete_session(session_id)
    return {"status": "ok", "session_id": session_id}
