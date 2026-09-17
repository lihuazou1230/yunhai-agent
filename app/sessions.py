"""会话与消息存储：SQLite（stdlib，不引 ORM 也不引额外依赖）。

存的是"问答过程"：问题、回答、当次引用块、命中分数、兜底模式。
用途有两个——前端的历史侧栏，以及排查"为什么这题答歪了"（引用和分数都留着）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.errors import SessionNotFound

_SCHEMA = """
create table if not exists sessions (
  id          text primary key,
  title       text not null default '',
  mode        text not null default 'semantic',
  created_at  text not null,
  updated_at  text not null
);
create table if not exists messages (
  id          text primary key,
  session_id  text not null references sessions(id) on delete cascade,
  role        text not null,
  content     text not null,
  citations   text not null default '[]',
  meta        text not null default '{}',
  created_at  text not null
);
create index if not exists idx_messages_session on messages(session_id, created_at);
"""


class SessionStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + 一把锁：FastAPI 的线程池里会被多个线程碰到
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ---------- 会话 ----------

    def create_session(self, title: str = "", mode: str = "semantic") -> dict[str, Any]:
        session_id = uuid.uuid4().hex[:16]
        now = _now()
        with self._lock:
            self._conn.execute(
                "insert into sessions (id, title, mode, created_at, updated_at) values (?, ?, ?, ?, ?)",
                (session_id, title[:60], mode, now, now),
            )
            self._conn.commit()
        return {"id": session_id, "title": title[:60], "mode": mode, "created_at": now, "updated_at": now}

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                select s.*, (select count(*) from messages m where m.session_id = s.id) as message_count
                from sessions s order by s.updated_at desc limit ?
                """,
                (limit,),
            ).fetchall()
        return [_session_row(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute("select * from sessions where id = ?", (session_id,)).fetchone()
            if row is None:
                raise SessionNotFound(f"会话不存在：{session_id}")
            messages = self._conn.execute(
                "select * from messages where session_id = ? order by created_at, rowid",
                (session_id,),
            ).fetchall()
        session = _session_row(row)
        session["messages"] = [_message_row(m) for m in messages]
        return session

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("delete from sessions where id = ?", (session_id,))
            self._conn.execute("delete from messages where session_id = ?", (session_id,))
            self._conn.commit()

    def ensure_session(self, session_id: str | None, question: str, mode: str) -> str:
        """有 session_id 就沿用，没有就按首问建一个新的（标题 = 问题前 20 字）。"""
        if session_id:
            with self._lock:
                row = self._conn.execute("select id from sessions where id = ?", (session_id,)).fetchone()
            if row is not None:
                return session_id
        return self.create_session(title=question.strip()[:20], mode=mode)["id"]

    # ---------- 消息 ----------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        citations: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message_id = uuid.uuid4().hex[:16]
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                insert into messages (id, session_id, role, content, citations, meta, created_at)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    session_id,
                    role,
                    content,
                    json.dumps(citations or [], ensure_ascii=False),
                    json.dumps(meta or {}, ensure_ascii=False),
                    now,
                ),
            )
            self._conn.execute("update sessions set updated_at = ? where id = ?", (now, session_id))
            self._conn.commit()
        return {
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "citations": citations or [],
            "meta": meta or {},
            "created_at": now,
        }


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _session_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data.setdefault("message_count", 0)
    return data


def _message_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    try:
        data["citations"] = json.loads(data.get("citations") or "[]")
    except json.JSONDecodeError:
        data["citations"] = []
    try:
        data["meta"] = json.loads(data.get("meta") or "{}")
    except json.JSONDecodeError:
        data["meta"] = {}
    return data
