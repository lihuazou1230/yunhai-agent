"""待续跑的 Agent 运行状态（规划 11.1 的 client 工具回环用）。

后端要"停"在等前端执行工具的地方，就必须把这一轮的 messages 存下来——
这是**有状态**的一步，也是整套设计里唯一需要落盘的东西。

存 SQLite（stdlib，与 sessions 同一套路）而不是纯内存：进程重启（`--reload`、部署）后
前端那次 resume 不该直接 404。带 TTL 清理，运行状态不是长期数据。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.errors import RunNotFound

_SCHEMA = """
create table if not exists agent_runs (
  run_id      text primary key,
  session_id  text not null,
  state       text not null,
  created_at  text not null
);
create index if not exists idx_agent_runs_created on agent_runs(created_at);
"""


@dataclass
class PendingRun:
    """一次"停在等客户端工具"的运行。"""

    run_id: str
    session_id: str
    question: str
    messages: list[dict[str, Any]]
    mode: str = "semantic"
    fallback_mode: str = "refuse"
    rounds: int = 0
    tokens: int = 0
    content: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)
    failures: dict[str, int] = field(default_factory=dict)
    kinds: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, run_id: str, payload: str) -> PendingRun:
        data = json.loads(payload)
        data["run_id"] = run_id
        return cls(**data)


class RunStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def save(self, run: PendingRun) -> None:
        if not run.created_at:
            run.created_at = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            self._conn.execute(
                "insert or replace into agent_runs (run_id, session_id, state, created_at) values (?, ?, ?, ?)",
                (run.run_id, run.session_id, run.to_json(), run.created_at),
            )
            self._conn.commit()

    def load(self, run_id: str) -> PendingRun:
        with self._lock:
            row = self._conn.execute(
                "select state from agent_runs where run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise RunNotFound(f"这次运行状态已过期或不存在：{run_id}（请重新提问）")
        return PendingRun.from_json(run_id, row["state"])

    def delete(self, run_id: str) -> None:
        with self._lock:
            self._conn.execute("delete from agent_runs where run_id = ?", (run_id,))
            self._conn.commit()

    def purge_expired(self, ttl_seconds: int) -> int:
        """清掉过期的运行状态（默认 1 小时）：它不是长期数据，留久了只是垃圾。"""
        cutoff = (datetime.now() - timedelta(seconds=ttl_seconds)).isoformat(timespec="seconds")
        with self._lock:
            cursor = self._conn.execute("delete from agent_runs where created_at < ?", (cutoff,))
            self._conn.commit()
            return cursor.rowcount or 0

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:16]
