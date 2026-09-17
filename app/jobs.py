"""后台任务登记表（规划 10.1：大文件异步入库返回任务 ID）。

故意做得极简：进程内字典 + 线程池。个人知识库的入库负载是"偶尔传个几十页 PDF"，
为它引入 Celery/Redis 是过度设计；但**接口形态**要按异步来设计
（返回 task_id + 状态查询），这样以后换成真队列，前端一行都不用改。
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from app.errors import JobNotFound


class JobRegistry:
    def __init__(self, max_workers: int = 2):
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ingest")

    def submit(self, name: str, func: Callable[[], Any]) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "name": name,
                "status": "pending",
                "result": None,
                "error": None,
                "created_at": _now(),
                "updated_at": _now(),
            }
        self._pool.submit(self._run, job_id, func)
        return job_id

    def _run(self, job_id: str, func: Callable[[], Any]) -> None:
        self._update(job_id, status="running")
        try:
            result = func()
        except Exception as exc:  # noqa: BLE001 - 任务失败要落进任务状态，不能吞
            self._update(job_id, status="failed", error=str(exc))
            return
        self._update(job_id, status="succeeded", result=result)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(fields)
            job["updated_at"] = _now()

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(f"任务不存在：{job_id}")
            return dict(job)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
