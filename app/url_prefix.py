"""挂在子路径下时的前缀剥离（纯 ASGI 中间件）。

**为什么需要它**

自有服务器上本服务挂在 IIS 的子应用 `/yhai`（80 端口，同源反代，不开新端口）。
IIS 的 HttpPlatformHandler 会把请求**原样**转给子进程，也就是说 Python 侧收到的
还是 `/yhai/api/health`；而本服务的路由全是 `/api/*`，不处理就是一片 404。

于是由 `AGENT_URL_PREFIX=/yhai` 告诉服务"你自己被挂在 /yhai 下"，进来先把这段剥掉。
两种反代行为都能兼容：如果哪天换成会剥前缀的代理（如 ARR），这里就是彻底的空操作。

**为什么不用 BaseHTTPMiddleware**

Starlette 的 `BaseHTTPMiddleware` 会把响应重新包装一遍，流式响应会被揉成整块再吐，
SSE 就不再是流式的——而"逐字吐"正是本服务最不能丢的特性。所以这里只改写 `scope`，
`receive` / `send` 原样透传，对流式零影响。
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Receive = Callable[[], Any]
Send = Callable[[Any], Any]


def normalize_prefix(raw: str | None) -> str:
    """统一成 `/yhai` 这种形态：补前导斜杠、去掉尾斜杠；空值/纯斜杠视为不启用。"""
    value = (raw or "").strip().strip("/")
    return f"/{value}" if value else ""


class UrlPrefixMiddleware:
    """把进入请求路径上的前缀剥掉，命中时同时把前缀记进 `root_path`。

    - 前缀不匹配的请求（例如本机直连 `127.0.0.1:8000/api/health`）原样放行；
    - 只处理 `http` scope，`lifespan` / `websocket` 不动；
    - `raw_path` 一并剥：它在 scope 里是 bytes，某些工具/中间件按它取原始路径。
    """

    def __init__(self, app: Callable, prefix: str):
        self._app = app
        self._prefix = normalize_prefix(prefix)
        self._prefix_bytes = self._prefix.encode("ascii")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http" and self._prefix:
            path = scope.get("path") or ""
            if path == self._prefix or path.startswith(self._prefix + "/"):
                new_scope = dict(scope)
                new_scope["path"] = path[len(self._prefix) :] or "/"
                new_scope["root_path"] = f"{scope.get('root_path') or ''}{self._prefix}"
                raw_path = scope.get("raw_path")
                if raw_path:
                    new_scope["raw_path"] = raw_path[len(self._prefix_bytes) :] or b"/"
                scope = new_scope
        await self._app(scope, receive, send)
