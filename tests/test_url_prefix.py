"""子路径前缀剥离的测试（服务器形态挂在 IIS 子应用 /yhai 下）。

两条路都要覆盖：
1. 中间件本身的单元测试（直接喂 ASGI scope，看 path/root_path/raw_path 被改成什么样）；
2. 走真应用的集成测试（`AGENT_URL_PREFIX=/yhai` 时 `/yhai/api/health` 通、`/api/health` 不通）。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.runtime import Runtime, set_runtime
from app.url_prefix import UrlPrefixMiddleware, normalize_prefix


def test_normalize_prefix() -> None:
    assert normalize_prefix("") == ""
    assert normalize_prefix(None) == ""
    assert normalize_prefix("/") == ""
    assert normalize_prefix("   ") == ""
    assert normalize_prefix("yhai") == "/yhai"
    assert normalize_prefix("/yhai") == "/yhai"
    assert normalize_prefix("/yhai/") == "/yhai"
    assert normalize_prefix("  /yhai/  ") == "/yhai"


async def _collect(prefix: str, path: str, raw_path: bytes | None = None) -> dict:
    """跑一次中间件，把最终到达内层应用的 scope 记下来。"""
    seen: dict = {}

    async def inner(scope, receive, send):  # noqa: ANN001
        seen.update(scope)

    scope = {"type": "http", "path": path, "root_path": "", "method": "GET"}
    if raw_path is not None:
        scope["raw_path"] = raw_path
    await UrlPrefixMiddleware(inner, prefix)(scope, None, None)
    return seen


async def test_prefix_is_stripped() -> None:
    seen = await _collect("/yhai", "/yhai/api/health")
    assert seen["path"] == "/api/health"
    assert seen["root_path"] == "/yhai"


async def test_prefix_only_path_becomes_root() -> None:
    seen = await _collect("/yhai", "/yhai")
    assert seen["path"] == "/"


async def test_raw_path_is_stripped_too() -> None:
    seen = await _collect("/yhai", "/yhai/api/ask", raw_path=b"/yhai/api/ask")
    assert seen["raw_path"] == b"/api/ask"


async def test_non_matching_path_passes_through() -> None:
    # 前缀不匹配就原样放行：本机直连 127.0.0.1:8000/api/health 必须照旧能用
    seen = await _collect("/yhai", "/api/health")
    assert seen["path"] == "/api/health"
    assert seen["root_path"] == ""


async def test_similar_prefix_is_not_stripped() -> None:
    # /yhaixxx 不是 /yhai 下的路径，不能被误剥成 xxx
    seen = await _collect("/yhai", "/yhaixxx/api/health")
    assert seen["path"] == "/yhaixxx/api/health"


async def test_empty_prefix_is_noop() -> None:
    seen = await _collect("", "/yhai/api/health")
    assert seen["path"] == "/yhai/api/health"
    assert seen["root_path"] == ""


async def test_non_http_scope_untouched() -> None:
    seen: dict = {}

    async def inner(scope, receive, send):  # noqa: ANN001
        seen.update(scope)

    await UrlPrefixMiddleware(inner, "/yhai")({"type": "lifespan"}, None, None)
    assert "path" not in seen


def test_prefixed_requests_reach_routes(settings: Settings, stub_llm) -> None:
    """集成：设了前缀后，带前缀的路径能进路由，health 也会回显前缀。"""
    prefixed = settings.model_copy(update={"agent_url_prefix": "/yhai"})
    set_runtime(Runtime.build(prefixed, llm=stub_llm))
    try:
        with TestClient(create_app()) as client:
            response = client.get("/yhai/api/health")
            assert response.status_code == 200
            assert response.json()["url_prefix"] == "/yhai"

            # 不带前缀的老路径**故意保留**：
            #  - 服务器上这个进程只被 IIS 子应用 /yhai 暴露出去，外网到不了裸 /api/*；
            #  - 而"装完自己先探活一下"（install.ps1 直连 127.0.0.1:PORT/api/health）走的就是裸路径。
            assert client.get("/api/health").status_code == 200
    finally:
        set_runtime(None)


def test_without_prefix_nothing_changes(client) -> None:
    """不设前缀时行为与改动前完全一致（本地开发形态）。"""
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["url_prefix"] == ""


def test_env_var_name_is_agent_url_prefix(monkeypatch) -> None:
    """环境变量名必须是 `AGENT_URL_PREFIX`。

    pydantic-settings 按**字段名**推环境变量名：字段若只叫 `url_prefix`，
    它读的是 `URL_PREFIX`，而 .env 里写的 `AGENT_URL_PREFIX` 会被静默忽略
    —— 现象是"前缀怎么设都不生效、子路径全 404"。这个坑实测踩过一次，
    所以在这里钉一条：拿真环境变量构造 Settings，断言字段真被填上。
    """
    monkeypatch.setenv("AGENT_URL_PREFIX", "/yhai")
    # _env_file=None：不读 .env，确保命中的是环境变量本身
    assert Settings(_env_file=None).agent_url_prefix == "/yhai"
