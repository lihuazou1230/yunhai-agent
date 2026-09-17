"""LLM 客户端测试：流式增量解析、错误中文映射、未配置时的明确报错。"""

from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.errors import LLMError, NotConfigured
from app.rag.generator import LLMClient, _describe_status, _parse_delta


def test_parse_delta_extracts_content():
    line = 'data: {"choices":[{"delta":{"content":"分块"}}]}'
    assert _parse_delta(line) == "分块"


@pytest.mark.parametrize(
    "line",
    [
        "",
        ": keep-alive",
        "data: [DONE]",
        "data: {坏 JSON",
        'data: {"choices":[]}',
        'data: {"choices":[{"delta":{}}]}',
    ],
)
def test_parse_delta_ignores_non_content_lines(line: str):
    assert _parse_delta(line) == ""


def test_describe_status_is_readable_chinese():
    assert "Key 无效" in _describe_status(401, "unauthorized")
    assert "余额" in _describe_status(402, "")
    assert "频繁" in _describe_status(429, "")
    assert "模型名" in _describe_status(404, "")
    assert "500" in _describe_status(500, "boom")


async def test_unconfigured_llm_raises_not_configured():
    client = LLMClient(Settings(llm_api_key="", _env_file=None))
    with pytest.raises(NotConfigured) as error:
        async for _ in client.stream_chat([{"role": "user", "content": "hi"}]):
            pass
    assert "LLM_API_KEY" in error.value.message


async def test_http_error_is_mapped_to_llm_error(monkeypatch):
    """把 httpx.AsyncClient 换成会返回 402 的假客户端，验证错误映射。"""

    class FakeResponse:
        status_code = 402

        async def aread(self) -> bytes:
            return b'{"error":"Insufficient Balance"}'

    class FakeStream:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, *args):
            return False

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return FakeStream()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    client = LLMClient(Settings(llm_api_key="k", _env_file=None))
    with pytest.raises(LLMError) as error:
        async for _ in client.stream_chat([{"role": "user", "content": "hi"}]):
            pass
    assert "余额不足" in error.value.message


async def test_stream_yields_pieces(monkeypatch):
    class FakeResponse:
        status_code = 200

        async def aiter_lines(self):
            for line in (
                'data: {"choices":[{"delta":{"content":"甲"}}]}',
                "",
                'data: {"choices":[{"delta":{"content":"乙"}}]}',
                "data: [DONE]",
            ):
                yield line

    class FakeStream:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, *args):
            return False

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return FakeStream()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    client = LLMClient(Settings(llm_api_key="k", _env_file=None))
    pieces = [piece async for piece in client.stream_chat([{"role": "user", "content": "hi"}])]
    assert pieces == ["甲", "乙"]
