"""生成侧 LLM 客户端：OpenAI 兼容 + 流式（规划 10.1「输出一律流式」）。

不引 OpenAI SDK，用 httpx 手写 SSE 解析——和前端「原生 fetch 不引 axios」是同一条技术选择：
多一层 SDK 只多一份版本兼容负担，而协议本身只有十几行。

错误一律翻成中文可读信息（401 无效 Key / 402 余额不足 / 429 限流 / 超时），
前端拿到 `error` 事件就能直接显示，不需要自己做状态码映射。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.config import Settings
from app.errors import LLMError, NotConfigured


class LLMClient:
    def __init__(self, settings: Settings):
        self._settings = settings

    @property
    def configured(self) -> bool:
        return self._settings.llm_configured

    @property
    def model(self) -> str:
        return self._settings.llm_model

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """逐块吐出增量文本。未配 Key 时抛 `NotConfigured`（前端据此引导去配置）。"""
        if not self.configured:
            raise NotConfigured("后端未配置 LLM Key：请在 yunhai-agent/.env 里填 LLM_API_KEY")

        url = f"{self._settings.llm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self._settings.llm_model,
            "messages": messages,
            "stream": True,
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        timeout = httpx.Timeout(self._settings.llm_timeout_s, connect=10.0)
        try:
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream("POST", url, json=payload, headers=headers) as response,
            ):
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", errors="replace")[:300]
                    raise LLMError(_describe_status(response.status_code, detail))
                async for line in response.aiter_lines():
                    delta = _parse_delta(line)
                    if delta:
                        yield delta
        except httpx.TimeoutException as exc:
            raise LLMError("生成超时，请缩短问题或稍后重试") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"生成服务连接失败：{exc}") from exc


def _parse_delta(line: str) -> str:
    """解析一行 OpenAI 兼容的流式响应，返回增量文本（无内容时返回空串）。"""
    if not line or not line.startswith("data:"):
        return ""
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return ""
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return ""
    choices = payload.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _describe_status(status: int, detail: str) -> str:
    base = (
        "LLM Key 无效或已过期"
        if status == 401
        else "LLM 账户余额不足"
        if status == 402
        else "请求过于频繁，请稍后再试"
        if status == 429
        else "接口地址或模型名不正确"
        if status == 404
        else f"生成服务报错（HTTP {status}）"
    )
    return f"{base}：{detail}" if detail else base
