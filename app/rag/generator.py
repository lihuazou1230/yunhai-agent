"""生成侧 LLM 客户端：OpenAI 兼容 + 流式（规划 10.1「输出一律流式」）。

不引 OpenAI SDK，用 httpx 手写 SSE 解析——和前端「原生 fetch 不引 axios」是同一条技术选择：
多一层 SDK 只多一份版本兼容负担，而协议本身只有十几行。

两个入口：
- `stream_chat`：纯文本流（第十阶段的 RAG 直答路径继续用它）；
- `stream_with_tools`：带工具调用的流（第十一阶段的 ReAct 循环用）。
  工具调用在流里是**碎片**：id/name 只来一次、arguments 一段段拼——这是最容易写错的地方，
  所以累积逻辑单独放在 `TurnResult.absorb_tool_call_delta` 里并有专门用例钉住。

错误一律翻成中文可读信息（401 无效 Key / 402 余额不足 / 429 限流 / 超时），
前端拿到 `error` 事件就能直接显示，不需要自己做状态码映射。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import Settings
from app.errors import LLMError, NotConfigured


@dataclass
class ToolCall:
    """一次工具调用（OpenAI 兼容格式的累积结果）。"""

    id: str
    name: str
    arguments: str  # 原始 JSON 字符串（可能被模型写坏，解析失败由调用方兜底）

    def parsed_arguments(self) -> dict[str, Any]:
        try:
            data = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


@dataclass
class TurnResult:
    """一轮 LLM 调用的收口信息（流式过程中由 `stream_with_tools` 填充）。

    为什么用「调用方传入的可变对象」而不是让生成器 return：
    Python 的异步生成器不能把返回值交给 `async for`，而工具调用必须等流结束才完整。
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    total_tokens: int = 0
    # 流式 tool_calls 的分片累积（index -> {id, name, arguments}）
    partial: dict[int, dict[str, str]] = field(default_factory=dict, repr=False)

    def absorb_tool_call_delta(self, delta: dict[str, Any]) -> None:
        index = int(delta.get("index", 0))
        slot = self.partial.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if delta.get("id"):
            slot["id"] = str(delta["id"])
        function = delta.get("function") or {}
        if function.get("name"):
            slot["name"] += str(function["name"])
        if function.get("arguments"):
            slot["arguments"] += str(function["arguments"])

    def finish_tool_calls(self) -> None:
        self.tool_calls = [
            ToolCall(id=slot["id"] or f"call_{index}", name=slot["name"], arguments=slot["arguments"])
            for index, slot in sorted(self.partial.items())
        ]

    def as_tool_call_messages(self) -> list[dict[str, Any]]:
        """把这一轮的工具调用转成回填给模型的 assistant 消息格式。"""
        return [
            {
                "role": "assistant",
                "content": self.content or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments or "{}"},
                    }
                    for call in self.tool_calls
                ],
            }
        ]


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
        result = TurnResult()
        async for piece in self.stream_with_tools(messages, tools=None, result=result):
            yield piece

    async def stream_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        result: TurnResult,
    ) -> AsyncIterator[str]:
        """流式调用一次 chat/completions，逐块 yield 增量文本，并把收口信息写进 `result`。

        `tools=None` 时退化成纯文本流（第十阶段的路径），payload 里也不带 tools 字段。
        """
        if not self.configured:
            raise NotConfigured("后端未配置 LLM Key：请在 yunhai-agent/.env 里填 LLM_API_KEY")

        url = f"{self._settings.llm_base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "stream": True,
            # 让流里也带上用量，token 预算护栏才有真实数字可用
            "stream_options": {"include_usage": True},
            "temperature": self._settings.llm_temperature,
            "max_tokens": self._settings.llm_max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

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
                    chunk = _parse_chunk(line, result)
                    if chunk:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise LLMError("生成超时，请缩短问题或稍后重试") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"生成服务连接失败：{exc}") from exc

        result.finish_tool_calls()


def _parse_chunk(line: str, result: TurnResult) -> str:
    """解析一行流式响应：把工具调用碎片与用量写进 result，返回可展示的增量文本。"""
    if not line or not line.startswith("data:"):
        return ""
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return ""
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return ""

    usage = payload.get("usage")
    if isinstance(usage, dict) and usage.get("total_tokens"):
        result.total_tokens = int(usage["total_tokens"])

    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if choice.get("finish_reason"):
        result.finish_reason = str(choice["finish_reason"])

    delta = choice.get("delta") or {}
    for call_delta in delta.get("tool_calls") or []:
        result.absorb_tool_call_delta(call_delta)

    content = delta.get("content")
    if isinstance(content, str) and content:
        result.content += content
        return content
    return ""


def _parse_delta(line: str) -> str:
    """兼容旧接口：只要增量文本（单测与 RAG 路径都在用）。"""
    return _parse_chunk(line, TurnResult())


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
