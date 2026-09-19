"""可插拔工具注册表（规划 11.1）。

**tool = 名称 + 描述 + JSON Schema + 执行函数**，运行时注入——加一个工具只要 `register()`。

两类执行器（这是本阶段最关键的一个架构决定）：

| executor | 谁执行 | 例子 | 为什么 |
| --- | --- | --- | --- |
| `server` | yunhai-agent 进程内 | `search_knowledge`、`get_date` | 数据就在后端（向量库、系统时间） |
| `client` | **前端**，在工作台数据上执行 | `task_crud`、`get_weather` | 任务与天气缓存只存在于用户的工作台里，后端不该为了"能调工具"复制一份真相 |

client 工具的往返：后端发 `tool_call(executor=client)` 并把这一轮的 messages 存起来 →
前端执行 → `POST /api/ask/resume` 带观察结果回来 → 循环继续。这样后端始终无状态地"想"，
前端始终是数据的唯一真相。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

ExecutorKind = Literal["server", "client"]


@dataclass
class ToolOutcome:
    """工具执行结果。

    `content` 是**给模型看的观察结果**（自然语言/JSON 文本），
    `meta` 是给前端与日志用的结构化结果，`citations` 是需要展示给用户的引用块。
    """

    ok: bool
    content: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def as_observation(self) -> str:
        """回填给模型的内容：失败也要回填，让模型有机会换个参数重试。"""
        if self.ok:
            return self.content or "（工具执行成功，但没有返回内容）"
        return f"工具执行失败：{self.error or '未知错误'}"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    executor: ExecutorKind = "server"
    handler: Callable[..., Awaitable[ToolOutcome]] | None = None
    timeout_s: float = 10.0

    def spec(self) -> dict[str, Any]:
        """转成 OpenAI 兼容的 tools 定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


EMPTY_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}, "required": []}


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.executor == "server" and tool.handler is None:
            raise ValueError(f"服务端工具 {tool.name} 必须提供 handler")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    def client_tool_names(self) -> list[str]:
        return [name for name, tool in self._tools.items() if tool.executor == "client"]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        """执行服务端工具：带**参数校验 + 超时 + 异常兜底**，永不向上抛。

        永不抛是刻意的：工具失败对 agent 来说是一条**观察结果**（"这条路走不通"），
        不是流程终止。否则一个写错参数的调用就会把整轮对话打断。
        """
        tool = self.get(name)
        if tool is None:
            return ToolOutcome(ok=False, content="", error=f"没有名为 {name} 的工具", meta={"tool": name})
        if tool.executor != "server" or tool.handler is None:
            return ToolOutcome(
                ok=False,
                content="",
                error=f"{name} 是客户端工具，必须在工作台前端执行",
                meta={"tool": name, "executor": tool.executor},
            )

        problems = validate_arguments(tool.parameters, arguments)
        if problems:
            return ToolOutcome(
                ok=False,
                content="",
                error="参数不合法：" + "；".join(problems),
                meta={"tool": name, "arguments": arguments},
            )

        try:
            return await asyncio.wait_for(tool.handler(**arguments), timeout=tool.timeout_s)
        except asyncio.TimeoutError:
            return ToolOutcome(
                ok=False,
                content="",
                error=f"工具执行超时（>{tool.timeout_s:g}s）",
                meta={"tool": name},
            )
        except TypeError as exc:  # handler 签名与 schema 不一致：属于我们的 bug，如实说
            return ToolOutcome(ok=False, content="", error=f"工具参数不匹配：{exc}", meta={"tool": name})
        except Exception as exc:  # noqa: BLE001 - 任何异常都降级成观察结果
            return ToolOutcome(ok=False, content="", error=str(exc) or exc.__class__.__name__, meta={"tool": name})


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """极简 JSON Schema 校验（只覆盖我们写 schema 时用到的关键字）。

    为什么自己写而不是引 jsonschema：规则只用到 required/type/enum/边界，
    自己写能给出**中文、可操作**的报错（"due_date 需要 YYYY-MM-DD 格式"），
    这些报错会直接回填给模型，措辞质量决定它能不能自我修正。
    """
    problems: list[str] = []
    if not isinstance(arguments, dict):
        return ["参数必须是一个 JSON 对象"]

    properties = schema.get("properties") or {}
    for key in schema.get("required") or []:
        if arguments.get(key) in (None, ""):
            problems.append(f"缺少必填参数 {key}")

    for key, value in arguments.items():
        rule = properties.get(key)
        if rule is None:
            if schema.get("additionalProperties") is False:
                problems.append(f"不认识的参数 {key}")
            continue
        problems.extend(_check_value(key, value, rule))
    return problems


def _check_value(key: str, value: Any, rule: dict[str, Any]) -> list[str]:
    expected = rule.get("type")
    if value is None:
        return []
    if expected == "string":
        if not isinstance(value, str):
            return [f"{key} 需要字符串"]
        if rule.get("minLength") and len(value) < int(rule["minLength"]):
            return [f"{key} 太短（至少 {rule['minLength']} 个字符）"]
        if rule.get("maxLength") and len(value) > int(rule["maxLength"]):
            return [f"{key} 太长（最多 {rule['maxLength']} 个字符）"]
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return [f"{key} 需要整数"]
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return [f"{key} 需要数字"]
    elif expected == "boolean":
        if not isinstance(value, bool):
            return [f"{key} 需要 true/false"]
    elif expected == "array":
        if not isinstance(value, list):
            return [f"{key} 需要数组"]
        item_rule = rule.get("items")
        if isinstance(item_rule, dict):
            for item in value:
                problems = _check_value(f"{key} 的每一项", item, item_rule)
                if problems:
                    return problems
    elif expected == "object" and not isinstance(value, dict):
        return [f"{key} 需要对象"]

    allowed = rule.get("enum")
    if allowed and value not in allowed:
        return [f"{key} 只能是 {'/'.join(str(a) for a in allowed)}，收到的是 {value!r}"]
    pattern_hint = rule.get("format")
    if pattern_hint == "date" and isinstance(value, str) and not _looks_like_date(value):
        return [f"{key} 需要 YYYY-MM-DD 格式"]
    return []


def _looks_like_date(value: str) -> bool:
    parts = value.split("-")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return False
    year, month, day = (int(p) for p in parts)
    return 1970 <= year <= 2999 and 1 <= month <= 12 and 1 <= day <= 31
