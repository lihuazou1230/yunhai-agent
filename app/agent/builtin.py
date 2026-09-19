"""内置工具（规划 11.2）。

| 工具 | executor | 说明 |
| --- | --- | --- |
| `search_knowledge` | server | 知识库检索；命中即产出 citation（前端照旧渲染引用块） |
| `get_date` | server | 系统日期/时间/星期——零风险即时查询 |
| `task_crud` | **client** | 任务的增删查改；数据在前端，由前端执行后回环 |
| `get_weather` | **client** | 读前端已有的天气缓存（不新增数据源）；缓存为空就如实说 |

client 工具在这里**只有声明**（名称/描述/schema），没有 handler——
`ToolRegistry.register` 也会拦住"client 工具带 handler"这种混淆。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from app.agent.tools import EMPTY_PARAMETERS, Tool, ToolOutcome, ToolRegistry
from app.rag.prompt import build_context, citations_of
from app.rag.types import Retrieved

SEARCH_KNOWLEDGE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "检索用的自然语言问题（用用户的原话或改写后的短句都行）",
            "minLength": 2,
        },
        "top_k": {
            "type": "integer",
            "description": "返回片段数，默认 4；需要更全的上下文时再调大",
        },
    },
    "required": ["query"],
}

TASK_CRUD_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["list", "create", "update", "complete", "uncomplete", "delete", "archive"],
            "description": (
                "list=查任务（今天还剩哪些活/有没有 X）；create=新建；update=改标题/日期/优先级；"
                "complete=标记完成；uncomplete=取消完成；delete=删除；archive=归档"
            ),
        },
        "title": {
            "type": "string",
            "description": "create 用：任务标题，动词开头、去掉「帮我」这类赘词，不要带时间词",
            "maxLength": 120,
        },
        "due_date": {
            "type": "string",
            "format": "date",
            "description": "YYYY-MM-DD。用户说「明天/后天/下周一」时必须换算成绝对日期再传",
        },
        "priority": {
            "type": "string",
            "enum": ["urgent", "high", "medium", "low"],
            "description": "优先级。出现「紧急/重要/马上」判 urgent 或 high；没说就 medium",
        },
        "query": {
            "type": "string",
            "description": "list/update/complete/delete 用：按关键字定位任务（用户提到的那几个字）",
        },
        "ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "已经知道任务 id 时直接用 id，比 query 更准（上一轮工具结果里会给）",
        },
        "filter": {
            "type": "string",
            "enum": ["all", "today", "active", "completed", "overdue"],
            "description": "list 用：筛选范围，默认 all（未归档的全部任务）",
        },
        "subtasks": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "拆解出的执行步骤。用户说「把 X 拆成可执行的步骤/子任务」时用："
                "先 create 一条标题是目标本身的任务，再把 3~6 条步骤放进这里——"
                "每条动词开头、能独立完成。update 时表示往已有任务上加步骤（见 subtasks_mode）"
            ),
        },
        "subtasks_mode": {
            "type": "string",
            "enum": ["append", "replace"],
            "description": "update 时如何处置已有子任务：append（默认，追加）或 replace（整体替换）",
        },
    },
    "required": ["action"],
}

GET_WEATHER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "城市名；不传就用用户上次定位的城市"},
    },
    "required": [],
}


def search_knowledge_tool(retriever, settings) -> Tool:
    async def handler(query: str, top_k: int | None = None) -> ToolOutcome:
        k = max(1, min(top_k or settings.top_k, 10))
        # 语义与字面**都查一遍再合并**：第十阶段实测两者互补（语义 recall@4 15/16、字面 16/16，
        # 且漏的题各不相同）。工具层做并集，比只挑一路少漏——代价是上下文里多几块，
        # 所以上限压在 2k 并用 RRF 排序，超了就丢尾部。
        semantic = await asyncio.to_thread(retriever.filtered, query, mode="semantic", top_k=k)
        lexical = await asyncio.to_thread(retriever.filtered, query, mode="lexical", top_k=k)
        mode = "semantic" if len(semantic) >= len(lexical) else "lexical"
        hits = merge_by_rank([semantic, lexical], limit=k * 2)
        if not hits:
            return ToolOutcome(
                ok=True,
                content="知识库里没有检索到与该查询相关的内容（阈值拦截）。",
                meta={"hits": 0, "query": query, "modes": ["semantic", "lexical"]},
            )
        citations = citations_of(hits)
        return ToolOutcome(
            ok=True,
            # 片段自带 [编号]，与 citations 一一对应，模型只要照抄编号就能对上引用
            content=build_context(hits),
            citations=citations,
            meta={
                "hits": len(hits),
                "query": query,
                "mode": mode,
                "modes": ["semantic", "lexical"],
                "sources": sorted({hit.chunk.source for hit in hits}),
                "scores": [round(hit.score, 4) for hit in hits],
            },
        )

    return Tool(
        name="search_knowledge",
        description=(
            "检索用户的知识库（上传的文档、FAQ、技术笔记）。"
            "凡是问「文档里怎么写的 / 工作台怎么用 / 某个参数是多少」都先用它，"
            "并且只依据它返回的片段作答。返回内容里的 [编号] 就是引用来源的编号。"
        ),
        parameters=SEARCH_KNOWLEDGE_PARAMETERS,
        executor="server",
        handler=handler,
        timeout_s=15.0,
    )


def get_date_tool() -> Tool:
    async def handler() -> ToolOutcome:
        now = datetime.now()
        weekdays = "一二三四五六日"
        iso = now.isocalendar()
        payload = {
            "date": now.strftime("%Y-%m-%d"),
            "weekday": f"星期{weekdays[now.weekday()]}",
            "time": now.strftime("%H:%M"),
            "iso_week": f"{iso.year}-W{iso.week:02d}",
            "timestamp": int(now.timestamp()),
        }
        text = (
            f"今天是 {payload['date']}（{payload['weekday']}），当前时间 {payload['time']}，"
            f"ISO 周 {payload['iso_week']}。"
        )
        return ToolOutcome(ok=True, content=text, meta=payload)

    return Tool(
        name="get_date",
        description="获取当前日期、星期与时间。凡是涉及「今天/明天/这周」的计算都先用它，不要凭记忆猜日期。",
        parameters=EMPTY_PARAMETERS,
        executor="server",
        handler=handler,
    )


def task_crud_tool() -> Tool:
    return Tool(
        name="task_crud",
        description=(
            "读写用户工作台里的任务：查询（list）、新建（create）、修改（update）、"
            "完成/取消完成（complete/uncomplete）、删除（delete）、归档（archive）。"
            "用户提到「我的任务/待办/今天还剩什么/帮我加一条」都用它。"
            "**拆解也用它**：用户说「把 X 拆成可执行的步骤」时，create 一条标题是 X 的任务，"
            "把拆出的 3~6 条步骤放进 `subtasks`——不要把步骤只写在回答里，拆解结果要真的落进工作台。"
            "任务数据只存在于用户的工作台里，不调用它你看不到任何任务。"
        ),
        parameters=TASK_CRUD_PARAMETERS,
        executor="client",
    )


def get_weather_tool() -> Tool:
    return Tool(
        name="get_weather",
        description=(
            "读取用户本机已缓存的天气（不新增数据源）。"
            "缓存为空或已过期时会返回失败，这时如实告诉用户去仪表板刷新天气，不要编造天气数据。"
        ),
        parameters=GET_WEATHER_PARAMETERS,
        executor="client",
    )


def merge_by_rank(ranked_lists: list[list[Retrieved]], limit: int) -> list[Retrieved]:
    """把多路检索结果按名次融合（RRF）：`1/(60+rank)` 累加，纯看名次、不看分数量纲。

    去重按 `chunk_id`；保留第一次出现的 Retrieved（它的 score 是那一路自己的量纲，
    引用块里展示的就是"这一路认为它有多相关"）。融合只影响**顺序与截断**，
    所以在 k 取够大时，召回等于并集。
    """
    scores: dict[str, float] = {}
    picked: dict[str, Retrieved] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
            picked.setdefault(chunk_id, hit)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:limit]
    return [picked[cid] for cid in ordered]


def build_builtin_tools(runtime) -> list[Tool]:
    """内置工具清单（运行时注入）。

    返回 list 而不是直接造一个注册表：Runtime 需要先拿到「那个」注册表对象
    （AgentLoop 持有它的引用），再把工具逐个注册进去。
    """
    return [
        search_knowledge_tool(runtime.retriever, runtime.settings),
        get_date_tool(),
        task_crud_tool(),
        get_weather_tool(),
    ]


def build_registry(runtime) -> ToolRegistry:
    """便捷入口（测试与脚本用）：直接拿一个装好内置工具的注册表。"""
    return ToolRegistry(build_builtin_tools(runtime))
