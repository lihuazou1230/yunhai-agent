"""第十一阶段：工具注册表、ReAct 循环与护栏。

全部用假 LLM（`StubLLM` 的 script）驱动：真模型不配合的事情（反复调用同一个失败工具、
token 爆预算、只输出纯文本），假模型必须配合，否则这些护栏根本测不到。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.agent.tools import Tool, ToolOutcome, ToolRegistry, validate_arguments
from app.errors import RunNotFound
from app.rag.types import Chunk, Retrieved
from app.runtime import Runtime
from tests.conftest import StubLLM

# ---------------- 工具注册表 ----------------


def ok_handler(**kwargs) -> ToolOutcome:  # 占位（真实工具都是 async，这里只在同步注册用例里用不到）
    raise AssertionError("不会被调用")


async def test_registry_registers_and_exposes_openai_specs():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="get_date",
            description="取当前日期",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_echo_outcome,
        )
    )
    registry.register(
        Tool(name="task_crud", description="任务增删查改", parameters={"type": "object", "properties": {}}, executor="client")
    )
    assert registry.names() == ["get_date", "task_crud"]
    assert registry.client_tool_names() == ["task_crud"]
    spec = registry.specs()[0]
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "get_date"


def test_server_tool_without_handler_is_rejected():
    with pytest.raises(ValueError):
        ToolRegistry().register(Tool(name="bad", description="x", parameters={}, executor="server"))


async def _echo_outcome(**kwargs) -> ToolOutcome:
    return ToolOutcome(ok=True, content=json.dumps(kwargs, ensure_ascii=False), meta={"echo": True})


async def test_registry_call_validates_arguments_before_running():
    registry = ToolRegistry(
        [
            Tool(
                name="task_crud",
                description="x",
                parameters={
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
                handler=_echo_outcome,
            )
        ]
    )
    outcome = await registry.call("task_crud", {})
    assert outcome.ok is False
    assert "缺少必填参数 title" in outcome.error
    ok = await registry.call("task_crud", {"title": "写周报"})
    assert ok.ok and ok.meta["echo"] is True


async def test_registry_call_never_raises_and_reports_unknown_or_client_tools():
    registry = ToolRegistry([Tool(name="task_crud", description="x", parameters={}, executor="client")])
    unknown = await registry.call("nope", {})
    assert unknown.ok is False and "没有名为 nope 的工具" in unknown.error
    client = await registry.call("task_crud", {})
    assert client.ok is False and "客户端工具" in client.error


async def test_registry_call_times_out_and_wraps_exceptions():
    async def slow(**kwargs):
        await asyncio.sleep(0.5)
        return ToolOutcome(ok=True, content="太慢了")

    async def boom(**kwargs):
        raise RuntimeError("外部接口 500")

    registry = ToolRegistry(
        [
            Tool(name="slow", description="x", parameters={}, handler=slow, timeout_s=0.05),
            Tool(name="boom", description="x", parameters={}, handler=boom),
        ]
    )
    timeout = await registry.call("slow", {})
    assert timeout.ok is False and "超时" in timeout.error
    failed = await registry.call("boom", {})
    assert failed.ok is False and "外部接口 500" in failed.error


def test_validate_arguments_covers_types_enum_date_and_unknown_keys():
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "list"]},
            "due_date": {"type": "string", "format": "date"},
            "top_k": {"type": "integer"},
            "ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action"],
        "additionalProperties": False,
    }
    assert validate_arguments(schema, {"action": "create", "due_date": "2026-09-18"}) == []
    assert "只能是 create/list" in validate_arguments(schema, {"action": "drop"})[0]
    assert "YYYY-MM-DD" in validate_arguments(schema, {"action": "create", "due_date": "明天"})[0]
    assert "需要整数" in validate_arguments(schema, {"action": "list", "top_k": "3"})[0]
    assert "需要数组" in validate_arguments(schema, {"action": "list", "ids": "a"})[0]
    assert "不认识的参数" in validate_arguments(schema, {"action": "list", "nope": 1})[0]


async def test_search_knowledge_merges_semantic_and_lexical(settings, sample_md):
    """工具层做两路并集：语义漏掉的题字面能召回（第十阶段实测两者互补）。"""
    runtime = Runtime.build(settings, llm=StubLLM())
    runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    tool = runtime.tools.get("search_knowledge")
    assert tool is not None
    outcome = await tool.handler(query="分块默认块长", top_k=2)
    assert outcome.ok is True
    assert outcome.meta["hits"] >= 1
    assert outcome.meta["modes"] == ["semantic", "lexical"]
    assert "500" in outcome.content  # 命中块里带着答案（字面那一路把它捞了回来）


def test_merge_by_rank_dedupes_and_prefers_shared_hits():
    from app.agent.builtin import merge_by_rank

    def hit(chunk_id: str, score: float):
        return Retrieved(
            chunk=Chunk(
                chunk_id=chunk_id,
                text=chunk_id,
                source="s.md",
                source_type="md",
                doc_id="d",
                item_id="raw",
                chunk_index=0,
                uploaded_at="",
            ),
            score=score,
        )

    merged = merge_by_rank([[hit("a", 0.9), hit("b", 0.5)], [hit("b", 0.8), hit("c", 0.7)]], limit=3)
    # b 在两路都靠前，融合后第一；a、c 各只在一条路里出现过
    assert [item.chunk.chunk_id for item in merged] == ["b", "a", "c"]
    assert merged[0].score == 0.5  # 保留"第一次出现"那一路的分数（量纲不混用）


# ---------------- ReAct 循环：知识库工具 ----------------


def make_runtime(settings, script: list[dict], sample_md: str) -> tuple[Runtime, StubLLM]:
    llm = StubLLM(script=script)
    runtime = Runtime.build(settings, llm=llm)
    runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    return runtime, llm


def parse_frames(frames: list[str]) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for frame in frames:
        name = ""
        data = "{}"
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        if name:
            events.append((name, json.loads(data)))
    return events


async def collect(runtime: Runtime, question: str, **kwargs) -> list[tuple[str, dict]]:
    return parse_frames([frame async for frame in runtime.agent.stream(question, **kwargs)])


async def test_agent_decides_to_search_knowledge_then_answers(settings, sample_md):
    """验收项 2：agent 能自主决定"先查知识库再回答"。"""
    runtime, llm = make_runtime(
        settings,
        [
            {"tool_calls": [{"name": "search_knowledge", "arguments": {"query": "分块默认块长"}}]},
            {"content": "分块默认 500 个字符、重叠 80。[1]"},
        ],
        sample_md,
    )
    events = await collect(runtime, "分块默认的块长和重叠是多少？")

    kinds = [kind for kind, _ in events]
    assert "tool_call" in kinds and "tool_result" in kinds
    call = next(data for kind, data in events if kind == "tool_call")
    assert call["name"] == "search_knowledge" and call["executor"] == "server"
    result = next(data for kind, data in events if kind == "tool_result")
    assert result["ok"] is True and result["citations"]
    done = events[-1][1]
    assert done["status"] == "ok" and done["kind"] == "kb" and done["fallback"] == "kb"
    assert done["rounds"] == 1 and done["tokens"] > 0
    assert "分块默认 500" in "".join(data["text"] for kind, data in events if kind == "token")
    # 第二次调用模型时，工具结果已经作为 tool 消息回填
    second_call_messages = llm.calls[1]["messages"]
    assert any(m.get("role") == "tool" for m in second_call_messages)


async def test_agent_knowledge_miss_falls_back_to_refuse_mode(settings):
    """知识库是空的：工具调用成功但 0 命中 → 依据类型回落到配置的兜底模式。"""
    runtime = Runtime.build(settings, llm=StubLLM(script=[
        {"tool_calls": [{"name": "search_knowledge", "arguments": {"query": "世界杯冠军"}}]},
        {"content": "知识库里没有相关内容，我不作回答。"},
    ]))
    events = await collect(runtime, "世界杯冠军是谁？", fallback_mode="refuse")
    done = events[-1][1]
    assert done["kind"] == "tool" and done["fallback"] == "refuse" and done["hit_count"] == 0


async def test_agent_plain_chat_without_tools_is_marked_chat(settings):
    """验收项 4 前半：闲聊不产生工具调用，kind=chat。"""
    runtime = Runtime.build(settings, llm=StubLLM(script=[{"content": "今天天气不错，适合出门走走。"}]))
    events = await collect(runtime, "讲个冷笑话")
    done = events[-1][1]
    assert done["kind"] == "chat" and done["fallback"] == "chat" and done["tools"] == []
    # 系统提示里带了"闲聊直接回答、且不会被记住"的口径
    assert "闲聊" in runtime.llm.calls[0]["messages"][0]["content"]


async def test_agent_history_excludes_light_chat(settings, sample_md):
    """验收项 4 后半：闲聊不进任务上下文（用户那句与回答都不进）。"""
    runtime, llm = make_runtime(settings, [{"content": "今天好好休息。"}], sample_md)

    chat = await collect(runtime, "今天好累啊")
    session_id = chat[-1][1]["session_id"]
    # 闲聊那一轮：**用户那句与回答都被打上 chat 标记**（否则用户半句照样会被回填）
    kinds = [(m["meta"] or {}).get("kind") for m in runtime.sessions.get_session(session_id)["messages"]]
    assert kinds == ["chat", "chat"]

    # 第二轮：真正的知识库问答
    llm.script = [
        {"tool_calls": [{"name": "search_knowledge", "arguments": {"query": "分块默认块长"}}]},
        {"content": "500 字符。[1]"},
    ]
    await collect(runtime, "分块默认多大？", session_id=session_id)

    # 第三轮：看回填给模型的上下文
    llm.script = [{"content": "好的"}]
    await collect(runtime, "还有别的吗？", session_id=session_id)
    contents = " ".join(str(m.get("content") or "") for m in llm.calls[-1]["messages"])
    assert "今天好累啊" not in contents  # 闲聊被剔除
    assert "500 字符。[1]" in contents  # 知识库那轮保留


# ---------------- ReAct 循环：客户端工具回环 ----------------


async def test_client_tool_pauses_then_resume_completes_the_turn(settings):
    """client 工具：后端停在等前端执行，resume 后继续同一轮。"""
    script = [
        {"tool_calls": [{"name": "task_crud", "arguments": {"action": "create", "title": "交周报", "due_date": "2026-09-18"}}]},
        {"content": "已经帮你加好「交周报」，截止 2026-09-18。"},
    ]
    runtime = Runtime.build(settings, llm=StubLLM(script=script))

    first = await collect(runtime, "帮我加个明天交周报的任务")
    done = first[-1][1]
    assert done["status"] == "awaiting_client" and done["pending"]
    pending = done["pending"][0]
    assert pending["name"] == "task_crud" and pending["arguments"]["title"] == "交周报"
    call_event = next(data for kind, data in first if kind == "tool_call")
    assert call_event["executor"] == "client" and call_event["status"] == "awaiting_client"
    # 停在客户端时**不写** assistant 消息（这一轮还没完）
    session = runtime.sessions.get_session(done["session_id"])
    assert [m["role"] for m in session["messages"]] == ["user"]

    frames = [
        frame
        async for frame in runtime.agent.resume(
            done["run_id"],
            [
                {
                    "tool_call_id": pending["id"],
                    "name": "task_crud",
                    "ok": True,
                    "summary": "已创建任务「交周报」（id=t1，截止 2026-09-18）",
                    "result": {"id": "t1"},
                }
            ],
        )
    ]
    resumed = parse_frames(frames)
    resumed_done = resumed[-1][1]
    assert resumed_done["status"] == "ok" and resumed_done["kind"] == "tool"
    # 客户端工具也走统一的 tool_result 事件（前端不用自己拿 done.tools 补一条）
    client_result = next(data for kind, data in resumed if kind == "tool_result")
    assert client_result["ok"] is True and "id=t1" in client_result["summary"]
    message = runtime.sessions.get_session(resumed_done["session_id"])["messages"][-1]
    assert message["role"] == "assistant" and "交周报" in message["content"]
    tools = message["meta"]["tools"]
    assert tools[0]["name"] == "task_crud"
    # 参数与摘要都落进 meta：会话历史上工具标签才画得出来
    assert tools[0]["arguments"]["title"] == "交周报"
    assert "id=t1" in tools[0]["summary"]
    # 前端回传的观察结果确实进了模型上下文
    assert any("id=t1" in str(m.get("content")) for m in runtime.llm.calls[1]["messages"])


async def test_resume_with_unknown_run_id_is_a_clear_error(settings):
    runtime = Runtime.build(settings, llm=StubLLM())
    with pytest.raises(RunNotFound):
        runtime.runs.load("nope")


async def test_client_tool_failure_round_trips_as_observation(settings):
    script = [
        {"tool_calls": [{"name": "get_weather", "arguments": {}}]},
        {"content": "本机还没有天气缓存，去仪表板刷新一下再问我。"},
    ]
    runtime = Runtime.build(settings, llm=StubLLM(script=script))
    first = await collect(runtime, "今天天气怎么样？")
    done = first[-1][1]
    pending = done["pending"][0]
    frames = [
        frame
        async for frame in runtime.agent.resume(
            done["run_id"],
            [{"tool_call_id": pending["id"], "name": "get_weather", "ok": False, "error": "本地没有天气缓存"}],
        )
    ]
    resumed = parse_frames(frames)
    assert resumed[-1][1]["status"] == "ok"
    tool_message = next(m for m in runtime.llm.calls[1]["messages"] if m.get("role") == "tool")
    assert "本地没有天气缓存" in str(tool_message["content"])


# ---------------- 护栏 ----------------


async def test_repeated_tool_failure_trips_the_breaker_and_wraps_up_gracefully(settings):
    """验收项 3：构造一个必失败任务，agent 在轮数上限内优雅收场而非死循环。"""

    async def always_fail(**kwargs) -> ToolOutcome:
        raise RuntimeError("演示用的必失败工具")

    runtime = Runtime.build(settings, llm=StubLLM())
    runtime.tools.register(
        Tool(name="boom", description="必然失败", parameters={"type": "object", "properties": {}}, handler=always_fail)
    )
    # 模型每轮都调同一个必失败工具，一直调到熔断；第 4 次调用是"摘掉工具"后的收尾
    runtime.llm.script = [{"tool_calls": [{"name": "boom", "arguments": {}}]} for _ in range(12)]
    runtime.llm.script.insert(
        settings.agent_max_tool_failures, {"content": "我没能完成任务：演示用的必失败工具一直报错。"}
    )

    events = await collect(runtime, "帮我做点做不到的事")
    done = events[-1][1]
    assert done["status"] == "guardrail" and done["kind"] == "guardrail"
    # 熔断在 3 次，不会真的跑满 8 轮，更不会无限循环
    assert done["rounds"] <= settings.agent_max_tool_failures
    failures = [data for kind, data in events if kind == "tool_result" and not data["ok"]]
    assert len(failures) == settings.agent_max_tool_failures
    assert "我没能完成任务" in "".join(data["text"] for kind, data in events if kind == "token")
    # 收尾那次调用必须把工具摘掉
    assert runtime.llm.calls[-1]["tools"] is None


async def test_max_rounds_guardrail_stops_a_tool_loop(settings):
    async def ok_tool(**kwargs) -> ToolOutcome:
        return ToolOutcome(ok=True, content="好的")

    runtime = Runtime.build(settings, llm=StubLLM())
    runtime.tools.register(
        Tool(name="loop", description="总是被调用", parameters={"type": "object", "properties": {}}, handler=ok_tool)
    )
    runtime.llm.script = [{"tool_calls": [{"name": "loop", "arguments": {}}]} for _ in range(20)]
    runtime.llm.script.insert(settings.agent_max_rounds, {"content": "轮数到上限了，先说结论。"})

    events = await collect(runtime, "一直调用工具")
    done = events[-1][1]
    assert done["status"] == "guardrail"
    assert done["rounds"] == settings.agent_max_rounds
    assert "轮数达到上限" in runtime.llm.calls[-1]["messages"][-1]["content"]


async def test_token_budget_guardrail(settings):
    runtime = Runtime.build(settings, llm=StubLLM(script=[{"tool_calls": [{"name": "get_date", "arguments": {}}], "tokens": 99999}]))
    runtime.llm.script.append({"content": "预算用尽，先收口。"})
    events = await collect(runtime, "现在几点？")
    done = events[-1][1]
    assert done["status"] == "guardrail" and done["tokens"] >= settings.agent_token_budget


async def test_unknown_tool_returns_observation_instead_of_crashing(settings):
    runtime = Runtime.build(settings, llm=StubLLM(script=[
        {"tool_calls": [{"name": "not_a_tool", "arguments": {}}]},
        {"content": "换个方式回答。"},
    ]))
    events = await collect(runtime, "随便问问")
    failure = next(data for kind, data in events if kind == "tool_result")
    assert failure["ok"] is False and "没有名为 not_a_tool 的工具" in failure["error"]
    assert events[-1][1]["status"] == "ok"


async def test_tools_can_be_disabled_entirely(settings):
    runtime = Runtime.build(settings, llm=StubLLM(script=[{"content": "纯对话回答。"}]))
    await collect(runtime, "你好", tools_enabled=False)
    assert runtime.llm.calls[0]["tools"] is None


async def test_llm_failure_surfaces_error_event(settings):
    runtime = Runtime.build(settings, llm=StubLLM(fail="boom"))
    events = await collect(runtime, "问题")
    assert events[0][0] == "error"
    assert events[0][1]["code"] == "llm_error"


async def test_transient_llm_failure_is_retried_once(settings):
    """provider 偶发 5xx/超时不该直接把用户打断：一个字都没吐出来时重试一次。"""
    llm = StubLLM(script=[{"content": "重试之后的答案"}], fail_calls=1)
    runtime = Runtime.build(settings, llm=llm)
    events = await collect(runtime, "问题")
    assert not [data for kind, data in events if kind == "error"]
    assert "重试之后的答案" in "".join(data["text"] for kind, data in events if kind == "token")
    assert events[-1][1]["status"] == "ok"
    assert len(llm.calls) == 2  # 第一次失败、第二次成功


async def test_citations_are_renumbered_across_multiple_searches(settings, sample_md):
    runtime, _ = make_runtime(
        settings,
        [
            {"tool_calls": [{"name": "search_knowledge", "arguments": {"query": "分块默认块长"}}]},
            {"tool_calls": [{"name": "search_knowledge", "arguments": {"query": "余弦距离 阈值"}}]},
            {"content": "见 [1] 与 [2]。"},
        ],
        sample_md,
    )
    events = await collect(runtime, "分块和阈值分别是什么？")
    indices = [data["index"] for kind, data in events if kind == "citation"]
    assert indices == sorted(set(indices)), f"引用编号必须全局唯一递增，实际 {indices}"
    tool_observation = runtime.llm.calls[2]["messages"]
    # 第二轮检索的观察结果里，编号已经被平移到全局（不会又从 [1] 开始）
    assert any("[3]" in str(m.get("content")) or "[2]" in str(m.get("content")) for m in tool_observation)
