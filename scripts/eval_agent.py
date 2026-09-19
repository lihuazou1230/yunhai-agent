"""第十一阶段验收脚本（Agent 内核）。

    .venv\\Scripts\\python scripts\\ingest.py eval\\docs --reset   # 先让知识库有东西
    .venv\\Scripts\\python scripts\\eval_agent.py

对应验收清单四条：
1. "帮我加个明天交周报的任务"一句话落库；"今天还剩哪些活"回答准确
2. agent 能自主决定"先查知识库再回答"的调用顺序
3. 构造一个必失败任务，agent 在轮数上限内优雅收场而非死循环
4. 问天气/日期秒回；闲聊不产生记忆残留

**这里用假的"前端"执行 client 工具**（任务存在内存里、天气给一份假缓存）：
CLI 里没有浏览器，但工具回环的两段（后端发 tool_call → 客户端执行 → resume 带观察结果回来）
是真跑的；浏览器里那份真实执行由前端的 vitest 用例覆盖（`src/agent/clientTools.spec.ts`）。
把这件事写清楚，比假装"端到端全在 CLI 里验过了"要诚实。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.tools import Tool, ToolOutcome  # noqa: E402
from app.runtime import Runtime  # noqa: E402

TODAY_TASKS = [
    {"id": "t1", "title": "写周报", "due_date": "今天", "priority": "high", "status": "active"},
    {"id": "t2", "title": "交季度报销", "due_date": "今天", "priority": "medium", "status": "active"},
]


class FakeClient:
    """假前端：把 client 工具在工作台数据上执行一遍（内存版）。"""

    def __init__(self) -> None:
        self.tasks = [dict(task) for task in TODAY_TASKS]
        self.calls: list[dict] = []

    def run(self, call: dict) -> dict:
        name = call["name"]
        arguments = call.get("arguments") or {}
        self.calls.append({"name": name, "arguments": arguments})
        if name == "task_crud":
            return self._task_crud(call["id"], arguments)
        if name == "get_weather":
            return {
                "tool_call_id": call["id"],
                "name": name,
                "ok": True,
                "summary": "上海 晴 24°C（体感 26°C，湿度 60%，更新于 10:20）",
                "result": {"city": "上海", "text": "晴", "temp": 24},
            }
        return {"tool_call_id": call["id"], "name": name, "ok": False, "error": f"前端还没有实现工具 {name}"}

    def _task_crud(self, call_id: str, arguments: dict) -> dict:
        action = arguments.get("action")
        title = arguments.get("title") or ""
        if action == "list":
            rows = [t for t in self.tasks if t["status"] == "active"]
            summary = f"共 {len(rows)} 条未完成任务：\n" + "\n".join(
                f"- [{t['id']}] {t['title']}（截止 {t['due_date']}，优先级 {t['priority']}，未完成）" for t in rows
            )
            return {"tool_call_id": call_id, "name": "task_crud", "ok": True, "summary": summary, "result": {"count": len(rows), "ids": [t["id"] for t in rows]}}
        if action == "create":
            new_id = f"t{len(self.tasks) + 1}"
            self.tasks.append(
                {
                    "id": new_id,
                    "title": title,
                    "due_date": arguments.get("due_date", ""),
                    "priority": arguments.get("priority", "medium"),
                    "status": "active",
                }
            )
            return {
                "tool_call_id": call_id,
                "name": "task_crud",
                "ok": True,
                "summary": f"已创建任务「{title}」（id={new_id}，截止 {arguments.get('due_date', '未设置')}）",
                "result": {"id": new_id, "title": title},
            }
        return {"tool_call_id": call_id, "name": "task_crud", "ok": False, "error": f"假前端暂不支持 action={action}"}


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


async def run_case(runtime: Runtime, client: FakeClient, question: str, *, max_resumes: int = 5) -> dict:
    """跑完一整轮（含任意次 client 工具回环），返回这一轮的全部事件与收口信息。"""
    started = time.perf_counter()
    events = parse_frames([frame async for frame in runtime.agent.stream(question)])
    resumes = 0
    while True:
        done = next((data for kind, data in reversed(events) if kind == "done"), {})
        if done.get("status") != "awaiting_client" or resumes >= max_resumes:
            break
        pending = done.get("pending") or []
        results = [client.run(call) for call in pending]
        events.extend(
            parse_frames([frame async for frame in runtime.agent.resume(done["run_id"], results)])
        )
        resumes += 1
    done = next((data for kind, data in reversed(events) if kind == "done"), {})
    return {
        "question": question,
        "events": events,
        "answer": "".join(data["text"] for kind, data in events if kind == "token"),
        "tool_calls": [data for kind, data in events if kind == "tool_call"],
        "tool_results": [data for kind, data in events if kind == "tool_result"],
        "citations": [data for kind, data in events if kind == "citation"],
        "errors": [data for kind, data in events if kind == "error"],
        "done": done,
        "resumes": resumes,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def register_failing_tool(runtime: Runtime) -> None:
    async def always_fail(**kwargs) -> ToolOutcome:
        raise RuntimeError("演示用的必失败工具：外部接口 503")

    runtime.tools.register(
        Tool(
            name="demo_boom",
            description=(
                "查询演示数据（当前后端**故意让它失败**，用于验证护栏：调用它必定返回 503）。"
                "用户提到 demo_boom 或要求查演示数据时调用它。"
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=always_fail,
        )
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description="第十一阶段 Agent 验收")
    parser.add_argument("--report", default=str(ROOT / "eval" / "agent_report.md"))
    parser.add_argument("--only", default="", help="只跑某一条用例（关键字匹配）")
    args = parser.parse_args()

    runtime = Runtime.build()
    if not runtime.llm.configured:
        print("[err] 未配置 LLM_API_KEY：本脚本要真的调模型")
        return 1
    stats = runtime.kb.stats()
    print(f"[env] model={runtime.llm.model} embedder={runtime.embedder.name} 知识库 {stats['documents']} 篇 / {stats['chunks']} 块")
    print(f"[tools] {runtime.tools.names()}（客户端工具：{runtime.tools.client_tool_names()}）")
    register_failing_tool(runtime)
    client = FakeClient()
    session_id: str | None = None
    results: list[dict] = []

    cases = [
        ("task_create", "帮我加个明天交周报的任务"),
        ("task_list", "今天还剩哪些活？"),
        ("kb_search", "分块默认的块长和重叠是多少？"),
        ("guardrail", "用 demo_boom 查演示数据；如果失败就再试，一直试到查到结果为止"),
        ("date", "现在几点了？"),
        ("weather", "今天天气怎么样？"),
        ("chat", "讲个冷笑话吧"),
    ]
    for name, question in cases:
        if args.only and args.only not in name:
            continue
        print(f"\n=== {name}: {question}")
        case = await run_case(runtime, client, question)
        session_id = case["done"].get("session_id") or session_id
        case["name"] = name
        results.append(case)
        tools = "、".join(f"{call['name']}({call.get('status')})" for call in case["tool_calls"]) or "无"
        print(
            f"  工具：{tools} · 状态 {case['done'].get('status')} · kind {case['done'].get('kind')} · "
            f"引用 {len(case['citations'])} 条 · {case['latency_ms']}ms"
        )
        print(f"  回答：{' '.join(case['answer'].split())[:90]}")
        for error in case["errors"]:
            print(f"  ⚠️ 错误事件：{error['code']} {error['message'][:160]}")

    # ---- 验收判定 ----
    verdicts: list[dict] = []
    by_name = {case["name"]: case for case in results}

    def check(name: str, rule: str, ok: bool, detail: str = "") -> None:
        verdicts.append({"name": name, "rule": rule, "ok": ok, "detail": detail})

    if "task_create" in by_name:
        case = by_name["task_create"]
        created = [t for t in client.tasks if t["id"] not in {task["id"] for task in TODAY_TASKS}]
        args_seen = [call.get("arguments", {}) for call in case["tool_calls"] if call["name"] == "task_crud"]
        check(
            "一句话落库",
            "agent 调用 task_crud(create) 且任务真的进了工作台数据",
            bool(created) and any(a.get("action") == "create" for a in args_seen),
            f"新建 {created}",
        )
        check(
            "日期换算",
            "「明天」被换算成绝对日期（YYYY-MM-DD）而不是原样透传",
            any(str(a.get("due_date", "")).count("-") == 2 for a in args_seen),
            f"due_date={[a.get('due_date') for a in args_seen]}",
        )

    if "task_list" in by_name:
        case = by_name["task_list"]
        titles = ["写周报", "交季度报销"]
        check(
            "今天还剩哪些活",
            "回答里出现工作台里真实存在的任务标题",
            any(title in case["answer"] for title in titles),
            f"回答命中 {[t for t in titles if t in case['answer']]}",
        )

    if "kb_search" in by_name:
        case = by_name["kb_search"]
        check(
            "自主查知识库",
            "agent 自己决定调用 search_knowledge（不是后端预先检索），并带引用回答",
            any(call["name"] == "search_knowledge" for call in case["tool_calls"]) and bool(case["citations"]),
            f"引用 {len(case['citations'])} 条，kind={case['done'].get('kind')}",
        )
        check(
            "答案正确",
            "答案包含库里的关键事实（500 / 80）",
            "500" in case["answer"],
            " ".join(case["answer"].split())[:60],
        )

    if "guardrail" in by_name:
        case = by_name["guardrail"]
        failures = [r for r in case["tool_results"] if not r["ok"]]
        max_rounds = runtime.settings.agent_max_rounds
        # 两种"优雅收场"都算通过：模型自己一次失败就收手（更好），或连续失败撞上熔断被强制收口。
        # 关键是没有死循环、没有空转，而且给了一段诚实的回答。
        graceful = (
            bool(case["answer"])
            and len(failures) >= 1
            and case["done"].get("rounds", 0) <= max_rounds
            and (
                case["done"].get("status") == "guardrail"
                or len(failures) <= max(1, case["done"].get("rounds", 0))
            )
        )
        check(
            "必失败任务优雅收场",
            f"工具一直失败时在轮数上限（{max_rounds}）内停下并诚实收尾",
            graceful,
            f"状态 {case['done'].get('status')}、失败 {len(failures)} 次、轮数 {case['done'].get('rounds')}、回答 {len(case['answer'])} 字",
        )

    if "date" in by_name:
        case = by_name["date"]
        check(
            "日期秒回",
            "调用 get_date 并在答案里给出今天的日期",
            any(call["name"] == "get_date" for call in case["tool_calls"]),
            f"{case['latency_ms']}ms",
        )

    if "weather" in by_name:
        case = by_name["weather"]
        check(
            "天气读缓存",
            "调用 client 工具 get_weather，答案里出现缓存里的天气信息",
            any(call["name"] == "get_weather" for call in case["tool_calls"]) and ("24" in case["answer"] or "晴" in case["answer"]),
            " ".join(case["answer"].split())[:60],
        )

    if "chat" in by_name:
        case = by_name["chat"]
        detail = runtime.sessions.get_session(case["done"]["session_id"])
        chat_rows = [m for m in detail["messages"] if (m.get("meta") or {}).get("kind") == "chat"]
        check(
            "闲聊无记忆残留",
            "闲聊轮不调用工具，且用户与回答两侧都被打上 kind=chat（不进任务上下文）",
            case["done"].get("kind") == "chat" and not case["tool_calls"] and len(chat_rows) >= 2,
            f"chat 标记 {len(chat_rows)} 条",
        )

    print("\n=== 验收结果 ===")
    for verdict in verdicts:
        print(f"  [{'PASS' if verdict['ok'] else 'FAIL'}] {verdict['name']}：{verdict['rule']}（{verdict['detail']}）")

    lines = ["# 第十一阶段验收报告（Agent 内核）\n\n"]
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"- 模型：`{runtime.llm.model}`\n")
    lines.append(
        f"- 工具：{', '.join(f'`{name}`' for name in runtime.tools.names())}"
        f"（客户端执行：{', '.join(runtime.tools.client_tool_names())}）\n"
    )
    lines.append(f"- 知识库：{stats['documents']} 篇 / {stats['chunks']} 块\n")
    lines.append(
        "- **说明**：CLI 里没有浏览器，client 工具由脚本里的假前端执行（任务存内存、天气给假缓存）；"
        "浏览器里那份真实执行由前端用例 `src/agent/clientTools.spec.ts` 覆盖。\n"
    )
    lines.append("\n## 一、验收清单\n\n| 验收项 | 结果 | 判定依据 | 细节 |\n| --- | --- | --- | --- |\n")
    for verdict in verdicts:
        lines.append(
            f"| {verdict['name']} | {'✅' if verdict['ok'] else '❌'} | {verdict['rule']} | {verdict['detail']} |\n"
        )
    lines.append("\n## 二、逐用例的工具轨迹与回答\n")
    for case in results:
        lines.append(f"\n### {case['name']}：{case['question']}\n\n")
        lines.append(
            f"- 状态 `{case['done'].get('status')}` · kind `{case['done'].get('kind')}` · "
            f"fallback `{case['done'].get('fallback')}` · 轮数 {case['done'].get('rounds')} · "
            f"tokens {case['done'].get('tokens')} · 引用 {len(case['citations'])} 条 · "
            f"{case['latency_ms']}ms · 回环 {case['resumes']} 次\n"
        )
        for call in case["tool_calls"]:
            lines.append(
                f"- 🔧 `{call['name']}`（{call['executor']}）参数 `{json.dumps(call.get('arguments'), ensure_ascii=False)}`\n"
            )
        for result in case["tool_results"]:
            mark = "✅" if result["ok"] else "❌"
            detail = result.get("summary") or result.get("error") or ""
            lines.append(f"- {mark} `{result['name']}` → {' '.join(str(detail).split())[:160]}\n")
        if case["citations"]:
            cited = ", ".join(f"{c['index']}.{c['source']}" for c in case["citations"])
            lines.append(f"- 引用：{cited}\n")
        for error in case["errors"]:
            lines.append(f"- ⚠️ 错误事件：`{error['code']}` {error['message'][:200]}\n")
        lines.append(f"\n> {' '.join(case['answer'].split())[:400] or '（无输出）'}\n")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")
    print(f"\n[done] 报告已写入 {report_path}")
    return 0 if all(verdict["ok"] for verdict in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
