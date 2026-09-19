"""ReAct 循环与护栏（规划 11.1）。

    思考 → 选工具 → 执行 → 观察 → 再思考

**护栏三件套**（面试考点，也是真的防失控）：
1. 最大轮数（默认 8）：轮数是"模型又调了一次工具"的次数，不是 LLM 调用次数；
2. token 预算（默认 12000）：用流里的 usage 累计，缺失时按字符估算；
3. 工具超时：注册表里每个工具自带（检索 15s、其余 10s），由 `ToolRegistry.call` 兜住。

外加两条工程护栏，都是被"必然失败"的场景逼出来的：
4. **同一工具连续失败 N 次熔断**：换个参数重试是合理的，同样的失败重试 3 次就是空转；
5. **整轮墙钟超时**：防止"每轮都不超时、但整体跑十分钟"。

撞上任何一条都不是抛错，而是**把工具摘掉再问一次**——让模型用已有信息给一个诚实的收尾
（"我完成了什么、没完成什么、卡在哪"）。用户看到的是一段完整回答，而不是断流或死循环。
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from app import sse
from app.agent.prompts import build_agent_system_prompt
from app.agent.runs import PendingRun, RunStore
from app.agent.tools import ToolOutcome, ToolRegistry
from app.config import Settings
from app.errors import AgentError
from app.rag.generator import LLMClient, TurnResult
from app.sessions import SessionStore

# kind：这条回答是靠什么立起来的
KIND_KB = "kb"
KIND_TOOL = "tool"
KIND_CHAT = "chat"
KIND_GUARDRAIL = "guardrail"

_CITATION_RE = re.compile(r"\[(\d+)\]")


class AgentLoop:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        registry: ToolRegistry,
        sessions: SessionStore,
        runs: RunStore,
    ):
        self._settings = settings
        self._llm = llm
        self._registry = registry
        self._sessions = sessions
        self._runs = runs

    # ---------------- 对外入口 ----------------

    async def stream(
        self,
        question: str,
        *,
        session_id: str | None = None,
        mode: str = "semantic",
        fallback_mode: str | None = None,
        tools_enabled: bool = True,
    ) -> AsyncIterator[str]:
        started = time.perf_counter()
        fallback = (fallback_mode or self._settings.fallback_mode).strip() or "refuse"
        resolved = await asyncio.to_thread(self._sessions.ensure_session, session_id, question, mode)
        await asyncio.to_thread(
            self._sessions.add_message, resolved, "user", question, meta={"mode": mode}
        )
        run = PendingRun(
            run_id=RunStore.new_id(),
            session_id=resolved,
            question=question,
            messages=self._initial_messages(resolved, question, fallback),
            mode=mode,
            fallback_mode=fallback,
        )
        async for frame in self._drive(run, started, tools_enabled=tools_enabled):
            yield frame

    async def resume(self, run_id: str, results: list[dict[str, Any]]) -> AsyncIterator[str]:
        """前端执行完 client 工具后回来续跑。"""
        started = time.perf_counter()
        run = await asyncio.to_thread(self._runs.load, run_id)
        await asyncio.to_thread(self._runs.delete, run_id)  # 旧状态作废（续跑可能再次落盘）

        for item in results:
            call = next((c for c in run.pending if c["id"] == item["tool_call_id"]), None)
            name = str(item.get("name") or (call or {}).get("name") or "unknown")
            ok = bool(item.get("ok"))
            summary = str(item.get("summary") or item.get("result") or "")
            error = str(item.get("error") or "")
            observation = summary if ok else f"工具执行失败：{error or '未知错误'}"
            run.messages.append(
                {"role": "tool", "tool_call_id": item["tool_call_id"], "content": observation or "（无返回内容）"}
            )
            run.tools.append({"id": item["tool_call_id"], "name": name, "ok": ok, "executor": "client", "error": error})
            if ok:
                run.failures[name] = 0
            else:
                run.failures[name] = run.failures.get(name, 0) + 1
        run.pending = []

        async for frame in self._drive(run, started, tools_enabled=True):
            yield frame

    # ---------------- 主循环 ----------------

    async def _drive(self, run: PendingRun, started: float, *, tools_enabled: bool) -> AsyncIterator[str]:
        while True:
            reason = self._guardrail_reason(run, started)
            if reason:
                async for frame in self._wrap_up(run, reason, started):
                    yield frame
                return

            turn = TurnResult()
            try:
                specs = self._registry.specs() if tools_enabled else None
                async for piece in self._llm.stream_with_tools(run.messages, tools=specs, result=turn):
                    run.content += piece
                    yield sse.token_event(piece)
            except AgentError as exc:
                yield sse.error_event(exc.message, exc.code)
                return
            except Exception as exc:  # noqa: BLE001 - 流一旦开始就不能抛
                yield sse.error_event(f"生成失败：{exc}", "internal_error")
                return

            run.tokens += turn.total_tokens or _estimate_tokens(run.messages, run.content)

            if not turn.tool_calls:
                async for frame in self._finish(run, started, status="ok"):
                    yield frame
                return

            run.rounds += 1
            run.messages.extend(turn.as_tool_call_messages())
            client_calls: list[dict[str, Any]] = []

            for call in turn.tool_calls:
                tool = self._registry.get(call.name)
                arguments = call.parsed_arguments()

                if tool is None:
                    observation = f"没有名为 {call.name} 的工具，请改用可用工具或直接回答。"
                    run.tools.append({"id": call.id, "name": call.name, "ok": False, "error": observation})
                    run.messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})
                    yield sse.tool_result_event(call.id, call.name, False, error=observation)
                    continue

                if tool.executor == "client":
                    client_calls.append(
                        {"id": call.id, "name": call.name, "arguments": arguments, "executor": "client"}
                    )
                    continue

                yield sse.tool_call_event(call.id, call.name, arguments, executor="server", status="running")
                outcome = await self._registry.call(call.name, arguments)
                offset = len(run.citations)
                citations = _renumber(outcome.citations, offset)
                run.citations.extend(citations)
                run.tools.append(
                    {
                        "id": call.id,
                        "name": call.name,
                        "ok": outcome.ok,
                        "executor": "server",
                        "arguments": arguments,
                        "error": outcome.error,
                        "meta": outcome.meta,
                    }
                )
                if outcome.ok:
                    run.failures[call.name] = 0
                else:
                    run.failures[call.name] = run.failures.get(call.name, 0) + 1
                run.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": _renumber_text(outcome.as_observation(), offset),
                    }
                )
                yield sse.tool_result_event(
                    call.id,
                    call.name,
                    outcome.ok,
                    summary=_summarize(outcome),
                    error=outcome.error,
                    citations=citations,
                    meta=outcome.meta,
                )
                # 命中即发 citation 事件：前端在答案还没写完时就能把引用块画出来，
                # 而且与第十阶段的 RAG 直答走**同一套**渲染路径
                for citation in citations:
                    yield sse.citation_event(citation)

            if client_calls:
                # 停在这里等前端执行——把状态存下来，前端 resume 时接着跑
                run.pending = client_calls
                for item in client_calls:
                    yield sse.tool_call_event(
                        item["id"], item["name"], item["arguments"], executor="client", status="awaiting_client"
                    )
                await asyncio.to_thread(self._runs.save, run)
                async for frame in self._done_frames(run, started, status="awaiting_client"):
                    yield frame
                return

    # ---------------- 收口 ----------------

    async def _wrap_up(self, run: PendingRun, reason: str, started: float) -> AsyncIterator[str]:
        """撞护栏后的优雅收场：摘掉工具，让模型用已有信息给一个诚实的收尾。"""
        run.messages.append(
            {
                "role": "system",
                "content": (
                    f"系统提示：{reason}，因此本轮到此为止。**不要再调用任何工具**，"
                    "直接用已有信息给用户一个诚实的收尾：说明你完成了什么、没完成什么、卡在哪里。"
                ),
            }
        )
        turn = TurnResult()
        try:
            async for piece in self._llm.stream_with_tools(run.messages, tools=None, result=turn):
                run.content += piece
                yield sse.token_event(piece)
        except Exception:  # noqa: BLE001 - 收尾也不能把流炸掉
            notice = f"\n\n（{reason}，我先停在这里。可以缩小范围再试一次。）"
            run.content += notice
            yield sse.token_event(notice)
        run.kinds.append(KIND_GUARDRAIL)
        async for frame in self._finish(run, started, status="guardrail"):
            yield frame

    async def _finish(self, run: PendingRun, started: float, *, status: str) -> AsyncIterator[str]:
        kind = _classify(run)
        fallback = _fallback_of(run, kind)
        if kind == KIND_CHAT:
            # 闲聊那一轮**连用户那句一起**打标：只标回答的话，用户那半句下一轮还是会被回填
            await asyncio.to_thread(self._sessions.tag_last_user_message, run.session_id, KIND_CHAT)
        message = await asyncio.to_thread(
            self._sessions.add_message,
            run.session_id,
            "assistant",
            run.content,
            citations=run.citations,
            meta={
                "kind": kind,
                "fallback": fallback,
                "mode": run.mode,
                "rounds": run.rounds,
                "tokens": run.tokens,
                "tools": run.tools,
                "status": status,
            },
        )
        async for frame in self._done_frames(run, started, status=status, message_id=message["id"], kind=kind, fallback=fallback):
            yield frame

    async def _done_frames(
        self,
        run: PendingRun,
        started: float,
        *,
        status: str,
        message_id: str = "",
        kind: str = "",
        fallback: str = "",
    ) -> AsyncIterator[str]:
        yield sse.done_event(
            session_id=run.session_id,
            run_id=run.run_id,
            message_id=message_id,
            citations=run.citations,
            fallback=fallback or _fallback_of(run, kind or _classify(run)),
            kind=kind or _classify(run),
            hit_count=len(run.citations),
            latency_ms=int((time.perf_counter() - started) * 1000),
            status=status,
            rounds=run.rounds,
            tokens=run.tokens,
            tools=run.tools,
            pending=run.pending,
        )

    # ---------------- 护栏与上下文 ----------------

    def _guardrail_reason(self, run: PendingRun, started: float) -> str:
        if run.rounds >= self._settings.agent_max_rounds:
            return f"工具调用轮数达到上限（{self._settings.agent_max_rounds} 轮）"
        if run.tokens >= self._settings.agent_token_budget:
            return f"token 预算用尽（约 {run.tokens}）"
        if time.perf_counter() - started >= self._settings.agent_run_timeout_s:
            return f"整轮耗时超过 {self._settings.agent_run_timeout_s:g} 秒"
        for name, count in run.failures.items():
            if count >= self._settings.agent_max_tool_failures:
                return f"工具 {name} 连续失败 {count} 次"
        return ""

    def _initial_messages(self, session_id: str, question: str, fallback: str) -> list[dict[str, Any]]:
        history = self._history(session_id)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_agent_system_prompt(fallback, has_history=bool(history))}
        ]
        messages.extend(history)
        messages.append({"role": "user", "content": question})
        return messages

    def _history(self, session_id: str) -> list[dict[str, str]]:
        """回填最近几条历史——**剔除轻量闲聊**（11.4：闲聊不进任务上下文）。"""
        limit = self._settings.agent_history_messages
        if limit <= 0:
            return []
        detail = self._sessions.get_session(session_id)
        rows = [m for m in detail["messages"] if (m.get("meta") or {}).get("kind") != KIND_CHAT]
        # 去掉最后一条（就是本轮刚写进去的用户消息）
        if rows and rows[-1]["role"] == "user":
            rows = rows[:-1]
        rows = rows[-limit:]
        return [
            {"role": str(m["role"]), "content": str(m["content"])}
            for m in rows
            if str(m.get("content", "")).strip()
        ]


def _classify(run: PendingRun) -> str:
    if KIND_GUARDRAIL in run.kinds:
        return KIND_GUARDRAIL
    names = {tool["name"] for tool in run.tools}
    if "search_knowledge" in names and run.citations:
        return KIND_KB
    if names:
        return KIND_TOOL
    return KIND_CHAT


def _fallback_of(run: PendingRun, kind: str) -> str:
    """给前端看的"依据类型"。

    知识库被调用过但没命中 → 用配置的兜底模式（refuse/bare/web），
    这样界面上会如实显示"知识库无相关内容"，而不是假装这只是一次普通工具调用。
    """
    names = {tool["name"] for tool in run.tools}
    if kind == KIND_GUARDRAIL:
        return KIND_GUARDRAIL
    if "search_knowledge" in names:
        return KIND_KB if run.citations else run.fallback_mode
    if names:
        return KIND_TOOL
    return KIND_CHAT


def _renumber(citations: list[dict[str, Any]], offset: int) -> list[dict[str, Any]]:
    if not offset:
        return citations
    return [{**citation, "index": int(citation.get("index", 0)) + offset} for citation in citations]


def _renumber_text(text: str, offset: int) -> str:
    """把观察结果里的 [n] 平移到全局编号（多次检索时编号才不会串台）。"""
    if not offset:
        return text
    return _CITATION_RE.sub(lambda match: f"[{int(match.group(1)) + offset}]", text)


def _summarize(outcome: ToolOutcome) -> str:
    if not outcome.ok:
        return ""
    if outcome.meta.get("hits") is not None:
        return f"命中 {outcome.meta['hits']} 块"
    return outcome.content[:120]


def _estimate_tokens(messages: list[dict[str, Any]], content: str) -> int:
    """usage 缺失时的兜底估算：中文大致 1 字 ≈ 1 token，这里按字符数/2 保守估。"""
    size = sum(len(str(m.get("content") or "")) for m in messages) + len(content)
    return size // 2
