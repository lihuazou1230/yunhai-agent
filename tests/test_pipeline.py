"""入库与问答主流程测试（含三种兜底模式）。"""

from __future__ import annotations

import json

import pytest

from app.errors import EmptyDocument, UnsupportedFileType
from app.rag.generator import LLMClient
from app.rag.pipeline import REFUSAL_TEXT
from app.runtime import Runtime


def parse_frames(frames: list[str]) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for frame in frames:
        event = ""
        data = "{}"
        for line in frame.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        events.append((event, json.loads(data)))
    return events


async def collect(runtime: Runtime, question: str, **kwargs) -> list[tuple[str, dict]]:
    frames = [frame async for frame in runtime.ask.stream(question, **kwargs)]
    return parse_frames(frames)


def text_of(events: list[tuple[str, dict]]) -> str:
    return "".join(data["text"] for event, data in events if event == "token")


# ---------------- 入库 ----------------


def test_ingest_reports_items_and_chunks(runtime: Runtime, sample_md: str):
    result = runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    assert result.skipped is False
    assert result.items == 1
    assert result.chunks >= 2
    assert result.doc_id
    assert runtime.store.count() == result.chunks
    assert runtime.lexical.size == result.chunks
    assert (runtime.settings.uploads_dir / f"{result.doc_id}.md").exists()


def test_reingest_same_content_is_skipped(runtime: Runtime, sample_md: str):
    first = runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    second = runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    assert second.skipped is True
    assert runtime.store.count() == first.chunks  # 一块向量都没新增
    assert runtime.lexical.size == first.chunks


def test_reingest_changed_content_replaces_source(runtime: Runtime, sample_md: str):
    runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    changed = sample_md + "\n\n## 新增章节\n\n补充了一条新规则：知识库单次问答不留多轮上下文。\n"
    result = runtime.kb.ingest("手册.md", changed.encode("utf-8"))
    assert result.skipped is False
    documents = runtime.kb.documents()
    assert len(documents) == 1  # 同名来源不会新旧并存
    assert runtime.store.count() == result.chunks
    assert runtime.lexical.size == result.chunks


def test_force_reingest_bypasses_hash_check(runtime: Runtime, sample_md: str):
    runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    again = runtime.kb.ingest("手册.md", sample_md.encode("utf-8"), force=True)
    assert again.skipped is False


def test_delete_document_clears_vectors_and_lexical(runtime: Runtime, sample_md: str):
    result = runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    runtime.kb.delete_document(result.doc_id)
    assert runtime.store.count() == 0
    assert runtime.lexical.size == 0
    assert runtime.kb.documents() == []
    assert not (runtime.settings.uploads_dir / f"{result.doc_id}.md").exists()


def test_reset_clears_everything(runtime: Runtime, sample_md: str):
    runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    runtime.kb.reset()
    assert runtime.store.count() == 0
    assert runtime.lexical.size == 0
    assert runtime.kb.stats()["documents"] == 0


def test_unsupported_and_empty_are_rejected(runtime: Runtime):
    with pytest.raises(UnsupportedFileType):
        runtime.kb.ingest("图.png", b"\x89PNG")
    with pytest.raises(EmptyDocument):
        runtime.kb.ingest("空.md", b"   \n")


def test_lexical_index_rebuilds_from_vector_store(runtime: Runtime, sample_md: str):
    runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    runtime.lexical.clear()
    rebuilt = Runtime.build(runtime.settings, llm=runtime.llm)
    assert rebuilt.lexical.size == rebuilt.store.count() > 0


# ---------------- 问答：命中路径 ----------------


async def test_ask_with_hit_streams_citation_then_tokens(
    runtime: Runtime, stub_llm, sample_md: str
):
    runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    events = await collect(runtime, "分块默认块长是多少？")

    assert events[0][0] == "citation"
    assert events[0][1]["source"] == "手册.md"
    assert events[0][1]["index"] == 1
    assert text_of(events) == "分块默认 500 字符。[1]"

    kind, done = events[-1]
    assert kind == "done"
    assert done["fallback"] == "kb"
    assert done["hit_count"] >= 1
    assert done["citations"]

    # 会话与消息都落库了
    detail = runtime.sessions.get_session(done["session_id"])
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["citations"][0]["source"] == "手册.md"
    assert detail["messages"][1]["meta"]["fallback"] == "kb"

    # 送进模型的上下文里带着检索到的原文（StubLLM 现在记录的是 {messages, tools} 结构）
    assert "分块默认块长" in stub_llm.calls[0]["messages"][1]["content"]


async def test_ask_reuses_given_session(runtime: Runtime, sample_md: str):
    runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    session = runtime.sessions.create_session("老会话")
    events = await collect(runtime, "分块默认块长是多少？", session_id=session["id"])
    assert events[-1][1]["session_id"] == session["id"]
    assert len(runtime.sessions.list_sessions()) == 1


# ---------------- 问答：兜底三模式 ----------------


async def test_fallback_refuse_does_not_call_llm(runtime: Runtime, stub_llm, sample_md: str):
    runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    events = await collect(runtime, "世界杯冠军是哪支球队？", fallback_mode="refuse")
    assert text_of(events) == REFUSAL_TEXT
    assert events[-1][1]["fallback"] == "refuse"
    assert stub_llm.calls == []  # 拒答不花一分钱额度


async def test_fallback_bare_calls_llm_with_warning(runtime: Runtime, stub_llm, sample_md: str):
    runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    events = await collect(runtime, "世界杯冠军是哪支球队？", fallback_mode="bare")
    assert events[-1][1]["fallback"] == "bare"
    assert "不基于知识库" in stub_llm.calls[0]["messages"][0]["content"]
    assert text_of(events)


async def test_fallback_web_reports_not_implemented(runtime: Runtime, sample_md: str):
    runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    events = await collect(runtime, "世界杯冠军是哪支球队？", fallback_mode="web")
    kinds = [kind for kind, _ in events]
    assert "error" in kinds
    error = next(data for kind, data in events if kind == "error")
    assert error["code"] == "web_search_unavailable"
    assert events[-1][1]["fallback"] == "web"


async def test_invalid_fallback_mode_falls_back_to_refuse(runtime: Runtime, sample_md: str):
    runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    events = await collect(runtime, "世界杯冠军是哪支球队？", fallback_mode="乱写的")
    assert events[-1][1]["fallback"] == "refuse"


async def test_empty_knowledge_base_refuses(runtime: Runtime):
    events = await collect(runtime, "分块默认块长是多少？")
    assert events[-1][1]["fallback"] == "refuse"
    assert events[-1][1]["hit_count"] == 0


# ---------------- 问答：异常路径 ----------------


async def test_llm_failure_surfaces_error_event(runtime: Runtime, stub_llm, sample_md: str):
    runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    stub_llm.fail = "boom"
    events = await collect(runtime, "分块默认块长是多少？")
    error = next(data for kind, data in events if kind == "error")
    assert error["code"] == "llm_error"
    assert "stub 生成失败" in error["message"]


async def test_unconfigured_llm_returns_config_hint(settings, sample_md: str):
    settings.llm_api_key = ""
    unconfigured = Runtime.build(settings, llm=LLMClient(settings))
    unconfigured.kb.ingest("手册.md", sample_md.encode("utf-8"))
    events = await collect(unconfigured, "分块默认块长是多少？")
    error = next(data for kind, data in events if kind == "error")
    assert error["code"] == "llm_not_configured"
    assert "LLM_API_KEY" in error["message"]
