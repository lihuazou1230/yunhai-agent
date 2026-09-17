"""Prompt 组装与 SSE 协议测试。"""

from __future__ import annotations

import json

from app import sse
from app.rag.prompt import (
    BARE_HINT,
    SYSTEM_PROMPT,
    build_context,
    build_messages,
    citations_of,
)
from app.rag.types import Chunk, Retrieved


def make_hit(text: str, *, index: int = 0, page: int | None = None, score: float = 0.9) -> Retrieved:
    return Retrieved(
        chunk=Chunk(
            chunk_id=f"d1:{index}",
            text=text,
            source="手册.md",
            source_type="md",
            doc_id="d1",
            item_id="raw",
            chunk_index=index,
            uploaded_at="2026-09-17T10:00:00",
            page=page,
        ),
        score=score,
    )


def test_context_has_numbered_blocks_and_location():
    context = build_context([make_hit("分块 500 字符"), make_hit("阈值 0.45", index=1, page=3)])
    assert "[1] 来源：手册.md（第 1 块）" in context
    assert "[2] 来源：手册.md（第 3 页）" in context


def test_messages_carry_context_and_question():
    messages = build_messages("分块多大？", [make_hit("分块 500 字符")])
    assert messages[0]["role"] == "system"
    assert "只依据" in messages[0]["content"] or "只能依据" in messages[0]["content"]
    assert "分块 500 字符" in messages[1]["content"]
    assert "分块多大？" in messages[1]["content"]


def test_bare_mode_adds_the_not_based_on_kb_hint():
    messages = build_messages("无关问题", [], bare=True)
    assert BARE_HINT in messages[0]["content"]
    assert "（无）" in messages[1]["content"]


def test_system_prompt_forbids_fabrication():
    assert "不要编造" in SYSTEM_PROMPT
    assert "知识库里没有相关内容" in SYSTEM_PROMPT


def test_citations_are_numbered_and_snippet_is_trimmed():
    citations = citations_of([make_hit("甲" * 300)])
    assert citations[0]["index"] == 1
    assert len(citations[0]["snippet"]) == 160


def test_sse_frame_format_keeps_chinese_readable():
    frame = sse.token_event("你好")
    assert frame.startswith("event: token\ndata: ")
    assert frame.endswith("\n\n")
    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload["text"] == "你好"
    assert "\\u4f60" not in frame  # 不转义成 \\uXXXX，省带宽也方便调试


def test_error_event_carries_code_and_message():
    frame = sse.error_event("后端未配置 Key", "llm_not_configured")
    body = json.loads(frame.split("data: ", 1)[1].strip())
    assert body["code"] == "llm_not_configured"
    assert body["message"] == "后端未配置 Key"
