"""分块器测试：中文边界、重叠、硬切、块序号。"""

from __future__ import annotations

from app.rag.chunker import build_chunks, split_text
from app.rag.types import LoadedItem


def test_short_text_is_single_chunk():
    assert split_text("很短的一句话。", 100, 10) == ["很短的一句话。"]


def test_blank_text_yields_nothing():
    assert split_text("   \n\n  ", 100, 10) == []


def test_splits_on_chinese_sentence_end_and_keeps_punctuation():
    text = "第一句话说的是甲。" * 12
    chunks = split_text(text, 40, 0)
    assert len(chunks) > 1
    # 标点跟随前一块：不能出现以"。"开头的块
    assert all(not chunk.startswith("。") for chunk in chunks)
    assert all(chunk.endswith("。") for chunk in chunks)


def test_overlap_prefixes_next_chunk():
    text = "".join(f"第{i}句内容。甲" for i in range(1, 30))
    chunks = split_text(text, 60, 20)
    assert len(chunks) >= 3
    tail = chunks[0][-10:]
    assert tail in chunks[1]  # 重叠区真的带过去了


def test_hard_split_when_no_separator():
    text = "甲" * 250
    chunks = split_text(text, 100, 0)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_overlap_is_capped_below_chunk_size():
    """重叠 >= 块长会把切分变成死循环，必须夹住；重叠也不能把块撑过上限。"""
    text = "。".join("甲" * 30 for _ in range(20))
    chunks = split_text(text, 50, 999)
    assert len(chunks) < 200
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_build_chunks_keeps_global_index_and_page():
    items = [
        LoadedItem(item_id="p1", text="第一页内容。" * 20, meta={"page": 1}),
        LoadedItem(item_id="p2", text="第二页内容。" * 20, meta={"page": 2}),
    ]
    chunks = build_chunks(
        items,
        source="手册.pdf",
        source_type="pdf",
        doc_id="abc123",
        uploaded_at="2026-09-17T10:00:00",
        chunk_size=40,
        chunk_overlap=0,
    )
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id == "abc123:0"
    assert chunks[0].page == 1
    assert chunks[-1].page == 2
    assert all(c.source == "手册.pdf" for c in chunks)


def test_invalid_chunk_size_raises():
    import pytest

    with pytest.raises(ValueError):
        split_text("甲" * 10, 0, 0)
