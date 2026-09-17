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


def test_markdown_sections_are_not_merged():
    """小节是硬边界：三个短小节不该被合并成一块（第十阶段答案验收抓到的坑）。

    合并后语义被平均掉，"深色模式下图表发白是什么原因"这类**原文就在库里**的问题
    会因为分数差 0.005 被阈值拒答。
    """
    text = (
        "# 手册\n\n"
        "## 一、AI 边界\n\nKey 放在后端保管，前端只拿一个基地址。\n\n"
        "## 二、快捷键\n\nCtrl+K 打开搜索，Ctrl+B 折叠侧边栏。\n\n"
        "## 三、故障排查\n\n深色模式下图表发白：容器背景必须显式设为 transparent。\n"
    )
    chunks = split_text(text, 500, 80)
    assert len(chunks) == 4  # 前言 + 三个小节，各自成块
    troubleshooting = [c for c in chunks if "图表发白" in c]
    assert len(troubleshooting) == 1
    # 关键断言：故障排查那一块里不该混进别的小节正文
    assert "Ctrl+K" not in troubleshooting[0]
    assert "前端只拿一个基地址" not in troubleshooting[0]
    # 标题留在所属小节里，引用片段能看出出处
    assert troubleshooting[0].startswith("## 三、故障排查")


def test_markdown_sections_reduce_information_dilution():
    """同一份内容，小节切分后目标块的语义更"纯"——用块内词占比粗略验证。"""
    section = "深色模式下图表发白是因为容器背景没有设为 transparent。" * 6
    other = "快捷键与部署路径说明。" * 6
    merged = split_text(f"## A\n\n{other}\n\n## B\n\n{section}", 500, 80)
    target = [c for c in merged if "transparent" in c][0]
    assert "快捷键与部署路径" not in target


def test_preamble_before_first_heading_kept():
    text = "这是没有标题的前言。\n\n## 小节\n\n小节正文。"
    chunks = split_text(text, 500, 0)
    assert any(c.startswith("这是没有标题的前言") for c in chunks)
    assert any("小节正文" in c for c in chunks)


def test_long_section_still_splits_with_overlap():
    """小节超过 chunk_size 时仍要在小节内继续切（标题自成一块是允许的）。"""
    section = "".join(f"第{i}句内容。" for i in range(1, 40))
    chunks = split_text(f"## 长小节\n\n{section}", 60, 20)
    headings = [c for c in chunks if c.startswith("## ")]
    body = [c for c in chunks if not c.startswith("## ")]
    assert headings == ["## 长小节"]  # 标题单独成块，正文从下一块开始
    assert len(body) > 2
    assert all(len(c) <= 60 for c in body)
    # 小节内仍然重叠（重叠行为本身在 test_overlap_prefixes_next_chunk 里单独钉过）
    assert body[0][-6:] in body[1]


def test_text_without_headings_behaves_as_before():
    text = "第一句话说的是甲。" * 12
    chunks = split_text(text, 40, 0)
    assert len(chunks) > 1
    assert all(chunk.endswith("。") for chunk in chunks)
