"""字面索引（BM25）测试：排序、删除、持久化、归一化。"""

from __future__ import annotations

from pathlib import Path

from app.rag.lexical import LexicalIndex
from app.rag.types import Chunk


def make_chunk(chunk_id: str, text: str, source: str = "笔记.md", doc_id: str = "d1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        source_type="md",
        doc_id=doc_id,
        item_id="raw",
        chunk_index=int(chunk_id.split(":")[-1]),
        uploaded_at="2026-09-17T10:00:00",
    )


def test_search_ranks_matching_chunk_first():
    index = LexicalIndex()
    index.add(
        [
            make_chunk("d1:0", "分块默认块长 500 个字符，重叠 80 个字符。"),
            make_chunk("d1:1", "主题跟随系统，支持浅色与深色两种模式。"),
            make_chunk("d1:2", "部署在 IIS 的 80 端口 /workspace/ 子路径下。"),
        ]
    )
    hits = index.search("分块默认多少字符？", top_k=3)
    assert hits
    assert "500" in hits[0].chunk.text
    # 分数用固定尺度压缩到 (0,1)：既保排序，又保留绝对量纲（阈值才有意义）
    assert 0.0 < hits[0].score < 1.0
    assert hits[0].score >= hits[-1].score


def test_search_on_empty_index_returns_nothing():
    assert LexicalIndex().search("随便问问") == []


def test_search_with_only_stopwords_returns_nothing():
    index = LexicalIndex()
    index.add([make_chunk("d1:0", "有内容的一句话")])
    assert index.search("的了是") == []


def test_remove_by_source_and_doc():
    index = LexicalIndex()
    index.add(
        [
            make_chunk("d1:0", "第一份文档的内容", source="a.md", doc_id="d1"),
            make_chunk("d2:0", "第二份文档的内容", source="b.md", doc_id="d2"),
        ]
    )
    assert index.remove_by(source="a.md") == 1
    assert index.size == 1
    # 删掉的是 a.md 这一块；b.md 还有词面重合，被检索到是 BM25 的正常行为
    assert all(hit.chunk.source != "a.md" for hit in index.search("第一份文档"))
    assert index.remove_by(doc_id="d2") == 1
    assert index.size == 0
    assert index.search("第一份文档") == []


def test_add_is_idempotent_for_same_id():
    index = LexicalIndex()
    chunk = make_chunk("d1:0", "重复入库不应该让词频翻倍")
    index.add([chunk])
    index.add([chunk])
    assert index.size == 1
    assert 0.0 < index.search("重复入库")[0].score < 1.0


def test_persistence_round_trip(tmp_path: Path):
    path = tmp_path / "lexical.json"
    index = LexicalIndex(path)
    index.add([make_chunk("d1:0", "持久化之后还能检索到这句话")])
    index.save()

    reloaded = LexicalIndex(path)
    assert reloaded.size == 1
    assert reloaded.search("持久化检索")[0].chunk.chunk_id == "d1:0"


def test_corrupted_index_file_does_not_crash(tmp_path: Path):
    path = tmp_path / "lexical.json"
    path.write_text("{ 这不是 JSON", encoding="utf-8")
    index = LexicalIndex(path)
    assert index.size == 0
    assert index.search("任意") == []
