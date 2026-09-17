"""向量库测试：幂等 upsert、分数区间、按元数据删除、清库重建。"""

from __future__ import annotations

from pathlib import Path

from app.rag.embedder import HashingEmbedder
from app.rag.store import VectorStore
from app.rag.types import Chunk

EMBEDDER = HashingEmbedder(dim=256)


def make_chunk(chunk_id: str, text: str, *, source: str = "笔记.md", doc_id: str = "d1", index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        source_type="md",
        doc_id=doc_id,
        item_id="raw",
        chunk_index=index,
        uploaded_at="2026-09-17T10:00:00",
    )


def build(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", "test_collection")


def test_upsert_same_ids_is_idempotent(tmp_path: Path):
    store = build(tmp_path)
    chunks = [make_chunk("d1:0", "分块默认 500 字符"), make_chunk("d1:1", "重叠 80 字符", index=1)]
    store.upsert(chunks, EMBEDDER.encode([c.text for c in chunks]))
    assert store.count() == 2
    # 重复入库：同一批 id 覆盖，不新增
    store.upsert(chunks, EMBEDDER.encode([c.text for c in chunks]))
    assert store.count() == 2


def test_query_returns_normalised_similarity(tmp_path: Path):
    store = build(tmp_path)
    chunks = [
        make_chunk("d1:0", "分块默认块长 500 个字符"),
        make_chunk("d1:1", "部署在 IIS 的 80 端口子路径", index=1),
    ]
    store.upsert(chunks, EMBEDDER.encode([c.text for c in chunks]))

    hits = store.query(EMBEDDER.encode(["分块默认块长"], is_query=True)[0], top_k=2)
    assert hits
    assert "500" in hits[0].chunk.text
    assert 0.0 <= hits[0].score <= 1.0
    assert hits[0].score >= hits[-1].score


def test_query_on_empty_store(tmp_path: Path):
    store = build(tmp_path)
    assert store.query(EMBEDDER.encode(["随便"])[0]) == []


def test_delete_by_source_and_doc(tmp_path: Path):
    store = build(tmp_path)
    chunks = [
        make_chunk("d1:0", "第一篇内容", source="a.md", doc_id="d1"),
        make_chunk("d2:0", "第二篇内容", source="b.md", doc_id="d2"),
    ]
    store.upsert(chunks, EMBEDDER.encode([c.text for c in chunks]))
    assert store.delete_by(source="a.md") == 1
    assert store.count() == 1
    assert store.delete_by(doc_id="d2") == 1
    assert store.count() == 0


def test_delete_requires_a_filter(tmp_path: Path):
    import pytest

    store = build(tmp_path)
    with pytest.raises(ValueError):
        store.delete_by()


def test_list_documents_aggregates_chunks_and_pages(tmp_path: Path):
    store = build(tmp_path)
    chunks = [
        make_chunk("d1:0", "第一页", doc_id="d1", index=0),
        make_chunk("d1:1", "第二页", doc_id="d1", index=1),
    ]
    store.upsert(chunks, EMBEDDER.encode([c.text for c in chunks]))
    documents = store.list_documents()
    assert len(documents) == 1
    assert documents[0]["chunks"] == 2
    assert documents[0]["doc_id"] == "d1"


def test_all_chunks_round_trip_metadata(tmp_path: Path):
    store = build(tmp_path)
    chunk = make_chunk("d1:0", "带元数据的一块")
    store.upsert([chunk], EMBEDDER.encode([chunk.text]))
    restored = store.all_chunks()
    assert len(restored) == 1
    assert restored[0].source == "笔记.md"
    assert restored[0].chunk_id == "d1:0"


def test_reset_clears_collection(tmp_path: Path):
    store = build(tmp_path)
    chunk = make_chunk("d1:0", "内容")
    store.upsert([chunk], EMBEDDER.encode([chunk.text]))
    store.reset()
    assert store.count() == 0
    # 清库后还能继续写（集合被重建过）
    store.upsert([chunk], EMBEDDER.encode([chunk.text]))
    assert store.count() == 1
