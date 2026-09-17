"""检索与阈值拦截测试。"""

from __future__ import annotations

from app.runtime import Runtime


def ingest_sample(runtime: Runtime, sample_md: str) -> str:
    result = runtime.kb.ingest("手册.md", sample_md.encode("utf-8"))
    return result.doc_id


def test_semantic_search_hits_relevant_chunk(runtime: Runtime, sample_md: str):
    ingest_sample(runtime, sample_md)
    hits = runtime.retriever.search("分块默认块长是多少？", mode="semantic", top_k=3)
    assert hits
    assert "500" in hits[0].chunk.text


def test_threshold_blocks_out_of_scope_question(runtime: Runtime, sample_md: str):
    ingest_sample(runtime, sample_md)
    hits = runtime.retriever.search("世界杯冠军是谁", mode="semantic", top_k=3)
    assert all(hit.score < runtime.settings.score_threshold for hit in hits)
    # 阈值拦截后应当判定为「知识库里没有」
    assert runtime.retriever.filtered("世界杯冠军是谁", mode="semantic", top_k=3) == []


def test_threshold_keeps_in_scope_question(runtime: Runtime, sample_md: str):
    ingest_sample(runtime, sample_md)
    hits = runtime.retriever.filtered("分块默认块长是多少？", mode="semantic", top_k=3)
    assert hits
    assert all(hit.score >= runtime.settings.score_threshold for hit in hits)
    assert any("500" in hit.chunk.text for hit in hits)


def test_explicit_threshold_overrides_config(runtime: Runtime, sample_md: str):
    ingest_sample(runtime, sample_md)
    assert runtime.retriever.filtered("分块默认块长", mode="semantic", threshold=0.999) == []


def test_lexical_mode_uses_its_own_threshold(runtime: Runtime, sample_md: str):
    ingest_sample(runtime, sample_md)
    hits = runtime.retriever.filtered("余弦距离 阈值", mode="lexical", top_k=3)
    assert hits
    assert runtime.retriever.threshold_for("lexical") == runtime.settings.lexical_score_threshold
    assert runtime.retriever.threshold_for("semantic") == runtime.settings.score_threshold


def test_where_filter_limits_sources(runtime: Runtime, sample_md: str):
    ingest_sample(runtime, sample_md)
    runtime.kb.ingest("别的.md", "分块默认块长 500 个字符，来自另一份文档。".encode())
    hits = runtime.retriever.search(
        "分块默认块长", mode="semantic", top_k=5, where={"source": "别的.md"}
    )
    assert hits
    assert {hit.chunk.source for hit in hits} == {"别的.md"}


def test_min_hits_can_force_miss(runtime: Runtime, sample_md: str):
    ingest_sample(runtime, sample_md)
    runtime.settings.min_hits = 5
    assert runtime.retriever.filtered("分块默认块长是多少？", mode="semantic", top_k=2) == []
