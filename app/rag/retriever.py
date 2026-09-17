"""检索与阈值拦截（规划 10.3）。

**库外问题不进生成环节**是这一层唯一的硬指标：命中分数低于阈值就当作"库里没有"，
交给兜底策略处理。阈值不是拍的——`scripts/eval_retrieval.py` 用库内/库外两套
问题集扫一遍，取"库内召回最大化且库外零漏网"的那条线。
"""

from __future__ import annotations

from app.config import Settings
from app.rag.embedder import Embedder
from app.rag.lexical import LexicalIndex
from app.rag.store import VectorStore
from app.rag.types import Retrieved


class Retriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        lexical: LexicalIndex,
    ):
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._lexical = lexical

    # ---------- 查询 ----------

    def search(
        self,
        question: str,
        *,
        mode: str = "semantic",
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[Retrieved]:
        """按指定策略取 top-k（不判阈值；判阈值是 `filtered` 的事）。"""
        k = top_k or self._settings.top_k
        if mode == "lexical":
            hits = self._lexical.search(question, top_k=k)
            return self._apply_where(hits, where)
        vector = self._embedder.encode([question], is_query=True)[0]
        return self._store.query(vector, top_k=k, where=where)

    def filtered(
        self,
        question: str,
        *,
        mode: str = "semantic",
        top_k: int | None = None,
        threshold: float | None = None,
        where: dict | None = None,
    ) -> list[Retrieved]:
        """取 top-k 并做阈值拦截——返回空列表 = 「知识库里没有」。"""
        hits = self.search(question, mode=mode, top_k=top_k, where=where)
        limit = self.threshold_for(mode) if threshold is None else threshold
        kept = [hit for hit in hits if hit.score >= limit]
        if len(kept) < self._settings.min_hits:
            return []
        return kept

    def threshold_for(self, mode: str) -> float:
        if mode == "lexical":
            return self._settings.lexical_score_threshold
        return self._settings.score_threshold

    @staticmethod
    def _apply_where(hits: list[Retrieved], where: dict | None) -> list[Retrieved]:
        """字面索引不支持原生过滤，在这里补一遍（元数据量小，够用）。"""
        if not where:
            return hits
        kept = []
        for hit in hits:
            meta = hit.chunk.metadata()
            if all(meta.get(key) == value for key, value in where.items()):
                kept.append(hit)
        return kept
