"""字面检索基线：BM25（规划 10.2 要求与语义方案做对比）。

存在的意义有两个：
1. **对比结论**——"中文语义模型到底值不值那 400MB"要用数据回答，不能靠感觉；
2. **兜底**——没有模型（或模型没下完）时，知识库仍然可用。

索引持久化成一个 JSON：纯 Python、可读、可手改、可进 git 做小样本评测。
规模上限是几万块，个人知识库完全够；再大就该换倒排库了。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from app.rag.tokenize import tokenize
from app.rag.types import Chunk, Retrieved

K1 = 1.5
B = 0.75
# BM25 原始分 -> (0,1) 的压缩尺度：score = raw / (raw + SCALE)
# 为什么不用「除以本次查询最高分」：那样每条查询的第一名恒为 1.0，
# 阈值就永远拦不住任何东西（评测里当场抓到这个坑）。固定尺度保留了绝对量纲，
# 阈值才有意义；SCALE 由 scripts/eval_retrieval.py 的分数分布定。
SCALE = 8.0


class LexicalIndex:
    """BM25 倒排索引，支持按文档/来源删除与幂等重建。"""

    def __init__(self, path: Path | None = None):
        self.path = path
        self._chunks: dict[str, Chunk] = {}
        self._tokens: dict[str, Counter[str]] = {}
        self._lengths: dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._avg_len = 0.0
        if path and path.exists():
            self.load()

    # ---------- 写入 ----------

    def add(self, chunks: list[Chunk]) -> int:
        """覆盖式写入：同 id 先删再加（重跑解析不会翻倍）。"""
        added = 0
        for chunk in chunks:
            if chunk.chunk_id in self._chunks:
                self.remove_ids([chunk.chunk_id])
            counts = Counter(tokenize(chunk.text))
            if not counts:
                continue
            self._chunks[chunk.chunk_id] = chunk
            self._tokens[chunk.chunk_id] = counts
            self._lengths[chunk.chunk_id] = sum(counts.values())
            for token in counts:
                self._df[token] += 1
            added += 1
        self._recompute_avg()
        return added

    def remove_ids(self, chunk_ids: list[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            counts = self._tokens.pop(chunk_id, None)
            if counts is None:
                continue
            self._chunks.pop(chunk_id, None)
            self._lengths.pop(chunk_id, None)
            for token in counts:
                self._df[token] -= 1
                if self._df[token] <= 0:
                    del self._df[token]
            removed += 1
        self._recompute_avg()
        return removed

    def remove_by(self, *, doc_id: str | None = None, source: str | None = None) -> int:
        targets = [
            chunk_id
            for chunk_id, chunk in self._chunks.items()
            if (doc_id is None or chunk.doc_id == doc_id) and (source is None or chunk.source == source)
        ]
        return self.remove_ids(targets)

    def clear(self) -> None:
        self._chunks.clear()
        self._tokens.clear()
        self._lengths.clear()
        self._df.clear()
        self._avg_len = 0.0

    # ---------- 查询 ----------

    def search(self, query: str, top_k: int = 4) -> list[Retrieved]:
        """返回分数归一到 (0, 1) 的命中列表。

        BM25 原始分没有绝对量纲（跟查询长度、语料规模都有关），
        这里用**固定尺度**的压缩函数 `raw / (raw + SCALE)` 映射到 (0,1)：
        它保持排序不变，又不像"除以本次最高分"那样让第一名恒为 1.0
        （那样阈值形同虚设——阈值校准实验里正是这么暴露出来的）。
        代价是阈值必须单独校准，不能直接套用余弦相似度那条线。
        """
        query_tokens = Counter(tokenize(query))
        if not query_tokens or not self._chunks:
            return []

        scores: dict[str, float] = {}
        for token, qtf in query_tokens.items():
            df = self._df.get(token)
            if not df:
                continue
            idf = math.log(1 + (len(self._chunks) - df + 0.5) / (df + 0.5))
            for chunk_id, counts in self._tokens.items():
                tf = counts.get(token)
                if not tf:
                    continue
                length = self._lengths[chunk_id]
                denominator = tf + K1 * (1 - B + B * length / (self._avg_len or 1))
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * tf * (K1 + 1) / denominator * qtf

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_k)]
        return [Retrieved(chunk=self._chunks[cid], score=raw / (raw + SCALE)) for cid, raw in ranked]

    # ---------- 持久化 ----------

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "source": c.source,
                    "source_type": c.source_type,
                    "doc_id": c.doc_id,
                    "item_id": c.item_id,
                    "chunk_index": c.chunk_index,
                    "uploaded_at": c.uploaded_at,
                    "page": c.page,
                }
                for c in self._chunks.values()
            ],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # 索引坏了不该让服务起不来：丢掉重建即可（向量库才是真相）
            self.clear()
            return
        self.clear()
        for raw in payload.get("chunks", []):
            chunk = Chunk(
                chunk_id=raw["chunk_id"],
                text=raw["text"],
                source=raw["source"],
                source_type=raw["source_type"],
                doc_id=raw["doc_id"],
                item_id=raw["item_id"],
                chunk_index=raw["chunk_index"],
                uploaded_at=raw["uploaded_at"],
                page=raw.get("page"),
            )
            self.add([chunk])

    @property
    def size(self) -> int:
        return len(self._chunks)

    def _recompute_avg(self) -> None:
        """平均块长参与 BM25 的长度归一化，增删后必须重算。"""
        if not self._lengths:
            self._avg_len = 0.0
            return
        self._avg_len = sum(self._lengths.values()) / len(self._lengths)
