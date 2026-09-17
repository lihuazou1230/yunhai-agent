"""RAG 领域内的数据模型（纯数据类，不依赖 FastAPI）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LoadedItem:
    """解析阶段的一条原始条目。

    `item_id` 是**入库幂等**的最小单位：同一条目重跑解析不会产生新向量。
    pdf 一页一条、jsonl 一行一条、md/txt 一段一条（段落由分块器再细分）。
    """

    item_id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """入库的最小单位（向量 + 元数据）。"""

    chunk_id: str
    text: str
    source: str
    source_type: str
    doc_id: str
    item_id: str
    chunk_index: int
    uploaded_at: str
    page: int | None = None

    def metadata(self) -> dict[str, Any]:
        """Chroma 只接受 str/int/float/bool —— None 一律不进元数据。"""
        meta: dict[str, Any] = {
            "source": self.source,
            "source_type": self.source_type,
            "doc_id": self.doc_id,
            "item_id": self.item_id,
            "chunk_index": self.chunk_index,
            "uploaded_at": self.uploaded_at,
        }
        if self.page is not None:
            meta["page"] = self.page
        return meta


@dataclass(frozen=True)
class Retrieved:
    """检索命中的一块，带分数（0~1，越大越相关）。"""

    chunk: Chunk
    score: float

    def as_citation(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "source": self.chunk.source,
            "source_type": self.chunk.source_type,
            "doc_id": self.chunk.doc_id,
            "chunk_index": self.chunk.chunk_index,
            "page": self.chunk.page,
            "score": round(self.score, 4),
            "snippet": self.chunk.text[:160],
        }
