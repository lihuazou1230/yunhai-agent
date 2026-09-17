"""向量库：Chroma 嵌入式（规划 10.2）。

两个刻意的选择：
1. `hnsw:space = cosine` + **我们自己传向量**（不注册 embedding_function）。
   理由：embedding 模型可能换（bge → api → hash），把"算向量"留在应用层，
   换模型时只需要清库重建，不用动存储层；也避免了 Chroma 偷偷用默认英文模型。
2. `upsert` 而不是 `add`：chunk_id 由 `doc_id:序号` 决定，
   同一份文件重跑入库就是覆盖同一批 id —— **幂等**（规划验收项之一）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.rag.types import Chunk, Retrieved


class VectorStore:
    def __init__(self, path: Path, collection_name: str):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------- 写入 ----------

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("块数与向量数不一致")
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
        )
        return len(chunks)

    def drop_ids(self, chunk_ids: list[str]) -> int:
        """按 id 精确删除（重建某文档时先清旧块，处理"文件改短了"的情况）。"""
        if not chunk_ids:
            return 0
        self._collection.delete(ids=chunk_ids)
        return len(chunk_ids)

    def delete_by(self, *, doc_id: str | None = None, source: str | None = None) -> int:
        """按元数据批量删除：删某个来源可清其全部向量（规划验收项）。"""
        if doc_id is None and source is None:
            raise ValueError("必须给出 doc_id 或 source")
        where: dict[str, Any] = {}
        if doc_id is not None:
            where["doc_id"] = doc_id
        if source is not None:
            where["source"] = source
        chunk_ids = self.ids_where(where)
        if chunk_ids:
            self._collection.delete(ids=chunk_ids)
        return len(chunk_ids)

    def reset(self) -> None:
        """换 embedding 模型必须清库重建（向量维度和语义空间都变了）。"""
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    # ---------- 查询 ----------

    def query(
        self,
        embedding: list[float],
        top_k: int = 4,
        where: dict[str, Any] | None = None,
    ) -> list[Retrieved]:
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(max(1, top_k), total),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[Retrieved] = []
        for index, chunk_id in enumerate(ids):
            meta = dict(metadatas[index] or {})
            # 余弦距离 -> 相似度；浮点误差可能压出 [-1,1] 之外，夹一下
            similarity = max(0.0, min(1.0, 1.0 - float(distances[index])))
            hits.append(
                Retrieved(
                    chunk=Chunk(
                        chunk_id=chunk_id,
                        text=documents[index] or "",
                        source=str(meta.get("source", "")),
                        source_type=str(meta.get("source_type", "")),
                        doc_id=str(meta.get("doc_id", "")),
                        item_id=str(meta.get("item_id", "")),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        uploaded_at=str(meta.get("uploaded_at", "")),
                        page=int(meta["page"]) if meta.get("page") is not None else None,
                    ),
                    score=similarity,
                )
            )
        return hits

    # ---------- 元信息 ----------

    def count(self) -> int:
        return int(self._collection.count())

    def ids_where(self, where: dict[str, Any] | None = None) -> list[str]:
        result = self._collection.get(where=where or None, include=[])
        return list(result.get("ids") or [])

    def all_chunks(self, page_size: int = 1000) -> list[Chunk]:
        """把库里所有块捞回来（用于从向量库重建字面索引）。"""
        chunks: list[Chunk] = []
        offset = 0
        while True:
            result = self._collection.get(
                limit=page_size, offset=offset, include=["documents", "metadatas"]
            )
            ids = list(result.get("ids") or [])
            documents = list(result.get("documents") or [])
            metadatas = list(result.get("metadatas") or [])
            for index, chunk_id in enumerate(ids):
                meta = dict(metadatas[index] or {})
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=documents[index] or "",
                        source=str(meta.get("source", "")),
                        source_type=str(meta.get("source_type", "")),
                        doc_id=str(meta.get("doc_id", "")),
                        item_id=str(meta.get("item_id", "")),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        uploaded_at=str(meta.get("uploaded_at", "")),
                        page=int(meta["page"]) if meta.get("page") is not None else None,
                    )
                )
            if len(ids) < page_size:
                break
            offset += page_size
        return chunks

    def all_metadatas(self, page_size: int = 1000) -> list[dict[str, Any]]:
        """分页捞元数据（Chroma 的 get 有默认上限，不分页会静默截断）。"""
        collected: list[dict[str, Any]] = []
        offset = 0
        while True:
            result = self._collection.get(limit=page_size, offset=offset, include=["metadatas"])
            batch = list(result.get("metadatas") or [])
            collected.extend(dict(m or {}) for m in batch)
            if len(batch) < page_size:
                break
            offset += page_size
        return collected

    def list_documents(self) -> list[dict[str, Any]]:
        """按 doc_id 聚合出文档清单（前端知识库侧栏用）。"""
        grouped: dict[str, dict[str, Any]] = {}
        for meta in self.all_metadatas():
            doc_id = str(meta.get("doc_id", ""))
            entry = grouped.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "source": str(meta.get("source", "")),
                    "source_type": str(meta.get("source_type", "")),
                    "uploaded_at": str(meta.get("uploaded_at", "")),
                    "chunks": 0,
                    "pages": set(),
                },
            )
            entry["chunks"] += 1
            if meta.get("page") is not None:
                entry["pages"].add(int(meta["page"]))
        documents = []
        for entry in grouped.values():
            pages = entry.pop("pages")
            if pages:
                entry["pages"] = max(pages)
            documents.append(entry)
        documents.sort(key=lambda d: d.get("uploaded_at", ""), reverse=True)
        return documents
