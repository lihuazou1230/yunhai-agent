"""入库与问答的主流程编排。

入库（规划 10.2）：
  解析 → 分块 → 向量化 → 写入 Chroma + 字面索引，并同时落一份原文件
  （`data/uploads/`），这样换分块参数或换模型时可以**离线重建**，不用让用户重传。

幂等（规划验收项）：
  - `doc_id = sha256(内容)[:16]`：文件没变 -> 直接判定 skipped，一块向量都不新增；
  - 同名来源重新上传 -> 先按 source 清掉旧块再写新的（改过的文件不会新旧并存）。

问答（规划 10.3）：
  检索 → 阈值拦截 → 命中则约束生成 + 引用；未命中则走兜底三模式。
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app import sse
from app.config import Settings
from app.errors import AgentError, EmptyDocument, FileTooLarge, NotConfigured
from app.rag.chunker import build_chunks
from app.rag.embedder import Embedder
from app.rag.generator import LLMClient
from app.rag.lexical import LexicalIndex
from app.rag.loader import load_items, source_type_for
from app.rag.prompt import WEB_HINT, build_messages, citations_of
from app.rag.retriever import Retriever
from app.rag.store import VectorStore
from app.sessions import SessionStore

# 兜底模式名（规划 10.3）
FALLBACK_REFUSE = "refuse"
FALLBACK_BARE = "bare"
FALLBACK_WEB = "web"
FALLBACK_MODES = (FALLBACK_REFUSE, FALLBACK_BARE, FALLBACK_WEB)

REFUSAL_TEXT = (
    "知识库里没有检索到与该问题相关的内容，因此不作答（避免编造）。\n"
    "可以试试：换个说法再问一次，或先把相关文档上传到知识库。"
)


@dataclass
class IngestResult:
    doc_id: str
    source: str
    source_type: str
    items: int
    chunks: int
    skipped: bool
    uploaded_at: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class KnowledgeBase:
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

    # ---------- 入库 ----------

    def ingest(self, filename: str, data: bytes, *, force: bool = False) -> IngestResult:
        source_type = source_type_for(filename)  # 白名单在这里兜底（接口层也会拦一次）
        if len(data) > self._settings_max_bytes():
            raise FileTooLarge("文件超过 10MB 上限")

        doc_id = hashlib.sha256(data).hexdigest()[:16]
        existing = {doc["doc_id"] for doc in self._store.list_documents()}
        if doc_id in existing and not force:
            # 内容一模一样：一块向量都不用重算（重复入库不产生重复向量）
            return IngestResult(
                doc_id=doc_id,
                source=filename,
                source_type=source_type,
                items=0,
                chunks=self._store.count(),
                skipped=True,
                uploaded_at=_now(),
                size_bytes=len(data),
            )

        items = load_items(filename, data, self._settings.jsonl_fields)
        uploaded_at = _now()
        chunks = build_chunks(
            items,
            source=filename,
            source_type=source_type,
            doc_id=doc_id,
            uploaded_at=uploaded_at,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            raise EmptyDocument("分块后没有内容，文件可能只有空白")

        # 同名来源先清干净：改过的文件重传时不会新旧并存
        self._store.delete_by(source=filename)
        self._lexical.remove_by(source=filename)

        for batch in _batched(chunks, self._settings.embed_batch_size):
            vectors = self._embedder.encode([c.text for c in batch])
            self._store.upsert(batch, vectors)
        self._lexical.add(chunks)
        self._lexical.save()
        self._save_upload(doc_id, filename, data)

        return IngestResult(
            doc_id=doc_id,
            source=filename,
            source_type=source_type,
            items=len(items),
            chunks=len(chunks),
            skipped=False,
            uploaded_at=uploaded_at,
            size_bytes=len(data),
        )

    def _settings_max_bytes(self) -> int:
        from app.config import MAX_UPLOAD_BYTES

        return MAX_UPLOAD_BYTES

    def _save_upload(self, doc_id: str, filename: str, data: bytes) -> None:
        suffix = Path(filename).suffix.lower()
        (self._settings.uploads_dir / f"{doc_id}{suffix}").write_bytes(data)

    # ---------- 删除 ----------

    def delete_document(self, doc_id: str) -> int:
        removed = self._store.delete_by(doc_id=doc_id)
        self._lexical.remove_by(doc_id=doc_id)
        self._lexical.save()
        for path in self._settings.uploads_dir.glob(f"{doc_id}.*"):
            path.unlink(missing_ok=True)
        return removed

    def delete_source(self, source: str) -> int:
        removed = self._store.delete_by(source=source)
        self._lexical.remove_by(source=source)
        self._lexical.save()
        return removed

    def reset(self) -> None:
        """换 embedding 模型必须清库重建（规划 10.2）。"""
        self._store.reset()
        self._lexical.clear()
        self._lexical.save()

    # ---------- 读取 ----------

    def documents(self) -> list[dict[str, Any]]:
        return self._store.list_documents()

    def stats(self) -> dict[str, Any]:
        documents = self._store.list_documents()
        return {
            "documents": len(documents),
            "chunks": self._store.count(),
            "lexical_chunks": self._lexical.size,
            "sources": [d["source"] for d in documents],
        }


class AskPipeline:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        llm: LLMClient,
        sessions: SessionStore,
        kb: KnowledgeBase,
    ):
        self._settings = settings
        self._retriever = retriever
        self._llm = llm
        self._sessions = sessions
        self._kb = kb

    async def stream(
        self,
        question: str,
        *,
        session_id: str | None = None,
        mode: str = "semantic",
        fallback_mode: str | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> AsyncIterator[str]:
        started = time.perf_counter()
        fallback = (fallback_mode or self._settings.fallback_mode).strip() or FALLBACK_REFUSE
        if fallback not in FALLBACK_MODES:
            fallback = FALLBACK_REFUSE

        resolved_session = session_id
        try:
            hits = await asyncio.to_thread(
                self._retriever.filtered,
                question,
                mode=mode,
                top_k=top_k,
                threshold=threshold,
            )
            resolved_session = await asyncio.to_thread(
                self._sessions.ensure_session, session_id, question, mode
            )
            await asyncio.to_thread(
                self._sessions.add_message, resolved_session, "user", question, meta={"mode": mode}
            )

            citations = citations_of(hits)
            for citation in citations:
                yield sse.citation_event(citation)

            answer_parts: list[str] = []
            used_fallback = "kb" if hits else fallback

            if hits:
                async for piece in self._generate(build_messages(question, hits)):
                    answer_parts.append(piece)
                    yield sse.token_event(piece)
            else:
                async for frame in self._fallback_stream(question, fallback, answer_parts):
                    yield frame

            answer = "".join(answer_parts)
            message = await asyncio.to_thread(
                self._sessions.add_message,
                resolved_session,
                "assistant",
                answer,
                citations=citations,
                meta={
                    "fallback": used_fallback,
                    "mode": mode,
                    "hits": [round(h.score, 4) for h in hits],
                    "threshold": self._retriever.threshold_for(mode) if threshold is None else threshold,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            yield sse.done_event(
                session_id=resolved_session,
                message_id=message["id"],
                citations=citations,
                fallback=used_fallback,
                hit_count=len(hits),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except AgentError as exc:
            yield sse.error_event(exc.message, exc.code)
        except Exception as exc:  # noqa: BLE001 - 流一旦开始就不能抛，否则前端只看到连接断
            yield sse.error_event(f"服务内部错误：{exc}", "internal_error")

    async def _generate(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        async for piece in self._llm.stream_chat(messages):
            yield piece

    async def _fallback_stream(
        self, question: str, fallback: str, sink: list[str]
    ) -> AsyncIterator[str]:
        """兜底三模式（规划 10.3）。"""
        if fallback == FALLBACK_REFUSE:
            # 模式 A：不花 token、也绝不编造——把固定文案按块"流"出去，前端只认一套协议
            for piece in _chunks_of(REFUSAL_TEXT, 12):
                sink.append(piece)
                yield sse.token_event(piece)
            return

        if fallback == FALLBACK_BARE:
            async for piece in self._generate(build_messages(question, [], bare=True)):
                sink.append(piece)
                yield sse.token_event(piece)
            return

        # 模式 C（联网搜索）规划里明确后置：这里给明确提示，不假装答得出来
        for piece in _chunks_of(WEB_HINT, 12):
            sink.append(piece)
            yield sse.token_event(piece)
        yield sse.error_event("联网搜索兜底（模式 C）尚未接入，本阶段按拒答处理", "web_search_unavailable")


def _chunks_of(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _batched(items: list, size: int) -> list[list]:
    step = max(1, size)
    return [items[i : i + step] for i in range(0, len(items), step)]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


__all__ = [
    "AskPipeline",
    "IngestResult",
    "KnowledgeBase",
    "REFUSAL_TEXT",
    "FALLBACK_REFUSE",
    "FALLBACK_BARE",
    "FALLBACK_WEB",
    "NotConfigured",
]
