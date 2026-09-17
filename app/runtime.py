"""运行时容器：把配置、模型、存储、管道装配成一个对象，全局懒加载。

启动时**不做重活**（不加载 torch、不扫全库），第一次真正用到才初始化；
`EMBEDDER=bge` 但环境里没有 sentence-transformers 时，不静默降级成 hash，
而是明确记下 `degraded_reason` 并退回 hash，让 /api/health 和前端都能看到这条警告。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.jobs import JobRegistry
from app.rag.embedder import Embedder, HashingEmbedder, build_embedder
from app.rag.generator import LLMClient
from app.rag.lexical import LexicalIndex
from app.rag.pipeline import AskPipeline, KnowledgeBase
from app.rag.retriever import Retriever
from app.rag.store import VectorStore
from app.sessions import SessionStore

VERSION = "0.1.0"


@dataclass
class Runtime:
    settings: Settings
    embedder: Embedder
    store: VectorStore
    lexical: LexicalIndex
    retriever: Retriever
    llm: LLMClient
    sessions: SessionStore
    kb: KnowledgeBase
    ask: AskPipeline
    jobs: JobRegistry = field(default_factory=JobRegistry)
    degraded_reason: str = ""

    @classmethod
    def build(cls, settings: Settings | None = None, llm: LLMClient | None = None) -> Runtime:
        settings = settings or get_settings()
        settings.ensure_dirs()
        degraded = ""
        try:
            embedder: Embedder = build_embedder(settings)
        except RuntimeError as exc:
            degraded = f"{exc}；已临时改用零依赖哈希向量（检索质量会下降）"
            embedder = HashingEmbedder(dim=settings.hash_embed_dim)

        store = VectorStore(settings.chroma_dir, settings.collection)
        lexical = LexicalIndex(settings.lexical_path)
        if lexical.size == 0 and store.count() > 0:
            # 字面索引是派生数据：JSON 丢了就从向量库重建，不该让对比实验没法做
            lexical.add(store.all_chunks())
            lexical.save()
        retriever = Retriever(settings, embedder, store, lexical)
        llm = llm or LLMClient(settings)
        sessions = SessionStore(settings.sessions_db)
        kb = KnowledgeBase(settings, embedder, store, lexical)
        ask = AskPipeline(settings, retriever, llm, sessions, kb)
        return cls(
            settings=settings,
            embedder=embedder,
            store=store,
            lexical=lexical,
            retriever=retriever,
            llm=llm,
            sessions=sessions,
            kb=kb,
            ask=ask,
            degraded_reason=degraded,
        )

    def health(self) -> dict:
        return {
            "status": "ok",
            "version": VERSION,
            "llm_configured": self.llm.configured,
            "llm_model": self.settings.llm_model if self.llm.configured else "",
            "embedder": self.embedder.name,
            "embedder_model": self.settings.embed_model if self.embedder.name == "bge" else "",
            "degraded_reason": self.degraded_reason,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "top_k": self.settings.top_k,
            "score_threshold": self.settings.score_threshold,
            "lexical_score_threshold": self.settings.lexical_score_threshold,
            "fallback_mode": self.settings.fallback_mode,
            **self.kb.stats(),
        }


_runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime.build()
    return _runtime


def set_runtime(runtime: Runtime | None) -> None:
    """测试用：注入一个跑在临时目录上的运行时。"""
    global _runtime
    _runtime = runtime
