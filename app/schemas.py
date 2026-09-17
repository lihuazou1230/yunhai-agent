"""接口出入参模型（Pydantic）。

只描述**前端会看到的**结构；内部数据类留在 app/rag/types.py。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RetrievalMode = Literal["semantic", "lexical"]
FallbackMode = Literal["refuse", "bare", "web"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, description="用户问题")
    session_id: str | None = Field(default=None, description="留空则新建会话")
    mode: RetrievalMode = Field(default="semantic", description="检索策略")
    fallback_mode: FallbackMode | None = Field(default=None, description="留空用后端默认")
    top_k: int | None = Field(default=None, ge=1, le=20)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    source_type: str
    uploaded_at: str
    chunks: int
    pages: int | None = None


class DocumentList(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class IngestResponse(BaseModel):
    status: Literal["done", "queued"]
    document: DocumentInfo | None = None
    stats: dict[str, Any] | None = None
    job_id: str | None = None
    message: str = ""


class DeleteResponse(BaseModel):
    doc_id: str
    removed_chunks: int


class JobInfo(BaseModel):
    job_id: str
    name: str
    status: Literal["pending", "running", "succeeded", "failed"]
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class MessageInfo(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class SessionInfo(BaseModel):
    id: str
    title: str
    mode: str
    created_at: str
    updated_at: str
    message_count: int = 0


class SessionDetail(SessionInfo):
    messages: list[MessageInfo] = Field(default_factory=list)


class SessionList(BaseModel):
    sessions: list[SessionInfo]


class ErrorBody(BaseModel):
    code: str
    message: str
