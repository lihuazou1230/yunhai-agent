"""知识库文档接口（规划 10.1）。

`POST /api/documents` 的两条路径：
- 小文件（≤2MB）：同步入库，直接返回统计 —— 上传完立刻能问；
- 大文件（>2MB）：**先返回任务 ID**，后台线程入库，前端轮询 `/api/jobs/{id}`。
  50 页 PDF 走向量化要几十秒，卡在 HTTP 请求里既容易超时、用户也看不到进度。
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import LARGE_FILE_BYTES, MAX_UPLOAD_BYTES
from app.errors import FileTooLarge
from app.rag.loader import source_type_for
from app.runtime import get_runtime
from app.schemas import DeleteResponse, DocumentInfo, DocumentList, IngestResponse, JobInfo

router = APIRouter(prefix="/api", tags=["knowledge"])

CHUNK_PREVIEW_FIELDS = ("doc_id", "source", "source_type", "uploaded_at", "chunks", "pages")


@router.post("/documents", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...), force: bool = Form(False)):
    """上传并入库（类型白名单 + 10MB 上限）。"""
    filename = file.filename or "untitled"
    source_type_for(filename)  # 白名单：不合法直接 400，不浪费读取
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise FileTooLarge("文件超过 10MB 上限")

    runtime = get_runtime()
    if len(data) > LARGE_FILE_BYTES:
        job_id = runtime.jobs.submit(
            f"ingest:{filename}",
            lambda: runtime.kb.ingest(filename, data, force=force).as_dict(),
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=IngestResponse(
                status="queued",
                job_id=job_id,
                message="文件较大，已转后台入库，请轮询任务状态",
            ).model_dump(),
        )

    result = await run_in_threadpool(runtime.kb.ingest, filename, data, force=force)
    return IngestResponse(
        status="done",
        document=_document_of(result.as_dict()),
        stats=runtime.kb.stats(),
        message="内容与库中已有文档一致，未重复入库" if result.skipped else "已入库",
    )


@router.get("/documents", response_model=DocumentList)
async def list_documents():
    runtime = get_runtime()
    documents = [DocumentInfo(**{k: d.get(k) for k in CHUNK_PREVIEW_FIELDS}) for d in runtime.kb.documents()]
    return DocumentList(documents=documents, total_chunks=runtime.store.count())


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """删某个文档的全部向量（规划验收项：删某来源可清其全部向量）。"""
    runtime = get_runtime()
    removed = await run_in_threadpool(runtime.kb.delete_document, doc_id)
    return DeleteResponse(doc_id=doc_id, removed_chunks=removed)


@router.post("/knowledge/reset", response_model=dict)
async def reset_knowledge():
    """清库重建入口：换 embedding 模型必须走这一步（规划 10.2）。"""
    runtime = get_runtime()
    await run_in_threadpool(runtime.kb.reset)
    return {"status": "ok", "message": "向量库与字面索引已清空", **runtime.kb.stats()}


@router.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job(job_id: str):
    return JobInfo(**get_runtime().jobs.get(job_id))


def _document_of(payload: dict) -> DocumentInfo:
    return DocumentInfo(
        doc_id=payload["doc_id"],
        source=payload["source"],
        source_type=payload["source_type"],
        uploaded_at=payload["uploaded_at"],
        chunks=payload["chunks"],
        pages=None,
    )
