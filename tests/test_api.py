"""HTTP 接口测试（TestClient，走真路由、真 SSE、真会话库）。"""

from __future__ import annotations

import json
import time

import pytest

import app.api.documents as documents_module


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event = ""
        data = "{}"
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        events.append((event, json.loads(data)))
    return events


def upload(client, name: str, content: bytes, mime: str = "text/markdown"):
    return client.post("/api/documents", files={"file": (name, content, mime)})


# ---------------- 健康检查 ----------------


def test_health_reports_capabilities(client):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["embedder"] == "hash"
    assert payload["llm_configured"] is True
    assert payload["fallback_mode"] == "refuse"
    assert payload["documents"] == 0


def test_embedder_warmup_returns_dimension(client, settings):
    payload = client.get("/api/health/embedder").json()
    assert payload["status"] == "ok"
    assert payload["dim"] == settings.hash_embed_dim


# ---------------- 文档接口 ----------------


def test_upload_then_list(client, sample_md):
    response = upload(client, "手册.md", sample_md.encode("utf-8"))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "done"
    assert body["document"]["source"] == "手册.md"
    assert body["message"] == "已入库"
    assert body["stats"]["chunks"] > 0

    listed = client.get("/api/documents").json()
    assert [d["source"] for d in listed["documents"]] == ["手册.md"]
    assert listed["total_chunks"] == body["document"]["chunks"]


def test_upload_same_file_again_is_skipped(client, sample_md):
    upload(client, "手册.md", sample_md.encode("utf-8"))
    again = upload(client, "手册.md", sample_md.encode("utf-8")).json()
    assert again["message"].startswith("内容与库中已有文档一致")
    assert client.get("/api/documents").json()["total_chunks"] == again["document"]["chunks"]


def test_upload_rejects_unsupported_type(client):
    response = upload(client, "图.png", b"\x89PNG\r\n", "image/png")
    assert response.status_code == 400
    assert response.json()["code"] == "unsupported_file_type"
    assert "不支持的文件类型" in response.json()["message"]


def test_upload_rejects_oversized_file(client, monkeypatch, sample_md):
    monkeypatch.setattr(documents_module, "MAX_UPLOAD_BYTES", 50)
    response = upload(client, "手册.md", sample_md.encode("utf-8"))
    assert response.status_code == 413
    assert response.json()["code"] == "file_too_large"


def test_large_file_goes_async_and_reports_job(client, monkeypatch, sample_md):
    monkeypatch.setattr(documents_module, "LARGE_FILE_BYTES", 16)
    response = upload(client, "手册.md", sample_md.encode("utf-8"))
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]

    job = None
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert job is not None
    assert job["status"] == "succeeded", job
    assert job["result"]["chunks"] > 0
    assert client.get("/api/documents").json()["total_chunks"] > 0


def test_unknown_job_returns_404(client):
    response = client.get("/api/jobs/nope")
    assert response.status_code == 404
    assert response.json()["code"] == "job_not_found"


def test_delete_document_clears_vectors(client, sample_md):
    doc_id = upload(client, "手册.md", sample_md.encode("utf-8")).json()["document"]["doc_id"]
    deleted = client.delete(f"/api/documents/{doc_id}").json()
    assert deleted["removed_chunks"] > 0
    assert client.get("/api/documents").json()["documents"] == []


def test_reset_knowledge(client, sample_md):
    upload(client, "手册.md", sample_md.encode("utf-8"))
    payload = client.post("/api/knowledge/reset").json()
    assert payload["status"] == "ok"
    assert payload["chunks"] == 0


# ---------------- 问答接口（SSE）----------------


def test_ask_streams_sse_with_citations(client, sample_md):
    upload(client, "手册.md", sample_md.encode("utf-8"))
    response = client.post("/api/ask", json={"question": "分块默认块长是多少？", "strategy": "rag"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"

    events = parse_sse(response.text)
    assert events[0][0] == "citation"
    assert events[-1][0] == "done"
    assert events[-1][1]["fallback"] == "kb"
    assert any(kind == "token" for kind, _ in events)


def test_ask_out_of_scope_refuses(client, sample_md):
    upload(client, "手册.md", sample_md.encode("utf-8"))
    events = parse_sse(client.post("/api/ask", json={"question": "世界杯冠军是谁？", "strategy": "rag"}).text)
    assert events[-1][1]["fallback"] == "refuse"
    assert not any(kind == "citation" for kind, _ in events)


def test_ask_rejects_empty_question(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_ask_rejects_bad_mode(client):
    assert client.post("/api/ask", json={"question": "问题", "mode": "乱写"}).status_code == 422


# ---------------- 会话接口 ----------------


def test_session_endpoints(client, sample_md):
    upload(client, "手册.md", sample_md.encode("utf-8"))
    done = parse_sse(client.post("/api/ask", json={"question": "分块默认块长是多少？", "strategy": "rag"}).text)[-1][1]
    session_id = done["session_id"]

    listed = client.get("/api/sessions").json()["sessions"]
    assert [s["id"] for s in listed] == [session_id]
    assert listed[0]["message_count"] == 2

    detail = client.get(f"/api/sessions/{session_id}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["citations"][0]["source"] == "手册.md"

    assert client.delete(f"/api/sessions/{session_id}").json()["status"] == "ok"
    assert client.get("/api/sessions").json()["sessions"] == []


def test_unknown_session_returns_404(client):
    response = client.get("/api/sessions/nope")
    assert response.status_code == 404
    assert response.json()["code"] == "session_not_found"


def test_ask_can_continue_an_existing_session(client, sample_md):
    upload(client, "手册.md", sample_md.encode("utf-8"))
    first = parse_sse(client.post("/api/ask", json={"question": "分块默认块长是多少？", "strategy": "rag"}).text)[-1][1]
    second = parse_sse(
        client.post(
            "/api/ask",
            json={"question": "向量是多少维的？", "session_id": first["session_id"], "strategy": "rag"},
        ).text
    )[-1][1]
    assert second["session_id"] == first["session_id"]
    detail = client.get(f"/api/sessions/{first['session_id']}").json()
    assert len(detail["messages"]) == 4


def test_ask_with_lexical_mode(client, sample_md):
    upload(client, "手册.md", sample_md.encode("utf-8"))
    events = parse_sse(
        client.post("/api/ask", json={"question": "余弦距离 阈值", "mode": "lexical", "strategy": "rag"}).text
    )
    assert events[-1][1]["fallback"] == "kb"


@pytest.mark.parametrize("question", ["分块默认块长是多少？"])
def test_ask_top_k_is_respected(client, sample_md, question):
    upload(client, "手册.md", sample_md.encode("utf-8"))
    events = parse_sse(client.post("/api/ask", json={"question": question, "top_k": 1}).text)
    citations = [data for kind, data in events if kind == "citation"]
    assert len(citations) <= 1
