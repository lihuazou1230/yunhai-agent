"""会话存储测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.errors import SessionNotFound
from app.sessions import SessionStore


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions.db")


def test_create_and_list_sessions(store: SessionStore):
    created = store.create_session("分块多大？")
    sessions = store.list_sessions()
    assert [s["id"] for s in sessions] == [created["id"]]
    assert sessions[0]["message_count"] == 0
    assert sessions[0]["title"] == "分块多大？"


def test_add_message_and_read_detail(store: SessionStore):
    session = store.create_session("阈值")
    store.add_message(session["id"], "user", "阈值是多少？")
    store.add_message(
        session["id"],
        "assistant",
        "0.45 [1]",
        citations=[{"index": 1, "source": "笔记.md"}],
        meta={"fallback": "kb"},
    )
    detail = store.get_session(session["id"])
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["citations"][0]["source"] == "笔记.md"
    assert detail["messages"][1]["meta"]["fallback"] == "kb"
    # 更新过会话后 message_count 跟着变
    assert store.list_sessions()[0]["message_count"] == 2


def test_ensure_session_reuses_existing_and_creates_by_first_question(store: SessionStore):
    existing = store.create_session("老的")
    assert store.ensure_session(existing["id"], "新问题", "semantic") == existing["id"]
    created = store.ensure_session(None, "第一句话就是标题来源很长很长", "lexical")
    assert created != existing["id"]
    assert store.get_session(created)["title"] == "第一句话就是标题来源很长很长"[:20]


def test_ensure_session_ignores_unknown_id(store: SessionStore):
    created = store.ensure_session("不存在的会话", "问题", "semantic")
    assert store.get_session(created)["id"] == created


def test_delete_session_removes_messages(store: SessionStore):
    session = store.create_session("要删掉的")
    store.add_message(session["id"], "user", "问题")
    store.delete_session(session["id"])
    assert store.list_sessions() == []
    with pytest.raises(SessionNotFound):
        store.get_session(session["id"])


def test_missing_session_raises(store: SessionStore):
    with pytest.raises(SessionNotFound):
        store.get_session("nope")


def test_corrupted_json_columns_do_not_crash(store: SessionStore):
    session = store.create_session("坏数据")
    store.add_message(session["id"], "assistant", "内容")
    # 直接写坏 citations 字段（模拟历史遗留/手工改库）
    store._conn.execute("update messages set citations = 'not-json' where session_id = ?", (session["id"],))
    store._conn.commit()
    detail = store.get_session(session["id"])
    assert detail["messages"][0]["citations"] == []
