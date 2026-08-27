"""Cold-start / serverless schema self-heal smoke tests.

Vercel's @vercel/python runtime does not reliably fire FastAPI startup events,
so tables and the built-in local account must be created lazily on first DB
access (see ragchat.db.ensure_db). These tests prove that path works against a
fresh SQLite database, including the real-world case where an existing
`messages` table was created by an OLDER app version and is missing columns the
current models expect (citations / eval_line / eval_data) -> which previously
made every chat query 500 with UndefinedColumn.

Runs with NO network and NO external DB (temp SQLite file).

Run:  .venv/Scripts/python -m pytest tests/test_cold_start.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text as _t

# Use an isolated temp SQLite DB; never touch Neon/.env.
_DB_PATH = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SESSION_SECRET"] = "test-secret"
for _k in ("PG_DATABASE_URL", "rag_gel_DATABASE_URL"):
    os.environ.pop(_k, None)


@pytest.fixture()
def client():
    # Import after env is set so ragchat.db picks up the temp SQLite engine.
    from ragchat import app as rapp
    from ragchat.db import engine, inspect

    # ensure_db() caches a module-global _initialized flag; reset it so each
    # test gets a genuine cold-start self-heal (mirrors a fresh Vercel process).
    import ragchat.db as _db
    _db._initialized = False

    # Fully fresh DB each test.
    for tbl in ("messages", "conversations", "users", "documents"):
        with engine.begin() as conn:
            conn.execute(_t(f"DROP TABLE IF EXISTS {tbl}"))
    yield TestClient(rapp.app, raise_server_exceptions=True)


def _message_columns():
    from ragchat.db import engine, inspect
    return {c["name"] for c in inspect(engine).get_columns("messages")}


def test_fresh_db_bootstraps_and_chat_works(client):
    """Fresh DB (no tables): first request self-heals schema + local account."""
    r = client.post("/api/auth/local-login")
    assert r.status_code == 200, r.text

    assert client.get("/api/chats").status_code == 200
    cid = client.post("/api/chats").json()["id"]
    ask = client.post(
        f"/api/chats/{cid}/ask",
        json={"question": "What is retrieval augmented generation?"},
    )
    assert ask.status_code == 200, ask.text


def test_column_drift_reconciled_on_existing_messages_table(client):
    """Reproduce the deployed failure: an OLD messages table missing columns.

    Before the fix this 500'd with UndefinedColumn (column messages.eval_data
    does not exist). After the fix, ensure_db reconciles the missing columns.
    """
    from ragchat.db import engine

    # Simulate the OLD schema: messages missing citations/eval_line/eval_data.
    with engine.begin() as conn:
        conn.execute(_t("DROP TABLE IF EXISTS messages"))
        conn.execute(_t("DROP TABLE IF EXISTS conversations"))
        conn.execute(
            _t("CREATE TABLE conversations ("
               "id VARCHAR PRIMARY KEY, user_id VARCHAR, title VARCHAR, created_at FLOAT)")
        )
        conn.execute(
            _t("CREATE TABLE messages ("
               "id VARCHAR PRIMARY KEY, conversation_id VARCHAR, "
               "role VARCHAR, content TEXT, created_at FLOAT)")
        )

    assert _message_columns() == {
        "id", "conversation_id", "role", "content", "created_at"
    }

    # Trigger the self-heal.
    assert client.post("/api/auth/local-login").status_code == 200

    # Columns must now be present (the core regression guard).
    assert "citations" in _message_columns()
    assert "eval_line" in _message_columns()
    assert "eval_data" in _message_columns()

    # And the chat flow must work end-to-end.
    assert client.get("/api/chats").status_code == 200
    cid = client.post("/api/chats").json()["id"]
    assert client.post(
        f"/api/chats/{cid}/ask",
        json={"question": "test"},
    ).status_code == 200
