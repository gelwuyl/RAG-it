"""Emptying a workspace has to empty the vector store too.

The Neon `chunks` table is the part that grows without anyone watching. A
document row is a few hundred bytes; its vectors are 768 floats per chunk. So
"delete" that removed rows and left chunks would look completely correct in the
UI while the database filled up with data nothing points at any more.

The other half is scope. This route is destructive and irreversible, so what it
does NOT touch is as much a part of the contract as what it does: conversations
are not embedded, cost no vector storage, and carry their citations inline —
deleting documents must not quietly take chat history with it.

No network: the vector store is stubbed and the calls it receives are recorded.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text as _t  # noqa: E402

_DB_PATH = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SESSION_SECRET"] = "test-secret"
for _k in ("PG_DATABASE_URL", "rag_gel_DATABASE_URL"):
    os.environ.pop(_k, None)


@pytest.fixture()
def rig(monkeypatch):
    from ragchat import app as rapp
    from ragchat import vectordb
    from ragchat.db import engine
    import ragchat.db as _db

    _db._initialized = False
    for tbl in ("messages", "conversations", "users", "documents", "folders"):
        with engine.begin() as conn:
            conn.execute(_t(f"DROP TABLE IF EXISTS {tbl}"))

    cleared: list[list[str]] = []
    monkeypatch.setattr(vectordb, "delete_users_chunks",
                        lambda ids: cleared.append(list(ids)) or 0)

    c = TestClient(rapp.app, raise_server_exceptions=True)
    c.post("/api/auth/local-login")
    uid = c.get("/api/auth/status").json()["user"]["id"]
    return c, uid, cleared


def _seed(user_id: str, n_docs: int = 3, n_folders: int = 1) -> None:
    from ragchat.db import Document, FolderSource, SessionLocal

    s = SessionLocal()
    for i in range(n_docs):
        s.add(Document(user_id=user_id, source_type="upload", title=f"doc{i}.md",
                       status="ready", n_chunks=4, size_bytes=10))
    for i in range(n_folders):
        s.add(FolderSource(user_id=user_id, path=f"/tmp/f{i}"))
    s.commit()
    s.close()


def _counts(user_id: str) -> dict:
    from ragchat.db import Conversation, Document, FolderSource, Message, SessionLocal

    s = SessionLocal()
    out = {
        "documents": s.query(Document).filter(Document.user_id == user_id).count(),
        "folders": s.query(FolderSource).filter(FolderSource.user_id == user_id).count(),
        "conversations": s.query(Conversation).filter(
            Conversation.user_id == user_id).count(),
        "messages": s.query(Message).count(),
    }
    s.close()
    return out


# --- it empties what it says it empties -----------------------------------

def test_it_removes_every_document_and_folder(rig):
    client, uid, _ = rig
    _seed(uid)
    assert _counts(uid)["documents"] == 3

    r = client.delete("/api/documents")
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "documents": 3, "folders": 1}
    assert _counts(uid)["documents"] == 0
    assert _counts(uid)["folders"] == 0


def test_the_vectors_go_with_them(rig):
    """The half that grows. Rows are bytes; vectors are 768 floats per chunk,
    and orphaned ones are invisible to everything except the prune command."""
    client, uid, cleared = rig
    _seed(uid)
    client.delete("/api/documents")
    assert cleared == [[uid]], (
        "the vector store was not cleared — the UI would look empty while Neon "
        "kept every embedding"
    )


def test_vectors_are_cleared_before_the_rows(rig, monkeypatch):
    """Rows without vectors is recoverable: re-index rebuilds it. Vectors
    without rows is orphaned data nothing points at."""
    from ragchat import vectordb
    from ragchat.db import Document, SessionLocal

    client, uid, _ = rig
    _seed(uid)
    seen = {}

    def _spy(ids):
        s = SessionLocal()
        seen["docs_at_vector_time"] = s.query(Document).filter(
            Document.user_id == uid).count()
        s.close()
        return 0

    monkeypatch.setattr(vectordb, "delete_users_chunks", _spy)
    client.delete("/api/documents")
    assert seen["docs_at_vector_time"] == 3


# --- and leaves alone what it says it leaves ------------------------------

def test_conversations_survive(rig):
    """"Delete all" in a pane called Sources must not be read as taking the
    chats. They are not embedded and they cost no vector storage."""
    client, uid, _ = rig
    _seed(uid)
    cid = client.post("/api/chats").json()["id"]
    assert _counts(uid)["conversations"] == 1

    client.delete("/api/documents")
    assert _counts(uid)["conversations"] == 1
    assert client.get(f"/api/chats/{cid}").status_code == 200


def test_another_users_workspace_is_untouched(rig):
    from ragchat.db import Document, SessionLocal

    client, uid, cleared = rig
    _seed(uid)
    _seed("somebody-else", n_docs=2, n_folders=0)

    client.delete("/api/documents")
    s = SessionLocal()
    theirs = s.query(Document).filter(Document.user_id == "somebody-else").count()
    s.close()
    assert theirs == 2
    assert cleared == [[uid]], "the vector clear was not scoped to one user"


# --- who may press it ------------------------------------------------------

def test_a_guest_may_not(rig, monkeypatch):
    """Their two demo documents were vector-copied and they cannot get them
    back without a re-seed — and the workspace throws itself away in 30
    minutes anyway."""
    from ragchat import app as rapp

    client, uid, _ = rig
    _seed(uid)
    monkeypatch.setattr(rapp.guests, "is_guest", lambda u: True)
    r = client.delete("/api/documents")
    assert r.status_code == 403, r.text
    assert _counts(uid)["documents"] == 3


def test_an_empty_workspace_is_not_an_error(rig):
    """The button is disabled-by-emptiness in the UI, but the route is the one
    that has to be safe."""
    client, _, cleared = rig
    r = client.delete("/api/documents")
    assert r.status_code == 200
    assert r.json()["documents"] == 0
