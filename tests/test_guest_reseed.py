"""A returning guest must not be left holding unreachable documents.

Chunks are only retrievable under the config fingerprint they were written
with. So when the embedding model changes, a guest workspace seeded before the
change keeps listing its two sample documents as READY, with chunk counts —
and every question answers "I couldn't find this in your documents". The
workspace looks perfect and is empty underneath.

It was worse than a cosmetic lie: a guest cannot re-index (account-only route),
so the only remedy was to wait out the 30-minute reap. This is the first thing a
visitor sees.

Two things had to be true and neither was:

  * `guest-seed` was idempotent on EXISTENCE — it counted demo rows and
    returned "already seeded" without ever looking at the fingerprint;
  * nothing on the boot path called it anyway, because the client seeds only
    for a visitor who is not yet authenticated, and a returning guest already
    is.

So the server reports staleness on /api/auth/status and the client acts on it.

No network: the vector store is stubbed, since what is under test is the
detect-and-repair contract rather than embedding.
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

STALE_FP = "0000staleoldfp"


@pytest.fixture()
def client(monkeypatch):
    from ragchat import app as rapp
    from ragchat.db import engine
    import ragchat.db as _db

    _db._initialized = False
    for tbl in ("messages", "conversations", "users", "documents", "folders"):
        with engine.begin() as conn:
            conn.execute(_t(f"DROP TABLE IF EXISTS {tbl}"))

    # Seeding copies vectors between users. Neither backend is available here,
    # and neither is what these tests are about.
    monkeypatch.setattr(rapp.guests, "seed_demo_corpus", lambda db, guest: 2)
    monkeypatch.setattr(rapp, "delete_document_chunks", lambda *a, **k: 0)
    yield TestClient(rapp.app, raise_server_exceptions=True)


def _become_guest(client):
    r = client.post("/api/auth/guest-login")
    assert r.status_code == 200, r.text
    return client.get("/api/auth/status").json()


def _give_demo_docs(user_id: str, fingerprint: str) -> None:
    from ragchat.db import Document, SessionLocal

    s = SessionLocal()
    for title in ("helios_energy_handbook.md", "meridian_coffee_ops.md"):
        s.add(Document(user_id=user_id, source_type="upload", title=title,
                       status="ready", is_demo=True, n_chunks=1,
                       config_fingerprint=fingerprint, size_bytes=10))
    s.commit()
    s.close()


def _current_fp() -> str:
    from ragchat.config import load_config

    return load_config().fingerprint()


def _demo_count(user_id: str) -> int:
    from ragchat.db import Document, SessionLocal

    s = SessionLocal()
    n = s.query(Document).filter(
        Document.user_id == user_id, Document.is_demo.is_(True)
    ).count()
    s.close()
    return n


# --- the server has to notice ---------------------------------------------

def test_a_stale_workspace_is_reported_as_needing_a_reseed(client):
    status = _become_guest(client)
    _give_demo_docs(status["user"]["id"], STALE_FP)

    assert client.get("/api/auth/status").json()["demo_needs_reseed"] is True


def test_a_current_workspace_is_not(client):
    status = _become_guest(client)
    _give_demo_docs(status["user"]["id"], _current_fp())

    assert client.get("/api/auth/status").json()["demo_needs_reseed"] is False


def test_an_empty_workspace_is_not_stale(client):
    """Nothing to repair is not the same as something broken; saying otherwise
    would re-seed on every boot before the first copy has landed."""
    _become_guest(client)
    assert client.get("/api/auth/status").json()["demo_needs_reseed"] is False


# --- and the repair has to actually run -----------------------------------

def test_reseeding_replaces_the_stale_documents(client):
    status = _become_guest(client)
    uid = status["user"]["id"]
    _give_demo_docs(uid, STALE_FP)
    assert _demo_count(uid) == 2

    r = client.post("/api/auth/guest-seed")
    assert r.status_code == 200, r.text
    # The stale rows are gone. Leaving them would hand the visitor two of each
    # sample document: one that answers and one that cannot.
    assert _demo_count(uid) == 0 or all(
        fp == _current_fp() for fp in _fingerprints(uid)
    )


def _fingerprints(user_id: str) -> list[str]:
    from ragchat.db import Document, SessionLocal

    s = SessionLocal()
    out = [
        d.config_fingerprint
        for d in s.query(Document).filter(
            Document.user_id == user_id, Document.is_demo.is_(True)
        )
    ]
    s.close()
    return out


def test_a_current_workspace_is_not_reseeded(client, monkeypatch):
    """Re-copying on every call would put the corpus back in front of a visitor
    who already has it, on every page load."""
    from ragchat import app as rapp

    status = _become_guest(client)
    _give_demo_docs(status["user"]["id"], _current_fp())

    calls = {"n": 0}

    def _counting(db, guest):
        calls["n"] += 1
        return 2

    monkeypatch.setattr(rapp.guests, "seed_demo_corpus", _counting)
    body = client.post("/api/auth/guest-seed").json()
    assert calls["n"] == 0
    assert body["reason"] == "already seeded"


def test_the_client_repairs_a_returning_guest():
    """The seeding call on the boot path runs only for a visitor who is NOT yet
    authenticated. A returning guest already is, so without this branch nothing
    would ever notice."""
    js = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "demo_needs_reseed" in js, (
        "the client no longer reacts to a stale workspace — a returning guest "
        "would see two READY documents that answer nothing"
    )
