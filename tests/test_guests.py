"""Guest workspace lifecycle.

Guests are the only accounts the app deletes on its own, so the reaper gets the
most attention here: it must destroy idle guest workspaces and nothing else. A
bug that widened its WHERE clause would silently delete real users' documents.

Runs against a temp SQLite DB with a stubbed vector store — no network.

Run:  .venv/Scripts/python -m pytest tests/test_guests.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

_DB_PATH = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SESSION_SECRET"] = "test-secret"
for _k in ("PG_DATABASE_URL", "rag_gel_DATABASE_URL"):
    os.environ.pop(_k, None)

from ragchat import guests as _guests  # noqa: E402
from ragchat.db import (  # noqa: E402
    Conversation,
    Document,
    Message,
    SessionLocal,
    User,
    init_db,
)


@pytest.fixture(autouse=True)
def clean_db(monkeypatch):
    init_db()
    db = SessionLocal()
    for model in (Message, Conversation, Document, User):
        db.query(model).delete()
    db.commit()
    # The vector store is exercised elsewhere; here it must not be touched at all.
    monkeypatch.setattr("ragchat.vectordb.delete_document_chunks", lambda *a, **k: None)
    yield db
    db.close()


def _guest(db, *, idle_seconds: float) -> User:
    g = User(provider="guest", sub=f"guest-{idle_seconds}", name="Guest",
             last_seen_at=time.time() - idle_seconds)
    db.add(g)
    db.commit()
    return g


def _doc(db, user, title="f.txt", is_demo=False) -> Document:
    d = Document(user_id=user.id, source_type="upload", title=title,
                 size_bytes=1024, is_demo=is_demo)
    db.add(d)
    db.commit()
    return d


def test_reaper_deletes_only_idle_guests(clean_db):
    db = clean_db
    stale = _guest(db, idle_seconds=_guests.GUEST_IDLE_TTL_SECONDS + 60)
    active = _guest(db, idle_seconds=60)
    _doc(db, stale)
    _doc(db, active)

    assert _guests.reap_stale_guests(db) == 1
    remaining = {u.id for u in db.query(User).all()}
    assert stale.id not in remaining, "idle guest should be gone"
    assert active.id in remaining, "a guest active a minute ago must survive"
    # The idle guest's documents go with it; the active guest keeps theirs.
    assert db.query(Document).filter(Document.user_id == stale.id).count() == 0
    assert db.query(Document).filter(Document.user_id == active.id).count() == 1


def test_reaper_never_touches_real_accounts(clean_db):
    db = clean_db
    # A signed-in account that has not been seen for a year must be untouched:
    # only guests are ephemeral, and last_seen_at is written for everyone.
    real = User(provider="google", sub="g-1", email="a@b.c", name="Real",
                last_seen_at=time.time() - 365 * 24 * 3600)
    db.add(real)
    db.commit()
    _doc(db, real)

    assert _guests.reap_stale_guests(db) == 0
    assert db.get(User, real.id) is not None
    assert db.query(Document).filter(Document.user_id == real.id).count() == 1


def test_demo_documents_do_not_consume_the_upload_allowance(clean_db):
    db = clean_db
    g = _guest(db, idle_seconds=0)
    for i in range(2):
        _doc(db, g, title=f"demo{i}.md", is_demo=True)
    # Seeded demo content is supplied by the app; charging it to the visitor's
    # quota left them a single usable slot.
    assert _guests.upload_allowance(db, g, 1024) is None

    for i in range(_guests.GUEST_MAX_DOCUMENTS):
        _doc(db, g, title=f"own{i}.txt")
    denied = _guests.upload_allowance(db, g, 1024)
    assert denied and "sign in" in denied.lower()


def test_byte_cap_is_enforced(clean_db):
    db = clean_db
    g = _guest(db, idle_seconds=0)
    big = Document(user_id=g.id, source_type="upload", title="big.txt",
                   size_bytes=_guests.GUEST_MAX_UPLOAD_BYTES - 100)
    db.add(big)
    db.commit()
    assert _guests.upload_allowance(db, g, 50) is None
    assert _guests.upload_allowance(db, g, 5000) is not None


def test_signed_in_accounts_are_never_capped(clean_db):
    db = clean_db
    real = User(provider="google", sub="g-2", name="Real", last_seen_at=time.time())
    db.add(real)
    db.commit()
    for i in range(_guests.GUEST_MAX_DOCUMENTS + 5):
        _doc(db, real, title=f"d{i}.txt")
    assert _guests.upload_allowance(db, real, 10 * 1024 * 1024) is None


def test_promote_moves_work_and_retires_the_guest(clean_db, monkeypatch):
    db = clean_db
    monkeypatch.setattr("ragchat.vectordb.reassign_user_chunks", lambda *a, **k: 0)
    g = _guest(db, idle_seconds=0)
    target = User(provider="google", sub="g-3", name="Real", last_seen_at=time.time())
    db.add(target)
    db.commit()
    _doc(db, g, title="kept.txt")
    conv = Conversation(user_id=g.id)
    db.add(conv)
    db.commit()

    moved = _guests.promote_guest(db, g, target)
    assert moved["documents"] == 1 and moved["conversations"] == 1
    assert db.query(Document).filter(Document.user_id == target.id).count() == 1
    assert db.query(Conversation).filter(Conversation.user_id == target.id).count() == 1
    assert db.get(User, g.id) is None, "the guest row exists only to own data"


def test_purge_removes_messages_with_their_conversation(clean_db):
    db = clean_db
    g = _guest(db, idle_seconds=0)
    conv = Conversation(user_id=g.id)
    db.add(conv)
    db.commit()
    db.add(Message(conversation_id=conv.id, role="user", content="hi"))
    db.commit()

    summary = _guests.purge_user_data(db, g, drop_user=True)
    assert summary["messages"] == 1
    assert db.query(Message).count() == 0
    assert db.query(Conversation).count() == 0
