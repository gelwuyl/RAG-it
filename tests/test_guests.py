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
from ragchat.config import load_config  # noqa: E402
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


class _FakeVectorStore:
    """Stands in for whichever backend is configured, recording chunks per
    (user, doc) so a test can assert that a seeded document really has vectors.

    Both real backends can fail a copy in two distinct ways, and both matter
    here: raising (Chroma losing a collection mid-iteration, pgvector dropping
    the connection) and quietly copying nothing (the source rows are gone).
    """

    def __init__(self):
        self.chunks: dict[tuple[str, str], int] = {}
        self.raise_for: set[str] = set()  # src doc ids whose copy explodes
        self.zero_for: set[str] = set()   # src doc ids whose copy finds nothing
        self.unembeddable: set[str] = set()  # titles the provider refuses

    def ingest(self, user_id, doc_id, title, text, cfg):
        if title in self.unembeddable:
            raise RuntimeError("embedding provider is out of quota")
        n = max(1, len(text) // 400)
        self.chunks[(user_id, doc_id)] = n
        return n

    def copy(self, src_user, src_doc, dst_user, dst_doc):
        if src_doc in self.raise_for:
            raise RuntimeError("vector store unavailable")
        if src_doc in self.zero_for:
            return 0
        n = self.chunks.get((src_user, src_doc), 0)
        if n:
            self.chunks[(dst_user, dst_doc)] = n
        return n

    def delete(self, user_id, doc_id, embedding_model=None):
        self.chunks.pop((user_id, doc_id), None)

    def count(self, user_id, doc_id) -> int:
        return self.chunks.get((user_id, doc_id), 0)


@pytest.fixture
def vectors(monkeypatch) -> _FakeVectorStore:
    store = _FakeVectorStore()
    monkeypatch.setattr("ragchat.pipeline.ingest_document_text", store.ingest)
    monkeypatch.setattr("ragchat.vectordb.copy_user_chunks", store.copy)
    monkeypatch.setattr("ragchat.vectordb.delete_document_chunks", store.delete)
    return store


def _seeded(db, guest) -> dict[str, Document]:
    return {
        d.title: d
        for d in db.query(Document).filter(Document.user_id == guest.id).all()
        if d.is_demo
    }


def test_seeding_gives_a_guest_every_demo_file_with_its_vectors(clean_db, vectors):
    """The regression this file exists for: a guest used to land with one of the
    two demo files and no chunks behind it, because the clone row was committed
    before its vectors were copied."""
    db = clean_db
    guest = _guests.create_guest(db)
    assert _guests.seed_demo_corpus(db, guest) == len(_guests.DEMO_CORPUS_FILES)

    docs = _seeded(db, guest)
    assert set(docs) == set(_guests.DEMO_CORPUS_FILES)
    for title, doc in docs.items():
        assert doc.status == "ready", title
        # The row claiming chunks is not the same as the store holding them.
        assert doc.n_chunks > 0, title
        assert vectors.count(guest.id, doc.id) > 0, f"{title} has no vectors"


def test_a_copy_that_raises_never_leaves_a_chunkless_document(clean_db, vectors):
    """A failure on ONE file must cost the guest only that file — and must not
    leave its row behind, since a demo document with no vectors answers nothing
    and cannot be told apart from a working one in the UI."""
    db = clean_db
    template = _guests.ensure_demo_template(db, load_config())
    first, second = _guests.DEMO_CORPUS_FILES
    doomed = (
        db.query(Document)
        .filter(Document.user_id == template.id, Document.title == first)
        .one()
    )
    vectors.raise_for.add(doomed.id)

    guest = _guests.create_guest(db)
    assert _guests.seed_demo_corpus(db, guest) == 1

    docs = _seeded(db, guest)
    assert first not in docs, "a document whose vectors failed must be rolled back"
    assert second in docs, "one file's failure must not abort the rest of the loop"
    assert vectors.count(guest.id, docs[second].id) > 0


def test_a_copy_that_moves_no_chunks_counts_as_a_failure(clean_db, vectors):
    """Copying zero chunks raises nothing, but produces exactly the same broken
    state as a throw: a document the retriever can never return."""
    db = clean_db
    template = _guests.ensure_demo_template(db, load_config())
    first = _guests.DEMO_CORPUS_FILES[0]
    empty = (
        db.query(Document)
        .filter(Document.user_id == template.id, Document.title == first)
        .one()
    )
    vectors.zero_for.add(empty.id)

    guest = _guests.create_guest(db)
    assert _guests.seed_demo_corpus(db, guest) == len(_guests.DEMO_CORPUS_FILES) - 1
    assert first not in _seeded(db, guest)


def test_a_file_that_cannot_be_embedded_costs_only_itself(clean_db, vectors):
    """An embedding failure on one demo file used to propagate out of
    seed_demo_corpus and leave the arriving visitor with a completely empty
    workspace. It should cost them that file and nothing more — and it must not
    leave a `failed` template row to be cloned, since it has no vectors."""
    db = clean_db
    first, second = _guests.DEMO_CORPUS_FILES
    vectors.unembeddable.add(first)

    guest = _guests.create_guest(db)
    assert _guests.seed_demo_corpus(db, guest) == 1

    docs = _seeded(db, guest)
    assert first not in docs
    assert second in docs and vectors.count(guest.id, docs[second].id) > 0

    # And once the provider recovers, the next visitor gets the full corpus
    # without anyone having to clear the failed row by hand.
    vectors.unembeddable.clear()
    later = _guests.create_guest(db)
    assert _guests.seed_demo_corpus(db, later) == len(_guests.DEMO_CORPUS_FILES)
    assert set(_seeded(db, later)) == set(_guests.DEMO_CORPUS_FILES)


def test_seeding_only_ever_hands_over_the_two_synthetic_files(clean_db, vectors):
    """The rest of eval/corpus is real business content and must never reach an
    anonymous visitor."""
    db = clean_db
    guest = _guests.create_guest(db)
    _guests.seed_demo_corpus(db, guest)
    assert set(_seeded(db, guest)) <= set(_guests.DEMO_CORPUS_FILES)


def test_reaper_spares_the_demo_template(clean_db, vectors):
    """The template is a guest-provider row that nothing ever touches, so it
    looks permanently idle. Reaping it deletes the corpus every visitor is
    seeded from and bills the next visitor for a re-embed."""
    db = clean_db
    template = _guests.ensure_demo_template(db, load_config())
    template.last_seen_at = time.time() - _guests.GUEST_IDLE_TTL_SECONDS - 60
    db.commit()
    stale = _guest(db, idle_seconds=_guests.GUEST_IDLE_TTL_SECONDS + 60)

    # The exemption must be narrow: ordinary idle guests still go.
    assert _guests.reap_stale_guests(db) == 1
    assert db.get(User, stale.id) is None
    survivor = _guests._demo_template(db)
    assert survivor is not None, "the demo template must outlive the reaper"
    assert (
        db.query(Document).filter(Document.user_id == survivor.id).count()
        == len(_guests.DEMO_CORPUS_FILES)
    )


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
