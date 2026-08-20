"""The Neon schema check runs once per process, not once per call.

_ensure_table is called from all eight store operations and does six round
trips: CREATE EXTENSION, a catalog query, SQLAlchemy's reflecting create_all,
and two CREATE INDEX statements. Every one is a network hop. A guest sign-in
performs three store operations, so it paid eighteen — measured as the bulk of
a 14.3s demo-corpus seed inside a request with a 10 second budget.

Imported without psycopg2 or a database: only the memoisation is under test.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

pytest.importorskip("pgvector", reason="Neon backend deps not installed")

from ragchat import store_neon as sn  # noqa: E402


class _Conn:
    """Counts statements instead of running them."""

    def __init__(self):
        self.executed = []

    def execute(self, stmt, *a, **k):
        self.executed.append(str(stmt))

        class _R:
            rowcount = 0

            def mappings(self):
                return self

            def all(self):
                return []

            def scalar(self):
                return None

            def first(self):
                return None
        return _R()


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(sn, "_SCHEMA_READY_FOR_DIM", None)
    monkeypatch.setattr(sn, "_live_embedding_dim", lambda conn: None)
    monkeypatch.setattr(sn._metadata, "create_all", lambda conn: None)
    yield


def test_the_schema_is_prepared_on_the_first_call():
    c = _Conn()
    sn._ensure_table(c)
    assert c.executed, "the first call must actually prepare the schema"


def test_and_skipped_on_every_call_after_that():
    first, second = _Conn(), _Conn()
    sn._ensure_table(first)
    sn._ensure_table(second)
    assert second.executed == [], (
        f"schema DDL re-ran: {second.executed}"
    )


def test_a_failure_part_way_is_not_remembered_as_success(monkeypatch):
    """Otherwise the process believes in a schema it never finished building."""
    class _Boom(_Conn):
        def execute(self, stmt, *a, **k):
            raise RuntimeError("connection lost")

    with pytest.raises(RuntimeError):
        sn._ensure_table(_Boom())
    assert sn._SCHEMA_READY_FOR_DIM is None

    ok = _Conn()
    sn._ensure_table(ok)
    assert ok.executed, "a failed attempt blocked the retry"


def test_a_changed_target_dimension_re_runs_the_check(monkeypatch):
    """The drift check is the reason this is keyed by dimension and not a bool."""
    sn._ensure_table(_Conn())
    monkeypatch.setattr(sn, "_target_dim", lambda: 1536)
    after = _Conn()
    sn._ensure_table(after)
    assert after.executed, "a dimension change skipped the drift check"


def test_the_vector_store_shares_the_app_engine_when_it_is_the_same_database(monkeypatch):
    """Two pools to one Neon database halves how often either connection gets
    reused, so the vector pool — touched only by uploads, deletes and asks —
    routinely handed out one that had aged past pool_recycle and paid a fresh
    TLS handshake. That was most of a 2.7s document delete."""
    import ragchat.db as _db

    monkeypatch.setattr(sn, "_engine", None)
    monkeypatch.setattr(sn.settings, "pg_url", "postgresql://same/db")
    monkeypatch.setattr(sn.settings, "db_url", "postgresql://same/db")
    assert sn._get_engine() is _db.engine


def test_it_keeps_its_own_engine_when_the_databases_differ(monkeypatch):
    """A separate PG_DATABASE_URL exists to point the vector store somewhere
    else; sharing then would silently send chunks to the wrong server."""
    import ragchat.db as _db

    monkeypatch.setattr(sn, "_engine", None)
    monkeypatch.setattr(sn.settings, "pg_url", "postgresql://vectors/db")
    monkeypatch.setattr(sn.settings, "db_url", "postgresql://app/db")
    made = {}
    monkeypatch.setattr(sn, "create_engine", lambda url, **k: made.setdefault("url", url) or "ENGINE")
    assert sn._get_engine() != _db.engine
    assert made["url"] == "postgresql://vectors/db"


def test_the_delete_path_gets_its_own_index():
    """Deleting a document filters on (user_id, doc_id). The other index starts
    with user_id and then goes to columns the delete does not filter on, so it
    scanned every chunk the user owns."""
    import inspect

    src = inspect.getsource(sn._ensure_table)
    assert "chunks_user_doc_idx" in src
    assert "(user_id, doc_id)" in src
