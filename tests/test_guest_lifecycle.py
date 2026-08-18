"""Guest workspaces expire; signed-in ones never do.

Everything here guards a path that deletes data, where the failure mode is
silent and the loser is a visitor who cannot report it. Four in particular:

- the sweep endpoint must be authenticated, and DISABLED rather than open when
  no secret is configured;
- the reaper must never take a signed-in account or the demo template;
- the close beacon must not delete, because `pagehide` fires on a reload;
- promotion must survive the beacon, or signing in destroys the work signing in
  is supposed to keep.

Runs against a temp SQLite DB with no network.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_DB_PATH = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SESSION_SECRET"] = "test-secret"
for _k in ("PG_DATABASE_URL", "rag_gel_DATABASE_URL"):
    os.environ.pop(_k, None)

SWEEP_SECRET = "sweep-me-please"


@pytest.fixture()
def db(monkeypatch):
    import ragchat.db as _db

    _db._initialized = False
    _db.init_db()
    # No vector store in these tests: the relational side is what is under test,
    # and the bulk chunk delete has its own coverage in the store modules.
    monkeypatch.setattr("ragchat.vectordb.delete_users_chunks", lambda ids: 0)
    s = _db.SessionLocal()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def client(monkeypatch):
    from ragchat import app as rapp
    import ragchat.db as _db

    _db._initialized = False
    _db.init_db()
    monkeypatch.setattr("ragchat.vectordb.delete_users_chunks", lambda ids: 0)
    monkeypatch.setattr("ragchat.vectordb.delete_document_chunks", lambda *a, **k: None)
    monkeypatch.setattr(rapp.settings, "sweep_secret", SWEEP_SECRET)
    with TestClient(rapp.app, raise_server_exceptions=True) as c:
        yield c


def _guest(db, *, idle_seconds: float = 0.0):
    from ragchat.db import User
    from ragchat import guests

    u = User(provider=guests.GUEST_PROVIDER, sub=f"guest-{os.urandom(4).hex()}",
             name="Guest", last_seen_at=time.time() - idle_seconds)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


# --- the TTL ---------------------------------------------------------------

def test_ttl_is_thirty_minutes():
    from ragchat import guests

    # The sweeper runs every 15 minutes, so the TTL has to be a comfortable
    # multiple of that or "30 minutes" is really "30 to 45".
    assert guests.GUEST_IDLE_TTL_SECONDS == 30 * 60


def test_an_idle_guest_is_reaped(db):
    from ragchat import guests

    stale = _guest(db, idle_seconds=guests.GUEST_IDLE_TTL_SECONDS + 60)
    assert guests.reap_stale_guests(db) == 1
    from ragchat.db import User
    assert db.query(User).filter(User.id == stale.id).first() is None


def test_a_guest_inside_the_ttl_is_left_alone(db):
    from ragchat import guests
    from ragchat.db import User

    fresh = _guest(db, idle_seconds=60)
    guests.reap_stale_guests(db)
    assert db.query(User).filter(User.id == fresh.id).first() is not None


def test_a_signed_in_account_is_never_reaped_however_idle(db):
    from ragchat import guests
    from ragchat.db import User

    acct = User(provider="password", sub=f"real-{os.urandom(4).hex()}",
                name="Real", last_seen_at=time.time() - 365 * 24 * 3600)
    db.add(acct)
    db.commit()
    guests.reap_stale_guests(db)
    assert db.query(User).filter(User.id == acct.id).first() is not None


def test_the_demo_template_is_never_reaped(db):
    """Nothing ever calls touch() on it, so it looks idle from birth. Reaping it
    deletes the corpus every visitor is seeded from and bills the next arrival
    for a re-embed."""
    from ragchat import guests
    from ragchat.db import User

    tpl = User(provider=guests.GUEST_PROVIDER, sub=guests.DEMO_TEMPLATE_SUB,
               name="Demo corpus",
               last_seen_at=time.time() - guests.GUEST_IDLE_TTL_SECONDS * 10)
    db.add(tpl)
    db.commit()
    guests.reap_stale_guests(db)
    assert db.query(User).filter(User.id == tpl.id).first() is not None


# --- the close beacon ------------------------------------------------------

def test_the_beacon_back_dates_and_does_not_delete(db):
    """pagehide fires on a reload as readily as on a close. Deleting here would
    destroy a workspace the visitor is about to come straight back to."""
    from ragchat import guests
    from ragchat.db import User

    g = _guest(db)
    guests.back_date(db, g)
    assert db.query(User).filter(User.id == g.id).first() is not None
    # Inside the TTL still, so a visitor who returns finds their work; past it
    # by the next sweep if they do not.
    idle = time.time() - g.last_seen_at
    assert 0 < idle < guests.GUEST_IDLE_TTL_SECONDS


def test_a_back_dated_guest_is_collected_by_the_next_sweep(db):
    from ragchat import guests
    from ragchat.db import User

    g = _guest(db)
    guests.back_date(db, g)
    # One sweep interval later.
    g.last_seen_at -= 15 * 60
    db.commit()
    guests.reap_stale_guests(db)
    assert db.query(User).filter(User.id == g.id).first() is None


def test_returning_after_the_beacon_restores_a_full_life(db):
    from ragchat import guests

    g = _guest(db)
    guests.back_date(db, g)
    g.last_seen_at = 0.0     # defeat touch()'s write throttle
    db.commit()
    guests.touch(db, g)
    assert time.time() - g.last_seen_at < 5


def test_the_beacon_does_nothing_to_a_signed_in_account(db):
    from ragchat import guests
    from ragchat.db import User

    acct = User(provider="password", sub=f"real-{os.urandom(4).hex()}",
                name="Real", last_seen_at=time.time())
    db.add(acct)
    db.commit()
    before = acct.last_seen_at
    guests.back_date(db, acct)
    assert acct.last_seen_at == before


def test_the_beacon_route_answers_without_a_session(client):
    """It is fired from a page being torn down. It must never 500 or 401."""
    client.cookies.clear()
    assert client.post("/api/auth/leaving").status_code == 204


# --- the sweep endpoint ----------------------------------------------------

def test_sweep_rejects_a_missing_secret(client):
    assert client.post("/api/admin/sweep-guests").status_code == 403


def test_sweep_rejects_a_wrong_secret(client):
    r = client.post("/api/admin/sweep-guests",
                    headers={"x-sweep-secret": "not-it"})
    assert r.status_code == 403


def test_sweep_is_disabled_not_open_when_no_secret_is_configured(client, monkeypatch):
    """The dangerous default. An unset secret must close the route, never turn
    it into an unauthenticated deletion endpoint on a public repo."""
    from ragchat import app as rapp

    monkeypatch.setattr(rapp.settings, "sweep_secret", "")
    assert client.post("/api/admin/sweep-guests").status_code == 404
    r = client.post("/api/admin/sweep-guests", headers={"x-sweep-secret": ""})
    assert r.status_code == 404


def test_sweep_with_the_secret_reports_what_it_did(client):
    r = client.post("/api/admin/sweep-guests",
                    headers={"x-sweep-secret": SWEEP_SECRET})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"reaped", "hit_limit", "idle_ttl_seconds"}
    assert body["idle_ttl_seconds"] == 30 * 60


# --- guest-login cost ------------------------------------------------------

def test_guest_login_reaps_at_most_two_workspaces_inline(db, monkeypatch):
    """The acceptance criterion is a guest-login under 10s. It was 39.7s, and
    the reason was twenty full workspace deletions in front of a visitor."""
    from ragchat import guests

    assert guests.INLINE_REAP_LIMIT == 2
    # Track OUR ids: the tests share one database, so counting stale rows
    # globally would fold in whatever an earlier test left behind.
    mine = [_guest(db, idle_seconds=guests.GUEST_IDLE_TTL_SECONDS + 60).id
            for _ in range(5)]

    seen = {}
    real = guests.reap_stale_guests

    def spy(session, *, limit=200):
        seen["limit"] = limit
        return real(session, limit=limit)

    monkeypatch.setattr(guests, "reap_stale_guests", spy)
    guests.create_guest(db)
    assert seen["limit"] == guests.INLINE_REAP_LIMIT

    from ragchat.db import User
    left = db.query(User).filter(User.id.in_(mine)).count()
    assert left == 3, "the inline reap took more than its backstop share"


def test_purge_is_set_based_not_per_row(db, monkeypatch):
    """One vector-store call for the whole sweep, not one per document.

    Per-document deletes were most of the 39.7s: on Neon each one is a network
    round trip, and a sweep of twenty workspaces made hundreds of them.
    """
    from ragchat import guests
    from ragchat.db import Document

    calls = []
    monkeypatch.setattr("ragchat.vectordb.delete_users_chunks",
                        lambda ids: calls.append(list(ids)) or 0)

    victims = [_guest(db, idle_seconds=guests.GUEST_IDLE_TTL_SECONDS + 60)
               for _ in range(3)]
    for g in victims:
        for i in range(4):
            db.add(Document(user_id=g.id, source_type="upload",
                            title=f"d{i}", size_bytes=10))
    db.commit()

    ids = [g.id for g in victims]
    guests.purge_users(db, victims, drop_users=True)
    assert len(calls) == 1, f"expected one bulk call, got {len(calls)}"
    assert sorted(calls[0]) == sorted(ids)
    # Scoped to these users: the test files share one database, so a global
    # count would fold in rows an earlier module left behind.
    assert db.query(Document).filter(Document.user_id.in_(ids)).count() == 0


def test_single_user_purge_goes_through_the_same_path(db, monkeypatch):
    """Account deletion and guest reaping must not drift — a table forgotten in
    one would silently strand rows."""
    from ragchat import guests
    from ragchat.db import Document

    calls = []
    monkeypatch.setattr("ragchat.vectordb.delete_users_chunks",
                        lambda ids: calls.append(list(ids)) or 0)
    g = _guest(db)
    db.add(Document(user_id=g.id, source_type="upload", title="d", size_bytes=1))
    db.commit()

    summary = guests.purge_user_data(db, g, drop_user=True)
    assert calls == [[g.id]]
    assert summary["documents"] == 1
    assert "users" not in summary, "single-user callers read a per-table summary"


def test_the_secret_check_refuses_non_ascii_rather_than_raising():
    """`presented` is a header an unauthenticated caller controls, and
    hmac.compare_digest on str raises TypeError the moment either side is
    non-ASCII. Starlette decodes header bytes as latin-1, so a raw client can
    put é there; a security check that errors on input the attacker chooses
    is not a check.

    Tested below the HTTP layer on purpose: httpx refuses to SEND a non-ASCII
    header, so a request-level test cannot reach the code it is aiming at.
    """
    from ragchat.app import _secret_matches

    assert _secret_matches("s3cret", "s3cret") is True
    assert _secret_matches("café-not-the-secret", "s3cret") is False
    assert _secret_matches("s3cret", "café") is False
    assert _secret_matches("", "s3cret") is False


def test_sweep_ignores_a_junk_limit_instead_of_erroring(client):
    r = client.post("/api/admin/sweep-guests?limit=abc",
                    headers={"x-sweep-secret": SWEEP_SECRET})
    assert r.status_code == 200, r.text


# --- sign-in is one INSERT; the sample documents follow separately ----------

def test_guest_login_does_not_seed_or_reap(client, monkeypatch):
    """Both measured as pure waiting in front of an empty screen: 6.4s of
    copying sample documents and 1.7s of clearing up after previous visitors,
    neither of which the arriving visitor asked for."""
    from ragchat import guests

    called = []
    monkeypatch.setattr(guests, "seed_demo_corpus",
                        lambda db, g: called.append("seed") or 0)
    monkeypatch.setattr(guests, "reap_stale_guests",
                        lambda db, **k: called.append("reap") or 0)

    r = client.post("/api/auth/guest-login")
    assert r.status_code == 200, r.text
    assert r.json()["seeded"] is False, "the client must know to ask for seeding"
    assert called == [], f"sign-in did work it should have deferred: {called}"


def test_the_seed_request_does_both(client, monkeypatch):
    from ragchat import guests

    called = []
    monkeypatch.setattr(guests, "seed_demo_corpus",
                        lambda db, g: called.append("seed") or 2)
    monkeypatch.setattr(guests, "reap_stale_guests",
                        lambda db, **k: called.append("reap") or 0)

    client.post("/api/auth/guest-login")
    r = client.post("/api/auth/guest-seed")
    assert r.status_code == 200, r.text
    assert r.json()["seeded"] is True
    assert called == ["seed", "reap"], called


def test_seeding_twice_does_not_copy_the_corpus_twice(client, monkeypatch):
    """The client can retry, and a double-fired request must cost a SELECT
    rather than a second copy of every sample document."""
    from ragchat import guests
    from ragchat.db import Document, User

    monkeypatch.setattr(guests, "reap_stale_guests", lambda db, **k: 0)

    def _fake_seed(db, g):
        db.add(Document(user_id=g.id, source_type="upload", title="demo.md",
                        status="ready", is_demo=True, size_bytes=1))
        db.commit()
        return 1

    monkeypatch.setattr(guests, "seed_demo_corpus", _fake_seed)
    client.post("/api/auth/guest-login")
    assert client.post("/api/auth/guest-seed").json()["seeded"] is True

    monkeypatch.setattr(guests, "seed_demo_corpus",
                        lambda db, g: pytest.fail("seeded a second time"))
    again = client.post("/api/auth/guest-seed")
    assert again.json() == {"seeded": True, "documents": 1, "reason": "already seeded"}


def test_seeding_refuses_a_signed_in_workspace(client):
    """It would put the demo corpus into somebody's real documents."""
    client.post("/api/auth/register",
                json={"username": f"u{os.urandom(3).hex()}", "password": "pw-12345678"})
    r = client.post("/api/auth/guest-seed")
    assert r.status_code == 200
    assert r.json()["seeded"] is False
