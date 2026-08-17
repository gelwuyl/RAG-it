"""What an anonymous caller and a guest are allowed to write.

These are the routes that made guest-first mode unsafe to turn on.

`config_overrides` is a SINGLE shared row (db.py:121), so a config write is not
a personal preference — it re-points the embedding model for every user and
invalidates their chunks. Four eval routes had no authentication dependency at
all, which meant any unauthenticated caller on the public deployment could do
that, or start a 46-question benchmark against the deployment's LLM quota.

The mirror-image risk is over-correcting: a guest workspace has to stay a real
workspace. Uploading, asking, deleting and pruning must all keep working, or
guest mode is a display case rather than a trial. Both directions are asserted.

Runs against a temp SQLite DB with no network.

Run:  .venv/Scripts/python -m pytest tests/test_guest_permissions.py -q
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

_DB_PATH = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SESSION_SECRET"] = "test-secret"
for _k in ("PG_DATABASE_URL", "rag_gel_DATABASE_URL"):
    os.environ.pop(_k, None)


# Every route that writes GLOBAL state or spends real budget, with a body that
# would otherwise succeed. Kept as data so adding a route to require_account
# without adding it here is visible as a gap.
GLOBAL_WRITE_ROUTES = [
    ("POST", "/api/eval/hybrid-search", None),
    ("POST", "/api/eval/web-augmentation", None),
    ("PUT", "/api/eval/config", {"top_k": 5}),
    ("POST", "/api/eval/run", {"retrieval_only": True}),
    ("POST", "/api/eval/step", {"retrieval_only": True}),
    ("POST", "/api/documents/reindex", None),
    ("POST", "/api/folders", {"path": "."}),
]


@pytest.fixture()
def client(monkeypatch):
    from ragchat import app as rapp
    import ragchat.db as _db

    _db._initialized = False
    _db.init_db()
    # No vector store and no LLM in this test: only the permission layer is
    # under test, and it must reject before reaching either.
    monkeypatch.setattr("ragchat.vectordb.delete_document_chunks", lambda *a, **k: None)
    with TestClient(rapp.app, raise_server_exceptions=True) as c:
        yield c


def _as_guest(client: TestClient) -> str:
    """Put a real guest session cookie on the client. Returns the guest's id.

    The id matters: tests share one DB, so several guest rows accumulate and
    "the guest with provider=guest" is ambiguous. Only the id ties a row to the
    session cookie actually being used.
    """
    r = client.post("/api/auth/guest-login")
    assert r.status_code == 200, r.text
    assert r.json()["guest"] is True
    return r.json()["id"]


def _as_account(client: TestClient) -> None:
    """Put a real signed-in (non-guest) session cookie on the client."""
    r = client.post(
        "/api/auth/register",
        json={"username": f"real-{os.urandom(4).hex()}", "password": "pw-12345678"},
    )
    assert r.status_code == 200, r.text


def _call(client: TestClient, method: str, path: str, body):
    return client.request(method, path, json=body) if body else client.request(method, path)


# --------------------------------------------------------------------------
# Anonymous callers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method,path,body", GLOBAL_WRITE_ROUTES,
                         ids=[f"{m} {p}" for m, p, _ in GLOBAL_WRITE_ROUTES])
def test_anonymous_cannot_write_global_state(client, method, path, body):
    """No cookie at all must not be able to re-point the deployment's config.

    Four of these routes previously took no user dependency whatsoever, so this
    was reachable over the open internet on the public deployment.
    """
    client.cookies.clear()
    r = _call(client, method, path, body)
    assert r.status_code in (401, 403), (
        f"{method} {path} answered {r.status_code} to an anonymous caller"
    )


# --------------------------------------------------------------------------
# Guests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method,path,body", GLOBAL_WRITE_ROUTES,
                         ids=[f"{m} {p}" for m, p, _ in GLOBAL_WRITE_ROUTES])
def test_guest_cannot_write_global_state(client, method, path, body):
    _as_guest(client)
    r = _call(client, method, path, body)
    assert r.status_code == 403, (
        f"{method} {path} answered {r.status_code} to a guest"
    )
    # The refusal has to say what to do about it. A bare "Forbidden" on a
    # portfolio demo reads as a broken app.
    assert "sign in" in r.json()["detail"].lower()


def test_guest_config_write_leaves_the_shared_config_untouched(client):
    """The point of the guard, asserted on the actual side effect.

    A 403 that still wrote the row would be worse than no guard at all, because
    it would look protected.
    """
    from ragchat.config import load_config

    before = load_config().top_k
    _as_guest(client)
    client.put("/api/eval/config", json={"top_k": before + 3})
    assert load_config().top_k == before


def test_signed_in_accounts_keep_full_access(client):
    """The guard must key on GUEST, not on 'not the owner'.

    If this fails the app has quietly become single-user again.
    """
    _as_account(client)
    r = client.post("/api/eval/hybrid-search")
    assert r.status_code == 200, r.text
    assert "hybrid_search" in r.json()


# --------------------------------------------------------------------------
# What a guest keeps
# --------------------------------------------------------------------------


def test_guest_workspace_stays_usable(client):
    """Guest mode is a trial, not a diorama — these must NOT be locked down."""
    _as_guest(client)
    assert client.get("/api/documents").status_code == 200
    assert client.get("/api/folders").status_code == 200
    assert client.get("/api/chats").status_code == 200
    # Reading the last benchmark is deliberately open: the scorecard is the most
    # portfolio-legible thing in the app, and showing it costs nothing.
    assert client.get("/api/eval").status_code == 200
    # Pruning only ever touches the caller's own orphaned chunks.
    assert client.post("/api/documents/prune").status_code == 200


def test_guest_upload_cap_covers_urls_too(client, monkeypatch):
    """A URL add is an embedded document billed to the deployment, same as an
    upload. Guarding only /upload left the cap bypassable by pasting links."""
    from ragchat.db import Document, SessionLocal
    from ragchat import guests

    guest_id = _as_guest(client)
    db = SessionLocal()
    try:
        for i in range(guests.GUEST_MAX_DOCUMENTS):
            db.add(Document(user_id=guest_id, source_type="upload",
                            title=f"f{i}.txt", status="ready", size_bytes=10))
        db.commit()
    finally:
        db.close()

    # Must be refused BEFORE the fetch — reaching the network at all would mean
    # the cap costs a request per rejection.
    def _boom(*a, **k):
        raise AssertionError("fetch_url must not run for a capped guest")

    monkeypatch.setattr("ragchat.app.fetch_url", _boom)
    r = client.post("/api/documents/url", json={"url": "https://example.com"})
    assert r.status_code == 422
    assert "sign in" in r.json()["detail"].lower()


# --------------------------------------------------------------------------
# Session cookie flags
# --------------------------------------------------------------------------


def test_session_cookie_is_hardened(client):
    """httponly + samesite always; `secure` only where HTTPS is real.

    An unconditional `secure` flag is silently dropped on http://localhost, so
    local sign-in would appear to succeed and every later request would be
    anonymous — a failure with no error message anywhere.
    """
    from ragchat import auth as authn

    r = client.post("/api/auth/guest-login")
    raw = r.headers.get("set-cookie", "")
    assert authn.SESSION_COOKIE in raw
    assert "httponly" in raw.lower()
    assert "samesite=lax" in raw.lower()
    # VERCEL is unset under pytest, so this run is the local-HTTP case.
    assert "secure" not in raw.lower()
