"""`/api/auth/local-login` must not exist where real sign-in does.

It takes no credentials and returns a full non-guest session. On the live
deployment the account it hands out held the owner's real business documents,
so anyone who knew the path could read or delete them — and it also walked past
the signed-in-only gate on web search, handing out a metered third-party quota.

It survived because the frontend stopped calling it and nothing pointed at it
any more. An unauthenticated endpoint that nothing uses is exactly the kind
that stops being looked at, so this file points at it.

No network: temp SQLite, no external calls.
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
def client():
    from ragchat import app as rapp
    from ragchat.db import engine
    import ragchat.db as _db

    _db._initialized = False
    for tbl in ("messages", "conversations", "users", "documents", "folders"):
        with engine.begin() as conn:
            conn.execute(_t(f"DROP TABLE IF EXISTS {tbl}"))
    yield TestClient(rapp.app, raise_server_exceptions=True)


def _oauth(monkeypatch, configured: bool):
    from ragchat import app as rapp

    monkeypatch.setattr(rapp.authn, "oauth_configured", lambda: configured)


def test_it_is_gone_where_google_sign_in_exists(client, monkeypatch):
    _oauth(monkeypatch, True)
    r = client.post("/api/auth/local-login")
    assert r.status_code == 404, (
        "an unauthenticated route handed out a non-guest session on a "
        "deployment that has real sign-in"
    )


def test_it_leaves_no_session_behind_when_refused(client, monkeypatch):
    """A 404 that still set the cookie would be worse than no check at all."""
    _oauth(monkeypatch, True)
    client.post("/api/auth/local-login")
    status = client.get("/api/auth/status").json()
    assert status["authenticated"] is False


def test_the_refusal_does_not_advertise_the_route(client, monkeypatch):
    """404, not 403: a 'forbidden' confirms there is something there."""
    _oauth(monkeypatch, True)
    assert client.post("/api/auth/local-login").status_code == 404


def test_it_still_works_in_local_development(client, monkeypatch):
    """Where there is no OAuth there is no other way in, and this is a laptop."""
    _oauth(monkeypatch, False)
    r = client.post("/api/auth/local-login")
    assert r.status_code == 200, r.text
    assert client.get("/api/auth/status").json()["authenticated"] is True


def test_the_web_tool_is_not_reachable_through_it(client, monkeypatch):
    """The signed-in-only gate on web search is only as strong as the weakest
    way of becoming signed in."""
    _oauth(monkeypatch, True)
    client.post("/api/auth/local-login")
    status = client.get("/api/auth/status").json()
    assert status.get("web_search_available") is not True
