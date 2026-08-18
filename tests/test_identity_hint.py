"""The identity hint cookie: fast paint, and never a lie.

The session cookie is httpOnly, so the page cannot read it and used to wait on
/api/auth/status before knowing whether to draw a guest badge or a sign-in
button — 1.2s warm and about 3s cold, during which the top right corner is
blank. That is the "the sign-in button is missing" report.

`ragchat_kind` carries only "guest" or "account" and is readable by the page.
It is a RENDERING hint: forging it changes what your own browser draws for a
moment and nothing else, because every authorisation decision still rests on
the httpOnly session cookie. These tests hold it to two things — that it is
always written where a session is, and never survives one.
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

_DB_PATH = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SESSION_SECRET"] = "test-secret"
for _k in ("PG_DATABASE_URL", "rag_gel_DATABASE_URL"):
    os.environ.pop(_k, None)

KIND = "ragchat_kind"


@pytest.fixture()
def client(monkeypatch):
    from ragchat import app as rapp
    import ragchat.db as _db

    _db._initialized = False
    _db.init_db()
    monkeypatch.setattr("ragchat.vectordb.delete_document_chunks", lambda *a, **k: None)
    monkeypatch.setattr("ragchat.vectordb.delete_users_chunks", lambda ids: 0)
    monkeypatch.setattr("ragchat.guests.seed_demo_corpus", lambda db, g: 0)
    with TestClient(rapp.app, raise_server_exceptions=True) as c:
        yield c


def test_a_guest_login_writes_the_hint(client):
    r = client.post("/api/auth/guest-login")
    assert r.status_code == 200
    assert r.cookies.get(KIND) == "guest"


def test_the_hint_is_readable_by_the_page(client):
    """httpOnly would defeat the entire purpose — the page must read it."""
    r = client.post("/api/auth/guest-login")
    raw = "; ".join(r.headers.get_list("set-cookie"))
    kind_line = [c for c in r.headers.get_list("set-cookie") if c.startswith(KIND)]
    assert kind_line, raw
    assert "httponly" not in kind_line[0].lower()


def test_signing_in_flips_the_hint_to_account(client):
    client.post("/api/auth/guest-login")
    r = client.post("/api/auth/register",
                    json={"username": f"u{os.urandom(3).hex()}", "password": "pw-12345678"})
    assert r.status_code == 200, r.text
    assert r.cookies.get(KIND) == "account"


def test_status_backfills_a_session_that_predates_the_hint(client):
    """The case that would otherwise never be covered: an existing visitor
    never calls guest-login again, because status keeps answering fine, so
    without this their top bar paints blank forever."""
    client.post("/api/auth/guest-login")
    client.cookies.delete(KIND)          # a session from before the hint shipped
    r = client.get("/api/auth/status")
    assert r.json()["is_guest"] is True
    assert r.cookies.get(KIND) == "guest"


def test_status_corrects_a_hint_that_disagrees(client):
    client.post("/api/auth/guest-login")
    client.cookies.set(KIND, "account")  # drifted, or edited by hand
    r = client.get("/api/auth/status")
    assert r.cookies.get(KIND) == "guest"


def test_logging_out_clears_the_hint(client):
    """Left behind, it paints a signed-in top bar over a signed-out session."""
    client.post("/api/auth/register",
                json={"username": f"u{os.urandom(3).hex()}", "password": "pw-12345678"})
    r = client.post("/api/auth/logout")
    set_cookies = "; ".join(r.headers.get_list("set-cookie"))
    assert KIND in set_cookies, "logout did not clear the hint"
    assert client.cookies.get(KIND) in (None, "", '""')


def test_a_hint_alone_authorises_nothing(client):
    """It is a rendering hint. Forging it must not make anyone anybody."""
    client.cookies.clear()
    client.cookies.set(KIND, "account")
    r = client.get("/api/auth/status")
    assert r.json()["authenticated"] is False
    # ...and the lie is cleaned up rather than left to paint again next load.
    # Asserted on the header the server sent, not on the test client's jar:
    # the jar entry here was created without a domain and httpx will not match
    # a deletion against it, which is a harness detail rather than behaviour.
    cleared = [c for c in r.headers.get_list("set-cookie") if c.startswith(KIND)]
    assert cleared, "a forged hint was left in place"
    assert 'Max-Age=0' in cleared[0] or 'expires=' in cleared[0].lower()
