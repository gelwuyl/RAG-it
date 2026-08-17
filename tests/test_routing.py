"""Where sign-in lands, now that "/" is a landing page.

This is the ordering trap the handoff got wrong: the OAuth callback redirect to
/app and the routing split (vercel.json rewrite + vite multi-entry) MUST ship
together. The redirect alone 404s because /app does not exist yet; the split
alone dumps every signed-in user onto the marketing page.

Neither half is visible in normal use — the callback only runs at the end of a
real Google round trip, which no other test exercises — so it needs a guard
that fails loudly if someone "tidies" the redirect back to "/".

Runs against a temp SQLite DB with the Google token exchange stubbed. No network.

Run:  .venv/Scripts/python -m pytest tests/test_routing.py -q
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


@pytest.fixture()
def client():
    from ragchat import app as rapp
    import ragchat.db as _db

    _db._initialized = False
    _db.init_db()
    with TestClient(rapp.app, raise_server_exceptions=True) as c:
        yield c


def _complete_oauth(client: TestClient, monkeypatch, sub: str = "google-user-1"):
    """Drive the callback with a valid state and a stubbed token exchange."""
    from ragchat import app as rapp

    async def _fake_exchange(code):
        return {"sub": sub, "email": f"{sub}@example.com", "name": "Test User"}

    monkeypatch.setattr(rapp.authn, "google_exchange_code", _fake_exchange)

    # The callback rejects a state that does not match the cookie, so set both.
    state = "test-state-value"
    client.cookies.set("oauth_state", state)
    return client.get(
        f"/api/auth/google/callback?code=abc&state={state}",
        follow_redirects=False,
    )


def test_oauth_callback_lands_on_the_app_not_the_landing_page(client, monkeypatch):
    r = _complete_oauth(client, monkeypatch)
    assert r.status_code in (302, 307), r.status_code
    location = r.headers["location"]
    assert location == "/app", (
        f"callback redirects to {location!r}; '/' is the landing page, so a user "
        "who just signed in would be dropped back on the marketing pitch"
    )


def test_oauth_callback_redirect_matches_the_configured_app_path(client, monkeypatch):
    """The redirect must come from APP_PATH, not a second hardcoded string.

    Two copies of "/app" drift; this asserts there is only one source.
    """
    from ragchat import app as rapp

    monkeypatch.setattr(rapp, "APP_PATH", "/somewhere-else")
    r = _complete_oauth(client, monkeypatch, sub="google-user-2")
    assert r.headers["location"] == "/somewhere-else"


def test_oauth_callback_still_issues_the_session_cookie(client, monkeypatch):
    """A redirect with no cookie would land on the app and 401 immediately."""
    from ragchat import auth as authn

    r = _complete_oauth(client, monkeypatch, sub="google-user-3")
    raw = r.headers.get("set-cookie", "")
    assert authn.SESSION_COOKIE in raw
    assert "httponly" in raw.lower()
    assert "samesite=lax" in raw.lower(), (
        "the callback is a cross-site top-level navigation; under samesite=strict "
        "the browser withholds the cookie it was just handed and the user lands "
        "signed out"
    )


def test_vercel_rewrite_and_vite_input_exist_for_app_html():
    """The other half of the split. Without these, /app is a 404 in production
    and the redirect above points at nothing."""
    import json

    vercel = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    sources = {r["source"]: r["destination"] for r in vercel["rewrites"]}
    assert sources.get("/app") == "/app.html", (
        "vercel.json has no /app -> /app.html rewrite; the OAuth callback "
        "redirect would 404 on the deploy"
    )
    # The API rewrite must survive alongside it.
    assert "/api/(.*)" in sources

    vite = (ROOT / "frontend" / "vite.config.js").read_text(encoding="utf-8")
    assert "app.html" in vite, (
        "vite.config.js has no app.html entry, so app.html never reaches dist/ "
        "and the rewrite points at a file that was never built"
    )

    assert (ROOT / "frontend" / "app.html").exists()
    assert (ROOT / "frontend" / "index.html").exists()


def test_landing_page_ships_no_app_js_and_calls_no_api():
    """The landing page is CDN-served and must cost zero serverless invocations
    for visitors who never enter."""
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "app.js" not in html, "landing page pulls in the app bundle"
    for forbidden in ("fetch(", "XMLHttpRequest", "/api/auth", "/api/eval", "/api/documents"):
        assert forbidden not in html, f"landing page references {forbidden}"
    # It must still read the same theme key, or a light-theme visitor gets a
    # dark pitch page followed by a light workspace.
    assert "ragchat-theme" in html
