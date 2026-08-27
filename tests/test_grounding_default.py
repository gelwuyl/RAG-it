"""The web tool is OFF unless the reader asks for it.

This app's claim is that answers are grounded in the documents you gave it. A
tool that quietly reaches outside whenever those documents fall short makes the
claim conditional without telling anyone — "usually grounded" is not what is on
the tin.

So the default is a product promise, not a preference, and it is pinned here
rather than left to whoever next edits AskIn. Deep search stays ON by contrast:
it reads the reader's OWN documents, so using it is the same promise pursued
harder.

No network: `ask` is stubbed and only the request contract is under test.
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
    from ragchat.db import engine
    import ragchat.db as _db

    _db._initialized = False
    for tbl in ("messages", "conversations", "users", "documents"):
        with engine.begin() as conn:
            conn.execute(_t(f"DROP TABLE IF EXISTS {tbl}"))

    seen = {}

    def _fake_ask(user_id, query, history, cfg, deep_search=None, web_search=None, grade=True):
        seen["deep"] = deep_search is not None
        seen["web"] = web_search is not None
        return {"answer": "ok", "not_found": False, "citations": [], "eval_line": ""}

    monkeypatch.setattr(rapp, "ask", _fake_ask)
    # The web tool is configured and the caller is an account, so nothing but
    # the DEFAULT can be what withholds it.
    monkeypatch.setattr(rapp.websearch, "is_configured", lambda: True)
    monkeypatch.setattr(rapp.guests, "is_guest", lambda u: False)

    c = TestClient(rapp.app, raise_server_exceptions=True)
    c.post("/api/auth/local-login")
    cid = c.post("/api/chats").json()["id"]
    return c, cid, seen


def test_a_plain_question_gets_no_web_tool(rig):
    client, cid, seen = rig
    client.post(f"/api/chats/{cid}/ask", json={"question": "anything"})
    assert seen["web"] is False, (
        "the app was handed the web tool on a question nobody asked to widen"
    )


def test_a_plain_question_still_gets_the_deep_tool(rig):
    """Searching the reader's own documents harder is the same promise, not a
    wider one."""
    client, cid, seen = rig
    client.post(f"/api/chats/{cid}/ask", json={"question": "anything"})
    assert seen["deep"] is True


def test_asking_for_it_turns_it_on(rig):
    client, cid, seen = rig
    client.post(f"/api/chats/{cid}/ask", json={"question": "anything", "web_search": True})
    assert seen["web"] is True


def test_the_switch_can_still_remove_deep_search(rig):
    client, cid, seen = rig
    client.post(f"/api/chats/{cid}/ask", json={"question": "anything", "deep_search": False})
    assert seen["deep"] is False


def test_the_ui_ships_the_web_switch_dark(rig):
    """A lit switch means the tool is available, so the web one must not start
    lit — the page would be contradicting the request it is about to send."""
    import re

    html = (ROOT / "frontend" / "app.html").read_text(encoding="utf-8")

    def button(el_id: str) -> str:
        m = re.search(rf'<button[^>]*id="{el_id}"[^>]*>', html)
        assert m, f"#{el_id} is gone or was renamed"
        return m.group(0)

    web = button("web-toggle")
    assert "switch-btn on" not in web, f"the web switch ships lit: {web}"
    assert 'aria-pressed="false"' in web, web

    deep = button("deep-toggle")
    assert "switch-btn on" in deep, "deep search should still ship enabled"
    assert 'aria-pressed="true"' in deep, deep
