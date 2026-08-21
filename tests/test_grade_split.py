"""The answer arrives first; the grade arrives in a second request.

Grading is two more sequential model calls and, measured against the live
provider, they cost MORE than writing the answer did (10.1s against 7.8s). So
`/ask` no longer waits for them and `/grade` runs them afterwards.

Two things have to hold for that split to be safe, and neither is visible from
the UI:

  * the judges must see the passages the answer was actually built from — a
    request later, retrieval is gone, so the context travels with the message;
  * a retry must not spend two more judge calls, and must never overwrite a
    verdict with a fresh failure.

No network: the pipeline is stubbed, because what is under test is the routing
and persistence contract, not the models.
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

CONTEXT = "[1] The milk fridge must read between 1 and 4 degrees Celsius."
ANSWER = "Between 1 and 4 degrees Celsius. [1]"


@pytest.fixture()
def client():
    from ragchat import app as rapp
    from ragchat.db import engine
    import ragchat.db as _db

    _db._initialized = False
    for tbl in ("messages", "conversations", "users", "documents", "folders"):
        with engine.begin() as conn:
            conn.execute(_t(f"DROP TABLE IF EXISTS {tbl}"))
    c = TestClient(rapp.app, raise_server_exceptions=True)
    c.post("/api/auth/local-login")
    yield c


@pytest.fixture()
def answered(client, monkeypatch):
    """A chat holding one answered-but-ungraded assistant message."""
    from ragchat import app as rapp

    def _fake_ask(user_id, query, history, cfg, deep_search=None, grade=True):
        assert grade is False, "the chat route must not wait for the judges"
        return {
            "answer": ANSWER,
            "not_found": False,
            "citations": [],
            "eval_line": "top sim 0.38 - 8255 ms",
            "eval": {"pending": True, "faithful": None, "relevant": None,
                     "top_sim": 0.38, "deep_n": 0, "latency_ms": 8255},
            "context": CONTEXT,
            "effective_query": "milk fridge temperature",
        }

    monkeypatch.setattr(rapp, "ask", _fake_ask)
    cid = client.post("/api/chats").json()["id"]
    r = client.post(f"/api/chats/{cid}/ask", json={"question": "how cold?"})
    assert r.status_code == 200, r.text
    return client, cid, r.json()


def _stub_judges(monkeypatch, verdict=(True, True)):
    """Replace the judges and count how often they actually run."""
    from ragchat import pipeline

    calls = {"n": 0}

    def _fake(question, answer, context_text, cfg):
        calls["n"] += 1
        _fake.seen = {"q": question, "answer": answer, "context": context_text}
        return {"faithful": verdict[0], "faithful_reason": "r",
                "relevant": verdict[1], "relevant_reason": "r"}

    monkeypatch.setattr(pipeline, "_eval_answer", _fake)
    return calls, _fake


# --- the answer comes back ungraded ---------------------------------------

def test_the_answer_returns_without_a_verdict(answered):
    _, _, body = answered
    assert body["answer"] == ANSWER
    assert body["eval"]["pending"] is True
    assert body["eval"]["faithful"] is None
    assert body["message_id"], "the client cannot ask for a grade without this"


def test_the_retrieval_context_does_not_travel_to_the_browser(answered):
    """It exists so the judge can be honest; the reader already has citations."""
    _, _, body = answered
    assert "context" not in body and "effective_query" not in body


# --- grading it -----------------------------------------------------------

def test_grading_fills_the_verdicts_in(answered, monkeypatch):
    client, cid, body = answered
    calls, _ = _stub_judges(monkeypatch)
    r = client.post(f"/api/chats/{cid}/messages/{body['message_id']}/grade")
    assert r.status_code == 200, r.text
    ev = r.json()["eval"]
    assert calls["n"] == 1
    assert ev["faithful"] is True and ev["relevant"] is True
    assert ev["pending"] is False
    assert "grade_ms" in ev


def test_the_judge_sees_what_the_answer_was_built_from(answered, monkeypatch):
    """Re-running retrieval to rebuild the context would grade the answer
    against a different set of passages than the model actually saw."""
    client, cid, body = answered
    _, fake = _stub_judges(monkeypatch)
    client.post(f"/api/chats/{cid}/messages/{body['message_id']}/grade")
    assert fake.seen["context"] == CONTEXT
    assert fake.seen["answer"] == ANSWER
    assert fake.seen["q"] == "milk fridge temperature", "the REWRITTEN query is judged"


def test_the_answer_latency_survives_grading(answered, monkeypatch):
    """`latency_ms` is what the reader waited for the ANSWER. Grading adds its
    own number; it must not overwrite that one."""
    client, cid, body = answered
    _stub_judges(monkeypatch)
    ev = client.post(f"/api/chats/{cid}/messages/{body['message_id']}/grade").json()["eval"]
    assert ev["latency_ms"] == 8255


def test_the_rebuilt_grey_line_carries_the_verdicts(answered, monkeypatch):
    client, cid, body = answered
    _stub_judges(monkeypatch, verdict=(True, False))
    line = client.post(f"/api/chats/{cid}/messages/{body['message_id']}/grade").json()["eval_line"]
    assert "faith PASS" in line and "rel FAIL" in line
    assert "top sim 0.38" in line, "retrieval facts must survive into the second request"


# --- it must be safe to call twice ----------------------------------------

def test_a_second_grade_call_spends_nothing(answered, monkeypatch):
    """The client retries on timeout. Two more judge calls per retry is how a
    free tier gets exhausted by one flaky connection."""
    client, cid, body = answered
    calls, _ = _stub_judges(monkeypatch)
    first = client.post(f"/api/chats/{cid}/messages/{body['message_id']}/grade").json()
    second = client.post(f"/api/chats/{cid}/messages/{body['message_id']}/grade").json()
    assert calls["n"] == 1
    assert second["eval"]["faithful"] == first["eval"]["faithful"]


def test_a_graded_verdict_is_never_replaced_by_a_later_failure(answered, monkeypatch):
    from ragchat import pipeline

    client, cid, body = answered
    _stub_judges(monkeypatch)
    client.post(f"/api/chats/{cid}/messages/{body['message_id']}/grade")

    def _broken(*a, **k):
        return {"faithful": None, "relevant": None, "judge_error": "504"}

    monkeypatch.setattr(pipeline, "_eval_answer", _broken)
    ev = client.post(f"/api/chats/{cid}/messages/{body['message_id']}/grade").json()["eval"]
    assert ev["faithful"] is True and "judge_error" not in ev


# --- it stays in its lane -------------------------------------------------

def test_another_users_message_cannot_be_graded(answered, monkeypatch):
    """The grade route reads a stored answer and spends model calls on it."""
    client, cid, body = answered
    from ragchat.db import SessionLocal, Conversation

    s = SessionLocal()
    conv = s.get(Conversation, cid)
    conv.user_id = "somebody-else"
    s.commit()
    s.close()

    calls, _ = _stub_judges(monkeypatch)
    r = client.post(f"/api/chats/{cid}/messages/{body['message_id']}/grade")
    assert r.status_code == 404
    assert calls["n"] == 0


def test_a_message_from_another_chat_is_rejected(answered, monkeypatch):
    client, cid, body = answered
    other = client.post("/api/chats").json()["id"]
    calls, _ = _stub_judges(monkeypatch)
    r = client.post(f"/api/chats/{other}/messages/{body['message_id']}/grade")
    assert r.status_code == 404
    assert calls["n"] == 0


# --- nothing to grade -----------------------------------------------------

def test_an_answer_with_no_context_stops_pending_instead_of_hanging(client, monkeypatch):
    """A not-found reply has no passages. The bars must stop waiting rather
    than spin forever on a grade that can never arrive."""
    from ragchat import app as rapp

    def _not_found(user_id, query, history, cfg, deep_search=None, grade=True):
        return {"answer": "No match in your documents.",
                "not_found": True, "citations": [], "eval_line": "12 ms"}

    monkeypatch.setattr(rapp, "ask", _not_found)
    cid = client.post("/api/chats").json()["id"]
    body = client.post(f"/api/chats/{cid}/ask", json={"question": "?"}).json()

    calls, _ = _stub_judges(monkeypatch)
    ev = client.post(f"/api/chats/{cid}/messages/{body['message_id']}/grade").json()["eval"]
    assert calls["n"] == 0
    assert ev.get("pending") is False
