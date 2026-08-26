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

import json
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

    def _fake_ask(user_id, query, history, cfg, deep_search=None, web_search=None, grade=True):
        assert grade is False, "the chat route must not wait for the judges"
        return {
            "answer": ANSWER,
            "not_found": False,
            "citations": [],
            "eval_line": "top sim 0.38 - 8255 ms",
            "eval": {"pending": True, "faithful": None, "relevant": None,
                     "context_relevance": None, "context_sufficiency": None,
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

    def _fake(question, answer, context_text, cfg, previous=None):
        calls["n"] += 1
        _fake.seen = {"q": question, "answer": answer, "context": context_text,
                      "previous": previous}
        return {"faithful": verdict[0], "faithful_reason": "r",
                "relevant": verdict[1], "relevant_reason": "r",
                "context_relevance": verdict[0], "context_relevance_reason": "r",
                "context_sufficiency": verdict[1], "context_sufficiency_reason": "r"}

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


def test_partial_grade_stays_pending_until_the_missing_judge_recovers(answered, monkeypatch):
    """A completed result survives while only the missing judge is retried."""
    from ragchat import pipeline

    client, cid, body = answered
    calls = []

    def _partial_then_complete(question, answer, context_text, cfg, previous=None):
        calls.append(previous or {})
        if len(calls) == 1:
            return {
                "faithful": True,
                "faithful_reason": "supported",
                "relevant": None,
                "relevant_reason": "429 rate limited",
                "context_relevance": True,
                "context_relevance_reason": "on point",
                "context_sufficiency": None,
                "context_sufficiency_reason": "judge timeout",
                "judge_error": "Answer relevancy unavailable: 429 rate limited"
                " · Context sufficiency unavailable: judge timeout",
            }
        assert previous["faithful"] is True
        assert previous["context_relevance"] is True
        return {
            "faithful": previous["faithful"],
            "faithful_reason": previous["faithful_reason"],
            "relevant": True,
            "relevant_reason": "on-topic",
            "context_relevance": previous["context_relevance"],
            "context_relevance_reason": previous["context_relevance_reason"],
            "context_sufficiency": True,
            "context_sufficiency_reason": "enough",
        }

    monkeypatch.setattr(pipeline, "_eval_answer", _partial_then_complete)
    url = f"/api/chats/{cid}/messages/{body['message_id']}/grade"
    first = client.post(url).json()["eval"]
    assert first["faithful"] is True and first["relevant"] is None
    assert first["context_relevance"] is True and first["context_sufficiency"] is None
    assert first["pending"] is True
    assert first["grade_attempts"] == 1

    second = client.post(url).json()["eval"]
    assert len(calls) == 2
    assert second["faithful"] is True and second["relevant"] is True
    assert second["context_sufficiency"] is True
    assert second["pending"] is False
    assert second["grade_attempts"] == 2
    assert "judge_error" not in second


def test_legacy_partial_result_is_eligible_for_the_new_retry(answered, monkeypatch):
    """Pre-fix messages said pending=false despite a missing verdict."""
    from ragchat.db import Message, SessionLocal

    client, cid, body = answered
    session = SessionLocal()
    msg = session.get(Message, body["message_id"])
    msg.eval_data = json.dumps({
        "pending": False,
        "faithful": True,
        "faithful_reason": "supported",
        "relevant": None,
        "relevant_reason": "judge timeout",
        "context_relevance": None,
        "context_sufficiency": None,
    })
    session.commit()
    session.close()

    calls, fake = _stub_judges(monkeypatch)
    ev = client.post(f"/api/chats/{cid}/messages/{body['message_id']}/grade").json()["eval"]

    assert calls["n"] == 1
    assert fake.seen["previous"]["faithful"] is True
    assert ev["faithful"] is True and ev["relevant"] is True
    assert ev["pending"] is False


def test_partial_grade_exhaustion_stops_after_the_bounded_budget(answered, monkeypatch):
    from ragchat import app as rapp
    from ragchat import pipeline

    client, cid, body = answered
    calls = {"n": 0}

    def _always_missing(question, answer, context_text, cfg, previous=None):
        calls["n"] += 1
        return {
            "faithful": True,
            "faithful_reason": "supported",
            "relevant": None,
            "relevant_reason": "judge timeout",
            "context_relevance": True,
            "context_relevance_reason": "on point",
            "context_sufficiency": None,
            "context_sufficiency_reason": "judge timeout",
            "judge_error": "Answer relevancy unavailable: judge timeout"
            " · Context sufficiency unavailable: judge timeout",
        }

    monkeypatch.setattr(pipeline, "_eval_answer", _always_missing)
    url = f"/api/chats/{cid}/messages/{body['message_id']}/grade"
    results = [client.post(url).json()["eval"] for _ in range(rapp.GRADE_MAX_ATTEMPTS)]
    assert [ev["pending"] for ev in results] == [True, True, False]
    exhausted = results[-1]
    assert exhausted["grade_exhausted"] is True
    assert "unavailable after 3 attempts" in exhausted["judge_error"]
    # The recovered check is named too — a partial outage does not hide behind
    # the one judge that stayed down.
    assert "Answer relevancy unavailable: judge timeout" in exhausted["judge_error"]
    assert "Context sufficiency unavailable: judge timeout" in exhausted["judge_error"]

    client.post(url)
    assert calls["n"] == rapp.GRADE_MAX_ATTEMPTS


def test_grading_disabled_midflight_does_not_spend_a_retry(answered, monkeypatch):
    from ragchat import app as rapp

    class _Cfg:
        eval_show = False

    client, cid, body = answered
    calls, _ = _stub_judges(monkeypatch)
    monkeypatch.setattr(rapp, "load_config", lambda: _Cfg())
    url = f"/api/chats/{cid}/messages/{body['message_id']}/grade"

    first = client.post(url).json()["eval"]
    second = client.post(url).json()["eval"]

    assert calls["n"] == 0
    assert first["pending"] is False
    assert first["grade_unavailable"] == "Live grading is disabled in Settings."
    assert "grade_attempts" not in first
    assert "grade_exhausted" not in first
    assert second == first


def test_disabled_grading_preserves_an_answer_timing_line(answered, monkeypatch):
    """A stray grade call must not erase facts recorded while grading was off."""
    from ragchat import app as rapp
    from ragchat.db import Message, SessionLocal

    class _Cfg:
        eval_show = False

    client, cid, body = answered
    original_line = "top sim 0.38 - 8255 ms"
    session = SessionLocal()
    msg = session.get(Message, body["message_id"])
    msg.eval_data = None
    msg.eval_line = original_line
    session.commit()
    session.close()

    calls, _ = _stub_judges(monkeypatch)
    monkeypatch.setattr(rapp, "load_config", lambda: _Cfg())
    url = f"/api/chats/{cid}/messages/{body['message_id']}/grade"
    response = client.post(url)

    assert response.status_code == 200, response.text
    assert response.json() == {"eval": None, "eval_line": original_line}
    assert calls["n"] == 0

    session = SessionLocal()
    stored = session.get(Message, body["message_id"])
    assert stored.eval_data is None
    assert stored.eval_line == original_line
    session.close()


def test_grade_request_locks_its_message_before_judging(answered, monkeypatch):
    """The production row lock serializes attempts across browser tabs."""
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.orm import Session

    client, cid, body = answered
    calls, _ = _stub_judges(monkeypatch)
    locked = []
    execute = Session.execute

    def _track_lock(self, statement, *args, **kwargs):
        if getattr(statement, "_for_update_arg", None) is not None:
            locked.append(statement)
        return execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(Session, "execute", _track_lock)
    r = client.post(f"/api/chats/{cid}/messages/{body['message_id']}/grade")

    assert r.status_code == 200, r.text
    assert calls["n"] == 1
    assert len(locked) == 1
    assert "FOR UPDATE" in str(locked[0].compile(dialect=postgresql.dialect()))


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

    def _not_found(user_id, query, history, cfg, deep_search=None, web_search=None, grade=True):
        return {"answer": "No match in your documents.",
                "not_found": True, "citations": [], "eval_line": "12 ms"}

    monkeypatch.setattr(rapp, "ask", _not_found)
    cid = client.post("/api/chats").json()["id"]
    body = client.post(f"/api/chats/{cid}/ask", json={"question": "?"}).json()

    calls, _ = _stub_judges(monkeypatch)
    ev = client.post(f"/api/chats/{cid}/messages/{body['message_id']}/grade").json()["eval"]
    assert calls["n"] == 0
    assert ev.get("pending") is False
    assert ev["grade_unavailable"] == "No source passages were stored for this answer."


def test_the_thread_carries_message_ids(answered):
    """A reader who reloads while an answer is being graded needs to be able to
    ask for that verdict again. Without an id on the message there is nothing
    to ask about, and the chip spins forever."""
    client, cid, body = answered
    msgs = client.get(f"/api/chats/{cid}").json()["messages"]
    assistant = [m for m in msgs if m["role"] == "assistant"]
    assert assistant and assistant[0]["id"] == body["message_id"]
    assert assistant[0]["eval_data"]["pending"] is True
