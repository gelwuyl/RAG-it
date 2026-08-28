"""Bank-referenced correctness: a matched demo question's HUMAN answer is the
grading reference, never a model-drafted one.

The demo bank (eval/golden.py) now carries a human ``expected`` answer per
question. When a live question matches an answerable entry, correctness must
grade against THAT answer, must mark the reading ``expected_source: "bank"`` so
the UI can promote it out of the estimates strip, and must never call
``synthesize_expected`` — not on the answer request, not on the deferred grade
request, not on any retry. Everything else stays exactly as it was: unmatched
questions keep the drafted-reference path, known-unanswerable matches stay on
the refusal path, ``use_gold=False`` (the benchmark) is untouched, and a broken
judge still means "not graded" (None), never a false FAIL.

No network: every judge (and the drafter) is stubbed. What is under test is
which reference the correctness judge RECEIVES and what is persisted around it.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Isolated sqlite before ragchat imports, same as test_grade_split.
_DB_PATH = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SESSION_SECRET"] = "test-secret"
for _k in ("PG_DATABASE_URL", "rag_gel_DATABASE_URL"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text as _t  # noqa: E402

from eval import golden, judges  # noqa: E402
from ragchat import pipeline as _pl  # noqa: E402

HUMAN = "Between 1 and 4 degrees Celsius; above 5, call extension 204."
CONTEXT = "[1] Milk fridges must read between 1 and 4 degrees Celsius."
ANSWER = "Between 1 and 4 degrees Celsius. [1]"

GOLD = {
    "question": "What temperature should the milk fridges read?",
    "unanswerable": False,
    "expected": HUMAN,
    "golden_passages": ["Milk fridges must read\nbetween 1 and 4 degrees Celsius."],
    "golden_doc": "meridian_coffee_ops.md",
    "_src": "demo",
    "_idx": 70,
}
UNANS = {
    "question": "How much does the SunPak 5 battery cost?",
    "unanswerable": True,
    "expected": "",
    "golden_passages": [],
    "golden_doc": "helios_energy_handbook.md",
    "_src": "demo",
    "_idx": 71,
}


def _make_cfg(**over):
    base = dict(
        chunk_size=512, chunk_overlap=75, splitter="recursive",
        top_k=4, candidate_k=20, similarity_threshold=0.0,
        hybrid_search=False, reranker=False, query_rewrite=False,
        llm_model="stub", router_model="stub-router", temperature=0.0,
        embedding_model="stub-embed", embedding_provider="gemini",
        reranker_provider="gemini", eval_show=True,
    )
    base.update(over)
    from ragchat import config as _cfg
    return _cfg.PipelineConfig(**base)


def _stub_judges(monkeypatch, correct=(True, 0.87, "covers the facts")):
    """Replace every judge with an instant stub; count what correctness
    receives and whether the drafter ran at all."""
    calls = {"synth": 0, "correct_refs": []}

    def _synth(question, context):
        calls["synth"] += 1
        return "drafted reference.", ""

    def _correct(question, expected, answer):
        calls["correct_refs"].append(expected)
        if isinstance(correct, Exception):
            raise correct
        return correct

    monkeypatch.setattr(judges, "synthesize_expected", _synth)
    monkeypatch.setattr(judges, "answer_correctness_scored", _correct)
    for name in (
        "faithfulness_scored",
        "answer_relevancy_scored",
        "context_relevance_scored",
        "context_sufficiency_scored",
    ):
        monkeypatch.setattr(judges, name, lambda *a, **k: (True, 0.9, "ok"))
    return calls


@pytest.fixture()
def rig(monkeypatch):
    """The pipeline answer path with retrieval/generation stubbed: one chunk,
    one answer, no tools."""
    state = {"pool": [{"text": GOLD["golden_passages"][0],
                       "similarity": 0.62, "doc_id": "m", "title": "m.md",
                       "ref": "", "chunk_id": "m:0"}],
             "answers": [ANSWER]}
    monkeypatch.setattr(_pl, "retrieve", lambda *a, **k: list(state["pool"]))
    monkeypatch.setattr(_pl, "_chat", lambda model, messages, temperature: state["answers"][0])
    monkeypatch.setattr(_pl, "_golden_ndcg", lambda *a, **k: 0.9)
    return state


# ---------- 1. the bank reference reaches the judge; the drafter never runs ----------

def test_matched_bank_question_grades_against_the_human_answer(monkeypatch):
    calls = _stub_judges(monkeypatch)
    cfg = _make_cfg()
    out = _pl._eval_answer("how cold?", ANSWER, CONTEXT, cfg,
                           expected=HUMAN, expected_source="bank")
    assert calls["synth"] == 0, "a known answer must never be re-drafted"
    assert calls["correct_refs"] == [HUMAN]
    assert out["correct"] is True
    assert out["correct_score"] == 0.87
    assert out["expected_answer"] == HUMAN
    assert out["expected_source"] == "bank"


def test_ask_persists_the_bank_reference_on_the_answer(monkeypatch, rig):
    calls = _stub_judges(monkeypatch)
    monkeypatch.setattr(golden, "match_question", lambda q: GOLD)
    cfg = _make_cfg()
    out = _pl.ask("u", "how cold?", [], cfg, grade=True, use_gold=True)
    ev = out["eval"]
    assert calls["synth"] == 0
    assert calls["correct_refs"] == [HUMAN]
    assert ev["expected_answer"] == HUMAN
    assert ev["expected_source"] == "bank"
    assert ev["correct"] is True
    # The gold retrieval readings ride along unchanged.
    assert ev["gold"]["idx"] == GOLD["_idx"] and ev["gold"]["mrr"] == 1.0


# ---------- 2. the deferred grade request and its retries ----------

def test_deferred_grade_reuses_the_persisted_bank_reference(monkeypatch):
    """The stored eval dict from the answer request carries the reference; a
    grade request a request later must score against it and still never
    draft."""
    calls = _stub_judges(monkeypatch)
    cfg = _make_cfg()
    stored = {
        "pending": True,
        **{f: None for f in _pl.LIVE_GRADE_FIELDS},
        "expected_answer": HUMAN,
        "expected_source": "bank",
        "latency_ms": 1200,
        "top_sim": 0.62,
        "deep_n": 0,
    }
    out = _pl.grade_answer("how cold?", ANSWER, CONTEXT, cfg, stored)
    assert calls["synth"] == 0
    assert calls["correct_refs"] == [HUMAN]
    assert out["correct"] is True
    assert out["expected_source"] == "bank"


def test_retry_after_a_judge_failure_still_never_drafts(monkeypatch):
    """First pass: the correctness judge fails (outage). The reference is
    persisted anyway, so the retry scores against the SAME human answer."""
    calls = _stub_judges(monkeypatch, correct=RuntimeError("judge 404"))
    cfg = _make_cfg()
    first = _pl._eval_answer("how cold?", ANSWER, CONTEXT, cfg,
                             expected=HUMAN, expected_source="bank")
    assert first["correct"] is None and first["correct_score"] is None
    assert first["expected_answer"] == HUMAN
    assert first["expected_source"] == "bank"

    calls2 = _stub_judges(monkeypatch)
    second = _pl._eval_answer("how cold?", ANSWER, CONTEXT, cfg, previous=first)
    assert calls2["synth"] == 0
    assert calls2["correct_refs"] == [HUMAN]
    assert second["correct"] is True
    assert second["expected_source"] == "bank"


def test_route_level_grade_keeps_the_marker(monkeypatch):
    """Through the real HTTP split: the ask route persists the reference, the
    /grade route replays it, and expected_source survives into the stored
    verdict the UI reads."""
    calls = _stub_judges(monkeypatch)
    from ragchat import app as rapp
    from ragchat.db import engine
    import ragchat.db as _db

    _db._initialized = False
    for tbl in ("messages", "conversations", "users", "documents"):
        with engine.begin() as conn:
            conn.execute(_t(f"DROP TABLE IF EXISTS {tbl}"))
    client = TestClient(rapp.app, raise_server_exceptions=True)
    client.post("/api/auth/local-login")

    def _fake_ask(user_id, query, history, cfg, deep_search=None, web_search=None, grade=True):
        assert grade is False
        return {
            "answer": ANSWER,
            "not_found": False,
            "citations": [],
            "eval_line": "top sim 0.62 - 100 ms",
            "eval": {"pending": True, **{f: None for f in _pl.LIVE_GRADE_FIELDS},
                     "expected_answer": HUMAN, "expected_source": "bank",
                     "top_sim": 0.62, "deep_n": 0, "latency_ms": 100},
            "context": CONTEXT,
            "effective_query": "how cold?",
        }

    monkeypatch.setattr(rapp, "ask", _fake_ask)
    cid = client.post("/api/chats").json()["id"]
    body = client.post(f"/api/chats/{cid}/ask", json={"question": "how cold?"}).json()
    r = client.post(f"/api/chats/{cid}/messages/{body['message_id']}/grade")
    assert r.status_code == 200, r.text
    ev = r.json()["eval"]
    assert calls["synth"] == 0
    assert calls["correct_refs"] == [HUMAN]
    assert ev["correct"] is True
    assert ev["expected_source"] == "bank"
    # ...and it is what a reload would serve from the stored message.
    chat = client.get(f"/api/chats/{cid}").json()
    stored = next(
        m["eval_data"] for m in chat["messages"] if m["role"] == "assistant"
    )
    assert stored["expected_source"] == "bank"
    assert stored["correct"] is True


# ---------- 3. everyone else keeps today's behavior ----------

def test_unmatched_question_still_drafts_its_reference(monkeypatch):
    calls = _stub_judges(monkeypatch)
    cfg = _make_cfg()
    out = _pl._eval_answer("how cold?", ANSWER, CONTEXT, cfg)
    assert calls["synth"] == 1
    assert calls["correct_refs"] == ["drafted reference."]
    assert out["correct"] is True
    assert "expected_source" not in out, "a drafted reference must not be labelled bank truth"
    assert out["expected_answer"] == "drafted reference."


def test_unmatched_ask_has_no_bank_marker(monkeypatch, rig):
    calls = _stub_judges(monkeypatch)
    monkeypatch.setattr(golden, "match_question", lambda q: None)
    out = _pl.ask("u", "how cold?", [], _make_cfg(), grade=True, use_gold=True)
    assert calls["synth"] == 1
    assert "expected_source" not in out["eval"]


def test_known_unanswerable_match_never_gets_a_bank_reference(monkeypatch, rig):
    """A wrongly-answered unanswerable question is graded exactly as before —
    drafted reference, no bank marker. Bank membership alone must not hand it
    a reference (it has none) or change its verdict shape."""
    calls = _stub_judges(monkeypatch)
    monkeypatch.setattr(golden, "match_question", lambda q: UNANS)
    out = _pl.ask("u", "how much does the battery cost?", [], _make_cfg(),
                  grade=True, use_gold=True)
    ev = out["eval"]
    assert calls["synth"] == 1, "the drafted path is unchanged for unanswerables"
    assert "expected_source" not in ev
    assert ev["gold"]["refused"] is False
    assert ev["gold"]["unanswerable"] is True


def test_known_unanswerable_refusal_attaches_no_reference(monkeypatch, rig):
    """The model refuses: that is the measured verdict, and no correctness
    judge runs at all — bank entry or not."""
    rig["pool"] = []
    calls = _stub_judges(monkeypatch)
    monkeypatch.setattr(golden, "match_question", lambda q: UNANS)
    out = _pl.ask("u", "how much does the battery cost?", [], _make_cfg(),
                  grade=False, use_gold=True)
    assert out["not_found"] is True
    assert out["eval"]["gold"]["refused"] is True
    assert "expected_source" not in out["eval"]


def test_use_gold_false_benchmark_is_untouched(monkeypatch, rig):
    """The harness IS the golden run: no matching, no bank reference, drafted
    path exactly as before."""
    calls = _stub_judges(monkeypatch)
    seen = {"n": 0}

    def _spy(q):
        seen["n"] += 1
        return GOLD
    monkeypatch.setattr(golden, "match_question", _spy)
    out = _pl.ask("u", "how cold?", [], _make_cfg(), grade=True, use_gold=False)
    assert seen["n"] == 0
    assert calls["synth"] == 1
    assert "expected_source" not in out["eval"]
    assert "gold" not in out["eval"]


# ---------- 4. a broken judge stays ungraded, never a false FAIL ----------

def test_bank_judge_failure_stays_ungraded_with_an_actionable_reason(monkeypatch):
    calls = _stub_judges(monkeypatch, correct=RuntimeError("judge model 404"))
    out = _pl._eval_answer("how cold?", ANSWER, CONTEXT, _make_cfg(),
                           expected=HUMAN, expected_source="bank")
    assert out["correct"] is None, "an outage must not become a verdict"
    assert out["correct_score"] is None
    assert out["expected_answer"] == HUMAN, "the reference persists for the retry"
    assert out["expected_source"] == "bank"
    assert calls["synth"] == 0
    assert "Answer correctness unavailable" in out.get("judge_error", "")
