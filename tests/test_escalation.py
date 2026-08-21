"""The app reaches for the literal scan on its own.

Deep search used to be a switch the visitor held down. It is now a TOOL that is
always handed to `ask()`, which decides for itself whether to use it — the first
place in this pipeline where the system chooses an action instead of executing a
fixed line.

It escalates at exactly the two points retrieval already knows it failed:

  1. nothing cleared the similarity threshold, so it is about to refuse;
  2. the model read the passages and said the answer is not in the documents.

Both are checked BEFORE the response is sent, and the scan costs no model call
at all, so nothing is spent on a question that was going to be answered anyway.

What these tests are really guarding is the boundary. An escalation ladder in a
serverless function that is frozen the instant it responds, with 60 seconds to
work in, must be provably finite — so "the tool is used at most once" and "the
answer is generated at most twice" are load-bearing assertions, not trivia.

Deliberately NOT driven by the judges: `NOT_FOUND_ANSWER` is entirely faithful
to its context and squarely answers the question, so both judges PASS it. The
grader cannot see this failure, which is why the trigger is retrieval and the
model's own refusal instead.

No network: retrieval, generation and the judges are all stubbed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from ragchat import config as _cfg  # noqa: E402
from ragchat import pipeline as _pl  # noqa: E402

GOOD = "Per [1] that is right."
LITERAL = "The Probat roaster at Riverside is serviced every 400 hours."


def _make_cfg(**over):
    base = dict(
        chunk_size=512, chunk_overlap=75, splitter="recursive",
        top_k=4, candidate_k=20, similarity_threshold=0.0,
        hybrid_search=False, reranker=False, query_rewrite=False,
        llm_model="stub", temperature=0.0,
        embedding_model="text-embedding-005", embedding_provider="gemini",
        reranker_provider="gemini", eval_show=True,
    )
    base.update(over)
    return _cfg.PipelineConfig(**base)


def _chunk(text="Riverside store has Probat roaster.", sim=0.62):
    return {"text": text, "similarity": sim, "doc_id": "m",
            "title": "meridian_coffee_ops.md", "ref": "~40%", "chunk_id": "m:0"}


def _deep_hit(text=LITERAL):
    return {"text": text, "similarity": None, "doc_id": "m",
            "title": "meridian_coffee_ops.md", "ref": "~40%", "deep": True}


@pytest.fixture()
def rig(monkeypatch):
    """Stub retrieval + generation, and count what the pipeline reaches for."""
    state = {
        "pool": [_chunk()],
        "answers": [GOOD],       # popped in order; the last one repeats
        "deep": [_deep_hit()],   # what the scan returns
        "scans": 0,
        "generations": 0,
        "prompts": [],
    }

    monkeypatch.setattr(_pl, "retrieve", lambda *a, **k: list(state["pool"]))
    monkeypatch.setattr(_pl, "_eval_answer", lambda *a, **k: None)

    def _chat(model, messages, temperature):
        state["generations"] += 1
        state["prompts"].append(messages[-1]["content"])
        return state["answers"][min(state["generations"] - 1, len(state["answers"]) - 1)]

    monkeypatch.setattr(_pl, "_chat", _chat)

    def _tool(_query):
        state["scans"] += 1
        return list(state["deep"])

    state["tool"] = _tool
    return state


def _ask(rig, **kw):
    return _pl.ask("u", "servicing interval", [], _make_cfg(**kw.pop("cfg", {})),
                   deep_search=rig["tool"], **kw)


# --- the happy path stays free --------------------------------------------

def test_a_good_answer_never_touches_the_tool(rig):
    """The escalation must cost nothing on questions that were going to work.
    If this fails, every question just got slower for no reason."""
    res = _ask(rig)
    assert res["answer"] == GOOD
    assert rig["scans"] == 0
    assert rig["generations"] == 1


def test_the_switch_still_forces_it(rig):
    """A literal hit the ranker missed matters MOST when the ranker looked
    confident, which is when nobody looks twice. Forcing must stay
    unconditional."""
    res = _ask(rig, force_deep=True)
    assert rig["scans"] == 1
    assert LITERAL in rig["prompts"][0]
    assert res["answer"] == GOOD


# --- escalation 1: retrieval was too weak to bother generating ------------

def test_weak_retrieval_reaches_for_the_scan_before_refusing(rig):
    """Below the threshold the old pipeline refused without generating. That is
    the cheapest possible moment to look harder — the scan costs no model call."""
    rig["pool"] = [_chunk(sim=0.01)]
    res = _ask(rig, cfg={"similarity_threshold": 0.5})
    assert rig["scans"] == 1
    assert res["not_found"] is False
    assert LITERAL in rig["prompts"][0], "the rescued passage never reached the model"
    assert res["eval"]["escalated"] == "weak_retrieval"


def test_weak_retrieval_with_nothing_to_find_still_refuses(rig):
    rig["pool"] = [_chunk(sim=0.01)]
    rig["deep"] = []
    res = _ask(rig, cfg={"similarity_threshold": 0.5})
    assert rig["scans"] == 1
    assert res["not_found"] is True
    assert rig["generations"] == 0, "refused without spending a generation, as before"


# --- escalation 2: the model read it and said no --------------------------

def test_the_models_own_refusal_triggers_a_second_look(rig):
    """The strongest signal in the system that ranking dropped something: it
    comes from the one component that actually read the text."""
    rig["answers"] = [_pl.NOT_FOUND_ANSWER, GOOD]
    res = _ask(rig)
    assert rig["scans"] == 1
    assert rig["generations"] == 2
    assert res["not_found"] is False and res["answer"] == GOOD
    assert res["eval"]["escalated"] == "model_refused"
    assert LITERAL in rig["prompts"][1], "the retry was generated without the new passage"


def test_the_retry_is_told_what_the_first_attempt_was_not(rig):
    """The first prompt must NOT contain the literal hit and the second must —
    otherwise the retry is just the same question asked twice."""
    rig["answers"] = [_pl.NOT_FOUND_ANSWER, GOOD]
    _ask(rig)
    assert LITERAL not in rig["prompts"][0]
    assert LITERAL in rig["prompts"][1]


def test_a_scan_that_finds_nothing_costs_no_second_generation(rig):
    rig["answers"] = [_pl.NOT_FOUND_ANSWER]
    rig["deep"] = []
    res = _ask(rig)
    assert rig["scans"] == 1
    assert rig["generations"] == 1, "regenerating against an unchanged pool buys nothing"
    assert res["not_found"] is True


# --- the ladder has exactly one rung --------------------------------------

def test_the_tool_is_never_used_twice(rig):
    """A serverless function is frozen the instant it responds and has 60s to
    work in, so this is a ladder with one rung and not a `while`."""
    rig["answers"] = [_pl.NOT_FOUND_ANSWER, _pl.NOT_FOUND_ANSWER, GOOD]
    res = _ask(rig)
    assert rig["scans"] == 1
    assert rig["generations"] == 2, "a second escalation would be an unbounded loop"
    assert res["not_found"] is True


def test_forcing_the_scan_disarms_the_escalation(rig):
    """The tool already ran; running it again would return the same passages."""
    rig["answers"] = [_pl.NOT_FOUND_ANSWER]
    res = _ask(rig, force_deep=True)
    assert rig["scans"] == 1
    assert rig["generations"] == 1
    assert res["not_found"] is True


# --- failures degrade to what we already had ------------------------------

def test_a_broken_scan_does_not_cost_the_answer(rig, monkeypatch):
    def _boom(_q):
        rig["scans"] += 1
        raise RuntimeError("source_text unreadable")

    rig["tool"] = _boom
    rig["answers"] = [_pl.NOT_FOUND_ANSWER]
    res = _ask(rig)
    assert rig["scans"] == 1
    assert res["not_found"] is True, "a tool failure must not become a 500"


def test_a_failed_retry_keeps_the_refusal_it_already_had(rig, monkeypatch):
    """The retry is a bonus. Losing it must not lose the truthful refusal."""
    calls = {"n": 0}

    def _chat(model, messages, temperature):
        calls["n"] += 1
        if calls["n"] == 1:
            return _pl.NOT_FOUND_ANSWER
        raise RuntimeError("provider 503")

    monkeypatch.setattr(_pl, "_chat", _chat)
    res = _ask(rig)
    assert calls["n"] == 2, "the retry was attempted"
    assert res["not_found"] is True
    assert res["answer"].startswith(_pl.NOT_FOUND_ANSWER)


# --- it says so ------------------------------------------------------------

def test_a_refusal_after_a_scan_says_the_documents_were_read(rig):
    """"I couldn't find this" after a ranked search and after reading every
    document word for word are different claims, and the second is stronger."""
    rig["pool"] = [_chunk(sim=0.01)]
    rig["deep"] = []
    res = _ask(rig, cfg={"similarity_threshold": 0.5})
    assert res["answer"] != _pl.NOT_FOUND_ANSWER
    assert "word for word" in res["answer"]


def test_a_refusal_without_a_scan_makes_the_smaller_claim(rig):
    rig["pool"] = [_chunk(sim=0.01)]
    res = _pl.ask("u", "q", [], _make_cfg(similarity_threshold=0.5), deep_search=None)
    assert res["answer"] == _pl.NOT_FOUND_ANSWER


def test_the_refusal_still_starts_with_the_phrase_everything_matches_on(rig):
    """`_refused()` and the frontend's not-found styling both match the prefix.
    Rewording the front of this string breaks the escalation's own trigger."""
    assert _pl._refusal_text(True).startswith(_pl.NOT_FOUND_ANSWER)
    assert _pl._refused(_pl._refusal_text(True))


def test_an_unescalated_answer_reports_no_escalation(rig):
    """`grade=False` is what the chat route uses, so this is the shape the UI
    actually receives."""
    res = _ask(rig, grade=False)
    assert res["eval"]["escalated"] is None
