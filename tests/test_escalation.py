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


def test_a_tool_switched_off_is_never_reached_for(rig):
    """The switch removes the tool; it does not merely stop forcing it. There
    is no mode where the visitor makes a tool run — only one where they take it
    away."""
    rig["answers"] = [_pl.NOT_FOUND_ANSWER, GOOD]
    res = _pl.ask("u", "q", [], _make_cfg(), deep_search=None, web_search=None)
    assert rig["scans"] == 0
    assert rig["generations"] == 1
    assert res["not_found"] is True
    assert res["answer"] == _pl.NOT_FOUND_ANSWER, "no tool ran, so no larger claim"


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


def test_a_tool_is_spent_once_across_both_escalations(rig):
    """Escalation 1 uses the scan, the model still refuses, and escalation 2
    must not run the same tool again for the same answer."""
    rig["pool"] = [_chunk(sim=0.01)]
    rig["answers"] = [_pl.NOT_FOUND_ANSWER]
    res = _ask(rig, cfg={"similarity_threshold": 0.5})
    assert rig["scans"] == 1
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
    res = _pl.ask("u", "q", [], _make_cfg(similarity_threshold=0.5),
                  deep_search=None, web_search=None)
    assert res["answer"] == _pl.NOT_FOUND_ANSWER


def test_the_refusal_still_starts_with_the_phrase_everything_matches_on(rig):
    """`_refused()` and the frontend's not-found styling both match the prefix.
    Rewording the front of this string breaks the escalation's own trigger."""
    for used in ([], ["deep"], ["web"], ["deep", "web"]):
        assert _pl._refusal_text(used).startswith(_pl.NOT_FOUND_ANSWER)
        assert _pl._refused(_pl._refusal_text(used))


def test_an_unescalated_answer_reports_no_escalation(rig):
    """`grade=False` is what the chat route uses, so this is the shape the UI
    actually receives."""
    res = _ask(rig, grade=False)
    assert res["eval"]["escalated"] is None


# ==========================================================================
# The third tool: the web.
#
# Web passages are the only ones in the pool that are NOT the user's own
# documents. That distinction is the app's single promise, so every test below
# is really the same test asked in a different place: does the "this came from
# outside" flag survive?
# ==========================================================================

WEB = "Independent testing put the figure at 11.4 kWh."


def _web_hit(text=WEB, url="https://example.com/report"):
    return {"text": text, "similarity": None, "doc_id": None,
            "title": "Example battery report", "ref": url, "web": True}


@pytest.fixture()
def two_tools(rig):
    """Both tools available; the web one returns a hit, deep search finds
    nothing, so the ladder has to walk past the first rung."""
    rig["deep"] = []
    rig["web_hits"] = [_web_hit()]
    rig["web_calls"] = 0

    def _web(_query):
        rig["web_calls"] += 1
        return list(rig["web_hits"])

    rig["web_tool"] = _web
    return rig


def _ask2(rig, **kw):
    return _pl.ask("u", "servicing interval", [], _make_cfg(**kw.pop("cfg", {})),
                   deep_search=rig["tool"], web_search=rig["web_tool"], **kw)


def test_the_documents_are_searched_before_the_web(two_tools):
    """Reaching outside before reading the user's own material is how the
    deleted web-augmentation feature broke the grounding promise."""
    two_tools["answers"] = [_pl.NOT_FOUND_ANSWER, GOOD]
    _ask2(two_tools)
    assert two_tools["scans"] == 1, "deep search must be tried first"
    assert two_tools["web_calls"] == 1, "and the web only after it found nothing"


def test_the_web_is_not_touched_when_the_documents_answer(two_tools):
    two_tools["deep"] = [_deep_hit()]
    two_tools["answers"] = [_pl.NOT_FOUND_ANSWER, GOOD]
    res = _ask2(two_tools)
    assert two_tools["scans"] == 1
    assert two_tools["web_calls"] == 0, "the documents had it; nothing to look outside for"
    assert res["eval"]["escalated"] == "model_refused"


def test_a_good_answer_touches_neither_tool(two_tools):
    _ask2(two_tools)
    assert two_tools["scans"] == 0 and two_tools["web_calls"] == 0


def test_web_passages_are_marked_as_web_in_the_prompt(two_tools):
    """The model cannot label what it cannot distinguish."""
    two_tools["answers"] = [_pl.NOT_FOUND_ANSWER, GOOD]
    _ask2(two_tools)
    retry_prompt = two_tools["prompts"][1]
    assert WEB in retry_prompt
    assert "WEB —" in retry_prompt, "a web passage reached the model unlabelled"


def test_the_system_prompt_tells_the_model_what_the_marker_means(two_tools):
    """A marker the model was never told about is decoration."""
    assert "WEB —" in _pl.SYSTEM_PROMPT
    assert "not from the user" in _pl.SYSTEM_PROMPT.lower()


def test_the_web_flag_is_per_citation_not_per_answer(two_tools):
    """An answer can rest on a document AND a web page at once, which is
    exactly when saying which is which matters most."""
    two_tools["answers"] = [_pl.NOT_FOUND_ANSWER, "Both [1] and [2] agree."]
    res = _ask2(two_tools)
    cites = res["citations"]
    assert len(cites) == 2, cites
    assert [c["is_web"] for c in cites] == [False, True], (
        "the document and the web page were not told apart"
    )


def test_a_web_passage_carries_its_url_not_a_document_id(two_tools):
    """Giving it a doc_id would let it be mistaken for one of the user's own."""
    two_tools["answers"] = [_pl.NOT_FOUND_ANSWER, "Both [1] and [2] agree."]
    res = _ask2(two_tools)
    web_cite = next(c for c in res["citations"] if c["is_web"])
    assert web_cite["doc_id"] is None
    assert web_cite["ref"].startswith("https://")


def test_the_web_never_votes_in_the_not_found_decision(two_tools):
    """It carries `similarity: None`, so it has no standing in a cosine
    judgement about the user's documents."""
    two_tools["pool"] = [_chunk(sim=0.01)]
    two_tools["answers"] = [GOOD]
    res = _ask2(two_tools, cfg={"similarity_threshold": 0.5})
    assert res["not_found"] is False, "the web hit rescued a below-threshold pool"
    assert res["eval"]["escalated"] == "weak_retrieval"


def test_a_refusal_that_tried_both_says_both(two_tools):
    two_tools["web_hits"] = []
    two_tools["answers"] = [_pl.NOT_FOUND_ANSWER]
    res = _ask2(two_tools)
    assert "word for word" in res["answer"] and "searched the web" in res["answer"]


def test_the_ladder_still_costs_at_most_one_extra_generation(two_tools):
    """Two tools widen the ladder; they must not deepen it. `_reach` stops at
    the first tool that finds something, so the retry count is unchanged."""
    two_tools["answers"] = [_pl.NOT_FOUND_ANSWER, _pl.NOT_FOUND_ANSWER, GOOD]
    _ask2(two_tools)
    assert two_tools["generations"] == 2, "a third generation is an unbounded loop"


def test_a_broken_web_tool_falls_through_to_the_refusal(two_tools):
    def _boom(_q):
        two_tools["web_calls"] += 1
        raise RuntimeError("tavily 503")

    two_tools["web_tool"] = _boom
    two_tools["answers"] = [_pl.NOT_FOUND_ANSWER]
    res = _ask2(two_tools)
    assert two_tools["web_calls"] == 1
    assert res["not_found"] is True


def test_tools_used_names_the_tools_and_not_the_citations(two_tools):
    """It shipped as [1].

    `ask()` already had a local called `used` holding the citation markers
    parsed out of the answer, and the tool list was given the same name — so a
    perfectly ordinary answer citing [1] overwrote the record of which tools
    had run. Both are small lists, neither is type-checked, and nothing failed.
    """
    two_tools["answers"] = [_pl.NOT_FOUND_ANSWER, "Both [1] and [2] agree."]
    res = _ask2(two_tools)
    assert res["eval"]["tools_used"] == ["deep", "web"], res["eval"]["tools_used"]


def test_an_answer_that_needed_no_tool_reports_none_used(two_tools):
    res = _ask2(two_tools, grade=False)
    assert res["eval"]["tools_used"] == []
