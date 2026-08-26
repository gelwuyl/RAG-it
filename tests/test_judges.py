"""Regression guard for the 'always FAIL' eval bug.

Two independent defects made both live metrics render as a confident FAIL on
every answer:

1. ``JUDGE_MODEL`` was a module-level constant read from ``RAG_LLM_MODEL`` (the
   boot env default) rather than the live config. On this deployment the env
   value was the bare ``gemma-4-26b-it`` while the served id is
   ``models/gemma-4-26b-a4b-it``, so every judge call 404'd.
2. ``_parse_verdict`` failed CLOSED: an empty or unparseable reply returned
   ``(False, "")``, indistinguishable from a genuine hallucination finding.
   With ``max_tokens=96``, a thinking-capable model could burn the whole budget
   on reasoning and return empty content — permanent FAIL/FAIL.

The contract now: no verdict => JudgeError => the caller records None
("not graded"), never False ("graded, failed").
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from eval import judges
from eval.judges import JudgeError, _parse_verdict


# ---------- verdict parsing ----------

def test_parses_well_formed_pass():
    ok, reason = _parse_verdict("VERDICT: PASS\nREASON: Every claim maps to source [1].")
    assert ok is True
    assert "source [1]" in reason


def test_parses_well_formed_fail():
    ok, reason = _parse_verdict("VERDICT: FAIL\nREASON: The date is not in the context.")
    assert ok is False
    assert "date" in reason


def test_empty_reply_raises_instead_of_failing_closed():
    """The core bug: empty output must NOT be reported as a failed answer."""
    with pytest.raises(JudgeError):
        _parse_verdict("")


def test_reply_without_verdict_raises():
    with pytest.raises(JudgeError):
        _parse_verdict("I think the answer looks reasonable overall.")


def test_reasoning_only_reply_raises():
    """A thinking model that spends its whole token budget reasoning."""
    with pytest.raises(JudgeError):
        _parse_verdict("<thinking>Let me check each claim against the context...</thinking>")


def test_reasoning_wrapper_is_stripped_before_parsing():
    out = (
        "<thinking>The answer says X. Context says X. So this should not FAIL.</thinking>\n"
        "VERDICT: PASS\nREASON: Supported by [2]."
    )
    ok, reason = _parse_verdict(out)
    assert ok is True, "a FAIL mentioned inside the reasoning must not flip the verdict"
    assert "Supported by [2]" in reason


def test_truncated_reasoning_block_is_ungraded_not_failed():
    # max_tokens cut the reply mid-thought, so the wrapper is never closed. The
    # closed-tag stripper leaves it intact, and the stray "FAIL" inside the
    # trace used to be parsed as a real verdict -> a confident hallucination
    # finding produced by a grader that never actually graded.
    out = "<thought>The answer says 5.1 kWh. If it did not match I would FAIL it. Let me check"
    with pytest.raises(JudgeError):
        _parse_verdict(out)


def test_truncated_reasoning_after_real_verdict_keeps_verdict():
    # A complete verdict followed by an unterminated trace must still grade.
    out = "VERDICT: PASS\nREASON: Supported by [2].\n<thought>Double-checking the"
    ok, reason = _parse_verdict(out)
    assert ok is True
    assert "Supported by [2]" in reason


def test_labelled_verdict_wins_over_stray_token():
    out = "The grader was told to answer PASS or FAIL.\nVERDICT: FAIL\nREASON: Unsupported figure."
    ok, _ = _parse_verdict(out)
    assert ok is False


def test_reason_is_truncated_to_first_sentence():
    ok, reason = _parse_verdict(
        "VERDICT: PASS\nREASON: First sentence here. Second sentence rambles on."
    )
    assert reason == "First sentence here."


# ---------- judge model resolution ----------

def test_judge_model_follows_live_config(monkeypatch):
    """The judge must use the model the app actually generates with."""
    class _Cfg:
        llm_model = "models/gemma-4-26b-a4b-it"

    monkeypatch.setattr(judges, "load_config", lambda: _Cfg())
    assert judges.judge_model() == "models/gemma-4-26b-a4b-it"


def test_judge_model_falls_back_when_config_unavailable(monkeypatch):
    def _boom():
        raise RuntimeError("no DB")

    monkeypatch.setattr(judges, "load_config", _boom)
    # Must still return something usable rather than raising.
    assert judges.judge_model() == judges.settings.default_llm_model


def test_judge_raises_on_empty_content(monkeypatch):
    """An empty completion is a broken grader, not a failed answer."""
    class _Msg:
        content = ""

    class _Choice:
        message = _Msg()
        finish_reason = "length"

    class _Resp:
        choices = [_Choice()]

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _Resp()

    monkeypatch.setattr(judges, "openai_client", lambda: _Client())
    monkeypatch.setattr(judges, "judge_model", lambda: "some-model")
    with pytest.raises(JudgeError) as exc:
        judges._judge("prompt")
    assert "empty content" in str(exc.value)


def test_judge_raises_on_model_404(monkeypatch):
    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("Error code: 404 - NOT_FOUND")

    monkeypatch.setattr(judges, "openai_client", lambda: _Client())
    monkeypatch.setattr(judges, "judge_model", lambda: "gemma-4-26b-it")
    with pytest.raises(JudgeError) as exc:
        judges._judge("prompt")
    assert "404" in str(exc.value)


def test_empty_answer_is_a_real_fail_not_an_error():
    """Grading an empty answer is still a legitimate FAIL, not a judge outage."""
    assert judges.faithfulness("q", "ctx", "   ") == (False, "empty answer")
    assert judges.answer_relevancy("q", "  ") == (False, "empty answer")


# ---------- live-grade retries ----------


def test_live_retry_only_calls_the_judge_without_a_verdict(monkeypatch):
    from ragchat import pipeline

    calls = {"faithful": 0, "relevant": 0}

    def _faithful(*args):
        calls["faithful"] += 1
        return False, "unsupported figure"

    def _relevant(*args):
        calls["relevant"] += 1
        return True, "on-topic"

    class _Cfg:
        eval_show = True

    monkeypatch.setattr(judges, "faithfulness", _faithful)
    monkeypatch.setattr(judges, "answer_relevancy", _relevant)
    monkeypatch.setattr(judges, "context_relevance", lambda *a: (True, "on point"))
    monkeypatch.setattr(judges, "context_sufficiency", lambda *a: (True, "enough"))
    monkeypatch.setattr(
        judges, "synthesize_expected", lambda *a: ("the expected answer.", "")
    )
    monkeypatch.setattr(judges, "answer_correctness", lambda *a: (True, "matches"))
    out = pipeline.grade_answer(
        "q",
        "a",
        "ctx",
        _Cfg(),
        {"faithful": False, "faithful_reason": "unsupported figure", "relevant": None},
    )

    assert calls == {"faithful": 0, "relevant": 1}
    assert out["faithful"] is False
    assert out["relevant"] is True
    # The two context checks grade the SAME stored context, so a completed
    # verdict is preserved exactly like the answer judges' is.
    assert out["context_relevance"] is True
    assert out["context_sufficiency"] is True


def test_live_judge_error_identifies_the_missing_metric(monkeypatch):
    from ragchat import pipeline

    class _Cfg:
        eval_show = True

    monkeypatch.setattr(judges, "faithfulness", lambda *args: (True, "supported"))
    monkeypatch.setattr(judges, "context_relevance", lambda *a: (True, "on point"))
    monkeypatch.setattr(judges, "context_sufficiency", lambda *a: (True, "enough"))
    monkeypatch.setattr(
        judges, "synthesize_expected", lambda *a: ("the expected answer.", "")
    )
    monkeypatch.setattr(judges, "answer_correctness", lambda *a: (True, "matches"))

    def _broken_relevancy(*args):
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(judges, "answer_relevancy", _broken_relevancy)
    out = pipeline.grade_answer("q", "a", "ctx", _Cfg())

    assert out["faithful"] is True and out["relevant"] is None
    assert "Answer relevancy unavailable: 429 rate limited" in out["judge_error"]
    assert "Faithfulness unavailable" not in out["judge_error"]


def test_context_checks_are_reported_when_they_fail(monkeypatch):
    """A FAIL from a reference-free proxy is a real verdict, not an outage."""
    from ragchat import pipeline

    class _Cfg:
        eval_show = True

    monkeypatch.setattr(judges, "faithfulness", lambda *a: (True, "supported"))
    monkeypatch.setattr(judges, "answer_relevancy", lambda *a: (True, "on-topic"))
    monkeypatch.setattr(judges, "context_relevance", lambda *a: (False, "mostly unrelated"))
    monkeypatch.setattr(judges, "context_sufficiency", lambda *a: (True, "enough"))
    monkeypatch.setattr(
        judges, "synthesize_expected", lambda *a: ("the expected answer.", "")
    )
    monkeypatch.setattr(judges, "answer_correctness", lambda *a: (True, "matches"))
    out = pipeline.grade_answer("q", "a", "ctx", _Cfg())

    assert out["context_relevance"] is False
    assert out["context_relevance_reason"] == "mostly unrelated"
    assert out["context_sufficiency"] is True
    assert "judge_error" not in out


# ---------- estimated correctness (option 3) ----------


def test_correctness_scores_against_a_drafted_reference(monkeypatch):
    """No gold answer exists for an arbitrary chat; the judge drafts one from
    the passages and scores against it. The draft is persisted so a retry does
    not re-purchase it."""
    from ragchat import pipeline

    class _Cfg:
        eval_show = True

    seen = {}

    def _synth(question, context):
        seen["synth_ctx"] = context
        return "The fridge must read 1-4C.", ""

    def _correct(question, expected, answer):
        seen["expected_used"] = expected
        return False, "range mismatch"

    monkeypatch.setattr(judges, "faithfulness", lambda *a: (True, "supported"))
    monkeypatch.setattr(judges, "answer_relevancy", lambda *a: (True, "on-topic"))
    monkeypatch.setattr(judges, "context_relevance", lambda *a: (True, "on point"))
    monkeypatch.setattr(judges, "context_sufficiency", lambda *a: (True, "enough"))
    monkeypatch.setattr(judges, "synthesize_expected", _synth)
    monkeypatch.setattr(judges, "answer_correctness", _correct)
    out = pipeline.grade_answer("q", "a", "the passages", _Cfg())

    assert out["correct"] is False
    assert seen["expected_used"] == "The fridge must read 1-4C."
    # The drafter never sees the system's answer — it reads the passages.
    assert "the passages" in seen["synth_ctx"]
    assert out["expected_answer"] == "The fridge must read 1-4C."
    assert "judge_error" not in out


def test_no_derivable_reference_is_a_graded_not_an_outage(monkeypatch):
    """When the passages cannot answer the question at all, the synthesized
    'no answer' ruling is a completed FAIL verdict for correctness — the retry
    must not re-run synthesis or report a judge_error."""
    from ragchat import pipeline

    class _Cfg:
        eval_show = True

    calls = {"n": 0}

    def _no_answer(*a):
        calls["n"] += 1
        return "", pipeline.NO_ANSWER_DERIVABLE

    monkeypatch.setattr(judges, "faithfulness", lambda *a: (True, "supported"))
    monkeypatch.setattr(judges, "answer_relevancy", lambda *a: (True, "on-topic"))
    monkeypatch.setattr(judges, "context_relevance", lambda *a: (False, "unrelated"))
    monkeypatch.setattr(judges, "context_sufficiency", lambda *a: (False, "not enough"))
    monkeypatch.setattr(judges, "synthesize_expected", _no_answer)

    fresh = pipeline.grade_answer("q", "a", "empty-ish context", _Cfg())
    assert fresh["correct"] is False
    assert fresh["expected_reason"] == pipeline.NO_ANSWER_DERIVABLE
    assert "judge_error" not in fresh
    assert calls["n"] == 1

    # A retry resumes the stored verdict without spending another call.
    resumed = pipeline.grade_answer("q", "a", "empty-ish context", _Cfg(), previous={
        "expected_answer": "", "expected_reason": pipeline.NO_ANSWER_DERIVABLE})
    assert resumed["correct"] is False
    assert calls["n"] == 1
    assert "judge_error" not in resumed


def test_synthesis_outage_is_named_and_retry_heals(monkeypatch):
    """A failed DRAFT is an outage, not a verdict: it names the cause, stays
    None, and a healed retry finishes correctness without re-running the
    already-finished judges."""
    from ragchat import pipeline

    class _Cfg:
        eval_show = True

    judges.faithfulness  # keep the import referenced even if stubs change
    monkeypatch.setattr(judges, "faithfulness", lambda *a: (True, "supported"))
    monkeypatch.setattr(judges, "answer_relevancy", lambda *a: (True, "on-topic"))
    monkeypatch.setattr(judges, "context_relevance", lambda *a: (True, "on point"))
    monkeypatch.setattr(judges, "context_sufficiency", lambda *a: (True, "enough"))

    def _boom(*a):
        raise RuntimeError("draft call 504")

    monkeypatch.setattr(judges, "synthesize_expected", _boom)
    out = pipeline.grade_answer("q", "a", "ctx", _Cfg())

    assert out["correct"] is None
    assert out["expected_reason"] == "draft call 504"
    assert "Answer correctness unavailable: draft call 504" in out["judge_error"]

    finished_judge_calls = {"n": 0}

    def _counting_faithful(*a):
        finished_judge_calls["n"] += 1
        return True, "supported"

    monkeypatch.setattr(judges, "faithfulness", _counting_faithful)
    monkeypatch.setattr(
        judges, "synthesize_expected", lambda *a: ("drafted text.", "")
    )
    monkeypatch.setattr(judges, "answer_correctness", lambda *a: (True, "matches"))

    healed = pipeline.grade_answer("q", "a", "ctx", _Cfg(), previous={
        "faithful": True, "faithful_reason": "supported",
        "relevant": True, "relevant_reason": "on-topic",
        "context_relevance": True, "context_relevance_reason": "on point",
        "context_sufficiency": True, "context_sufficiency_reason": "enough",
        "expected_answer": "", "expected_reason": "draft call 504"})

    assert healed["correct"] is True
    assert finished_judge_calls["n"] == 0, "finished judges are never re-run"
    assert "judge_error" not in healed
