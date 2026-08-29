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
    """A dedicated judge_model config field wins; it is not the answerer's job."""

    class _Cfg:
        llm_model = "models/gemma-4-26b-a4b-it"
        judge_model = "models/gemini-3.5-flash-lite"

    monkeypatch.setattr(judges, "load_config", lambda: _Cfg())
    assert judges.judge_model() == "models/gemini-3.5-flash-lite"


def test_empty_judge_model_grades_with_the_answerer(monkeypatch):
    """"Empty means the answerer grades" — the historical behaviour, kept as the
    explicit fallback choice in Settings."""

    class _Cfg:
        llm_model = "models/gemma-4-26b-a4b-it"
        judge_model = ""

    monkeypatch.setattr(judges, "load_config", lambda: _Cfg())
    assert judges.judge_model() == "models/gemma-4-26b-a4b-it"


def test_judge_field_tolerates_an_older_config_object(monkeypatch):
    """Configs built before the field existed (stale callers, test doubles) must
    not break judge resolution — getattr, not attribute access."""

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
        return False, 0.0, "unsupported figure"

    def _relevant(*args):
        calls["relevant"] += 1
        return True, 1.0, "on-topic"

    class _Cfg:
        eval_show = True

    monkeypatch.setattr(judges, "faithfulness_scored", _faithful)
    monkeypatch.setattr(judges, "answer_relevancy_scored", _relevant)
    monkeypatch.setattr(judges, "context_precision_scored", lambda *a: (True, 1.0, "on point"))
    monkeypatch.setattr(judges, "context_recall_scored", lambda *a: (True, 1.0, "enough"))
    monkeypatch.setattr(
        judges, "synthesize_expected", lambda *a: ("the expected answer.", "")
    )
    out = pipeline.grade_answer(
        "q",
        "a",
        "ctx",
        _Cfg(),
        {"faithful": False, "faithful_reason": "unsupported figure", "relevant": None},
        passages=["p1", "p2"],
    )

    assert calls == {"faithful": 0, "relevant": 1}
    assert out["faithful"] is False
    assert out["relevant"] is True
    # The two retrieval readings grade the SAME stored material — the ordered
    # pool for precision, the context text for recall — so a completed verdict
    # is preserved exactly like the answer judges' is.
    assert out["context_precision"] is True
    assert out["context_recall"] is True


def test_live_judge_error_identifies_the_missing_metric(monkeypatch):
    from ragchat import pipeline

    class _Cfg:
        eval_show = True

    monkeypatch.setattr(judges, "faithfulness_scored", lambda *args: (True, 1.0, "supported"))
    monkeypatch.setattr(judges, "context_precision_scored", lambda *a: (True, 1.0, "on point"))
    monkeypatch.setattr(judges, "context_recall_scored", lambda *a: (True, 1.0, "enough"))
    monkeypatch.setattr(
        judges, "synthesize_expected", lambda *a: ("the expected answer.", "")
    )

    def _broken_relevancy(*args):
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(judges, "answer_relevancy_scored", _broken_relevancy)
    out = pipeline.grade_answer("q", "a", "ctx", _Cfg(), passages=["p1"])

    assert out["faithful"] is True and out["relevant"] is None
    assert "Answer relevancy unavailable: 429 rate limited" in out["judge_error"]
    assert "Faithfulness unavailable" not in out["judge_error"]


def test_context_readings_are_reported_when_they_fail(monkeypatch):
    """A FAIL from a retrieval judge is a real verdict, not an outage."""
    from ragchat import pipeline

    class _Cfg:
        eval_show = True

    monkeypatch.setattr(judges, "faithfulness_scored", lambda *a: (True, 1.0, "supported"))
    monkeypatch.setattr(judges, "answer_relevancy_scored", lambda *a: (True, 1.0, "on-topic"))
    monkeypatch.setattr(judges, "context_precision_scored", lambda *a: (False, 0.0, "relevant material ranked last"))
    monkeypatch.setattr(judges, "context_recall_scored", lambda *a: (True, 1.0, "enough"))
    monkeypatch.setattr(
        judges, "synthesize_expected", lambda *a: ("the expected answer.", "")
    )
    out = pipeline.grade_answer("q", "a", "ctx", _Cfg(), passages=["p1"])

    assert out["context_precision"] is False
    assert out["context_precision_reason"] == "relevant material ranked last"
    assert out["context_recall"] is True
    assert "judge_error" not in out


def test_scored_judges_store_their_reading_for_the_bar(monkeypatch):
    """The scorecard draws bars from the 0-1 reading (65% fills to 65 against
    the benchmark tick); a binary pass rendering as 100% was indistinguishable
    from a measurement. The score rides next to each verdict in eval_data."""
    from ragchat import pipeline

    class _Cfg:
        eval_show = True

    monkeypatch.setattr(judges, "faithfulness_scored", lambda *a: (True, 0.75, "3 of 4 claims"))
    monkeypatch.setattr(judges, "answer_relevancy_scored", lambda *a: (True, 0.9, "direct"))
    monkeypatch.setattr(judges, "context_precision_scored", lambda *a: (True, 0.65, "mixed"))
    monkeypatch.setattr(judges, "context_recall_scored", lambda *a: (True, 0.8, "enough"))
    monkeypatch.setattr(judges, "synthesize_expected", lambda *a: ("expected.", ""))
    out = pipeline.grade_answer("q", "a", "ctx", _Cfg(), passages=["p1"])

    assert out["faithful_score"] == 0.75
    assert out["relevant_score"] == 0.9
    assert out["context_precision_score"] == 0.65
    assert out["context_recall_score"] == 0.8
    assert out["context_recall"] is True
    # A judge that omits its SCORE line degrades to the verdict, not an error.
    monkeypatch.setattr(judges, "faithfulness_scored", lambda *a: (True, 1.0, "no line"))
    out2 = pipeline.grade_answer("q", "a", "ctx", _Cfg(), passages=["p1"])
    assert out2["faithful_score"] == 1.0
    assert "judge_error" not in out2


def test_precision_without_stored_passages_is_a_data_gap_not_a_failure(monkeypatch):
    """Context precision grades the ORDERED pool, and answers stored before
    that order was persisted have none. It must sit honestly ungraded with a
    reason — never a verdict, never a broken-grader claim about the other
    judges."""
    from ragchat import pipeline

    class _Cfg:
        eval_show = True

    called = {"precision": 0}

    monkeypatch.setattr(judges, "faithfulness_scored", lambda *a: (True, 1.0, "supported"))
    monkeypatch.setattr(judges, "answer_relevancy_scored", lambda *a: (True, 1.0, "on-topic"))
    monkeypatch.setattr(judges, "context_recall_scored", lambda *a: (True, 1.0, "enough"))
    monkeypatch.setattr(judges, "synthesize_expected", lambda *a: ("expected.", ""))

    def _no_call(*a):
        called["precision"] += 1
        return True, 1.0, "should not run"

    monkeypatch.setattr(judges, "context_precision_scored", _no_call)
    out = pipeline.grade_answer("q", "a", "ctx", _Cfg())

    assert called["precision"] == 0
    assert out["context_precision"] is None
    assert "ordered passages" in out["context_precision_reason"]
    # The gap is TERMINAL: marked so the /grade route spends no retry budget
    # on it, and kept out of judge_error — it is a data gap, not a broken
    # grader.
    assert out["precision_data_gap"] is True
    assert out["context_recall"] is True
    assert "judge_error" not in out


# ---------- the reference-backed recall pair (provenance) ----------


def test_recall_scores_against_a_drafted_reference_marked_estimated(monkeypatch):
    """No gold answer exists for an arbitrary chat; the judge drafts one from
    the passages and grades the retrieval against it. The draft is persisted so
    a retry does not re-purchase it, and its provenance is "draft" — the UI
    renders the retrieval rows "estimated" from that marker."""
    from ragchat import pipeline

    class _Cfg:
        eval_show = True

    seen = {}

    def _synth(question, context):
        seen["synth_ctx"] = context
        return "The fridge must read 1-4C.", ""

    def _recall(question, reference, context):
        seen["reference_used"] = reference
        seen["recall_ctx"] = context
        return False, 0.4, "missing a claim"

    monkeypatch.setattr(judges, "faithfulness_scored", lambda *a: (True, 1.0, "supported"))
    monkeypatch.setattr(judges, "answer_relevancy_scored", lambda *a: (True, 1.0, "on-topic"))
    monkeypatch.setattr(judges, "context_precision_scored", lambda *a: (True, 1.0, "on point"))
    monkeypatch.setattr(judges, "synthesize_expected", _synth)
    monkeypatch.setattr(judges, "context_recall_scored", _recall)
    out = pipeline.grade_answer("q", "a", "the passages", _Cfg(), passages=["p1"])

    assert out["context_recall"] is False
    assert seen["reference_used"] == "The fridge must read 1-4C."
    # The drafter never sees the system's answer — it reads the passages.
    assert "the passages" in seen["synth_ctx"]
    # The recall judge grades the CONTEXT, not the answer.
    assert seen["recall_ctx"] == "the passages"
    assert out["expected_answer"] == "The fridge must read 1-4C."
    assert out["expected_source"] == "draft"
    assert "judge_error" not in out


def test_no_derivable_reference_is_a_graded_not_an_outage(monkeypatch):
    """When the passages cannot answer the question at all, the synthesized
    'no answer' ruling is a completed FAIL-at-0 verdict for context recall —
    nothing of a reference the passages cannot yield can be IN the passages —
    and the retry must not re-run synthesis or report a judge_error."""
    from ragchat import pipeline

    class _Cfg:
        eval_show = True

    calls = {"n": 0}

    def _no_answer(*a):
        calls["n"] += 1
        return "", pipeline.NO_ANSWER_DERIVABLE

    monkeypatch.setattr(judges, "faithfulness_scored", lambda *a: (True, 1.0, "supported"))
    monkeypatch.setattr(judges, "answer_relevancy_scored", lambda *a: (True, 1.0, "on-topic"))
    monkeypatch.setattr(judges, "context_precision_scored", lambda *a: (True, 1.0, "on point"))
    monkeypatch.setattr(judges, "synthesize_expected", _no_answer)

    fresh = pipeline.grade_answer("q", "a", "empty-ish context", _Cfg(), passages=["p1"])
    assert fresh["context_recall"] is False
    assert fresh["context_recall_score"] == 0.0
    assert fresh["expected_reason"] == pipeline.NO_ANSWER_DERIVABLE
    assert "judge_error" not in fresh
    assert calls["n"] == 1

    # A retry resumes the stored verdict without spending another call.
    resumed = pipeline.grade_answer("q", "a", "empty-ish context", _Cfg(),
                                    previous={
                                        "expected_answer": "",
                                        "expected_reason": pipeline.NO_ANSWER_DERIVABLE},
                                    passages=["p1"])
    assert resumed["context_recall"] is False
    assert calls["n"] == 1
    assert "judge_error" not in resumed


def test_refusal_token_quoted_in_a_trace_is_not_a_refusal(monkeypatch):
    """Found on the preview deployment: the drafter reasoned 'maybe reply
    NO_ANSWER_DERIVABLE?' inside a thinking trace and still drafted a real
    reference, but a substring match read the quoted token as a refusal — so an
    answerable question got a None reading with 'context does not contain an
    answer'. The refusal must be the WHOLE reply to count."""
    from eval import judges

    raw = "<thought>Do NOT reply NO_ANSWER_DERIVABLE here.</thought>\nThe fridge must read 1-4C."
    monkeypatch.setattr(judges, "_judge", lambda prompt, max_tokens=512: raw)
    text, reason = judges.synthesize_expected("how cold?", "[1] The fridge must read 1-4C.")
    assert text == "The fridge must read 1-4C."
    assert reason == ""

    # And the genuine all-integer-refusal case still parses.
    monkeypatch.setattr(judges, "_judge", lambda prompt, max_tokens=512: f" {judges.NO_ANSWER_DERIVABLE_SENTINEL.upper()} ")
    text2, reason2 = judges.synthesize_expected("how cold?", "[1] nothing relevant")
    assert text2 == ""
    assert reason2 == judges.NO_ANSWER_DERIVABLE_SENTINEL


def test_synthesis_outage_is_named_and_retry_heals(monkeypatch):
    """A failed DRAFT is an outage, not a verdict: it names the cause, stays
    None, and a healed retry finishes recall without re-running the
    already-finished judges."""
    from ragchat import pipeline

    class _Cfg:
        eval_show = True

    judges.faithfulness  # keep the import referenced even if stubs change
    monkeypatch.setattr(judges, "faithfulness_scored", lambda *a: (True, 1.0, "supported"))
    monkeypatch.setattr(judges, "answer_relevancy_scored", lambda *a: (True, 1.0, "on-topic"))
    monkeypatch.setattr(judges, "context_precision_scored", lambda *a: (True, 1.0, "on point"))

    def _boom(*a):
        raise RuntimeError("draft call 504")

    monkeypatch.setattr(judges, "synthesize_expected", _boom)
    out = pipeline.grade_answer("q", "a", "ctx", _Cfg(), passages=["p1"])

    assert out["context_recall"] is None
    assert out["expected_reason"] == "draft call 504"
    assert "Context recall unavailable: draft call 504" in out["judge_error"]

    finished_judge_calls = {"n": 0}

    def _counting_faithful(*a):
        finished_judge_calls["n"] += 1
        return True, 1.0, "supported"

    monkeypatch.setattr(judges, "faithfulness_scored", _counting_faithful)
    monkeypatch.setattr(
        judges, "synthesize_expected", lambda *a: ("drafted text.", "")
    )
    monkeypatch.setattr(judges, "context_recall_scored", lambda *a: (True, 1.0, "matches"))

    healed = pipeline.grade_answer("q", "a", "ctx", _Cfg(), previous={
        "faithful": True, "faithful_reason": "supported",
        "relevant": True, "relevant_reason": "on-topic",
        "context_precision": True, "context_precision_reason": "on point",
        "expected_answer": "", "expected_reason": "draft call 504"},
        passages=["p1"])

    assert healed["context_recall"] is True
    assert finished_judge_calls["n"] == 0, "finished judges are never re-run"
    assert "judge_error" not in healed


# ---------- context_precision_scored: rank-aware, fail-open ----------


def _precision_reply(marks, verdict="PASS", reason="marks explain it"):
    lines = [f"PASSAGE {i}: {m}" for i, m in enumerate(marks, start=1)]
    lines.append(f"VERDICT: {verdict}")
    lines.append(f"REASON: {reason}.")
    return "\n".join(lines)


def test_precision_scores_a_perfect_ranking(monkeypatch):
    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: _precision_reply(
            ["RELEVANT", "RELEVANT"])
    )
    ok, score, why = judges.context_precision_scored("q", ["a", "b"])
    assert ok is True
    assert score == 1.0
    assert "marks" in why


def test_precision_punishes_relevant_material_ranked_low(monkeypatch):
    """The whole point of the rank-aware metric: the same one relevant passage
    scores 1.0 when it ranks first and 0.5 when it ranks second."""
    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: _precision_reply(
            ["RELEVANT", "IRRELEVANT"])
    )
    ok_hi, hi, _ = judges.context_precision_scored("q", ["a", "b"])
    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: _precision_reply(
            ["IRRELEVANT", "RELEVANT"])
    )
    ok_lo, lo, _ = judges.context_precision_scored("q", ["a", "b"])
    assert hi == 1.0 and lo == 0.5
    assert ok_hi is True and ok_lo is True  # the 0.5 verdict line sits on PASS


def test_precision_with_no_relevant_passage_scores_zero(monkeypatch):
    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: _precision_reply(
            ["IRRELEVANT", "IRRELEVANT"], verdict="FAIL")
    )
    ok, score, _ = judges.context_precision_scored("q", ["a", "b"])
    assert ok is False and score == 0.0


def test_precision_with_a_partial_mark_set_is_a_broken_grader(monkeypatch):
    """A rank-aware statistic computed over a SUBSET of the pool would silently
    distort the ordering it exists to measure. Missing marks are a truncated
    reply — JudgeError, "not graded", never a score."""
    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: _precision_reply(
            ["RELEVANT"])
    )
    with pytest.raises(JudgeError):
        judges.context_precision_scored("q", ["a", "b"])


def test_precision_without_a_verdict_fails_open(monkeypatch):
    """The fail-open invariant, on the new judge: no verdict line → JudgeError
    → the caller records None + reason, never a confident FAIL."""
    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: "PASSAGE 1: RELEVANT\n"
    )
    with pytest.raises(JudgeError):
        judges.context_precision_scored("q", ["a"])


def test_precision_ignores_marks_inside_a_thinking_trace(monkeypatch):
    """A reasoning model may rehearse marks inside <thought> before emitting
    the real ones; only the lines outside the wrapper may be parsed, or the
    trace's first pass would clobber the final marks."""
    raw = (
        "<thought>First thought: PASSAGE 1: IRRELEVANT</thought>\n"
        "PASSAGE 1: RELEVANT\nVERDICT: PASS\nREASON: supported."
    )
    monkeypatch.setattr(judges, "_judge", lambda p, max_tokens=512: raw)
    _, score, _ = judges.context_precision_scored("q", ["a"])
    assert score == 1.0


def test_precision_on_an_empty_pool_is_measured_not_graded_broken(monkeypatch):
    """Nothing retrieved is a retrieval fact (recall 0's twin), not an outage."""
    ok, score, why = judges.context_precision_scored("q", [])
    assert ok is False and score == 0.0 and "no passages" in why


# ---------- context_recall_scored: reference-backed, fail-open ----------


def test_recall_parses_a_scored_reply(monkeypatch):
    monkeypatch.setattr(
        judges, "_judge",
        lambda p, max_tokens=512: "VERDICT: PASS\nSCORE: 75\nREASON: 3 of 4 claims.",
    )
    ok, score, why = judges.context_recall_scored("q", "the reference", "the context")
    assert ok is True and score == 0.75 and "claims" in why


def test_recall_prompt_carries_reference_and_context(monkeypatch):
    seen = {}
    def _spy(prompt, max_tokens=512):
        seen["prompt"] = prompt
        return "VERDICT: PASS\nSCORE: 100\nREASON: all claims."
    monkeypatch.setattr(judges, "_judge", _spy)
    judges.context_recall_scored("how cold?", "the fridge reads 1-4C", "[1] 1-4C context")
    assert "the fridge reads 1-4C" in seen["prompt"]
    assert "[1] 1-4C context" in seen["prompt"]
    assert "how cold?" in seen["prompt"]


def test_recall_without_a_verdict_fails_open(monkeypatch):
    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: "The context looks decent overall."
    )
    with pytest.raises(JudgeError):
        judges.context_recall_scored("q", "reference", "context")


def test_recall_with_an_empty_reference_is_a_caller_bug_not_a_zero(monkeypatch):
    """An empty reference would grade as 0 and look like a retrieval finding.
    It is a bug upstream — fail open instead of manufacturing a measurement."""
    with pytest.raises(JudgeError):
        judges.context_recall_scored("q", "   ", "context")


def test_recall_on_an_empty_context_is_measured_zero(monkeypatch):
    """Nothing retrieved means nothing of the reference can be in it — that is
    a real 0, not an outage."""
    ok, score, why = judges.context_recall_scored("q", "reference", "   ")
    assert ok is False and score == 0.0 and "empty context" in why


# ---------- faithfulness claim marks (2026-08-29 tightening) ----------


def _claims_reply(marks, verdict="PASS", reason="marks explain it"):
    lines = [f"CLAIM {i}: {m}" for i, m in enumerate(marks, start=1)]
    lines.append(f"VERDICT: {verdict}")
    lines.append(f"REASON: {reason}.")
    return "\n".join(lines)


def test_faithfulness_scored_computes_the_ratio_from_marks(monkeypatch):
    """2 of 3 claims supported -> 0.667, computed in Python from the marks,
    not taken from anything the judge says about a SCORE."""
    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: _claims_reply(
            ["SUPPORTED", "UNSUPPORTED", "SUPPORTED"], verdict="FAIL")
    )
    ok, score, why = judges.faithfulness_scored("q", "ctx", "answer")
    assert ok is False, "verdict is the measured one: not every claim supported"
    assert abs(score - 2 / 3) < 1e-9
    assert "marks" in why


def test_faithfulness_scored_all_supported_is_a_full_pass(monkeypatch):
    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: _claims_reply(
            ["SUPPORTED", "SUPPORTED"])
    )
    ok, score, why = judges.faithfulness_scored("q", "ctx", "answer")
    assert ok is True and score == 1.0


def test_faithfulness_scored_no_claims_refusal_scores_full(monkeypatch):
    """A proper refusal makes no factual claims and cannot be unfaithful to
    anything — the RAGAS convention. Evasion is answer relevancy's job."""
    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: _claims_reply(
            ["NONE"], reason="the answer asserts nothing")
    )
    ok, score, why = judges.faithfulness_scored("q", "ctx", "not in documents")
    assert ok is True and score == 1.0


def test_faithfulness_scored_gap_in_marks_is_a_broken_grader(monkeypatch):
    """A missing claim mark would silently inflate the ratio — JudgeError,
    "not graded", never a score."""
    # Claim 3 exists in the reply but claim 2's line is missing entirely.
    raw = "CLAIM 1: SUPPORTED\nCLAIM 3: SUPPORTED\nVERDICT: PASS\nREASON: x."
    monkeypatch.setattr(judges, "_judge", lambda p, max_tokens=512: raw)
    with pytest.raises(JudgeError):
        judges.faithfulness_scored("q", "ctx", "answer")


def test_faithfulness_scored_conflicting_duplicate_marks_raise(monkeypatch):
    raw = (
        "CLAIM 1: SUPPORTED\nCLAIM 1: UNSUPPORTED\n"
        "VERDICT: PASS\nREASON: x."
    )
    monkeypatch.setattr(judges, "_judge", lambda p, max_tokens=512: raw)
    with pytest.raises(JudgeError):
        judges.faithfulness_scored("q", "ctx", "answer")


def test_faithfulness_scored_none_mixed_with_marks_is_ambiguous(monkeypatch):
    raw = (
        "CLAIM 1: NONE\nCLAIM 2: SUPPORTED\n"
        "VERDICT: PASS\nREASON: x."
    )
    monkeypatch.setattr(judges, "_judge", lambda p, max_tokens=512: raw)
    with pytest.raises(JudgeError):
        judges.faithfulness_scored("q", "ctx", "answer")


def test_faithfulness_scored_without_marks_or_verdict_fails_open(monkeypatch):
    """The core fail-open invariant on the tightened judge: no marks and no
    verdict -> JudgeError -> the caller records None, never a score."""
    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: "Looks well grounded to me."
    )
    with pytest.raises(JudgeError):
        judges.faithfulness_scored("q", "ctx", "answer")


def test_faithfulness_scored_ignores_marks_inside_a_thinking_trace(monkeypatch):
    raw = (
        "<thought>First pass: CLAIM 1: UNSUPPORTED</thought>\n"
        "CLAIM 1: SUPPORTED\nVERDICT: PASS\nREASON: supported."
    )
    monkeypatch.setattr(judges, "_judge", lambda p, max_tokens=512: raw)
    _, score, _ = judges.faithfulness_scored("q", "ctx", "answer")
    assert score == 1.0


def test_boolean_faithfulness_fails_on_any_unsupported_claim(monkeypatch):
    """The benchmark judge shares the scored sibling's meaning: one
    unsupported claim is a FAIL, computed from the marks not the vibe."""
    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: _claims_reply(
            ["SUPPORTED", "SUPPORTED", "UNSUPPORTED"], verdict="PASS")
    )
    ok, why = judges.faithfulness("q", "ctx", "answer")
    assert ok is False

    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: _claims_reply(
            ["SUPPORTED", "SUPPORTED"])
    )
    ok2, _ = judges.faithfulness("q", "ctx", "answer")
    assert ok2 is True


def test_boolean_faithfulness_passes_a_no_claims_refusal(monkeypatch):
    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: _claims_reply(["NONE"])
    )
    ok, _ = judges.faithfulness("q", "ctx", "I couldn't find this in your documents.")
    assert ok is True


def test_boolean_faithfulness_without_marks_fails_open(monkeypatch):
    monkeypatch.setattr(
        judges, "_judge", lambda p, max_tokens=512: "VERDICT: PASS\nREASON: fine."
    )
    with pytest.raises(JudgeError):
        judges.faithfulness("q", "ctx", "answer")


def test_empty_answer_still_a_real_fail_on_the_tightened_judges():
    assert judges.faithfulness_scored("q", "ctx", "  ") == (False, 0.0, "empty answer")
    assert judges.faithfulness("q", "ctx", "  ") == (False, "empty answer")


def test_tightened_prompts_bar_the_benefit_of_the_doubt(monkeypatch):
    """The leniency levers must actually be IN the prompts the judges send."""
    seen = {}

    def _spy(prompt, max_tokens=512):
        seen.setdefault("prompts", []).append(prompt)
        if "CONTEXT PRECISION" in prompt:
            return _precision_reply(["RELEVANT"])
        return _claims_reply(["SUPPORTED"])

    monkeypatch.setattr(judges, "_judge", _spy)
    judges.faithfulness("q", "ctx", "a")
    judges.faithfulness_scored("q", "ctx", "a")
    judges.answer_relevancy("q", "a")
    judges.answer_relevancy_scored("q", "a")
    judges.context_precision_scored("q", ["p1"])
    joined = "\n".join(seen["prompts"])
    assert "benefit of the doubt" in joined
    assert "EXPLICITLY" in joined
    assert "DIFFERENT aspect" in joined, "relevancy must demand specificity"
    assert "NEEDED to answer" in joined, "precision bar must be need, not topic"


def test_both_faithfulness_judges_share_one_prompt_body(monkeypatch):
    """The pairing rule: benchmark verdict and live ratio must MEAN the same
    thing, which falls apart if the two prompts drift apart."""
    seen = []

    def _spy(prompt, max_tokens=512):
        seen.append(prompt)
        return _claims_reply(["SUPPORTED"])

    monkeypatch.setattr(judges, "_judge", _spy)
    judges.faithfulness("q", "ctx", "a")
    judges.faithfulness_scored("q", "ctx", "a")
    scored_body = seen[1].replace(
        "\nSCORE is computed from your marks: SUPPORTED claims divided by "
        "total claims (about half unsupported is 50).", ""
    )
    assert scored_body == seen[0]
