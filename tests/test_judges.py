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
