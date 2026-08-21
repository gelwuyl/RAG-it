"""Query rewriting must yield a QUERY, not a reasoning trace.

The configured chat model (models/gemma-4-26b-a4b-it) is thinking-capable and
answers the rewrite prompt with ~1400 characters of <thought> followed by the
one-line query — despite being told to reply with the query and nothing else.
That whole blob used to be returned raw, so it became the effective query: it
was embedded for retrieval, handed to generation as the question, and shown to
the judge as the "Question" field.

The symptoms were an answer that echoed the rewritten question back instead of
answering it, and a judge reporting that the question "contains a thought
process and a final query". Retrieval was silently poisoned on EVERY follow-up,
since rewriting only fires once a conversation has history.

Nothing here touches the network.

Run:  .venv/Scripts/python -m pytest tests/test_rewrite.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from ragchat.pipeline import _MAX_REWRITE_CHARS, _clean_rewrite, rewrite_query

ORIGINAL = "And what does oat milk add?"
WANTED = "How much does oat milk cost extra at Meridian Coffee?"

# Captured verbatim from the live model, trimmed in the middle. The shape is
# what matters: a closed <thought> block with the query immediately after the
# closing tag and no newline between them.
REAL_REPLY = (
    "<thought>*   Input: A conversation between a user and an assistant.\n"
    "    *   Task: Rewrite the user's latest question into a standalone search query.\n"
    '    *   The "what" refers to the cost/price (based on the previous context).\n'
    f'    *   "{WANTED}" is a clear, standalone search query.\n'
    f'    *   "{WANTED}"</thought>{WANTED}'
)


def test_the_real_model_reply_yields_only_the_query():
    assert _clean_rewrite(REAL_REPLY, ORIGINAL) == WANTED


def test_reasoning_never_survives_into_the_query():
    """The whole point: no trace text may reach the embedder."""
    out = _clean_rewrite(REAL_REPLY, ORIGINAL)
    for leak in ("<thought>", "Input:", "Task:", "standalone search query."):
        assert leak not in out
    assert len(out) < 200


def test_plain_reply_passes_through():
    assert _clean_rewrite(WANTED, ORIGINAL) == WANTED


@pytest.mark.parametrize("label", ["Query:", "query:", "Rewritten query:", "Search query:"])
def test_label_prefixes_are_removed(label):
    assert _clean_rewrite(f"{label} {WANTED}", ORIGINAL) == WANTED


@pytest.mark.parametrize("wrapped", [f'"{WANTED}"', f"'{WANTED}'", f"`{WANTED}`", f"**{WANTED}**"])
def test_surrounding_quotes_and_markdown_are_removed(wrapped):
    assert _clean_rewrite(wrapped, ORIGINAL) == WANTED


# --------------------------------------------------------------------------
# Fallbacks. Every one of these must return the ORIGINAL query — rewriting is
# an optimization for follow-ups, never a requirement for answering.
# --------------------------------------------------------------------------


def test_unterminated_wrapper_falls_back():
    """max_tokens ran out mid-reasoning, so there is no query after the block.

    This is the case that defeated the judges: a pattern requiring a closing tag
    lets the entire trace through. Critically, the wrapper's INNER text must not
    be recovered here — that text is the reasoning, and using it as the search
    query is the bug itself. _clean_answer does recover it, which is why this
    path does not reuse it.
    """
    truncated = "<thought>Let me think about what the user means by 'add'. The"
    assert _clean_rewrite(truncated, ORIGINAL) == ORIGINAL


def test_fully_wrapped_reply_falls_back_rather_than_using_the_reasoning():
    wrapped = "<thinking>The user wants the oat milk surcharge</thinking>"
    assert _clean_rewrite(wrapped, ORIGINAL) == ORIGINAL


@pytest.mark.parametrize("junk", ["", "   ", "\n\n"])
def test_empty_replies_fall_back(junk):
    assert _clean_rewrite(junk, ORIGINAL) == ORIGINAL


def test_overlong_reply_falls_back():
    """A paragraph is not a search query, however clean it looks."""
    assert _clean_rewrite("x" * (_MAX_REWRITE_CHARS + 1), ORIGINAL) == ORIGINAL


def test_leftover_angle_bracket_falls_back():
    """A surviving tag fragment means the strip did not fully work; refuse
    rather than embed markup as a query."""
    assert _clean_rewrite("<reasoning_content>the price", ORIGINAL) == ORIGINAL


# --------------------------------------------------------------------------
# rewrite_query's own guards — these must not spend an LLM call at all.
# --------------------------------------------------------------------------


class _Cfg:
    def __init__(self, query_rewrite=True, llm_model="m"):
        self.query_rewrite = query_rewrite
        self.llm_model = llm_model


def _explode(*a, **k):
    raise AssertionError("no model call should happen here")


def test_no_history_means_no_rewrite_and_no_call(monkeypatch):
    monkeypatch.setattr("ragchat.pipeline._chat", _explode)
    assert rewrite_query(ORIGINAL, [], _Cfg()) == ORIGINAL


def test_disabled_means_no_rewrite_and_no_call(monkeypatch):
    monkeypatch.setattr("ragchat.pipeline._chat", _explode)
    history = [{"role": "user", "content": "prices?"}]
    assert rewrite_query(ORIGINAL, history, _Cfg(query_rewrite=False)) == ORIGINAL


def test_model_failure_falls_back_to_the_original(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("proxy 503")

    monkeypatch.setattr("ragchat.pipeline._chat", _boom)
    history = [{"role": "user", "content": "prices?"}]
    assert rewrite_query(ORIGINAL, history, _Cfg()) == ORIGINAL


def test_end_to_end_rewrite_strips_the_trace(monkeypatch):
    monkeypatch.setattr("ragchat.pipeline._chat", lambda *a, **k: REAL_REPLY)
    history = [{"role": "user", "content": "What are the menu prices?"}]
    assert rewrite_query(ORIGINAL, history, _Cfg()) == WANTED


# --------------------------------------------------------------------------
# The prompt itself carries a correctness requirement, so it is asserted.
#
# Told only to "resolve pronouns and references", the model resolved ones that
# were never there: after two turns about a solar battery, "What is the boiler
# pressure range for the espresso machine?" came back — deterministically — as
# "...for the SunPak 5 espresso machine". Retrieval then hunted for a product
# that does not exist and the reader was told the fact was not in their
# documents, while it sat in the corpus a paragraph away from an answer they
# had already been given.
#
# Asserting on prompt text is unusual and deliberate. The instruction is load
# bearing and invisible: nothing else in the suite fails if it is dropped,
# because the bug needs a live model AND several turns to appear.
# --------------------------------------------------------------------------


def _captured_prompt(monkeypatch) -> str:
    seen = {}

    def _capture(model, messages, temperature):
        seen["p"] = messages[0]["content"]
        return "anything"

    monkeypatch.setattr("ragchat.pipeline._chat", _capture)
    rewrite_query(
        "What is the boiler pressure range for the espresso machine?",
        [{"role": "user", "content": "How long is the SunPak warranty?"},
         {"role": "assistant", "content": "10 years."}],
        _Cfg(),
    )
    return seen["p"]


def test_the_prompt_tells_the_model_to_leave_a_standalone_question_alone(monkeypatch):
    p = _captured_prompt(monkeypatch).lower()
    assert "unchanged" in p, (
        "the rewrite prompt no longer tells the model to return a standalone "
        "question unchanged — see this section's comment for what that costs"
    )


def test_the_prompt_forbids_importing_a_subject_from_an_earlier_turn(monkeypatch):
    p = _captured_prompt(monkeypatch).lower()
    assert "never add" in p, (
        "the rewrite prompt no longer forbids adding a subject the latest "
        "question does not refer to — this is the SunPak-espresso-machine bug"
    )


def test_the_conversation_and_the_question_still_reach_the_model(monkeypatch):
    """The guards above must not have crowded out what the step is for."""
    p = _captured_prompt(monkeypatch)
    assert "SunPak" in p and "boiler pressure range" in p
