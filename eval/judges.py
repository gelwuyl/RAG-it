"""LLM-as-judge metrics for answer and context quality.

The reference-based correctness judge is used by the benchmark. The live chat
also uses two explicitly reference-free context checks — context relevance and
context sufficiency — plus synthesize_expected, which DRAFTS a reference from
the retrieved passages so arbitrary answers can carry an estimated correctness.
Proxies and synthesized references must not be mistaken for golden-set recall,
precision, or correctness; callers label them estimated in the UI.

All judges reuse the same proxy LLM (qwen3.8-max) per user decision 2026-08-15.
The judge returns PASS/FAIL (or a 0-1 score) plus one line of reasoning.
The eval harness calls the reference-based judges only in the full
(non --retrieval-only) run.

Tightened 2026-08-16: the judge prompts now forbid chain-of-thought and force a
single-line verdict + one-sentence reason. Previously the model could emit a
thinking trace before the verdict, which leaked into the parsed "reason" and
occasionally flipped the verdict parse. We also cap max_tokens so the model
cannot ramble, and the parser only reads the verdict line + first sentence.
"""
from __future__ import annotations

import re

from ragchat.embeddings import openai_client
from ragchat.config import load_config, settings


class JudgeError(RuntimeError):
    """The judge could not produce a verdict (model 404, quota, empty reply).

    Raised instead of silently returning FAIL so callers can distinguish
    "the answer is bad" from "the grader is broken".
    """


# What synthesize_expected returns as the reason when the passages cannot
# answer the question, and what ragchat.pipeline stores in eval_data's
# expected_reason to mark correctness as graded-by-refusal. Kept here because
# the judge is the producer; the pipeline imports this constant rather than
# duplicating a magic string on both sides.
NO_ANSWER_DERIVABLE_SENTINEL = "no answer derivable"


def judge_model() -> str:
    """Model used for LLM-as-judge — resolved from the LIVE config each call.

    This used to be a module-level constant read from ``settings.default_llm_model``,
    i.e. the ``RAG_LLM_MODEL`` env var at import time. That is the *boot default*,
    not what the app is actually generating with: config.yaml (and the DB
    override written by the Settings UI) can name a different model. On this
    deployment the env default was the bare ``gemma-4-26b-it`` while the served
    id is ``models/gemma-4-26b-a4b-it``, so every judge call 404'd and both
    metrics rendered as FAIL. Reading the live config keeps the judge on the
    same model that just answered the question.
    """
    try:
        model = (load_config().llm_model or "").strip()
        if model:
            return model
    except Exception:
        pass
    return settings.default_llm_model

# Strict output contract the judge must follow. We repeat it in every prompt so
# the parser can rely on "first line = verdict, second line = one sentence".
_VERDICT_CONTRACT = (
    "Reply in exactly this format, with NO preamble, NO chain-of-thought, "
    "NO commentary outside these two lines:\n"
    "VERDICT: PASS or FAIL\n"
    "REASON: one short sentence citing the specific evidence.\n"
)

# The live scorecard draws per-answer readings on the same percentage bars the
# benchmark reports pass rates on, so a binary verdict can only ever fill a bar
# at 0% or 100% — indistinguishable from a measurement. The scored variants ask
# for a 0-100 reading too. The BENCHMARK keeps using the boolean judges above:
# its published numbers are pass rates over 53 golden questions, and switching
# its judges would silently change what those numbers mean.
_VERDICT_CONTRACT_SCORED = (
    "Reply in exactly this format, with NO preamble, NO chain-of-thought, "
    "NO commentary outside these three lines:\n"
    "VERDICT: PASS or FAIL\n"
    "SCORE: an integer from 0 to 100\n"
    "REASON: one short sentence citing the specific evidence.\n"
)

_SCORE_LINE = re.compile(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


# Reasoning-tuned models wrap output in these; strip before parsing so a
# thinking trace can neither hide the verdict nor leak into the reason.
_THINK_BLOCK = re.compile(
    r"<(thought|thinking|reasoning)[\s>].*?</\1>", re.IGNORECASE | re.DOTALL
)

# The SAME wrappers, but never closed — what you get when the reasoning trace
# runs into max_tokens and the reply is cut mid-thought. The configured judge
# (models/gemma-4-26b-a4b-it) is thinking-capable and emits <thought>, so this
# is reachable in production, and it defeated the closed-tag pattern above:
# the raw trace survived to the parser, which found a stray "FAIL" inside the
# reasoning and reported it as a confident verdict. A truncated reply is a
# BROKEN GRADER, not a failed answer — strip the unterminated block so nothing
# is left to parse and the caller records "not graded".
_UNCLOSED_THINK_BLOCK = re.compile(
    r"<(thought|thinking|reasoning)[\s>].*\Z", re.IGNORECASE | re.DOTALL
)


def _judge(prompt: str) -> str:
    """Ask the judge model for a verdict. Raises JudgeError if it can't answer.

    ``max_tokens`` is deliberately generous (was 96). On thinking-capable models
    the reasoning tokens are billed against this budget, so a tight cap made the
    model spend the whole allowance thinking and return EMPTY content — which
    the old parser then scored as FAIL.
    """
    client = openai_client()
    model = judge_model()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
    except Exception as exc:  # 404 on an unserved model, quota, network...
        raise JudgeError(f"judge model {model!r} failed: {exc}") from exc
    choice = resp.choices[0] if resp.choices else None
    out = ((choice.message.content if choice else None) or "").strip()
    if not out:
        finish = getattr(choice, "finish_reason", None) if choice else None
        raise JudgeError(
            f"judge model {model!r} returned empty content (finish_reason={finish})"
        )
    return out


def _parse_verdict(out: str) -> tuple[bool, str]:
    """Parse a strict 'VERDICT: PASS/FAIL\\nREASON: ...' response.

    Robust to minor formatting drift: finds the PASS/FAIL token anywhere, and
    takes the reason from the REASON: line (or the text after the verdict),
    truncated to the first sentence.

    Raises JudgeError when no verdict token is present. It previously returned
    ``(False, "")`` in that case — failing CLOSED — so any unparseable reply was
    indistinguishable from a genuine hallucination finding, and the UI showed a
    confident FAIL with no reason. "Unknown" and "failed" are different states
    and must not be conflated.
    """
    text = _THINK_BLOCK.sub("", out or "")
    # Order matters: closed blocks first, so a well-formed trace followed by a
    # real verdict keeps its verdict. Only a block left open to end-of-string
    # (i.e. a truncated reply) is removed here.
    text = _UNCLOSED_THINK_BLOCK.sub("", text).strip()
    if not text:
        raise JudgeError("judge returned only a reasoning block, no verdict")

    # Look for the verdict on its labelled line first; only then anywhere.
    verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.IGNORECASE)
    if verdict_match is None:
        verdict_match = re.search(r"\b(PASS|FAIL)\b", text, re.IGNORECASE)
    if verdict_match is None:
        raise JudgeError(f"no PASS/FAIL verdict in judge reply: {text[:120]!r}")
    verdict = verdict_match.group(1).upper() == "PASS"

    # Prefer an explicit REASON: line; otherwise the text after the verdict line.
    reason_match = re.search(r"REASON:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        # Strip the verdict token and any leading label, take what remains.
        reason = re.sub(
            r"VERDICT:\s*(PASS|FAIL)", "", text, flags=re.IGNORECASE
        ).strip()
    # First sentence only — drop any trailing ramble.
    reason = re.split(r"(?<=[.!?])\s", reason)[0].strip()
    return verdict, reason


def _parse_verdict_scored(out: str) -> tuple[bool, float, str]:
    """Parse VERDICT/SCORE/REASON. Returns (verdict, score 0.0-1.0, reason).

    Raises JudgeError when there is no verdict — the same fail-open rule as
    _parse_verdict. A missing SCORE line degrades to the verdict alone: PASS
    scores 1.0, FAIL scores 0.0, so a judge that ignores the new line still
    produces a usable reading instead of poisoning the retry budget.
    """
    verdict, reason = _parse_verdict(out)
    m = _SCORE_LINE.search(out or "")
    if m is None:
        return verdict, (1.0 if verdict else 0.0), reason
    score = max(0.0, min(1.0, float(m.group(1)) / 100.0))
    return verdict, score, reason


def faithfulness(question: str, context: str, answer: str) -> tuple[bool, str]:
    """Is every claim in the answer supported by the retrieved context?

    This is the missing anti-hallucination check. The judge must decide ONLY
    from the provided context — not from prior knowledge.
    """
    if not answer.strip():
        return False, "empty answer"
    prompt = (
        "You are grading a RAG system for HALLUCINATION. Decide whether EVERY "
        "factual claim in the answer is supported by the context. If the answer "
        "states something not present in or contradicted by the context, it is a "
        "FAIL (hallucination). Minor phrasing is fine. Do NOT use outside "
        f"knowledge.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer: {answer}\n\n"
        + _VERDICT_CONTRACT
    )
    return _parse_verdict(_judge(prompt))


def answer_relevancy(question: str, answer: str) -> tuple[bool, str]:
    """Does the answer address the question (regardless of source grounding)?"""
    if not answer.strip():
        return False, "empty answer"
    prompt = (
        "You are grading a RAG system for ANSWER RELEVANCY. Decide whether the "
        "answer actually addresses what was asked. Off-topic, evasive, or "
        "'I don't know' answers when an answer exists are FAIL. A correct-in-"
        "spirit answer (including an appropriate 'not found' when warranted) is "
        "PASS.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        + _VERDICT_CONTRACT
    )
    return _parse_verdict(_judge(prompt))


def context_relevance(question: str, context: str) -> tuple[bool, str]:
    """Is the retrieved context materially relevant to the question?

    This is a reference-free live proxy for retrieval precision. It judges the
    passages against the question itself, not against a hidden expected answer,
    so it must never be reported as canonical Context Precision or Recall.
    """
    if not context.strip():
        return False, "empty context"
    prompt = (
        "You are checking the CONTEXT RELEVANCE of passages retrieved for a RAG "
        "question, without a reference answer. PASS when the context contains "
        "useful, directly relevant evidence for the question and has no "
        "substantial irrelevant filler. FAIL when the passages are mostly "
        "unrelated, too weak, or do not bear on what was asked. Judge only the "
        "question and supplied context; do not use outside knowledge and do not "
        "judge the generated answer. This is a reference-free proxy, not a gold "
        "Context Precision or Context Recall score.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        + _VERDICT_CONTRACT
    )
    return _parse_verdict(_judge(prompt))


def context_sufficiency(question: str, context: str) -> tuple[bool, str]:
    """Does the context contain enough information to answer the question?

    This is a reference-free live proxy for answer-passage coverage. It asks
    whether the supplied passages are sufficient, not whether every gold passage
    was retrieved, because arbitrary chat questions have no gold passage set.
    """
    if not context.strip():
        return False, "empty context"
    prompt = (
        "You are checking CONTEXT SUFFICIENCY for a RAG question, without a "
        "reference answer. PASS when the supplied passages contain enough "
        "specific information to answer the question completely from the "
        "passages alone. FAIL when key information is absent, the context is "
        "too incomplete, or it cannot support an answer. Judge only the "
        "question and supplied context; do not fill gaps with outside knowledge. "
        "This is a reference-free coverage proxy, not canonical Context Recall.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        + _VERDICT_CONTRACT
    )
    return _parse_verdict(_judge(prompt))


# --- scored variants for the live scorecard --------------------------------
#
# Same questions, same criteria, one extra 0-100 reading. Only the live chat
# calls these; the benchmark keeps the boolean functions above so its published
# pass-rate numbers keep meaning exactly what they meant when published.


def faithfulness_scored(question: str, context: str, answer: str) -> tuple[bool, float, str]:
    """Faithfulness as a ratio of supported claims (RAGAS-style), not a verdict.

    The pass/fail twin is the anti-hallucination gate; this one measures HOW
    MUCH of the answer stands on the context — 3 of 4 claims supported is 75,
    which is a reading a single verdict destroys.
    """
    if not answer.strip():
        return False, 0.0, "empty answer"
    prompt = (
        "You are grading a RAG system for FAITHFULNESS. List the distinct "
        "factual claims in the answer, then decide for each whether the context "
        "supports it. SCORE is the percentage of claims the context supports "
        "(an answer with no claims scores 100 only if it is a correct refusal). "
        "VERDICT is PASS when every claim is supported. Judge from the context "
        "only, never outside knowledge. A hallucinated detail drags the score "
        "down even if the rest is grounded.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer: {answer}\n\n"
        + _VERDICT_CONTRACT_SCORED
    )
    return _parse_verdict_scored(_judge(prompt))


def answer_relevancy_scored(question: str, answer: str) -> tuple[bool, float, str]:
    """How completely the answer addresses what was asked, 0-100."""
    if not answer.strip():
        return False, 0.0, "empty answer"
    prompt = (
        "You are grading a RAG system for ANSWER RELEVANCY. SCORE is how well "
        "the answer addresses what was asked: 100 fully and directly answers "
        "the question; partial or incomplete answers score in between; "
        "off-topic or evasive answers score near 0. An appropriate 'not found' "
        "when the sources lack the answer still scores 80+. VERDICT is PASS "
        "unless the answer misses the question.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        + _VERDICT_CONTRACT_SCORED
    )
    return _parse_verdict_scored(_judge(prompt))


def context_relevance_scored(question: str, context: str) -> tuple[bool, float, str]:
    """How much of the retrieved context bears on the question, 0-100."""
    if not context.strip():
        return False, 0.0, "empty context"
    prompt = (
        "You are checking the CONTEXT RELEVANCE of passages retrieved for a "
        "RAG question, without a reference answer. SCORE is the percentage of "
        "the context that is useful, directly relevant evidence for the "
        "question; mostly unrelated passages score near 0. Judge only the "
        "question and supplied context; do not use outside knowledge and do "
        "not judge the generated answer. This is a reference-free proxy, not a "
        "gold Context Precision score.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        + _VERDICT_CONTRACT_SCORED
    )
    return _parse_verdict_scored(_judge(prompt))


def context_sufficiency_scored(question: str, context: str) -> tuple[bool, float, str]:
    """Whether the passages suffice to answer completely, as a 0-100 coverage."""
    if not context.strip():
        return False, 0.0, "empty context"
    prompt = (
        "You are checking CONTEXT SUFFICIENCY for a RAG question, without a "
        "reference answer. SCORE is how completely the passages cover what is "
        "needed to answer the question: 100 when they suffice entirely, lower "
        "when key information is missing. Judge only the question and supplied "
        "context; do not fill gaps with outside knowledge. This is a "
        "reference-free coverage proxy, not canonical Context Recall.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        + _VERDICT_CONTRACT_SCORED
    )
    return _parse_verdict_scored(_judge(prompt))


def answer_correctness_scored(
    question: str, expected: str, answer: str
) -> tuple[bool, float, str]:
    """How much of the expected answer's key facts the answer conveys, 0-100."""
    if not answer.strip():
        return False, 0.0, "empty answer"
    prompt = (
        "You are grading a RAG system for ANSWER CORRECTNESS against an "
        "expected answer. SCORE is the percentage of the expected answer's key "
        "facts the system's answer conveys correctly; wrong facts score lower "
        "than missing ones. Minor phrasing differences are fine. VERDICT is "
        "PASS when the key facts are conveyed.\n\n"
        f"Question: {question}\n\n"
        f"Expected answer: {expected}\n\n"
        f"System answer: {answer}\n\n"
        + _VERDICT_CONTRACT_SCORED
    )
    return _parse_verdict_scored(_judge(prompt))


def synthesize_expected(question: str, context: str) -> tuple[str, str]:
    """Draft the expected answer a gold author would have written, from context.

    Live grading of arbitrary questions has no human reference. The honest
    substitute is to DERIVE one from the retrieved passages and label the result
    estimated everywhere it is shown. Two deliberate properties:

    * It reads only the question and the passages. It never sees the system's
      answer, so scoring against it is not the model agreeing with itself.
    * If the passages genuinely cannot answer the question, that is returned as
      a verdict ("no answer derivable"), NOT invented text — grading a real
      answer against a hallucinated reference would manufacture failures out
      of thin air.

    Raises JudgeError when the model cannot produce usable text.
    """
    if not context.strip():
        raise JudgeError("empty context")
    prompt = (
        "You are writing the reference answer for a RAG evaluation set.\n"
        "Answer the question using ONLY the passages below, in 1-3 sentences.\n"
        "If and only if the passages do not contain enough information to "
        "answer, reply with exactly:\n"
        f"{NO_ANSWER_DERIVABLE_SENTINEL.upper()}\n"
        "Do not use outside knowledge. Do not mention the passages or this task.\n\n"
        f"Question: {question}\n\n"
        f"Passages:\n{context}\n\n"
        "Reference answer:"
    )
    # Reuses the same client/model as every other judge; raw text here rather
    # than a verdict, so _judge + strip directly.
    out = _judge(prompt)
    text = _THINK_BLOCK.sub("", out or "")
    text = _UNCLOSED_THINK_BLOCK.sub("", text).strip()
    if not text:
        raise JudgeError("reference synthesis returned empty content")
    # The refusal token must be the WHOLE reply. A trace that merely quotes it
    # while reasoning toward a real answer is not a refusal; matching anywhere
    # turned a correctly drafted reference into a false "no answer" (found on
    # the preview deployment, 2026-08-27).
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1 and lines[0].upper() == NO_ANSWER_DERIVABLE_SENTINEL.upper():
        return "", NO_ANSWER_DERIVABLE_SENTINEL
    return text, ""


def answer_correctness(
    question: str, expected: str, answer: str, golden_passages: list[str] | None = None
) -> tuple[bool, str]:
    """Factual correctness vs the expected answer (upgraded from the old string match).

    When golden_passages are supplied (the exact source spans that answer the
    question), the judge may also credit an answer that paraphrases those spans
    correctly even if it differs from the expected wording. This reduces
    mis-flags from phrasing differences.
    """
    if not answer.strip():
        return False, "empty answer"
    golden_block = ""
    if golden_passages:
        golden_block = (
            "\n\nSource passages that answer the question (an answer matching "
            "these in meaning is correct):\n"
            + "\n".join(f"- {p}" for p in golden_passages)
        )
    prompt = (
        "You are grading a RAG system for ANSWER CORRECTNESS. Decide whether the "
        "system's answer conveys the key facts of the expected answer. Minor "
        "phrasing differences are fine; wrong or missing key facts are FAIL."
        f"{golden_block}\n\n"
        f"Question: {question}\n\n"
        f"Expected answer: {expected}\n\n"
        f"System answer: {answer}\n\n"
        + _VERDICT_CONTRACT
    )
    return _parse_verdict(_judge(prompt))
