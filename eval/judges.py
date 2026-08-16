"""LLM-as-judge generation metrics (faithfulness, relevancy, correctness).

All three reuse the same proxy LLM (qwen3.8-max) per user decision 2026-08-15.
The judge returns PASS/FAIL (or a 0-1 score) plus one line of reasoning.
The eval harness calls these only in the full (non --retrieval-only) run.

Tightened 2026-08-16: the judge prompts now forbid chain-of-thought and force a
single-line verdict + one-sentence reason. Previously the model could emit a
thinking trace before the verdict, which leaked into the parsed "reason" and
occasionally flipped the verdict parse. We also cap max_tokens so the model
cannot ramble, and the parser only reads the verdict line + first sentence.
"""
from __future__ import annotations

import re

from ragchat.embeddings import openai_client
from ragchat.config import settings

# Judge model follows the deployment default (now gemini-2.5-flash on Google's
# endpoint). Was previously hard-coded to qwen3.8-max which would 404 on the
# Gemini endpoint — derive it from settings so it tracks config.yaml.
JUDGE_MODEL = settings.default_llm_model

# Strict output contract the judge must follow. We repeat it in every prompt so
# the parser can rely on "first line = verdict, second line = one sentence".
_VERDICT_CONTRACT = (
    "Reply in exactly this format, with NO preamble, NO chain-of-thought, "
    "NO commentary outside these two lines:\n"
    "VERDICT: PASS or FAIL\n"
    "REASON: one short sentence citing the specific evidence.\n"
)


def _judge(prompt: str) -> str:
    client = openai_client()
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=96,  # tight: one verdict line + one reason line, nothing more
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_verdict(out: str) -> tuple[bool, str]:
    """Parse a strict 'VERDICT: PASS/FAIL\\nREASON: ...' response.

    Robust to minor formatting drift: finds the PASS/FAIL token anywhere, and
    takes the reason from the REASON: line (or the text after the verdict),
    truncated to the first sentence. Never lets a thinking trace flip the call.
    """
    if not out:
        return False, ""
    verdict_match = re.search(r"\b(PASS|FAIL)\b", out, re.IGNORECASE)
    verdict = bool(verdict_match) and verdict_match.group(1).upper() == "PASS"

    # Prefer an explicit REASON: line; otherwise the text after the verdict line.
    reason = ""
    reason_match = re.search(r"REASON:\s*(.+)", out, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()
    else:
        # Strip the verdict token and any leading label, take what remains.
        tail = re.sub(r"VERDICT:\s*(PASS|FAIL)", "", out, flags=re.IGNORECASE).strip()
        reason = tail
    # First sentence only — drop any trailing ramble.
    reason = re.split(r"(?<=[.!?])\s", reason)[0].strip()
    return verdict, reason


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
