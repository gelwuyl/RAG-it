"""LLM-as-judge generation metrics (faithfulness, relevancy, correctness).

All three reuse the same proxy LLM (qwen3.8-max) per user decision 2026-08-15.
The judge returns PASS/FAIL (or a 0-1 score) plus one line of reasoning.
The eval harness calls these only in the full (non --retrieval-only) run.
"""
from __future__ import annotations

import re

from ragchat.embeddings import openai_client

JUDGE_MODEL = "qwen3.8-max"


def _judge(prompt: str) -> str:
    client = openai_client()
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_verdict(out: str) -> tuple[bool, str]:
    m = re.search(r"(PASS|FAIL)", out, re.IGNORECASE)
    verdict = bool(m) and m.group(1).upper() == "PASS"
    reason = out.split("\n", 1)[1].strip() if "\n" in out else out
    return verdict, reason


def faithfulness(question: str, context: str, answer: str) -> tuple[bool, str]:
    """Is every claim in the answer supported by the retrieved context?
    This is the missing anti-hallucination check."""
    if not answer.strip():
        return False, "empty answer"
    prompt = (
        "You are grading a RAG system for HALLUCINATION. Given the question, the "
        "retrieved source context, and the system's answer, decide whether EVERY "
        "factual claim in the answer is supported by the context. If the answer "
        "states something not present in or contradicted by the context, it is a "
        "FAIL (hallucination). Minor phrasing is fine.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer: {answer}\n\n"
        "Reply exactly: PASS or FAIL, then a newline, then one sentence of reasoning."
    )
    return _parse_verdict(_judge(prompt))


def answer_relevancy(question: str, answer: str) -> tuple[bool, str]:
    """Does the answer address the question (regardless of source grounding)?"""
    if not answer.strip():
        return False, "empty answer"
    prompt = (
        "You are grading a RAG system for ANSWER RELEVANCY. Given the question and "
        "the system's answer, decide whether the answer actually addresses what "
        "was asked (it may be correct-in-spirit even if it says 'not found' when "
        "appropriate). Off-topic or evasive answers are FAIL.\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Reply exactly: PASS or FAIL, then a newline, then one sentence of reasoning."
    )
    return _parse_verdict(_judge(prompt))


def answer_correctness(question: str, expected: str, answer: str) -> tuple[bool, str]:
    """Factual correctness vs the expected answer (upgraded from the old string match)."""
    if not answer.strip():
        return False, "empty answer"
    prompt = (
        "You are grading a RAG system for ANSWER CORRECTNESS. Given the question, "
        "the expected answer, and the system's answer, decide whether the system's "
        "answer conveys the key facts of the expected answer. Minor phrasing "
        "differences are fine; wrong or missing key facts are FAIL.\n\n"
        f"Question: {question}\n\n"
        f"Expected answer: {expected}\n\n"
        f"System answer: {answer}\n\n"
        "Reply exactly: PASS or FAIL, then a newline, then one sentence of reasoning."
    )
    return _parse_verdict(_judge(prompt))
