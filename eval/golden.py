"""The runtime question bank: known truth for questions the app is asked live.

The benchmark scores retrieval against ``golden_set.jsonl`` — each question
ships with the passages that answer it, so MRR/NDCG/hit rate have something
measured to compare against. A chat answer has no such labels, which is why the
scorecard's ranking rows read "needs a known answer": nobody can judge whether
the RIGHT passage came back without knowing which passage that is.

This module closes that gap for the questions where truth IS known. It loads
the golden set plus ``demo_golden.jsonl`` (pairs over the two demo documents,
so a first-time visitor's questions can light the rows up too) and matches the
asked question against it. A match hands ``pipeline.ask`` the ground truth for
that question and the ranking rows carry a MEASURED reading instead of a note.

Matching is deliberately strict — an exact match after normalization, or a
>=0.90 difflib ratio against the best candidate. A near-miss that matches the
WRONG question would present fiction as ground truth, which is the one failure
this feature must not have, so anything ambiguous matches nothing and the rows
stay honestly grey.
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent

BANK_FILES = (
    (EVAL_DIR / "golden_set.jsonl", "golden"),
    (EVAL_DIR / "demo_golden.jsonl", "demo"),
)

# Anything below this is a different question wearing a similar coat.
MATCH_RATIO = 0.90

_bank: list[dict] | None = None


def load_bank() -> list[dict]:
    """Parse both bank files once per process and cache them.

    Reading repo files at runtime is fine on Vercel (only WRITES are impossible)
    and the cache only has to live as long as the instance does. Any read or
    parse problem yields an empty bank: an ungreying feature that fails must
    fail back to the grey rows, never break the answer.
    """
    global _bank
    if _bank is not None:
        return _bank
    bank: list[dict] = []
    idx = 0
    for path, src in BANK_FILES:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            item["_src"] = src
            item["_idx"] = idx
            idx += 1
            bank.append(item)
    _bank = bank
    return bank


def _norm(q: str) -> str:
    """Case- and punctuation-insensitive question identity."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (q or "").lower())).strip()


def match_question(question: str) -> dict | None:
    """The bank entry for this question, or None when nothing matches.

    Exact normalized equality wins outright. Otherwise the best difflib ratio
    is taken only at >=0.90 — and callers must treat None as "no opinion",
    exactly like router.choose_tool's: the caller cannot tell "no match" from
    "the bank could not be read", and must not care.
    """
    try:
        bank = load_bank()
        if not bank or not (question or "").strip():
            return None
        q = _norm(question)
        if not q:
            return None
        for item in bank:
            if _norm(item.get("question", "")) == q:
                return item
        best, best_ratio = None, 0.0
        for item in bank:
            ratio = difflib.SequenceMatcher(None, q, _norm(item.get("question", ""))).ratio()
            if ratio > best_ratio:
                best, best_ratio = item, ratio
        return best if best is not None and best_ratio >= MATCH_RATIO else None
    except Exception:
        return None
