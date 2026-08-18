"""Retrieval + generation metric functions for the eval harness.

Retrieval metrics are deterministic and do NOT need an LLM: they compare the
retrieved chunk set against the question's golden answer-passages using
embedding-cosine similarity (same model as the pipeline, per user decision
2026-08-15). A golden passage is "recalled" if any retrieved chunk exceeds
MATCH_THRESHOLD cosine similarity.

Generation metrics (faithfulness / relevancy / correctness) live in judges.py
because they require an LLM-as-judge call.
"""
from __future__ import annotations

import math
from typing import Iterable

# Cosine threshold for counting a retrieved chunk as a match for a golden
# passage.
#
# CALIBRATED 2026-08-18, on the synthetic corpus with qwen3-embedding-8b. The
# method: every golden passage is a verbatim substring of some chunk (the
# generator enforces it), so the cosine between a passage and the chunk that
# literally contains it is a ground-truth "this really is a match". Anything
# scoring below the threshold there is a FALSE MISS — retrieval worked and the
# metric called it a failure. Measured against 98 passages and 21 chunks:
#
#   threshold   false misses      false positives
#   0.60          79.6%              0.7%
#   0.50          31.6%              7.6%
#   0.45          20.4%             17.2%
#   0.40           9.2%             31.6%
#   0.30           0.0%             74.1%
#
# 0.6 was scoring FOUR IN FIVE true containments as misses, which is most of why
# context recall read so low — it was largely a broken measurement, not broken
# retrieval. The cause is length, not language: a one-line passage against a
# ~500-token chunk scores low even when it is verbatim inside it, and Latin
# passages fared worse (83.1%) than CJK (66.7%).
#
# No threshold is good. 0.45 is the least-bad point on the curve and is chosen
# deliberately over exact substring matching, which would be exact and free but
# would stop being the RAGAS-style embedding-cosine metric this harness reports
# (user decision, 2026-08-18).
#
# Live with the consequence: recall carries roughly a fifth error in BOTH
# directions, so small movements between runs are noise, not signal.
MATCH_THRESHOLD = 0.45


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _chunk_matches_passage(chunk_emb: list[float], passage_emb: list[float]) -> bool:
    return cosine(chunk_emb, passage_emb) >= MATCH_THRESHOLD


def best_sim_for_passage(chunk_embs: Iterable[list[float]], passage_emb: list[float]) -> float:
    """Highest cosine between a golden passage and any retrieved chunk."""
    return max((cosine(ce, passage_emb) for ce in chunk_embs), default=0.0)


def context_recall(retrieved_embs: list[list[float]], golden_embs: list[list[float]]) -> float:
    """Fraction of golden passages that have at least one retrieved chunk >= threshold."""
    if not golden_embs:
        return 0.0
    hits = 0
    for ge in golden_embs:
        if any(_chunk_matches_passage(ce, ge) for ce in retrieved_embs):
            hits += 1
    return hits / len(golden_embs)


def precision_at_k(
    retrieved_embs: list[list[float]], golden_embs: list[list[float]], k: int
) -> float:
    """Fraction of the top-k retrieved chunks that match ANY golden passage."""
    top = retrieved_embs[:k]
    if not top:
        return 0.0
    rel = 0
    for ce in top:
        if any(_chunk_matches_passage(ce, ge) for ge in golden_embs):
            rel += 1
    return rel / len(top)


def hit_rate_at_k(
    retrieved_embs: list[list[float]], golden_embs: list[list[float]], k: int
) -> int:
    """1 if any of the top-k chunks matches a golden passage, else 0."""
    return int(any(_chunk_matches_passage(ce, ge)
                   for ce in retrieved_embs[:k] for ge in golden_embs))


def mrr_at_k(
    retrieved_embs: list[list[float]], golden_embs: list[list[float]], k: int
) -> float:
    """Reciprocal rank of the first retrieved chunk that matches a golden passage."""
    for rank, ce in enumerate(retrieved_embs[:k], start=1):
        if any(_chunk_matches_passage(ce, ge) for ge in golden_embs):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_embs: list[list[float]], golden_embs: list[list[float]], k: int
) -> float:
    """Graded relevance: each chunk scored by max cosine to any golden passage,
    then normalized DCG over the top-k."""
    def _rel(ce: list[float]) -> float:
        return max((cosine(ce, ge) for ge in golden_embs), default=0.0)

    dcg = 0.0
    for i, ce in enumerate(retrieved_embs[:k], start=1):
        dcg += _rel(ce) / math.log2(i + 1)
    # Ideal DCG: sort all relevance scores descending.
    all_rel = sorted((_rel(ce) for ce in retrieved_embs), reverse=True)
    idcg = 0.0
    for i, r in enumerate(all_rel[:k], start=1):
        idcg += r / math.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Exact containment metrics — deterministic, free, and what the CI gate uses.
#
# The cosine metrics above answer "is this chunk ABOUT the golden passage".
# These answer "does this chunk CONTAIN it", which for a golden set whose
# passages are guaranteed verbatim substrings (the generator enforces it) is
# decidable exactly, with no threshold and no embedding call.
#
# They exist because the cosine metric has a pathology that makes it unfit to
# fail a build on. Measured on this corpus at MATCH_THRESHOLD 0.45:
#
#   corpus size    false misses    false positives
#   21 chunks         20.4%           17.2%
#   39 chunks         20.4%           19.5%
#
# The false-positive rate RISES with corpus size, because more same-domain
# chunks drift above the threshold. So cosine scores go UP as the corpus grows —
# adding filler improves the number. A gate on that can be gamed by writing more
# documents, and it carries a fifth error in both directions besides.
#
# The cosine metrics remain the reported, RAGAS-comparable ones. These are the
# ones the gate compares, so a red build means something actually changed.
# ---------------------------------------------------------------------------


def _contains_any(text: str, passages: Iterable[str]) -> bool:
    return any(p in text for p in passages)


def exact_context_recall(retrieved_texts: list[str], passages: list[str]) -> float:
    """Fraction of golden passages present verbatim in some retrieved chunk."""
    if not passages:
        return 0.0
    hits = sum(1 for p in passages if any(p in t for t in retrieved_texts))
    return hits / len(passages)


def exact_precision_at_k(retrieved_texts: list[str], passages: list[str], k: int) -> float:
    """Fraction of the top-k chunks that carry at least one golden passage."""
    top = retrieved_texts[:k]
    if not top:
        return 0.0
    return sum(1 for t in top if _contains_any(t, passages)) / len(top)


def exact_hit_rate_at_k(retrieved_texts: list[str], passages: list[str], k: int) -> int:
    """1 if any of the top-k chunks carries a golden passage, else 0."""
    return int(any(_contains_any(t, passages) for t in retrieved_texts[:k]))


def exact_mrr_at_k(retrieved_texts: list[str], passages: list[str], k: int) -> float:
    """Reciprocal rank of the first top-k chunk carrying a golden passage."""
    for rank, t in enumerate(retrieved_texts[:k], start=1):
        if _contains_any(t, passages):
            return 1.0 / rank
    return 0.0
