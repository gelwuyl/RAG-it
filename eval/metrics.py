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
