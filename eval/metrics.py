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
# passage. Tunable; raised to 0.6 because 0.45 let a single shared keyword
# (e.g. "system") cross the bar against a multi-token golden passage under a
# coarse embedder. A real embedding model separates meaning better, but 0.6
# is a safer default. CALIBRATE on the real KFD run: if Context Recall comes
# out implausibly high, lower this; if golden passages that ARE retrieved
# score as missed, raise it. Measurement-first (user decision #3).
MATCH_THRESHOLD = 0.6


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
