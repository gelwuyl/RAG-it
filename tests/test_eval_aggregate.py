"""Aggregate-level regression guards for benchmark grading availability."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_eval import aggregate


def _answerable(**overrides):
    row = {
        "unanswerable": False,
        "context_recall": 1.0,
        "precision_at_k": 1.0,
        "mrr": 1.0,
        "ndcg_at_k": 1.0,
        "hit_rate_at_k": 1.0,
        "faithful": True,
        "relevant": True,
        "correct": True,
    }
    row.update(overrides)
    return row


def test_correctness_only_outage_counts_as_ungraded():
    """A broken correctness judge cannot disappear from benchmark provenance."""
    metrics = aggregate([_answerable(correct=None)])

    assert metrics["answer_correctness"] is None
    assert metrics["n_ungraded"] == 1
