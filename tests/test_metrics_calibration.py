"""The match threshold is a measured value, not a taste.

MATCH_THRESHOLD decides whether a retrieved chunk counts as containing a golden
passage, and at 0.6 it was scoring roughly four in five TRUE containments as
misses — which is most of why context recall read so low. These tests pin the
properties that made 0.45 the choice, so a future edit has to argue with data.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import baseline as _baseline  # noqa: E402
from eval.metrics import MATCH_THRESHOLD  # noqa: E402


def test_threshold_sits_where_it_was_calibrated():
    # 0.6 gave 79.6% false misses; 0.30 gave 74.1% false positives. 0.45 is the
    # least-bad point measured on the synthetic corpus with qwen3-embedding-8b.
    assert 0.40 <= MATCH_THRESHOLD <= 0.50, (
        "MATCH_THRESHOLD moved outside the calibrated band — recalibrate with "
        "the method documented in eval/metrics.py before changing it"
    )


def test_baseline_is_committed_and_matches_the_shipped_config():
    from ragchat.config import load_config

    b = _baseline.load()
    assert b is not None, "eval/baseline.json missing — run python -m eval.baseline"
    cfg = load_config()
    assert b["fingerprint"] == cfg.fingerprint(), (
        "baseline was generated under a different chunking/embedding config; "
        "regenerate it or the gate compares two different questions"
    )
    assert b["embedding_model"] == cfg.embedding_model


def test_a_drop_beyond_tolerance_fails_and_a_small_one_does_not():
    # Named from GATED_METRICS rather than hardcoded: which metrics are gated is
    # a decision that has already changed once, and a test that pins the names
    # breaks for the wrong reason when it changes again.
    first = _baseline.GATED_METRICS[0]
    b = {"metrics": {k: 0.80 for k in _baseline.GATED_METRICS}}
    assert _baseline.compare({k: 0.80 for k in _baseline.GATED_METRICS}, b)["ok"]
    # Inside tolerance: noise, not a regression.
    assert _baseline.compare({k: 0.77 for k in _baseline.GATED_METRICS}, b)["ok"]
    # Beyond it: a real drop.
    bad = _baseline.compare({**{k: 0.80 for k in _baseline.GATED_METRICS},
                             first: 0.60}, b)
    assert not bad["ok"]
    assert any(r["metric"] == first and r["status"] == "FAIL" for r in bad["rows"])


def test_a_missing_metric_is_skipped_not_silently_passed():
    first, second = _baseline.GATED_METRICS[0], _baseline.GATED_METRICS[1]
    b = {"metrics": {k: 0.80 for k in _baseline.GATED_METRICS}}
    res = _baseline.compare({first: 0.80}, b)
    statuses = {r["metric"]: r["status"] for r in res["rows"]}
    assert statuses[first] == "ok"
    assert statuses[second] == "skipped", "an unchecked metric must not report as a pass"


# --- exact containment: what the gate actually compares -------------------
#
# The cosine metrics carry ~20% error in both directions AND drift upward as the
# corpus grows (17.2% false positives at 21 chunks, 19.5% at 39). A gate on them
# could be improved by adding filler documents. These are string containment, so
# they move only when retrieval moves.


def test_exact_metrics_have_no_false_positives():
    from eval.metrics import (exact_context_recall, exact_hit_rate_at_k,
                              exact_mrr_at_k, exact_precision_at_k)

    chunks = ["the quick brown fox", "jumps over the lazy dog"]
    # A topically identical passage that is NOT present scores zero, where
    # cosine would have rated it highly similar.
    assert exact_context_recall(chunks, ["a fast auburn fox"]) == 0.0
    assert exact_hit_rate_at_k(chunks, ["a fast auburn fox"], 2) == 0
    assert exact_precision_at_k(chunks, ["a fast auburn fox"], 2) == 0.0
    assert exact_mrr_at_k(chunks, ["a fast auburn fox"], 2) == 0.0


def test_exact_metrics_have_no_false_misses():
    from eval.metrics import exact_context_recall, exact_hit_rate_at_k

    # A short passage inside a long chunk is the case cosine got wrong 80% of
    # the time at threshold 0.6. Containment is exact regardless of length.
    long_chunk = "x " * 400 + "form HE-104 is required" + " y" * 400
    assert exact_context_recall([long_chunk], ["form HE-104 is required"]) == 1.0
    assert exact_hit_rate_at_k([long_chunk], ["form HE-104 is required"], 1) == 1


def test_exact_recall_is_a_fraction_of_passages():
    from eval.metrics import exact_context_recall

    chunks = ["alpha", "beta"]
    assert exact_context_recall(chunks, ["alpha", "beta"]) == 1.0
    assert exact_context_recall(chunks, ["alpha", "gamma"]) == 0.5
    assert exact_context_recall(chunks, []) == 0.0


def test_the_gate_watches_exact_metrics_only():
    # If this ever names a cosine metric again, the gate has become gameable by
    # adding documents. That is the failure this whole calibration exists to
    # prevent, so it is worth a test rather than a comment.
    assert all(m.startswith("exact_") for m in _baseline.GATED_METRICS), (
        f"gate must compare deterministic metrics, got {_baseline.GATED_METRICS}"
    )
