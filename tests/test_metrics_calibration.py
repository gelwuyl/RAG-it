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
    b = {"metrics": {k: 0.80 for k in _baseline.GATED_METRICS}}
    assert _baseline.compare({k: 0.80 for k in _baseline.GATED_METRICS}, b)["ok"]
    # Inside tolerance: noise, not a regression.
    assert _baseline.compare({k: 0.77 for k in _baseline.GATED_METRICS}, b)["ok"]
    # Beyond it: a real drop.
    bad = _baseline.compare({**{k: 0.80 for k in _baseline.GATED_METRICS},
                             "context_recall": 0.60}, b)
    assert not bad["ok"]
    assert any(r["metric"] == "context_recall" and r["status"] == "FAIL"
               for r in bad["rows"])


def test_a_missing_metric_is_skipped_not_silently_passed():
    b = {"metrics": {k: 0.80 for k in _baseline.GATED_METRICS}}
    res = _baseline.compare({"context_recall": 0.80}, b)
    statuses = {r["metric"]: r["status"] for r in res["rows"]}
    assert statuses["context_recall"] == "ok"
    assert statuses["mrr"] == "skipped", "an unchecked metric must not report as a pass"
