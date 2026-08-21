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


def test_the_threshold_in_use_is_the_one_that_was_measured():
    """This used to assert a fixed band, 0.40-0.50, which was right while one
    model was hardcoded and became wrong the moment a second existed: pplx's
    calibrated value is 0.27 and is not a mistake. What must hold is that the
    number in use came from a measurement, not from taste."""
    import json

    from ragchat.config import load_config

    from eval.metrics import match_threshold

    model = load_config().embedding_model
    store = json.loads((ROOT / "eval" / "thresholds.json").read_text(encoding="utf-8"))
    assert match_threshold(model) == store[model]["threshold"]


def test_the_threshold_separates_the_two_populations():
    """A threshold outside the gap between true and false medians is not a
    dividing line, it is a floor or a ceiling — it would call everything a
    match or nothing one."""
    import json

    store = json.loads((ROOT / "eval" / "thresholds.json").read_text(encoding="utf-8"))
    for model, e in store.items():
        assert e["median_false"] < e["threshold"] < e["median_true"], (
            f"{model}: threshold {e['threshold']} sits outside the gap between "
            f"false {e['median_false']} and true {e['median_true']}"
        )


def test_the_derivation_reproduces_the_hand_calibrated_value():
    """The method is only trustworthy if it independently lands where a human
    landed. qwen was tuned by hand to 0.45 in August; eval/calibrate.py derives
    0.48 from labelled pairs without being told."""
    import json

    store = json.loads((ROOT / "eval" / "thresholds.json").read_text(encoding="utf-8"))
    qwen = store.get("qwen/qwen3-embedding-8b")
    assert qwen, "the validation case is gone — recalibrate qwen and keep it"
    assert abs(qwen["threshold"] - 0.45) <= 0.05, (
        f"derived {qwen['threshold']} against a hand-calibrated 0.45; if the "
        f"method has drifted that far, do not trust it for other models either"
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


# ==========================================================================
# The threshold belongs to the EMBEDDING MODEL, not to the harness.
#
# Different models put their similarities on different scales. On the same
# known-true pairs — a golden passage against the chunk that literally contains
# it — qwen's median is 0.492 and pplx's is 0.300. Against one hardcoded 0.45,
# qwen cleared 8 of 12 and pplx cleared 1, which reads as "pplx cannot
# retrieve" and actually means the ruler was built for another model.
#
# That mistake has now been made twice here: once when 0.6 scored four in five
# true containments as misses, and again when comparing these two models. These
# tests are what stops it being made a third time.
# ==========================================================================

def test_an_uncalibrated_model_falls_back_rather_than_guessing():
    from eval.metrics import FALLBACK_MATCH_THRESHOLD, match_threshold

    assert match_threshold("some/model-nobody-measured") == FALLBACK_MATCH_THRESHOLD


def test_the_fallback_is_the_historically_calibrated_value():
    """Anything already measured must keep behaving as it did."""
    from eval.metrics import FALLBACK_MATCH_THRESHOLD

    assert FALLBACK_MATCH_THRESHOLD == 0.45


def test_the_configured_model_has_a_calibrated_threshold():
    """Shipping an embedder with no calibration means the cosine metrics in the
    Evaluation pane are measured with someone else's ruler."""
    import json

    from ragchat.config import load_config

    store = json.loads((ROOT / "eval" / "thresholds.json").read_text(encoding="utf-8"))
    model = load_config().embedding_model
    assert model in store, (
        f"{model} has no calibrated threshold — run "
        f"`python -m eval.calibrate --model {model} --write`"
    )
    assert 0.0 < store[model]["threshold"] < 1.0


def test_each_calibration_records_what_it_cost():
    """A threshold without its error rates is a number someone liked. The whole
    argument for 0.45 was the shape of the curve around it."""
    import json

    store = json.loads((ROOT / "eval" / "thresholds.json").read_text(encoding="utf-8"))
    assert store, "eval/thresholds.json is empty"
    for model, entry in store.items():
        for field in ("threshold", "false_miss_rate", "false_positive_rate",
                      "youden_j", "n_true", "n_false"):
            assert field in entry, f"{model} calibration is missing {field}"
        assert entry["n_true"] > 20, f"{model} was calibrated on too few true pairs"


def test_the_metrics_honour_a_threshold_they_are_handed():
    """If they read the module global instead, an --embedding-model override is
    scored with the configured model's ruler and the comparison is a fiction."""
    from eval.metrics import context_recall

    # Two identical vectors: cosine 1.0. Two orthogonal ones: cosine 0.0.
    same = [1.0, 0.0]
    other = [0.0, 1.0]
    assert context_recall([same], [same], 0.9) == 1.0
    assert context_recall([other], [same], 0.9) == 0.0
    # A threshold of 0.0 must count even the orthogonal pair, proving the
    # argument reaches the comparison rather than being ignored.
    assert context_recall([other], [same], 0.0) == 1.0
