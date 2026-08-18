"""The CI gate must fail on our mistakes and never on someone else's outage.

A gate that goes red when a provider rate-limits gets ignored within a week, and
an ignored gate is worse than none — it is a green check people trust. A gate
that stays green when the baseline no longer describes the pipeline is worse
still: it reports success for a comparison it never made.

These tests pin that split, because it is a judgement call encoded in an
exception filter and nothing else would notice if it drifted.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from eval import baseline as _baseline  # noqa: E402
from eval import gate as _gate  # noqa: E402


# --- what counts as "the measurement could not be taken" -------------------

class RateLimitError(Exception):
    """Same class NAME the provider SDKs use, which is what the filter matches."""


def test_a_provider_error_is_not_a_regression():
    assert _gate._is_provider_error(RateLimitError("429"))


def test_a_provider_error_is_recognised_through_a_wrapper():
    # run_benchmark wraps failures as it unwinds; the cause chain is where the
    # real error ends up, and checking only the outermost type missed it.
    try:
        try:
            raise RateLimitError("429")
        except RateLimitError as inner:
            raise RuntimeError("indexing corpus failed") from inner
    except RuntimeError as outer:
        assert _gate._is_provider_error(outer)


def test_our_own_bug_is_not_excused_as_a_provider_error():
    assert not _gate._is_provider_error(TypeError("'<' not supported"))
    assert not _gate._is_provider_error(KeyError("similarity"))


# --- what counts as "the baseline no longer describes this run" ------------

def _report(**over):
    metrics = {k: 0.80 for k in _baseline.GATED_METRICS}
    metrics["n_answerable"] = 53
    metrics.update(over.pop("metrics", {}))
    cfg = {"fingerprint": "abc123", "embedding_model": "m", "top_k": 4,
           "candidate_k": 20}
    cfg.update(over.pop("config", {}))
    return {"mode": over.pop("mode", "retrieval-pre-rerank"),
            "config": cfg, "metrics": metrics}


def _base(**over):
    b = {"mode": "retrieval-pre-rerank", "fingerprint": "abc123",
         "embedding_model": "m", "top_k": 4, "candidate_k": 20,
         "n_questions": 53, "tolerance": _baseline.TOLERANCE,
         "metrics": {k: 0.80 for k in _baseline.GATED_METRICS}}
    b.update(over)
    return b


def test_an_aligned_run_reports_no_drift():
    assert _gate._describe_drift(_report(), _base()) == []


@pytest.mark.parametrize("field, report_over", [
    ("fingerprint", {"config": {"fingerprint": "deadbe"}}),
    ("embedding_model", {"config": {"embedding_model": "other"}}),
    ("top_k", {"config": {"top_k": 6}}),
    ("mode", {"mode": "full"}),
    ("n_questions", {"metrics": {"n_answerable": 61}}),
])
def test_each_thing_that_moves_the_numbers_is_caught(field, report_over):
    drift = _gate._describe_drift(_report(**report_over), _base())
    assert any(field in line for line in drift), (
        f"{field} changed and the gate would have compared anyway"
    )


# --- the exit codes --------------------------------------------------------

def _run_gate(monkeypatch, *, report=None, raises=None, base=None):
    monkeypatch.setattr(_baseline, "load", lambda: base if base is not None else _base())

    def fake_run_benchmark(**_kw):
        if raises is not None:
            raise raises
        return report

    import eval.run_eval as _re
    monkeypatch.setattr(_re, "run_benchmark", fake_run_benchmark)
    return _gate.main()


def test_no_baseline_yet_passes_rather_than_blocking_the_first_push(monkeypatch):
    assert _run_gate(monkeypatch, report=_report(), base=None) == 0


def test_provider_outage_passes(monkeypatch):
    assert _run_gate(monkeypatch, raises=RateLimitError("429")) == 0


def test_a_crash_in_our_code_fails(monkeypatch):
    assert _run_gate(monkeypatch, raises=TypeError("boom")) == 1


def test_a_pipeline_change_without_a_new_baseline_fails(monkeypatch):
    report = _report(config={"fingerprint": "changed"})
    assert _run_gate(monkeypatch, report=report) == 1


def test_scores_holding_passes(monkeypatch):
    assert _run_gate(monkeypatch, report=_report()) == 0


def test_a_regression_beyond_tolerance_fails(monkeypatch):
    dropped = {_baseline.GATED_METRICS[0]: 0.80 - _baseline.TOLERANCE - 0.01}
    assert _run_gate(monkeypatch, report=_report(metrics=dropped)) == 1


def test_a_drop_inside_tolerance_passes(monkeypatch):
    jitter = {_baseline.GATED_METRICS[0]: 0.80 - _baseline.TOLERANCE + 0.01}
    assert _run_gate(monkeypatch, report=_report(metrics=jitter)) == 0


# --- the baseline file the gate and the scorecard share --------------------

def test_baseline_records_every_metric_not_only_the_gated_ones():
    # The scorecard draws its marker from this file too. It shows the cosine
    # (RAGAS-comparable) numbers, so a baseline carrying only exact_* would
    # leave every visible bar measured against the old unreachable aspiration.
    b = _baseline.load()
    assert b, "eval/baseline.json missing"
    for k in _baseline.GATED_METRICS:
        assert k in b["metrics"]
    assert "context_recall" in b["metrics"], (
        "regenerate with python -m eval.baseline — the scorecard needs the "
        "cosine metrics recorded too"
    )


def test_baseline_records_the_mode_it_was_measured_in():
    # Without this, a full-mode scorecard would be compared against a
    # pre-rerank baseline: two different retrievals, one number.
    b = _baseline.load()
    assert b and b.get("mode") == "retrieval-pre-rerank"


def test_baseline_carries_no_run_shape_counts_as_scores():
    b = _baseline.load()
    assert not [k for k in b["metrics"] if k.startswith("n_")]
