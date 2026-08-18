"""The committed baseline: what this pipeline currently scores.

Targets used to be hardcoded aspirations in the frontend (context recall >= 0.80
and so on). Nothing ever met them, so the scorecard was permanently red and
therefore unreadable — a dashboard that is always failing tells you nothing on
the day it starts failing for a real reason.

A baseline is the opposite: it records what the pipeline ACTUALLY scores on a
known-good run, and red then means "worse than we were", which is a fact you can
act on. The same file drives the CI gate, so the scorecard and the build agree
on what counts as a regression instead of holding two opinions.

Regenerate after any deliberate change to the corpus, the golden set, the
chunking config or the embedding model — all of which legitimately move the
numbers:

    python -m eval.baseline

Then read the diff before committing it. A baseline regenerated without reading
the diff launders a regression into the new normal, which is the one failure
mode this file has.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

BASELINE_FILE = Path(__file__).resolve().parent / "baseline.json"

# Metrics the gate watches. Deliberately the retrieval ones only: they need no
# LLM call, so the gate stays free and runs on every push. Generation and judge
# scores move with model updates outside our control and would fail the build
# for something nobody changed.
GATED_METRICS = (
    "context_recall",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "hit_rate_at_k",
)

# How far a metric may fall before the gate fails.
#
# Not tight, and deliberately so: MATCH_THRESHOLD is 0.45, which carries roughly
# 20% false misses and 17% false positives (see metrics.py), so run-to-run
# movement of a few points is measurement noise rather than a code change. A
# tolerance below that noise floor produces a gate that cries wolf, and a gate
# people learn to ignore is worse than no gate.
TOLERANCE = 0.05


def load() -> dict | None:
    if not BASELINE_FILE.exists():
        return None
    try:
        return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def compare(metrics: dict, baseline: dict | None = None,
            tolerance: float = TOLERANCE) -> dict:
    """Compare a run's metrics against the baseline.

    Returns {"ok": bool, "rows": [...]}. A metric missing from either side is
    reported as "skipped" rather than as a pass: silently passing something we
    did not actually check is how a gate stops meaning anything.
    """
    baseline = baseline if baseline is not None else load()
    if not baseline:
        return {"ok": True, "rows": [], "note": "no baseline committed yet"}

    base_metrics = baseline.get("metrics", {})
    rows = []
    ok = True
    for key in GATED_METRICS:
        now, was = metrics.get(key), base_metrics.get(key)
        if now is None or was is None:
            rows.append({"metric": key, "status": "skipped",
                         "now": now, "baseline": was, "delta": None})
            continue
        delta = round(now - was, 4)
        failed = delta < -tolerance
        ok = ok and not failed
        rows.append({"metric": key, "status": "FAIL" if failed else "ok",
                     "now": round(now, 4), "baseline": round(was, 4),
                     "delta": delta})
    return {"ok": ok, "rows": rows}


def render(result: dict) -> str:
    """Human-readable comparison, for CI logs."""
    if not result.get("rows"):
        return result.get("note", "nothing to compare")
    width = max(len(r["metric"]) for r in result["rows"])
    lines = [f"{'metric'.ljust(width)}  {'baseline':>9} {'now':>9} {'delta':>8}  status"]
    for r in result["rows"]:
        base = "—" if r["baseline"] is None else f"{r['baseline']:.4f}"
        now = "—" if r["now"] is None else f"{r['now']:.4f}"
        delta = "—" if r["delta"] is None else f"{r['delta']:+.4f}"
        lines.append(
            f"{r['metric'].ljust(width)}  {base:>9} {now:>9} {delta:>8}  {r['status']}"
        )
    return "\n".join(lines)


def build(limit: int | None = None) -> Path:
    """Run the free retrieval-only benchmark and write baseline.json."""
    from eval.run_eval import run_benchmark

    report = run_benchmark(limit=limit, retrieval_only=True)
    cfg = report["config"]
    blob = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        # Everything that legitimately moves the numbers. A baseline compared
        # across a different fingerprint or a different golden set is comparing
        # two different questions, so record enough to notice.
        "mode": report.get("mode"),
        "fingerprint": cfg.get("fingerprint"),
        "embedding_model": cfg.get("embedding_model"),
        "top_k": cfg.get("top_k"),
        "candidate_k": cfg.get("candidate_k"),
        "reranker": cfg.get("reranker"),
        "n_questions": report["metrics"].get("n_answerable"),
        "tolerance": TOLERANCE,
        "metrics": {k: report["metrics"].get(k) for k in GATED_METRICS},
    }
    BASELINE_FILE.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {BASELINE_FILE}")
    print(json.dumps(blob, indent=2))
    return BASELINE_FILE


if __name__ == "__main__":
    build()
