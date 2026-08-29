"""The benchmark result that ships with the app.

The Evaluation pane used to be empty until you ran the benchmark yourself, and
running it means the browser driving ~56 sliced requests through the live
pipeline: several minutes, real model spend, and a visibly busy app while it
happens. Every visitor paid that to see numbers that are the same for all of
them — the corpus is fixed, the questions are fixed, and the pipeline is the
same one for everyone.

So a known-good run is committed here and served immediately. The pane has
content on first paint, the landing page's claims have something behind them,
and "Run benchmark" becomes what it always should have been: a way to
re-measure after changing something, not a toll on the front door.

This is NOT eval/baseline.json. That file is four deterministic retrieval
metrics for the CI gate, deliberately cheap to reproduce on every push. This is
the full picture including generation and the judges — expensive, occasional,
and for reading rather than gating.

Regenerate after a change that should move the published numbers:

    python -m eval.run_eval            # the expensive part
    python -m eval.published --from-run latest

Read the diff before committing it. Publishing a run without looking at it is
how a regression becomes the number on the landing page.

PROVENANCE CAVEAT: the committed run was generated 2026-08-21, BEFORE the
judge-prompt tightening of 2026-08-29 (see eval/judges.py and
eval/judge_calibration.py). Its faithfulness and answer_relevancy pass rates
were read by the earlier, more lenient judges and are NOT directly comparable
to readings taken after that date — the tightened judges discriminate, so
sub-100% values appear where the old judges flatly passed everything.
Re-publishing requires a full benchmark re-run, which is a separate
user-approved model spend.
"""
from __future__ import annotations

import json
from pathlib import Path

PUBLISHED_FILE = Path(__file__).resolve().parent / "published_run.json"

# Per-question rows kept in the shipped file. The pane lists them, but all 56
# with their generated answers is a payload every visitor downloads to look at
# the first screenful. Enough to show the shape of the run, including the
# unanswerable ones, which are the interesting rows.
MAX_RESULTS = 56


def load() -> dict | None:
    if not PUBLISHED_FILE.exists():
        return None
    try:
        return json.loads(PUBLISHED_FILE.read_text(encoding="utf-8"))
    except Exception:
        # A malformed file must not take the Evaluation pane down with it; the
        # caller renders "no run yet", which is what it did before this existed.
        return None


def build(from_run: str | Path = "latest") -> Path:
    """Write published_run.json from a completed run directory."""
    from eval.baseline import _resolve_run

    run_dir = _resolve_run(from_run)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})

    blob = {
        "published": True,
        "generated": report.get("timestamp"),
        "run": run_dir.name,
        # Which pipeline these numbers describe. The scorecard refuses to
        # compare against a baseline measured in a different mode, and this is
        # what it checks (ragchat/app.py:_run_payload, frontend scoreReference).
        "mode": report.get("mode"),
        "config": report.get("config", {}),
        "metrics": metrics,
        "results": report.get("results", [])[:MAX_RESULTS],
        "n_corpus_files": len(
            [p for p in (Path(__file__).resolve().parent / "corpus").iterdir()
             if p.is_file()]
        ),
    }
    PUBLISHED_FILE.write_text(json.dumps(blob, indent=1) + "\n", encoding="utf-8")
    size_kb = PUBLISHED_FILE.stat().st_size / 1024
    print(f"wrote {PUBLISHED_FILE} ({size_kb:.0f} KB)")
    print(f"  mode      {blob['mode']}")
    print(f"  questions {len(blob['results'])}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:34s} {v:.4f}")
    return PUBLISHED_FILE


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-run", default="latest", metavar="DIR|latest")
    build(ap.parse_args().from_run)
