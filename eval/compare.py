"""Compare two eval runs side by side (PRD §7.4).

Usage:
    .venv/bin/python -m eval.compare eval/runs/A eval/runs/B
    .venv/bin/python -m eval.compare --last-two
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
METRICS = ["recall_at_k", "mrr", "answer_correctness", "not_found_rate_unanswerables"]


def load_report(run_path: str | Path) -> dict:
    p = Path(run_path) / "report.json"
    if not p.exists():
        raise SystemExit(f"No report.json in {run_path}")
    return json.loads(p.read_text())


def print_comparison(a: dict, b: dict, name_a: str, name_b: str) -> None:
    print(f"{'':32} {name_a:>18} {name_b:>18}")
    print("-" * 70)
    for key in METRICS:
        va = a["metrics"].get(key)
        vb = b["metrics"].get(key)
        va_s = f"{va:.4f}" if isinstance(va, (int, float)) else "—"
        vb_s = f"{vb:.4f}" if isinstance(vb, (int, float)) else "—"
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = vb - va
            delta = f"  ({d:+.4f})"
        print(f"{key:32} {va_s:>18} {vb_s:>18}{delta}")
    print()
    print("Configs:")
    print(f"  {name_a}: {json.dumps(a['config'])}")
    print(f"  {name_b}: {json.dumps(b['config'])}")


def main() -> None:
    if "--last-two" in sys.argv:
        runs = sorted((EVAL_DIR / "runs").iterdir()) if (EVAL_DIR / "runs").exists() else []
        if len(runs) < 2:
            raise SystemExit("Fewer than two runs exist")
        print_comparison(
            load_report(runs[-2]), load_report(runs[-1]), runs[-2].name, runs[-1].name
        )
    elif len(sys.argv) == 3:
        print_comparison(
            load_report(sys.argv[1]),
            load_report(sys.argv[2]),
            Path(sys.argv[1]).name,
            Path(sys.argv[2]).name,
        )
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
