"""CI entry point: score retrieval and fail the build if it got worse.

    python -m eval.gate

Runs the retrieval-only benchmark (no generation, no judges, no rerank) and
compares the deterministic `exact_*` metrics against eval/baseline.json.

Three outcomes, and the distinction between them is the whole design:

  exit 0  the numbers held, or the gate could not be run at all
  exit 1  retrieval regressed beyond the tolerance
  exit 1  the baseline no longer describes this pipeline

The middle case is the point. The last one deserves an explanation: if the
chunking, the embedding model or the golden set changed, the committed baseline
was measured on a different pipeline and comparing to it is not a measurement.
That is the author's doing and should stop the build until they regenerate it —
otherwise the gate quietly compares nothing and reports success forever.

The first case is the same "fail open, never closed" rule the judges follow
(CLAUDE.md). A provider that answers 403, 429 or 5xx is a broken grader, not a
broken retriever, and failing a merge because someone else's rate limit was hit
teaches everyone to ignore the gate. Those are reported loudly and pass. A
TypeError is NOT caught: that is our own code and the build should break.
"""
from __future__ import annotations

import sys
import traceback

from eval import baseline as bl

# Errors that mean "the measurement could not be taken", not "the code is worse".
# Matched by class name so this module does not import the provider SDKs, which
# would make an unrelated dependency change able to break the gate itself.
_PROVIDER_ERRORS = {
    "APIStatusError", "APIConnectionError", "APITimeoutError", "APIError",
    "RateLimitError", "InternalServerError", "PermissionDeniedError",
    "AuthenticationError", "ConnectionError", "Timeout", "ReadTimeout",
    "HTTPError", "SSLError",
}


def _is_provider_error(exc: BaseException) -> bool:
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        for klass in type(cur).__mro__:
            if klass.__name__ in _PROVIDER_ERRORS:
                return True
        cur = cur.__cause__ or cur.__context__
    return False


def _describe_drift(report: dict, base: dict) -> list[str]:
    """Reasons the committed baseline does not describe this run."""
    cfg = report.get("config", {})
    checks = [
        ("mode", report.get("mode"), base.get("mode")),
        ("fingerprint", cfg.get("fingerprint"), base.get("fingerprint")),
        ("embedding_model", cfg.get("embedding_model"), base.get("embedding_model")),
        ("top_k", cfg.get("top_k"), base.get("top_k")),
        ("candidate_k", cfg.get("candidate_k"), base.get("candidate_k")),
        ("n_questions", report.get("metrics", {}).get("n_answerable"),
         base.get("n_questions")),
    ]
    return [
        f"  {name}: baseline {was!r}, this run {now!r}"
        for name, now, was in checks
        if was is not None and now is not None and now != was
    ]


def main() -> int:
    base = bl.load()
    if not base:
        print("no eval/baseline.json committed — nothing to compare against.")
        print("run: python -m eval.baseline")
        return 0

    try:
        from eval.run_eval import run_benchmark

        report = run_benchmark(retrieval_only=True)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, then narrowed
        if _is_provider_error(exc):
            print()
            print("=" * 68)
            print("GATE SKIPPED — the embedding provider did not answer.")
            print(f"  {type(exc).__name__}: {exc}")
            print()
            print("This is not a retrieval regression and does not fail the build.")
            print("=" * 68)
            return 0
        traceback.print_exc()
        print()
        print("GATE FAILED — the benchmark itself raised. That is our code.")
        return 1

    drift = _describe_drift(report, base)
    if drift:
        print()
        print("=" * 68)
        print("GATE FAILED — the baseline was measured on a different pipeline,")
        print("so comparing against it would not be a measurement:")
        print("\n".join(drift))
        print()
        print("If the change was deliberate, regenerate and read the diff:")
        print("  python -m eval.baseline --from-run latest")
        print("=" * 68)
        return 1

    result = bl.compare(report["metrics"], base)
    print()
    print(bl.render(result))
    print()
    if result["ok"]:
        print(f"GATE PASSED — no gated metric fell more than {base.get('tolerance')}.")
        return 0
    print("GATE FAILED — retrieval regressed. Either fix it, or, if the change")
    print("was deliberate, regenerate the baseline and say why in the commit.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
