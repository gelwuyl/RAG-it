"""Live calibration gate for the tightened judges (2026-08-29 discrimination pass).

Runs the tightened judge functions against CONSTRUCTED cases with REAL model
calls — the same proxy judge model the app configures, resolved exactly the
way ``judges._judge`` resolves it (via ``load_config()``), never a settings
boot default. Every case asserts DIRECTION, never an exact number:

* Faithfulness: an answer built only from its passages reads high; an answer
  mixing grounded claims with fabricated ones reads sub-100; a fully
  fabricated answer reads near 0.
* Answer relevancy: a direct answer reads high; a true answer to a DIFFERENT
  aspect of the question reads sub-100; an evasive non-answer reads near 0.
* Context precision: an all-relevant pool reads high; the relevant passage
  buried under filler reads low (the judge must mark the filler
  IRRELEVANT — the AP math converts the marks, so a low AP proves it).

Then it re-grades a handful of REAL golden-set questions — the recorded
answers of the published 2026-08-21 run, graded against their golden
passages through the tightened boolean judges — as evidence that real
material still scores high and the tightening is not just failing
everything. This costs no generation spend: the answers already exist in
eval/published_run.json.

Usage:
    .venv python -m eval.judge_calibration            # writes eval/calibration_output.txt
    .venv python -m eval.judge_calibration --out path # elsewhere

Requires GEMINI_API_KEY in the environment. Exits non-zero if any direction
assertion fails or a judge call is ungraded (JudgeError) — a calibration run
that cannot grade is a broken grader, not a pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT))

from eval import judges  # noqa: E402
from eval.judges import JudgeError  # noqa: E402

DEFAULT_OUT = EVAL_DIR / "calibration_output.txt"


class _Reporter:
    """Collects everything the run saw — raw judge replies first, assertions
    second — and writes it to the evidence file the orchestrator reads."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures: list[str] = []
        self.n_asserts = 0

    def section(self, title: str) -> None:
        self.lines += ["", "=" * 72, title, "=" * 72]

    def raw(self, reply: str) -> None:
        self.lines.append("--- raw judge reply ---")
        self.lines.append(reply.strip())
        self.lines.append("-----------------------")

    def assert_direction(self, claim: str, ok: bool, detail: str) -> None:
        self.n_asserts += 1
        status = "PASS" if ok else "FAIL"
        self.lines.append(f"  [{status}] {claim}  ({detail})")
        if not ok:
            self.failures.append(f"{claim}  ({detail})")

    def ungraded(self, case: str, exc: Exception) -> None:
        self.failures.append(f"{case}: judge call ungraded — {exc}")
        self.lines.append(f"  [FAIL] {case}: JudgeError — {exc}")

    def write(self, path: Path, model: str) -> None:
        verdict = (
            f"RESULT: ALL {self.n_asserts} DIRECTION ASSERTIONS PASSED"
            if not self.failures
            else f"RESULT: {len(self.failures)} FAILURE(S) of {self.n_asserts}"
        )
        lines = self.lines + ["", "=" * 72, verdict]
        if self.failures:
            lines += [f"  - {f}" for f in self.failures]
        lines += [
            "Judge model: " + model,
            "Provenance: readings from these tightened judges are NOT directly",
            "comparable to the published 2026-08-21 run, which predates them.",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n{verdict}")
        print(f"Evidence written to {path}")


# Tee _judge so every reply the judges parse lands in the evidence file —
# and pace the calls: the free-tier judge endpoint allows ~15 requests per
# minute per model, and a calibration run spends a few dozen. A 429 gets a
# backoff-and-retry (it is an outage shape, but a recoverable one here).
_real_judge = judges._judge
_reporter: _Reporter

_MIN_INTERVAL = 4.5
_last_call = [0.0]


def _capturing_judge(prompt: str, max_tokens: int = 512) -> str:
    since = time.monotonic() - _last_call[0]
    if since < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - since)
    for attempt in range(4):
        _last_call[0] = time.monotonic()
        try:
            raw = _real_judge(prompt, max_tokens=max_tokens)
            break
        except JudgeError as exc:
            if "429" not in str(exc) or attempt == 3:
                raise
            print("  (judge 429 rate-limited — backing off 20s)")
            time.sleep(20)
    _reporter.lines.append(f"\n[prompt] {prompt[:150].strip()}...")
    _reporter.raw(raw)
    return raw


def _pct(score: float) -> str:
    return f"{round(score * 100)}"


# --- constructed material ---------------------------------------------------
#
# Deliberately synthetic and self-contained: the judge must grade from the
# context alone, so the passages are invented and small, and the fabricated
# claims are provably absent from them.

TF300_CTX = (
    "[1] The TF-300 air fryer has a 4-litre basket and a 1500 W heating "
    "element.\n"
    "[2] The TF-300 ships with a two-year manufacturer warranty covering "
    "heating element defects.\n"
    "[3] Cleaning: the inner basket is dishwasher-safe; wipe the exterior "
    "with a damp cloth."
)

FAITH_CASES = [
    (
        "faithfulness / grounded answer",
        "The TF-300 has a 4-litre basket, a 1500 W heating element, and a "
        "two-year manufacturer warranty.",
        lambda s: s >= 0.8,
        "expect >= 0.8",
        True,
    ),
    (
        "faithfulness / mixed (2 of 4 claims fabricated)",
        "The TF-300 has a 4-litre basket, a 1500 W heating element, a "
        "9-hour cooking timer, and a stainless steel body.",
        lambda s: s < 0.9,
        "expect sub-100",
        False,
    ),
    (
        "faithfulness / fully fabricated answer",
        "The TF-300 has a 9-hour cooking timer and costs 249 euros.",
        lambda s: s <= 0.35,
        "expect near 0",
        False,
    ),
]

WARRANTY_Q = "What is the warranty period of the TF-300 air fryer?"

RELEVANCY_CASES = [
    (
        "relevancy / direct answer",
        "The TF-300 ships with a two-year manufacturer warranty.",
        lambda s: s >= 0.8,
        "expect >= 0.8",
        True,
    ),
    (
        "relevancy / true but different aspect",
        "The TF-300 has a 4-litre basket and a 1500 W heating element.",
        lambda s: s < 0.9,
        "expect sub-100",
        False,
    ),
    (
        "relevancy / evasive non-answer",
        "Air fryers are a popular kind of kitchen appliance.",
        lambda s: s <= 0.4,
        "expect near 0",
        False,
    ),
]


def run_constructed(rep: _Reporter) -> None:
    rep.section("CONSTRUCTED CASES (synthetic passages, real judge calls)")

    for name, answer, direction, note, want_bool_pass in FAITH_CASES:
        rep.lines.append(f"\n{name}\n  answer: {answer}")
        try:
            verdict, score, reason = judges.faithfulness_scored(
                "Tell me about the TF-300.", TF300_CTX, answer
            )
            rep.lines.append(f"  faithfulness_scored: {_pct(score)}  ({reason})")
            rep.assert_direction(f"{name}: scored {note}", direction(score), _pct(score))
            rep.assert_direction(
                f"{name}: scored verdict {'PASS' if want_bool_pass else 'FAIL'}",
                verdict is want_bool_pass,
                f"verdict={verdict}",
            )
            bverdict, breason = judges.faithfulness("Tell me about the TF-300.", TF300_CTX, answer)
            rep.lines.append(f"  faithfulness (benchmark): {bverdict}  ({breason})")
            rep.assert_direction(
                f"{name}: boolean {'PASS' if want_bool_pass else 'FAIL'}",
                bverdict is want_bool_pass,
                f"verdict={bverdict}",
            )
        except JudgeError as exc:
            rep.ungraded(name, exc)

    direct_reading: list[float] = []

    for name, answer, direction, note, want_bool_pass in RELEVANCY_CASES:
        rep.lines.append(f"\n{name}\n  answer: {answer}")
        try:
            verdict, score, reason = judges.answer_relevancy_scored(WARRANTY_Q, answer)
            rep.lines.append(f"  answer_relevancy_scored: {_pct(score)}  ({reason})")
            rep.assert_direction(f"{name}: scored {note}", direction(score), _pct(score))
            if "direct" in name:
                direct_reading.append(score)
                rep.assert_direction(
                    f"{name}: scored verdict PASS", verdict is True, f"verdict={verdict}"
                )
            if "different aspect" in name and direct_reading:
                rep.assert_direction(
                    f"{name}: below the direct answer's reading",
                    score < direct_reading[0],
                    f"{_pct(score)} < {_pct(direct_reading[0])}",
                )
            bverdict, breason = judges.answer_relevancy(WARRANTY_Q, answer)
            rep.lines.append(f"  answer_relevancy (benchmark): {bverdict}  ({breason})")
            if want_bool_pass or "evasive" in name:
                rep.assert_direction(
                    f"{name}: boolean {'PASS' if want_bool_pass else 'FAIL'}",
                    bverdict is want_bool_pass,
                    f"verdict={bverdict}",
                )
        except JudgeError as exc:
            rep.ungraded(name, exc)

    rep.lines.append("\ncontext precision / relevant passage ranked first")
    try:
        verdict, ap, reason = judges.context_precision_scored(
            WARRANTY_Q,
            [
                "The TF-300 ships with a two-year manufacturer warranty "
                "covering heating element defects.",
                "The TF-300's warranty is administered by the retailer it was "
                "purchased from; claims within the warranty window go through "
                "the retailer.",
            ],
        )
        rep.lines.append(f"  context_precision_scored: AP {_pct(ap)}  ({reason})")
        # With the needed passage top-ranked, AP is 1.0 even if the judge
        # marks the second passage either way — the reading this case pins is
        # that a top-ranked relevant passage yields a HIGH score, and the
        # raw reply shows whether the tightened bar treated the
        # administration detail as RELEVANT or IRRELEVANT.
        rep.assert_direction("precision / top-ranked: expect >= 0.8", ap >= 0.8, _pct(ap))
        rep.assert_direction("precision / top-ranked: verdict PASS", verdict is True, f"verdict={verdict}")
    except JudgeError as exc:
        rep.ungraded("precision / top-ranked", exc)

    rep.lines.append("\ncontext precision / relevant passage buried under filler")
    try:
        verdict, ap, reason = judges.context_precision_scored(
            WARRANTY_Q,
            [
                TF300_CTX.split("\n")[2].replace("[3] ", ""),
                "Air fryers circulate hot air around food to cook it quickly.",
                TF300_CTX.split("\n")[0].replace("[1] ", ""),
                TF300_CTX.split("\n")[1].replace("[2] ", ""),
            ],
        )
        rep.lines.append(f"  context_precision_scored: AP {_pct(ap)}  ({reason})")
        # The warranty passage ranks 4th of 4 with exactly one relevant: AP
        # = (1/4)/1 = 0.25 IF the judge marks all three fillers IRRELEVANT.
        # A low AP is therefore direct proof that the tightened bar marked
        # same-topic filler as IRRELEVANT.
        rep.assert_direction("precision / buried: expect <= 0.6", ap <= 0.6, _pct(ap))
        rep.assert_direction("precision / buried: verdict FAIL", verdict is False, f"verdict={verdict}")
    except JudgeError as exc:
        rep.ungraded("precision / buried", exc)


def _golden_context(item: dict) -> str:
    """The material the answer's claims live in: the FULL golden source
    document(s), not the golden_passage excerpts.

    The published answers were generated from retrieved chunks of the whole
    corpus, and a question's 2-3 golden passage excerpts routinely omit a
    fact the answer legitimately cites — grading against the excerpts would
    manufacture UNSUPPORTED verdicts for well-grounded answers (seen live in
    the first calibration pass: 'Havenmark Method in Chinese' is stated
    verbatim in havenmark_method.txt but not in its golden passages).
    Corpus files are small (1.4-6 KB), so the whole document is affordable.
    """
    docs = [
        d.strip()
        for d in (item.get("golden_doc") or "").replace(";", ",").split(",")
        if d.strip()
    ]
    parts = []
    for name in docs:
        path = EVAL_DIR / "corpus" / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts) if parts else "\n\n".join(item.get("golden_passages", []))


def run_real_questions(rep: _Reporter) -> None:
    rep.section(
        "REAL GOLDEN-SET RE-GRADES (published 2026-08-21 answers, tightened judges)"
    )
    published = json.loads((EVAL_DIR / "published_run.json").read_text(encoding="utf-8"))
    golden = {
        json.loads(l)["question"]: json.loads(l)
        for l in (EVAL_DIR / "golden_set.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    answerable = [r for r in published["results"] if not r["unanswerable"]][:3]
    unanswerable = [r for r in published["results"] if r["unanswerable"]][:1]
    rows = answerable + unanswerable

    faith_pass = 0
    relevant_pass = 0
    for r in rows:
        q = r["question"]
        rep.lines.append(f"\nQ: {q}")
        rep.lines.append(f"A: {(r['answer'] or '')[:300]}")
        item = golden.get(q, {})
        ctx = _golden_context(item)
        rep.lines.append(
            f"  graded against: {item.get('golden_doc') or '(no golden doc — refusal case)'}"
        )
        try:
            fverdict, freason = judges.faithfulness(q, ctx, r["answer"])
            rep.lines.append(f"  faithfulness (benchmark): {fverdict}  ({freason})")
            fverdict_s, fscore, freason_s = judges.faithfulness_scored(q, ctx, r["answer"])
            rep.lines.append(
                f"  faithfulness_scored: {_pct(fscore)}  ({freason_s})"
            )
            if not r["unanswerable"]:
                faith_pass += bool(fverdict)
            rverdict, rreason = judges.answer_relevancy(q, r["answer"])
            rep.lines.append(f"  answer_relevancy (benchmark): {rverdict}  ({rreason})")
            rverdict_s, rscore, rreason_s = judges.answer_relevancy_scored(q, r["answer"])
            rep.lines.append(
                f"  answer_relevancy_scored: {_pct(rscore)}  ({rreason_s})"
            )
            if not r["unanswerable"]:
                relevant_pass += bool(rverdict)
            if r["unanswerable"]:
                # A published refusal makes no claims and must stay faithful;
                # the tightened relevancy prompt still credits a proper
                # 'not found'.
                rep.assert_direction(
                    "real refusal: faithfulness PASS", fverdict is True, f"{_pct(fscore)}"
                )
                rep.assert_direction(
                    "real refusal: relevancy PASS", rverdict is True, f"{_pct(rscore)}"
                )
        except JudgeError as exc:
            rep.ungraded(f"real question: {q[:60]}", exc)

    rep.assert_direction(
        "real material still scores high on relevancy (>= 2 of 3 PASS)",
        relevant_pass >= 2,
        f"{relevant_pass}/3",
    )
    rep.assert_direction(
        "real material still scores high on faithfulness (>= 2 of 3 PASS)",
        faith_pass >= 2,
        f"{faith_pass}/3",
    )


def main() -> None:
    global _reporter
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    # Patched here, not at import time: this module is a script, but if
    # anything ever imported it, an import-time patch would silently capture
    # and rate-limit every judge call in that process.
    judges._judge = _capturing_judge
    _reporter = _Reporter()
    _reporter.lines.append("LIVE JUDGE CALIBRATION — tightened judges (2026-08-29)")
    _reporter.lines.append("Direction assertions only; every raw reply below is verbatim.")
    print(f"Judge model: {judges.judge_model()}")

    run_constructed(_reporter)
    run_real_questions(_reporter)
    _reporter.write(Path(args.out), judges.judge_model())
    if _reporter.failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
