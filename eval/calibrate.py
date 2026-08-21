"""Derive MATCH_THRESHOLD for an embedding model, from labelled data.

The cosine retrieval metrics decide "does this retrieved chunk contain the
golden passage?" by comparing two embeddings against a threshold. That
threshold is a property of the EMBEDDING MODEL, not of the harness: different
models put their similarities on different scales, so a value tuned for one is
meaningless against another.

Measured on this corpus, for pairs that are known-true because the golden
passage is a verbatim substring of the chunk:

    qwen/qwen3-embedding-8b        median 0.492
    perplexity/pplx-embed-v1-0.6b  median 0.300

At a single hardcoded 0.45, qwen cleared 8 of 12 known-true pairs and pplx
cleared 1. Read naively that says pplx cannot retrieve; it actually says the
ruler was built for a different model. That mistake has been made twice in this
repo now — first when 0.6 scored four in five true containments as misses, and
again when comparing these two models — so the threshold is derived rather than
chosen.

WHAT THE LABELS ARE

`eval/build_golden_set.py` guarantees every golden passage is a verbatim
substring of exactly one corpus document and absent from every distractor. That
gives free, exact labels with no judgement involved:

    TRUE  — passage vs a chunk that literally contains it
    FALSE — passage vs a chunk that does not

WHAT IS OPTIMISED

Youden's J (sensitivity + specificity - 1), which weights a false miss and a
false positive equally. That matters here: the two failures are not symmetrical
in their consequences but they are symmetrical in their capacity to mislead — a
low threshold inflates recall by counting unrelated chunks, a high one deflates
it by rejecting real ones, and this repo has been burned by both directions.

USAGE

    .venv/Scripts/python -m eval.calibrate
    .venv/Scripts/python -m eval.calibrate --model perplexity/pplx-embed-v1-0.6b
    .venv/Scripts/python -m eval.calibrate --model <m> --write

`--write` updates eval/thresholds.json, which metrics.py reads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ragchat.config import load_config  # noqa: E402
from ragchat.embeddings import ProxyEmbeddings  # noqa: E402
from ragchat.pipeline import plan_chunks  # noqa: E402

from eval.metrics import cosine  # noqa: E402

CORPUS = ROOT / "eval" / "corpus"
GOLDEN = ROOT / "eval" / "golden_set.jsonl"
THRESHOLDS = ROOT / "eval" / "thresholds.json"

# Enough negatives to characterise the false-positive side without embedding the
# whole corpus against every passage. Drawn from documents OTHER than the one
# holding the passage, which is where a confusable corpus does its work: these
# are the same firm in the same words with different facts.
NEGATIVES_PER_PASSAGE = 6


def _labelled_pairs(cfg) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(true_pairs, false_pairs) of (passage, chunk_text)."""
    rows = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    chunks_by_doc: dict[str, list[str]] = {}
    for path in sorted(CORPUS.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks_by_doc[path.name] = [c.text for c in plan_chunks(text, path.name, cfg)]

    true_pairs: list[tuple[str, str]] = []
    false_pairs: list[tuple[str, str]] = []
    for row in rows:
        for passage in row.get("golden_passages") or []:
            holder = None
            for doc, chunks in chunks_by_doc.items():
                for chunk in chunks:
                    if passage in chunk:
                        true_pairs.append((passage, chunk))
                        holder = doc
            if holder is None:
                # The generator forbids this; if it happens the golden set and
                # the chunker disagree and every cosine number is meaningless.
                print(f"  ! no chunk contains: {passage[:60]!r}")
                continue
            others = [
                chunk
                for doc, chunks in chunks_by_doc.items()
                if doc != holder
                for chunk in chunks
            ]
            for chunk in others[:NEGATIVES_PER_PASSAGE]:
                if passage not in chunk:
                    false_pairs.append((passage, chunk))
    return true_pairs, false_pairs


def _scores(pairs: list[tuple[str, str]], emb: ProxyEmbeddings) -> list[float]:
    cache: dict[str, list[float]] = {}

    def vec(text: str) -> list[float]:
        if text not in cache:
            cache[text] = emb.embed_query(text)
        return cache[text]

    return [cosine(vec(a), vec(b)) for a, b in pairs]


def calibrate(model: str, cfg) -> dict:
    emb = ProxyEmbeddings(model)
    true_pairs, false_pairs = _labelled_pairs(cfg)
    print(f"  {len(true_pairs)} true pairs, {len(false_pairs)} false pairs")
    pos = _scores(true_pairs, emb)
    neg = _scores(false_pairs, emb)

    best = None
    for i in range(1, 100):
        t = i / 100
        sensitivity = sum(1 for s in pos if s >= t) / len(pos)
        specificity = sum(1 for s in neg if s < t) / len(neg)
        j = sensitivity + specificity - 1
        if best is None or j > best["youden_j"]:
            best = {
                "threshold": round(t, 2),
                "youden_j": round(j, 4),
                "false_miss_rate": round(1 - sensitivity, 4),
                "false_positive_rate": round(1 - specificity, 4),
            }
    best["n_true"] = len(pos)
    best["n_false"] = len(neg)
    best["median_true"] = round(sorted(pos)[len(pos) // 2], 4)
    best["median_false"] = round(sorted(neg)[len(neg) // 2], 4)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="defaults to the configured embedding model")
    ap.add_argument("--write", action="store_true", help="save into eval/thresholds.json")
    args = ap.parse_args()

    cfg = load_config()
    model = args.model or cfg.embedding_model
    print(f"Calibrating MATCH_THRESHOLD for {model}\n")
    result = calibrate(model, cfg)

    print(f"\n  threshold            {result['threshold']}")
    print(f"  Youden's J           {result['youden_j']}")
    print(f"  false misses         {result['false_miss_rate']:.1%}")
    print(f"  false positives      {result['false_positive_rate']:.1%}")
    print(f"  median true / false  {result['median_true']} / {result['median_false']}")

    if args.write:
        store = json.loads(THRESHOLDS.read_text(encoding="utf-8")) if THRESHOLDS.exists() else {}
        store[model] = result
        THRESHOLDS.write_text(json.dumps(store, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\n  written to {THRESHOLDS.relative_to(ROOT)}")
    else:
        print("\n  (not saved — pass --write)")


if __name__ == "__main__":
    main()
