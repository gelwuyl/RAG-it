"""Eval harness (PRD §7.2): index eval/corpus under the current config, run
the golden set, and write a comparable report to eval/runs/<timestamp>/.

Usage:
    .venv/bin/python -m eval.run_eval            # full run (retrieval + answers)
    .venv/bin/python -m eval.run_eval --retrieval-only   # skip generation/judge
    .venv/bin/python -m eval.run_eval --limit 5          # smoke run

The report includes the full config snapshot (F19), so any two runs under
eval/runs/ can be compared side by side (see eval/compare.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT))

from ragchat.config import load_config  # noqa: E402
from ragchat.embeddings import openai_client  # noqa: E402
from ragchat.pipeline import NOT_FOUND_ANSWER, ask, ingest_document_text, retrieve  # noqa: E402
from ragchat.store import collection_name, get_client  # noqa: E402

EVAL_USER = "__eval__"
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN = EVAL_DIR / "golden.jsonl"
JUDGE_MODEL = "qwen3.8-max"


def reset_eval_collection() -> None:
    client = get_client()
    # Collection names are now namespaced by (user, embedding model) in
    # store.py, so delete the exact collection the eval will write to.
    name = collection_name(EVAL_USER, cfg.embedding_model) if "cfg" in dir() else None
    if name is None:
        # cfg not yet loaded (shouldn't happen) — sweep any __eval__ collections
        for col in client.list_collections():
            if col.name.startswith("user-__eval__"):
                try:
                    client.delete_collection(col.name)
                except Exception:
                    pass
        return
    try:
        client.delete_collection(name)
    except Exception:
        pass


def load_corpus(cfg) -> None:
    for path in sorted(CORPUS_DIR.glob("*")):
        text = path.read_text(encoding="utf-8", errors="replace")
        n = ingest_document_text(EVAL_USER, path.name, path.name, text, cfg)
        print(f"  indexed {path.name} ({n} chunks)")


def judge_answer(question: str, expected: str, answer: str) -> tuple[bool, str]:
    client = openai_client()
    prompt = (
        "You are grading a RAG system. Given a question, an expected answer, "
        "and the system's generated answer, decide whether the generated "
        "answer correctly conveys the key facts of the expected answer. "
        "Minor phrasing differences are fine; wrong or missing key facts are not.\n\n"
        f"Question: {question}\nExpected answer: {expected}\n"
        f"Generated answer: {answer}\n\n"
        "Reply exactly: PASS or FAIL, then a newline, then one sentence of reasoning."
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    out = (resp.choices[0].message.content or "").strip()
    verdict = out.upper().startswith("PASS")
    reason = out.split("\n", 1)[1].strip() if "\n" in out else out
    return verdict, reason


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    print(f"Config fingerprint: {cfg.fingerprint()}")
    print("Indexing corpus...")
    reset_eval_collection()
    load_corpus(cfg)

    items = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]

    results = []
    for i, item in enumerate(items, start=1):
        question = item["question"]
        chunks = retrieve(EVAL_USER, question, cfg)
        hits = [c for c in chunks if c["title"] == item.get("source")]
        recall = 1.0 if hits else 0.0
        rr = 0.0
        for rank, c in enumerate(chunks, start=1):
            if c["title"] == item.get("source"):
                rr = 1.0 / rank
                break

        entry = {
            "question": question,
            "unanswerable": item["unanswerable"],
            "expected_source": item.get("source", ""),
            "recall_at_k": recall,
            "reciprocal_rank": rr,
            "retrieved": [
                {"title": c["title"], "similarity": round(c["similarity"], 4)}
                for c in chunks
            ],
        }

        if not args.retrieval_only:
            res = ask(EVAL_USER, question, [], cfg)
            entry["answer"] = res["answer"]
            entry["not_found"] = res["not_found"]
            if item["unanswerable"]:
                entry["correct"] = bool(res["not_found"])
            else:
                verdict, reason = judge_answer(question, item["expected"], res["answer"])
                entry["correct"] = verdict
                entry["judge_reason"] = reason
        results.append(entry)
        status = "✓" if entry.get("correct", recall > 0) else "✗"
        print(f"  [{i}/{len(items)}] {status} {question[:60]}")

    answerable = [r for r in results if not r["unanswerable"]]
    unanswerable = [r for r in results if r["unanswerable"]]
    metrics = {
        "recall_at_k": round(sum(r["recall_at_k"] for r in answerable) / max(len(answerable), 1), 4),
        "mrr": round(sum(r["reciprocal_rank"] for r in answerable) / max(len(answerable), 1), 4),
    }
    if not args.retrieval_only:
        metrics["answer_correctness"] = round(
            sum(1 for r in answerable if r["correct"]) / max(len(answerable), 1), 4
        )
        metrics["not_found_rate_unanswerables"] = round(
            sum(1 for r in unanswerable if r["correct"]) / max(len(unanswerable), 1), 4
        )
    metrics["n_answerable"] = len(answerable)
    metrics["n_unanswerable"] = len(unanswerable)

    run_dir = EVAL_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "splitter": cfg.splitter,
            "top_k": cfg.top_k,
            "candidate_k": cfg.candidate_k,
            "similarity_threshold": cfg.similarity_threshold,
            "hybrid_search": cfg.hybrid_search,
            "reranker": cfg.reranker,
            "query_rewrite": cfg.query_rewrite,
            "llm_model": cfg.llm_model,
            "temperature": cfg.temperature,
            "embedding_model": cfg.embedding_model,
            "fingerprint": cfg.fingerprint(),
        },
        "metrics": metrics,
        "results": results,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    md = [
        f"# Eval run {run_dir.name}",
        "",
        f"- Config: `{json.dumps(report['config'])}`",
        "",
        "| Metric | Value | Target |",
        "|---|---|---|",
        f"| Recall@k | {metrics['recall_at_k']} | ≥ 0.80 |",
        f"| MRR | {metrics['mrr']} | ≥ 0.65 |",
    ]
    if not args.retrieval_only:
        md += [
            f"| Answer correctness | {metrics['answer_correctness']} | ≥ 0.80 |",
            f"| Not-found rate (unanswerables) | {metrics['not_found_rate_unanswerables']} | ≥ 0.90 |",
        ]
    md += [
        "",
        "| Question | Expected source | Recall | Correct |",
        "|---|---|---|---|",
    ]
    for r in results:
        md.append(
            f"| {r['question'][:60]} | {r['expected_source'] or '—'} | "
            f"{r['recall_at_k']:.0f} | {'—' if args.retrieval_only else ('✓' if r.get('correct') else '✗')} |"
        )
    (run_dir / "report.md").write_text("\n".join(md))
    print(f"\nMetrics: {json.dumps(metrics, indent=2)}")
    print(f"Report written to {run_dir}")


if __name__ == "__main__":
    main()
