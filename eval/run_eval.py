"""Eval harness (PRD §7.2): index eval/corpus under the current config, run
the golden set, and write a comparable report to eval/runs/<timestamp>/.

Usage:
    .venv/bin/python -m eval.run_eval            # full run (retrieval + answers + judges)
    .venv/bin/python -m eval.run_eval --retrieval-only   # skip generation/judge
    .venv/bin/python -m eval.run_eval --limit 5          # smoke run

The report includes the full config snapshot (F19) and now BOTH a retrieval
block (Context Recall / Precision@k / MRR / NDCG / HitRate, computed by
embedding-cosine match against golden passages) and a generation block
(Faithfulness / Answer Relevancy / Answer Correctness via LLM-as-judge).
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
from ragchat.embeddings import ProxyEmbeddings, openai_client  # noqa: E402
from ragchat.pipeline import NOT_FOUND_ANSWER, ask, ingest_document_text, retrieve  # noqa: E402
from ragchat.vectordb import collection_name  # noqa: E402
from ragchat.store import get_client  # noqa: E402
from eval.metrics import (  # noqa: E402
    MATCH_THRESHOLD,
    context_recall,
    precision_at_k,
    mrr_at_k,
    ndcg_at_k,
    hit_rate_at_k,
)
from eval import judges  # noqa: E402

EVAL_USER = "__eval__"
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN = EVAL_DIR / "golden.jsonl"
JUDGE_MODEL = judges.JUDGE_MODEL


def reset_eval_collection(cfg) -> None:
    client = get_client()
    name = collection_name(EVAL_USER, cfg.embedding_model)
    try:
        client.delete_collection(name)
    except Exception:
        pass


def load_corpus(cfg) -> None:
    for path in sorted(CORPUS_DIR.glob("*")):
        if path.suffix.lower() in (".md", ".txt"):
            text = path.read_text(encoding="utf-8", errors="replace")
            n = ingest_document_text(EVAL_USER, path.name, path.name, text, cfg)
            print(f"  indexed {path.name} ({n} chunks)")


def embed_passages(passages: list[str], model: str) -> list[list[float]]:
    if not passages:
        return []
    emb = ProxyEmbeddings(model)
    return emb.embed_documents(passages)


def judge_answer(question: str, expected: str, answer: str) -> tuple[bool, str]:
    # Kept for backwards compatibility; correctness now routed through judges.py
    return judges.answer_correctness(question, expected, answer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    print(f"Config fingerprint: {cfg.fingerprint()}")
    print(f"Judge model: {JUDGE_MODEL} | MATCH_THRESHOLD (cosine): {MATCH_THRESHOLD}")
    print("Indexing corpus...")
    reset_eval_collection(cfg)
    load_corpus(cfg)

    items = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]

    # Pre-embed golden passages once per question.
    for it in items:
        it["_golden_embs"] = embed_passages(it.get("golden_passages", []), cfg.embedding_model)

    results = []
    for i, item in enumerate(items, start=1):
        question = item["question"]
        chunks = retrieve(EVAL_USER, question, cfg)
        # Embed the retrieved chunk texts with the same model used for the
        # golden passages so we can cosine-match (user decision #3: embedding-cosine).
        chunk_texts = [c["text"] for c in chunks]
        _emb = ProxyEmbeddings(cfg.embedding_model)
        chunk_embs = _emb.embed_documents(chunk_texts) if chunk_texts else []

        k = cfg.top_k
        cr = context_recall(chunk_embs, item["_golden_embs"])
        prec = precision_at_k(chunk_embs, item["_golden_embs"], k)
        mrr = mrr_at_k(chunk_embs, item["_golden_embs"], k)
        ndcg = ndcg_at_k(chunk_embs, item["_golden_embs"], k)
        hr = hit_rate_at_k(chunk_embs, item["_golden_embs"], k)

        entry = {
            "question": question,
            "unanswerable": item["unanswerable"],
            "needs": item.get("needs", ["single_passage"]),
            "type": item.get("type", ""),
            "expected_source": item.get("golden_doc", ""),
            "context_recall": round(cr, 4),
            "precision_at_k": round(prec, 4),
            "mrr": round(mrr, 4),
            "ndcg_at_k": round(ndcg, 4),
            "hit_rate_at_k": hr,
            "retrieved": [
                {"title": c["title"], "similarity": round(c["similarity"], 4)}
                for c in chunks[:k]
            ],
        }

        if not args.retrieval_only:
            res = ask(EVAL_USER, question, [], cfg)
            entry["answer"] = res["answer"]
            entry["not_found"] = res["not_found"]
            context_text = "\n\n".join(
                f"[{j+1}] {c['title']}\n{c['text']}" for j, c in enumerate(chunks[:k])
            )
            if item["unanswerable"]:
                # Correct iff the system refused (did not fabricate).
                entry["correct"] = bool(res["not_found"])
                entry["faithful"] = None
                entry["relevant"] = None
            else:
                fh, fh_r = judges.faithfulness(question, context_text, res["answer"])
                rv, rv_r = judges.answer_relevancy(question, res["answer"])
                cr_ok, cr_r = judges.answer_correctness(
                    question, item["expected"], res["answer"]
                )
                entry["faithful"] = fh
                entry["faithful_reason"] = fh_r
                entry["relevant"] = rv
                entry["relevant_reason"] = rv_r
                entry["correct"] = cr_ok
                entry["correct_reason"] = cr_r

        results.append(entry)
        status = "✓" if entry.get("correct", cr > 0) else "✗"
        print(f"  [{i}/{len(items)}] {status} {question[:60]}")

    # Aggregate
    answerable = [r for r in results if not r["unanswerable"]]
    unanswerable = [r for r in results if r["unanswerable"]]
    multi_doc = [r for r in answerable if "multi_doc" in r["needs"]]

    def _mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    metrics = {
        "context_recall": _mean([r["context_recall"] for r in answerable]),
        "precision_at_k": _mean([r["precision_at_k"] for r in answerable]),
        "mrr": _mean([r["mrr"] for r in answerable]),
        "ndcg_at_k": _mean([r["ndcg_at_k"] for r in answerable]),
        "hit_rate_at_k": _mean([r["hit_rate_at_k"] for r in answerable]),
    }
    if not args.retrieval_only:
        metrics["faithfulness"] = _mean([1 if r["faithful"] else 0 for r in answerable if r["faithful"] is not None])
        metrics["answer_relevancy"] = _mean([1 if r["relevant"] else 0 for r in answerable if r["relevant"] is not None])
        metrics["answer_correctness"] = _mean([1 if r["correct"] else 0 for r in answerable if r.get("correct") is not None])
        metrics["not_found_rate_unanswerables"] = _mean([1 if r["correct"] else 0 for r in unanswerable]) if unanswerable else None
        if multi_doc:
            metrics["context_recall_multi_doc"] = _mean([r["context_recall"] for r in multi_doc])
    metrics["n_answerable"] = len(answerable)
    metrics["n_unanswerable"] = len(unanswerable)
    metrics["n_multi_doc"] = len(multi_doc)

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
        f"| Context Recall | {metrics['context_recall']} | ≥ 0.80 |",
        f"| Precision@k | {metrics['precision_at_k']} | ≥ 0.70 |",
        f"| MRR | {metrics['mrr']} | ≥ 0.65 |",
        f"| NDCG@k | {metrics['ndcg_at_k']} | ≥ 0.70 |",
        f"| Hit Rate@k | {metrics['hit_rate_at_k']} | ≥ 0.80 |",
    ]
    if not args.retrieval_only:
        md += [
            f"| Faithfulness | {metrics['faithfulness']} | ≥ 0.90 |",
            f"| Answer Relevancy | {metrics['answer_relevancy']} | ≥ 0.85 |",
            f"| Answer Correctness | {metrics['answer_correctness']} | ≥ 0.80 |",
            f"| Not-found rate (unanswerables) | {metrics['not_found_rate_unanswerables']} | ≥ 0.90 |",
        ]
    md += [
        "",
        "| Question | Type | CtxRecall | Faithful | Correct |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        fh = "—" if r.get("faithful") is None else ("✓" if r["faithful"] else "✗")
        cd = "—" if r.get("correct") is None else ("✓" if r["correct"] else "✗")
        md.append(
            f"| {r['question'][:55]} | {','.join(r.get('needs', []))} | "
            f"{r['context_recall']:.2f} | {fh} | {cd} |"
        )
    (run_dir / "report.md").write_text("\n".join(md))
    print(f"\nMetrics: {json.dumps(metrics, indent=2)}")
    print(f"Report written to {run_dir}")


if __name__ == "__main__":
    main()
