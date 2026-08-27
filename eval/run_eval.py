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
from ragchat.pipeline import (  # noqa: E402
    NOT_FOUND_ANSWER,
    _rerank,
    ask,
    ingest_document_text,
    retrieve,
)
from ragchat.vectordb import delete_document_chunks  # noqa: E402
from eval.metrics import (  # noqa: E402
    MATCH_THRESHOLD,
    match_threshold,
    exact_context_recall,
    exact_hit_rate_at_k,
    exact_mrr_at_k,
    exact_precision_at_k,
    context_recall,
    precision_at_k,
    mrr_at_k,
    ndcg_at_k,
    hit_rate_at_k,
)
from eval import judges  # noqa: E402

EVAL_USER = "__eval__"
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN = EVAL_DIR / "golden_set.jsonl"


def corpus_files() -> list[Path]:
    """Corpus documents, in the order they are indexed."""
    return [
        p for p in sorted(CORPUS_DIR.glob("*")) if p.suffix.lower() in (".md", ".txt")
    ]


def reset_eval_collection(cfg) -> None:
    """Drop previously-indexed eval chunks, on EITHER vector backend.

    This used to reach straight into ``ragchat.store.get_client()`` (Chroma),
    which raises on a Neon/pgvector deploy. Deleting per document id is the
    backend-agnostic equivalent — and unlike ``prune_chunks(user, set())`` it
    actually removes rows (prune deliberately no-ops on an empty valid-doc set
    so it can never wipe a user who simply has no documents yet).

    ``load_corpus`` uses the filename as the doc_id, so the two agree.
    """
    for path in corpus_files():
        try:
            delete_document_chunks(EVAL_USER, path.name, cfg.embedding_model)
        except Exception as exc:  # a stale-index warning must not abort the run
            print(f"  (could not reset {path.name}: {exc})")


def load_corpus(cfg) -> int:
    """Index every corpus file for the eval user. Returns the chunk count."""
    total = 0
    for path in corpus_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        n = ingest_document_text(EVAL_USER, path.name, path.name, text, cfg)
        total += n
        print(f"  indexed {path.name} ({n} chunks)")
    return total


def load_golden(limit: int | None = None) -> list[dict]:
    """Parse golden_set.jsonl (optionally truncated to `limit` questions)."""
    items = [
        json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()
    ]
    return items[:limit] if limit else items


def embed_passages(
    passages: list[str], model: str, provider: str | None = None
) -> list[list[float]]:
    """Embed golden passages with the LIVE embedding provider.

    `provider` is required in practice: omitting it made ProxyEmbeddings fall
    back to the EMBEDDING_PROVIDER *env default*, so with the Settings UI set to
    OpenRouter every benchmark run sent an OpenRouter model id to Google's
    endpoint and 404'd on the first scoring step (indexing had already used the
    right provider, which is why only scoring broke). Env vars are boot
    defaults, never live behaviour — read the config.
    """
    if not passages:
        return []
    emb = ProxyEmbeddings(model, provider=provider)
    return emb.embed_documents(passages)


def judge_answer(question: str, expected: str, answer: str) -> tuple[bool, str]:
    # Kept for backwards compatibility; correctness now routed through judges.py
    return judges.answer_correctness(question, expected, answer)


def _safe_judge(fn, *args) -> tuple[bool | None, str]:
    """Run a judge, converting a JudgeError into (None, reason).

    None means "not graded" — distinct from False ("graded, failed"). The
    caller drops None from the aggregate rather than counting it as a miss,
    so a broken judge can no longer silently drag a metric to 0%.
    """
    try:
        return fn(*args)
    except judges.JudgeError as exc:
        return None, f"judge unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"judge error: {exc}"


def score_item(
    item: dict,
    cfg,
    retrieval_only: bool = False,
    rerank: bool = False,
    ceiling: bool = False,
) -> dict:
    """Score ONE golden question end-to-end and return its result entry.

    Factored out of run_benchmark so the serverless chunked runner can score a
    handful of questions per HTTP request (see ragchat/app.py) while the CLI
    still runs the whole set in one process.

    Every retrieval metric is measured over the SAME list — the passages the
    generator would actually be handed. That was not true before, and it made
    the scorecard incoherent: retrieve() returns cfg.candidate_k chunks (the
    pool), ask() then reranks and keeps cfg.top_k. context_recall was computed
    over all 40, while precision/MRR/hit/NDCG sliced [:6]. So "Context Recall
    49%" sat beside "Hit rate 41%" describing different retrievals, and the
    headline metric was inflated by 34 chunks nothing downstream ever sees.

    `rerank` decides which list that is:
      False -> the pool's own top_k, PRE-rerank. Costs no model call, which is
               what makes the CI gate free. Report it as a pre-rerank number.
      True  -> the reranked top_k, i.e. what the deployment really answers from.
               Costs one Cohere call per question.
    The old behaviour reranked in neither case, so a preset with reranker: True
    scored exactly as if it were False.

    `ceiling` additionally reports recall over the whole candidate pool. That is
    the diagnostic that separates "the right passage was never retrieved" from
    "it was retrieved and the ranking buried it" — different bugs with different
    fixes. Off by default because it means embedding candidate_k chunks per
    question instead of top_k, several times the embedding spend.
    """
    question = item["question"]
    golden_passages = item.get("golden_passages", [])
    golden_embs = item.get("_golden_embs")
    if golden_embs is None:
        golden_embs = embed_passages(
            item.get("golden_passages", []),
            cfg.embedding_model,
            provider=cfg.embedding_provider,
        )

    pool = retrieve(EVAL_USER, question, cfg)
    # A full run ALWAYS reranks for the retrieval metrics, whatever the caller
    # passed. In full mode ask() reranks internally, so leaving these metrics on
    # the pre-rerank order would describe a different list from the one the
    # judges graded — faithfulness scoring the reranked passages while context
    # recall scored the unreranked ones. The cost is one extra rerank call per
    # question; ask() cannot hand its chunk list back without a wider refactor,
    # and a cheap duplicated call is the better trade against an incoherent
    # scorecard. --with-rerank is therefore only meaningful with
    # --retrieval-only, where skipping the call is the whole point.
    if not retrieval_only:
        rerank = True
    # Narrow the pool to what the generator would receive, by the same route the
    # pipeline uses. _rerank is a no-op when cfg.reranker is off, so asking for
    # rerank on a preset that does not rerank still measures that preset.
    if rerank:
        final = _rerank(question, [dict(c) for c in pool], cfg)
    else:
        final = pool[: cfg.top_k]

    # Embed with the same model used for the golden passages so we can
    # cosine-match (user decision #3: embedding-cosine).
    _emb = ProxyEmbeddings(cfg.embedding_model, provider=cfg.embedding_provider)
    final_texts = [c["text"] for c in final]
    chunk_embs = _emb.embed_documents(final_texts) if final_texts else []

    k = cfg.top_k
    # The ruler belongs to the embedder, not to the harness. Reading the module
    # global here would measure an --embedding-model override with a threshold
    # calibrated for a DIFFERENT model, which is how a perfectly good embedder
    # reads as catastrophic (eval/metrics.py).
    thr = match_threshold(cfg.embedding_model)
    cr = context_recall(chunk_embs, golden_embs, thr)
    entry = {
        "question": question,
        "unanswerable": item["unanswerable"],
        "needs": item.get("needs", ["single_passage"]),
        "type": item.get("type", ""),
        "expected_source": item.get("golden_doc", ""),
        "expected": item.get("expected", ""),
        "context_recall": round(cr, 4),
        "precision_at_k": round(precision_at_k(chunk_embs, golden_embs, k, thr), 4),
        "mrr": round(mrr_at_k(chunk_embs, golden_embs, k, thr), 4),
        "ndcg_at_k": round(ndcg_at_k(chunk_embs, golden_embs, k, thr), 4),
        "hit_rate_at_k": hit_rate_at_k(chunk_embs, golden_embs, k, thr),
        # Deterministic twins of the four above. No threshold, no embedding, no
        # false anything — the CI gate compares these, because the cosine ones
        # drift upward as the corpus grows (see metrics.py).
        "exact_context_recall": round(exact_context_recall(final_texts, golden_passages), 4),
        "exact_precision_at_k": round(exact_precision_at_k(final_texts, golden_passages, k), 4),
        "exact_mrr": round(exact_mrr_at_k(final_texts, golden_passages, k), 4),
        "exact_hit_rate_at_k": exact_hit_rate_at_k(final_texts, golden_passages, k),
        "retrieved": [
            {"title": c["title"], "similarity": round(c["similarity"], 4)}
            for c in final
            if c.get("similarity") is not None
        ],
    }
    if ceiling:
        pool_texts = [c["text"] for c in pool]
        pool_embs = _emb.embed_documents(pool_texts) if pool_texts else []
        entry["context_recall_at_candidate_k"] = round(
            context_recall(pool_embs, golden_embs, thr), 4
        )
    if retrieval_only:
        return entry

    # use_gold=False: this IS the golden run. Every question here matches the
    # bank by construction, so re-matching would only spend embeddings on
    # scoring the harness already does itself.
    res = ask(EVAL_USER, question, [], cfg, use_gold=False)
    entry["answer"] = res["answer"]
    entry["not_found"] = res["not_found"]
    # The context the model was ACTUALLY given, straight from ask().
    #
    # This line read `chunks[:k]`, and `chunks` has not existed since the
    # retrieval refactor — so every full-mode run died here with a NameError,
    # including the in-app "Run benchmark" button, which reaches this code as
    # soon as it finishes indexing. Only --retrieval-only was ever exercised,
    # so nothing caught it.
    #
    # Rebuilding from `final` would be a RECONSTRUCTION, and grading an answer
    # against one is the same error c002445 fixed for the retrieval metrics:
    # ask() calls rewrite_query, which is a model call, so a second retrieval
    # can legitimately return a different list than the one behind the answer.
    # The fallback covers the not-found path, which returns before building a
    # context at all.
    context_text = res.get("context") or "\n\n".join(
        f"[{j+1}] {c['title']}\n{c['text']}" for j, c in enumerate(final[:k])
    )
    if item["unanswerable"]:
        # Correct iff the system refused (did not fabricate).
        entry["correct"] = bool(res["not_found"])
        entry["faithful"] = None
        entry["relevant"] = None
    else:
        entry["faithful"], entry["faithful_reason"] = _safe_judge(
            judges.faithfulness, question, context_text, res["answer"]
        )
        entry["relevant"], entry["relevant_reason"] = _safe_judge(
            judges.answer_relevancy, question, res["answer"]
        )
        entry["correct"], entry["correct_reason"] = _safe_judge(
            judges.answer_correctness,
            question,
            item["expected"],
            res["answer"],
            item.get("golden_passages"),
        )
    return entry


def aggregate(results: list[dict], retrieval_only: bool = False) -> dict:
    """Aggregate per-question entries into the scorecard metrics."""
    answerable = [r for r in results if not r["unanswerable"]]
    unanswerable = [r for r in results if r["unanswerable"]]
    multi_doc = [r for r in answerable if "multi_doc" in r.get("needs", [])]

    def _mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def _rate(rows, key):
        """Pass-rate over rows that were actually graded (None = ungraded)."""
        graded = [r for r in rows if r.get(key) is not None]
        return _mean([1 if r[key] else 0 for r in graded]) if graded else None

    metrics = {
        "context_recall": _mean([r["context_recall"] for r in answerable]),
        "precision_at_k": _mean([r["precision_at_k"] for r in answerable]),
        "mrr": _mean([r["mrr"] for r in answerable]),
        "ndcg_at_k": _mean([r["ndcg_at_k"] for r in answerable]),
        "hit_rate_at_k": _mean([r["hit_rate_at_k"] for r in answerable]),
        "exact_context_recall": _mean([r.get("exact_context_recall", 0.0) for r in answerable]),
        "exact_precision_at_k": _mean([r.get("exact_precision_at_k", 0.0) for r in answerable]),
        "exact_mrr": _mean([r.get("exact_mrr", 0.0) for r in answerable]),
        "exact_hit_rate_at_k": _mean([r.get("exact_hit_rate_at_k", 0) for r in answerable]),
    }
    # Present only on a --ceiling run. Recall over the whole candidate pool: how
    # much the ranking is leaving on the table. A large gap between this and
    # context_recall is a RANKING problem (the passage was fetched and buried);
    # both being low is a RETRIEVAL problem (it was never fetched at all).
    ceiling_rows = [
        r for r in answerable if r.get("context_recall_at_candidate_k") is not None
    ]
    if ceiling_rows:
        metrics["context_recall_at_candidate_k"] = _mean(
            [r["context_recall_at_candidate_k"] for r in ceiling_rows]
        )
    if not retrieval_only:
        metrics["faithfulness"] = _rate(answerable, "faithful")
        metrics["answer_relevancy"] = _rate(answerable, "relevant")
        metrics["answer_correctness"] = _rate(answerable, "correct")
        metrics["not_found_rate_unanswerables"] = _rate(unanswerable, "correct")
        if multi_doc:
            metrics["context_recall_multi_doc"] = _mean(
                [r["context_recall"] for r in multi_doc]
            )
        # How many judge calls came back ungraded — surfaced so a broken judge
        # shows up as an explicit count instead of a mysteriously low score.
        metrics["n_ungraded"] = sum(
            1
            for r in answerable
            if (
                r.get("faithful") is None
                or r.get("relevant") is None
                or r.get("correct") is None
            )
        )
    metrics["n_answerable"] = len(answerable)
    metrics["n_unanswerable"] = len(unanswerable)
    metrics["n_multi_doc"] = len(multi_doc)
    return metrics


def config_snapshot(cfg) -> dict:
    """The config the run was executed under (PRD F19), for report comparability."""
    return {
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "splitter": cfg.splitter,
        "top_k": cfg.top_k,
        "candidate_k": cfg.candidate_k,
        "similarity_threshold": cfg.similarity_threshold,
        "hybrid_search": cfg.hybrid_search,
        "reranker": cfg.reranker,
        "reranker_provider": cfg.reranker_provider,
        "query_rewrite": cfg.query_rewrite,
        "llm_model": cfg.llm_model,
        "judge_model": judges.judge_model(),
        "temperature": cfg.temperature,
        "embedding_model": cfg.embedding_model,
        "embedding_provider": cfg.embedding_provider,
        "fingerprint": cfg.fingerprint(),
    }


def run_benchmark(
    limit: int | None = None,
    retrieval_only: bool = False,
    rerank: bool = False,
    ceiling: bool = False,
    preset: str | None = None,
    embedding_model: str | None = None,
) -> dict:
    """Run the full eval harness and return the report dict.

    Callable form used by the in-app Evaluation tab (POST /api/eval/run).
    Mirrors the CLI `main()` exactly, but returns the report instead of only
    printing it, so the web layer can persist it to a status file and the UI
    can poll for progress.

    `preset` scores one of the named Settings configurations (ragchat.presets)
    instead of the live config. The override is applied IN MEMORY with
    dataclasses.replace and never persisted: `config_overrides` is a single row
    shared by every user of the deployment, so a benchmark that wrote it would
    re-point the pipeline for everyone for the duration of the run — and leave it
    re-pointed if the run died halfway.
    """
    cfg = load_config()
    if preset:
        from dataclasses import replace

        from ragchat.presets import get_preset

        entry = get_preset(preset)
        if entry is None:
            raise SystemExit(f"Unknown preset: {preset}")
        cfg = replace(cfg, **entry["values"])
        print(f"Preset: {entry['name']} ({preset}) — {entry['values']}")
    if embedding_model:
        # Comparing embedders is exactly what this harness is for, and until now
        # the only way to do it was to edit config.yaml — which re-points the
        # pipeline for every caller and leaves it re-pointed if the run dies.
        # Applied in memory like --preset, and it changes the fingerprint, so
        # the corpus is re-indexed under the new model rather than scored
        # against vectors made by the old one.
        from dataclasses import replace as _replace

        cfg = _replace(cfg, embedding_model=embedding_model)
        print(f"Embedding model: {embedding_model}")
    print(f"Config fingerprint: {cfg.fingerprint()}")
    print(
        f"Judge model: {judges.judge_model()} | match threshold (cosine): "
        f"{match_threshold(cfg.embedding_model)} for {cfg.embedding_model}"
    )
    print("Indexing corpus...")
    reset_eval_collection(cfg)
    load_corpus(cfg)

    items = load_golden(limit)

    # Pre-embed golden passages once per question.
    for it in items:
        it["_golden_embs"] = embed_passages(
            it.get("golden_passages", []),
            cfg.embedding_model,
            provider=cfg.embedding_provider,
        )

    results = []
    for i, item in enumerate(items, start=1):
        entry = score_item(
            item, cfg, retrieval_only=retrieval_only,
            rerank=rerank, ceiling=ceiling,
        )
        results.append(entry)
        status = "✓" if entry.get("correct", entry["context_recall"] > 0) else "✗"
        print(f"  [{i}/{len(items)}] {status} {entry['question'][:60]}")

    metrics = aggregate(results, retrieval_only=retrieval_only)

    run_dir = EVAL_DIR / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        # Which named configuration this run scored, if any. Without it, comparing
        # four runs means reverse-engineering the preset from the config snapshot.
        "preset": preset,
        "embedding_model_override": embedding_model,
        # Which pipeline the numbers describe. A pre-rerank run and a reranked
        # run measure different lists, so their scores are not comparable — and
        # a report that does not say which it was invites exactly that mistake.
        "mode": (
            "full" if not retrieval_only
            else "retrieval+rerank" if rerank
            else "retrieval-pre-rerank"
        ),
        "config": config_snapshot(cfg),
        "metrics": metrics,
        "results": results,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))

    def _m(key):
        """Render a metric, or an explicit dash when nothing was graded."""
        v = metrics.get(key)
        return "—" if v is None else v

    md = [
        f"# Eval run {run_dir.name}",
        "",
        f"- Config: `{json.dumps(report['config'])}`",
        "",
        "| Metric | Value | Target |",
        "|---|---|---|",
        f"| Context Recall | {_m('context_recall')} | ≥ 0.80 |",
        f"| Precision@k | {_m('precision_at_k')} | ≥ 0.70 |",
        f"| MRR | {_m('mrr')} | ≥ 0.65 |",
        f"| NDCG@k | {_m('ndcg_at_k')} | ≥ 0.70 |",
        f"| Hit Rate@k | {_m('hit_rate_at_k')} | ≥ 0.80 |",
    ]
    if not retrieval_only:
        md += [
            f"| Faithfulness | {_m('faithfulness')} | ≥ 0.90 |",
            f"| Answer Relevancy | {_m('answer_relevancy')} | ≥ 0.85 |",
            f"| Answer Correctness | {_m('answer_correctness')} | ≥ 0.80 |",
            f"| Not-found rate (unanswerables) | {_m('not_found_rate_unanswerables')} | ≥ 0.90 |",
        ]
        if metrics.get("n_ungraded"):
            md += ["", f"> ⚠ {metrics['n_ungraded']} question(s) could not be graded — check the judge model."]
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
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip generation and the judges. Free unless --with-rerank is given.",
    )
    ap.add_argument(
        "--with-rerank",
        action="store_true",
        help=(
            "Score the RERANKED top_k — what the deployment actually answers "
            "from — instead of the pool's pre-rerank order. Costs one rerank "
            "call per question, so the CI gate leaves it off."
        ),
    )
    ap.add_argument(
        "--ceiling",
        action="store_true",
        help=(
            "Also report recall over the whole candidate pool, to separate a "
            "ranking problem from a retrieval one. Embeds candidate_k chunks "
            "per question instead of top_k."
        ),
    )
    ap.add_argument(
        "--embedding-model",
        default=None,
        help="score a different embedding model instead of the configured one "
             "(applied in memory, never persisted). Changes the fingerprint, so "
             "the corpus is re-indexed under it.",
    )
    ap.add_argument(
        "--preset",
        default=None,
        help="score a named Settings preset instead of the live config "
             "(applied in memory, never persisted)",
    )
    args = ap.parse_args()
    run_benchmark(
        embedding_model=args.embedding_model,
        limit=args.limit,
        retrieval_only=args.retrieval_only,
        rerank=args.with_rerank,
        ceiling=args.ceiling,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()
