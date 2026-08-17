"""The RAG pipeline: ingest (chunk -> embed -> store) and ask
(rewrite -> retrieve -> rerank -> generate with citations, PRD F7, F11-F13).

All pipeline knobs come from config.yaml via PipelineConfig (F16).

Retrieval notes (correctness fixes):
- hybrid_search is REAL keyword fusion (vector + BM25 via RRF), not web
  search. It promotes chunks that pure-vector ranking misses (exact IDs,
  codes, names) without ever pulling in unverified external text.
- web_augmentation is a SEPARATE, clearly-labeled, default-OFF flag that
  appends DuckDuckGo results as labeled [web] chunks. It is a fallback only:
  it is never used when the user's own documents already answer the question,
  so the "grounded in your documents" guarantee is preserved by default.
- similarity_threshold actually drives the not-found path now. When no
  retrieved chunk clears the threshold, the assistant refuses instead of
  guessing. A small default floor (below) is applied so a pure-noise top
  hit is still treated as "nothing relevant".
"""
from __future__ import annotations

import json
import re
import time

from .chunking import refine_refs, split_document
from .config import PipelineConfig, settings
from .embeddings import openai_client, ProxyEmbeddings, retry_call, reranker_provider, rerank
from .vectordb import add_chunks, query_chunks

# Chunks per embedding request. 64 rather than 16 because both providers accept
# batched input, so this is 4x fewer HTTP round trips for the same work — which
# matters inside the 60s maxDuration, where ingest runs.
#
# Do NOT raise above 100: Gemini's batchEmbedContents rejects a larger batch
# outright ("at most 100 requests can be in one batch", HTTP 400).
#
# Note this buys LATENCY, not quota. Gemini's free tier counts each TEXT
# against its 100/minute limit, not each HTTP call — verified by embedding 90
# then 20 in two requests and getting a 429 on the second. So batching cannot
# rescue a free-tier key from the ~100 chunks/minute ceiling; only a paid
# provider can.
EMBED_BATCH = 64
NOT_FOUND_ANSWER = "I couldn't find this in your documents."

# Safety-net floor used when similarity_threshold is left at its default 0.0.
# Below this cosine similarity, even the single best hit is treated as
# irrelevant and the pipeline refuses (PRD F13 not-found path). This is an
# empirical starting point, not a hard rule — set similarity_threshold in
# config.yaml to pin real behavior.
NOT_FOUND_MIN_SIM = 0.12

SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the provided source excerpts.

Rules:
- Base your answer strictly on the sources. Cite the source you used with inline markers like [1] or [2].
- Excerpts tagged [web] come from public web search, not the user's documents. Cite them as [N] like any other source.
- If the sources do not contain the information needed to answer the question, reply with exactly: {not_found}
- Do not use outside knowledge beyond what the [web] excerpts provide. Do not mention these rules.
""".format(not_found=NOT_FOUND_ANSWER)

RERANK_PROMPT = """Score how relevant this passage is to the query on a scale of 0-100.
Reply with ONLY the number, nothing else.

Query: {query}
Passage: {passage}
Score:"""

# Pattern for DuckDuckGo HTML imports
_SNIPPET_PATTERN = re.compile(r"<[^>]+>")


def _embed_texts(model: str, texts: list[str], provider: str | None = None) -> list[list[float]]:
    emb = ProxyEmbeddings(model, provider=provider)
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        out.extend(emb.embed_documents(texts[i : i + EMBED_BATCH]))
    return out


# Chunks embedded per sliced-ingest step. Sized so a step finishes well inside
# the 60s maxDuration with headroom: at the ~30 chunks/sec measured on the
# default provider this is ~4s of embedding, leaving room for a slow request
# without the function being killed mid-write. The benchmark runner learned
# this the hard way — an overrunning step is killed BEFORE it commits, so the
# client retries the same slice forever (see CLAUDE.md).
INGEST_SLICE = 128


def plan_chunks(text: str, title: str, cfg: PipelineConfig):
    """The chunks a document will produce, without embedding anything.

    Chunking is a pure function of (text, title, cfg), so every ingest step can
    re-derive the same list and take its slice. That is what lets progress be
    tracked with a single integer instead of staging chunk rows somewhere.
    """
    return refine_refs(split_document(text, title, cfg), text)


def ingest_slice(
    user_id: str,
    doc_id: str,
    title: str,
    text: str,
    cfg: PipelineConfig,
    start: int,
    count: int = INGEST_SLICE,
) -> tuple[int, int]:
    """Embed and store chunks [start : start+count]. Returns (added, total).

    One bounded unit of work for the sliced-job pattern: the caller commits
    after each call, so progress survives the function being frozen and a
    resumed run picks up from `start` rather than re-embedding what is done.
    """
    chunks = plan_chunks(text, title, cfg)
    total = len(chunks)
    window = chunks[start : start + count]
    if not window:
        return 0, total
    texts = [c.text for c in window]
    embeddings = _embed_texts(cfg.embedding_model, texts, provider=cfg.embedding_provider)
    add_chunks(
        user_id, doc_id, title, cfg.fingerprint(), texts, embeddings,
        [c.ref for c in window],
        embedding_model=cfg.embedding_model,
        start_index=start,
    )
    return len(window), total


def ingest_document_text(
    user_id: str,
    doc_id: str,
    title: str,
    text: str,
    cfg: PipelineConfig,
) -> int:
    """Chunk, embed, and store a document's text in ONE call. Returns the count.

    Kept for callers that are not request-bound — folder sync, the benchmark
    corpus, demo seeding. The upload path uses ingest_slice instead, because a
    large document cannot finish inside one serverless request.
    """
    chunks = split_document(text, title, cfg)
    chunks = refine_refs(chunks, text)
    if not chunks:
        return 0
    texts = [c.text for c in chunks]
    embeddings = _embed_texts(cfg.embedding_model, texts, provider=cfg.embedding_provider)
    refs = [c.ref for c in chunks]
    # Embedding model is part of the collection name (store.py), so chunks
    # from different dimensions never collide.
    add_chunks(
        user_id, doc_id, title, cfg.fingerprint(), texts, embeddings, refs,
        embedding_model=cfg.embedding_model,
    )
    return len(chunks)


def _chat(model: str, messages: list[dict], temperature: float) -> str:
    client = openai_client()
    resp = retry_call(
        client.chat.completions.create,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
    )
    return (resp.choices[0].message.content or "").strip()


# A rewritten query is one short line. Anything longer is not a query — it is
# the model narrating — and must not reach the embedder.
_MAX_REWRITE_CHARS = 300

_THINK_CLOSED = re.compile(
    r"<(thought|thinking|reasoning)[\s>].*?</\1>", re.IGNORECASE | re.DOTALL
)
# Same unterminated-wrapper case that defeated the judges (eval/judges.py): when
# reasoning runs into max_tokens the closing tag never arrives, and a pattern
# requiring one lets the whole trace through.
_THINK_OPEN = re.compile(
    r"<(thought|thinking|reasoning)[\s>].*\Z", re.IGNORECASE | re.DOTALL
)


def _clean_rewrite(raw: str, fallback: str) -> str:
    """Extract the rewritten query from a possibly reasoning-wrapped reply.

    The configured chat model is thinking-capable and really does answer this
    prompt with ~1400 characters of ``<thought>`` followed by the one-line query,
    despite being told to reply with the query and nothing else. Returning that
    blob raw meant the ENTIRE reasoning trace became the effective query — so it
    was embedded for retrieval, handed to generation as the question (which is
    why answers echoed the rewritten query back), and shown to the judge as the
    "Question" field.

    Deliberately NOT _clean_answer(): that helper recovers the wrapper's inner
    text when everything sits inside the wrapper, which is right for an answer
    and exactly wrong here — that inner text is the reasoning, and using it as a
    search query is the defect this function exists to prevent.

    Falling back to the original query is always safe: rewriting is an
    optimization for follow-ups, never a requirement for answering.
    """
    text = _THINK_CLOSED.sub("", raw or "")
    text = _THINK_OPEN.sub("", text).strip()
    if not text:
        return fallback
    # Preambles come first and the query last, so take the final non-empty line.
    line = [ln.strip() for ln in text.splitlines() if ln.strip()][-1]
    for label in ("rewritten query:", "search query:", "query:"):
        if line.lower().startswith(label):
            line = line[len(label):].strip()
            break
    line = line.strip("\"'`*").strip()
    # A leftover "<" means a wrapper survived the strip; reject rather than embed it.
    if not line or len(line) > _MAX_REWRITE_CHARS or "<" in line:
        return fallback
    return line


def rewrite_query(
    query: str, history: list[dict], cfg: PipelineConfig
) -> str:
    """Resolve follow-ups against chat history into a standalone query (PRD §5)."""
    if not cfg.query_rewrite or not history:
        return query
    tail = history[-6:]
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in tail)
    prompt = (
        "Rewrite the user's latest question into a standalone search query, "
        "resolving pronouns and references using the conversation. Reply with "
        "only the rewritten query, nothing else.\n\n"
        f"Conversation:\n{convo}\n\nLatest question: {query}"
    )
    try:
        rewritten = _chat(cfg.llm_model, [{"role": "user", "content": prompt}], 0.0)
        return _clean_rewrite(rewritten, query)
    except Exception:
        return query


def retrieve(
    user_id: str, query: str, cfg: PipelineConfig, n_results: int | None = None
) -> list[dict]:
    """Ranked chunks for the user under the current config fingerprint.

    If cfg.hybrid_search is on, the vector top-k and a BM25 keyword top-k are
    fused (RRF) in the store so exact-match terms surface (PRD §5). The
    embedding model is passed so the correct per-model collection is queried.
    """
    emb = ProxyEmbeddings(cfg.embedding_model, provider=cfg.embedding_provider)
    qvec = emb.embed_query(query)
    n = n_results or cfg.candidate_k
    chunks = query_chunks(
        user_id,
        qvec,
        cfg.fingerprint(),
        n,
        embedding_model=cfg.embedding_model,
        bm25_index=cfg.hybrid_search,
        query_text=query if cfg.hybrid_search else None,
    )
    return chunks


# Score given to a chunk the reranker could not score itself. Mid-scale, so an
# unscored chunk neither wins nor is buried.
_NEUTRAL_RERANK_SCORE = 0.5


def _fallback_score(c: dict) -> float:
    """A sortable score for a chunk with no rerank result of its own.

    Two kinds of chunk arrive without a cosine: web results, which never had
    one, and BM25-only chunks, which fusion marks `similarity: None` because
    keyword search found them and vector search did not.

    Both used to reach `scored.sort()` as None and raise
    `TypeError: '<' not supported between NoneType and float`, turning a single
    rate-limited rerank call into a 500 for the whole answer. Keyword fusion is
    unconditional now, so BM25-only chunks are routine and this path is live.

    Not `similarity or 0.5` — that rewrites a legitimate 0.0 into 0.5 and
    promotes the worst chunk in the pool above genuinely mid-ranked ones.
    """
    sim = c.get("similarity")
    return _NEUTRAL_RERANK_SCORE if sim is None else float(sim)


def _rerank(
    query: str, chunks: list[dict], cfg: PipelineConfig
) -> list[dict]:
    """Re-rank chunks to keep the top_k most relevant to `query`.

    When cfg.reranker is off, or there are few enough chunks already, we
    return the vector top_k unchanged.

    Provider behaviour (from cfg.reranker_provider, i.e. the Settings choice):
      - "openrouter"      -> Cohere rerank-v3.5 at OpenRouter's /v1/rerank
        endpoint (fast, cheap, purpose-built). All passages (including [web]
        ones) are reranked together.
      - "gemini" (default) -> the original LLM cross-encoder: each non-web
        chunk is scored 0-100 by the generation LLM. Web chunks keep a neutral
        score so they aren't given a spurious LLM number.
    """
    if not cfg.reranker or len(chunks) <= cfg.top_k:
        return chunks[: cfg.top_k]

    # Use the LIVE config's choice, falling back to the env default only when
    # the config doesn't carry one. This previously read reranker_provider()
    # (env-only), so the Settings dropdown had no effect on a deploy.
    provider = (cfg.reranker_provider or reranker_provider()).lower()
    if provider == "openrouter":
        try:
            docs = [c["text"] for c in chunks]
            order = rerank(query, docs, top_n=cfg.top_k)
            return [chunks[i] for i in order]
        except Exception:
            # Reranker unavailable (e.g. key missing) — fall back to vector order.
            return chunks[: cfg.top_k]

    # Default: slow LLM cross-encoder.
    scored = []
    for c in chunks:
        if str(c.get("doc_id", "")).startswith("web:"):
            scored.append((_fallback_score(c), c))
            continue
        prompt = RERANK_PROMPT.format(query=query, passage=c["text"][:1200])
        try:
            raw = _chat(cfg.llm_model, [{"role": "user", "content": prompt}], 0.0)
            score = float(raw.strip()) / 100.0
        except Exception:
            score = _fallback_score(c)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[: cfg.top_k]]


def _web_search(query: str, n: int) -> list[dict]:
    """Search the web and return chunk-shaped results, labeled [web]."""
    try:
        from ddgs import DDGS
    except ImportError:
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=n))
    except Exception:
        return []
    chunks = []
    for i, r in enumerate(results):
        body = _SNIPPET_PATTERN.sub("", r.get("body", ""))
        chunks.append(
            {
                "text": f"Title: {r.get('title', '')}\n{body}",
                "similarity": None,
                "doc_id": f"web:{i}",
                "title": r.get("title", "Web result"),
                "ref": r.get("href", ""),
            }
        )
    return chunks


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        where = f" ({c['ref']})" if c.get("ref") else ""
        tag = " [web]" if str(c.get("doc_id", "")).startswith("web:") else ""
        parts.append(f"[{i}] {c['title']}{tag}{where}\n{c['text']}")
    return "\n\n".join(parts)


def _effective_threshold(cfg: PipelineConfig) -> float:
    """Threshold used to decide 'nothing relevant' in ask()."""
    return cfg.similarity_threshold if cfg.similarity_threshold > 0 else NOT_FOUND_MIN_SIM


def _eval_answer(question: str, answer: str, context_text: str, cfg: PipelineConfig) -> dict | None:
    """LLM-as-judge faithfulness + relevancy for the live grey eval line.

    Returns {faithful, faithful_reason, relevant, relevant_reason} or None when
    eval is disabled or the judge call fails. Never raises — a judge failure
    must not turn a good answer into a 500.
    """
    if not cfg.eval_show:
        return None
    try:
        from eval.judges import faithfulness, answer_relevancy
    except Exception as exc:
        return {"judge_error": f"judges unavailable: {exc}"}

    def _run(fn, *args):
        """Return (verdict|None, reason). None = not graded, NOT a failure.

        A judge that 404s, times out, or replies without a verdict must never
        be reported as FAIL — that renders as a confident hallucination finding
        when in reality nothing was graded at all.
        """
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

    fh, fh_r = _run(faithfulness, question, context_text, answer)
    ar, ar_r = _run(answer_relevancy, question, answer)
    out = {
        "faithful": fh,
        "faithful_reason": fh_r,
        "relevant": ar,
        "relevant_reason": ar_r,
    }
    # Surface *why* grading is missing so a misconfigured judge model is
    # visible in the UI instead of silently blanking the metrics.
    if fh is None or ar is None:
        out["judge_error"] = fh_r or ar_r or "judge returned no verdict"
    return out


def _clean_answer(answer: str) -> str:
    """Strip model reasoning wrappers (e.g. ``,
    ``<think:6124c78e>...</think:6124c78e>``, ``<reasoning>...</reasoning>``) from a
    generated answer so the user-facing text is clean and citation parsing
    isn't poisoned by a stray not-found string inside the reasoning block.

    Some reasoning-tuned models wrap output in these tags. The final,
    user-facing sentence usually sits OUTSIDE the wrapper, but if the whole
    answer is inside it we keep the inner text so nothing is lost.
    """
    import re as _re

    text = answer or ""
    # Remove full wrapper blocks (tags + contents), case-insensitive.
    stripped = _re.sub(
        r"<(thought|thinking|reasoning)[\s>].*?</\1>",
        "",
        text,
        flags=_re.IGNORECASE | _re.DOTALL,
    )
    stripped = stripped.strip()
    # Fallback: if everything was inside the wrapper, recover its inner text.
    if not stripped:
        inner = _re.search(
            r"<(thought|thinking|reasoning)[\s>].*?</\1>",
            text,
            flags=_re.IGNORECASE | _re.DOTALL,
        )
        if inner:
            stripped = inner.group(0).split(">", 1)[-1].rsplit("<", 1)[0].strip()
    # Drop any leftover self-closing/empty tags and stray markers.
    stripped = _re.sub(r"</?(thought|thinking|reasoning)\s*/?>", "", stripped, flags=_re.IGNORECASE)
    return stripped.strip()


def _build_eval_line(eval_d: dict | None, chunks: list[dict], latency_ms: float) -> str:
    """Compact grey-line string: retrieval sim + web count + judge verdicts + latency."""
    parts: list[str] = []
    sims = [c["similarity"] for c in chunks if c.get("similarity") is not None]
    if sims:
        parts.append(f"top sim {max(sims):.2f}")
    web = sum(1 for c in chunks if str(c.get("doc_id", "")).startswith("web:"))
    if web:
        parts.append(f"{web} web")
    if eval_d:
        if eval_d.get("faithful") is not None:
            parts.append("faith " + ("PASS" if eval_d["faithful"] else "FAIL"))
        if eval_d.get("relevant") is not None:
            parts.append("rel " + ("PASS" if eval_d["relevant"] else "FAIL"))
    if latency_ms is not None:
        parts.append(f"{latency_ms:.0f} ms")
    return " · ".join(parts)


def ask(
    user_id: str,
    query: str,
    history: list[dict],
    cfg: PipelineConfig,
) -> dict:
    """Answer a question. Returns {answer, not_found, citations, eval_line, eval}."""
    t0 = time.time()
    effective_query = rewrite_query(query, history, cfg)
    try:
        chunks = retrieve(user_id, effective_query, cfg)
    except Exception as exc:
        # Embedding/retrieval failure (e.g. transient 429 after retries) must
        # not crash the request with a 500 — return a clean, user-facing answer.
        return {
            "answer": f"I couldn't search your documents right now ({exc}). Please try again in a moment.",
            "not_found": True,
            "citations": [],
        }

    # Decision: do the user's own documents answer this?
    doc_sims = [c["similarity"] for c in chunks if c.get("similarity") is not None]
    doc_max = max(doc_sims) if doc_sims else None
    docs_answer = doc_max is not None and doc_max >= _effective_threshold(cfg)

    # Web augmentation is a fallback ONLY — never used when documents already
    # clear the bar. This preserves grounding-by-default (PRD F13).
    web_chunks: list[dict] = []
    if not docs_answer and cfg.web_augmentation:
        web_chunks = _web_search(effective_query, cfg.top_k)

    pool = chunks + web_chunks
    if pool:
        pool = _rerank(effective_query, pool, cfg)
    else:
        pool = chunks[: cfg.top_k]

    if not pool:
        return {"answer": NOT_FOUND_ANSWER, "not_found": True, "citations": []}

    context = _build_context(pool)
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    user_prompt = (
        f"Sources:\n{context}\n\n"
        + (f"Conversation so far:\n{convo}\n\n" if convo else "")
        + f"Question: {effective_query}"
    )
    try:
        answer = _chat(
            cfg.llm_model,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            cfg.temperature,
        )
    except Exception as exc:
        # Generation failure (quota, model error) must not crash with a 500.
        return {
            "answer": f"I couldn't generate an answer right now ({exc}). Your documents are still indexed — please try again shortly.",
            "not_found": True,
            "citations": [],
        }

    # Strip reasoning wrappers some models emit (e.g. ``) so the
    # user-facing text is clean and the not-found guard / citation parsing
    # aren't poisoned by a stray phrase inside the reasoning block.
    answer = _clean_answer(answer)

    if NOT_FOUND_ANSWER.lower() in answer.lower():
        return {
            "answer": NOT_FOUND_ANSWER,
            "not_found": True,
            "citations": [],
            "eval_line": _build_eval_line(None, pool, (time.time() - t0) * 1000),
        }

    used = sorted(
        {int(m) for m in re.findall(r"\[(\d+)\]", answer) if 1 <= int(m) <= len(pool)}
    )
    citations = []
    for num in used:
        c = pool[num - 1]
        citations.append(
            {
                "number": num,
                "doc_id": c["doc_id"],
                "title": c["title"],
                "ref": c.get("ref") or "",
                "excerpt": c["text"][:400],
                "is_web": str(c.get("doc_id", "")).startswith("web:"),
            }
        )
    if not citations:
        citations = [
            {
                "number": i + 1,
                "doc_id": c["doc_id"],
                "title": c["title"],
                "ref": c.get("ref") or "",
                "excerpt": c["text"][:400],
                "is_web": str(c.get("doc_id", "")).startswith("web:"),
            }
            for i, c in enumerate(pool[: min(2, len(pool))])
        ]
    eval_d = _eval_answer(effective_query, answer, context, cfg)
    # Enrich the eval dict with the retrieval top-similarity and end-to-end
    # latency so the UI can render a self-contained, readable performance block
    # (no need to parse the terse eval_line string).
    if eval_d is not None:
        sims = [c["similarity"] for c in pool if c.get("similarity") is not None]
        eval_d["top_sim"] = round(max(sims), 4) if sims else None
        eval_d["latency_ms"] = round((time.time() - t0) * 1000)
    return {
        "answer": answer,
        "not_found": False,
        "citations": citations,
        "eval_line": _build_eval_line(eval_d, pool, (time.time() - t0) * 1000),
        "eval": eval_d,
    }
