"""The RAG pipeline: ingest (chunk -> embed -> store) and ask
(rewrite -> retrieve -> rerank -> generate with citations, PRD F7, F11-F13).

All pipeline knobs come from config.yaml via PipelineConfig (F16).

Retrieval notes (correctness fixes):
- hybrid_search is REAL keyword fusion (vector + BM25 via RRF). It promotes
  chunks that pure-vector ranking misses (exact IDs, codes, names).
- deep search is a PER-REQUEST flag that adds an exhaustive literal scan of the
  user's own documents (ragchat/deepsearch.py) to the pool. It replaced web
  augmentation, which appended DuckDuckGo snippets: the honest answer to "it is
  in my document and it did not find it" is to search the documents harder, not
  to search somewhere else. This pipeline no longer fetches anything external.
- similarity_threshold drives the not-found path: when no retrieved chunk
  clears it and deep search found nothing literal either, ask() refuses before
  generating rather than spending a call to be told the same. A small default
  floor (below) applies when it is left at 0.0, so a pure-noise top hit still
  counts as "nothing relevant".
"""
from __future__ import annotations

import json
import logging
import re
import time

from .chunking import refine_refs, split_document
from .config import PipelineConfig, settings
from .embeddings import openai_client, ProxyEmbeddings, retry_call, reranker_provider, rerank
from .vectordb import add_chunks, query_chunks

log = logging.getLogger(__name__)

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
- If the sources do not contain the information needed to answer the question, reply with exactly: {not_found}
- Do not use outside knowledge beyond what the excerpts provide. Do not mention these rules.
""".format(not_found=NOT_FOUND_ANSWER)

RERANK_PROMPT = """Score how relevant this passage is to the query on a scale of 0-100.
Reply with ONLY the number, nothing else.

Query: {query}
Passage: {passage}
Score:"""



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

    Two kinds of chunk arrive without a cosine: BM25-only chunks, which fusion
    marks `similarity: None` because keyword search found them and vector search
    did not, and deep-search passages, which are literal hits and never had a
    measured distance at all.

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
        endpoint (fast, cheap, purpose-built). Every passage is reranked
        together, deep-search hits included: a literal match is a candidate and
        not a verdict, so it has to earn its place like anything else.
      - "gemini" (default) -> the original LLM cross-encoder: one chat call per
        chunk, scored 0-100 by the generation LLM.
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
        prompt = RERANK_PROMPT.format(query=query, passage=c["text"][:1200])
        try:
            raw = _chat(cfg.llm_model, [{"role": "user", "content": prompt}], 0.0)
            score = float(raw.strip()) / 100.0
        except Exception:
            score = _fallback_score(c)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[: cfg.top_k]]


_WS = re.compile(r"\s+")


def _drop_duplicates(extra: list[dict], have: list[dict]) -> list[dict]:
    """Remove extra passages whose text the ranked pool already carries.

    Deep search reads whole documents, so on a small corpus its window and a
    retrieved chunk are frequently the SAME prose. Sending both spends context
    on a second copy and, measured live, made answers worse rather than better:
    a question that was answered correctly with deep search off came back "I
    couldn't find this" with it on, because most of what the model received was
    the same text twice.

    Containment is checked on a middle slice rather than the whole passage: the
    two are rarely byte-identical (different boundaries), but if a window's core
    already appears in a retrieved chunk then the chunk is the better copy —
    it is the unit that was indexed, and it carries a real similarity score.
    """
    if not extra or not have:
        return extra
    corpus = "\n".join(_WS.sub(" ", c.get("text") or "") for c in have)
    out = []
    for c in extra:
        t = _WS.sub(" ", c.get("text") or "").strip()
        if not t:
            continue
        # A slice from the middle, long enough to be distinctive and short
        # enough to survive the boundaries differing at both ends.
        start = max(0, len(t) // 2 - 80)
        core = t[start:start + 160]
        if core and core in corpus:
            continue
        out.append(c)
    return out


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        where = f" ({c['ref']})" if c.get("ref") else ""
        parts.append(f"[{i}] {c['title']}{where}\n{c['text']}")
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


def _ungraded_eval(cfg: PipelineConfig) -> dict | None:
    """The eval dict an answer carries before anything has graded it.

    `pending` is what the UI draws its in-progress state from. It is NOT the
    same as `judge_error`: nothing has failed here, the grading simply has not
    happened yet. Conflating the two would put a broken-grader message in front
    of a reader who is two seconds from getting a verdict.
    """
    if not cfg.eval_show:
        return None
    return {"pending": True, "faithful": None, "relevant": None}


def grade_answer(question: str, answer: str, context_text: str, cfg: PipelineConfig) -> dict:
    """Run the judges on an answer that has already been delivered.

    Split out of ask() so the reader is not kept waiting on two model calls
    that grade what they are already reading. Returns the same shape
    _eval_answer does, plus how long grading took.
    """
    t0 = time.time()
    out = _eval_answer(question, answer, context_text, cfg) or {}
    out["grade_ms"] = round((time.time() - t0) * 1000)
    out["pending"] = False
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


def eval_line_from(eval_d: dict | None, latency_ms: float | None) -> str:
    """Rebuild the grey line from a stored eval dict alone.

    The grade route runs a request later and has no chunk pool to count, so the
    two facts the line needs from retrieval — top similarity and how many
    passages came from deep search — are carried in the eval dict itself.
    """
    return _line(eval_d, eval_d.get("top_sim") if eval_d else None,
                 (eval_d or {}).get("deep_n") or 0, latency_ms)


def _build_eval_line(eval_d: dict | None, chunks: list[dict], latency_ms: float) -> str:
    """Compact grey-line string: retrieval sim + deep hits + judge verdicts + latency."""
    sims = [c["similarity"] for c in chunks if c.get("similarity") is not None]
    return _line(eval_d, max(sims) if sims else None,
                 sum(1 for c in chunks if c.get("deep")), latency_ms)


def _line(eval_d: dict | None, top_sim: float | None, deep: int, latency_ms: float | None) -> str:
    parts: list[str] = []
    if top_sim is not None:
        parts.append(f"top sim {top_sim:.2f}")
    if deep:
        parts.append(f"{deep} deep")
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
    deep_search=None,
    grade: bool = True,
) -> dict:
    """Answer a question. Returns {answer, not_found, citations, eval_line, eval}.

    `grade=False` returns the answer WITHOUT running the judges. They are two
    more sequential model calls and, measured on the live provider, they cost
    more than writing the answer did (10.1s against 7.8s) — so the chat route
    hands the answer over first and grades it in a second request via
    `grade_answer`. The benchmark passes grade=True: nothing waits on it there.

    `deep_search` is an optional callable taking the rewritten query and
    returning extra chunk-shaped passages — in practice
    `deepsearch.searcher(db, user_id)`. Passed as a function rather than a
    session so this module stays free of the ORM, and given the REWRITTEN query
    because a follow-up like "and the second one?" must not be scanned for the
    word "second".
    """
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

    # Deep search runs ALWAYS when asked for, not as a fallback. Web
    # augmentation was gated on "the documents did not answer" because pulling
    # in outside text when they did would have broken the grounding promise.
    # This searches the same documents, so there is nothing to protect against
    # — and gating it would defeat the point: a literal hit the ranker missed
    # is most valuable precisely when the ranker is confident, because that is
    # when nobody looks twice.
    deep_chunks: list[dict] = []
    if deep_search is not None:
        try:
            deep_chunks = _drop_duplicates(deep_search(effective_query), chunks)
        except Exception:
            # A scan failing must not cost the visitor their answer; ranked
            # retrieval already has one.
            log.exception("deep search failed for user %s", user_id)

    # Nothing cleared the bar and nothing matched literally: refuse here rather
    # than spending a generation call to be told the same thing.
    #
    # This is where similarity_threshold takes effect. Until now its ONLY use
    # was gating web augmentation, so removing web search would have quietly
    # turned an exposed, tunable setting into a decoration on a meter — a
    # setting that lies is worse than one that does not exist. Deep hits carry
    # `similarity: None` and deliberately count as evidence here: a verbatim
    # occurrence is worth showing the model even when the ranker was cold.
    doc_sims = [c["similarity"] for c in chunks if c.get("similarity") is not None]
    # NO cosine at all is not evidence of irrelevance — it is the absence of
    # evidence. A pool can be entirely BM25-only (fusion marks those
    # `similarity: None`), and that happens on exactly the queries keyword
    # fusion exists for: part numbers, form codes, names. Refusing there would
    # break the case hybrid retrieval was added to fix. Only refuse when there
    # ARE scores to judge by and every one of them falls short.
    nothing_relevant = (
        not deep_chunks
        and bool(doc_sims)
        and max(doc_sims) < _effective_threshold(cfg)
    )
    if chunks and nothing_relevant:
        return {
            "answer": NOT_FOUND_ANSWER,
            "not_found": True,
            "citations": [],
            "eval_line": _build_eval_line(None, chunks, (time.time() - t0) * 1000),
        }

    pool = chunks + deep_chunks
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
                "is_deep": bool(c.get("deep")),
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
                "is_deep": bool(c.get("deep")),
            }
            for i, c in enumerate(pool[: min(2, len(pool))])
        ]
    # Stop the clock HERE. What follows is two judge calls that exist to fill
    # the scorecard, not to answer the question — and on a slow provider they
    # cost more than the answer did (measured: 10.1s of grading against 7.8s of
    # answering). Counting them made a graded answer look twice as slow as the
    # not-found path, which returns before this point and always did report the
    # honest number. The UI renders this as "Answered in".
    answer_ms = (time.time() - t0) * 1000
    eval_d = _eval_answer(effective_query, answer, context, cfg) if grade else _ungraded_eval(cfg)
    # Enrich the eval dict with the retrieval top-similarity and end-to-end
    # latency so the UI can render a self-contained, readable performance block
    # (no need to parse the terse eval_line string).
    if eval_d is not None:
        sims = [c["similarity"] for c in pool if c.get("similarity") is not None]
        eval_d["top_sim"] = round(max(sims), 4) if sims else None
        eval_d["latency_ms"] = round(answer_ms)
        # What grading added on top. Reported separately rather than folded
        # in: the reader waits for both, so hiding one would be the same
        # dishonesty in the other direction.
        if grade:
            eval_d["grade_ms"] = round((time.time() - t0) * 1000 - answer_ms)
        # The grader needs the count of deep hits to rebuild this line later,
        # and it will not have the pool by then.
        eval_d["deep_n"] = sum(1 for c in pool if c.get("deep"))
    return {
        "answer": answer,
        "not_found": False,
        "citations": citations,
        "eval_line": _build_eval_line(eval_d, pool, answer_ms),
        "eval": eval_d,
        # The passages this answer was ACTUALLY built from. The benchmark's
        # faithfulness judge needs them, and reconstructing them by re-running
        # retrieval is how you end up grading an answer against a different set
        # of passages than the model saw — the same mistake c002445 fixed for
        # the retrieval metrics. rewrite_query is a model call, so a second
        # retrieval is not guaranteed to return the same list.
        "context": context,
        # The query the answer was built from — the rewrite, not what the user
        # typed. A later grade call must judge the same one.
        "effective_query": effective_query,
    }
