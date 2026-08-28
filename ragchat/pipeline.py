"""The RAG pipeline: ingest (chunk -> embed -> store) and ask
(rewrite -> retrieve -> rerank -> generate with citations, PRD F7, F11-F13).

All pipeline knobs come from config.yaml via PipelineConfig (F16).

Retrieval notes (correctness fixes):
- hybrid_search is REAL keyword fusion (vector + BM25 via RRF). It promotes
  chunks that pure-vector ranking misses (exact IDs, codes, names).
- ask() is handed TOOLS, not flags. Deep search (ragchat/deepsearch.py, an
  exhaustive literal scan of the user's own documents) and web search
  (ragchat/websearch.py) are callables of the rewritten query returning
  chunk-shaped passages. Whether either RUNS is ask()'s decision: it reaches for
  them only when it is about to refuse, documents first and the web last. A
  caller passing None says the tool does not exist for this request.
- The web tool is per-request and is never written to shared config. The old
  web-augmentation feature was deleted for two reasons that still apply: it
  answered "it is in my document and it did not find it" by looking somewhere
  else instead of harder, and it lived in config_overrides, a single row shared
  by the whole deployment.
- similarity_threshold drives the not-found path: when no retrieved chunk
  clears it and deep search found nothing literal either, ask() refuses before
  generating rather than spending a call to be told the same. A small default
  floor (below) applies when it is left at 0.0, so a pure-noise top hit still
  counts as "nothing relevant".
"""
from __future__ import annotations

import json
import logging
import math
import re
import time

from .chunking import refine_refs, split_document
from . import router
from .config import PipelineConfig, settings
from .embeddings import openai_client, ProxyEmbeddings, retry_call, reranker_provider, rerank
from .vectordb import add_chunks, query_chunks

# Deferred at the bottom of the module instead: eval.judges imports ragchat.config,
# and importing it here at module load would risk a circular import. See the
# _eval_answer import block.
NO_ANSWER_DERIVABLE = None  # placeholder; bound lazily in _eval_answer

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

# A browser drives these bounded follow-up requests after an answer arrives.
# They are deliberately not an in-process retry: Vercel freezes a function when
# it responds, so durable grading progress has to be carried by Message.eval_data
# and resumed by another HTTP request.
GRADE_MAX_ATTEMPTS = 4
GRADE_RETRY_AFTER_MS = 1_500

# Wall-clock ceiling for ONE /grade request's judging work. Six sequential judge
# calls (four readings + reference draft + correctness) at an honest 4096-token
# budget no longer fit the 60s maxDuration on the deployment — the probe that
# found the thinking-budget bug made every /grade 504, and a 504 discards the
# whole purchase, draft included. So the grader honours a deadline: when it is
# about to expire it stops rather than starts another call, returns what it has,
# and the caller persists the partial result. The client's existing retry loop
# picks the remaining fields up in a fresh request — sliced across requests
# exactly the way the benchmark is, because a serverless function frozen the
# instant it responds cannot host a long loop.
#
# 36s, not 56s: below we refuse to START a call with less than _SLICE_MIN_SECONDS
# left, and a judge call itself can run ~25s on the proxy — starting one at the
# 44th second would sail past 60s and hand everything back to the 504 problem.
GRADE_MAX_SECONDS = 36
_SLICE_MIN_SECONDS = 12
# Every normal chat answer gets these five live readings. Faithfulness and
# relevancy are judged directly. Correctness is ESTIMATED: the judge drafts an
# expected answer from the retrieved passages and scores the real answer against
# it, because arbitrary questions have no gold answer. The two context checks
# map onto the Precision@k and Context Recall bars the same way - close enough
# to be useful, not the same statistic, so the UI tags all three "estimated".
# Renaming these keys orphans verdicts persisted in Message.eval_data.
LIVE_GRADE_FIELDS = (
    "faithful",
    "relevant",
    "context_relevance",
    "context_sufficiency",
    "correct",
)
_LIVE_GRADE_LABELS = {
    "faithful": "Faithfulness",
    "relevant": "Answer relevancy",
    "context_relevance": "Context relevance",
    "context_sufficiency": "Context sufficiency",
    "correct": "Answer correctness",
}
# Live grading runs the SCORED judge variants, which return a 0-1 reading in
# addition to the verdict; it lands in eval_data as f"{field}_score". The
# scorecard draws the bar from the score (65% fills at 65 against the 86%
# benchmark tick) and keeps the verdict for the passed/failed chip. A score of
# None means the judge predates scoring or did not emit one — the bar falls
# back to the binary 100/0 rendering.
LIVE_SCORE_FIELDS = {f: f"{f}_score" for f in LIVE_GRADE_FIELDS}
# Value carried by eval_data["expected_reason"] when synthesize_expected ruled
# that the passages cannot answer the question. Bound lazily from the judge
# module (its producer) inside _eval_answer — importing it at module load risks
# a circular import, and a module-level re-export would also pin the value at
# import time, the exact bug class that bit JUDGE_MODEL.
NO_ANSWER_DERIVABLE = None  # placeholder; bound lazily in _eval_answer

SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the provided source excerpts.

Rules:
- Base your answer strictly on the sources. Cite the source you used with inline markers like [1] or [2].
- If the sources do not contain the information needed to answer the question, reply with exactly: {not_found}
- Do not use outside knowledge beyond what the excerpts provide. Do not mention these rules.
- A source marked "WEB —" came from a web search, NOT from the user's own documents. Where you rely on one, say so in the sentence itself, for example "According to the web, ...". Never present a WEB source as if it came from their documents.
- Where a WEB source and one of the user's own documents disagree, prefer the document and say that they disagree.
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

    Kept for callers that are not request-bound — the benchmark corpus, demo
    seeding. The upload path uses ingest_slice instead, because a large
    document cannot finish inside one serverless request.
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
    """Resolve follow-ups against chat history into a standalone query (PRD §5).

    Telling the model to leave a standalone question ALONE is not padding — it
    is the correctness of this step. Instructed only to "resolve references",
    it resolves ones that were never there: after two turns about a solar
    battery, "What is the boiler pressure range for the espresso machine?" came
    back, deterministically, as "...for the SunPak 5 espresso machine".
    Retrieval then hunted for a product that does not exist, nothing cleared
    the threshold, and the reader was told "I couldn't find this in your
    documents" — about a fact sitting in the corpus, in the same paragraph as
    the answer they had been given two turns earlier.

    That failure cannot be seen one question at a time, which is how it
    survived: rewriting is skipped entirely when there is no history.
    """
    if not cfg.query_rewrite or not history:
        return query
    tail = history[-6:]
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in tail)
    prompt = (
        "Rewrite the user's latest question into a standalone search query.\n"
        "Resolve a pronoun or reference ONLY where the latest question is "
        "incomplete without the conversation, such as 'what about the second "
        "one?'.\n"
        "If the latest question already stands on its own, reply with it "
        "UNCHANGED.\n"
        "Never add a name, product or subject from an earlier turn that the "
        "latest question does not itself refer to.\n"
        "Reply with only the query, nothing else.\n\n"
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
    """Number the passages for citation, and mark the ones from the web.

    The WEB marker is not decoration. Web passages are the only ones in the
    pool that are not the user's own material, and an answer that blends the
    two without saying which is which breaks the single promise this app makes.
    The model is told what the marker means (SYSTEM_PROMPT), the citation
    carries `is_web`, and this is where the distinction enters the prompt.
    """
    parts = []
    for i, c in enumerate(chunks, start=1):
        where = f" ({c['ref']})" if c.get("ref") else ""
        tag = "WEB — " if c.get("web") else ""
        parts.append(f"[{i}] {tag}{c['title']}{where}\n{c['text']}")
    return "\n\n".join(parts)


def _effective_threshold(cfg: PipelineConfig) -> float:
    """Threshold used to decide 'nothing relevant' in ask()."""
    return cfg.similarity_threshold if cfg.similarity_threshold > 0 else NOT_FOUND_MIN_SIM


def _eval_answer(
    question: str,
    answer: str,
    context_text: str,
    cfg: PipelineConfig,
    previous: dict | None = None,
    expected: str | None = None,
    expected_source: str | None = None,
) -> dict | None:
    """Judge unresolved live checks without replacing a completed verdict.

    A returned boolean is durable evidence from a completed judge call. Only a
    ``None`` is retried, so one transient provider failure cannot spend another
    call on a metric that already completed — or replace that result with a
    later outage. Finality belongs to the HTTP route, where the persisted attempt
    count is available.

    ``expected``/``expected_source`` carry a KNOWN reference for this question
    (a matched demo-bank entry): correctness is then scored against the human
    answer instead of a drafted one, and the provenance lands in the payload so
    the UI can tell the two apart.
    """
    if not cfg.eval_show:
        return None

    # Deadline for this pass. The route's 60s maxDuration, minus response
    # overhead, decides when to stop STARTING calls; anything unfinished is
    # picked up by the next grade request (persisted partials + client retry).
    started_at = time.monotonic()

    def _time_left() -> float:
        return GRADE_MAX_SECONDS - (time.monotonic() - started_at)

    previous = previous or {}
    done = {field: previous.get(field) is not None for field in LIVE_GRADE_FIELDS}
    values = {
        field: previous.get(field) if done[field] else None
        for field in LIVE_GRADE_FIELDS
    }
    reasons = {
        field: previous.get(f"{field}_reason") if done[field] else ""
        for field in LIVE_GRADE_FIELDS
    }
    # The synthesized reference is an INPUT to correctness, not a verdict. But
    # "no answer derivable" IS a completed verdict for `correct`: the judge
    # decided with certainty that the passages cannot answer, and FAIL is then
    # the honest grade (there was nothing to be right about). It is persisted as
    # expected_answer="" + expected_reason=NO_ANSWER_DERIVABLE, which also lets
    # a retry skip re-purchasing the same draft call.
    try:
        from eval.judges import (
            NO_ANSWER_DERIVABLE_SENTINEL,
            answer_correctness_scored,
            answer_relevancy_scored,
            context_relevance_scored,
            context_sufficiency_scored,
            faithfulness_scored,
            synthesize_expected,
        )
        # Bind the module-level placeholder so comparisons below (and any
        # caller inspecting pipeline.NO_ANSWER_DERIVABLE) see the real sentinel.
        globals()["NO_ANSWER_DERIVABLE"] = NO_ANSWER_DERIVABLE_SENTINEL
    except Exception as exc:
        reason = f"judges unavailable: {exc}"
        for field in LIVE_GRADE_FIELDS:
            if not done[field]:
                reasons[field] = reason
        missing = [
            f"{_LIVE_GRADE_LABELS[field]} unavailable: {reason}"
            for field in LIVE_GRADE_FIELDS
            if not done[field]
        ]
        out = {
            **values,
            **{f"{field}_reason": reasons[field] for field in LIVE_GRADE_FIELDS},
        }
        if missing:
            out["judge_error"] = " · ".join(missing)
        return out

    # A matched bank question's HUMAN reference. It outranks any drafted one:
    # the draft call is never purchased on this path, and a retry re-derives
    # the reference from the persisted fields rather than re-drafting it.
    bank_expected = (expected or "").strip()
    if not bank_expected and previous.get("expected_source") == "bank":
        bank_expected = (previous.get("expected_answer") or "").strip()
    expected_text = previous.get("expected_answer")
    expected_reason = previous.get("expected_reason") or ""
    if values["correct"] is None and (
        previous.get("expected_answer") == ""
        and expected_reason == NO_ANSWER_DERIVABLE
    ):
        # Resume a graded "nothing to be right about" verdict instead of
        # re-running synthesis for an answer it already ruled on.
        values["correct"] = False
        values[LIVE_SCORE_FIELDS["correct"]] = 0.0
        reasons["correct"] = "no reference derivable: the passages do not answer this"
    correct_done = values["correct"] is not None
    # A draft that failed once is retried (the outage may have healed); one that
    # already RANKED as no-answer-derivable was a completed verdict, handled above.

    def _run(fn, *args):
        """Return (verdict, score|None, reason). None verdict = not graded.

        A judge that 404s, times out, or replies without a verdict must never
        be reported as FAIL — that renders as a confident hallucination finding
        when in reality nothing was graded at all. The score rides along: None
        when no verdict, and also None when the judge omitted its SCORE line
        (the bar then falls back to the binary rendering).
        """
        try:
            verdict, score, why = fn(*args)
            return verdict, score, why
        except Exception as exc:  # noqa: BLE001
            return None, None, str(exc)

    def _put(field, res):
        verdict, score, why = res
        values[field] = verdict
        values[LIVE_SCORE_FIELDS[field]] = score
        reasons[field] = why

    # Each stage refuses to START a judge call when too little wall clock
    # remains to expect it back — an unfinished pass is a persisted partial the
    # retry heals, while a 504 would discard even the calls that did finish.
    def _stage(field, fn, *args):
        if done[field]:
            return
        if _time_left() <= _SLICE_MIN_SECONDS:
            reasons[field] = (
                "paused at this request's time limit; grading continues next try"
            )
            return
        _put(field, _run(fn, *args))

    _stage("faithful", faithfulness_scored, question, context_text, answer)
    _stage("relevant", answer_relevancy_scored, question, answer)
    _stage("context_relevance", context_relevance_scored, question, context_text)
    _stage("context_sufficiency", context_sufficiency_scored, question, context_text)

    def _run2(fn, *args):
        """_run for the two-tuple synthesis judge (text, reason)."""
        try:
            text, why = fn(*args)
            return text, why
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

    if not correct_done and _time_left() > _SLICE_MIN_SECONDS:
        if bank_expected:
            # The known answer is DATA, not a model call, so correctness is
            # ONE judge call on this path — there is no draft to buy and no
            # window to pause between reference and score. Both fields persist
            # even if the judge below fails: a retry must re-use this
            # reference, never replace it with a drafted one.
            values["expected_answer"] = bank_expected
            values["expected_source"] = "bank"
            _put("correct", _run(
                answer_correctness_scored, question, bank_expected, answer
            ))
        else:
            # Resume a draft-outage marker ("expected_answer": "" + a reason) by
            # re-running synthesis; NO_ANSWER_DERIVABLE was already converted to a
            # FAIL verdict above. An absent key means "not yet attempted".
            if previous.get("expected_answer") == "" or expected_text is None:
                expected_text, expected_reason = _run2(
                    synthesize_expected, question, context_text
                )
            if expected_text:
                # A real reference exists: score against it and persist it so a
                # retry never re-purchases the same draft call. Correctness is TWO
                # sequential calls (draft + score), so the deadline is re-checked
                # between them rather than only before the pair.
                if _time_left() > _SLICE_MIN_SECONDS:
                    values["expected_answer"] = expected_text
                    _put("correct", _run(
                        answer_correctness_scored, question, expected_text, answer
                    ))
                else:
                    # Draft bought but not yet scored: persist just the text.
                    # A resumed pass finds a non-empty expected_answer with
                    # correct=None, skips synthesis above, and scores directly.
                    values["expected_answer"] = expected_text
            else:
                # No usable text. Two different meanings: the passages genuinely
                # cannot answer (a graded verdict — correctness is FAIL by
                # definition, since there was nothing to be right about), or the
                # draft call itself failed (an outage the retry will heal, and the
                # draft failure reason is stored as data, not an error string).
                values["expected_answer"] = ""
                if expected_reason:
                    values["expected_reason"] = expected_reason
                if expected_reason == NO_ANSWER_DERIVABLE:
                    values["correct"] = False
                    values[LIVE_SCORE_FIELDS["correct"]] = 0.0
                    reasons["correct"] = (
                        "no reference derivable: the passages do not answer this"
                    )

    out = {
        **values,
        **{f"{field}_reason": reasons[field] for field in LIVE_GRADE_FIELDS},
    }
    missing = []
    paused = False
    for field in LIVE_GRADE_FIELDS:
        if values[field] is not None:
            continue
        # A graded "the passages do not answer this" carries its own reason in
        # expected_reason; it is a verdict, not a judge outage.
        if field == "correct" and values.get("expected_reason") == NO_ANSWER_DERIVABLE:
            continue
        cause = values.get("expected_reason") if field == "correct" else ""
        cause = cause or (reasons[field] or "judge returned no verdict")
        if cause.endswith("grading continues next try"):
            # Out of wall clock, out of judge failure: the retry loop will
            # finish this. Naming it "unavailable" would render as a broken
            # grader and waste the reader's patience on a non-problem.
            paused = True
            continue
        if field == "correct":
            # The reference draft is upstream of the correctness judge; if
            # IT failed, naming "no verdict" would send someone debugging
            # the wrong call.
            cause = values.get("expected_reason") or (
                reasons[field] or "judge returned no verdict"
            )
        missing.append(f"{_LIVE_GRADE_LABELS[field]} unavailable: {cause}")
    if missing:
        out["judge_error"] = " · ".join(missing)
    elif paused:
        out["paused"] = True
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
    return {
        "pending": True,
        **{field: None for field in LIVE_GRADE_FIELDS},
        "grade_attempts": 0,
        "grade_max_attempts": GRADE_MAX_ATTEMPTS,
    }


def grade_answer(
    question: str,
    answer: str,
    context_text: str,
    cfg: PipelineConfig,
    previous: dict | None = None,
) -> dict:
    """Run only unresolved judges on an answer already delivered to the reader."""
    t0 = time.time()
    out = _eval_answer(question, answer, context_text, cfg, previous) or {}
    out["grade_ms"] = round((time.time() - t0) * 1000)
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
        if eval_d.get("context_relevance") is not None:
            parts.append("ctx rel " + ("PASS" if eval_d["context_relevance"] else "FAIL"))
        if eval_d.get("context_sufficiency") is not None:
            parts.append("ctx suff " + ("PASS" if eval_d["context_sufficiency"] else "FAIL"))
        if eval_d.get("correct") is not None:
            # Prefer the score: "correct 75" says what "correct PASS" hides.
            cs = eval_d.get("correct_score")
            parts.append(
                "correct " + (f"{round(cs * 100)}" if cs is not None
                              else "PASS" if eval_d["correct"] else "FAIL")
            )
    if latency_ms is not None:
        parts.append(f"{latency_ms:.0f} ms")
    return " · ".join(parts)


def _rank_pool(effective_query: str, chunks: list[dict], deep_chunks: list[dict],
               cfg: PipelineConfig) -> list[dict]:
    """Merge the ranked and literal pools and order them. Called twice when the
    app escalates, which is why it is a function rather than inline code."""
    pool = chunks + deep_chunks
    if pool:
        return _rerank(effective_query, pool, cfg)
    return chunks[: cfg.top_k]


def _write_answer(effective_query: str, pool: list[dict], history: list[dict],
                  cfg: PipelineConfig) -> tuple[str, str]:
    """Generate one answer from one pool. Returns (answer, context).

    Raises whatever the provider raises — the caller decides whether a failure
    is fatal (first attempt) or survivable (an escalation retry, where an
    answer already exists).
    """
    context = _build_context(pool)
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    user_prompt = (
        f"Sources:\n{context}\n\n"
        + (f"Conversation so far:\n{convo}\n\n" if convo else "")
        + f"Question: {effective_query}"
    )
    answer = _chat(
        cfg.llm_model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        cfg.temperature,
    )
    # Strip reasoning wrappers some models emit (e.g. <thought>...) so the
    # user-facing text is clean and the not-found guard / citation parsing
    # aren't poisoned by a stray phrase inside the reasoning block.
    return _clean_answer(answer), context


# A citation marker, in every form the model actually writes one.
#
# It was `\[(\d+)\]`, which reads "[2]" and silently ignores "[2, 3, 4]" — and
# the model writes the second form whenever an answer rests on several sources
# at once, which is most web answers. The result was references in the prose
# with no chip to click, on an app whose whole promise is that you can click
# the citation. Found on the deployment, not in the tests.
_CITE_MARKER = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def cited_in(answer: str) -> list[int]:
    """Every source number referenced in an answer, in order of appearance.

    Shared with the frontend by contract rather than by code: app.js renders
    the same forms as separate clickable markers, so what the reader can click
    and what the citation list contains stay the same set.
    """
    out: list[int] = []
    for group in _CITE_MARKER.findall(answer or ""):
        out.extend(int(part) for part in group.split(",") if part.strip().isdigit())
    return out


def _refused(answer: str) -> bool:
    return NOT_FOUND_ANSWER.lower() in (answer or "").lower()


def _refusal_text(used: list[str]) -> str:
    """What a refusal says depends on how hard the app actually looked.

    "I couldn't find this" after a ranked search, after reading every document
    word for word, and after searching the web as well are three different
    claims, each stronger than the last. Saying which one applies is the
    difference between a system that gave up and one that finished looking.

    Kept as a SUFFIX so the string still starts with NOT_FOUND_ANSWER -
    `_refused` and the frontend's not-found styling both match on that prefix,
    and `_refused` is the escalation's own trigger.
    """
    did = [w for w in (
        "searched every document word for word" if "deep" in used else "",
        "searched the web" if "web" in used else "",
    ) if w]
    if not did:
        return NOT_FOUND_ANSWER
    return NOT_FOUND_ANSWER + " I also " + " and ".join(did) + "."


# ---------------------------------------------------------------------------
# Known-answer readings — what un-greys the ranking rows.
#
# The benchmark measures MRR/NDCG/hit rate against golden_set.jsonl, where each
# question ships with the passages that answer it. A chat answer has no such
# labels, so those rows say "needs a known answer". Two things change that:
#
# 1. A GOLD MATCH. eval/golden.py matches the asked question against the golden
#    bank (the 56 benchmark questions plus demo pairs over the demo corpus).
#    A match hands us the passages that are KNOWN to answer it, and the exact
#    containment twins in eval/metrics.py score the pool against them with no
#    model call at all — the same functions the CI gate trusts.
# 2. A RANK ESTIMATE. Every answer cites pool positions (citation marker [3]
#    IS pool index 3), so "the passage this answer was built from ranked #k of
#    n" is always computable. It measures ordering only — passages that never
#    came back are invisible to it — so it is labelled estimated everywhere it
#    renders, exactly like the judge-drafted correctness estimate.
#
# Neither ever blocks or delays an answer: matching is a string comparison, the
# exact metrics are substring checks, and the one embedding call (graded NDCG,
# below) happens only on a gold match and fails open.
# ---------------------------------------------------------------------------

# Golden-passage vectors, keyed by (provider, model, passage) — keyed by model
# per the CLAUDE.md rule on singletons, never one global. ~150 passages at
# 768 dims is negligible memory.
_GOLD_EMB_CACHE: dict[tuple, list[float]] = {}


def _gold_identity(gold: dict) -> dict:
    """The fields every gold reading carries so the UI can say WHAT was
    measured: which bank question matched, and from which bank."""
    return {
        "idx": gold.get("_idx"),
        "src": gold.get("_src"),
        "question": gold.get("question", ""),
        "unanswerable": bool(gold.get("unanswerable")),
    }


def _golden_ndcg(pool_texts: list[str], passages: list[str], k: int,
                 cfg: PipelineConfig) -> float:
    """Graded NDCG@k against the matched question's passages.

    The exact containment twins need no embeddings, but NDCG grades each chunk
    by its cosine to the passages — the SAME computation run_eval publishes for
    the benchmark bar, so this reading and the bar are one statistic. Any
    embedding failure degrades to None upstack and the row falls back to the
    rank estimate; it must never break an answer that was already written.
    """
    from eval.metrics import ndcg_at_k

    from .embeddings import ProxyEmbeddings
    emb = ProxyEmbeddings(cfg.embedding_model, provider=cfg.embedding_provider)
    key = (cfg.embedding_provider, cfg.embedding_model)
    missing = [p for p in passages if (key, p) not in _GOLD_EMB_CACHE]
    if missing:
        for p, e in zip(missing, emb.embed_documents(missing)):
            _GOLD_EMB_CACHE[(key, p)] = e
    golden_embs = [_GOLD_EMB_CACHE[(key, p)] for p in passages]
    chunk_embs = emb.embed_documents(pool_texts) if pool_texts else []
    return round(ndcg_at_k(chunk_embs, golden_embs, k), 4)


def _gold_scores(gold: dict, pool_texts: list[str],
                 cfg: PipelineConfig) -> dict:
    """Measured retrieval readings for one gold-matched question.

    The exact metrics (verbatim containment) are free, deterministic and the
    ones the CI gate compares — the cosine twins carry a fifth error in both
    directions and drift with corpus size (eval/metrics.py), so a reading that
    claims to be MEASURED uses the twin that cannot lie about containment.
    """
    from eval.metrics import (
        exact_context_recall,
        exact_hit_rate_at_k,
        exact_mrr_at_k,
        exact_precision_at_k,
    )
    passages = gold.get("golden_passages") or []
    k = cfg.top_k
    out = {
        "mrr": round(exact_mrr_at_k(pool_texts, passages, k), 4),
        "hit_rate_at_k": exact_hit_rate_at_k(pool_texts, passages, k),
        "context_recall": round(exact_context_recall(pool_texts, passages), 4),
        "precision_at_k": round(exact_precision_at_k(pool_texts, passages, k), 4),
        "ndcg_at_k": None,
    }
    try:
        out["ndcg_at_k"] = _golden_ndcg(pool_texts, passages, k, cfg)
    except Exception:
        log.exception("golden ndcg scoring failed; row falls back to estimate")
    return out


def _gold_attach(result: dict, gold: dict | None, eval_show: bool) -> dict:
    """Attach the known-answer verdict to results `_ask` could not score.

    `_ask` scores the matched question's retrieval when a pool exists. These
    are the paths left over: refusals (nothing was retrieved — for an
    unanswerable question that refusal IS the verdict being measured) and a
    normal answer to a known-unanswerable question (answering it is the miss
    the not-found rate counts). Infrastructure failures attach nothing: a
    broken retrieval is not a correct refusal, and saying so would flatter the
    system precisely when it is broken.
    """
    if not gold or not eval_show:
        return result
    if result.get("errored"):
        return result
    eval_d = result.get("eval")
    if isinstance(eval_d, dict) and eval_d.get("gold"):
        return result
    refused = bool(result.get("not_found"))
    if gold.get("unanswerable"):
        entry = {**_gold_identity(gold), "refused": refused}
    elif not refused:
        # Answered normally, so the retrieval scores were attached in `_ask`.
        return result
    else:
        # Refused on a question with known passages: nothing cleared the pool,
        # which is a measured zero, not an absence of measurement.
        entry = {
            **_gold_identity(gold),
            "mrr": 0.0, "ndcg_at_k": 0.0, "hit_rate_at_k": 0,
            "context_recall": 0.0, "precision_at_k": 0.0,
        }
    base = eval_d if isinstance(eval_d, dict) else {"pending": False}
    base["gold"] = entry
    result["eval"] = base
    return result


def ask(
    user_id: str,
    query: str,
    history: list[dict],
    cfg: PipelineConfig,
    deep_search=None,
    web_search=None,
    grade: bool = True,
    use_gold: bool = True,
) -> dict:
    """Answer a question — the public entry, with known-answer matching wrapped
    around `_ask` so every exit path (including refusals) gets the verdict.

    `use_gold=False` is for the benchmark harness: it IS the golden run, and
    re-matching each question against the bank would only spend embeddings on
    answers the harness scores itself.

    See `_ask` for the pipeline itself."""
    gold = None
    if use_gold and cfg.eval_show:
        try:
            from eval.golden import match_question
            gold = match_question(query)
        except Exception:
            log.exception("golden match failed; rows stay benchmark-only")
    result = _ask(
        user_id, query, history, cfg,
        deep_search=deep_search, web_search=web_search, grade=grade, gold=gold,
    )
    return _gold_attach(result, gold, cfg.eval_show)


def _ask(
    user_id: str,
    query: str,
    history: list[dict],
    cfg: PipelineConfig,
    deep_search=None,
    web_search=None,
    grade: bool = True,
    gold: dict | None = None,
) -> dict:
    """Answer a question. Returns {answer, not_found, citations, eval_line, eval}.

    `gold` is a matched eval/golden.py bank entry (None = no known answer for
    this question, the overwhelmingly common case). When set, the answerable
    questions' pool is scored against the bank's known passages below.

    `grade=False` returns the answer WITHOUT running the judges. They are two
    more sequential model calls and, measured on the live provider, they cost
    more than writing the answer did (10.1s against 7.8s) — so the chat route
    hands the answer over first and grades it in a second request via
    `grade_answer`. The benchmark passes grade=True: nothing waits on it there.

    `deep_search` and `web_search` are TOOLS: callables taking the rewritten
    query and returning extra chunk-shaped passages - in practice
    `deepsearch.searcher(db, user_id)` and `websearch.searcher()`. Passed as
    functions rather than as a session or an HTTP client so this module stays
    free of both the ORM and the network, and given the REWRITTEN query because
    a follow-up like "and the second one?" must not be searched for the word
    "second".

    Passing None means the tool is NOT AVAILABLE - the visitor switched it off,
    or it is unconfigured, or they are not entitled to it. There is no "use this
    tool every time" mode: whether a tool runs is `ask`'s decision, and all the
    caller controls is which tools exist.
    """
    t0 = time.time()
    effective_query = rewrite_query(query, history, cfg)
    try:
        chunks = retrieve(user_id, effective_query, cfg)
    except Exception as exc:
        # Embedding/retrieval failure (e.g. transient 429 after retries) must
        # not crash the request with a 500 — return a clean, user-facing answer.
        # `errored` marks this as an infrastructure failure, NOT a refusal: a
        # gold-matched unanswerable question must not score a broken search as
        # a correct refusal (`_gold_attach` skips errored results).
        return {
            "answer": f"I couldn't search your documents right now ({exc}). Please try again in a moment.",
            "not_found": True,
            "citations": [],
            "errored": True,
        }

    # Deep search runs ALWAYS when asked for, not as a fallback. Web
    # augmentation was gated on "the documents did not answer" because pulling
    # in outside text when they did would have broken the grounding promise.
    # This searches the same documents, so there is nothing to protect against
    # — and gating it would defeat the point: a literal hit the ranker missed
    # is most valuable precisely when the ranker is confident, because that is
    # when nobody looks twice.
    extra: list[dict] = []          # passages won by a tool, in pool order
    tools_spent: list[str] = []      # which tools have been used, in order
    routed: list[str] = []          # ...and which of those a model chose
    escalated: str | None = None    # ...and what made the app reach for them

    # The tools, in the order the app is allowed to try them. DOCUMENTS FIRST is
    # not a preference: the web is the only source here that is not the user's
    # own material, and reaching for it before the documents have been read
    # literally would answer "your documents did not have it" by looking
    # somewhere else instead of by looking harder. That is the exact mistake the
    # deleted web-augmentation feature made.
    tools: list[tuple[str, object]] = []
    if deep_search is not None:
        tools.append(("deep", deep_search))
    if web_search is not None:
        tools.append(("web", web_search))

    def _reach(why: str) -> bool:
        """Try each unspent tool in order until one returns something.

        `used` is what bounds the escalation: every tool is spent at most once,
        so the ladder has as many rungs as there are tools and cannot become an
        open-ended reason/act loop — which a function frozen the instant it
        responds, with 60 seconds to work in, could not host.

        Returns True if anything new reached the pool.
        """
        nonlocal escalated
        # Ask a model which tool fits before falling back to the fixed order.
        #
        # The order alone cannot tell "this should be in their documents, the
        # ranker missed it" from "this could never have been in a private
        # document", so it pays for a full literal scan of every document
        # before every web search. The router makes that a judgement about the
        # question. It returns None whenever it has no opinion — including when
        # it fails — and then this is exactly the loop it was before.
        order = list(tools)
        if len([t for t in tools if t[0] not in tools_spent]) > 1:
            try:
                pick = router.choose_tool(
                    effective_query,
                    chunks,
                    [n for n, _ in tools if n not in tools_spent],
                    cfg,
                )
            except Exception:
                # choose_tool swallows its own failures, so this should be
                # unreachable — which is exactly why it is here. The guarantee
                # is "the router can improve the choice and can never block
                # it", and a guarantee that depends on another module policing
                # itself is not one.
                log.exception("tool router raised; using the fixed order")
                pick = None
            if pick:
                order.sort(key=lambda t: t[0] != pick)
                routed.append(pick)

        for name, tool in order:
            if name in tools_spent:
                continue
            tools_spent.append(name)
            try:
                found = _drop_duplicates(tool(effective_query), chunks + extra)
            except Exception:
                # A tool failing must not cost the visitor their answer; there
                # is already one in hand by the time any of this runs.
                log.exception("%s search failed for user %s", name, user_id)
                continue
            if found:
                extra.extend(found)
                escalated = why
                return True
        return False

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
    def _nothing_relevant() -> bool:
        return (
            not extra
            and bool(doc_sims)
            and max(doc_sims) < _effective_threshold(cfg)
        )

    # ESCALATION 1 — about to refuse on weak retrieval. Reach for a tool now.
    #
    # This is the cheapest possible place to be agentic: the literal scan costs
    # no model call at all (it reads Document.source_text and matches
    # literally), and this branch only runs on questions that were about to be
    # refused. Nothing is spent on a question that was going to be answered.
    if chunks and _nothing_relevant():
        _reach("weak_retrieval")

    if chunks and _nothing_relevant():
        return {
            "answer": _refusal_text(tools_spent),
            "not_found": True,
            "citations": [],
            "eval_line": _build_eval_line(None, chunks, (time.time() - t0) * 1000),
        }

    pool = _rank_pool(effective_query, chunks, extra, cfg)
    if not pool:
        return {"answer": _refusal_text(tools_spent), "not_found": True, "citations": []}

    try:
        answer, context = _write_answer(effective_query, pool, history, cfg)
    except Exception as exc:
        # Generation failure (quota, model error) must not crash with a 500.
        # `errored`: a broken model call is not a refusal, and a gold-matched
        # unanswerable question must not score it as one.
        return {
            "answer": f"I couldn't generate an answer right now ({exc}). Your documents are still indexed — please try again shortly.",
            "not_found": True,
            "citations": [],
            "errored": True,
        }

    # ESCALATION 2 — the model read the passages and still said the answer is
    # not there. That verdict is about the passages it was HANDED, not about the
    # documents, and it is the strongest signal in the system that ranking
    # dropped something: it comes from the one component that actually read the
    # text.
    #
    # Deliberately NOT driven by the judges. `NOT_FOUND_ANSWER` is entirely
    # faithful to its context and squarely answers the question, so both judges
    # pass it — the grader cannot see this failure. It also arrives a request
    # later now (see `grade`), by which time the reader is already reading.
    #
    # ONE regeneration, whatever the tools turn up. `_reach` walks the remaining
    # tools in order and stops at the first that finds anything, so the cost of
    # this branch is bounded at a single extra generation no matter how many
    # tools exist.
    if _refused(answer) and _reach("model_refused"):
        pool = _rank_pool(effective_query, chunks, extra, cfg)
        try:
            answer, context = _write_answer(effective_query, pool, history, cfg)
        except Exception:
            # The retry is a bonus. Failing it must not cost the refusal we
            # already had, which is a truthful answer in its own right.
            log.exception("escalated retry failed for user %s", user_id)

    if _refused(answer):
        return {
            "answer": _refusal_text(tools_spent),
            "not_found": True,
            "citations": [],
            "eval_line": _build_eval_line(None, pool, (time.time() - t0) * 1000),
        }

    # Named for what it holds. It was `used`, which silently overwrote the list
    # of tools the escalation had spent — `tools_used` shipped as [1] instead of
    # ["deep"], because a citation marker is also a small integer and nothing
    # complained.
    cited_numbers = sorted(
        {n for n in cited_in(answer) if 1 <= n <= len(pool)}
    )
    citations = []
    for num in cited_numbers:
        c = pool[num - 1]
        citations.append(
            {
                "number": num,
                "doc_id": c["doc_id"],
                "title": c["title"],
                "ref": c.get("ref") or "",
                "excerpt": c["text"][:400],
                "is_deep": bool(c.get("deep")),
                "is_web": bool(c.get("web")),
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
                "is_web": bool(c.get("web")),
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
    # A matched, ANSWERABLE bank question carries a human expected answer.
    # Correctness grades against it instead of a model-drafted reference —
    # known-unanswerable matches stay on the refusal path and never get one.
    bank_expected = ""
    if gold is not None and not gold.get("unanswerable"):
        bank_expected = (gold.get("expected") or "").strip()
    eval_d = (
        _eval_answer(
            effective_query, answer, context, cfg,
            expected=bank_expected or None,
            expected_source="bank" if bank_expected else None,
        )
        if grade
        else _ungraded_eval(cfg)
    )
    # An escalation is a thing the APP decided to do, so it is reported even
    # when evaluation is switched off entirely. Hiding a decision the reader did
    # not make would be the worst of the options here.
    if escalated and eval_d is None:
        eval_d = {}
    # Enrich the eval dict with the retrieval top-similarity and end-to-end
    # latency so the UI can render a self-contained, readable performance block
    # (no need to parse the terse eval_line string).
    if eval_d is not None:
        eval_d["escalated"] = escalated
        # WHICH tools were spent, not just that something was. "Searched the web"
        # and "read your documents word for word" are different claims to make
        # to a reader, and only one of them involves material that is not theirs.
        eval_d["tools_used"] = list(tools_spent)
        # Whether a MODEL picked the tool or the fixed order did. Worth
        # separating: one is a judgement about the question and the other
        # is a fallback, and only the first is the app being agentic.
        eval_d["routed"] = list(routed)
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
        # RANK ESTIMATE — the reading that gives the ranking rows something to
        # draw on every answer. Citation marker [3] IS pool index 3, so the
        # first marker names the position of the passage the answer was built
        # on in the reranker's final order. MRR's per-question statistic is the
        # reciprocal of exactly that rank; NDCG with one relevant item is
        # 1/log2(rank+1). There is deliberately NO hit@k estimate: the pool is
        # ALREADY the top-k cut, so anything cited is inside k by construction
        # and a "1" would be a tautology wearing a metric's clothes.
        #
        # Only real markers count. The UI courtesy that cites pool[:2] when the
        # model named nothing is not evidence the reranker ordered well.
        if cited_numbers:
            _r = cited_numbers[0]
            eval_d["cited_rank"] = _r
            eval_d["pool_n"] = len(pool)
            eval_d["mrr_est"] = round(1.0 / _r, 4)
            eval_d["ndcg_est"] = round(1.0 / math.log2(_r + 1), 4)
        # KNOWN ANSWER — the matched bank question's pool scored against its
        # passages (eval/golden.py). Refusals and known-unanswerables are
        # attached by `_gold_attach` back in `ask()`, which sees every exit
        # path; this is the one branch that needs the pool itself.
        if bank_expected:
            # Persist the human reference and its provenance alongside: the
            # deferred grade request (and any retry) finds them in the stored
            # eval dict, so it scores against the bank's known answer and can
            # label the reading as bank truth — never a drafted one.
            eval_d["expected_answer"] = bank_expected
            eval_d["expected_source"] = "bank"
        if gold is not None:
            if gold.get("unanswerable"):
                # The model ANSWERED a question with no document answer: the
                # not-found row's verdict is a miss, whoever wrote the prose.
                eval_d["gold"] = {**_gold_identity(gold), "refused": False}
            else:
                eval_d["gold"] = {
                    **_gold_identity(gold),
                    "refused": False,
                    **_gold_scores(gold, [c["text"] for c in pool], cfg),
                }
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
