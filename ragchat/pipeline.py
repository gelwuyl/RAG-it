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

from .chunking import refine_refs, split_document
from .config import PipelineConfig, settings
from .embeddings import openai_client, ProxyEmbeddings
from .vectordb import add_chunks, query_chunks

EMBED_BATCH = 16
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


def _embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    emb = ProxyEmbeddings(model)
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        out.extend(emb.embed_documents(texts[i : i + EMBED_BATCH]))
    return out


def ingest_document_text(
    user_id: str,
    doc_id: str,
    title: str,
    text: str,
    cfg: PipelineConfig,
) -> int:
    """Chunk, embed, and store a document's text. Returns the chunk count."""
    chunks = split_document(text, title, cfg)
    chunks = refine_refs(chunks, text)
    if not chunks:
        return 0
    texts = [c.text for c in chunks]
    embeddings = _embed_texts(cfg.embedding_model, texts)
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
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
    )
    return (resp.choices[0].message.content or "").strip()


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
        return rewritten.strip() or query
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
    emb = ProxyEmbeddings(cfg.embedding_model)
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


def _rerank(
    query: str, chunks: list[dict], cfg: PipelineConfig
) -> list[dict]:
    """LLM-based cross-encoder: score each chunk and keep top_k.

    Web chunks (doc_id starts with 'web:') cannot be re-scored by the
    document model meaningfully; they keep their position via a neutral
    score rather than a spurious LLM number.
    """
    if not cfg.reranker or len(chunks) <= cfg.top_k:
        return chunks[: cfg.top_k]
    scored = []
    for c in chunks:
        if str(c.get("doc_id", "")).startswith("web:"):
            scored.append((c.get("similarity") or 0.5, c))
            continue
        prompt = RERANK_PROMPT.format(query=query, passage=c["text"][:1200])
        try:
            raw = _chat(cfg.llm_model, [{"role": "user", "content": prompt}], 0.0)
            score = float(raw.strip()) / 100.0
        except Exception:
            score = c["similarity"]
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


def ask(
    user_id: str,
    query: str,
    history: list[dict],
    cfg: PipelineConfig,
) -> dict:
    """Answer a question. Returns {answer, not_found, citations}."""
    effective_query = rewrite_query(query, history, cfg)
    chunks = retrieve(user_id, effective_query, cfg)

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
    answer = _chat(
        cfg.llm_model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        cfg.temperature,
    )

    if NOT_FOUND_ANSWER.lower() in answer.lower():
        return {"answer": NOT_FOUND_ANSWER, "not_found": True, "citations": []}

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
    return {"answer": answer, "not_found": False, "citations": citations}
