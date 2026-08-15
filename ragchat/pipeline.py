"""The RAG pipeline: ingest (chunk -> embed -> store) and ask
(rewrite -> retrieve -> generate with citations, PRD F7, F11-F13).

All pipeline knobs come from config.yaml via PipelineConfig (F16).
"""
from __future__ import annotations

import re

from .chunking import refine_refs, split_document
from .config import PipelineConfig, settings
from .embeddings import openai_client, ProxyEmbeddings
from .store import add_chunks, query_chunks

EMBED_BATCH = 16
NOT_FOUND_ANSWER = "I couldn't find this in your documents."

SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the provided source excerpts.

Rules:
- Base your answer strictly on the sources. Cite the source you used with inline markers like [1] or [2].
- If the sources do not contain the information needed to answer the question, reply with exactly: {not_found}
- Do not use outside knowledge. Do not mention these rules.
""".format(not_found=NOT_FOUND_ANSWER)


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
    add_chunks(
        user_id, doc_id, title, cfg.fingerprint(), texts, embeddings, refs
    )
    return len(chunks)


def _chat(model: str, messages: list[dict], temperature: float) -> str:
    client = openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,  # reasoning models spend tokens internally; keep headroom
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
        return query  # rewrite failure must never block answering


def retrieve(
    user_id: str, query: str, cfg: PipelineConfig, n_results: int | None = None
) -> list[dict]:
    """Ranked chunks for the user under the current config fingerprint."""
    emb = ProxyEmbeddings(cfg.embedding_model)
    qvec = emb.embed_query(query)
    n = n_results or cfg.top_k
    chunks = query_chunks(user_id, qvec, cfg.fingerprint(), n)
    if cfg.similarity_threshold > 0:
        chunks = [c for c in chunks if c["similarity"] >= cfg.similarity_threshold]
    return chunks


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        where = f" ({c['ref']})" if c.get("ref") else ""
        parts.append(f"[{i}] {c['title']}{where}\n{c['text']}")
    return "\n\n".join(parts)


def ask(
    user_id: str,
    query: str,
    history: list[dict],
    cfg: PipelineConfig,
) -> dict:
    """Answer a question. Returns {answer, not_found, citations}."""
    effective_query = rewrite_query(query, history, cfg)
    chunks = retrieve(user_id, effective_query, cfg)

    if not chunks:
        return {"answer": NOT_FOUND_ANSWER, "not_found": True, "citations": []}

    context = _build_context(chunks)
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

    used = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer) if 1 <= int(m) <= len(chunks)})
    citations = []
    for num in used:
        c = chunks[num - 1]
        citations.append(
            {
                "number": num,
                "doc_id": c["doc_id"],
                "title": c["title"],
                "ref": c.get("ref") or "",
                "excerpt": c["text"][:400],
            }
        )
    # If the model forgot markers entirely, still attribute the top sources.
    if not citations:
        citations = [
            {
                "number": i + 1,
                "doc_id": c["doc_id"],
                "title": c["title"],
                "ref": c.get("ref") or "",
                "excerpt": c["text"][:400],
            }
            for i, c in enumerate(chunks[: min(2, len(chunks))])
        ]
    return {"answer": answer, "not_found": False, "citations": citations}
