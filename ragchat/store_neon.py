"""Postgres + pgvector vector store (backend "neon", for Vercel/serverless).

Drop-in replacement for ``ragchat.store`` (Chroma). Mirrors the exact public
interface (``collection_name``, ``add_chunks``, ``query_chunks``,
``delete_document_chunks``) so callers don't change.

Single table ``chunks``. Per-user + per-model isolation is enforced entirely by
``WHERE`` clauses (the ``id`` also bakes in ``user_id`` and ``doc_id`` for
extra safety). Embeddings are fixed at 768 dimensions (``text-embedding-004``)
so the pgvector HNSW index stays well under the 2000-dim ceiling.

Hybrid search (``bm25_index=True``) uses STATELESS Postgres full-text search
(``to_tsvector``/``ts_rank``) fused with the vector results via Reciprocal Rank
Fusion (k=60) in Python. No in-memory BM25 is used, because serverless has no
persistent memory between requests (PRD §5 hybrid_search).
"""
from __future__ import annotations

import re

from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine

from pgvector.sqlalchemy import Vector

from .config import settings

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DIM = 768

_metadata = MetaData()

chunks_table = Table(
    "chunks",
    _metadata,
    Column("id", String, primary_key=True),
    Column("user_id", String, nullable=False),
    Column("embedding_model", String, nullable=False),
    Column("doc_id", String, nullable=False),
    Column("title", String, nullable=False),
    Column("fingerprint", String, nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("ref", String, nullable=False),
    Column("text", String, nullable=False),
    Column("embedding", Vector(DIM), nullable=False),
)


_engine: Engine | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = settings.pg_url
        if not url:
            raise RuntimeError(
                "VECTOR_BACKEND=neon requires PG_DATABASE_URL (or DATABASE_URL) "
                "to be set, but it is empty."
            )
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def _ensure_table(conn) -> None:
    """Create the table, the pgvector extension, and the HNSW index once.

    Runs INSIDE the caller's `with eng.begin() as conn:` transaction, so it
    must NOT commit/rollback here — the context manager does that on exit. A
    manual commit inside the begin() context closes the transaction and makes
    the surrounding `with` block raise "Can't operate on closed transaction".
    """
    # CREATE EXTENSION IF NOT EXISTS is idempotent; on managed Postgres that
    # disallows it, the table/index DDL below surfaces a clear error instead.
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    _metadata.create_all(conn)
    # HNSW cosine index — valid at 768 dimensions (well under pgvector's 2000-dim HNSW ceiling).
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
    )
    # B-tree to speed the (user_id, embedding_model, fingerprint) WHERE clauses.
    conn.execute(
        text(
            "CREATE INDEX IF NOT EXISTS chunks_user_model_fp_idx "
            "ON chunks (user_id, embedding_model, fingerprint)"
        )
    )


# ---------------------------------------------------------------------------
# Naming (same as the Chroma store, for run_eval.py compatibility)
# ---------------------------------------------------------------------------


def _slug(user_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", user_id).strip("._-") or "user"
    return slug[:200]


def _model_slug(model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", model).strip("._-") or "emb"
    return slug[:200]


def collection_name(user_id: str, embedding_model: str) -> str:
    """Stable, dimension-isolated collection name for a (user, model) pair."""
    return f"user-{_slug(user_id)}__{_model_slug(embedding_model)}"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def add_chunks(
    user_id: str,
    doc_id: str,
    title: str,
    fingerprint: str,
    texts: list[str],
    embeddings: list[list[float]],
    refs: list[str],
    embedding_model: str = "text-embedding-005",
) -> None:
    """Upsert chunks for a document (delete-then-insert by stable id)."""
    eng = _get_engine()
    with eng.begin() as conn:
        _ensure_table(conn)
        ids = [f"{user_id}::{doc_id}:{i}" for i in range(len(texts))]
        # Remove any previous version of these exact chunks (idempotent upsert).
        conn.execute(chunks_table.delete().where(chunks_table.c.id.in_(ids)))
        rows = [
            {
                "id": ids[i],
                "user_id": user_id,
                "embedding_model": embedding_model,
                "doc_id": doc_id,
                "title": title,
                "fingerprint": fingerprint,
                "chunk_index": i,
                "ref": refs[i],
                "text": texts[i],
                "embedding": list(embeddings[i]),
            }
            for i in range(len(texts))
        ]
        if rows:
            conn.execute(chunks_table.insert(), rows)


def delete_document_chunks(
    user_id: str, doc_id: str, embedding_model: str | None = None
) -> None:
    """Delete all chunks for a document (optionally scoped to one model)."""
    eng = _get_engine()
    with eng.begin() as conn:
        _ensure_table(conn)
        if embedding_model:
            conn.execute(
                chunks_table.delete().where(
                    chunks_table.c.user_id == user_id,
                    chunks_table.c.doc_id == doc_id,
                    chunks_table.c.embedding_model == embedding_model,
                )
            )
        else:
            conn.execute(
                chunks_table.delete().where(
                    chunks_table.c.user_id == user_id,
                    chunks_table.c.doc_id == doc_id,
                )
            )


def query_chunks(
    user_id: str,
    embedding: list[float],
    fingerprint: str,
    n_results: int,
    embedding_model: str = "text-embedding-005",
    bm25_index: bool = False,
    query_text: str | None = None,
) -> list[dict]:
    """Ranked chunks for the user under the current config fingerprint.

    Cosine distance via ``embedding <=> :q`` (pgvector); similarity is
    ``1.0 - distance``. When ``bm25_index`` is set, the vector top-k and a
    stateless Postgres full-text top-k are fused with reciprocal rank fusion
    (RRF, k=60) in Python. ``query_text`` is required for FTS.
    """
    eng = _get_engine()
    with eng.begin() as conn:
        _ensure_table(conn)
        q = list(embedding)
        # Wider vector net when fusing, so FTS can promote chunks the vector
        # ranker missed.
        v_lim = max(n_results, 30) if bm25_index else n_results

        vec_sql = text(
            """
            SELECT id, text, doc_id, title, ref,
                   1.0 - (embedding <=> :q) AS similarity
            FROM chunks
            WHERE user_id = :u
              AND embedding_model = :m
              AND fingerprint = :f
            ORDER BY embedding <=> :q
            LIMIT :lim
            """
        )
        res = conn.execute(
            vec_sql,
            {"q": q, "u": user_id, "m": embedding_model, "f": fingerprint, "lim": v_lim},
        ).mappings().all()

        chunks = [
            {
                "chunk_id": r["id"],
                "text": r["text"],
                "similarity": float(r["similarity"]),
                "doc_id": r["doc_id"],
                "title": r["title"],
                "ref": r["ref"],
            }
            for r in res
        ]

        if not (bm25_index and query_text):
            return chunks

        # ---- Stateless Postgres full-text search (hybrid) ----
        fts_sql = text(
            """
            SELECT id, text, doc_id, title, ref,
                   ts_rank(to_tsvector('simple', text),
                           plainto_tsquery('simple', :q)) AS rank
            FROM chunks
            WHERE user_id = :u
              AND embedding_model = :m
              AND to_tsvector('simple', text) @@ plainto_tsquery('simple', :q)
            ORDER BY rank DESC
            LIMIT :lim
            """
        )
        fres = (
            conn.execute(
                fts_sql,
                {"q": query_text, "u": user_id, "m": embedding_model, "lim": v_lim},
            )
            .mappings()
            .all()
        )

        return _rrf_fuse(chunks, fres, n_results)


def _rrf_fuse(vector_chunks: list[dict], fts_rows: list, n_results: int) -> list[dict]:
    """Reciprocal Rank Fusion (k=60) of vector + Postgres FTS result lists."""
    K = 60
    fused: dict[str, dict] = {}

    # Vector ranking by cosine similarity (descending).
    for rank, c in enumerate(
        sorted(vector_chunks, key=lambda c: c["similarity"], reverse=True), start=1
    ):
        cid = c["chunk_id"]
        fused.setdefault(cid, {"chunk": c, "score": 0.0})
        fused[cid]["score"] += 1.0 / (K + rank)

    # FTS ranking by ts_rank (the result set is already ordered desc).
    for rank, r in enumerate(fts_rows, start=1):
        cid = r["id"]
        if cid not in fused:
            fused[cid] = {
                "chunk": {
                    "chunk_id": cid,
                    "text": r["text"],
                    "similarity": None,
                    "doc_id": r["doc_id"],
                    "title": r["title"],
                    "ref": r["ref"],
                },
                "score": 0.0,
            }
        fused[cid]["score"] += 1.0 / (K + rank)

    ranked = sorted(fused.values(), key=lambda e: e["score"], reverse=True)
    return [e["chunk"] for e in ranked[: max(n_results, 1)]]
