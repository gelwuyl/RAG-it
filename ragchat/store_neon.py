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

import os
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
from .embeddings import embedding_dim

# ---------------------------------------------------------------------------
# Schema — dimension-aware
# ---------------------------------------------------------------------------
#
# pgvector fixes a column's vector dimension on first row, so a single shared
# `chunks` table can only hold ONE dimension at a time. Embedding models differ
# in dimension (Gemini 768, OpenRouter 1536), so the column must match the
# ACTIVE embedding model. We derive DIM from the current embedding provider and
# recreate the table if the live column dimension differs.
#
# Recreation is destructive (drops all chunks), so it is gated behind
# NEON_ALLOW_DIM_MIGRATION=1. Without it, a dimension mismatch raises a clear,
# actionable error instead of silently losing data.

_metadata = MetaData()


def _target_dim() -> int:
    """Dimension required by the currently-selected embedding provider/model."""
    try:
        from .config import load_config

        cfg = load_config()
        return embedding_dim(cfg.embedding_provider, cfg.embedding_model)
    except Exception:
        return embedding_dim()


def _build_chunks_table(dim: int) -> Table:
    return Table(
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
        Column("embedding", Vector(dim), nullable=False),
    )


# Module-level handle used by add/delete/prune; rebuilt when dim changes.
chunks_table = _build_chunks_table(_target_dim())


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


def _live_embedding_dim(conn) -> int | None:
    """Return the dimension of chunks.embedding as defined in Postgres, or None
    if the table/column doesn't exist yet."""
    row = conn.execute(
        text(
            "SELECT atttypmod FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE c.relname = 'chunks' AND a.attname = 'embedding' "
            "AND NOT a.attisdropped"
        )
    ).fetchone()
    # pgvector stores dim in atttypmod; -1 means unconstrained (no dim set).
    if row is None:
        return None
    return None if row[0] == -1 else int(row[0])


# Which embedding dimension this PROCESS has already prepared the schema for.
#
# _ensure_table is called from all eight store operations and does six round
# trips every time: CREATE EXTENSION, a catalog query for the live dimension,
# SQLAlchemy's create_all (which reflects), and two CREATE INDEX statements.
# Every one is a network hop to Neon. A guest sign-in performs three store
# operations, so it was paying eighteen — measured as the bulk of a 14.3s
# seed_demo_corpus inside a request with a 10s budget.
#
# The DDL is idempotent and the docstring below always claimed "once"; it just
# was not true. Keyed by target dimension rather than a plain bool, so the
# dimension-drift check still re-runs if the target ever changes.
_SCHEMA_READY_FOR_DIM: int | None = None


def _ensure_table(conn) -> None:
    """Create the table, the pgvector extension, and the HNSW index once.

    Runs INSIDE the caller's `with eng.begin() as conn:` transaction, so it
    must NOT commit/rollback here — the context manager does that on exit. A
    manual commit inside the begin() context closes the transaction and makes
    the surrounding `with` block raise "Can't operate on closed transaction".

    Once per PROCESS, not once per call. On a serverless function a process is
    one warm instance, so a cold start still verifies the schema, and any
    instance that has already done so skips straight to the real work.
    """
    global chunks_table, _SCHEMA_READY_FOR_DIM
    target = _target_dim()
    if _SCHEMA_READY_FOR_DIM == target:
        return
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    live_dim = _live_embedding_dim(conn)
    if live_dim is not None and live_dim != target:
        # Dimension drift (e.g. switched Gemini 768 -> OpenRouter 1536). The
        # existing chunks were embedded with the OLD dimension and are invalid
        # for the new model, so the column must be recreated.
        if os.environ.get("NEON_ALLOW_DIM_MIGRATION") == "1":
            conn.execute(text("DROP TABLE IF EXISTS chunks"))
            chunks_table = _build_chunks_table(target)
        else:
            raise RuntimeError(
                f"chunks.embedding dimension mismatch: table is vector({live_dim}), "
                f"but the selected embedding model needs vector({target}). "
                "Switching embedding models requires recreating the chunks table. "
                "Set NEON_ALLOW_DIM_MIGRATION=1 and restart to migrate (drops all "
                "existing chunks, which are invalid for the new model anyway)."
            )
    _metadata.create_all(conn)
    # HNSW cosine index — valid up to pgvector's 2000-dim HNSW ceiling.
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
    # Last: only mark it done once every statement above has succeeded, or a
    # failure part-way would leave the process believing in a schema it never
    # finished building.
    _SCHEMA_READY_FOR_DIM = target


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
    start_index: int = 0,
) -> None:
    """Upsert chunks for a document (delete-then-insert by stable id).

    ``start_index`` is the position of ``texts[0]`` within the whole document.
    Sliced ingest calls this once per slice, and without an offset every slice
    would write ids 0..n-1 and silently overwrite the previous one, leaving a
    document with only its last slice indexed.
    """
    eng = _get_engine()
    with eng.begin() as conn:
        _ensure_table(conn)
        ids = [f"{user_id}::{doc_id}:{start_index + i}" for i in range(len(texts))]
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
                "chunk_index": start_index + i,
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
        # Pass the query vector as a vector-format STRING ("[x,y,z]"). psycopg2
        # serializes a Python list as a Postgres array literal "{x,y,z}", which
        # pgvector cannot parse; the string form casts cleanly via CAST(:q AS vector).
        q = "[" + ",".join(str(float(x)) for x in embedding) + "]"
        # Wider vector net when fusing, so FTS can promote chunks the vector
        # ranker missed.
        v_lim = max(n_results, 30) if bm25_index else n_results

        vec_sql = text(
            """
            SELECT id, text, doc_id, title, ref,
                   1.0 - (embedding <=> CAST(:q AS vector)) AS similarity
            FROM chunks
            WHERE user_id = :u
              AND embedding_model = :m
              AND fingerprint = :f
            ORDER BY embedding <=> CAST(:q AS vector)
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
              AND fingerprint = :f
              AND to_tsvector('simple', text) @@ plainto_tsquery('simple', :q)
            ORDER BY rank DESC
            LIMIT :lim
            """
        )
        fres = (
            conn.execute(
                fts_sql,
                # `fingerprint` matters here as much as on the vector side. Without
                # it the two halves of the fusion searched different corpora: the
                # vector query is fingerprint-scoped, so it correctly ignores chunks
                # left behind by a previous chunking config, but FTS matched them and
                # RRF promoted them into the results. Those rows genuinely persist —
                # delete_document_chunks() scopes by doc and model, not fingerprint,
                # so changing chunk_size/splitter and declining the re-index prompt
                # leaves them in place until prune_chunks() is run. The symptom is
                # hybrid search citing text split under the OLD chunking.
                {
                    "q": query_text,
                    "u": user_id,
                    "m": embedding_model,
                    "f": fingerprint,
                    "lim": v_lim,
                },
            )
            .mappings()
            .all()
        )

        return _rrf_fuse(chunks, fres, n_results)


def prune_chunks(
    user_id: str,
    valid_doc_ids: set[str],
    stale_fingerprints: set[str] | None = None,
) -> int:
    """Delete chunks that are no longer backed by a Document row (orphans),
    and optionally chunks under a stale config fingerprint.

    Returns the number of chunk rows removed. This is the cleanup that keeps
    the vector store from accumulating 'ghost' chunks after deletes/re-indexes.
    """
    eng = _get_engine()
    removed = 0
    with eng.begin() as conn:
        _ensure_table(conn)
        # Orphans: chunk doc_id with no live Document. Guard the empty-set case
        # (NOT IN () is invalid SQL and would otherwise match nothing — but we
        # must never delete a user's chunks just because they have no docs yet).
        if valid_doc_ids:
            res = conn.execute(
                chunks_table.delete().where(
                    chunks_table.c.user_id == user_id,
                    chunks_table.c.doc_id.notin_(valid_doc_ids),
                )
            )
            removed += getattr(res, "rowcount", 0) or 0
        # Stale fingerprints (e.g. old embedding model) — only when explicitly requested.
        if stale_fingerprints:
            res2 = conn.execute(
                chunks_table.delete().where(
                    chunks_table.c.user_id == user_id,
                    chunks_table.c.fingerprint.in_(stale_fingerprints),
                )
            )
            removed += getattr(res2, "rowcount", 0) or 0
    return removed


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


def reassign_user_chunks(old_user_id: str, new_user_id: str) -> int:
    """Re-point every chunk row from one owner to another (guest -> account)."""
    eng = _get_engine()
    with eng.begin() as conn:
        _ensure_table(conn)
        res = conn.execute(
            text("UPDATE chunks SET user_id = :new WHERE user_id = :old"),
            {"new": new_user_id, "old": old_user_id},
        )
        return int(res.rowcount or 0)


def delete_users_chunks(user_ids: list[str]) -> int:
    """Delete every chunk owned by any of these users, in ONE statement.

    The reaper used to remove chunks one document at a time, so clearing a
    guest workspace with the demo corpus plus two uploads cost four round trips
    to Neon before the relational rows were even touched. A sweep of twenty
    such workspaces spent most of its time waiting on the network.

    Deliberately NOT a no-op guard like prune_chunks has: there the empty set
    means "this user has no documents, so do not treat all their chunks as
    orphans", which is a real hazard. Here an empty list means there are no
    users to delete, and the early return says exactly that.
    """
    ids = [u for u in user_ids if u]
    if not ids:
        return 0
    eng = _get_engine()
    with eng.begin() as conn:
        _ensure_table(conn)
        res = conn.execute(
            chunks_table.delete().where(chunks_table.c.user_id.in_(ids))
        )
        return int(res.rowcount or 0)


def copy_user_chunks(
    src_user_id: str, src_doc_id: str, dst_user_id: str, dst_doc_id: str
) -> int:
    """Copy one document's chunks (embeddings included) to another user.

    A fresh `id` is generated per row because it is the primary key; everything
    else, crucially the embedding and the fingerprint, is carried across
    verbatim so the copy is retrievable exactly like the original.
    """
    eng = _get_engine()
    with eng.begin() as conn:
        _ensure_table(conn)
        res = conn.execute(
            text(
                """
                INSERT INTO chunks
                    (id, user_id, embedding_model, doc_id, title, fingerprint,
                     chunk_index, ref, text, embedding)
                SELECT gen_random_uuid()::text, :dst_user, embedding_model,
                       :dst_doc, title, fingerprint, chunk_index, ref, text, embedding
                FROM chunks
                WHERE user_id = :src_user AND doc_id = :src_doc
                """
            ),
            {
                "dst_user": dst_user_id, "dst_doc": dst_doc_id,
                "src_user": src_user_id, "src_doc": src_doc_id,
            },
        )
        return int(res.rowcount or 0)
