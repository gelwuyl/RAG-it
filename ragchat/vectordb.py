"""Vector-store dispatch.

Exposes ``add_chunks``, ``query_chunks``, ``delete_document_chunks`` and
``collection_name`` by reading ``settings.vector_backend`` and lazily importing
the right implementation. This keeps the two backends mutually exclusive at
import time:

* ``chroma`` (default, local dev) -> :mod:`ragchat.store`
* ``neon``  (Postgres + pgvector, Vercel) -> :mod:`ragchat.store_neon`

Because the implementation module is imported *inside each function*, importing
this module never pulls in ``chromadb`` (when backend is neon) nor
``psycopg2``/``pgvector`` (when backend is chroma).
"""
from __future__ import annotations

from .config import settings


def _backend() -> str:
    return (getattr(settings, "vector_backend", None) or "chroma").lower()


def _impl():
    if _backend() == "neon":
        from . import store_neon as impl
    else:
        from . import store as impl
    return impl


def collection_name(user_id: str, embedding_model: str) -> str:
    return _impl().collection_name(user_id, embedding_model)


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
    return _impl().add_chunks(
        user_id,
        doc_id,
        title,
        fingerprint,
        texts,
        embeddings,
        refs,
        embedding_model,
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
    return _impl().query_chunks(
        user_id,
        embedding,
        fingerprint,
        n_results,
        embedding_model,
        bm25_index,
        query_text,
    )


def delete_document_chunks(
    user_id: str, doc_id: str, embedding_model: str | None = None
) -> None:
    return _impl().delete_document_chunks(user_id, doc_id, embedding_model)


def prune_chunks(
    user_id: str,
    valid_doc_ids: set[str],
    stale_fingerprints: set[str] | None = None,
) -> int:
    return _impl().prune_chunks(user_id, valid_doc_ids, stale_fingerprints)
