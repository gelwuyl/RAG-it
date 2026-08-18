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
    start_index: int = 0,
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
        start_index,
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


def reassign_user_chunks(old_user_id: str, new_user_id: str) -> int:
    """Move every chunk owned by `old_user_id` to `new_user_id`.

    Used when a guest signs in: their vectors follow them into the permanent
    account instead of being re-embedded, so promotion costs no API calls.
    """
    return _impl().reassign_user_chunks(old_user_id, new_user_id)


def delete_users_chunks(user_ids: list[str]) -> int:
    """Delete every chunk owned by any of these users, in one operation.

    The guest sweeper's hot path. Per-document deletes made clearing one
    workspace several round trips and a sweep of twenty into a serverless
    timeout risk; this is a single statement on Neon.
    """
    return _impl().delete_users_chunks(user_ids)


def copy_user_chunks(
    src_user_id: str, src_doc_id: str, dst_user_id: str, dst_doc_id: str
) -> int:
    """Duplicate one document's chunks to another user, vectors and all.

    Lets the demo corpus be embedded ONCE under a template account and handed to
    each new guest as a pure database copy — otherwise every anonymous page load
    would spend embedding quota and add latency to first paint.
    """
    return _impl().copy_user_chunks(src_user_id, src_doc_id, dst_user_id, dst_doc_id)
