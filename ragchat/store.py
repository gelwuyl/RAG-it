"""Per-user persistent Chroma store (PRD T2, T6).

One collection per user on disk; every chunk carries metadata for citations
(doc_id, title, ref) and the config fingerprint it was built under (F18), so
chunks built under a different chunking/embedding config are never returned.
"""
from __future__ import annotations

import re

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import CHROMA_DIR

_client: chromadb.ClientAPI | None = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
    return _client


def collection_for(user_id: str):
    # Chroma names: 3-512 chars from [a-zA-Z0-9._-], starting/ending alphanumeric.
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", user_id).strip("._-") or "user"
    name = f"user-{slug}"
    return get_client().get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def add_chunks(
    user_id: str,
    doc_id: str,
    title: str,
    fingerprint: str,
    texts: list[str],
    embeddings: list[list[float]],
    refs: list[str],
) -> None:
    col = collection_for(user_id)
    col.add(
        ids=[f"{doc_id}:{i}" for i in range(len(texts))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "doc_id": doc_id,
                "title": title,
                "ref": ref,
                "fingerprint": fingerprint,
                "chunk_index": i,
            }
            for i, ref in enumerate(refs)
        ],
    )


def delete_document_chunks(user_id: str, doc_id: str) -> None:
    col = collection_for(user_id)
    col.delete(where={"doc_id": doc_id})


def query_chunks(
    user_id: str,
    embedding: list[float],
    fingerprint: str,
    n_results: int,
) -> list[dict]:
    """Return the user's chunks built under the current config fingerprint."""
    col = collection_for(user_id)
    if col.count() == 0:
        return []
    res = col.query(
        query_embeddings=[embedding],
        n_results=max(n_results, 1),
        where={"fingerprint": fingerprint},
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    docs = res["documents"][0] or []
    metas = res["metadatas"][0] or []
    dists = res["distances"][0] or []
    for doc, meta, dist in zip(docs, metas, dists):
        # cosine distance -> similarity
        chunks.append(
            {
                "text": doc,
                "similarity": 1.0 - dist,
                "doc_id": meta.get("doc_id"),
                "title": meta.get("title"),
                "ref": meta.get("ref"),
            }
        )
    return chunks
