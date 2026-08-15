"""Per-user persistent Chroma store (PRD T2, T6).

One collection per (user, embedding model) on disk. Chroma fixes a
collection's vector dimension on first write, so chunks from two
embedding models can never share a collection — otherwise a dimension
mismatch crashes ingest. We bake the model into the collection name, so
switching embedding models simply targets a fresh collection instead of
corrupting an existing one (PRD T3 / correctness fix).

Every chunk carries metadata for citations (doc_id, title, ref) and the
config fingerprint it was built under (F18), so chunks built under a
different chunking/embedding config are never returned.

A lightweight in-memory BM25 index (one per collection) backs hybrid
search: the vector top-k and the BM25 top-k are fused with reciprocal
rank fusion (RRF) so exact-match terms that pure-vector search misses
still surface (PRD §5 hybrid_search, real BM25 — not web search).
"""
from __future__ import annotations

import re
from collections import defaultdict

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

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


def _slug(user_id: str) -> str:
    # Chroma names: 3-512 chars from [a-zA-Z0-9._-], starting/ending alphanumeric.
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", user_id).strip("._-") or "user"
    return slug[:200]


def _model_slug(model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]", "-", model).strip("._-") or "emb"
    return slug[:200]


def collection_name(user_id: str, embedding_model: str) -> str:
    """Stable, dimension-isolated collection name for a (user, model) pair."""
    return f"user-{_slug(user_id)}__{_model_slug(embedding_model)}"


def collection_for(user_id: str, embedding_model: str):
    name = collection_name(user_id, embedding_model)
    return get_client().get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


# ---------- in-memory BM25 index (keyed by collection name) ----------

_BM25_DOCS: dict[str, dict[str, list[str]]] = defaultdict(dict)
_BM25_FLAT: dict[str, list[tuple[str, str, str, str, str]]] = {}
_BM25_OBJ: dict[str, BM25Okapi] = {}


def _rebuild_bm25(name: str) -> None:
    flat: list[tuple[str, str, str, str, str]] = []
    for doc_id, chunks in _BM25_DOCS[name].items():
        for i, t in enumerate(chunks):
            flat.append((f"{doc_id}:{i}", doc_id, t, _BM25_TITLE.get((name, doc_id), ""), _BM25_REF.get((name, doc_id, i), "")))
    _BM25_FLAT[name] = flat
    corpus = [t.split() for _, _, t, _, _ in flat]
    if corpus:
        _BM25_OBJ[name] = BM25Okapi(corpus)
    else:
        _BM25_OBJ.pop(name, None)


# title/ref lookups for BM25-only chunks (so we can reconstruct a full dict)
_BM25_TITLE: dict[tuple[str, str], str] = {}
_BM25_REF: dict[tuple[str, str, int], str] = {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


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
    col = collection_for(user_id, embedding_model)
    ids = [f"{doc_id}:{i}" for i in range(len(texts))]
    col.add(
        ids=ids,
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
    name = col.name
    _BM25_DOCS.setdefault(name, {})[doc_id] = list(texts)
    _BM25_TITLE[(name, doc_id)] = title
    for i, ref in enumerate(refs):
        _BM25_REF[(name, doc_id, i)] = ref
    _rebuild_bm25(name)


def _user_collection_prefix(user_id: str) -> str:
    return f"user-{_slug(user_id)}__"


def delete_document_chunks(
    user_id: str, doc_id: str, embedding_model: str | None = None
) -> None:
    client = get_client()
    if embedding_model:
        # Fast path when the indexing model is known.
        col = collection_for(user_id, embedding_model)
        col.delete(where={"doc_id": doc_id})
        _cleanup_bm25(col.name, doc_id)
        return
    # Without a model hint (e.g. a delete while the config model differs from
    # the one used to index), sweep every collection owned by this user so
    # stale-fingerprint chunks are still removed (handles embedding switches).
    prefix = _user_collection_prefix(user_id)
    for col in client.list_collections():
        if col.name.startswith(prefix):
            col.delete(where={"doc_id": doc_id})
            _cleanup_bm25(col.name, doc_id)


def _cleanup_bm25(name: str, doc_id: str) -> None:
    if name in _BM25_DOCS and doc_id in _BM25_DOCS[name]:
        del _BM25_DOCS[name][doc_id]
        _BM25_TITLE.pop((name, doc_id), None)
        _rebuild_bm25(name)


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

    When `bm25_index` is set, the vector top-k and a BM25 keyword top-k are
    fused with reciprocal rank fusion (RRF) and the combined list is
    returned (hybrid_search, PRD §5). `query_text` is required for BM25.
    """
    col = collection_for(user_id, embedding_model)
    if col.count() == 0:
        return []
    # Fetch a wider vector net when fusing, so BM25 can promote chunks the
    # vector ranker missed.
    v_n = max(n_results, 30) if bm25_index else n_results
    res = col.query(
        query_embeddings=[embedding],
        n_results=max(v_n, 1),
        where={"fingerprint": fingerprint},
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    docs = res["documents"][0] or []
    metas = res["metadatas"][0] or []
    dists = res["distances"][0] or []
    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append(
            {
                "chunk_id": f"{meta.get('doc_id')}:{meta.get('chunk_index')}",
                "text": doc,
                "similarity": 1.0 - dist,
                "doc_id": meta.get("doc_id"),
                "title": meta.get("title"),
                "ref": meta.get("ref"),
            }
        )

    if bm25_index and query_text and col.name in _BM25_OBJ:
        return _bm25_fuse(col.name, chunks, query_text, n_results)
    return chunks


def _bm25_fuse(
    name: str, vector_chunks: list[dict], query_text: str, n_results: int
) -> list[dict]:
    """Reciprocal rank fusion of vector + BM25 result lists (k=60)."""
    K = 60
    flat = _BM25_FLAT[name]
    bm25 = _BM25_OBJ[name]
    scores = bm25.get_scores(_tokenize(query_text))

    # BM25 ranking (only chunks with a positive score)
    bm25_by_id: dict[str, float] = {}
    for rank, i in enumerate(
        sorted(range(len(flat)), key=lambda j: scores[j], reverse=True), start=1
    ):
        if scores[i] > 0:
            bm25_by_id[flat[i][0]] = rank

    # Vector ranking by cosine similarity
    vec_sorted = sorted(vector_chunks, key=lambda c: c["similarity"], reverse=True)

    fused: dict[str, dict] = {}
    for rank, c in enumerate(vec_sorted, start=1):
        cid = c["chunk_id"]
        fused.setdefault(cid, {"chunk": c, "score": 0.0})
        fused[cid]["score"] += 1.0 / (K + rank)
    for cid, rank in bm25_by_id.items():
        fused.setdefault(
            cid,
            {
                "chunk": {
                    "chunk_id": cid,
                    "text": _flat_lookup(flat, cid, 2),
                    "similarity": None,
                    "doc_id": _flat_lookup(flat, cid, 1),
                    "title": _flat_lookup(flat, cid, 3),
                    "ref": _flat_lookup(flat, cid, 4),
                },
                "score": 0.0,
            },
        )
        fused[cid]["score"] += 1.0 / (K + rank)

    ranked = sorted(fused.values(), key=lambda e: e["score"], reverse=True)
    return [e["chunk"] for e in ranked[: max(n_results, 1)]]


def _flat_lookup(flat, cid, idx):
    for row in flat:
        if row[0] == cid:
            return row[idx]
    return None
