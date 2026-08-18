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
    start_index: int = 0,
) -> None:
    """Add chunks for a document.

    ``start_index`` is the position of ``texts[0]`` within the whole document.
    Sliced ingest calls this once per slice, and without an offset every slice
    would write ids 0..n-1 and overwrite the previous one, leaving a document
    with only its last slice indexed.
    """
    col = collection_for(user_id, embedding_model)
    ids = [f"{doc_id}:{start_index + i}" for i in range(len(texts))]
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
                "chunk_index": start_index + i,
            }
            for i, ref in enumerate(refs)
        ],
    )
    name = col.name
    # BM25 is an in-memory per-instance index keyed by doc; EXTEND rather than
    # replace, or a sliced ingest would leave only the final slice searchable
    # by keyword even though every chunk is in the vector store.
    bucket = _BM25_DOCS.setdefault(name, {})
    if start_index == 0:
        bucket[doc_id] = list(texts)
    else:
        bucket.setdefault(doc_id, []).extend(texts)
    _BM25_TITLE[(name, doc_id)] = title
    for i, ref in enumerate(refs):
        _BM25_REF[(name, doc_id, start_index + i)] = ref
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
    # Slice here too. `v_n` deliberately over-fetches (30) so BM25 has something
    # to promote from, and _bm25_fuse cuts back to n_results — but this return
    # is the path taken when fusion CANNOT run, and it used to hand back all 30.
    # candidate_k is a tunable setting the benchmark reports; a code path that
    # quietly ignores it makes every measurement taken on it wrong.
    #
    # That path is reached more often than it looks: _BM25_OBJ is an in-memory,
    # per-process index built during ingest, so a freshly started process has
    # none until something re-ingests. Chroma is local dev only — the Neon
    # backend uses Postgres full-text search, which is stateless and always
    # available — but it means local retrieval can differ from deployed
    # retrieval until the process has ingested something.
    return chunks[: max(n_results, 1)]


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


def prune_chunks(
    user_id: str,
    valid_doc_ids: set[str],
    stale_fingerprints: set[str] | None = None,
) -> int:
    """Delete chunks with no backing Document row, and optionally stale ones.

    The Neon twin of this function existed; this one did not, so "Prune ghosts"
    raised AttributeError inside vectordb._impl() and 500'd on the Chroma
    backend. The two stores must expose the same surface — vectordb.py dispatches
    to whichever is configured and cannot paper over a missing name.

    Returns the number of chunks removed.
    """
    client = get_client()
    prefix = _user_collection_prefix(user_id)
    removed = 0
    for col in client.list_collections():
        name = col.name if hasattr(col, "name") else str(col)
        if not name.startswith(prefix):
            continue
        collection = client.get_collection(name)
        got = collection.get(include=["metadatas"])
        ids = got.get("ids") or []
        metas = got.get("metadatas") or []
        doomed = []
        for cid, meta in zip(ids, metas):
            meta = meta or {}
            # Orphan check is skipped entirely on an empty valid set. Mirrors the
            # Neon guard and the no-op documented in CLAUDE.md: a user who has
            # not added any documents yet must not have their chunks wiped by a
            # prune that legitimately found nothing to compare against.
            if valid_doc_ids and meta.get("doc_id") not in valid_doc_ids:
                doomed.append(cid)
            elif stale_fingerprints and meta.get("fingerprint") in stale_fingerprints:
                doomed.append(cid)
        if doomed:
            collection.delete(ids=doomed)
            removed += len(doomed)
    if removed:
        # BM25 caches are keyed by collection name and now describe chunks that
        # no longer exist; drop them so they rebuild lazily from what is left.
        for cache in (_BM25_DOCS, _BM25_FLAT, _BM25_OBJ):
            cache.clear()
    return removed


def reassign_user_chunks(old_user_id: str, new_user_id: str) -> int:
    """Move a user's vectors by RENAMING their collections.

    Chroma encodes the owner in the collection name rather than a column, so
    there is no per-row user field to update. Renaming keeps the embeddings
    byte-for-byte and costs no API calls (guest -> signed-in promotion).
    """
    client = get_client()
    old_prefix = _user_collection_prefix(old_user_id)
    new_prefix = _user_collection_prefix(new_user_id)
    moved = 0
    for col in client.list_collections():
        name = col.name if hasattr(col, "name") else str(col)
        if not name.startswith(old_prefix):
            continue
        client.get_collection(name).modify(name=new_prefix + name[len(old_prefix):])
        moved += 1
    # In-memory BM25 caches are keyed by collection name and are per-instance
    # scratch; drop them so the renamed collections rebuild lazily.
    for cache in (_BM25_DOCS, _BM25_FLAT, _BM25_OBJ):
        cache.clear()
    return moved


def delete_users_chunks(user_ids: list[str]) -> int:
    """Drop every collection belonging to any of these users. Returns the count.

    The Chroma twin of the Neon bulk delete. Chroma encodes the owner in the
    collection NAME rather than a column, so "delete this user's vectors" is
    dropping their collections — which also disposes of the HNSW index, where
    a per-row delete would leave it behind.
    """
    prefixes = tuple(_user_collection_prefix(u) for u in user_ids if u)
    if not prefixes:
        return 0
    client = get_client()
    dropped = 0
    for col in client.list_collections():
        name = col.name if hasattr(col, "name") else str(col)
        if not name.startswith(prefixes):
            continue
        client.delete_collection(name)
        dropped += 1
    if dropped:
        # Per-instance scratch keyed by collection name. Rebuilt lazily, so
        # clearing is always safe and is cheaper than pruning by key.
        for cache in (_BM25_DOCS, _BM25_FLAT, _BM25_OBJ):
            cache.clear()
        _BM25_TITLE.clear()
    return dropped


def copy_user_chunks(
    src_user_id: str, src_doc_id: str, dst_user_id: str, dst_doc_id: str
) -> int:
    """Copy one document's chunks (with embeddings) into another user's space.

    Used to hand each new guest a private copy of the pre-embedded demo corpus
    without spending embedding quota on every anonymous page load.
    """
    client = get_client()
    src_prefix = _user_collection_prefix(src_user_id)
    copied = 0
    for col in client.list_collections():
        name = col.name if hasattr(col, "name") else str(col)
        if not name.startswith(src_prefix):
            continue
        model_slug = name[len(src_prefix):]
        src_col = client.get_collection(name)
        got = src_col.get(
            where={"doc_id": src_doc_id},
            include=["documents", "metadatas", "embeddings"],
        )
        docs = got.get("documents") or []
        if not docs:
            continue
        metas = got.get("metadatas") or []
        embs = got.get("embeddings")
        if embs is None or len(embs) == 0:
            continue  # nothing to copy without vectors; re-embedding is the caller's job
        dst_col = get_client().get_or_create_collection(
            name=f"{_user_collection_prefix(dst_user_id)}{model_slug}",
            metadata={"hnsw:space": "cosine"},
        )
        new_metas = []
        for m in metas:
            m = dict(m or {})
            m["doc_id"] = dst_doc_id  # retarget, keep title/ref/fingerprint/index
            new_metas.append(m)
        dst_col.add(
            ids=[f"{dst_doc_id}:{i}" for i in range(len(docs))],
            documents=list(docs),
            embeddings=[list(e) for e in embs],
            metadatas=new_metas,
        )
        copied += len(docs)
    return copied
