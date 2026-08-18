"""Precomputed embeddings for the demo corpus.

The first visitor to arrive on a fresh database used to pay for embedding the
whole demo corpus inside their own guest-login request, and that request has 60
seconds (vercel.json maxDuration). It did not fit: the very first
POST /api/auth/guest-login returned 504 FUNCTION_INVOCATION_TIMEOUT after 63s,
so the landing page's primary call to action was broken for whoever arrived
first. It self-healed on the second and third visit (41s, then ~10s once the
template existed), which is exactly what made it easy to miss — by the time
anyone looked, it worked.

Re-embedding is triggered by more than a brand-new database: the config
fingerprint covers chunk_size, chunk_overlap, splitter and embedding_model, so
changing any of them invalidates the template and puts the 504 straight back.

So the vectors ship WITH the repo. Seeding becomes a pure database insert and no
visitor ever waits on an embedding call. The corpus is two small files — 2
chunks, 1536 floats — so this costs a few KB, not megabytes.

Only the embeddings are stored, never the chunk text. Text is re-derived by
splitting the corpus file, which is deterministic for a given fingerprint, so
the two cannot drift apart: if the split changed, the fingerprint changed, and
the guard below rejects the file.

Regenerate with:  python -m ragchat.demo_vectors
"""
from __future__ import annotations

import json
from pathlib import Path

VECTOR_FILE = Path(__file__).resolve().parent / "demo_vectors.json"


def _corpus_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "eval" / "corpus"


def load(cfg) -> dict[str, list[list[float]]] | None:
    """Precomputed embeddings per filename, or None if they do not apply.

    None means "embed it live" — a missing file, a different embedding model, or
    a fingerprint that no longer matches. Returning None rather than raising is
    deliberate: stale vectors must degrade to the slow path, never to a wrong
    one. Chunks stored under a mismatched fingerprint are invisible to
    query_chunks anyway, so seeding from them would produce exactly the
    chunkless document the guest seeding fixes exist to prevent.
    """
    if not VECTOR_FILE.exists():
        return None
    try:
        blob = json.loads(VECTOR_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if blob.get("embedding_model") != cfg.embedding_model:
        return None
    if blob.get("fingerprint") != cfg.fingerprint():
        return None
    files = blob.get("files")
    return files if isinstance(files, dict) and files else None


def seed_document(user_id: str, doc_id: str, title: str, text: str, cfg,
                  embeddings: list[list[float]]) -> int:
    """Store one demo document from precomputed vectors. Returns the chunk count.

    Raises ValueError when the vector count does not match what the splitter
    produces, so a stale file fails loudly here instead of silently seeding a
    document with fewer chunks than it claims.
    """
    from .chunking import refine_refs, split_document
    from .vectordb import add_chunks

    chunks = refine_refs(split_document(text, title, cfg), text)
    if not chunks:
        return 0
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"{title}: {len(chunks)} chunks but {len(embeddings)} precomputed "
            "vectors — regenerate with python -m ragchat.demo_vectors"
        )
    add_chunks(
        user_id, doc_id, title, cfg.fingerprint(),
        [c.text for c in chunks], embeddings, [c.ref for c in chunks],
        embedding_model=cfg.embedding_model,
    )
    return len(chunks)


def build() -> Path:
    """Embed the demo corpus and write demo_vectors.json. Run this by hand."""
    from .config import load_config
    from .chunking import refine_refs, split_document
    from .embeddings import ProxyEmbeddings
    from .guests import DEMO_CORPUS_FILES

    cfg = load_config()
    emb = ProxyEmbeddings(cfg.embedding_model, provider=cfg.embedding_provider)
    files: dict[str, list[list[float]]] = {}
    for name in DEMO_CORPUS_FILES:
        path = _corpus_dir() / name
        if not path.exists():
            raise SystemExit(f"missing corpus file: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = refine_refs(split_document(text, name, cfg), text)
        files[name] = emb.embed_documents([c.text for c in chunks])
        print(f"  {name}: {len(chunks)} chunk(s) embedded")

    blob = {
        "embedding_model": cfg.embedding_model,
        "embedding_provider": cfg.embedding_provider,
        "fingerprint": cfg.fingerprint(),
        "dim": len(next(iter(files.values()))[0]) if files else 0,
        "files": files,
    }
    VECTOR_FILE.write_text(json.dumps(blob), encoding="utf-8")
    print(f"wrote {VECTOR_FILE} ({VECTOR_FILE.stat().st_size / 1024:.1f} KB)")
    print(f"  model={blob['embedding_model']} fingerprint={blob['fingerprint']} dim={blob['dim']}")
    return VECTOR_FILE


if __name__ == "__main__":
    build()
