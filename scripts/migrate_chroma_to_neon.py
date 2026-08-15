"""Migrate an existing local Chroma store into Neon (Postgres + pgvector).

USAGE
-----
    # Migrate every collection Chroma currently holds (one pass per user/model):
    .venv/Scripts/python scripts/migrate_chroma_to_neon.py --all

    # Migrate a single (user, embedding_model) collection:
    .venv/Scripts/python scripts/migrate_chroma_to_neon.py --user <USER_ID> --model <EMBEDDING_MODEL>

    # Dry run (read Chroma, print chunk counts, do NOT write to Neon):
    .venv/Scripts/python scripts/migrate_chroma_to_neon.py --all --dry-run

ENVIRONMENT
-----------
* VECTOR_BACKEND is irrelevant here — we read Chroma directly and write via
  the pgvector store, so both backends are imported explicitly.
* PG_DATABASE_URL (or DATABASE_URL) must be set and point at the Neon
  database; it is used by ragchat.store_neon.
* GEMINI_API_KEY is NOT needed (we copy the embeddings Chroma already has).

NOTES
-----
* Each Chroma collection is named ``user-<slug>__<modelslug>``. For ``--all``
  we use the slug as the Neon ``user_id`` (it is lossless for per-user
  isolation — the Neon store keys rows by that string). For a targeted
  ``--user`` run we use the exact ``USER_ID`` you pass.
* Chunks are upserted by stable id (``<user_id>::<doc_id>:<i>``), so re-running
  the migration is safe and idempotent.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragchat import store as chroma_store  # the Chroma backend (read side)
from ragchat import store_neon  # the pgvector backend (write side)


def _migrate_collection(user_id: str, embedding_model: str, dry_run: bool) -> int:
    """Copy one (user, model) Chroma collection into Neon. Returns chunk count."""
    col = chroma_store.collection_for(user_id, embedding_model)
    data = col.get(include=["documents", "embeddings", "metadatas"])
    ids = data["ids"] or []
    docs = data["documents"] or []
    emb = data["embeddings"] or []
    metas = data["metadatas"] or []

    if not ids:
        print("    (no chunks)")
        return 0

    # Group rows by doc_id, preserving chunk order.
    by_doc: dict[str, list[int]] = {}
    for i, meta in enumerate(metas):
        meta = meta or {}
        doc_id = meta.get("doc_id") or "unknown"
        by_doc.setdefault(doc_id, []).append(i)

    total = 0
    for doc_id, idxs in by_doc.items():
        idxs.sort(key=lambda j: (metas[j] or {}).get("chunk_index", j))
        texts = [docs[j] for j in idxs]
        embeddings = [list(emb[j]) for j in idxs]
        refs = [(metas[j] or {}).get("ref", "") for j in idxs]
        title = (metas[idxs[0]] or {}).get("title", "")
        fingerprint = (metas[idxs[0]] or {}).get("fingerprint", "")
        if not dry_run:
            store_neon.add_chunks(
                user_id,
                doc_id,
                title,
                fingerprint,
                texts,
                embeddings,
                refs,
                embedding_model,
            )
        print(f"    doc {doc_id!r}: {len(idxs)} chunks")
        total += len(idxs)
    return total


def migrate_all(dry_run: bool) -> None:
    client = chroma_store.get_client()
    collections = client.list_collections()
    if not collections:
        print("No Chroma collections found.")
        return
    for col in collections:
        name = col.name
        if not name.startswith("user-") or "__" not in name:
            continue
        user_slug, model_slug = name[len("user-"):].split("__", 1)
        # slug -> slug is idempotent via collection_name(), so the same
        # collection is read back correctly.
        print(f"Collection {name}:")
        try:
            n = _migrate_collection(user_slug, model_slug, dry_run)
            print(f"  -> migrated {n} chunks" + (" (dry run)" if dry_run else ""))
        except Exception as exc:  # keep going through the rest
            print(f"  -> ERROR: {exc}")


def migrate_one(user_id: str, embedding_model: str, dry_run: bool) -> None:
    print(f"Collection user-{user_id}__{embedding_model}:")
    n = _migrate_collection(user_id, embedding_model, dry_run)
    print(f"  -> migrated {n} chunks" + (" (dry run)" if dry_run else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="Migrate every Chroma collection.")
    ap.add_argument("--user", help="Target user id (requires --model).")
    ap.add_argument("--model", help="Embedding model of the collection to migrate.")
    ap.add_argument("--dry-run", action="store_true", help="Read Chroma, print counts, write nothing.")
    args = ap.parse_args()

    if args.all:
        migrate_all(args.dry_run)
    elif args.user and args.model:
        migrate_one(args.user, args.model, args.dry_run)
    else:
        ap.error("pass either --all, or both --user and --model")


if __name__ == "__main__":
    main()
