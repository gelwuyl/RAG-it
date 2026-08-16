"""Reassign one user's whole workspace to another account.

Written for the `local` -> Google-account handover: per-user isolation was always
enforced, so signing in with Google lands you in a brand-new empty space while
everything you uploaded stays attached to the built-in `local` user.

Moves documents, folder sources, conversations (messages follow their
conversation) and the vector chunks, so nothing has to be re-embedded.

The two vector backends scope users differently and both are handled:
  * neon   - `chunks.user_id` column -> UPDATE
  * chroma - collection NAME encodes the user (`user-<slug>__<model>`) -> rename

Dry-run by default. Nothing is written until you pass --commit.

    .venv/Scripts/python -m scripts.migrate_user_data --list
    .venv/Scripts/python -m scripts.migrate_user_data --to you@gmail.com
    .venv/Scripts/python -m scripts.migrate_user_data --to you@gmail.com --commit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragchat.config import settings  # noqa: E402
from ragchat.db import (  # noqa: E402
    Conversation,
    Document,
    FolderSource,
    Message,
    SessionLocal,
    User,
)


def _resolve(db, ident: str) -> User | None:
    """Find a user by id, email, or name — whichever the caller typed."""
    for column in (User.id, User.email, User.name, User.sub):
        user = db.query(User).filter(column == ident).first()
        if user:
            return user
    return None


def _describe(db, user: User) -> dict:
    convs = db.query(Conversation).filter(Conversation.user_id == user.id).all()
    return {
        "documents": db.query(Document).filter(Document.user_id == user.id).count(),
        "folders": db.query(FolderSource).filter(FolderSource.user_id == user.id).count(),
        "conversations": len(convs),
        "messages": sum(
            db.query(Message).filter(Message.conversation_id == c.id).count()
            for c in convs
        ),
    }


def _move_chunks_neon(old_id: str, new_id: str, commit: bool) -> int:
    from sqlalchemy import text as _t

    from ragchat.store_neon import _get_engine

    eng = _get_engine()
    with eng.begin() as conn:
        n = conn.execute(
            _t("SELECT COUNT(*) FROM chunks WHERE user_id = :u"), {"u": old_id}
        ).scalar_one()
        if commit and n:
            conn.execute(
                _t("UPDATE chunks SET user_id = :new WHERE user_id = :old"),
                {"new": new_id, "old": old_id},
            )
    return int(n)


def _move_chunks_chroma(old_id: str, new_id: str, commit: bool) -> int:
    """Rename each of the user's collections rather than re-embedding.

    The embeddings are identical — only the owner changes — so a rename keeps
    the vectors byte-for-byte and costs no API calls.
    """
    from ragchat.store import _slug, get_client

    client = get_client()
    old_prefix, new_prefix = f"user-{_slug(old_id)}__", f"user-{_slug(new_id)}__"
    moved = 0
    for col in client.list_collections():
        name = col.name if hasattr(col, "name") else str(col)
        if not name.startswith(old_prefix):
            continue
        moved += 1
        if commit:
            target = new_prefix + name[len(old_prefix):]
            client.get_collection(name).modify(name=target)
    return moved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="src", default="local",
                    help="source account (id/email/name/username). Default: local")
    ap.add_argument("--to", dest="dst", help="destination account (id/email/name)")
    ap.add_argument("--list", action="store_true", help="list accounts and exit")
    ap.add_argument("--commit", action="store_true",
                    help="actually write. Without this it is a dry run.")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        if args.list or not args.dst:
            print(f"{'provider':<10} {'name':<28} {'email':<30} id")
            for u in db.query(User).all():
                counts = _describe(db, u)
                print(f"{u.provider:<10} {(u.name or ''):<28} {(u.email or ''):<30} {u.id}")
                print(f"{'':<10} └─ {counts}")
            if not args.dst:
                print("\nNothing moved. Re-run with --to <account> to migrate.")
            return 0

        src = _resolve(db, args.src)
        dst = _resolve(db, args.dst)
        if not src:
            print(f"ERROR: source account {args.src!r} not found.")
            return 1
        if not dst:
            print(f"ERROR: destination account {args.dst!r} not found.")
            print("Sign in with Google once first — the account row is created on "
                  "first sign-in, and there is nothing to migrate into until then.")
            return 1
        if src.id == dst.id:
            print("ERROR: source and destination are the same account.")
            return 1

        before_src, before_dst = _describe(db, src), _describe(db, dst)
        print(f"FROM  {src.provider}/{src.name} ({src.id})\n      {before_src}")
        print(f"TO    {dst.provider}/{dst.name} ({dst.id})\n      {before_dst}")

        backend = (settings.vector_backend or "chroma").lower()
        mover = _move_chunks_neon if backend == "neon" else _move_chunks_chroma
        unit = "chunks" if backend == "neon" else "collections"

        n_docs = db.query(Document).filter(Document.user_id == src.id).update(
            {Document.user_id: dst.id}, synchronize_session=False
        )
        n_folders = db.query(FolderSource).filter(FolderSource.user_id == src.id).update(
            {FolderSource.user_id: dst.id}, synchronize_session=False
        )
        n_convs = db.query(Conversation).filter(Conversation.user_id == src.id).update(
            {Conversation.user_id: dst.id}, synchronize_session=False
        )
        n_vec = mover(src.id, dst.id, args.commit)

        print(f"\nvector backend: {backend}")
        print(f"  documents     {n_docs}")
        print(f"  folders       {n_folders}")
        print(f"  conversations {n_convs}")
        print(f"  {unit:<13} {n_vec}")

        if args.commit:
            db.commit()
            print("\nCOMMITTED. Sign in as the destination account to see everything.")
        else:
            db.rollback()
            print("\nDRY RUN — nothing written. Re-run with --commit to apply.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
