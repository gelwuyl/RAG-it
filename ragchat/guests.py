"""Ephemeral guest accounts.

A visitor who has not signed in gets their OWN throwaway account rather than
sharing one. That distinction is the whole point: the app scopes every query by
user id, so a single shared guest account would let strangers read and delete
each other's uploads — the exact mixing per-user isolation exists to prevent.

Guests are capped (they spend real embedding quota) and reaped after a period of
inactivity. Signing in with Google promotes the guest's work into the permanent
account instead of discarding it.

Reaping is opportunistic rather than scheduled: Vercel gives no cron on the
Hobby plan, and a background thread would be frozen the moment the response is
sent (see CLAUDE.md). Doing it on guest creation costs one indexed DELETE on an
event that is already infrequent.
"""
from __future__ import annotations

import time

from sqlalchemy.orm import Session

from .db import Conversation, Document, FolderSource, Message, User, new_id, now

GUEST_PROVIDER = "guest"

# Idle window before a guest's workspace is destroyed. Deliberately measured
# from last activity, not creation, so someone mid-session is never wiped.
GUEST_IDLE_TTL_SECONDS = 2 * 60 * 60  # 2 hours

# What a guest may upload before being asked to sign in.
#
# This is now the ONLY thing bounding what an anonymous visitor can spend.
# Embeddings run on a paid provider for guests and accounts alike: per-tier
# models were considered and rejected, because vectors from different models
# are not comparable, so a promoted guest workspace would go silent at exactly
# the moment the app promises "your work comes with you".
#
# Guests stay cheap by construction even so. The demo corpus is vector-COPIED
# rather than embedded, so a visitor who only reads costs nothing however many
# arrive, and asking a question costs one query embedding. Only uploads spend
# real money — and 2 MB is roughly a 100-page document, ample to try the app
# with and a 2.5x tighter worst case than the 5 MB it replaces.
GUEST_MAX_DOCUMENTS = 3
GUEST_MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB total across all their uploads

# Writing last_seen_at on literally every request would add a DB write to every
# call. Only refresh it when it is already this stale — reaping resolution does
# not need to be finer than this.
_LAST_SEEN_REFRESH_SECONDS = 5 * 60


def is_guest(user: User | None) -> bool:
    return bool(user is not None and user.provider == GUEST_PROVIDER)


def touch(db: Session, user: User) -> None:
    """Record activity, cheaply. Keeps a guest alive while they are using the app."""
    last = user.last_seen_at or 0.0
    if time.time() - last < _LAST_SEEN_REFRESH_SECONDS:
        return
    user.last_seen_at = now()
    db.commit()


def purge_user_data(db: Session, user: User, *, drop_user: bool) -> dict:
    """Delete everything belonging to `user`: chunks, documents, folders, chats.

    Shared by guest reaping and account deletion so the two can never drift —
    a forgotten table in one path would silently leave orphaned rows behind.
    Vector chunks are removed per document id: prune_chunks() deliberately
    no-ops on an empty valid-doc set, so it cannot be used to clear a user.
    """
    from .vectordb import delete_document_chunks

    docs = db.query(Document).filter(Document.user_id == user.id).all()
    for doc in docs:
        try:
            delete_document_chunks(user.id, doc.id)
        except Exception:
            # A vector-store hiccup must not strand the relational rows; the
            # orphan-prune endpoint exists to mop up anything left behind.
            pass

    convs = db.query(Conversation).filter(Conversation.user_id == user.id).all()
    n_msgs = 0
    for conv in convs:
        n_msgs += (
            db.query(Message).filter(Message.conversation_id == conv.id).delete()
        )
        db.delete(conv)

    n_folders = db.query(FolderSource).filter(FolderSource.user_id == user.id).delete()
    for doc in docs:
        db.delete(doc)

    summary = {
        "documents": len(docs),
        "folders": int(n_folders),
        "conversations": len(convs),
        "messages": int(n_msgs),
    }
    if drop_user:
        db.delete(user)
    db.commit()
    return summary


def reap_stale_guests(db: Session, *, limit: int = 20) -> int:
    """Destroy guest workspaces idle longer than the TTL.

    `limit` bounds the work so a backlog cannot blow the serverless time limit;
    the next call simply picks up where this one stopped.
    """
    cutoff = time.time() - GUEST_IDLE_TTL_SECONDS
    stale = (
        db.query(User)
        .filter(
            User.provider == GUEST_PROVIDER,
            # created_at covers rows written before last_seen_at existed.
            User.last_seen_at.isnot(None),
            User.last_seen_at < cutoff,
        )
        .limit(limit)
        .all()
    )
    for guest in stale:
        purge_user_data(db, guest, drop_user=True)
    return len(stale)


def create_guest(db: Session) -> User:
    """Provision a fresh, private guest workspace."""
    reap_stale_guests(db)
    guest = User(
        provider=GUEST_PROVIDER,
        sub=f"guest-{new_id()}",
        name="Guest",
        last_seen_at=now(),
    )
    db.add(guest)
    db.commit()
    db.refresh(guest)
    return guest


def promote_guest(db: Session, guest: User, target: User) -> dict:
    """Hand a guest's workspace over to the account they just signed into.

    Rows are re-pointed rather than copied, and the vector chunks move with
    them, so nothing is re-embedded. The guest row itself is dropped afterwards
    — it exists only to own data, and that data now belongs to `target`.
    """
    from .vectordb import reassign_user_chunks

    n_docs = (
        db.query(Document)
        .filter(Document.user_id == guest.id)
        .update({Document.user_id: target.id}, synchronize_session=False)
    )
    n_folders = (
        db.query(FolderSource)
        .filter(FolderSource.user_id == guest.id)
        .update({FolderSource.user_id: target.id}, synchronize_session=False)
    )
    n_convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == guest.id)
        .update({Conversation.user_id: target.id}, synchronize_session=False)
    )
    reassign_user_chunks(guest.id, target.id)
    db.delete(guest)
    db.commit()
    return {
        "documents": int(n_docs),
        "folders": int(n_folders),
        "conversations": int(n_convs),
    }


# Files served to every guest so the app is not empty on arrival. Deliberately
# ONLY the synthetic fixtures: the rest of eval/corpus is real business content
# and must never be exposed to anonymous visitors.
DEMO_CORPUS_FILES = ("helios_energy_handbook.md", "meridian_coffee_ops.md")

# The account that owns the canonical embedded copy of the demo corpus. It is
# never signed into; it exists so the demo can be embedded ONCE and then copied.
DEMO_TEMPLATE_SUB = "__demo_template__"


def _demo_template(db: Session) -> User | None:
    return (
        db.query(User)
        .filter(User.provider == GUEST_PROVIDER, User.sub == DEMO_TEMPLATE_SUB)
        .first()
    )


def ensure_demo_template(db: Session, cfg) -> User:
    """Embed the demo corpus once, under a template account.

    Embedding it per visitor would spend real API quota on every anonymous page
    load and add latency to first paint. Doing it once and copying the vectors
    makes each new guest a pure database operation.

    Re-embeds when the pipeline fingerprint changes, since chunks are only
    retrievable under the fingerprint they were written with.
    """
    from pathlib import Path

    from .pipeline import ingest_document_text

    template = _demo_template(db)
    if template is None:
        template = User(provider=GUEST_PROVIDER, sub=DEMO_TEMPLATE_SUB,
                        name="Demo corpus", last_seen_at=now())
        db.add(template)
        db.commit()
        db.refresh(template)

    fp = cfg.fingerprint()
    corpus_dir = Path(__file__).resolve().parent.parent / "eval" / "corpus"
    for filename in DEMO_CORPUS_FILES:
        path = corpus_dir / filename
        if not path.exists():
            continue
        doc = (
            db.query(Document)
            .filter(Document.user_id == template.id, Document.title == filename)
            .first()
        )
        if doc is not None and doc.config_fingerprint == fp and doc.status == "ready":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if doc is None:
            doc = Document(user_id=template.id, source_type="upload", title=filename,
                           size_bytes=len(text.encode("utf-8")))
            db.add(doc)
            db.commit()
        n = ingest_document_text(template.id, doc.id, doc.title, text, cfg)
        doc.status = "ready"
        doc.n_chunks = n
        doc.config_fingerprint = fp
        db.commit()
    return template


def seed_demo_corpus(db: Session, guest: User) -> int:
    """Give a new guest their own copy of the demo corpus — no embedding calls."""
    from .config import load_config
    from .vectordb import copy_user_chunks

    cfg = load_config()
    template = ensure_demo_template(db, cfg)
    copied = 0
    for src in db.query(Document).filter(Document.user_id == template.id).all():
        clone = Document(
            user_id=guest.id,
            source_type=src.source_type,
            title=src.title,
            content_hash=src.content_hash,
            config_fingerprint=src.config_fingerprint,
            status=src.status,
            n_chunks=src.n_chunks,
            size_bytes=src.size_bytes,
            is_demo=True,
        )
        db.add(clone)
        db.commit()
        copy_user_chunks(template.id, src.id, guest.id, clone.id)
        copied += 1
    return copied


def usage(db: Session, user: User) -> dict:
    """Current consumption against the guest allowance.

    Exposed so the UI can show "2 of 3 documents" BEFORE the visitor hits the
    wall. Limits alone are not enough: without usage the only way to discover
    the cap is to have an upload rejected, which is a bad way to learn a rule.
    Demo documents are excluded here for the same reason they are excluded from
    the cap itself — they are the app's content, not the visitor's.
    """
    docs = [
        d
        for d in db.query(Document).filter(Document.user_id == user.id).all()
        if not d.is_demo
    ]
    return {
        "documents": len(docs),
        "max_documents": GUEST_MAX_DOCUMENTS,
        "bytes": sum(d.size_bytes or 0 for d in docs),
        "max_bytes": GUEST_MAX_UPLOAD_BYTES,
        "idle_ttl_seconds": GUEST_IDLE_TTL_SECONDS,
    }


def upload_allowance(db: Session, user: User, incoming_bytes: int) -> str | None:
    """Return an error message if this upload would exceed a guest's cap.

    None means allowed. Signed-in accounts are never capped.
    """
    if not is_guest(user):
        return None
    # Count only what the visitor actually added. The seeded demo corpus is the
    # app's own content; charging it to their allowance left one usable slot.
    docs = [
        d
        for d in db.query(Document).filter(Document.user_id == user.id).all()
        if not d.is_demo
    ]
    if len(docs) >= GUEST_MAX_DOCUMENTS:
        return (
            f"Guests can add up to {GUEST_MAX_DOCUMENTS} documents. "
            "Sign in with Google to keep your files and add more."
        )
    used = sum(d.size_bytes or 0 for d in docs)
    if used + incoming_bytes > GUEST_MAX_UPLOAD_BYTES:
        mb = GUEST_MAX_UPLOAD_BYTES // (1024 * 1024)
        return (
            f"Guests can upload up to {mb} MB in total. "
            "Sign in with Google to keep your files and add more."
        )
    return None
