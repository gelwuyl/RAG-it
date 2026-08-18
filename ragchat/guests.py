"""Ephemeral guest accounts.

A visitor who has not signed in gets their OWN throwaway account rather than
sharing one. That distinction is the whole point: the app scopes every query by
user id, so a single shared guest account would let strangers read and delete
each other's uploads — the exact mixing per-user isolation exists to prevent.

Guests are capped (they spend real embedding quota) and reaped after a period of
inactivity. Signing in with Google promotes the guest's work into the permanent
account instead of discarding it.

Reaping is SCHEDULED, driven from outside the deployment. A background thread
would be frozen the moment the response is sent (CLAUDE.md), and Vercel's Hobby
cron — which does exist, contrary to what this docstring used to say — fires
once per DAY, which cannot honour a thirty-minute promise. So a GitHub Actions
schedule calls POST /api/admin/sweep-guests every 15 minutes, and create_guest()
keeps a two-workspace inline backstop for the day that schedule stops.
"""
from __future__ import annotations

import logging
import time

from sqlalchemy.orm import Session

from .db import Conversation, Document, FolderSource, Message, User, new_id, now

log = logging.getLogger(__name__)

GUEST_PROVIDER = "guest"

# Idle window before a guest's workspace is destroyed. Deliberately measured
# from last activity, not creation, so someone mid-session is never wiped.
#
# 30 minutes, down from 2 hours. Two hours only made sense while reaping was
# opportunistic — the TTL was really "whenever the next visitor happens to
# arrive", and a long window hid that. With a scheduled sweep every 15 minutes
# the promise is now real, so it can be a promise worth making: a workspace
# nobody has touched for half an hour is abandoned, and keeping it costs
# storage and leaves anonymous uploads sitting on disk longer than they need to.
#
# An OPEN TAB is not idle. The frontend pings /api/auth/status every few minutes
# while the page is visible, which calls touch(), so reading a long answer never
# races the reaper. Signed-in workspaces are permanent and are excluded by the
# provider filter, not by this number.
GUEST_IDLE_TTL_SECONDS = 30 * 60  # 30 minutes

# How many abandoned workspaces create_guest() clears on the way past.
#
# This is a BACKSTOP, not the mechanism: the scheduled sweeper does the real
# work. It was 20, which put up to twenty full workspace deletions in front of
# a visitor waiting for their first page — measured at 39.7s against 11.1s
# warm. Two keeps cleanup alive if the sweeper is ever disabled (a GitHub
# Actions schedule stops on a repo with no pushes for 60 days) without ever
# putting more than a moment in a visitor's path.
INLINE_REAP_LIMIT = 2

# How far back the close beacon dates a departing guest, as a fraction of the
# TTL. Not a delete: close-and-reopen has to survive, and a browser fires
# pagehide on a reload as readily as on a close. Back-dating to just inside the
# TTL means an abandoned workspace is collected by the next sweep, while a
# visitor who comes straight back finds their work and touch() restores them.
_BEACON_BACKDATE_FRACTION = 0.9

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


def purge_users(db: Session, users: list[User], *, drop_users: bool) -> dict:
    """Delete everything belonging to every user in `users`, set at a time.

    Shared by guest reaping and account deletion so the two can never drift —
    a forgotten table in one path would silently leave orphaned rows behind.

    Five statements regardless of how many workspaces or documents are
    involved, where the previous shape was six to eight round trips PER
    workspace: a vector delete per document, a message delete per conversation,
    and an ORM delete per row after that. On Neon every one of those is a
    network hop, and it is why an inline reap of twenty workspaces measured
    39.7s in front of a visitor's first page load.

    Vector chunks go through delete_users_chunks() rather than prune_chunks(),
    which deliberately no-ops on an empty valid-doc set and so cannot be used
    to clear anyone (CLAUDE.md).
    """
    from .vectordb import delete_users_chunks

    ids = [u.id for u in users]
    if not ids:
        return {"users": 0, "documents": 0, "folders": 0,
                "conversations": 0, "messages": 0}

    try:
        delete_users_chunks(ids)
    except Exception:
        # A vector-store hiccup must not strand the relational rows; the
        # orphan-prune endpoint exists to mop up anything left behind.
        log.exception("purge: clearing vectors for %d user(s) failed", len(ids))

    conv_ids = [
        c.id for c in
        db.query(Conversation.id).filter(Conversation.user_id.in_(ids)).all()
    ]
    n_msgs = 0
    if conv_ids:
        n_msgs = (
            db.query(Message)
            .filter(Message.conversation_id.in_(conv_ids))
            .delete(synchronize_session=False)
        )
    n_convs = (
        db.query(Conversation)
        .filter(Conversation.user_id.in_(ids))
        .delete(synchronize_session=False)
    )
    n_folders = (
        db.query(FolderSource)
        .filter(FolderSource.user_id.in_(ids))
        .delete(synchronize_session=False)
    )
    n_docs = (
        db.query(Document)
        .filter(Document.user_id.in_(ids))
        .delete(synchronize_session=False)
    )

    if drop_users:
        db.query(User).filter(User.id.in_(ids)).delete(synchronize_session=False)
    db.commit()
    if drop_users:
        # A bulk DELETE goes straight to the database and leaves the session's
        # identity map untouched, and SessionLocal sets expire_on_commit=False,
        # so db.get(User, id) would keep handing back a live-looking object for
        # a row that no longer exists. The per-row db.delete() this replaced
        # removed them for us.
        #
        # expunge, not expire: an expired instance re-SELECTs on the next
        # attribute access and raises when the row is gone, which would make
        # reading `guest.id` off the returned list an error. Expunged instances
        # keep the values they already hold — a snapshot, which is all a caller
        # wants from something it just deleted — while db.get() correctly misses.
        for u in users:
            try:
                db.expunge(u)
            except Exception:
                pass
    return {
        "users": len(ids),
        "documents": int(n_docs),
        "folders": int(n_folders),
        "conversations": int(n_convs),
        "messages": int(n_msgs),
    }


def purge_user_data(db: Session, user: User, *, drop_user: bool) -> dict:
    """Delete everything belonging to one user. Wraps the set-based path so the
    single-user case (account deletion) cannot drift from the sweep."""
    summary = purge_users(db, [user], drop_users=drop_user)
    summary.pop("users", None)
    return summary


def reap_stale_guests(db: Session, *, limit: int = 200) -> int:
    """Destroy guest workspaces idle longer than the TTL.

    `limit` bounds the work so a backlog cannot blow the serverless time limit;
    the next call simply picks up where this one stopped. The default suits the
    scheduled sweep, which has the request to itself; create_guest() passes
    INLINE_REAP_LIMIT instead, because there a visitor is waiting.
    """
    cutoff = time.time() - GUEST_IDLE_TTL_SECONDS
    stale = (
        db.query(User)
        .filter(
            User.provider == GUEST_PROVIDER,
            # The demo template is a guest-provider row too, and nothing ever
            # calls touch() on it — it is never signed into — so it looks idle
            # from the moment it is created and the reaper would destroy it two
            # hours later, corpus and vectors alike. That silently deletes the
            # content every visitor is seeded from and bills the next visitor
            # for a re-embed. It is infrastructure, not a workspace: exempt it.
            User.sub != DEMO_TEMPLATE_SUB,
            # created_at covers rows written before last_seen_at existed.
            User.last_seen_at.isnot(None),
            User.last_seen_at < cutoff,
        )
        .limit(limit)
        .all()
    )
    if not stale:
        return 0
    purge_users(db, stale, drop_users=True)
    return len(stale)


def back_date(db: Session, user: User) -> None:
    """Mark a departing guest as nearly-expired, without deleting anything.

    Called from the close beacon. It must NOT delete: `pagehide` fires on a
    reload and on a background-tab discard exactly as it does on a close, so
    deleting here would destroy a workspace the visitor is about to return to.

    Back-dating instead lets the next scheduled sweep collect it if they really
    did leave, while a visitor who comes back within the remaining window finds
    their work and touch() puts them back to full life.
    """
    if not is_guest(user):
        return
    user.last_seen_at = now() - GUEST_IDLE_TTL_SECONDS * _BEACON_BACKDATE_FRACTION
    db.commit()


def create_guest(db: Session) -> User:
    """Provision a fresh, private guest workspace."""
    # A visitor is waiting on this request, so the inline reap is a backstop
    # kept deliberately tiny — see INLINE_REAP_LIMIT. The scheduled sweeper is
    # what actually keeps the table clean.
    reap_stale_guests(db, limit=INLINE_REAP_LIMIT)
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

    from . import demo_vectors
    from .pipeline import ingest_document_text

    # Vectors that ship with the repo, when they match this exact model and
    # fingerprint. Without them the FIRST visitor on a fresh database pays for
    # embedding the whole corpus inside their own 60s guest-login request, and
    # it does not fit — that request 504'd at 63s. None means embed live.
    precomputed = demo_vectors.load(cfg)

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
        text = path.read_text(encoding="utf-8", errors="replace")
        if doc is None:
            doc = Document(user_id=template.id, source_type="upload", title=filename,
                           size_bytes=len(text.encode("utf-8")))
            db.add(doc)
        # Deep search reads source_text, and this path never went through
        # _stage_for_indexing (which is what sets it for uploads). Without this
        # the demo corpus is the one thing in a guest workspace deep search
        # cannot see — the exact documents a first-time visitor tries it on.
        #
        # Set BEFORE the up-to-date check, deliberately. Every deployment
        # already has a ready template at the current fingerprint, so a backfill
        # placed after the `continue` below would never run: the demo corpus
        # would stay un-deep-searchable until the next chunking change. Reading
        # a small file from disk on each call is the cheaper half of that trade.
        if doc.source_text != text:
            doc.source_text = text
            db.commit()
        if doc.config_fingerprint == fp and doc.status == "ready":
            continue
        db.commit()
        # One file's embedding failure must not deny the visitor the others. An
        # unhandled raise here used to propagate out of seed_demo_corpus and
        # leave the arriving guest with an entirely empty workspace, and it
        # would do so again on every subsequent visit, because the failed file
        # is retried first thing each time.
        try:
            vectors = (precomputed or {}).get(filename)
            if vectors:
                n = demo_vectors.seed_document(
                    template.id, doc.id, doc.title, text, cfg, vectors
                )
            else:
                n = ingest_document_text(template.id, doc.id, doc.title, text, cfg)
        except Exception:
            # Covers a stale vector file as well as a provider failure. Both mean
            # "this file is not ready", and the next visit retries it — by which
            # time the live path will have embedded it anyway.
            log.exception("demo corpus: preparing %s failed", filename)
            doc.status = "failed"
            db.commit()
            continue
        doc.status = "ready"
        doc.n_chunks = n
        doc.config_fingerprint = fp
        doc.error = None
        db.commit()
    return template


def seed_demo_corpus(db: Session, guest: User) -> int:
    """Give a new guest their own copy of the demo corpus — no embedding calls.

    Seeding is atomic PER DOCUMENT, and that is the whole point of the shape
    below. The clone row used to be committed BEFORE its vectors were copied,
    so anything the copy threw left the guest owning a document with nothing
    behind it — and aborted the loop, costing them every later file as well.
    That is the "visitor lands with one of the two demo files, and it answers
    nothing" bug: the row says ready with n_chunks set, while the vector store
    holds not a single chunk for it.

    So the row is only FLUSHED, and committed once the vectors are known to
    have landed. A failure rolls that one document back and moves to the next.

    A copy reporting ZERO chunks is a failure too, not a success. It means the
    template's vectors were not there to copy — mid re-index, or a reap that
    ran between the template being read and its chunks being fetched — and
    committing the row anyway is exactly what produces a document the retriever
    can never return. Both backends can report it: the Chroma path skips any
    source collection it finds empty, and the pgvector path returns the INSERT
    ... SELECT rowcount, which is 0 when the source rows are gone.
    """
    from .config import load_config
    from .vectordb import copy_user_chunks, delete_document_chunks

    cfg = load_config()
    template = ensure_demo_template(db, cfg)

    # Snapshot the sources up front. SessionLocal sets expire_on_commit=False,
    # so a commit alone would leave these usable — but a ROLLBACK expires every
    # instance regardless of that flag, and this loop rolls back on failure.
    # Iterating live ORM rows would therefore re-SELECT each remaining source
    # after the first failure, and raise ObjectDeletedError if the template had
    # been reaped underneath us. Plain tuples keep the loop independent of
    # session state.
    #
    # Only READY documents are cloned: a template file whose embedding failed
    # has no vectors to copy, and cloning it would hand the guest precisely the
    # chunkless document this function exists to prevent.
    sources = [
        (d.id, d.source_type, d.title, d.content_hash, d.config_fingerprint,
         d.n_chunks, d.size_bytes, d.source_text)
        for d in db.query(Document).filter(Document.user_id == template.id).all()
        if d.status == "ready"
    ]
    template_id, guest_id = template.id, guest.id

    copied = 0
    for src_id, source_type, title, content_hash, fingerprint, n_chunks, size, source_text in sources:
        clone = Document(
            user_id=guest_id,
            source_type=source_type,
            title=title,
            content_hash=content_hash,
            config_fingerprint=fingerprint,
            status="ready",
            n_chunks=n_chunks,
            size_bytes=size,
            is_demo=True,
            # Copied, not shared: deep search scans the CALLER'S documents, so a
            # guest whose clone has no source_text can vector-search the demo
            # corpus but not deep-search it — which reads as the feature being
            # broken on exactly the documents it is first tried on.
            source_text=source_text,
        )
        db.add(clone)
        db.flush()  # assigns clone.id without committing the row
        clone_id = clone.id
        try:
            n = copy_user_chunks(template_id, src_id, guest_id, clone_id)
        except Exception:
            db.rollback()
            log.exception(
                "demo seeding: copying %s to guest %s failed; skipping the row "
                "rather than seeding a document with no vectors", title, guest_id
            )
            continue
        if n <= 0:
            db.rollback()
            log.warning(
                "demo seeding: %s copied 0 chunks to guest %s (template vectors "
                "missing?); skipping the row", title, guest_id
            )
            continue
        try:
            db.commit()
        except Exception:
            # The vectors landed but the row did not. Orphan chunks are worse
            # than none: query_chunks scopes by user and fingerprint, not by
            # whether a document row still exists, so they would surface in
            # answers citing a document the visitor does not have.
            db.rollback()
            log.exception("demo seeding: committing %s for guest %s failed", title, guest_id)
            try:
                delete_document_chunks(guest_id, clone_id)
            except Exception:
                log.exception("demo seeding: could not clean up orphan chunks for %s", title)
            continue
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
