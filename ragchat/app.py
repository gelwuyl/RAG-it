"""FastAPI application: auth, sources, chats, eval (PRD F1-F19)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import auth as authn
from .config import (
    CONFIG_PATH,
    UPLOAD_DIR,
    load_config,
    model_catalog,
    embedding_models_for,
    is_known_model,
    save_config_override,
    settings,
)
from .db import (
    Conversation,
    Document,
    FolderSource,
    Message,
    SessionLocal,
    User,
    get_session,
    get_user,
    init_db,
    new_id,
    now,
)
from .loaders import fetch_url, load_bytes, page_title, TEXT_EXTENSIONS, HTML_EXTENSIONS, PDF_EXTENSIONS
from .pipeline import ingest_document_text, ask
from .vectordb import delete_document_chunks, prune_chunks

app = FastAPI(title="RAG Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


LOCAL_USERNAME = "local"


@app.on_event("startup")
def startup() -> None:
    init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Single-user mode: ensure the built-in local account exists so the UI
    # can sign itself in without any form (auth flow per PRD is deferred).
    db = SessionLocal()
    try:
        existing = (
            db.query(User)
            .filter(User.provider == "password", User.sub == LOCAL_USERNAME)
            .first()
        )
        if not existing:
            db.add(
                User(
                    provider="password",
                    sub=LOCAL_USERNAME,
                    name="Local user",
                    password_hash=authn.hash_password(new_id()),
                )
            )
            db.commit()
    finally:
        db.close()


# ---------- helpers ----------

SUPPORTED_SUFFIXES = TEXT_EXTENSIONS | HTML_EXTENSIONS | PDF_EXTENSIONS


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _index_document(db: Session, user: User, doc: Document, text: str) -> None:
    cfg = load_config()
    doc.status = "indexing"
    db.commit()
    try:
        n = ingest_document_text(user.id, doc.id, doc.title, text, cfg)
        doc.status = "ready"
        doc.n_chunks = n
        doc.config_fingerprint = cfg.fingerprint()
        doc.error = None
    except Exception as exc:  # surfaced to the UI as a failed source
        doc.status = "failed"
        doc.error = str(exc)[:500]
    db.commit()


def _sync_folder(db: Session, user: User, folder: FolderSource) -> dict:
    """Rescan a folder source: add new files, re-index changed ones, drop gone ones (F6a)."""
    root = Path(folder.path)
    if not root.is_dir():
        raise HTTPException(status_code=400, detail="Folder does not exist")
    root_resolved = root.resolve()
    if not str(root_resolved).startswith(str(settings.allowed_root)):
        raise HTTPException(status_code=403, detail="Folder is outside the allowed root")

    files = [
        p
        for p in sorted(root_resolved.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    seen_hashes = {}
    added = reindexed = unchanged = failed = 0

    existing = {
        d.path_or_url: d
        for d in db.query(Document).filter(
            Document.user_id == user.id,
            Document.source_type == "folder",
            Document.path_or_url.like(f"{root_resolved}%"),
        ).all()
    }

    for p in files:
        data = p.read_bytes()
        chash = _content_hash(data)
        seen_hashes[str(p)] = True
        doc = existing.get(str(p))
        if doc is None:
            doc = Document(
                user_id=user.id,
                source_type="folder",
                title=p.name,
                path_or_url=str(p),
                content_hash=chash,
            )
            db.add(doc)
            db.commit()
            try:
                text = load_bytes(p.name, data)
                _index_document(db, user, doc, text)
                added += 1
            except Exception as exc:
                doc.status = "failed"
                doc.error = str(exc)[:500]
                db.commit()
                failed += 1
        elif doc.content_hash != chash:
            delete_document_chunks(user.id, doc.id)
            doc.content_hash = chash
            try:
                text = load_bytes(p.name, data)
                _index_document(db, user, doc, text)
                reindexed += 1
            except Exception as exc:
                doc.status = "failed"
                doc.error = str(exc)[:500]
                db.commit()
                failed += 1
        else:
            unchanged += 1

    # Remove documents whose files disappeared from disk
    for path, doc in existing.items():
        if path not in seen_hashes:
            delete_document_chunks(user.id, doc.id)
            db.delete(doc)
    folder.last_scan_at = now()
    db.commit()
    return {"added": added, "reindexed": reindexed, "unchanged": unchanged, "failed": failed}


def _conversation_messages(db: Session, conversation_id: str) -> list[dict]:
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
        .all()
    )
    return [{"role": m.role, "content": m.content} for m in msgs]


# ---------- auth ----------

class RegisterIn(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(default="", max_length=200)


class LoginIn(RegisterIn):
    pass


@app.get("/api/auth/status")
def auth_status(request: Request, db: Session = Depends(get_session)):
    uid = authn.decode_session(request.cookies.get(authn.SESSION_COOKIE, ""))
    user = get_user(db, uid) if uid else None
    return {
        "authenticated": bool(user),
        "user": {"id": user.id, "name": user.name, "email": user.email} if user else None,
        "google_oauth": authn.oauth_configured(),
    }


@app.post("/api/auth/register")
def register(body: RegisterIn, response: Response, db: Session = Depends(get_session)):
    exists = (
        db.query(User)
        .filter(User.provider == "password", User.sub == body.username)
        .first()
    )
    if exists:
        raise HTTPException(status_code=409, detail="Username already taken")
    user, _err = authn.find_or_create_password_user(db, body.username, body.password)
    response.set_cookie(authn.SESSION_COOKIE, authn.encode_session(user.id), httponly=True)
    return {"id": user.id, "name": user.name}


@app.post("/api/auth/login")
def login(body: LoginIn, response: Response, db: Session = Depends(get_session)):
    try:
        user, _err = authn.find_or_create_password_user(db, body.username, body.password)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    response.set_cookie(authn.SESSION_COOKIE, authn.encode_session(user.id), httponly=True)
    return {"id": user.id, "name": user.name}


@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie(authn.SESSION_COOKIE)
    return {"ok": True}


@app.post("/api/auth/local-login")
def local_login(response: Response, db: Session = Depends(get_session)):
    """Single-user mode: sign in as the built-in local account without a form.

    Auth (Google OAuth / local accounts) is deferred per PRD; this keeps the
    multi-user plumbing intact so it can be turned back on later.
    """
    user = (
        db.query(User)
        .filter(User.provider == "password", User.sub == LOCAL_USERNAME)
        .first()
    )
    if not user:
        raise HTTPException(status_code=503, detail="Local user not initialized")
    response.set_cookie(authn.SESSION_COOKIE, authn.encode_session(user.id), httponly=True)
    return {"id": user.id, "name": user.name}


@app.get("/api/auth/google/login")
def google_login():
    if not authn.oauth_configured():
        raise HTTPException(status_code=400, detail="Google OAuth is not configured")
    state = authn.issue_state()
    url = authn.google_auth_url(state)
    resp = RedirectResponse(url)
    resp.set_cookie("oauth_state", state, httponly=True, max_age=600)
    return resp


@app.get("/api/auth/google/callback")
async def google_callback(request: Request, db: Session = Depends(get_session)):
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    expected = request.cookies.get("oauth_state")
    if not code or not state or state != expected:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")
    try:
        info = await authn.google_exchange_code(code)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Google OAuth exchange failed: {exc}")
    user = authn.find_or_create_google_user(db, info)
    resp = RedirectResponse("/")
    resp.set_cookie(authn.SESSION_COOKIE, authn.encode_session(user.id), httponly=True)
    resp.delete_cookie("oauth_state")
    return resp


# ---------- sources ----------

class UrlIn(BaseModel):
    url: str = Field(min_length=8, max_length=2000)


class FolderIn(BaseModel):
    path: str = Field(min_length=1, max_length=1000)


def _doc_view(d: Document) -> dict:
    return {
        "id": d.id,
        "source_type": d.source_type,
        "title": d.title,
        "path_or_url": d.path_or_url,
        "status": d.status,
        "error": d.error,
        "n_chunks": d.n_chunks,
        "created_at": d.created_at,
    }


@app.get("/api/documents")
def list_documents(user: User = Depends(authn.get_current_user), db: Session = Depends(get_session)):
    docs = (
        db.query(Document)
        .filter(Document.user_id == user.id)
        .order_by(Document.created_at)
        .all()
    )
    return [_doc_view(d) for d in docs]


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile,
    user: User = Depends(authn.get_current_user),
    db: Session = Depends(get_session),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        text = load_bytes(file.filename or "document", data)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    if not text.strip():
        raise HTTPException(status_code=422, detail="No extractable text (scanned PDF?)")

    doc = Document(
        user_id=user.id,
        source_type="upload",
        title=Path(file.filename or "document").name,
        content_hash=_content_hash(data),
    )
    db.add(doc)
    db.commit()
    # Keep the original bytes on disk so re-indexing works after config changes (F17).
    stored = UPLOAD_DIR / user.id
    stored.mkdir(parents=True, exist_ok=True)
    dest = stored / f"{doc.id}_{doc.title}"
    dest.write_bytes(data)
    doc.path_or_url = str(dest)
    _index_document(db, user, doc, text)
    db.commit()
    return _doc_view(doc)


@app.post("/api/documents/url")
def add_url_document(
    body: UrlIn,
    user: User = Depends(authn.get_current_user),
    db: Session = Depends(get_session),
):
    url = body.url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        final_url, content = fetch_url(url)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Fetch failed: {exc}")
    try:
        text = load_bytes("page.html", content, url=final_url)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    if not text.strip():
        raise HTTPException(status_code=422, detail="Page has no readable text")

    doc = Document(
        user_id=user.id,
        source_type="url",
        title=page_title(final_url, content),
        path_or_url=final_url,
        content_hash=_content_hash(content),
    )
    db.add(doc)
    db.commit()
    _index_document(db, user, doc, text)
    return _doc_view(doc)


@app.delete("/api/documents/{doc_id}")
def delete_document(
    doc_id: str,
    user: User = Depends(authn.get_current_user),
    db: Session = Depends(get_session),
):
    doc = db.get(Document, doc_id)
    if not doc or doc.user_id != user.id:
        raise HTTPException(status_code=404, detail="Document not found")
    delete_document_chunks(user.id, doc.id)
    db.delete(doc)
    db.commit()
    return {"ok": True}


@app.post("/api/documents/prune")
def prune_orphan_chunks(
    user: User = Depends(authn.get_current_user),
    db: Session = Depends(get_session),
):
    """Remove Neon vector chunks whose Document row no longer exists (orphans),
    plus any chunk under a stale config fingerprint. Keeps the vector store
    free of 'ghost' chunks after deletes / embedding-model changes.
    """
    valid = {d.id for d in db.query(Document).filter(Document.user_id == user.id).all()}
    current_fp = load_config().fingerprint()
    stale = {
        r.fingerprint
        for r in db.query(Document).filter(Document.user_id == user.id).all()
        if r.config_fingerprint and r.config_fingerprint != current_fp
    }
    removed = prune_chunks(user.id, valid, stale or None)
    return {"ok": True, "removed": removed}


@app.post("/api/documents/reindex")
def reindex_all(
    user: User = Depends(authn.get_current_user),
    db: Session = Depends(get_session),
):
    """Re-chunk and re-embed every source under the current config (F17)."""
    docs = db.query(Document).filter(Document.user_id == user.id).all()
    reindexed = 0
    for doc in docs:
        data = _load_source_text(doc)
        if data is None:
            doc.status = "failed"
            doc.error = "Source no longer readable"
            db.commit()
            continue
        delete_document_chunks(user.id, doc.id)
        _index_document(db, user, doc, data)
        reindexed += 1
    return {"reindexed": reindexed}


def _load_source_text(doc: Document) -> Optional[str]:
    try:
        if doc.source_type == "url":
            _url, content = fetch_url(doc.path_or_url)
            return load_bytes("page.html", content, url=doc.path_or_url)
        if doc.source_type in ("folder", "upload") and doc.path_or_url:
            p = Path(doc.path_or_url)
            if p.is_file():
                return load_bytes(p.name, p.read_bytes())
    except Exception:
        return None
    return None


@app.post("/api/folders")
def add_folder(
    body: FolderIn,
    user: User = Depends(authn.get_current_user),
    db: Session = Depends(get_session),
):
    root = Path(body.path).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(status_code=400, detail="Folder does not exist")
    if not str(root).startswith(str(settings.allowed_root)):
        raise HTTPException(status_code=403, detail="Folder is outside the allowed root")
    exists = (
        db.query(FolderSource)
        .filter(FolderSource.user_id == user.id, FolderSource.path == str(root))
        .first()
    )
    if exists:
        raise HTTPException(status_code=409, detail="Folder already added")
    folder = FolderSource(user_id=user.id, path=str(root))
    db.add(folder)
    db.commit()
    result = _sync_folder(db, user, folder)
    return {"id": folder.id, "path": folder.path, **result}


@app.get("/api/folders")
def list_folders(user: User = Depends(authn.get_current_user), db: Session = Depends(get_session)):
    folders = db.query(FolderSource).filter(FolderSource.user_id == user.id).all()
    out = []
    for f in folders:
        n_docs = (
            db.query(Document)
            .filter(
                Document.user_id == user.id,
                Document.source_type == "folder",
                Document.path_or_url.like(f"{f.path}%"),
            )
            .count()
        )
        out.append(
            {"id": f.id, "path": f.path, "n_docs": n_docs, "last_scan_at": f.last_scan_at}
        )
    return out


@app.post("/api/folders/{folder_id}/rescan")
def rescan_folder(
    folder_id: str,
    user: User = Depends(authn.get_current_user),
    db: Session = Depends(get_session),
):
    folder = db.get(FolderSource, folder_id)
    if not folder or folder.user_id != user.id:
        raise HTTPException(status_code=404, detail="Folder not found")
    return _sync_folder(db, user, folder)


@app.delete("/api/folders/{folder_id}")
def remove_folder(
    folder_id: str,
    user: User = Depends(authn.get_current_user),
    db: Session = Depends(get_session),
):
    folder = db.get(FolderSource, folder_id)
    if not folder or folder.user_id != user.id:
        raise HTTPException(status_code=404, detail="Folder not found")
    docs = (
        db.query(Document)
        .filter(
            Document.user_id == user.id,
            Document.source_type == "folder",
            Document.path_or_url.like(f"{folder.path}%"),
        )
        .all()
    )
    for d in docs:
        delete_document_chunks(user.id, d.id)
        db.delete(d)
    db.delete(folder)
    db.commit()
    return {"ok": True}


# ---------- chats ----------

@app.get("/api/chats")
def list_chats(user: User = Depends(authn.get_current_user), db: Session = Depends(get_session)):
    convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.created_at.desc())
        .all()
    )
    out = []
    for c in convs:
        last = (
            db.query(Message)
            .filter(Message.conversation_id == c.id)
            .order_by(Message.created_at.desc())
            .first()
        )
        # pending = user asked and the assistant reply hasn't landed yet; done otherwise
        status = "pending" if last and last.role == "user" else "done"
        out.append(
            {"id": c.id, "title": c.title, "created_at": c.created_at, "status": status}
        )
    return out


@app.delete("/api/chats/{chat_id}")
def delete_chat(
    chat_id: str,
    user: User = Depends(authn.get_current_user),
    db: Session = Depends(get_session),
):
    conv = db.get(Conversation, chat_id)
    if not conv or conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.query(Message).filter(Message.conversation_id == chat_id).delete()
    db.delete(conv)
    db.commit()
    return {"ok": True}


@app.post("/api/chats")
def create_chat(user: User = Depends(authn.get_current_user), db: Session = Depends(get_session)):
    conv = Conversation(user_id=user.id)
    db.add(conv)
    db.commit()
    return {"id": conv.id, "title": conv.title}


@app.get("/api/chats/{chat_id}")
def get_chat(
    chat_id: str,
    user: User = Depends(authn.get_current_user),
    db: Session = Depends(get_session),
):
    conv = db.get(Conversation, chat_id)
    if not conv or conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="Chat not found")
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == chat_id)
        .order_by(Message.created_at)
        .all()
    )
    return {
        "id": conv.id,
        "title": conv.title,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "citations": json.loads(m.citations) if m.citations else [],
                "eval_line": m.eval_line or "",
                "eval_data": json.loads(m.eval_data) if m.eval_data else None,
            }
            for m in msgs
        ],
    }


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


@app.post("/api/chats/{chat_id}/ask")
def ask_chat(
    chat_id: str,
    body: AskIn,
    user: User = Depends(authn.get_current_user),
    db: Session = Depends(get_session),
):
    conv = db.get(Conversation, chat_id)
    if not conv or conv.user_id != user.id:
        raise HTTPException(status_code=404, detail="Chat not found")

    cfg = load_config()
    history = _conversation_messages(db, chat_id)

    # persist the user turn first (F9)
    db.add(Message(conversation_id=chat_id, role="user", content=body.question))
    if len(history) == 0:
        conv.title = body.question[:60] or "New chat"
    db.commit()

    result = ask(user.id, body.question, history, cfg)

    db.add(
        Message(
            conversation_id=chat_id,
            role="assistant",
            content=result["answer"],
            citations=json.dumps(result["citations"]),
            eval_line=result.get("eval_line") or None,
            eval_data=json.dumps(result["eval"]) if result.get("eval") else None,
        )
    )
    db.commit()
    return result


# ---------- eval ----------

@app.get("/api/models")
def list_models(provider: str | None = None):
    """Live model catalog for the settings dropdowns.

    Chat models come from the generation endpoint (Gemini/proxy). Embedding
    models are discovered from the *embedding* provider's /v1/models so the
    dropdown reflects what the selected provider actually serves. Pass
    `?provider=openrouter` to fetch OpenRouter's embedding catalog instead of
    the default Gemini one. Falls back to a static list if discovery is
    unreachable so the UI never blanks.
    """
    # The requested provider is ALWAYS honoured — including "gemini". This
    # previously special-cased gemini and fell through to the default catalog,
    # which is keyed off the EMBEDDING_PROVIDER env var; on a deploy with
    # EMBEDDING_PROVIDER=openrouter that made the Gemini selection return
    # OpenRouter's models. When no provider is given we fall back to the live
    # config's provider (not a hardcoded "gemini") so the dropdown opens on the
    # list matching what is actually saved.
    emb_provider = (provider or load_config().embedding_provider or "gemini").lower()
    if emb_provider not in ("gemini", "openrouter"):
        emb_provider = "gemini"
    return {**model_catalog(emb_provider), "provider": emb_provider}


@app.get("/api/eval/config")
def eval_config():
    cfg = load_config()
    return {
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "splitter": cfg.splitter,
        "top_k": cfg.top_k,
        "candidate_k": cfg.candidate_k,
        "similarity_threshold": cfg.similarity_threshold,
        "hybrid_search": cfg.hybrid_search,
        "reranker": cfg.reranker,
        "query_rewrite": cfg.query_rewrite,
        "llm_model": cfg.llm_model,
        "temperature": cfg.temperature,
        "embedding_model": cfg.embedding_model,
        "embedding_provider": cfg.embedding_provider,
        "reranker_provider": cfg.reranker_provider,
        "openrouter_configured": bool(os.environ.get("OPENROUTER_API_KEY")),
        "web_augmentation": cfg.web_augmentation,
        "eval_show": cfg.eval_show,
        "fingerprint": cfg.fingerprint(),
    }


@app.post("/api/eval/hybrid-search")
def toggle_hybrid_search():
    """Toggle hybrid_search (KEYWORD/BM25 fusion) on/off. The config is
    re-read every request, so the change takes effect on the next ask. This is
    real vector+keyword fusion, NOT web search. Persisted to the DB (writable)
    because config.yaml is read-only on Vercel serverless."""
    from dataclasses import replace

    cfg = load_config()
    cfg = replace(cfg, hybrid_search=not cfg.hybrid_search)
    save_config_override(cfg)
    return {"hybrid_search": cfg.hybrid_search}


@app.post("/api/eval/web-augmentation")
def toggle_web_augmentation():
    """Toggle web_augmentation (DuckDuckGo fallback) on/off.

    This is a fallback ONLY — when on, web results are appended as labeled
    [web] chunks only when the user's own documents do not clear the
    relevance threshold (pipeline.py:ask). It never overrides grounded
    answers. Default off to preserve strict document grounding (PRD F13).
    Persisted to the DB (writable) because config.yaml is read-only on Vercel.
    """
    from dataclasses import replace

    cfg = load_config()
    cfg = replace(cfg, web_augmentation=not cfg.web_augmentation)
    save_config_override(cfg)
    return {"web_augmentation": cfg.web_augmentation}


class ConfigUpdateIn(BaseModel):
    """Partial config update — only the keys present are written.

    Bounds mirror the min/max on the Settings inputs. The HTML attributes are
    advisory only — the form is read with JS and posted as JSON, so nothing
    stopped an out-of-range value from being persisted. Unbounded values fail
    in ways that look like a broken app rather than a bad setting:
    ``top_k=0`` or ``similarity_threshold>1`` make retrieval return nothing, so
    every question answers "not found"; a negative ``chunk_size`` gets clamped
    to 1 in chunking.py and produces one-character chunks. A 422 naming the
    field is far easier to act on.
    """
    chunk_size: int | None = Field(default=None, ge=128, le=1024)
    chunk_overlap: int | None = Field(default=None, ge=0, le=256)
    splitter: Literal["recursive", "markdown_header", "semantic"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    candidate_k: int | None = Field(default=None, ge=1, le=100)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    hybrid_search: bool | None = None
    reranker: bool | None = None
    query_rewrite: bool | None = None
    llm_model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    embedding_model: str | None = None
    embedding_provider: str | None = None
    reranker_provider: str | None = None
    web_augmentation: bool | None = None
    eval_show: bool | None = None


@app.put("/api/eval/config")
def update_config(body: ConfigUpdateIn):
    """Persist tuning knobs to the DB-backed config store (config.yaml is
    read-only on Vercel serverless). Returns the new config snapshot and
    whether a re-index is needed (index-affecting keys changed)."""
    from dataclasses import replace

    updates = body.model_dump(exclude_none=True)
    # Reject unknown models up front instead of failing mid-ask with a proxy
    # error. "Unknown" means not in the live proxy catalog AND not the current
    # deployment default (so a discovery miss can't lock you out).
    if "llm_model" in updates and not is_known_model(updates["llm_model"], "chat"):
        raise HTTPException(status_code=422, detail=f"Unknown LLM model: {updates['llm_model']}")
    if "embedding_model" in updates:
        # Validate against the provider being saved (which may differ from the
        # currently-booted one), so a model valid for OpenRouter isn't rejected
        # just because the boot provider is Gemini.
        emb_provider = str(updates.get("embedding_provider") or load_config().embedding_provider).lower()
        if not is_known_model(updates["embedding_model"], "embedding", provider=emb_provider):
            raise HTTPException(
                status_code=422, detail=f"Unknown embedding model: {updates['embedding_model']}"
            )
    # Validate provider switches; block OpenRouter if no key is configured
    # (a silent failure later is worse than a clear 422 here). Gemini always
    # allowed (uses GEMINI_API_KEY / proxy).
    for key, env_var in (("embedding_provider", "OPENROUTER_API_KEY"),
                         ("reranker_provider", "OPENROUTER_API_KEY")):
        if key in updates:
            val = str(updates[key]).lower()
            if val not in ("gemini", "openrouter"):
                raise HTTPException(status_code=422, detail=f"Unknown {key}: {updates[key]}")
            if val == "openrouter" and not os.environ.get(env_var):
                raise HTTPException(
                    status_code=422,
                    detail=f"OpenRouter not configured — set {env_var} in .env before selecting OpenRouter for {key}.",
                )
    # Overlap must stay below chunk size or every chunk is mostly duplicated
    # context. chunking.py clamps this defensively, but silently honouring an
    # impossible pair hides a setting the user can see and fix. Compare against
    # the saved chunk_size when only one of the two is being changed.
    live = load_config()
    new_size = updates.get("chunk_size", live.chunk_size)
    new_overlap = updates.get("chunk_overlap", live.chunk_overlap)
    if new_overlap >= new_size:
        raise HTTPException(
            status_code=422,
            detail=(
                f"chunk_overlap ({new_overlap}) must be smaller than chunk_size "
                f"({new_size})."
            ),
        )
    index_affecting = {"chunk_size", "chunk_overlap", "splitter", "embedding_model", "embedding_provider"}
    needs_reindex = bool(index_affecting & set(updates.keys()))

    cfg = load_config()
    cfg = replace(cfg, **updates)
    save_config_override(cfg)
    cfg = load_config()  # re-read so the snapshot reflects what was persisted
    return {
        "config": {
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "splitter": cfg.splitter,
            "top_k": cfg.top_k,
            "candidate_k": cfg.candidate_k,
            "similarity_threshold": cfg.similarity_threshold,
            "hybrid_search": cfg.hybrid_search,
            "reranker": cfg.reranker,
            "query_rewrite": cfg.query_rewrite,
            "llm_model": cfg.llm_model,
            "temperature": cfg.temperature,
            "embedding_model": cfg.embedding_model,
            "embedding_provider": cfg.embedding_provider,
            "reranker_provider": cfg.reranker_provider,
            "openrouter_configured": bool(os.environ.get("OPENROUTER_API_KEY")),
            "web_augmentation": cfg.web_augmentation,
            "eval_show": cfg.eval_show,
            "fingerprint": cfg.fingerprint(),
        },
        "needs_reindex": needs_reindex,
    }


# ---------- eval benchmark (RAGAS-style scorecard, chunked) ----------
#
# The benchmark is CLIENT-DRIVEN and runs in slices. Each POST /api/eval/step
# does one bounded unit of work — index one corpus file, or score a few golden
# questions — persists the result to Postgres, and returns. The browser keeps
# calling until the run reports done.
#
# Why not a background thread (the previous design)? On Vercel:
#   1. The repo dir is read-only, so the old status file write raised EROFS
#      before the try block, killing the thread with no trace.
#   2. The function is frozen as soon as the response is sent, so a daemon
#      thread stops executing anyway.
#   3. A later poll can land on a different instance with no shared memory or
#      /tmp, so in-process state is invisible to it.
#   4. Scoring 46 questions x 3 LLM judges takes minutes, far past the function
#      time limit.
# Slices + DB-backed state solve all four, and give real progress feedback.

from pathlib import Path as _Path

_EVAL_DIR = _Path(__file__).resolve().parent.parent / "eval"

# Questions scored per /api/eval/step call. Each involves a retrieval, an
# embedding pass over the retrieved chunks, one generation and up to three
# judge calls, so keep this small enough to finish well inside the limit.
#
# Measured locally 2026-08-17 (gemma-4-26b-a4b-it judge, OpenRouter embeddings):
# ~40-54s to score ONE question, so a batch of 2 took 83s — past the 60s
# maxDuration in vercel.json. A step that overruns is killed before it commits,
# so the client retries the same slice forever and the run never advances.
# One question per step is the only value that fits, and even that leaves only
# ~6s of headroom; indexing steps are cheap (<9s) by comparison. If judge
# latency grows, slice the judges themselves rather than raising this.
EVAL_BATCH_DEFAULT = 1
EVAL_BATCH_MAX = 10


def _corpus_files() -> list[str]:
    """Corpus filenames, sourced from the harness so the two can't drift."""
    try:
        from eval.run_eval import corpus_files

        return [p.name for p in corpus_files()]
    except Exception:
        return sorted(
            p.name
            for p in _EVAL_DIR.glob("corpus/*")
            if p.suffix.lower() in (".md", ".txt")
        )


def _active_run(db: Session):
    """The most recent run row, or None."""
    from .db import EvalRun

    return db.query(EvalRun).order_by(EvalRun.started_at.desc()).first()


def _run_payload(run) -> dict:
    """Serialise a run row into the shape the Evaluation tab renders."""
    if run is None:
        return {"status": "none", "message": "No benchmark run yet."}
    results = json.loads(run.results) if run.results else []
    metrics = json.loads(run.metrics) if run.metrics else {}
    # While a run is in flight, show metrics computed from what's done so far so
    # the scorecard fills in progressively instead of staying blank.
    if run.status == "running" and results and not metrics:
        try:
            from eval.run_eval import aggregate

            metrics = aggregate(results, retrieval_only=bool(run.retrieval_only))
        except Exception:
            metrics = {}
    indexed = json.loads(run.indexed_files) if run.indexed_files else []
    return {
        "status": run.status,
        "run_id": run.id,
        "total": run.total or 0,
        "completed": run.completed or 0,
        "indexed_files": len(indexed),
        "total_files": len(_corpus_files()),
        "metrics": metrics,
        "results": results,
        "config": json.loads(run.config) if run.config else {},
        "error": run.error,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run.updated_at or run.started_at or time.time())),
    }


class EvalRunIn(BaseModel):
    """Options for starting a benchmark run."""

    limit: int | None = None          # score only the first N golden questions
    retrieval_only: bool = False      # skip generation + judges (much faster)
    batch: int | None = None          # questions per step (clamped server-side)


@app.get("/api/eval")
def get_eval(db: Session = Depends(get_session)):
    """Latest benchmark report (RAGAS-style scorecard) for the Evaluation tab."""
    return _run_payload(_active_run(db))


@app.post("/api/eval/run")
def start_eval(body: EvalRunIn | None = None, db: Session = Depends(get_session)):
    """Begin a benchmark run. Returns immediately; the client then calls
    POST /api/eval/step repeatedly until status != "running"."""
    from eval.run_eval import config_snapshot, load_golden, reset_eval_collection
    from .db import EvalRun

    body = body or EvalRunIn()
    cfg = load_config()

    # Supersede any run left in "running" (e.g. the tab was closed mid-run) so
    # a stale row can't block a fresh start forever.
    prev = _active_run(db)
    if prev is not None and prev.status == "running":
        prev.status = "cancelled"
        prev.updated_at = time.time()
        db.commit()

    try:
        items = load_golden(body.limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read golden set: {exc}")

    # Clear any chunks from a previous benchmark so stale vectors can't be
    # retrieved into this run's scores.
    reset_eval_collection(cfg)

    run = EvalRun(
        status="running",
        total=len(items),
        completed=0,
        indexed_files=json.dumps([]),
        retrieval_only=bool(body.retrieval_only),
        results=json.dumps([]),
        config=json.dumps(config_snapshot(cfg)),
        started_at=time.time(),
        updated_at=time.time(),
    )
    db.add(run)
    db.commit()
    return _run_payload(run)


@app.post("/api/eval/step")
def step_eval(body: EvalRunIn | None = None, db: Session = Depends(get_session)):
    """Advance the active run by one bounded slice of work.

    Order of work: index the corpus one file at a time, then score questions in
    batches. Every slice commits before returning, so progress survives the
    function being frozen or the next request landing elsewhere.
    """
    from eval.run_eval import (
        EVAL_USER,
        aggregate,
        embed_passages,
        load_golden,
        score_item,
    )

    body = body or EvalRunIn()
    run = _active_run(db)
    if run is None or run.status != "running":
        return _run_payload(run)

    batch = max(1, min(body.batch or EVAL_BATCH_DEFAULT, EVAL_BATCH_MAX))
    cfg = load_config()

    try:
        # --- phase 1: index the corpus, one file per step ---
        indexed = json.loads(run.indexed_files) if run.indexed_files else []
        pending = [f for f in _corpus_files() if f not in indexed]
        if pending:
            name = pending[0]
            path = _EVAL_DIR / "corpus" / name
            text_content = path.read_text(encoding="utf-8", errors="replace")
            ingest_document_text(EVAL_USER, name, name, text_content, cfg)
            indexed.append(name)
            run.indexed_files = json.dumps(indexed)
            run.updated_at = time.time()
            db.commit()
            return _run_payload(run)

        # --- phase 2: score the next `batch` questions ---
        items = load_golden(run.total)
        results = json.loads(run.results) if run.results else []
        start = run.completed or 0
        for item in items[start : start + batch]:
            item["_golden_embs"] = embed_passages(
                item.get("golden_passages", []),
                cfg.embedding_model,
                provider=cfg.embedding_provider,
            )
            results.append(score_item(item, cfg, retrieval_only=bool(run.retrieval_only)))

        run.results = json.dumps(results)
        run.completed = len(results)
        run.updated_at = time.time()
        if run.completed >= (run.total or 0):
            run.metrics = json.dumps(
                aggregate(results, retrieval_only=bool(run.retrieval_only))
            )
            run.status = "done"
        db.commit()
    except Exception as exc:  # record the failure instead of 500-ing the poll
        db.rollback()
        run = _active_run(db)
        if run is not None:
            run.status = "error"
            run.error = f"{type(exc).__name__}: {exc}"[:500]
            run.updated_at = time.time()
            db.commit()
    return _run_payload(run)


@app.get("/api/health")
def health():
    """Liveness + deploy self-check.

    Reports DB connectivity and which secrets are *present* (values are
    NEVER returned) so deploy issues can be diagnosed without the Vercel
    dashboard. A 200 here means the function booted and reached its DB.
    """
    import os as _os

    from sqlalchemy import text as _text

    from .config import DATA_DIR, settings as _s
    from .db import engine as _engine

    env_present = {
        "GEMINI_API_KEY": bool(_os.environ.get("GEMINI_API_KEY")),
        "OPENROUTER_API_KEY": bool(_os.environ.get("OPENROUTER_API_KEY")),
        "DATABASE_URL": bool(_os.environ.get("DATABASE_URL")),
        "PG_DATABASE_URL": bool(_os.environ.get("PG_DATABASE_URL")),
        "rag_gel_DATABASE_URL": bool(_os.environ.get("rag_gel_DATABASE_URL")),
        "VECTOR_BACKEND": bool(_os.environ.get("VECTOR_BACKEND")),
    }
    db_ok = False
    db_err = None
    try:
        with _engine.connect() as conn:
            conn.execute(_text("select 1"))
        db_ok = True
    except Exception as exc:  # surface the connection error, not a 500
        db_err = str(exc)[:200]

    # Effective (live) config, so a deploy can be diagnosed from the browser
    # without dashboard access. These are the values that actually get used —
    # env vars are only the boot defaults and are frequently NOT what's active,
    # because the Settings UI persists overrides to the database.
    cfg_info: dict = {}
    judge_info: dict = {}
    try:
        cfg = load_config()
        cfg_info = {
            "llm_model": cfg.llm_model,
            "embedding_model": cfg.embedding_model,
            "embedding_provider": cfg.embedding_provider,
            "reranker_provider": cfg.reranker_provider,
            "fingerprint": cfg.fingerprint(),
            "env_llm_model_default": settings.default_llm_model,
            "env_embedding_provider_default": settings.default_embedding_provider,
        }
    except Exception as exc:
        cfg_info = {"error": str(exc)[:200]}
    try:
        from eval.judges import judge_model

        judge_info = {"model": judge_model(), "importable": True}
    except Exception as exc:
        judge_info = {"importable": False, "error": str(exc)[:200]}

    # Is live model discovery actually reaching each provider, or are we
    # silently serving the static fallback catalog?
    discovery: dict = {}
    for _p in ("gemini", "openrouter"):
        try:
            discovery[_p] = embedding_models_for(_p)
        except Exception as exc:
            discovery[_p] = [f"error: {exc}"[:120]]

    return {
        "ok": True,
        "db_backend": _s.vector_backend,
        "app_db_is_postgres": ("postgres" in _s.db_url),
        "db_connected": db_ok,
        "db_error": db_err,
        "data_dir": str(DATA_DIR),
        "env_present": env_present,
        "effective_config": cfg_info,
        "judge": judge_info,
        "embedding_models_by_provider": discovery,
    }


# Serve the built frontend in production; in dev Vite serves it instead.
STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
