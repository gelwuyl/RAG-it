"""FastAPI application: auth, sources, chats, eval (PRD F1-F19)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from . import auth as authn
from .config import CONFIG_PATH, UPLOAD_DIR, load_config, settings
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
from .store import delete_document_chunks

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
    return [
        {"id": c.id, "title": c.title, "created_at": c.created_at}
        for c in convs
    ]


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
        )
    )
    db.commit()
    return result


# ---------- eval ----------

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
        "fingerprint": cfg.fingerprint(),
    }


@app.post("/api/eval/hybrid-search")
def toggle_hybrid_search():
    """Toggle hybrid_search on/off in config.yaml. The config is re-read every
    request, so the change takes effect on the next ask."""
    import yaml as _yaml

    data = _yaml.safe_load(CONFIG_PATH.read_text()) or {}
    data.setdefault("retrieval", {})["hybrid_search"] = not data.get(
        "retrieval", {}
    ).get("hybrid_search", False)
    CONFIG_PATH.write_text(_yaml.dump(data, default_flow_style=False))
    return {"hybrid_search": data["retrieval"]["hybrid_search"]}


class ConfigUpdateIn(BaseModel):
    """Partial config update — only the keys present are written."""
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    splitter: str | None = None
    top_k: int | None = None
    candidate_k: int | None = None
    similarity_threshold: float | None = None
    hybrid_search: bool | None = None
    reranker: bool | None = None
    query_rewrite: bool | None = None
    llm_model: str | None = None
    temperature: float | None = None
    embedding_model: str | None = None


@app.put("/api/eval/config")
def update_config(body: ConfigUpdateIn):
    """Write tuning knobs back to config.yaml. Returns the new config snapshot
    and whether a re-index is needed (index-affecting keys changed)."""
    import yaml as _yaml

    data = _yaml.safe_load(CONFIG_PATH.read_text()) or {}
    updates = body.model_dump(exclude_none=True)
    index_affecting = {"chunk_size", "chunk_overlap", "splitter", "embedding_model"}
    needs_reindex = bool(index_affecting & set(updates.keys()))

    # Map flat keys back to nested config.yaml sections
    section_map = {
        "chunk_size": ("chunking", "chunk_size"),
        "chunk_overlap": ("chunking", "chunk_overlap"),
        "splitter": ("chunking", "splitter"),
        "top_k": ("retrieval", "top_k"),
        "candidate_k": ("retrieval", "candidate_k"),
        "similarity_threshold": ("retrieval", "similarity_threshold"),
        "hybrid_search": ("retrieval", "hybrid_search"),
        "reranker": ("retrieval", "reranker"),
        "query_rewrite": ("query", "query_rewrite"),
        "llm_model": ("generation", "llm_model"),
        "temperature": ("generation", "temperature"),
        "embedding_model": ("embedding", "model"),
    }
    for key, value in updates.items():
        section, subkey = section_map.get(key, (None, None))
        if section:
            data.setdefault(section, {})[subkey] = value

    CONFIG_PATH.write_text(_yaml.dump(data, default_flow_style=False))
    cfg = load_config()
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
            "fingerprint": cfg.fingerprint(),
        },
        "needs_reindex": needs_reindex,
    }


@app.get("/api/health")
def health():
    return {"ok": True}


# Serve the built frontend in production; in dev Vite serves it instead.
STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
