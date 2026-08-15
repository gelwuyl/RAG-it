"""App settings (env vars) + hot-reloadable pipeline config (config.yaml).

Secrets live in environment variables only (PRD F15). Pipeline knobs live in
config.yaml and are re-read on every request so experiments need no restart
(PRD F16); index-affecting knobs (chunking + embedding model) are recorded
into a fingerprint stored with each chunk (PRD F18).
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Load .env (gitignored) if present, so local secrets don't need to be exported
# in the shell. python-dotenv is installed; failure to load is non-fatal.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "app.db"
UPLOAD_DIR = DATA_DIR / "uploads"
EVAL_DIR = ROOT / "eval"


class Settings:
    def __init__(self) -> None:
        # Endpoint + key. Default now points at Google's OpenAI-compatible
        # endpoint so a Gemini AI Studio key works directly. To use the class
        # proxy instead, set PROXY_BASE_URL back to the proxy /v1 and supply
        # ANTHROPIC_AUTH_TOKEN. GEMINI_API_KEY is the Gemini Studio key;
        # ANTHROPIC_AUTH_TOKEN is kept as a fallback for the class proxy.
        self.proxy_base_url = os.environ.get(
            "PROXY_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "ANTHROPIC_AUTH_TOKEN", ""
        )
        self.db_url = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")
        self.session_secret = os.environ.get("SESSION_SECRET", "dev-session-secret")
        self.allowed_root = Path(
            os.environ.get("RAG_ALLOWED_ROOT", str(Path.home()))
        ).resolve()
        # Google OAuth (PRD F1). Optional: when unset, only the password
        # fallback auth is available (documented risk fallback in PRD §9).
        self.google_client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
        self.google_client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
        self.google_redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "")
        # Default generation model; overridden by config.yaml
        self.default_llm_model = os.environ.get("RAG_LLM_MODEL", "gemma-3-27b-it")
        # Default embedding model name as exposed by the proxy. The proxy serves
        # "text-embedding-005" (768d) and "gemini-embedding" (3072d). We normalize
        # to the bare modelspec so the same physical model from different env
        # overrides still collides to one Chroma collection.
        self.default_embedding_model = normalize_embedding_model(
            os.environ.get("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
        )


def normalize_embedding_model(name: str) -> str:
    """Map a possibly-prefixed proxy embedding modelspec to its bare name.

    A bare name or one already under the proxy path both resolve to the same
    Chroma collection, so re-pointing RAG_EMBEDDING_MODEL at the identical
    model does not create a duplicate (empty) index.
    """
    return name.replace("text-embedding-3-small/", "").replace(
        "models/", ""
    ) or "text-embedding-005"


settings = Settings()

# Fallback catalog used only when live proxy discovery is unreachable. These
# keep the UI populated offline; they are NOT an allowlist — any model the
# proxy actually returns is accepted (see model_catalog below).
_FALLBACK_CHAT_MODELS = ["deepseek-v4-pro", "qwen3.8-max", "qwen3-coder"]
_FALLBACK_EMBEDDING_MODELS = ["text-embedding-005", "gemini-embedding"]

# How long a successful discovery result is cached (seconds). The proxy model
# list changes rarely; a short TTL avoids a round-trip on every settings open
# without going stale for long.
_MODEL_CACHE_TTL = 300.0

_cache: dict = {"chat": None, "embedding": None, "at": 0.0}


def _classify_model(model_id: str) -> str | None:
    """Classify a proxy model id as 'chat', 'embedding', or None (skip)."""
    mid = model_id.lower()
    if any(k in mid for k in ("embed", "embedding")):
        return "embedding"
    # Treat anything that isn't an embedder as a chat/generation model. The
    # proxy's /v1/models is the source of truth, so we don't maintain a
    # per-prefix allowlist here.
    return "chat"


def discover_models() -> dict[str, list[str]]:
    """Return live chat + embedding model lists from the proxy.

    Calls GET {proxy_base_url}/models (OpenAI-compatible). On any failure
    (no network, bad key, non-standard response) returns the fallback
    catalog so the UI never goes blank and config validation never locks
    you out during a transient proxy outage.
    """
    try:
        from .embeddings import openai_client

        client = openai_client()
        resp = client.models.list()
        chat, emb = [], []
        for m in getattr(resp, "data", []) or []:
            mid = getattr(m, "id", None)
            if not mid:
                continue
            kind = _classify_model(mid)
            if kind == "chat":
                chat.append(mid)
            elif kind == "embedding":
                emb.append(mid)
        if chat or emb:
            return {"chat": sorted(chat), "embedding": sorted(emb)}
    except Exception:
        pass
    return {"chat": list(_FALLBACK_CHAT_MODELS), "embedding": list(_FALLBACK_EMBEDDING_MODELS)}


def model_catalog() -> dict[str, list[str]]:
    """Cached live model catalog (chat + embedding)."""
    now = time.time()
    if _cache["chat"] is not None and (now - _cache["at"]) < _MODEL_CACHE_TTL:
        return {"chat": _cache["chat"], "embedding": _cache["embedding"]}
    catalog = discover_models()
    _cache["chat"] = catalog["chat"]
    _cache["embedding"] = catalog["embedding"]
    _cache["at"] = now
    return catalog


def is_known_model(model: str, kind: str) -> bool:
    """True if `model` is in the live catalog for `kind`, OR is the current
    deployment default for that kind (so a discovery miss can't reject a model
    the proxy will still serve)."""
    catalog = model_catalog()
    if model in catalog[kind]:
        return True
    if kind == "chat" and model == settings.default_llm_model:
        return True
    if kind == "embedding" and model == settings.default_embedding_model:
        return True
    return False


@dataclass(frozen=True)
class PipelineConfig:
    chunk_size: int
    chunk_overlap: int
    splitter: str
    top_k: int
    candidate_k: int
    similarity_threshold: float
    hybrid_search: bool
    reranker: bool
    query_rewrite: bool
    llm_model: str
    temperature: float
    embedding_model: str
    web_augmentation: bool

    def fingerprint(self) -> str:
        """Hash of the index-affecting settings (PRD F18).

        Note: web_augmentation is NOT in the fingerprint — web results are
        never embedded into the vector store, so they do not affect chunk
        storage or retrieval from a user's own documents.
        """
        raw = f"{self.chunk_size}|{self.chunk_overlap}|{self.splitter}|{self.embedding_model}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]


def load_config() -> PipelineConfig:
    """Re-read config.yaml on every call so tuning needs no restart."""
    data: dict = {}
    if CONFIG_PATH.exists():
        data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    c = data.get("chunking", {})
    r = data.get("retrieval", {})
    q = data.get("query", {})
    g = data.get("generation", {})
    e = data.get("embedding", {})
    return PipelineConfig(
        chunk_size=int(c.get("chunk_size", 512)),
        chunk_overlap=int(c.get("chunk_overlap", 75)),
        splitter=str(c.get("splitter", "recursive")),
        top_k=int(r.get("top_k", 4)),
        candidate_k=int(r.get("candidate_k", 20)),
        similarity_threshold=float(r.get("similarity_threshold", 0.0)),
        hybrid_search=bool(r.get("hybrid_search", False)),
        reranker=bool(r.get("reranker", False)),
        query_rewrite=bool(q.get("query_rewrite", True)),
        llm_model=str(g.get("llm_model", settings.default_llm_model)),
        temperature=float(g.get("temperature", 0.0)),
        embedding_model=str(e.get("model", settings.default_embedding_model)),
        web_augmentation=bool(g.get("web_augmentation", False)),
    )
