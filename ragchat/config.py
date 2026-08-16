"""App settings (env vars) + hot-reloadable pipeline config (config.yaml).

Secrets live in environment variables only (PRD F15). Pipeline knobs live in
config.yaml and are re-read on every request so experiments need no restart
(PRD F16); index-affecting knobs (chunking + embedding model) are recorded
into a fingerprint stored with each chunk (PRD F18).
"""
from __future__ import annotations

import hashlib
import json
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
# Local state (SQLite fallback DB, uploaded-file cache). On Vercel serverless
# the project directory is read-only, so any writable state must live in /tmp.
# RAG_DATA_DIR can override this explicitly; otherwise we use /tmp on Vercel
# (VERCEL=1 is set in the function environment) and <repo>/data locally.
_DATA_ROOT = os.environ.get("RAG_DATA_DIR") or (
    "/tmp/ragchat-data" if os.environ.get("VERCEL") else str(ROOT / "data")
)
DATA_DIR = Path(_DATA_ROOT)
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
        # App metadata DB (users, documents, chats). On Vercel the Neon
        # integration injects rag_gel_DATABASE_URL; fall back through the
        # same chain the vector store uses so metadata + vectors share one DB.
        # Local/dev without Postgres falls back to a SQLite file.
        self.db_url = (
            os.environ.get("DATABASE_URL")
            or os.environ.get("PG_DATABASE_URL")
            or os.environ.get("rag_gel_DATABASE_URL")
            or f"sqlite:///{DB_PATH}"
        )
        # Vector store backend: "chroma" (local disk, default/dev) or
        # "neon" (Postgres + pgvector, for Vercel/serverless deploy).
        self.vector_backend = os.environ.get("VECTOR_BACKEND", "chroma")
        # Postgres connection string for the Neon backend (pgvector). On Vercel
        # the Neon integration injects it as rag_gel_DATABASE_URL; fall back to
        # PG_DATABASE_URL / DATABASE_URL for local or manual setup.
        self.pg_url = (
            os.environ.get("PG_DATABASE_URL")
            or os.environ.get("rag_gel_DATABASE_URL")
            or os.environ.get("DATABASE_URL", "")
        )
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
        self.default_llm_model = os.environ.get("RAG_LLM_MODEL", "gemma-4-26b-it")
        # Default embedding model name as exposed by the proxy. The proxy
        # serves only gemini-embedding-* models; gemini-embedding-001 is
        # requested at 768 dims (see embeddings.py) to fit pgvector's HNSW
        # ceiling (2000). We normalize to the bare modelspec so the same
        # physical model from different env overrides still collides to one
        # Chroma collection.
        self.default_embedding_model = normalize_embedding_model(
            os.environ.get("RAG_EMBEDDING_MODEL", "models/gemini-embedding-001")
        )
        # Which provider serves embeddings / the reranker. "gemini" = the
        # Google OpenAI-compatible endpoint (free tier, rate-limited);
        # "openrouter" = OpenRouter's /v1 (embedding + rerank), far higher
        # limits. Switchable at runtime via the settings UI (persisted to the
        # DB override) — no restart needed.
        self.default_embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "gemini").lower()
        self.default_reranker_provider = os.environ.get("RERANKER_PROVIDER", "gemini").lower()


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
# Only 768-dim embedding models are exposed (Neon/pgvector stores ONE fixed
# dimension per chunks table). Anything else (qwen3-embedding-8b @1536,
# text-embedding-3-small/large) is intentionally hidden so a user can never
# select a model that fails to insert. See embeddings.EMBEDDING_768_MODELS.
_FALLBACK_EMBEDDING_MODELS: dict[str, list[str]] = {
    "gemini": ["models/gemini-embedding-001", "text-embedding-005"],
    "openrouter": ["google/gemini-embedding-001"],
}

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


def _client_for_provider(provider: str) -> "OpenAI":
    """OpenAI-compatible client for the given provider's model discovery.

    Gemini's /v1/models is the catalog of Google models; OpenRouter's
    /v1/models lists everything OpenRouter serves (including its embedding
    models). This lets the embedding dropdown auto-detect models per provider
    instead of only ever showing Gemini models.
    """
    from .embeddings import _PROVIDER_API_KEY, _PROVIDER_BASE_URL

    p = provider if provider in _PROVIDER_BASE_URL else "gemini"
    return OpenAI(api_key=_PROVIDER_API_KEY.get(p, ""), base_url=_PROVIDER_BASE_URL[p])


def discover_models(provider: str = "gemini") -> dict[str, list[str]]:
    """Return live chat + embedding model lists from the given provider's /v1/models.

    Chat models are always sourced from the generation (Gemini) endpoint; the
    `provider` arg selects which endpoint's embedding models we discover. On any
    failure (no network, bad key, non-standard response) returns the fallback
    catalog so the UI never goes blank and config validation never locks you
    out during a transient proxy outage.
    """
    emb_models: list[str] = []
    try:
        client = _client_for_provider(provider)
        resp = client.models.list()
        for m in getattr(resp, "data", []) or []:
            mid = getattr(m, "id", None)
            if not mid:
                continue
            if _classify_model(mid) == "embedding":
                emb_models.append(mid)
    except Exception:
        pass

    # Chat models: always from the generation endpoint (Gemini/proxy).
    chat_models: list[str] = []
    if provider == "gemini":
        # Only re-query when this is the generation endpoint too.
        try:
            gen = _client_for_provider("gemini")
            resp = gen.models.list()
            for m in getattr(resp, "data", []) or []:
                mid = getattr(m, "id", None)
                if mid and _classify_model(mid) == "chat":
                    chat_models.append(mid)
        except Exception:
            pass

    if emb_models:
        # Only expose 768-dim models (Neon/pgvector stores one fixed dimension
        # per chunks table). Intersect the live discovery with the known 768
        # allowlist so a 1536 model the provider catalogs can never reach the UI.
        allowed = set(_FALLBACK_EMBEDDING_MODELS.get(provider, []))
        emb_models = sorted(set(emb_models) & allowed) or sorted(allowed)
        return {"chat": sorted(chat_models), "embedding": emb_models}
    return {
        "chat": list(_FALLBACK_CHAT_MODELS),
        "embedding": list(_FALLBACK_EMBEDDING_MODELS.get(provider, _FALLBACK_EMBEDDING_MODELS["gemini"])),
    }


def embedding_models_for(provider: str) -> list[str]:
    """Live embedding-model list for `provider` (Gemini or OpenRouter)."""
    return discover_models(provider).get("embedding", [])


def model_catalog() -> dict[str, list[str]]:
    """Cached live model catalog (chat + embedding).

    Chat models come from the generation endpoint; embedding models come from
    the currently-selected embedding provider so the dropdown reflects what that
    provider actually serves.
    """
    now = time.time()
    if _cache["chat"] is not None and (now - _cache["at"]) < _MODEL_CACHE_TTL:
        return {"chat": _cache["chat"], "embedding": _cache["embedding"]}
    from .embeddings import embedding_provider

    catalog = discover_models(embedding_provider())
    _cache["chat"] = catalog["chat"]
    _cache["embedding"] = catalog["embedding"]
    _cache["at"] = now
    return catalog


def is_known_model(model: str, kind: str, provider: str | None = None) -> bool:
    """True if `model` is in the live catalog for `kind`, OR is a deployment
    default for that kind (so a discovery miss can't reject a model the
    provider will still serve).

    When `provider` is given (e.g. the embedding provider being saved), the
    check is scoped to that provider's model list, not the boot-default
    provider's catalog. This matters because a model valid for OpenRouter
    (e.g. ``qwen3-embedding-8b``) is not in the Gemini catalog, and a save
    that switches provider+model together must validate against the target
    provider, not the one currently booted.
    """
    from .embeddings import _PROVIDER_DEFAULT_MODEL

    if kind == "embedding":
        # Scope validation to the provider the model actually belongs to.
        check_provider = (provider or embedding_provider()).lower()
        if model in set(embedding_models_for(check_provider)):
            return True
        if model == settings.default_embedding_model:
            return True
        if model in set(_PROVIDER_DEFAULT_MODEL.values()):
            return True
        return False

    catalog = model_catalog()
    if model in catalog[kind]:
        return True
    if kind == "chat" and model == settings.default_llm_model:
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
    embedding_provider: str
    reranker_provider: str
    web_augmentation: bool
    eval_show: bool

    def fingerprint(self) -> str:
        """Hash of the index-affecting settings (PRD F18).

        Note: web_augmentation is NOT in the fingerprint — web results are
        never embedded into the vector store, so they do not affect chunk
        storage or retrieval from a user's own documents.
        """
        raw = f"{self.chunk_size}|{self.chunk_overlap}|{self.splitter}|{self.embedding_model}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]


def load_config() -> PipelineConfig:
    """Build the live pipeline config.

    Source order (later wins):
      1. config.yaml on disk (read-only on Vercel — fine for reading)
      2. a DB-backed override row (config_overrides), which is the writable
         store used by the settings UI on serverless/read-only deploys.

    Re-read on every call so tuning needs no restart (PRD F16).
    """
    data: dict = {}
    if CONFIG_PATH.exists():
        data = yaml.safe_load(CONFIG_PATH.read_text()) or {}

    # Merge the DB override (if any) on top of the file. The override is the
    # writable path the settings UI uses; on Vercel config.yaml is read-only so
    # all writes go to the DB.
    try:
        from .db import get_config_override

        row = get_config_override("pipeline")
        if row is not None and row.value:
            override = json.loads(row.value)
            for section, kv in override.items():
                if isinstance(kv, dict):
                    data.setdefault(section, {}).update(kv)
                else:
                    data[section] = kv
    except Exception:
        # Never let a DB hiccup break config loading.
        pass

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
        embedding_provider=str(e.get("provider", settings.default_embedding_provider)),
        reranker_provider=str(data.get("reranker", {}).get("provider", settings.default_reranker_provider)),
        web_augmentation=bool(g.get("web_augmentation", False)),
        eval_show=bool(data.get("evaluation", {}).get("show", True)),
    )


def save_config_override(cfg: "PipelineConfig") -> None:
    """Persist a config to the writable DB store (config_overrides).

    Used by the settings UI on serverless/read-only deploys where config.yaml
    cannot be written. Raises on failure so the caller can surface a 500 with a
    real reason instead of an opaque read-only-filesystem crash.
    """
    from .db import set_config_override

    payload = {
        "chunking": {"chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap, "splitter": cfg.splitter},
        "retrieval": {
            "top_k": cfg.top_k,
            "candidate_k": cfg.candidate_k,
            "similarity_threshold": cfg.similarity_threshold,
            "hybrid_search": cfg.hybrid_search,
            "reranker": cfg.reranker,
        },
        "query": {"query_rewrite": cfg.query_rewrite},
        "generation": {
            "llm_model": cfg.llm_model,
            "temperature": cfg.temperature,
            "web_augmentation": cfg.web_augmentation,
        },
        "embedding": {"model": cfg.embedding_model, "provider": cfg.embedding_provider},
        "reranker": {"provider": cfg.reranker_provider},
        "evaluation": {"show": cfg.eval_show},
    }
    set_config_override("pipeline", json.dumps(payload))
