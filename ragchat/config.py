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
from openai import OpenAI

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
        # Shared secret for POST /api/admin/sweep-guests, called on a schedule
        # from outside the deployment (GitHub Actions) because a serverless
        # function cannot run its own timer and Vercel's Hobby cron only fires
        # daily. Empty means the route is DISABLED, not open: an unset secret
        # must never degrade into an unauthenticated deletion endpoint.
        self.sweep_secret = os.environ.get("GUEST_SWEEP_SECRET", "")
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
        # Boot default only, like every other default_* here: the live
        # value comes from load_config(). Flash-lite because it is the
        # cheapest model on this endpoint that actually emits tool calls.
        self.default_router_model = os.environ.get(
            "RAG_ROUTER_MODEL", "models/gemini-3.5-flash-lite"
        )
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
        # Cohere rerank-v3.5 via OpenRouter. Boot default only — config.yaml's
        # `reranker.provider` and the Settings dropdown both override it, and
        # _rerank() reads the LIVE config, never this.
        self.default_reranker_provider = os.environ.get("RERANKER_PROVIDER", "openrouter").lower()


def normalize_embedding_model(name: str) -> str:
    """Map a possibly-prefixed proxy embedding modelspec to its bare name.

    A bare name or one already under the proxy path both resolve to the same
    Chroma collection, so re-pointing RAG_EMBEDDING_MODEL at the identical
    model does not create a duplicate (empty) index.
    """
    # The empty-name fallback must be a model the endpoint actually serves.
    # It used to be text-embedding-005, which 404s on Google's OpenAI-compatible
    # endpoint.
    return name.replace("text-embedding-3-small/", "").replace(
        "models/", ""
    ) or "gemini-embedding-001"


settings = Settings()

# Fallback catalog used only when live proxy discovery is unreachable. These
# keep the UI populated offline; they are NOT an allowlist — any model the
# proxy actually returns is accepted (see model_catalog below).
_FALLBACK_CHAT_MODELS = ["deepseek-v4-pro", "qwen3.8-max", "qwen3-coder"]
# Only 768-dim embedding models are exposed (Neon/pgvector stores ONE fixed
# dimension per chunks table). OpenRouter honors `dimensions=768` for every
# model listed below (verified live), so each is safe to select. See
# embeddings.EMBEDDING_768_MODELS.
#
# Gemini exposes exactly ONE *allowlisted* embedder here: gemini-embedding-001,
# confirmed live at 768 dims. Google also serves gemini-embedding-2 and
# -2-preview, deliberately excluded until their 768 support is verified — see
# embeddings.EMBEDDING_768_MODELS. (text-embedding-004/005 do NOT exist on
# Google's OpenAI-compatible endpoint and 404 — see config.yaml.) Everything
# else in the dropdown comes from OpenRouter.
_FALLBACK_EMBEDDING_MODELS: dict[str, list[str]] = {
    "gemini": ["models/gemini-embedding-001"],
    # Must stay in step with embeddings.EMBEDDING_768_MODELS — this is the list
    # served when live discovery fails, and a longer list here would offer models
    # the allowlist rejects. Two by choice, not by capability; the reasoning lives
    # with the allowlist.
    "openrouter": [
        "qwen/qwen3-embedding-8b",
        "perplexity/pplx-embed-v1-0.6b",
    ],
}

# How long a successful discovery result is cached (seconds). The proxy model
# list changes rarely; a short TTL avoids a round-trip on every settings open
# without going stale for long.
_MODEL_CACHE_TTL = 300.0

# Cache is keyed BY PROVIDER. It used to be a single global slot, which meant
# the embedding list discovered for whichever provider happened to be asked for
# first was then served for every provider — so picking "Gemini" in Settings
# could return OpenRouter's models (and vice versa). Each provider now gets its
# own entry: {provider: {"chat": [...], "embedding": [...], "at": ts}}.
_cache: dict[str, dict] = {}


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
    chat_models: list[str] = []
    try:
        client = _client_for_provider(provider)
        resp = client.models.list()
        for m in getattr(resp, "data", []) or []:
            mid = getattr(m, "id", None)
            if not mid:
                continue
            kind = _classify_model(mid)
            if kind == "embedding":
                emb_models.append(mid)
            # When this provider IS the generation endpoint, the same response
            # already carries the chat models — classify them here rather than
            # issuing a second identical /v1/models request.
            elif provider == "gemini":
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
    return model_catalog(provider).get("embedding", [])


# Cache key for the chat list. Chat models are NOT provider-scoped — generation
# always goes to the Gemini/proxy endpoint no matter which embedding provider is
# selected — so they get one shared entry rather than a copy per provider.
_CHAT_KEY = "__chat__"


def _is_live_chat(chat: list[str] | None) -> bool:
    """True only for a genuinely discovered chat list.

    ``discover_models`` returns ``_FALLBACK_CHAT_MODELS`` when the endpoint is
    unreachable, so a truthy list is NOT proof of a live result. The distinction
    matters because the fallback must never be cached: Google's free tier
    rate-limits bursts of /v1/models, and one throttled call caching its
    placeholder names would hide the real catalog for the whole TTL.
    """
    return bool(chat) and list(chat) != list(_FALLBACK_CHAT_MODELS)


def _chat_models() -> list[str]:
    """Chat models from the GENERATION endpoint, cached independently.

    Kept out of the per-provider entries deliberately. When the chat list lived
    there, discovery for OpenRouter (whose /v1/models carries no chat models)
    fell back to the static list and *cached it under that provider* — so
    selecting OpenRouter embeddings replaced a 51-model chat dropdown with 3
    stale placeholder names for the whole TTL, purely as a side effect of which
    provider happened to be asked for first. One shared entry, refreshed only
    from the endpoint that actually serves chat models, can't drift that way.
    """
    now = time.time()
    hit = _cache.get(_CHAT_KEY)
    if hit is not None and hit.get("chat") and (now - hit["at"]) < _MODEL_CACHE_TTL:
        return hit["chat"]

    chat = discover_models("gemini").get("chat")
    if not _is_live_chat(chat):
        # Serve the fallback but do NOT cache it, so the next call retries.
        return list(_FALLBACK_CHAT_MODELS)
    _cache[_CHAT_KEY] = {"chat": chat, "at": now}
    return chat


def model_catalog(provider: str | None = None) -> dict[str, list[str]]:
    """Cached live model catalog (chat + embedding) for one embedding provider.

    `provider` selects which endpoint's EMBEDDING models are returned. Chat
    models always come from the generation endpoint. When omitted, the
    deployment-default provider is used.

    Embedding lists are cached per provider, so asking for Gemini can never hand
    back the list that was discovered for OpenRouter.
    """
    from .embeddings import embedding_provider

    p = (provider or embedding_provider()).lower()
    now = time.time()
    hit = _cache.get(p)
    if hit is not None and (now - hit["at"]) < _MODEL_CACHE_TTL:
        return {"chat": _chat_models(), "embedding": hit["embedding"]}

    catalog = discover_models(p)
    _cache[p] = {"embedding": catalog["embedding"], "at": now}
    # The gemini response IS the generation endpoint's, so it already carries the
    # chat models. Seed the shared chat entry from it instead of letting
    # _chat_models() issue a second identical /v1/models request — but only when
    # the list is genuinely live, never the fallback.
    if _is_live_chat(catalog.get("chat")):
        _cache[_CHAT_KEY] = {"chat": catalog["chat"], "at": now}
    return {"chat": _chat_models(), "embedding": catalog["embedding"]}


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
    from .embeddings import (
        _PROVIDER_DEFAULT_MODEL,
        embedding_provider,
        same_embedding_model,
    )

    if kind == "embedding":
        # Scope validation to the provider the model actually belongs to.
        check_provider = (provider or embedding_provider()).lower()
        # Compare ignoring the vendor prefix so a legacy bare id already sitting
        # in the saved config (e.g. `qwen3-embedding-8b`) still validates against
        # the allowlist's prefixed spelling (`qwen/qwen3-embedding-8b`). Exact
        # matching locked users out of saving Settings at all — every save
        # re-submits the stored model, so a stored id the allowlist no longer
        # spelled the same way rejected every save, including ones that changed
        # nothing about the embedding.
        if any(same_embedding_model(model, m) for m in embedding_models_for(check_provider)):
            return True
        # The deployment-default escape hatch exists so a discovery miss cannot
        # reject a model the provider will still serve — but it must be SCOPED
        # to the provider that default belongs to. Unscoped, it accepted any
        # provider+model pairing as soon as the default belonged to the other
        # provider: with the default now qwen/qwen3-embedding-8b, saving
        # provider=gemini with a Qwen id validated fine and then 404'd on every
        # embedding call, which is exactly the near-miss this guard exists to
        # catch (see CLAUDE.md, "Model ids must be exact").
        if (
            check_provider == settings.default_embedding_provider
            and same_embedding_model(model, settings.default_embedding_model)
        ):
            return True
        if same_embedding_model(model, _PROVIDER_DEFAULT_MODEL.get(check_provider, "")):
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
    router_model: str
    temperature: float
    embedding_model: str
    embedding_provider: str
    reranker_provider: str
    eval_show: bool

    def fingerprint(self) -> str:
        """Hash of the index-affecting settings (PRD F18).

        Note deep search is not here and cannot be: it is a per-request flag,
        not config, and it embeds nothing — it reads Document.source_text
        directly. Nothing about it can invalidate a chunk.
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
        # NOT read from config — keyword fusion is unconditional.
        #
        # It was a setting, and that was wrong twice over. "Should BM25 scores be
        # fused into the vector ranking by RRF" is not a question a user can
        # answer, and it costs no model call, so there is nothing to trade. It is
        # simply how retrieval works here — the landing page has always said so.
        #
        # Hardcoded rather than defaulted, because a default can be masked: the
        # config_overrides DB row merges on top of config.yaml, so every
        # environment that had ever saved Settings would have kept running
        # vector-only until someone re-saved. Reading nothing is the only way the
        # promise holds everywhere without a migration.
        #
        # The field stays on PipelineConfig: the pipeline reads it, and the unit
        # tests still exercise both paths to prove fusion changes what it claims
        # to change (tests/test_retrieval_fixes.py).
        hybrid_search=True,
        # Whether to rerank IS a real choice — it trades a call for ordering — so
        # this one is still read from config. Only the default changed to True.
        reranker=bool(r.get("reranker", True)),
        query_rewrite=bool(q.get("query_rewrite", True)),
        llm_model=str(g.get("llm_model", settings.default_llm_model)),
        # The model that CHOOSES a tool, which is not the model that writes.
        # They are separated because the requirement differs: writing wants the
        # best prose available, choosing wants function calling and speed. The
        # answering model here emits no tool calls at all (verified live against
        # the endpoint), so a single-model design cannot choose anything.
        router_model=str(g.get("router_model", settings.default_router_model)),
        temperature=float(g.get("temperature", 0.0)),
        embedding_model=str(e.get("model", settings.default_embedding_model)),
        embedding_provider=str(e.get("provider", settings.default_embedding_provider)),
        # NOT read from config — the same reasoning as hybrid_search above.
        #
        # "Which vendor re-scores the candidate pool" is not a question a user can
        # answer, and the two options are not a trade: Cohere rerank-v3.5 is ONE
        # cheap purpose-built call for the whole pool, while the LLM cross-encoder
        # is one chat call PER passage — slower and dearer for no measured gain.
        # That is an implementation detail, not a setting.
        #
        # Hardcoded rather than defaulted for the reason fusion had to be: the
        # config_overrides row merges on top of config.yaml, so every deployment
        # that had ever saved Settings would have kept using the cross-encoder
        # until someone re-saved. `reranker` (on/off) above stays honoured from
        # the row, because that one really is the user's call.
        reranker_provider="openrouter",
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
            # hybrid_search is deliberately NOT persisted. load_config() no longer
            # reads it, so writing it would leave a value in the row that looks
            # authoritative and does nothing.
            "reranker": cfg.reranker,
        },
        "query": {"query_rewrite": cfg.query_rewrite},
        "generation": {
            "llm_model": cfg.llm_model,
            "router_model": cfg.router_model,
            "temperature": cfg.temperature,
        },
        "embedding": {"model": cfg.embedding_model, "provider": cfg.embedding_provider},
        # `reranker.provider` is deliberately NOT persisted, for the same reason
        # hybrid_search is not: load_config() hardcodes it, so writing it would
        # leave an authoritative-looking value in the row that changes nothing.
        "evaluation": {"show": cfg.eval_show},
    }
    set_config_override("pipeline", json.dumps(payload))
