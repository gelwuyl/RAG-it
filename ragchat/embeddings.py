"""Embeddings + reranking via provider endpoints (PRD T3).

Two embedding providers are supported, selected by the EMBEDDING_PROVIDER env
var (default "gemini" to preserve current behaviour):

  - "gemini"      -> Google's OpenAI-compatible endpoint (GEMINI_API_KEY).
                     Free tier is heavily rate-limited, so uploading many
                     documents stalls. Requested at 768 dims (fixed for
                     gemini-embedding-001).
  - "openrouter"  -> OpenRouter's OpenAI-compatible /v1/embeddings
                     (OPENROUTER_API_KEY). Far higher rate limits and a
                     choice of embedding models (text-embedding-3-small,
                     qwen3-embedding-*). Default model here is
                     openai/text-embedding-3-small at 1536 dims.

Generation (chat) and the LLM judges always use the GENERATION client
(openai_client()) — i.e. the proxy_base_url / GEMINI_API_KEY path — regardless
of which provider serves embeddings. Only embeddings + reranking switch.

The reranker is also provider-aware:
  - "gemini" (default)  -> the existing LLM cross-encoder (slow, uses the gen LLM).
  - "openrouter"        -> Cohere rerank-v3.5 at https://openrouter.ai/api/v1/rerank
                            (fast, cheap, purpose-built reranking).

One embedding model per deployment; the config fingerprint (F18) prevents
chunks from different models mixing in one query.

Resilience: transient 429 (RESOURCE_EXHAUSTED) and 5xx errors are retried
with exponential backoff so a brief quota dip doesn't drop a document
or fail a query outright.
"""
from __future__ import annotations

import os
import time

import requests
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from .config import settings

# Retry/backoff for transient 429 (RESOURCE_EXHAUSTED) and 5xx.
_MAX_RETRIES = 4
_BACKOFF_BASE = 1.5  # seconds

# Per-provider embedding dimension. Stored so we never send a model the
# endpoint rejects (e.g. gemini-embedding-001 is fixed at 768; OpenRouter's
# text-embedding-3-small defaults to 1536).
_PROVIDER_DIMS: dict[str, int] = {
    "gemini": 768,
    "openrouter": 1536,
}

# The Neon/pgvector backend uses a SINGLE shared `chunks` table with one fixed
# vector dimension (768), so every embedding model the UI exposes must be able
# to return 768-dim vectors. OpenRouter's embeddings API honors the `dimensions`
# param for all of these (verified live): the vector is materialized at 768 no
# matter the model's native default. Gemini's embedding models are natively
# 768. This allowlist is the UI's safe set — add a model here only after
# confirming `dimensions=768` actually returns 768.
EMBEDDING_768_MODELS: dict[str, list[str]] = {
    "gemini": ["models/gemini-embedding-001", "text-embedding-005"],
    "openrouter": [
        "openai/text-embedding-3-small",
        "openai/text-embedding-3-large",
        "qwen/qwen3-embedding-8b",
        "qwen/qwen3-embedding-4b",
        "perplexity/pplx-embed-v1-0.6b",
        "google/gemini-embedding-001",
    ],
}

# Provider -> base URL for the OpenAI-compatible /v1 API.
_PROVIDER_BASE_URL: dict[str, str] = {
    "gemini": os.environ.get(
        "PROXY_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    "openrouter": "https://openrouter.ai/api/v1",
}

# Provider -> API key env var.
_PROVIDER_API_KEY: dict[str, str] = {
    "gemini": settings.api_key,  # GEMINI_API_KEY (set in config.Settings)
    "openrouter": os.environ.get("OPENROUTER_API_KEY", ""),
}

# Default embedding model slug per provider (overridable via config.yaml /
# the settings UI; OpenRouter needs the provider prefix).
_PROVIDER_DEFAULT_MODEL: dict[str, str] = {
    "gemini": "models/gemini-embedding-001",
    "openrouter": "openai/text-embedding-3-small",
}

# Generation (chat) client — always Google / proxy base url (unchanged).
_gen_client: OpenAI | None = None
# Embedding client — provider-aware (gemini | openrouter).
_emb_client: OpenAI | None = None
# Reranker client — OpenRouter only.
_rerank_client: OpenAI | None = None


def embedding_provider() -> str:
    """Default embedding provider: env-selected (validated), else "gemini".

    This is the *deployment default*. The live UI selection flows through
    PipelineConfig.embedding_provider -> ProxyEmbeddings(provider=...), so a
    runtime switch in Settings is honoured without needing an env var or
    restart (see ProxyEmbeddings / embedding_client). The env var remains the
    fallback for deployments that only set environment variables (e.g. Vercel).
    """
    p = (os.environ.get("EMBEDDING_PROVIDER") or "gemini").lower()
    return p if p in _PROVIDER_BASE_URL else "gemini"


def reranker_provider() -> str:
    """Which provider serves the reranker (env-selected, validated)."""
    p = (os.environ.get("RERANKER_PROVIDER") or "gemini").lower()
    return p if p in ("gemini", "openrouter") else "gemini"


def _resolve_provider(provider: str | None) -> str:
    """Coerce a provider arg to a known key, falling back to the env default."""
    if provider in _PROVIDER_BASE_URL:
        return provider  # type: ignore[return-value]
    return embedding_provider()


def embedding_dim(provider: str | None = None, model: str | None = None) -> int:
    """Dimension to request for the given embedding provider/model.

    Returns 768 for every model the UI exposes. The Neon/pgvector backend
    stores ALL embeddings in one `chunks` table with a vector(768) column, and
    OpenRouter's embeddings API honors the `dimensions` param for every model
    we tested (text-embedding-3-small, Qwen3-Embedding-4B/8B,
    pplx-embed-v1-0.6b, google/gemini-embedding-001) — so requesting 768 keeps
    every model consistent with the stored column, regardless of the model's
    native default. Gemini's native embedding dim is also 768.
    """
    return 768


def openai_client() -> OpenAI:
    """Singleton OpenAI client for GENERATION (chat + judges).

    Always uses the proxy_base_url / GEMINI_API_KEY path, independent of the
    embedding-provider selection. This keeps chat generation on the Google
    endpoint while embeddings may run on OpenRouter.
    """
    global _gen_client
    if _gen_client is None:
        _gen_client = OpenAI(api_key=settings.api_key, base_url=settings.proxy_base_url)
    return _gen_client


def embedding_client(provider: str | None = None) -> OpenAI:
    """Singleton OpenAI client for the active EMBEDDING provider."""
    global _emb_client
    p = _resolve_provider(provider)
    if _emb_client is None:
        _emb_client = OpenAI(
            api_key=_PROVIDER_API_KEY[p],
            base_url=_PROVIDER_BASE_URL[p],
        )
    return _emb_client


def default_model_for(provider: str | None = None) -> str:
    """Default embedding model slug for a provider (overridable in config.yaml / UI)."""
    return _PROVIDER_DEFAULT_MODEL.get(_resolve_provider(provider), "models/gemini-embedding-001")


def rerank_client() -> OpenAI:
    """Singleton OpenAI client for the OpenRouter reranker endpoint."""
    global _rerank_client
    if _rerank_client is None:
        _rerank_client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        )
    return _rerank_client


def _is_retryable(exc: Exception) -> bool:
    """True if `exc` looks like a transient API error worth retrying."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    if isinstance(status, int) and 500 <= status < 600:
        return True
    code = getattr(exc, "code", None)
    if code in ("rate_limit_exceeded", "RESOURCE_EXHAUSTED", "internal_error"):
        return True
    return False


def retry_call(fn, *args, **kwargs):
    """Call fn with exponential backoff on transient API errors."""
    last: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not _is_retryable(exc):
                raise
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_BACKOFF_BASE * (2 ** attempt))
    # Unreachable: the last iteration re-raises. Kept for linter completeness.
    assert last is not None
    raise last


class ProxyEmbeddings(Embeddings):
    """Embeds text through the selected embedding provider, with a batch->sequential fallback."""

    def __init__(self, model: str, provider: str | None = None):
        self.provider = _resolve_provider(provider)
        self.model = model or default_model_for(self.provider)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = embedding_client(self.provider)
        dims = embedding_dim(self.provider, self.model)
        try:
            resp = retry_call(
                client.embeddings.create,
                model=self.model,
                input=texts,
                dimensions=dims,
            )
            return [d.embedding for d in resp.data]
        except Exception:
            # Some provider models reject list inputs; fall back to one-by-one.
            return [self._embed_one(t, client) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._embed_one(text, embedding_client(self.provider))
        except Exception as exc:
            # Surface a clean, catchable error so the chat endpoint can return
            # a 200 with a user-facing message instead of crashing with a 500.
            raise RuntimeError(f"Embedding failed: {exc}") from exc

    def _embed_one(self, text: str, client: OpenAI) -> list[float]:
        dims = embedding_dim(self.provider, self.model)
        resp = retry_call(
            client.embeddings.create,
            model=self.model,
            input=[text],
            dimensions=dims,
        )
        return resp.data[0].embedding


def rerank(query: str, documents: list[str], model: str = "cohere/rerank-v3.5", top_n: int | None = None) -> list[int]:
    """Rerank `documents` against `query` via OpenRouter's Cohere reranker.

    Calls OpenRouter's `/api/v1/rerank` REST endpoint directly. We avoid the
    OpenAI SDK's `client.rerank(...)` because that method is absent in several
    published `openai` versions (e.g. 2.24.0), and a direct HTTP call is stable
    across SDK releases. Returns document indices ordered by descending
    relevance. Raises on failure so the caller can fall back to vector order.
    """
    if not documents:
        return []
    n = top_n or len(documents)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    url = "https://openrouter.ai/api/v1/rerank"
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(n, len(documents)),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    # Each result has "index" and "relevance_score".
    ordered = sorted(
        results, key=lambda r: r.get("relevance_score", 0) or 0, reverse=True
    )
    return [r["index"] for r in ordered]
