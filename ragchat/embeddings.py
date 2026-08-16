"""Embeddings via the class proxy's OpenAI-compatible /v1/embeddings (PRD T3).

One embedding model per deployment; the config fingerprint (F18) prevents
chunks from different models mixing in one query.

Resilience: transient 429 (RESOURCE_EXHAUSTED) and 5xx errors are retried
with exponential backoff so a brief Google quota dip doesn't drop a document
or fail a query outright.
"""
from __future__ import annotations

import time

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from .config import settings

_client: OpenAI | None = None

# Retry/backoff for transient 429 (RESOURCE_EXHAUSTED) and 5xx.
_MAX_RETRIES = 4
_BACKOFF_BASE = 1.5  # seconds


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


def openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.api_key, base_url=settings.proxy_base_url)
    return _client


class ProxyEmbeddings(Embeddings):
    """Embeds text through the proxy, with a batch->sequential fallback."""

    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        client = openai_client()
        try:
            resp = retry_call(
                client.embeddings.create,
                model=self.model,
                input=texts,
                dimensions=768,
            )
            return [d.embedding for d in resp.data]
        except Exception:
            # Some proxy models reject list inputs; fall back to one-by-one.
            return [self._embed_one(t, client) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._embed_one(text, openai_client())
        except Exception as exc:
            # Surface a clean, catchable error so the chat endpoint can return
            # a 200 with a user-facing message instead of crashing with a 500.
            raise RuntimeError(f"Embedding failed: {exc}") from exc

    def _embed_one(self, text: str, client: OpenAI) -> list[float]:
        resp = retry_call(
            client.embeddings.create,
            model=self.model,
            input=[text],
            dimensions=768,
        )
        return resp.data[0].embedding
