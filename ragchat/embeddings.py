"""Embeddings via the class proxy's OpenAI-compatible /v1/embeddings (PRD T3).

One embedding model per deployment; the config fingerprint (F18) prevents
chunks from different models mixing in one query.
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from openai import OpenAI

from .config import settings

_client: OpenAI | None = None


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
            resp = client.embeddings.create(
                model=self.model, input=texts, dimensions=768
            )
            return [d.embedding for d in resp.data]
        except Exception:
            # Some proxy models reject list inputs; fall back to one-by-one.
            return [self._embed_one(t, client) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text, openai_client())

    def _embed_one(self, text: str, client: OpenAI) -> list[float]:
        resp = client.embeddings.create(
            model=self.model, input=[text], dimensions=768
        )
        return resp.data[0].embedding
