"""A rate-limited batch must not become N rate-limited requests.

ProxyEmbeddings.embed_documents sends one batch request, and falls back to
embedding texts one at a time if that fails. The fallback exists for providers
that reject LIST input — a permanent input-shape error.

Firing it on a TRANSIENT error inverts its purpose. Gemini's free tier allows
100 embedding requests per minute; a batch of 200 chunks that got a 429 used to
turn into 200 further requests against a quota that had just run out. Observed
live while tuning retrieval presets.

No network: the client is stubbed.

Run:  .venv/Scripts/python -m pytest tests/test_embedding_backoff.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from ragchat.embeddings import ProxyEmbeddings, _is_retryable


class _RateLimit(Exception):
    status_code = 429


class _BadInputShape(Exception):
    """What a provider that rejects list input actually returns: a 400."""
    status_code = 400


class _Resp:
    def __init__(self, vectors):
        self.data = [type("D", (), {"embedding": v})() for v in vectors]


class _Client:
    """Counts calls and fails the batch in a configurable way."""

    def __init__(self, batch_error):
        self.batch_error = batch_error
        self.calls = []
        outer = self

        class _Emb:
            def create(self, model, input, dimensions=None, **kw):
                outer.calls.append(list(input))
                if len(input) > 1 and outer.batch_error is not None:
                    raise outer.batch_error
                return _Resp([[0.1] * 3 for _ in input])

        self.embeddings = _Emb()


@pytest.fixture()
def no_backoff(monkeypatch):
    """retry_call sleeps between attempts; nothing here needs real waiting."""
    monkeypatch.setattr("ragchat.embeddings.time.sleep", lambda *_: None)


def _patch_client(monkeypatch, client):
    monkeypatch.setattr("ragchat.embeddings.embedding_client", lambda *_a, **_k: client)
    monkeypatch.setattr("ragchat.embeddings.embedding_dim", lambda *_a, **_k: 768)


def test_rate_limited_batch_does_not_fan_out(monkeypatch, no_backoff):
    """The regression. 200 texts must not become 200 requests on a 429."""
    client = _Client(_RateLimit("quota exceeded"))
    _patch_client(monkeypatch, client)
    texts = [f"chunk {i}" for i in range(200)]

    with pytest.raises(Exception):
        ProxyEmbeddings("m", provider="gemini").embed_documents(texts)

    single = [c for c in client.calls if len(c) == 1]
    assert not single, (
        f"fell back to {len(single)} individual requests after a rate limit; "
        "that is a stampede against an exhausted quota, not a retry"
    )
    # Only the batch, retried by retry_call's own backoff.
    assert all(len(c) == 200 for c in client.calls)


def test_input_shape_rejection_still_falls_back(monkeypatch, no_backoff):
    """The fallback must survive for the case it was written for."""
    client = _Client(_BadInputShape("input must be a string"))
    _patch_client(monkeypatch, client)
    texts = ["a", "b", "c"]

    out = ProxyEmbeddings("m", provider="gemini").embed_documents(texts)

    assert len(out) == 3
    assert [c for c in client.calls if len(c) == 1], "one-by-one fallback did not run"


def test_healthy_batch_is_a_single_request(monkeypatch, no_backoff):
    client = _Client(None)
    _patch_client(monkeypatch, client)

    out = ProxyEmbeddings("m", provider="gemini").embed_documents(["a", "b", "c"])

    assert len(out) == 3
    assert len(client.calls) == 1, "batching should cost exactly one request"


def test_rate_limit_is_classified_retryable():
    """The fix keys off this; if it stopped being true the guard would invert
    and the fallback would fire on quota errors again."""
    assert _is_retryable(_RateLimit()) is True
    assert _is_retryable(_BadInputShape()) is False
