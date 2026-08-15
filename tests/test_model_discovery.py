"""Verification of live proxy model discovery (replaces hardcoded
CHAT_MODELS/EMBEDDING_MODELS allowlists).

Runs without the class proxy by monkeypatching the OpenAI client's
models.list(). Proves:
1. Chat vs embedding classification from live /v1/models data.
2. Caching: a second call does not re-hit the (stubbed) proxy.
3. Fallback: when discovery raises, the static default catalog is returned
   and validation still allows the deployment defaults.
4. is_known_model accepts live-catalog models AND the deployment default even
   on a discovery miss (so a transient outage can't lock config saves).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

import ragchat.config as _cfg
import ragchat.embeddings as _emb


class _FakeModel:
    def __init__(self, mid):
        self.id = mid


class _FakeModelsList:
    def __init__(self, ids):
        self.data = [_FakeModel(i) for i in ids]


class _FakeClient:
    hits = 0

    def __init__(self, models):
        self._models = _FakeModelsList(models)

    @property
    def models(self):
        # OpenAI SDK exposes client.models.list()
        class _M:
            def __init__(self, owner):
                self._owner = owner

            def list(self):
                _FakeClient.hits += 1
                return self._owner._models

        return _M(self)


def _install_fake(models):
    _FakeClient.hits = 0
    fake = _FakeClient(models)

    def _fake_client():
        return fake

    _emb.openai_client = _fake_client


@pytest.fixture(autouse=True)
def reset_cache():
    _cfg._cache.update({"chat": None, "embedding": None, "at": 0.0})
    yield
    _cfg._cache.update({"chat": None, "embedding": None, "at": 0.0})


def test_classification_chat_vs_embedding():
    _install_fake(["deepseek-v4-pro", "qwen3.8-max", "text-embedding-005", "gemini-embedding"])
    cat = _cfg.model_catalog()
    assert "deepseek-v4-pro" in cat["chat"]
    assert "qwen3.8-max" in cat["chat"]
    assert "text-embedding-005" in cat["embedding"]
    assert "gemini-embedding" in cat["embedding"]
    assert "text-embedding-005" not in cat["chat"]


def test_caching_avoid_repeat_proxy_call():
    _install_fake(["qwen3.8-max", "text-embedding-005"])
    _cfg.model_catalog()
    _cfg.model_catalog()
    assert _FakeClient.hits == 1, "expected a single discovery call due to cache"


def test_fallback_when_discovery_fails():
    def _boom():
        raise RuntimeError("proxy down")

    _emb.openai_client = _boom
    cat = _cfg.model_catalog()
    # Falls back to the static defaults, not an empty/partial list.
    assert "qwen3.8-max" in cat["chat"]
    assert "text-embedding-005" in cat["embedding"]


def test_is_known_allows_live_and_default():
    _install_fake(["custom-llm-x", "some-embedder-y"])
    # Live catalog model is known.
    assert _cfg.is_known_model("custom-llm-x", "chat") is True
    # Deployment default is known even if not in the (tiny) live list.
    assert _cfg.is_known_model(_cfg.settings.default_llm_model, "chat") is True
    assert _cfg.is_known_model(_cfg.settings.default_embedding_model, "embedding") is True
    # A model neither live nor default is rejected.
    assert _cfg.is_known_model("totally-bogus-model", "chat") is False


def test_is_known_fallback_does_not_reject_default():
    def _boom():
        raise RuntimeError("proxy down")

    _emb.openai_client = _boom
    # Even with discovery down, the deployment defaults must validate so a
    # config save never gets locked out.
    assert _cfg.is_known_model(_cfg.settings.default_llm_model, "chat") is True
    assert _cfg.is_known_model(_cfg.settings.default_embedding_model, "embedding") is True
