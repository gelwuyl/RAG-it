"""Verification of live proxy model discovery + PROVIDER SCOPING.

These tests patch ``ragchat.config._client_for_provider`` — the seam discovery
actually uses. (An earlier version patched ``embeddings.openai_client``, which
``discover_models`` never calls, so every test passed vacuously against the
static fallback catalog and the provider-scoping bug went unnoticed.)

Proves:
1. Chat vs embedding classification from live /v1/models data.
2. Caching is PER PROVIDER — asking for gemini never returns openrouter's list.
3. ``/api/models?provider=`` is honoured for gemini too, not just openrouter.
4. Fallback: when discovery raises, the static per-provider catalog is returned
   and validation still allows the deployment defaults.
5. Gemini exposes exactly one embedding model; the rest belong to OpenRouter.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

import ragchat.config as _cfg


class _FakeModel:
    def __init__(self, mid):
        self.id = mid


class _FakeModelsList:
    def __init__(self, ids):
        self.data = [_FakeModel(i) for i in ids]


class _FakeClient:
    """Stands in for the OpenAI SDK client, per provider."""

    hits = 0

    def __init__(self, models):
        self._models = _FakeModelsList(models)

    @property
    def models(self):
        class _M:
            def __init__(self, owner):
                self._owner = owner

            def list(self):
                _FakeClient.hits += 1
                return self._owner._models

        return _M(self)


def _install_fake(by_provider: dict[str, list[str]]):
    """Patch discovery so each provider returns its own model list."""
    _FakeClient.hits = 0

    def _fake(provider: str):
        return _FakeClient(by_provider.get(provider, []))

    _cfg._client_for_provider = _fake


_REAL_CLIENT_FOR_PROVIDER = _cfg._client_for_provider


@pytest.fixture(autouse=True)
def reset_cache():
    _cfg._cache.clear()
    yield
    _cfg._cache.clear()
    _cfg._client_for_provider = _REAL_CLIENT_FOR_PROVIDER


def test_classification_chat_vs_embedding():
    _install_fake(
        {"gemini": ["deepseek-v4-pro", "qwen3.8-max", "models/gemini-embedding-001"]}
    )
    cat = _cfg.model_catalog("gemini")
    assert "deepseek-v4-pro" in cat["chat"]
    assert "qwen3.8-max" in cat["chat"]
    assert "models/gemini-embedding-001" in cat["embedding"]
    assert "models/gemini-embedding-001" not in cat["chat"]


def test_caching_avoids_repeat_discovery():
    _install_fake({"gemini": ["qwen3.8-max", "models/gemini-embedding-001"]})
    _cfg.model_catalog("gemini")
    _cfg.model_catalog("gemini")
    assert _FakeClient.hits == 1, "expected a single discovery call due to cache"


def test_cache_is_keyed_by_provider():
    """The regression that made every provider show the same six models.

    A single global cache slot meant whichever provider was queried first won,
    so selecting Gemini could return OpenRouter's catalog (and vice versa).
    """
    _install_fake(
        {
            "gemini": ["models/gemini-embedding-001"],
            "openrouter": ["perplexity/pplx-embed-v1-0.6b", "qwen/qwen3-embedding-8b"],
        }
    )
    # Warm OpenRouter FIRST — this is what used to poison the gemini result.
    openrouter = _cfg.model_catalog("openrouter")["embedding"]
    gemini = _cfg.model_catalog("gemini")["embedding"]

    assert gemini == ["models/gemini-embedding-001"]
    assert "qwen/qwen3-embedding-8b" in openrouter
    # No cross-contamination in either direction.
    assert not set(gemini) & set(openrouter)


def test_openrouter_first_does_not_pin_fallback_chat_list():
    """Chat models are NOT provider-scoped — generation always hits Gemini.

    OpenRouter's /v1/models carries no chat models, so discovery for it returns
    an empty chat list. When that empty result fell back to the static list and
    was cached under the provider, selecting OpenRouter embeddings swapped the
    real chat dropdown for a few placeholder names until the TTL expired.
    """
    _install_fake(
        {
            "gemini": ["qwen3.8-max", "deepseek-v4-pro", "models/gemini-embedding-001"],
            "openrouter": ["qwen/qwen3-embedding-8b"],
        }
    )
    # Warm OpenRouter FIRST, cold — the ordering that used to pin the fallback.
    chat_via_openrouter = _cfg.model_catalog("openrouter")["chat"]
    assert "qwen3.8-max" in chat_via_openrouter
    assert chat_via_openrouter == _cfg.model_catalog("gemini")["chat"], (
        "the chat list must not depend on which embedding provider was asked for"
    )


def test_fallback_chat_list_is_never_cached():
    """A throttled discovery call must not pin placeholder names for the TTL.

    discover_models() returns _FALLBACK_CHAT_MODELS when the endpoint is
    unreachable, so a non-empty list is not proof of a live result. Google's
    free tier rate-limits bursts of /v1/models, and caching one throttled reply
    hid the real 51-model catalog behind 3 placeholders until the TTL expired.
    """
    def _boom(provider):
        raise RuntimeError("429 rate limited")

    _cfg._client_for_provider = _boom
    assert _cfg.model_catalog("openrouter")["chat"] == list(_cfg._FALLBACK_CHAT_MODELS)
    assert _cfg._CHAT_KEY not in _cfg._cache, "the fallback must not be cached"

    # Once discovery recovers, the real list must appear immediately.
    _cfg._cache.clear()
    _install_fake({"gemini": ["qwen3.8-max", "models/gemini-embedding-001"]})
    assert "qwen3.8-max" in _cfg.model_catalog("openrouter")["chat"]


def test_legacy_bare_model_id_still_validates():
    """A stored id must not become unsaveable when the allowlist respells it.

    OpenRouter serves both `qwen3-embedding-8b` and `qwen/qwen3-embedding-8b`.
    A config saved under the bare spelling, validated against an allowlist
    holding the prefixed one, failed every save — and since each save re-sends
    the stored model, the Settings dialog could not be saved at all, even for
    an unrelated change like top_k. There was no way out from the UI.
    """
    _install_fake({"openrouter": ["qwen/qwen3-embedding-8b"]})
    assert _cfg.is_known_model("qwen/qwen3-embedding-8b", "embedding", provider="openrouter")
    assert _cfg.is_known_model("qwen3-embedding-8b", "embedding", provider="openrouter"), (
        "the bare spelling of an allowlisted model must remain valid"
    )
    # A genuinely different model is still rejected.
    assert not _cfg.is_known_model("acme/not-a-real-embedder", "embedding", provider="openrouter")


def test_gemini_exposes_exactly_one_embedding_model():
    """Gemini serves one usable embedder; every other option is OpenRouter's."""
    assert _cfg._FALLBACK_EMBEDDING_MODELS["gemini"] == ["models/gemini-embedding-001"]
    assert len(_cfg._FALLBACK_EMBEDDING_MODELS["openrouter"]) > 1


def test_discovery_is_intersected_with_768_allowlist():
    """A model the provider catalogs but that we do not allowlist never reaches the UI.

    The Neon `chunks` table has a single fixed vector(768) column, so a
    different-dimension model would fail at insert time.

    `openai/text-embedding-3-small` is the sharper case of the two: it really is
    served, and it really does honour dimensions=768. It is excluded anyway,
    because the allowlist is a product decision about how many near-identical
    embedders to put in front of someone, not only a capability check. So a
    model being genuinely usable is NOT sufficient to reach the dropdown.
    """
    _install_fake(
        {
            "openrouter": [
                "qwen/qwen3-embedding-8b",
                "openai/text-embedding-3-small",
                "some/unvetted-embedding-model",
            ]
        }
    )
    emb = _cfg.model_catalog("openrouter")["embedding"]
    assert "qwen/qwen3-embedding-8b" in emb
    assert "openai/text-embedding-3-small" not in emb
    assert "some/unvetted-embedding-model" not in emb


def test_fallback_when_discovery_fails():
    def _boom(provider):
        raise RuntimeError("proxy down")

    _cfg._client_for_provider = _boom
    # Falls back to the static per-provider defaults, not an empty/partial list.
    assert _cfg.model_catalog("gemini")["embedding"] == ["models/gemini-embedding-001"]
    assert "qwen3.8-max" in _cfg.model_catalog("gemini")["chat"]
    openrouter = _cfg.model_catalog("openrouter")["embedding"]
    assert "qwen/qwen3-embedding-8b" in openrouter
    # Even on a total discovery outage the lists stay provider-scoped.
    assert "qwen/qwen3-embedding-8b" not in _cfg.model_catalog("gemini")["embedding"]


def test_is_known_allows_live_and_default():
    _install_fake({"gemini": ["custom-llm-x", "models/gemini-embedding-001"]})
    assert _cfg.is_known_model("custom-llm-x", "chat") is True
    assert _cfg.is_known_model(_cfg.settings.default_llm_model, "chat") is True
    assert _cfg.is_known_model(_cfg.settings.default_embedding_model, "embedding") is True
    assert _cfg.is_known_model("totally-bogus-model", "chat") is False


def test_is_known_scopes_embedding_check_to_target_provider():
    """Saving provider+model together must validate against the TARGET provider."""
    _install_fake(
        {
            "gemini": ["models/gemini-embedding-001"],
            "openrouter": ["qwen/qwen3-embedding-8b"],
        }
    )
    assert _cfg.is_known_model("qwen/qwen3-embedding-8b", "embedding", provider="openrouter") is True
    assert _cfg.is_known_model("qwen/qwen3-embedding-8b", "embedding", provider="gemini") is False


def test_is_known_fallback_does_not_reject_default():
    def _boom(provider):
        raise RuntimeError("proxy down")

    _cfg._client_for_provider = _boom
    # Even with discovery down, the deployment defaults must validate so a
    # config save never gets locked out.
    assert _cfg.is_known_model(_cfg.settings.default_llm_model, "chat") is True
    assert _cfg.is_known_model(_cfg.settings.default_embedding_model, "embedding") is True


# --------------------------------------------------------------------------
# Undoing a Settings save
# --------------------------------------------------------------------------


def test_reset_returns_config_yaml_to_being_the_source():
    """A stored override FULLY replaces config.yaml, and nothing could remove
    it — so a deployment could sit on a top_k chosen months earlier while the
    repo said something else, with GET /api/health the only symptom.
    """
    from dataclasses import replace

    from ragchat.config import load_config, save_config_override
    from ragchat.db import clear_config_override, get_config_override

    # Start from a known state: other test modules in this suite save overrides,
    # and reading "shipped" through one would compare the override to itself.
    clear_config_override("pipeline")
    shipped = load_config()
    try:
        save_config_override(replace(shipped, top_k=shipped.top_k + 3))
        assert load_config().top_k == shipped.top_k + 3, "override did not take"

        assert clear_config_override("pipeline") is True
        assert get_config_override("pipeline") is None
        assert load_config().top_k == shipped.top_k, "config.yaml did not come back"

        # Clearing nothing reports nothing, rather than claiming a reset.
        assert clear_config_override("pipeline") is False
    finally:
        clear_config_override("pipeline")
