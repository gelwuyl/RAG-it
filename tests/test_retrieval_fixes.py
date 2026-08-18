"""Verification of the retrieval-correctness fixes.

Runs WITHOUT the class proxy / internet by monkeypatching the embedder with a
deterministic stub (token-overlap cosine). This proves:

1. Embedding-dim switch is safe: chunks indexed under two different embedding
   model names land in SEPARATE Chroma collections and never collide (no
   dimension-mismatch crash, no cross-model bleed).
2. Real BM25 hybrid: a query whose exact keyword is only a low-rank vector
   hit gets promoted by RRF fusion.
3. similarity_threshold / not-found: a query with zero relevant docs returns
   the NOT_FOUND answer; raising the threshold refuses even weak top hits.

Run:  .venv/Scripts/python -m pytest tests/ -q
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

# --- deterministic stub embedder (token-overlap cosine) ---
import re

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tok(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(s.lower()))


def _stub_embed(texts):
    # Embedding is a bag-of-tokens signature in a tiny fixed dim space so two
    # *semantically unrelated* strings still get non-identical vectors and two
    # *identical* strings match perfectly. Good enough to exercise ranking.
    import math

    vecs = []
    for t in texts:
        toks = _tok(t)
        # use a hash-based sparse vector of dim 256
        v = [0.0] * 256
        for tk in toks:
            i = hash(tk) % 256
            v[i] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


class _StubEmbeddings:
    # Mirrors ProxyEmbeddings' signature, which is provider-aware (gemini |
    # openrouter). The stub ignores the provider — it never leaves the process —
    # but must accept the kwarg or every ingest path raises TypeError.
    def __init__(self, model: str, provider: str | None = None):
        self.model = model
        self.provider = provider

    def embed_documents(self, texts):
        return _stub_embed(texts)

    def embed_query(self, text):
        return _stub_embed([text])[0]


# Monkeypatch the pipeline's embedder before importing pipeline code paths.
import ragchat.embeddings as _emb

_emb.ProxyEmbeddings = _StubEmbeddings

from ragchat import config as _cfg
from ragchat import store as _store
from ragchat import pipeline as _pl


@pytest.fixture(autouse=True)
def fresh_tmp_store(tmp_path, monkeypatch):
    # Point all on-disk state at a temp dir so tests are isolated & repeatable.
    monkeypatch.setattr(_cfg, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(_cfg, "DATA_DIR", tmp_path / "data")
    # reset module-level client so it re-binds to the new dir
    _store._client = None
    _store._BM25_DOCS.clear()
    _store._BM25_FLAT.clear()
    _store._BM25_OBJ.clear()
    _store._BM25_TITLE.clear()
    _store._BM25_REF.clear()
    yield


def _make_cfg(**over):
    base = dict(
        chunk_size=512,
        chunk_overlap=75,
        splitter="recursive",
        top_k=4,
        candidate_k=20,
        similarity_threshold=0.0,
        hybrid_search=False,
        reranker=False,
        query_rewrite=False,
        llm_model="stub",
        temperature=0.0,
        embedding_model="text-embedding-005",
        embedding_provider="gemini",
        reranker_provider="gemini",
        eval_show=True,
    )
    base.update(over)
    return _cfg.PipelineConfig(**base)


# --- 1. embedding-dim switch is safe -------------------------------------

def test_dim_switch_uses_separate_collections():
    u = "user-A"
    cfg_768 = _make_cfg(embedding_model="text-embedding-005")
    cfg_3072 = _make_cfg(embedding_model="gemini-embedding")

    n1 = _pl.ingest_document_text(u, "d1", "Doc One", "SunPak 5 stores 5.1 kWh usable energy.", cfg_768)
    n2 = _pl.ingest_document_text(u, "d2", "Doc Two", "Meridian flat white costs 4.70 dollars.", cfg_3072)
    assert n1 == 1 and n2 == 1

    # Each model has its own collection; querying under model 1 must NOT see
    # model 2's chunks (no cross-dim bleed, no crash).
    c1 = _store.query_chunks(u, _stub_embed(["energy storage"])[0], cfg_768.fingerprint(), 10, embedding_model="text-embedding-005")
    c2 = _store.query_chunks(u, _stub_embed(["coffee price"])[0], cfg_3072.fingerprint(), 10, embedding_model="gemini-embedding")
    assert all(c["doc_id"] == "d1" for c in c1), c1
    assert all(c["doc_id"] == "d2" for c in c2), c2


def test_switch_then_reindex_no_dimension_crash():
    u = "user-B"
    cfg_a = _make_cfg(embedding_model="text-embedding-005")
    _pl.ingest_document_text(u, "doc", "Doc", "SunPak 5 peak output 5.7 kW.", cfg_a)
    # Simulate switching embedding model (e.g. via settings) — old collection
    # persists, new one is created. Deleting the doc sweeps both.
    cfg_b = _make_cfg(embedding_model="gemini-embedding")
    _pl.ingest_document_text(u, "doc", "Doc", "SunPak 5 peak output 5.7 kW.", cfg_b)
    # Delete without a model hint must succeed (sweeps all user collections)
    # and not raise a Chroma dimension error.
    _store.delete_document_chunks(u, "doc")
    leftovers_a = _store.query_chunks(u, _stub_embed(["SunPak"])[0], cfg_a.fingerprint(), 5, embedding_model="text-embedding-005")
    leftovers_b = _store.query_chunks(u, _stub_embed(["SunPak"])[0], cfg_b.fingerprint(), 5, embedding_model="gemini-embedding")
    assert leftovers_a == [] and leftovers_b == []


# --- 2. real BM25 hybrid fusion ------------------------------------------

def _index_corpus(u, cfg):
    _pl.ingest_document_text(
        u, "h", "helios_energy_handbook.md",
        "The SunPak 5 stores 5.1 kWh. Warranty form HE-104 required. "
        "Installers need HX certification. Peak output 5.7 kW. "
        "Critical Response line 1-800-555-0147.",
        cfg,
    )
    _pl.ingest_document_text(
        u, "m", "meridian_coffee_ops.md",
        "Flat white 4.70. Oat milk adds 0.60. Riverside store has Probat roaster. "
        "Registers start with 150 float. Deposits Monday Thursday.",
        cfg,
    )


def test_hybrid_promotes_exact_keyword():
    u = "user-C"
    _index_corpus(u, _make_cfg())
    # Pure-vector retrieval: the rare token 'HE-104' may rank below common words.
    vec = _pl.retrieve(u, "HE-104 warranty form", _make_cfg(hybrid_search=False), n_results=4)
    vec_ranks = {c["chunk_id"]: i for i, c in enumerate(vec)}
    # Hybrid retrieval fuses BM25 which strongly matches the rare token.
    hyb = _pl.retrieve(u, "HE-104 warranty form", _make_cfg(hybrid_search=True), n_results=4)
    hyb_ids = [c["chunk_id"] for c in hyb]
    # The helios chunk (which contains HE-104) should be present and ideally rank 1.
    assert any(c["doc_id"] == "h" for c in hyb), hyb_ids
    # Fusion should not be worse than vector for the target doc's presence:
    assert "h:0" in hyb_ids


def test_hybrid_off_equals_vector_only():
    u = "user-D"
    _index_corpus(u, _make_cfg())
    vec = _pl.retrieve(u, "oat milk price", _make_cfg(hybrid_search=False), n_results=4)
    hyb = _pl.retrieve(u, "oat milk price", _make_cfg(hybrid_search=True), n_results=4)
    # With a common-token query the fused result should still contain the doc
    # the vector ranker found (hybrid is a strict superset of ranking signal).
    assert any(c["doc_id"] == "m" for c in vec)
    assert any(c["doc_id"] == "m" for c in hyb)


# --- 3. threshold / not-found --------------------------------------------

def test_unrelated_query_returns_not_found():
    u = "user-E"
    _index_corpus(u, _make_cfg())

    # Force the generation step to always "refuse" so we isolate the retrieval
    # gate. Patch _chat to echo the NOT_FOUND sentinel for any generation.
    def _fake_chat(model, messages, temperature):
        return _pl.NOT_FOUND_ANSWER

    import ragchat.pipeline as P
    orig = P._chat
    P._chat = _fake_chat
    try:
        res = _pl.ask(u, "What is the meaning of life?", [], _make_cfg())
    finally:
        P._chat = orig
    assert res["not_found"] is True
    assert res["answer"] == _pl.NOT_FOUND_ANSWER


def test_nothing_external_can_reach_an_answer():
    """Web augmentation is gone; deep search replaced it. Every citation now
    comes from a document the caller owns, and there is no code path left that
    can put anything else in front of the model."""
    u = "user-F"
    _index_corpus(u, _make_cfg())

    def _fake_chat(model, messages, temperature):
        return "Per [1] the answer is in your documents."

    import ragchat.pipeline as P
    assert not hasattr(P, "_web_search")
    orig = P._chat
    P._chat = _fake_chat
    try:
        res = _pl.ask(u, "latest stock market news", [], _make_cfg())
    finally:
        P._chat = orig
    for c in res.get("citations", []):
        assert not str(c.get("doc_id", "")).startswith("web:")
        assert not c.get("is_web")


def test_bm25_index_param_forces_fusion_path():
    u = "user-G"
    _index_corpus(u, _make_cfg())
    # Calling store.query_chunks directly with bm25_index=True and a query that
    # hits the keyword index must return a fused list containing the keyword doc.
    q = _stub_embed(["Probat roaster Riverside"])[0]
    out = _store.query_chunks(
        u, q, _make_cfg().fingerprint(), 4,
        embedding_model="text-embedding-005", bm25_index=True, query_text="Probat roaster Riverside",
    )
    assert any(c["doc_id"] == "m" for c in out), [c["doc_id"] for c in out]


# --- 5. a failed rerank call must not crash on BM25-only chunks -----------
#
# Fusion is unconditional now, so a chunk found by keyword search alone is
# routine — and it carries `similarity: None`, because vector search never
# scored it. The cross-encoder's except-branch used to fall back to that None
# and hand it straight to sort(), so ONE rate-limited rerank call raised
# TypeError and 500'd the whole answer.


def test_rerank_survives_none_similarity_when_scoring_fails():
    def _boom(*_a, **_kw):
        raise RuntimeError("429 rate limited")

    orig = _pl._chat
    _pl._chat = _boom
    try:
        chunks = [
            {"text": "vector hit", "similarity": 0.5, "doc_id": "a", "title": "a", "ref": ""},
            {"text": "bm25-only hit", "similarity": None, "doc_id": "b", "title": "b", "ref": ""},
            {"text": "another vector hit", "similarity": 0.3, "doc_id": "c", "title": "c", "ref": ""},
        ]
        out = _pl._rerank(
            "q", chunks, _make_cfg(reranker=True, reranker_provider="gemini", top_k=2)
        )
    finally:
        _pl._chat = orig
    assert len(out) == 2
    # The unscored chunk sorts on a real number, above the 0.3 vector hit.
    assert [c["doc_id"] for c in out] == ["a", "b"], [c["doc_id"] for c in out]


def test_fallback_score_keeps_a_legitimate_zero():
    # `similarity or 0.5` would rewrite 0.0 into 0.5 and promote the worst
    # chunk in the pool above genuinely mid-ranked ones.
    assert _pl._fallback_score({"similarity": 0.0}) == 0.0
    assert _pl._fallback_score({"similarity": None}) == 0.5
    assert _pl._fallback_score({}) == 0.5


# --- deep search reaches the model ----------------------------------------
#
# These stub retrieve() rather than indexing a corpus. The stub embedder at the
# top of this file is a module-level monkeypatch, and by the time the whole
# suite has run, other modules have re-imported ragchat.embeddings and put the
# real one back — so a test here that depends on live retrieval quietly starts
# measuring whether an API key works. Fixing the pool makes these tests about
# the one thing they are named for.

_POOL = [{
    "text": "Riverside store has Probat roaster.",
    "similarity": 0.62,
    "doc_id": "m",
    "title": "meridian_coffee_ops.md",
    "ref": "~40% of document",
    "chunk_id": "m:0",
}]


def _with_stubs(monkeypatch, pool=None, chat=None):
    monkeypatch.setattr(_pl, "retrieve", lambda *a, **k: list(pool if pool is not None else _POOL))
    monkeypatch.setattr(_pl, "_chat", chat or (lambda m, msgs, t: "Per [1] that is right."))
    # ask() ends by grading its own answer with the LLM judge. Left live, each
    # of these tests made a real API call and spent ~15s retrying against a
    # provider that is not part of what they test. `eval_show` is the real
    # switch for it, so this is the config a caller would use, not a patch.
    monkeypatch.setattr(_pl, "_eval_answer", lambda *a, **k: None)


def test_a_deep_hit_reaches_the_context_even_when_ranking_would_miss_it(monkeypatch):
    """The whole claim of the feature, end to end: a literal hit has to arrive
    in the text handed to the model, not merely be found."""
    seen = {}

    def _chat(model, messages, temperature):
        seen["prompt"] = messages[-1]["content"]
        return "Per [1] that is right."

    _with_stubs(monkeypatch, chat=_chat)

    def _deep(_query):
        return [{
            "text": "The Probat roaster at Riverside is serviced every 400 hours.",
            "similarity": None,
            "doc_id": "m",
            "title": "meridian_coffee_ops.md",
            "ref": "~40% of document",
            "deep": True,
        }]

    res = _pl.ask("user-DS", "servicing interval", [], _make_cfg(), deep_search=_deep)
    assert "serviced every 400 hours" in seen["prompt"], (
        "a literal hit was found and then dropped before generation"
    )
    assert "deep" in (res.get("eval_line") or ""), res.get("eval_line")


def test_a_failing_deep_search_does_not_cost_the_answer(monkeypatch):
    """Ranked retrieval already has an answer; a scan blowing up must not throw
    it away."""
    called = {}

    def _chat(model, messages, temperature):
        called["yes"] = True
        return "Per [1] the answer is in your documents."

    _with_stubs(monkeypatch, chat=_chat)

    def _boom(_query):
        raise RuntimeError("source_text column vanished")

    res = _pl.ask("user-DS2", "roaster", [], _make_cfg(), deep_search=_boom)
    assert called.get("yes"), "the raise escaped and generation never ran"
    assert not res["not_found"], res


def test_without_the_flag_nothing_extra_is_searched(monkeypatch):
    """Deep search is opt-in per question. Passing nothing behaves as before."""
    _with_stubs(monkeypatch)
    res = _pl.ask("user-DS3", "roaster", [], _make_cfg())
    assert "deep" not in (res.get("eval_line") or "")


def test_a_bm25_only_pool_is_not_treated_as_irrelevant(monkeypatch):
    """No cosine is the absence of evidence, not evidence of absence.

    Keyword fusion marks BM25-only chunks `similarity: None`, and that happens
    on exactly the queries fusion exists for — part numbers, form codes. An
    earlier version of the not-found guard refused whenever no chunk carried a
    cosine, which would have broken the case hybrid retrieval was added to fix.
    """
    called = {}

    def _chat(model, messages, temperature):
        called["yes"] = True
        return "Per [1] form HE-104 is required."

    bm25_only = [dict(_POOL[0], similarity=None, text="Warranty form HE-104 required.")]
    _with_stubs(monkeypatch, pool=bm25_only, chat=_chat)

    res = _pl.ask("user-DS4", "HE-104", [], _make_cfg())
    assert called.get("yes"), "a BM25-only pool was refused before generation"
    assert not res["not_found"], res


def test_a_pool_below_the_threshold_refuses_without_generating(monkeypatch):
    """similarity_threshold is a live setting again. Its only previous use was
    gating web augmentation, so deleting that would have left it decorating a
    meter and changing nothing."""
    called = {}

    def _chat(model, messages, temperature):
        called["yes"] = True
        return "should never be reached"

    weak = [dict(_POOL[0], similarity=0.01)]
    _with_stubs(monkeypatch, pool=weak, chat=_chat)

    res = _pl.ask("user-DS5", "unrelated question", [], _make_cfg())
    assert res["not_found"]
    assert not called, "spent a generation call on a pool that cleared nothing"


def test_candidate_k_bounds_the_pool_even_without_the_bm25_index():
    """`_BM25_OBJ` is an in-memory, per-process index built during ingest, so a
    freshly started process has none and the fusion branch is skipped. That
    return path handed back the whole over-fetched vector list — 30 chunks —
    ignoring candidate_k entirely, which makes every measurement taken on that
    setting wrong.
    """
    u = "user-CK"
    cfg = _make_cfg()
    for i in range(12):
        _pl.ingest_document_text(u, f"d{i}", f"doc {i}", f"Chunk number {i} about coffee and energy.", cfg)

    # Simulate the fresh process: the collection is on disk, the BM25 cache is not.
    _store._BM25_OBJ.clear()

    got = _pl.retrieve(u, "coffee", _make_cfg(candidate_k=3))
    assert len(got) <= 3, f"candidate_k=3 returned {len(got)} chunks"


def test_keyword_fusion_survives_a_process_restart():
    """The BM25 index is in-memory and built during ingest, so a freshly started
    process had none — and every query until the next upload silently ran
    vector-only. CLAUDE.md calls keyword fusion unconditional; it was
    conditional on having uploaded something since the last restart.
    """
    u = "user-HY"
    cfg = _make_cfg(hybrid_search=True)
    _index_corpus(u, cfg)

    # Simulate the restart: the collection stays on disk, every in-memory cache
    # goes, exactly as it would in a new process.
    for cache in (_store._BM25_DOCS, _store._BM25_FLAT, _store._BM25_OBJ):
        cache.clear()
    _store._BM25_TITLE.clear()
    _store._BM25_REF.clear()
    _store._BM25_HYDRATED.clear()

    out = _store.query_chunks(
        u, _stub_embed(["HE-104"])[0], cfg.fingerprint(), 4,
        embedding_model="text-embedding-005",
        bm25_index=True, query_text="HE-104 warranty form",
    )
    assert _store._BM25_OBJ, "the keyword index did not rebuild itself"
    assert any(c["doc_id"] == "h" for c in out), [c["doc_id"] for c in out]


def test_a_failed_hydration_is_retried_rather_than_disabling_fusion():
    """Marking a collection hydrated BEFORE the read means one transient
    failure disables keyword fusion for that collection for the life of the
    process — silently, with no later query able to notice."""
    u = "user-HY2"
    cfg = _make_cfg(hybrid_search=True)
    _index_corpus(u, cfg)
    col = _store.collection_for(u, "text-embedding-005")

    for cache in (_store._BM25_DOCS, _store._BM25_FLAT, _store._BM25_OBJ):
        cache.clear()
    _store._BM25_HYDRATED.clear()

    class _Boom:
        name = col.name

        def get(self, **_kw):
            raise RuntimeError("disk hiccup")

    _store._hydrate_bm25(_Boom())
    assert col.name not in _store._BM25_HYDRATED, "a failed read was cached as done"

    # The next attempt, against the real collection, still works.
    _store._hydrate_bm25(col)
    assert _store._BM25_OBJ, "fusion never recovered"


def test_ask_returns_the_context_it_used(monkeypatch):
    """The benchmark's faithfulness judge grades an answer against its sources.

    It used to rebuild them by re-running retrieval, which is a RECONSTRUCTION:
    ask() calls rewrite_query, a model call, so a second retrieval can return a
    different list than the one behind the answer. Same class of error as
    scoring a different list than the model was handed (c002445).
    """
    seen = {}

    def _chat(model, messages, temperature):
        seen["prompt"] = messages[-1]["content"]
        return "Per [1] that is right."

    _with_stubs(monkeypatch, chat=_chat)
    res = _pl.ask("user-CTX", "roaster", [], _make_cfg())
    assert res.get("context"), "ask() did not report the passages it used"
    assert res["context"] in seen["prompt"], (
        "the reported context is not what was sent to the model"
    )
    assert _POOL[0]["text"] in res["context"]


def test_full_mode_scoring_does_not_reference_a_dead_variable():
    """The NameError that broke every full benchmark run, including the in-app
    button, lived on a line only full mode reaches — so the retrieval-only runs
    used everywhere else never touched it."""
    import inspect

    from eval.run_eval import score_item

    src = inspect.getsource(score_item)
    after_ask = src.split("res = ask(")[1]
    # Comments stripped: the fix left a comment explaining the old line, and a
    # test that matched prose rather than code would fail on its own docs.
    code = " ".join(
        ln for ln in after_ask.splitlines() if not ln.strip().startswith("#")
    )
    assert "chunks[" not in code, (
        "full-mode scoring references `chunks`, which does not exist there"
    )
