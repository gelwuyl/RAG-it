"""Known-answer readings: how the ranking rows stop being greyed.

Two mechanisms, one contract — never block, never lie:

1. GOLD MATCH (eval/golden.py): the asked question matched against a bank of
   questions with KNOWN answer-passages. A match yields measured MRR/NDCG/hit
   for that one answer; a non-match yields None and the rows stay honestly
   greyed. Matching strictness is load-bearing — a near-miss against the WRONG
   question would present fiction as ground truth.
2. RANK ESTIMATE: citation marker [n] IS pool index n, so "the passage this
   answer was built from ranked #k of n" is always computable and always
   labelled estimated. It measures ordering only.

And one refusal: a broken retrieval or generation must never be scored as a
correct refusal on a known-unanswerable question — that would flatter the
system exactly when it is broken.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import math  # noqa: E402

import pytest  # noqa: E402

from eval import golden  # noqa: E402
from ragchat import pipeline as _pl  # noqa: E402

GOOD = "Per [1] that is right."


def _make_cfg(**over):
    base = dict(
        chunk_size=512, chunk_overlap=75, splitter="recursive",
        top_k=4, candidate_k=20, similarity_threshold=0.0,
        hybrid_search=False, reranker=False, query_rewrite=False,
        llm_model="stub", router_model="stub-router", temperature=0.0,
        embedding_model="stub-embed", embedding_provider="gemini",
        reranker_provider="gemini", eval_show=True,
    )
    base.update(over)
    from ragchat import config as _cfg
    return _cfg.PipelineConfig(**base)


def _chunk(text="The SunPak 5 ships with a 10-year warranty.", sim=0.62, doc="h"):
    return {"text": text, "similarity": sim, "doc_id": doc,
            "title": f"{doc}.md", "ref": "", "chunk_id": f"{doc}:0"}


GOLD = {
    "question": "What warranty does the SunPak 5 ship with?",
    "unanswerable": False,
    "expected": "A 10-year warranty.",
    "golden_passages": ["The SunPak 5 ships with a 10-year warranty."],
    "golden_doc": "helios_energy_handbook.md",
    "_src": "demo",
    "_idx": 61,
}
UNANS = {
    "question": "How much does the SunPak 5 battery cost?",
    "unanswerable": True,
    "golden_passages": [],
    "_src": "demo",
    "_idx": 63,
}


# ---------- question matching ----------

def test_exact_match_ignores_case_and_punctuation():
    m = golden.match_question("  What WARRANTY does the SunPak 5 ship with?!  ")
    # Asserted by identity, not index — the test must not encode bank layout.
    assert m is not None and m["_src"] == "demo"
    assert m["question"] == "What warranty does the SunPak 5 ship with?"


def test_reworded_question_matches_at_high_ratio():
    m = golden.match_question("What warranty does the SunPak 5 come with?")
    assert m is not None and m["_src"] == "demo"
    assert m["question"] == "What warranty does the SunPak 5 ship with?"


def test_a_different_question_matches_nothing():
    assert golden.match_question("what is the meaning of life") is None
    assert golden.match_question("") is None
    assert golden.match_question("   ") is None


def test_match_fails_open_when_the_bank_cannot_be_read(monkeypatch):
    def _boom():
        raise RuntimeError("disk gone")
    monkeypatch.setattr(golden, "load_bank", _boom)
    assert golden.match_question("anything") is None


# ---------- gold retrieval scoring ----------

def test_scores_measure_the_pool_against_known_passages(monkeypatch):
    # Golden passage verbatim in the SECOND pool chunk: reciprocal rank 1/2,
    # a hit inside k, recall 1.0. NDCG is stubbed — it alone needs embeddings.
    monkeypatch.setattr(_pl, "_golden_ndcg", lambda *a, **k: 0.8123)
    scores = _pl._gold_scores(GOLD, ["unrelated text", GOLD["golden_passages"][0]], _make_cfg())
    assert scores["mrr"] == 0.5
    assert scores["hit_rate_at_k"] == 1
    assert scores["context_recall"] == 1.0
    assert scores["precision_at_k"] == 0.5
    assert scores["ndcg_at_k"] == 0.8123


def test_ndcg_failure_degrades_to_none_not_an_error(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("embedding 429")
    monkeypatch.setattr(_pl, "_golden_ndcg", _boom)
    scores = _pl._gold_scores(GOLD, [GOLD["golden_passages"][0]], _make_cfg())
    assert scores["ndcg_at_k"] is None
    assert scores["mrr"] == 1.0  # the free twins still measured


# ---------- attaching verdicts to the paths _ask cannot score ----------

def test_refusal_on_known_unanswerable_is_the_measured_verdict():
    out = _pl._gold_attach(
        {"answer": "not found", "not_found": True, "citations": []},
        UNANS, True,
    )
    assert out["eval"]["gold"]["refused"] is True
    assert out["eval"]["gold"]["unanswerable"] is True
    assert out["eval"]["pending"] is False


def test_answering_a_known_unanswerable_is_a_miss():
    out = _pl._gold_attach(
        {"answer": "here you go", "not_found": False, "citations": []},
        UNANS, True,
    )
    assert out["eval"]["gold"]["refused"] is False


def test_refusal_on_answerable_is_a_measured_zero():
    out = _pl._gold_attach(
        {"answer": "not found", "not_found": True, "citations": []},
        GOLD, True,
    )
    g = out["eval"]["gold"]
    assert g["mrr"] == 0.0 and g["hit_rate_at_k"] == 0 and g["context_recall"] == 0.0


def test_broken_infrastructure_is_never_a_correct_refusal():
    """A retrieval outage on an unanswerable question must not score as a
    correct refusal — that flatters the system exactly when it is broken."""
    out = _pl._gold_attach(
        {"answer": "couldn't search", "not_found": True, "citations": [],
         "errored": True},
        UNANS, True,
    )
    assert "eval" not in out


def test_no_gold_or_no_eval_show_leaves_results_alone():
    r = {"answer": "a", "not_found": True, "citations": []}
    assert _pl._gold_attach(dict(r), None, True) == r
    assert _pl._gold_attach(dict(r), UNANS, False) == r


def test_existing_gold_is_never_overwritten():
    r = {"answer": "a", "not_found": False, "citations": [],
         "eval": {"gold": {"refused": False}}}
    out = _pl._gold_attach(r, UNANS, True)
    assert out["eval"]["gold"] == {"refused": False}


# ---------- the rank estimate, through the real ask() ----------

@pytest.fixture()
def rig(monkeypatch):
    state = {"pool": [_chunk()], "answers": [GOOD], "generations": 0}
    monkeypatch.setattr(_pl, "retrieve", lambda *a, **k: list(state["pool"]))
    monkeypatch.setattr(_pl, "_eval_answer", lambda *a, **k: None)

    def _chat(model, messages, temperature):
        state["generations"] += 1
        return state["answers"][0]

    monkeypatch.setattr(_pl, "_chat", _chat)
    return state


def test_citation_rank_becomes_the_estimate(rig):
    rig["pool"] = [_chunk("first passage"), _chunk("second passage")]
    rig["answers"] = ["Per [2] that is right."]
    out = _pl.ask("u", "q", [], _make_cfg(), grade=False, use_gold=False)
    assert out["eval"]["cited_rank"] == 2
    assert out["eval"]["pool_n"] == 2
    assert out["eval"]["mrr_est"] == 0.5
    assert out["eval"]["ndcg_est"] == pytest.approx(round(1.0 / math.log2(3), 4))


def test_rank_one_is_a_perfect_estimate(rig):
    rig["answers"] = ["Per [1] that is right."]
    out = _pl.ask("u", "q", [], _make_cfg(), grade=False, use_gold=False)
    assert out["eval"]["mrr_est"] == 1.0
    assert out["eval"]["ndcg_est"] == 1.0


def test_fallback_citations_produce_no_estimate(rig):
    """pool[:2] cited on the model's behalf is a UI courtesy, not evidence the
    reranker ordered well — the estimate must not claim it did."""
    rig["answers"] = ["An answer with no markers at all."]
    out = _pl.ask("u", "q", [], _make_cfg(), grade=False, use_gold=False)
    assert "mrr_est" not in out["eval"]
    assert "cited_rank" not in out["eval"]


# ---------- end to end: the wrapper ----------

def test_matched_unanswerable_refusal_arrives_scored(monkeypatch, rig):
    rig["pool"] = []  # nothing retrieved -> refusal
    monkeypatch.setattr(golden, "match_question", lambda q: UNANS)
    out = _pl.ask("u", "how much does the SunPak 5 cost?", [], _make_cfg(),
                  grade=False, use_gold=True)
    assert out["not_found"] is True
    assert out["eval"]["gold"]["refused"] is True


def test_matched_answerable_attaches_measured_scores(monkeypatch, rig):
    monkeypatch.setattr(golden, "match_question", lambda q: GOLD)
    monkeypatch.setattr(_pl, "_golden_ndcg", lambda *a, **k: 0.9)
    out = _pl.ask("u", "what warranty does the SunPak 5 have?", [], _make_cfg(),
                  grade=False, use_gold=True)
    g = out["eval"]["gold"]
    assert g["mrr"] == 1.0 and g["src"] == "demo" and g["idx"] == GOLD["_idx"]


def test_use_gold_false_skips_matching_even_for_bank_questions(monkeypatch, rig):
    called = {"n": 0}

    def _spy(q):
        called["n"] += 1
        return GOLD

    monkeypatch.setattr(golden, "match_question", _spy)
    out = _pl.ask("u", "q", [], _make_cfg(), grade=False, use_gold=False)
    assert called["n"] == 0
    assert "gold" not in out["eval"]
