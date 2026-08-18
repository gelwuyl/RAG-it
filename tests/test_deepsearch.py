"""Deep search finds what ranking misses, and stays inside the user's own data.

The claim this feature makes is strong: if the words are in your documents, they
reach the answer. These tests hold it to that — including the CJK case, where a
whitespace tokenizer would silently find nothing and look like "no match" rather
than like a bug.

The other half is scoping. A literal scan over documents is a search engine
pointed at private text, so "only the caller's documents" is not a nicety.

No network, no vector store, no LLM.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

_DB_PATH = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SESSION_SECRET"] = "test-secret"
for _k in ("PG_DATABASE_URL", "rag_gel_DATABASE_URL"):
    os.environ.pop(_k, None)

from ragchat import deepsearch  # noqa: E402

FILLER = "Padding sentence to give the window something to sit inside. " * 6


@pytest.fixture()
def db():
    import ragchat.db as _db

    _db._initialized = False
    _db.init_db()
    s = _db.SessionLocal()
    try:
        yield s
    finally:
        s.close()


def _doc(db, user_id, title, text):
    from ragchat.db import Document

    d = Document(user_id=user_id, source_type="upload", title=title,
                 source_text=text, status="ready", size_bytes=len(text))
    db.add(d)
    db.commit()
    db.refresh(d)
    return d


def _user():
    return f"u-{os.urandom(4).hex()}"


# --- it finds things ------------------------------------------------------

def test_a_literal_string_is_found(db):
    u = _user()
    _doc(db, u, "method.txt", FILLER + "Target average commission per case: not less than S$9,800." + FILLER)
    hits = deepsearch.literal_passages(db, u, "What is the S$9,800 figure?")
    assert hits, "a verbatim string in the document was not found"
    assert "S$9,800" in hits[0]["text"]


def test_the_match_arrives_with_its_context(db):
    """A number on its own tells the model nothing about what it is the price of."""
    u = _user()
    _doc(db, u, "method.txt", FILLER + "Target average commission per case: not less than S$9,800." + FILLER)
    hit = deepsearch.literal_passages(db, u, '"S$9,800"')[0]
    assert "commission per case" in hit["text"]
    assert len(hit["text"]) > 200


def test_chinese_is_searchable(db):
    """A whitespace tokenizer finds nothing here and looks like 'no match'.

    Chinese does not delimit words with spaces, so terms are taken per
    character; the distinct-term scoring is what stops that over-matching.
    """
    u = _user()
    _doc(db, u, "method.txt", FILLER + "周五数据不更新的顾问，下周不分配新线索。" + FILLER)
    hits = deepsearch.literal_passages(db, u, "周五数据不更新会怎样？")
    assert hits
    assert "下周不分配新线索" in hits[0]["text"]


def test_a_quoted_phrase_is_kept_whole(db):
    u = _user()
    _doc(db, u, "a.txt", FILLER + "the Practice Committee convenes within one working day" + FILLER)
    _doc(db, u, "b.txt", FILLER + "the committee practice of convening is described elsewhere" + FILLER)
    hits = deepsearch.literal_passages(db, u, 'find "Practice Committee convenes" please')
    assert hits
    assert hits[0]["doc_id"] and "convenes within one working day" in hits[0]["text"]


def test_more_distinct_terms_outranks_more_repetitions(db):
    """A window with three of the question's words beats one with the same word
    three times — which is what a raw hit count would have preferred."""
    u = _user()
    _doc(db, u, "repeat.txt", FILLER + ("caveat " * 12) + FILLER)
    _doc(db, u, "distinct.txt", FILLER + "a lodged caveat records a transacted price and a date" + FILLER)
    hits = deepsearch.literal_passages(db, u, "caveat transacted price date")
    assert hits[0]["title"] == "distinct.txt"


# --- it stays in its lane -------------------------------------------------

def test_only_the_callers_documents_are_searched(db):
    mine, theirs = _user(), _user()
    _doc(db, theirs, "secret.txt", FILLER + "the passphrase is hunter2" + FILLER)
    _doc(db, mine, "mine.txt", FILLER + "nothing of interest here" + FILLER)
    hits = deepsearch.literal_passages(db, mine, "what is the passphrase")
    assert all(h["title"] != "secret.txt" for h in hits)
    assert not any("hunter2" in h["text"] for h in hits)


def test_a_document_with_no_source_text_is_skipped_not_crashed(db):
    from ragchat.db import Document

    u = _user()
    db.add(Document(user_id=u, source_type="upload", title="empty.txt",
                    source_text=None, status="ready", size_bytes=0))
    db.commit()
    assert deepsearch.literal_passages(db, u, "anything at all") == []


# --- it does not flood the pool -------------------------------------------

def test_stopwords_alone_match_nothing(db):
    """Otherwise every question returns every document and deep search becomes
    a very expensive way to send the model noise."""
    u = _user()
    _doc(db, u, "a.txt", FILLER)
    assert deepsearch.literal_passages(db, u, "what is the of and to") == []


def test_the_number_of_passages_is_capped(db):
    u = _user()
    for i in range(20):
        _doc(db, u, f"d{i}.txt", FILLER + "havenmark appears here" + FILLER)
    hits = deepsearch.literal_passages(db, u, "havenmark")
    assert len(hits) <= deepsearch.MAX_PASSAGES


def test_overlapping_windows_are_merged(db):
    """Two hits a few characters apart are one passage, not two nearly
    identical ones eating two slots in a pool of four."""
    u = _user()
    _doc(db, u, "a.txt", FILLER + "havenmark and havenmark again" + FILLER)
    hits = deepsearch.literal_passages(db, u, "havenmark")
    assert len(hits) == 1


# --- the contract with the pipeline ---------------------------------------

def test_passages_carry_no_similarity_score(db):
    """A literal hit is not a measured distance. Reporting a number would let it
    vote in the not-found decision, which is a cosine judgement it has no part
    in — and pipeline._fallback_score is written for exactly this None."""
    u = _user()
    _doc(db, u, "a.txt", FILLER + "havenmark" + FILLER)
    hit = deepsearch.literal_passages(db, u, "havenmark")[0]
    assert hit["similarity"] is None
    assert hit["deep"] is True
    assert set(hit) >= {"text", "similarity", "doc_id", "title", "ref", "deep"}


def test_a_deep_hit_sorts_without_raising(db):
    """The regression that made BM25-only chunks crash the reranker was exactly
    this: `similarity: None` reaching a sort."""
    from ragchat.pipeline import _fallback_score

    u = _user()
    _doc(db, u, "a.txt", FILLER + "havenmark" + FILLER)
    hits = deepsearch.literal_passages(db, u, "havenmark")
    assert sorted(hits, key=_fallback_score) is not None


def test_the_searcher_closure_binds_one_user(db):
    u1, u2 = _user(), _user()
    _doc(db, u1, "one.txt", FILLER + "distinctivetoken" + FILLER)
    _doc(db, u2, "two.txt", FILLER + "distinctivetoken" + FILLER)
    fn = deepsearch.searcher(db, u1)
    hits = fn("distinctivetoken")
    assert len(hits) == 1 and hits[0]["title"] == "one.txt"
