"""Indexing is sliced across requests, and every slice must survive.

Ingest used to run inside the upload request. On Vercel that request is killed
at 60s and killed BEFORE it commits, so a large document did not index slowly —
it failed and lost the work. Upload now stages the text and returns; the client
drives bounded /index-step calls that each commit.

The dangerous bug in that design is silent: add_chunks numbered chunks from 0
on every call, so without an offset each slice would overwrite the previous one
and a 500-chunk document would end up holding only its final slice. Nothing
would error — retrieval would just quietly miss most of the document. Hence the
contiguity assertions here.

Runs against a temp SQLite DB with the embedder and vector store stubbed.

Run:  .venv/Scripts/python -m pytest tests/test_sliced_ingest.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

_DB_PATH = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SESSION_SECRET"] = "test-secret"
for _k in ("PG_DATABASE_URL", "rag_gel_DATABASE_URL"):
    os.environ.pop(_k, None)

from ragchat import pipeline as P  # noqa: E402


@pytest.fixture()
def captured(monkeypatch):
    """Record every add_chunks call instead of touching a vector store."""
    calls = []

    def _fake_add(user_id, doc_id, title, fingerprint, texts, embeddings, refs,
                  embedding_model="m", start_index=0):
        calls.append({"start": start_index, "texts": list(texts)})

    monkeypatch.setattr(P, "add_chunks", _fake_add)
    monkeypatch.setattr(P, "_embed_texts", lambda model, texts, provider=None: [[0.0] * 8 for _ in texts])
    return calls


class _Cfg:
    chunk_size = 200
    chunk_overlap = 20
    splitter = "recursive"
    embedding_model = "m"
    embedding_provider = "gemini"

    def fingerprint(self):
        return "fp"


def _long_text(n_words=4000):
    return " ".join(f"word{i}" for i in range(n_words))


def test_slices_cover_every_chunk_exactly_once(captured):
    """The whole document must be indexed, with no gaps and no duplicates."""
    text = _long_text()
    cfg = _Cfg()
    total = len(P.plan_chunks(text, "doc", cfg))
    assert total > 3, "test text must span several slices to be meaningful"

    start, guard = 0, 0
    while start < total and guard < 500:
        added, reported = P.ingest_slice("u", "d", "doc", text, cfg, start=start, count=3)
        assert reported == total, "total must not drift between steps"
        if added == 0:
            break
        start += added
        guard += 1

    assert start == total
    # Reconstruct what the store would hold from the recorded calls.
    stored = {}
    for c in captured:
        for i, t in enumerate(c["texts"]):
            stored[c["start"] + i] = t
    assert sorted(stored) == list(range(total)), (
        "chunk indices are not contiguous 0..n-1 — slices overwrote or skipped"
    )


def test_each_slice_is_offset_by_its_start(captured):
    """The offset is the thing that stops slice 2 overwriting slice 1."""
    text = _long_text()
    cfg = _Cfg()
    P.ingest_slice("u", "d", "doc", text, cfg, start=0, count=3)
    P.ingest_slice("u", "d", "doc", text, cfg, start=3, count=3)

    assert [c["start"] for c in captured] == [0, 3]
    # Different content in each slice: if the second re-embedded the first's
    # chunks, the texts would match.
    assert captured[0]["texts"] != captured[1]["texts"]


def test_slice_past_the_end_is_a_no_op(captured):
    """The client loop must terminate rather than spin on an exhausted doc."""
    text = _long_text(200)
    cfg = _Cfg()
    total = len(P.plan_chunks(text, "doc", cfg))
    added, reported = P.ingest_slice("u", "d", "doc", text, cfg, start=total, count=10)
    assert added == 0
    assert reported == total
    assert captured == [], "nothing should be written past the end"


def test_chunk_plan_is_deterministic():
    """Each step re-splits the text, so the plan must be stable — otherwise a
    later slice would index chunks that do not match the earlier ones."""
    text = _long_text()
    cfg = _Cfg()
    a = [c.text for c in P.plan_chunks(text, "doc", cfg)]
    b = [c.text for c in P.plan_chunks(text, "doc", cfg)]
    assert a == b


def test_slice_size_fits_the_function_budget():
    """INGEST_SLICE is a deadline guard, not a tuning knob.

    At the ~30 chunks/sec measured on the default provider, the slice must
    finish well inside the 60s maxDuration — an overrunning step is killed
    before it commits and the client retries the same slice forever.
    """
    assert P.INGEST_SLICE <= 256, "slice too large to reliably fit in 60s"
    assert P.INGEST_SLICE >= 16, "slice so small the round trips dominate"
