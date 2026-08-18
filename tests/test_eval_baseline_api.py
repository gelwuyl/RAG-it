"""The scorecard's contract with the baseline file.

Two things the UI depends on and nothing else would catch:

  1. `/api/eval/baseline` serves the committed baseline to everyone, guests
     included — it describes the pipeline, not a workspace.
  2. every run payload says which pipeline it measured, so the scorecard can
     refuse to compare a pre-rerank baseline against a full run. That is the
     mistake c002445 fixed inside the harness; without `mode` on the wire it
     would simply move into the browser.

Runs against a temp SQLite DB with no network.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_DB_PATH = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SESSION_SECRET"] = "test-secret"
for _k in ("PG_DATABASE_URL", "rag_gel_DATABASE_URL"):
    os.environ.pop(_k, None)


@pytest.fixture()
def client(monkeypatch):
    from ragchat import app as rapp
    import ragchat.db as _db

    _db._initialized = False
    _db.init_db()
    monkeypatch.setattr("ragchat.vectordb.delete_document_chunks", lambda *a, **k: None)
    with TestClient(rapp.app, raise_server_exceptions=True) as c:
        yield c


def test_anonymous_callers_get_nothing(client):
    client.cookies.clear()
    assert client.get("/api/eval/baseline").status_code in (401, 403)


def test_a_guest_can_read_the_baseline(client):
    # A guest cannot RUN the benchmark, but the scorecard still has to be drawn
    # correctly for the numbers they can see, and the baseline is a repo
    # constant rather than anyone's data.
    assert client.post("/api/auth/guest-login").status_code == 200
    r = client.get("/api/eval/baseline")
    assert r.status_code == 200, r.text
    b = r.json()["baseline"]
    assert b and b["mode"] == "retrieval-pre-rerank"
    assert "context_recall" in b["metrics"]
    assert b["tolerance"] > 0


def test_the_payload_says_which_pipeline_it_measured(client):
    from ragchat.app import _run_payload

    class _Run:
        id, status, error = "r1", "done", None
        total, completed = 10, 10
        results = metrics = config = indexed_files = None
        updated_at = started_at = 0.0
        retrieval_only = True

    r = _Run()
    assert _run_payload(r)["mode"] == "retrieval-pre-rerank"
    r.retrieval_only = False
    assert _run_payload(r)["mode"] == "full"


def test_run_modes_match_the_strings_the_harness_writes():
    """The two sides have to agree or the scorecard silently stops comparing.

    `_run_payload` builds its mode from a boolean; `run_benchmark` builds its
    own from three. A typo in either would not fail anything — the scorecard
    would just quietly fall back to targets forever.
    """
    import inspect

    from eval import run_eval

    src = inspect.getsource(run_eval.run_benchmark)
    for mode in ("retrieval-pre-rerank", "full"):
        assert f'"{mode}"' in src, f"harness no longer emits mode {mode!r}"
