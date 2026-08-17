"""A benchmark run belongs to whoever ran it, and to nobody else.

`_active_run` used to return the globally most recent `eval_runs` row with no
owner filter. Every read of /api/eval therefore served the same run to every
caller, so an anonymous guest opening the app was shown the deployment owner's
scorecard — and `results`, which carries each golden-set question together with
the answer generated for it. Nothing errored; the leak was silent and looked
exactly like a working feature.

Write access was already closed (test_guest_permissions.py). This is the read
side, which is the half that actually exposed data.

Runs against a temp SQLite DB with no network.

Run:  .venv/Scripts/python -m pytest tests/test_eval_scoping.py -q
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from fastapi.testclient import TestClient

_DB_PATH = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SESSION_SECRET"] = "test-secret"
for _k in ("PG_DATABASE_URL", "rag_gel_DATABASE_URL"):
    os.environ.pop(_k, None)

SECRET_Q = "What is KFD's core methodology in 9 Chinese characters?"


@pytest.fixture()
def client(monkeypatch):
    from ragchat import app as rapp
    import ragchat.db as _db

    _db._initialized = False
    _db.init_db()
    monkeypatch.setattr("ragchat.vectordb.delete_document_chunks", lambda *a, **k: None)
    with TestClient(rapp.app, raise_server_exceptions=True) as c:
        yield c


def _as_guest(client: TestClient) -> str:
    r = client.post("/api/auth/guest-login")
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _as_account(client: TestClient) -> str:
    r = client.post(
        "/api/auth/register",
        json={"username": f"real-{os.urandom(4).hex()}", "password": "pw-12345678"},
    )
    assert r.status_code == 200, r.text
    return r.json().get("id") or client.get("/api/auth/status").json()["user"]["id"]


def _plant_run(user_id, *, question=SECRET_Q, metrics=None):
    """Write a finished run straight to the DB, as start_eval/step_eval would.

    Going through the API would need the golden set, an embedder and two LLMs;
    the ownership filter is what is under test, not how the row got there.
    """
    from ragchat.db import EvalRun, SessionLocal

    s = SessionLocal()
    try:
        run = EvalRun(
            user_id=user_id,
            status="done",
            total=1,
            completed=1,
            results=json.dumps([{"question": question, "answer": "planted"}]),
            metrics=json.dumps(metrics or {"faithfulness": 1.0}),
            started_at=time.time(),
            updated_at=time.time(),
        )
        s.add(run)
        s.commit()
        return run.id
    finally:
        s.close()


def _body(client: TestClient) -> str:
    r = client.get("/api/eval")
    assert r.status_code == 200, r.text
    return r.text


# --------------------------------------------------------------------------
# The leak itself
# --------------------------------------------------------------------------


def test_guest_never_sees_an_account_holders_run(client):
    """The exact reported bug: a guest was served the owner's scorecard."""
    owner = _as_account(client)
    _plant_run(owner)

    client.cookies.clear()
    _as_guest(client)
    body = _body(client)

    assert SECRET_Q not in body, "golden-set question leaked into a guest response"
    payload = json.loads(body)
    assert payload["status"] == "none"
    assert not payload.get("metrics")


def test_one_account_never_sees_another_accounts_run(client):
    """Scoping has to hold between signed-in users too, not just guest vs owner."""
    a = _as_account(client)
    _plant_run(a, question="account A private question")

    client.cookies.clear()
    _as_account(client)  # a different registration -> a different user
    body = _body(client)

    assert "account A private question" not in body
    assert json.loads(body)["status"] == "none"


def test_rows_predating_the_owner_column_belong_to_nobody(client):
    """`user_id` is nullable so _reconcile_columns can add it to a live DB.

    A NULL owner must therefore be invisible, not universally visible — the
    migration must not leave the old leak in place for pre-existing rows.
    """
    _plant_run(None, question="legacy unowned run")

    _as_account(client)
    assert "legacy unowned run" not in _body(client)

    client.cookies.clear()
    _as_guest(client)
    assert "legacy unowned run" not in _body(client)


# --------------------------------------------------------------------------
# The owner still sees their own work
# --------------------------------------------------------------------------


def test_owner_still_sees_their_own_run(client):
    """Over-correcting would be its own bug: scoping must not hide the run
    from the person who paid for it."""
    owner = _as_account(client)
    _plant_run(owner)
    payload = json.loads(_body(client))
    assert payload["status"] == "done"
    assert payload["metrics"]["faithfulness"] == 1.0
    assert SECRET_Q in json.dumps(payload["results"])


def test_the_latest_of_the_owners_own_runs_wins(client):
    """Ordering still applies — but within the caller's own rows."""
    owner = _as_account(client)
    _plant_run(owner, question="older", metrics={"faithfulness": 0.1})
    time.sleep(0.01)
    _plant_run(owner, question="newer", metrics={"faithfulness": 0.9})
    payload = json.loads(_body(client))
    assert payload["metrics"]["faithfulness"] == 0.9


# --------------------------------------------------------------------------
# What the UI needs to render the right empty state
# --------------------------------------------------------------------------


def test_guest_response_is_flagged_locked(client):
    """A guest is not looking at an empty result, they are looking at a feature
    that is not theirs to run. The UI needs to tell those apart to show a
    sign-in prompt instead of "No benchmark run yet"."""
    _as_guest(client)
    assert json.loads(_body(client))["locked"] is True


def test_account_response_is_not_flagged_locked(client):
    _as_account(client)
    assert json.loads(_body(client)).get("locked") is not True
