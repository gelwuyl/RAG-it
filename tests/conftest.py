"""Suite-wide defaults.

The password sign-in routes (`/api/auth/register`, `/login`, `/local-login`)
404 wherever Google OAuth is configured — see ragchat.app._dev_auth_only. Half
a dozen test files use those routes to create an account, and
`ragchat.config` calls `load_dotenv()` at import, so a developer with real
`GOOGLE_CLIENT_ID` in their `.env` would watch those tests fail for a reason
that has nothing to do with what they test.

Popping the environment variable does not work: the next module to import
ragchat.config puts it straight back. Patching the predicate does.

Whether a test passes must not depend on which keys the person running it
happens to own.

`tests/test_local_login_closed.py` is the file that cares about the real
behaviour, and re-patches this per test in both directions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _password_auth_available(monkeypatch):
    try:
        from ragchat import auth as authn
    except Exception:  # pragma: no cover - import-time failures surface elsewhere
        return
    monkeypatch.setattr(authn, "oauth_configured", lambda: False, raising=False)
