"""Vercel Python serverless entrypoint for the agentic-RAG backend.

Vercel's @vercel/python runtime serves the ASGI application object named
`app` (or `handler`) exported from this module. We do NOT modify the
FastAPI app defined in ragchat/app.py — we simply import and re-export it
so the existing `app = FastAPI(...)` object is served as a single
serverless function.

The repo root (which contains the `ragchat` package) must be importable.
The @vercel/python runtime adds the `api/` directory to sys.path, not the
project root, so we insert the parent directory explicitly.
"""
import os
import sys

# Make the project root (containing the `ragchat` package) importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import the existing FastAPI ASGI app. This is the only app object the
# backend exposes; it is unchanged by this deployment config.
from ragchat.app import app  # noqa: E402  (FastAPI ASGI app)

# Some serverless adapters look for a symbol named `handler`.
handler = app

# Optionally serve the built Vite SPA (frontend/dist) from the function.
# `/api/*` routes registered on `app` take precedence; anything else falls
# through to these static files when a build is present. When Vercel serves
# frontend/dist as static output instead, this mount is simply unused.
_DIST = os.path.join(_ROOT, "frontend", "dist")
if os.path.isdir(_DIST):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_DIST, html=True), name="spa")
