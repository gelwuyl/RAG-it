# Deploying agentic-RAG on Vercel

Single serverless function hosting the FastAPI backend, plus the built Vite
SPA, using Neon (Postgres + pgvector) as the vector store.

## Backend (`api/index.py`)
- Entrypoint is `api/index.py`. It does `from ragchat.app import app` and
  exposes that existing ASGI `app` (also aliased as `handler`) for
  `@vercel/python` to serve. The FastAPI app itself is **not** rewritten.
- `@vercel/python` installs dependencies from **`requirements.txt` at the
  repository root** automatically at build time. Keep that file in sync
  with the backend's imports (chroma is pulled in via the `ragchat.store`
  import chain even though only the `neon` backend is used on Vercel).
- Python version is pinned to 3.11 in `vercel.json` (`pythonVersion`).

## Database (Neon pgvector)
- `DATABASE_URL` (and/or `PG_DATABASE_URL`) **must be provided from the
  Neon integration / Environment Variables** in the Vercel project
  dashboard. It is the Neon connection string and is **never** baked into
  the repo. Set `VECTOR_BACKEND=neon` so the app uses pgvector instead of
  the local Chroma dev default.

## Required environment variables (injected at deploy; no secrets in repo)
`GEMINI_API_KEY`, `PROXY_BASE_URL`, `RAG_LLM_MODEL`, `RAG_EMBEDDING_MODEL`,
`VECTOR_BACKEND=neon`, `PG_DATABASE_URL` / `DATABASE_URL`, `SESSION_SECRET`,
`RAG_ALLOWED_ROOT`.

Set these in the Vercel dashboard (**Project → Settings → Environment
Variables**) or via `vercel env add <NAME>` — never in the repo. They are read
at runtime by `os.environ.get(...)` (see `ragchat/config.py` / `embeddings.py`).

## Optional environment variables
- `OPENROUTER_API_KEY` — unlocks the **OpenRouter** provider for embeddings and
  reranking, escaping the Gemini free-tier rate limit (the ~5-document upload
  ceiling). Add it via `vercel env add OPENROUTER_API_KEY` or the dashboard. Once
  present, switch `EMBEDDING_PROVIDER` / `RERANKER_PROVIDER` to `openrouter`
  either here (add them as env vars) or live in the app's Settings UI.
- `EMBEDDING_PROVIDER` — `gemini` (default) or `openrouter`. Boot default only;
  the Settings UI overrides it at runtime (persisted to the DB).
- `RERANKER_PROVIDER` — `gemini` (default, LLM cross-encoder) or `openrouter`
  (Cohere `rerank-v3.5`, fast/cheap). Same switching rules as above.

## Frontend
- `vercel.json` runs `npm install && npm run build` in `frontend/` and
  serves `frontend/dist` as static output. `/api/*` is rewritten to the
  Python serverless function.
