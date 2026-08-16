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
- `vercel.json` sets `functions."api/index.py"` to `memory: 1024` and
  `maxDuration: 60`. The benchmark needs the longer duration; the default
  (10s on Hobby) is not enough for even a single scored question.

## Why the benchmark runs in slices
The Evaluation tab's Run button does **not** start a background job. On Vercel
that cannot work:

1. The repo directory (`/var/task`) is **read-only**, so writing a status file
   raises `EROFS`.
2. The function is **frozen the moment the response is sent** — a
   `threading.Thread` stops executing.
3. A later poll may hit a **different instance**, with no shared memory or
   `/tmp`.
4. Scoring 46 golden questions x 3 LLM judges takes **minutes**, far past
   `maxDuration`.

Instead the run is client-driven and stateful in Postgres:

- `POST /api/eval/run` creates a row in `eval_runs` and returns immediately.
- `POST /api/eval/step` performs **one bounded slice** — index one corpus file,
  or score `EVAL_BATCH_DEFAULT` questions — commits it, and returns progress.
- The browser calls `/step` until `status != "running"`. Closing the tab pauses
  the run; reopening the app resumes from the last committed slice.
- `GET /api/eval` returns the current (possibly partial) scorecard.

This is also why progress and partial results appear as the run proceeds.

**Slice size is latency-bound, not a preference.** Measured locally 2026-08-17,
one scored question costs **40–54s** (generation + up to three judge calls);
indexing a corpus file costs under 9s. `EVAL_BATCH_DEFAULT` is therefore **1** —
a batch of 2 measured 83s and blew the 60s `maxDuration`. That failure is silent
and self-perpetuating: an overrunning step is killed *before* it commits, so the
client retries the identical slice forever and the run never advances past
scoring. If judge latency grows, slice the judges across steps rather than
raising the batch — there is only ~6s of headroom left at 1.

## Diagnosing a deploy without the dashboard
`GET /api/health` reports the **effective** runtime config — not just which env
vars are set. Check `effective_config.llm_model`, `judge.model` and
`embedding_models_by_provider` there first: the Settings UI persists overrides
to the database, so env vars are only boot defaults and are frequently *not*
what the app is actually using.

**Open it immediately after every deploy** and confirm three things. All three
held locally on 2026-08-17; a mismatch on Vercel means an env var there is
overriding what you expect, which is otherwise invisible without dashboard
access:

1. `judge.model` **==** `effective_config.llm_model`. When these diverge the
   judge is grading with a model the app isn't answering with — and if the
   judge's id isn't served, every judge call 404s and the whole scorecard comes
   back ungraded.
2. `embedding_models_by_provider.gemini` has **exactly one** entry
   (`models/gemini-embedding-001`). More means the 768 allowlist was bypassed,
   and a non-768 model reaching the UI fails at insert against the fixed
   `vector(768)` column.
3. `effective_config.embedding_provider` is the provider you actually intend.
   It comes from the DB override, not the env var.

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
