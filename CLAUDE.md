# agentic-RAG — project notes for Claude

A document-grounded RAG chat app. FastAPI backend + vanilla-JS frontend,
deployed as a **single Vercel serverless function** with **Neon (Postgres +
pgvector)** as both the app DB and the vector store.

## Layout

```
api/index.py        Vercel entrypoint — re-exports ragchat.app:app as an ASGI handler
ragchat/app.py      FastAPI routes (auth, sources, chats, config, eval)
ragchat/config.py   Settings (env) + PipelineConfig (hot-reloaded) + model discovery
ragchat/pipeline.py ingest + ask (rewrite → retrieve → rerank → generate → judge)
ragchat/embeddings.py  Provider-aware embedding/rerank clients (gemini | openrouter)
ragchat/vectordb.py Dispatch to store.py (Chroma, local) or store_neon.py (pgvector)
ragchat/db.py       SQLAlchemy models + self-healing schema init
eval/               Golden set (46 Q), corpus (10 files), LLM judges, harness
frontend/           Vite, TWO entry points:
                      index.html  + landing.css  → the landing page at "/"
                      app.html    + styles.css + app.js → the workspace at "/app"
                      tokens.css  shared by both (palette, type scale, metrics)
```

## Commands

```bash
# Backend (local)
.venv\Scripts\python -m uvicorn ragchat.app:app --reload --port 8000

# Frontend (proxies /api → localhost:8000)
# "/" is the landing page. The APP is at /app.html in dev — the "/app" rewrite
# lives in vercel.json and does not exist on the Vite dev server.
cd frontend && npm run dev

# Screenshots: both pages x 5 breakpoints x 2 themes, into shots/<page>/<theme>/
# Fails the run on horizontal overflow or any page error.
node shot.mjs http://localhost:5173

# Workspace layout BEHAVIOUR (drag-resize, collapse, persistence) — the part a
# screenshot cannot see. Exits non-zero on the first failed assertion.
node layout_check.mjs http://localhost:5173

# Tests
.venv\Scripts\python -m pytest tests/ -v

# Benchmark CLI (full harness, writes eval/runs/<ts>/)
.venv\Scripts\python -m eval.run_eval --limit 5
.venv\Scripts\python -m eval.run_eval --retrieval-only
```

---

## Non-obvious constraints — read before changing anything

### Vercel serverless: no writes, no background work

The repo directory (`/var/task`) is **read-only**, and the function is **frozen
the instant the HTTP response is sent**. A later request may hit a **different
instance**.

Therefore:

- **Never write to the repo directory at runtime.** Writable state goes to the
  database, or to `/tmp` if it's genuinely per-instance scratch. `config.DATA_DIR`
  already switches to `/tmp` when `VERCEL` is set.
- **Never use `threading.Thread` / `BackgroundTasks` for real work.** It will
  appear to start and then silently stop. This exact bug made the Run Benchmark
  button a no-op for weeks.
- **Long jobs must be sliced.** The benchmark is the reference pattern: a client
  loop calls `POST /api/eval/step`, each step does one bounded unit and commits
  to the `eval_runs` table. See `DEPLOY_VERCEL.md`.
- **Anything periodic is driven from outside.** Vercel Hobby cron exists but
  fires once per *day*, which cannot honour a 30-minute guest TTL, so
  `.github/workflows/guest-sweeper.yml` calls `POST /api/admin/sweep-guests`
  every 15 minutes with a shared secret. Work done in front of a waiting
  visitor is a backstop only and stays tiny — `guests.INLINE_REAP_LIMIT` is 2,
  and it was 20, which put 39.7s in the path of a first page load.
- `maxDuration` is 60s (set in `vercel.json`). The Hobby default of 10s is not
  enough for a single scored question.

### Config precedence — env vars are NOT the source of truth

```
config.yaml  <  config_overrides DB row (written by the Settings UI)
```

`load_config()` re-reads both on **every call**, so tuning needs no restart.
Env vars (`RAG_LLM_MODEL`, `EMBEDDING_PROVIDER`, …) are only **boot defaults**
via `settings.*` and are frequently *not* what's actually running.

**Never read `settings.default_*` for live behaviour** — read `load_config()`.
Both major bugs in this repo's history came from violating this:
`judges.JUDGE_MODEL` pinned the env default at import time (→ 404 on every
judge call), and `_rerank()` read `reranker_provider()` instead of
`cfg.reranker_provider` (→ dropdown did nothing).

Same rule for module-level singletons: cache them **keyed by provider**, never as
a single global, or a runtime provider switch silently keeps using the old one.

### Model ids must be exact

Google's OpenAI-compatible endpoint serves specific ids and 404s on near-misses:

- `models/gemma-4-26b-a4b-it` ✅  — `gemma-4-26b-it` ❌
- `models/gemini-embedding-001` ✅ — `text-embedding-004` / `-005` ❌

When adding a model, confirm it appears in `client.models.list()` first.

### The 768-dimension invariant

The Neon `chunks` table has **one fixed `vector(768)` column** shared by all
models. Every embedding model exposed in the UI must return 768 dims —
`embedding_dim()` always requests 768, and OpenRouter honours the `dimensions`
param for the allowlisted models.

Only add to `EMBEDDING_768_MODELS` (`ragchat/embeddings.py`) after confirming
live that `dimensions=768` really returns 768. Changing the embedding model
changes `PipelineConfig.fingerprint()`, which invalidates existing chunks — the
UI prompts for a re-index.

Google serves **three** embedders (`gemini-embedding-001`, `-2`, `-2-preview`),
but only `-001` is allowlisted — the other two are excluded until their 768
support is confirmed live, so the Gemini dropdown shows exactly one entry. Every
other option in the dropdown is OpenRouter's.

Chat models are **not** provider-scoped: generation always goes to the
Gemini/proxy endpoint whatever the embedding provider is. They get their own
cache entry (`_CHAT_KEY`), and the static fallback is served but **never
cached** — `discover_models` returns it on failure, so caching one rate-limited
reply would hide the real catalog for the whole TTL.

### Evaluation must fail open, never closed

An LLM judge that 404s, times out, or replies without a verdict is a **broken
grader**, not a failed answer. Represent that as `None` ("not graded") and
surface the reason — never `False`. Rendering "FAIL" for an ungraded answer
looks like a confident hallucination finding and is actively misleading.

`judges._parse_verdict` raises `JudgeError` when no verdict is present;
`pipeline._eval_answer` and `run_eval._safe_judge` convert that to `None` plus a
`judge_error` string. `aggregate()` excludes ungraded rows from the mean and
counts them in `n_ungraded`.

Also: judges are thinking-model-sensitive. Keep `max_tokens` generous (reasoning
tokens are billed against it) and strip `<thinking>` wrappers before parsing.

### Two vector backends

`VECTOR_BACKEND=chroma` (local dev) or `neon` (deploy). `ragchat/vectordb.py`
imports the implementation **inside each function**, so importing it never pulls
in `chromadb` on Neon or `psycopg2` on Chroma. **Go through `vectordb.py`** —
importing `ragchat.store` directly breaks the Neon deploy. (`reset_eval_collection`
used to do exactly that.)

Note `prune_chunks(user, set())` deliberately **no-ops** on an empty valid-doc
set, so it can't wipe a user who has no documents yet. To actually clear chunks,
call `delete_document_chunks` per doc id.

### Diagnosing a deploy

`GET /api/health` reports the **effective** config — `effective_config.llm_model`,
`judge.model`, `embedding_models_by_provider`, DB connectivity, and which secrets
are present (never their values). Check it before assuming an env var is the
problem.

## Conventions

- Comments explain **why**, especially where the code looks odd because of a
  serverless constraint or a past bug. Keep them when refactoring.
- Frontend is dependency-free vanilla JS. No framework, no build step beyond Vite.
  Bump the `?v=` cache-buster in **`app.html`** when changing `app.js` or
  `styles.css` (and in `index.html` when changing `landing.css`).
- Errors that reach the user should be actionable strings, not 500s — the chat
  path deliberately catches embedding/generation failures and returns a readable
  answer instead.
