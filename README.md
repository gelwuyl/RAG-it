# RAG Chat

A multi-user **Retrieval-Augmented Generation (RAG)** web app: chat with an LLM that answers **only from your own documents** (PDFs, web pages, Markdown/text) with citations, a tunable pipeline, and built-in evaluation.

Deployed on **Vercel** (Python backend) with a **Neon Postgres + pgvector** database. Models are served through the class LLM proxy (OpenAI-compatible `/v1` API).

## Architecture

| Component | What it does | Tool / Service | Why this choice |
|-----------|--------------|----------------|-----------------|
| **Backend** | REST API: auth, upload, chat, config, eval | FastAPI (Python) | Lightweight async API; deploys cleanly on Vercel's Python (uv) runtime |
| **Frontend** | 3-pane chat UI (sources / chat / excerpt) | Vanilla JS + Vite | No framework overhead; fast and simple to maintain |
| **Database** | Users, documents, chats, messages, config | Neon Postgres | Serverless Postgres — survives Vercel's read-only filesystem and redeploys |
| **Vector DB** | Semantic search over document chunks | pgvector (inside Neon) | Lives in the same Postgres; no separate vector service to run |
| **Embeddings** | Text → vectors for storage & search | class proxy `/v1/embeddings` (`models/gemini-embedding-001`, 768-dim) | One embedding model per deployment; OpenAI-compatible |
| **Generation** | Produces the grounded answer | class proxy `/v1/chat/completions` (`models/gemma-4-26b-a4b-it`) | Grounds every claim in retrieved passages |
| **Question rewrite** | Turns follow-ups ("and what about X?") into standalone queries | LLM rewrite | Improves retrieval for conversational follow-ups |
| **Reranker** | Re-ranks candidates by relevance before the LLM sees them | LLM cross-encoder | Raises top-k quality; can be toggled off |
| **Evaluation** | Scores each answer (faithfulness, relevancy, latency, top similarity) | LLM-as-judge + offline harness (`eval/`) | Shows a grey performance line under every answer so tuning is measurable |
| **Authentication** | Sign-in + session cookies | Local username/password + optional Google OAuth | Works out of the box; OAuth is opt-in |

> The evaluation judge is a heuristic (LLM-as-judge) and can mis-flag edge cases — treat its verdicts as a signal, not ground truth. Web fallback and the reranker are basic first versions that can be improved.

## Pipeline configuration

All knobs live in `config.yaml` (or are tuned live from the Settings panel, which persists to the database). Changing chunking or embedding settings invalidates stored chunks (a config *fingerprint* tags them), so click **Re-index all** afterward.

- **`chunking`** — `chunk_size`, `chunk_overlap`, `splitter` (recursive | markdown_header | semantic). Controls how documents are split.
- **`embedding.model`** — embedding model spec. Switching models isolates old chunks (different vector dimension) until you re-index.
- **`retrieval.hybrid_search`** — **real BM25 keyword fusion**: vector top-k and a BM25 keyword top-k are fused (RRF). Surfaces exact terms (IDs, codes, names) pure-vector search misses. **Not web search** — your documents only.
- **`retrieval.similarity_threshold`** — minimum cosine similarity a chunk needs to be used. Below it → explicit "not found" answer.
- **`retrieval.top_k` / `candidate_k`** — chunks shown to the LLM / the wider pool the reranker narrows.
- **`retrieval.reranker`** — LLM cross-encoder re-scores candidates → top_k.
- **`generation.web_augmentation`** — **fallback only**, default OFF. Appends labeled `[web]` chunks (DuckDuckGo) **only** when your documents don't clear the relevance threshold. Never overrides a grounded answer.
- **`generation.llm_model` / `temperature`** — answering model + temperature.
- **Secrets (env vars only)** — `GEMINI_API_KEY` (proxy), `RAG_LLM_MODEL` / `RAG_EMBEDDING_MODEL` overrides.

## Run locally

```bash
# Backend (port 8000)
.venv/Scripts/python -m uvicorn ragchat.app:app --host 0.0.0.0 --port 8000

# Frontend (port 5173, proxies /api to the backend)
cd frontend && npm install && npm run dev
```

Set `GEMINI_API_KEY` (and optionally `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` for OAuth) before starting the backend. In production the app uses Neon (`rag_gel_DATABASE_URL`) and a `/tmp` data dir — no local database needed.

## Project layout

```
ragchat/          FastAPI backend
  app.py          routes: auth, sources, chats, eval, config
  auth.py         Google OAuth + password fallback, signed cookie sessions
  config.py       env settings + config.yaml loader + DB-backed override + fingerprint
  chunking.py     splitters per config
  loaders.py      PDF/HTML/text extraction, URL fetch
  store_neon.py   Neon pgvector chunks + documents
  vectordb.py     vector store dispatcher (Neon impl)
  pipeline.py     ingest + retrieve + rewrite + rerank + generate + eval
  db.py           SQLAlchemy models + Neon engine (users, documents, conversations, messages, config_overrides)
  embeddings.py   ProxyEmbeddings via /v1/embeddings + retry/backoff
frontend/         vanilla JS + Vite (3-pane UI)
eval/             golden set, corpus, judges, metrics, run_eval.py
config.yaml       all pipeline knobs
```
