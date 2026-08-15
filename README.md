# RAG Chat

A multi-user RAG web app built per [PRD.md](PRD.md): chat with an LLM grounded
in your own documents (PDFs, web pages, Markdown/text), with citations, a
tunable pipeline, and an evaluation harness to measure tuning changes.

Stack: **FastAPI + LangChain + Chroma (on-disk) + Vite**, models via the class
LLM proxy (OpenAI-compatible).

## Run it

Two terminals, from this directory:

```bash
# Terminal 1 — backend (port 8000)
.venv/bin/uvicorn ragchat.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend (port 5173, proxies /api to the backend)
cd frontend && npm run dev
```

Then open the **port 5173** URL from the workspace port panel. Create a local
account on the sign-in screen (see "Google OAuth" below to enable real OAuth).

### Configuration

## Pipeline knobs (config.yaml)

- `chunking` — chunk_size, chunk_overlap, splitter (recursive | markdown_header |
  semantic). Changing these invalidates stored chunks (fingerprint), so
  re-index after editing.
- `embedding.model` — embedding modelspec. **Switching models isolates your
  chunks into a fresh Chroma collection** (Chroma fixes vector dimension per
  collection), so old-model chunks are hidden until you re-index — no crash,
  no cross-dimension bleed. Re-index after switching.
- `retrieval.hybrid_search` — **real BM25 keyword fusion**: the vector top-k
  and a BM25 keyword top-k are fused with reciprocal rank fusion (RRF). This
  promotes exact-match terms (IDs, codes, names) that pure-vector search
  misses. It does NOT add web text — document grounding is unaffected.
- `retrieval.similarity_threshold` — minimum cosine similarity a chunk must
  clear to be used. When nothing clears it, you get the explicit "not found"
  answer (PRD F13). Leave at 0.0 and the pipeline applies a small safety
  floor so noise-only hits still refuse.
- `retrieval.top_k` / `candidate_k` — chunks handed to the LLM / the wider
  candidate pool the reranker narrows.
- `retrieval.reranker` — LLM cross-encoder scores candidates → top_k.
- `generation.web_augmentation` — **fallback only**, default OFF. When ON,
  web (DuckDuckGo) results are appended as labeled `[web]` chunks **only** when
  the user's own documents do not clear the relevance threshold. It never
  overrides a grounded answer. This is what the PRD called "hybrid_search";
  the config key was split so BM25 fusion and web fallback are independently
  controllable.
- `generation.llm_model` / `temperature` — answering model + temperature.
- **Secrets** — environment variables only:
  - `ANTHROPIC_AUTH_TOKEN` (already set in this workspace)
  - `RAG_ALLOWED_ROOT` — allowed root for folder sources (default: `$HOME`)
  - `RAG_LLM_MODEL` / `RAG_EMBEDDING_MODEL` — override config.yaml defaults

**Important:** after changing chunking or embedding settings in config.yaml,
click **Re-index all** in the sources panel (or re-add sources). Chunks are
tagged with a config fingerprint, and old-fingerprint chunks are hidden from
retrieval until re-indexed — so answers may come back empty until you do.

### Google OAuth (optional)

Without credentials the app falls back to local username/password accounts.
To enable Google sign-in:

1. Create an OAuth client (Web application) in a Google Cloud project.
2. Add your frontend URL + `/api/auth/google/callback` to the authorized
   redirect URIs.
3. Set env vars before starting the backend:
   `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`,
   `GOOGLE_REDIRECT_URI=http://<your-frontend-url>/api/auth/google/callback`.

The "Sign in with Google" button appears automatically when configured.

## Use it

- **Sources panel (left):** drag-and-drop or click to upload; paste a URL to
  add a web page; type a folder path to index a whole directory (recursive,
  PDF/MD/TXT/HTML). Folder sources have a rescan (↻) button that picks up
  new/changed/deleted files.
- **Chat (center):** ask questions; answers cite sources with `[1]`-style
  markers. Follow-ups work ("and what about X?"). Questions outside your
  documents get an explicit "not found" answer.
- **Excerpt (right):** click a citation marker or chip to see the source
  passage.

## Evaluate & tune it (PRD §7)

The eval harness indexes `eval/corpus/` under the current config and scores
the pipeline against `eval/golden.jsonl` (23 answerable + 5 unanswerable
questions):

```bash
# Baseline (uses config.yaml as-is)
.venv/bin/python -m eval.run_eval

# Cheap smoke run (no generation/LLM judge)
.venv/bin/python -m eval.run_eval --retrieval-only

# Change a knob in config.yaml, re-run, then compare:
.venv/bin/python -m eval.compare --last-two
```

Reports land in `eval/runs/<timestamp>/` with the full config snapshot, so
runs are reproducible and comparable. A change counts as an improvement only
if Recall@k / answer correctness rises **without** lowering the not-found
rate (PRD §7.4).

Default-config baseline measured 2026-08-15: Recall@k 1.0, MRR 1.0,
answer correctness 0.87, not-found rate on unanswerables 1.0.

## Project layout

```
ragchat/          FastAPI backend
  app.py          routes: auth, sources, chats, eval
  auth.py         Google OAuth + password fallback, signed cookie sessions
  config.py       env settings + config.yaml loader (+ fingerprint)
  chunking.py     splitters per config
  loaders.py      PDF/HTML/text extraction, URL fetch
  store.py        per-user Chroma collections
  pipeline.py     ingest + retrieve + generate with citations
  db.py           SQLite models (users, documents, folders, chats)
frontend/         vanilla JS + Vite (3-pane NotebookLM-style UI)
eval/             golden set, corpus, run_eval.py, compare.py
config.yaml       all pipeline knobs
data/             SQLite db, Chroma index, uploaded files (gitignore-worthy)
```
