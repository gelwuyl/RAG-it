# PRD — RAG Chat (working title)

**Version:** 0.2 (draft)
**Date:** 2026-08-15
**Status:** Draft for review

## 1. Summary

A web-based RAG (Retrieval-Augmented Generation) chat application. Users sign
in with Google OAuth, upload their own documents (PDFs, web pages,
Markdown/text), and chat with an LLM that answers strictly from those
documents, citing its sources. Built on LangChain (Python/FastAPI) so the
powering model can be swapped via configuration. The full RAG pipeline —
chunking, retrieval, generation — is parameter-driven, and a built-in
evaluation harness measures retrieval and answer quality so parameter changes
can be compared objectively. This system is intended as the retrieval
foundation for a larger application later.

## 2. Goals & Non-goals

### Goals

- Grounded answers: every answer is derived from the user's own documents.
- Citations: answers reference the source document and passage they came from.
- Multi-format ingestion: PDF, web pages (by URL), Markdown, plain text.
- **Tunable pipeline:** chunking, retrieval, and generation parameters live in
  one config file — no code changes to experiment.
- **Measurable quality:** an eval harness + golden QA set produces comparable
  scores per config, so "is this parameter change better?" has a numeric answer.
- Model-agnostic generation via LangChain + one config value.
- Multi-user: accounts via Google OAuth, fully isolated document stores and chats.
- A clean, well-separated codebase that a bigger app can build on.

### Non-goals (v1)

- Admin/moderation tooling, billing, quotas.
- Document sharing between users.
- Fine-grained document permissions, tags, folders.
- Re-crawling web pages on a schedule (ingest once at the URL given).
- Streaming answers (nice-to-have, see §10).
- OCR / scanned-image PDFs (text-layer PDFs only).
- A UI for tuning parameters — v1 is config-file driven.
- Mobile-native apps; a responsive web UI is sufficient.

## 3. Users

Individual users who want to ask questions over their own private document
collections. Each user's corpus is small (tens of documents). The system is
also the internal foundation for a future larger product, so clean module
boundaries matter as much as features. A secondary "user" is the developer
(its owner) tuning the pipeline: they need fast, reproducible before/after
comparisons when parameters change.

## 4. Functional requirements

### 4.1 Authentication & accounts

- **F1:** Users sign in via **Google OAuth** (Authorization Code flow).
- **F2:** First sign-in of a new OAuth identity creates a user account.
- **F3:** Users can sign out. All app pages except sign-in require auth.

### 4.2 Document ingestion

- **F4:** Users upload documents via the web UI. Supported formats:
  PDF (text layer), `.md`, `.txt`, and common plain-text variants.
- **F5:** Users add web pages by pasting a URL; the page content is fetched
  and indexed.
- **F6:** Users can attach one or more **server-side folders** (via the
  sources panel). The system recursively scans each folder and indexes the
  supported file types (PDF with text layer, `.md`, `.txt`, HTML) inside it,
  including subfolders.
- **F6a:** Folder sources stay in sync: a rescan (button per folder, plus a
  rescan command) indexes new files, re-indexes changed files (detected by
  content hash), and removes files that were deleted on disk.
- **F6b:** Folder paths must live under a configurable allowed root
  (default: the app user's home directory); paths outside it are rejected.
- **F7:** Uploaded/ingested documents are chunked, embedded, and added to the
  user's vector store incrementally (no full re-index needed to add a doc).
- **F8:** Users can see their ingested documents and delete them (deleting
  removes their chunks from the vector store).

### 4.3 Chat

- **F9:** A web chat UI: multi-turn conversation, persisted per user.
- **F10:** Chat history survives logout and browser restarts; users can reopen
  past conversations and start new ones.
- **F11:** Each question retrieves the top-k chunks from the user's own vector
  store and generates an answer from them, with conversation history included
  so follow-ups ("and what about X?") work.
- **F12:** Answers cite their sources — document title/name plus a passage
  reference, rendered as links/snippets in the UI.
- **F13:** If retrieval finds nothing relevant (no chunk clears the similarity
  threshold, or none retrieve), the assistant says so ("I couldn't find this
  in your documents") instead of answering from the model's own knowledge.

### 4.4 Configuration & secrets

- **F14:** The generation model is configurable via environment variable,
  defaulting to `qwen3.8-max` via the class LLM proxy
  (`https://llmproxy.mrchloep.com/v1`). Swapping to `deepseek-v4-pro` or
  `qwen3-coder` must require no code changes.
- **F15:** Secrets (LLM API key, OAuth client secret, session secret) are read
  from environment variables only — never hardcoded.

### 4.5 RAG pipeline configurability

- **F16:** Every pipeline parameter listed in §5.1 is settable from a single
  config file (`config.yaml`) — chunking, retrieval, and generation — with no
  code changes.
- **F17:** Changing index-affecting parameters (chunk size, overlap, splitter,
  embedding model) requires a re-index; the system provides a re-index command
  that rebuilds affected chunks. Chunks built under different configs must
  never be mixed in one query.
- **F18:** Each stored chunk records the config fingerprint it was built under
  (e.g. a hash of the chunking/embedding settings), so config drift is
  detectable and the re-index command can target stale chunks.
- **F19:** Every eval run (§7.2) records the full config snapshot it used
  alongside its scores, so any two runs are directly comparable.

### 4.6 User interface

NotebookLM-style three-pane layout: sources live in a persistent left panel,
chat in the center, and source excerpts open in a right panel on citation
click.

- **U1:** Two views: a **sign-in** screen (single "Sign in with Google"
  button), then a single-page app shell with three panes
  (sources / chat / excerpt).
- **U2:** Left **sources panel**: drag-and-drop/file-picker upload, a URL
  paste box ("add a web page"), and an "add folder" action (path input).
  Below, the source list: individual uploads show status
  (*indexing / ready / failed*); folder sources show the folder path and
  their document count, with **rescan** and **remove** actions. Deleting a
  folder source removes its documents from the index but leaves the files
  on disk.
- **U3:** Center **chat pane**: conversation switcher at the top (start new,
  reopen past conversations), user/assistant message bubbles below. Answers
  render asynchronously (no page reload); streaming is deferred (§10).
- **U4:** Citations: answers carry inline numbered markers (`[1]`, `[2]`),
  with a legend below the answer mapping each number to a source. Clicking a
  marker opens the source passage excerpt in the right panel.
- **U5:** A "not found in your documents" answer renders in a visibly distinct
  style so refusals are not mistaken for facts.
- **U6:** Empty states: no sources yet → the left panel prompts upload /
  add-URL; new chat → short hint on what the assistant can do; excerpt panel
  hidden until a citation is clicked.
- **U7:** Desktop-first responsive layout. UI framework is an implementation
  choice, not a requirement; the only hard constraint is the single-port
  proxy setup (T5).

```
┌──────────────────────────────────────────────────────┐
│ ● RAG Chat                          user@…  [Sign out]│
├─────────────┬───────────────────────────┬────────────┤
│ Sources     │  [▾ Conversation] [+ New] │ Excerpt    │
│             │                           │            │
│ 📁 ~/docs   │ You: What does X mean?    │ guide.pdf  │
│    12 docs  │                           │ p.3: "X is │
│  [↻][✕]     │ AI: X means … [1][2]      │ defined as │
│ 📄 guide.pdf│   [1] guide.pdf           │ …"         │
│    ✓ ready  │   [2] notes.md            │            │
│ 🌐 wiki/url │                           │            │
│    indexing │ You: And how is it used?  │            │
│             │                           │            │
│ [+ Upload]  │ AI: It is used to… [2]    │            │
│ [+ Add URL] │                           │            │
│ [+ Folder]  │ [ Ask about your docs…  ] │            │
└─────────────┴───────────────────────────┴────────────┘
```

## 5. RAG tuning parameters (config surface)

Research-backed starting points; every row below is a knob in `config.yaml`.
Consensus from LlamaIndex/Pinecone/LangChain guidance: **moderate chunks
(256–512 tokens) with 10–20% overlap** is the most validated default, and
retrieval-side levers (hybrid search, reranking) typically move accuracy more
than fine-tuning chunk size further.

| Group | Parameter | Default | Range | What it controls |
|---|---|---|---|---|
| Chunking | `chunk_size` | 512 tokens | 128–1024 | Granularity: smaller = more precise hits, larger = more surrounding context per hit. LlamaIndex testing found 512 best of {128–2048}. |
| Chunking | `chunk_overlap` | 75 tokens (~15%) | 0–25% | Preserves context across chunk boundaries. Beyond ~20% mostly adds index bloat and duplicate hits. |
| Chunking | `splitter` | structure-aware recursive | enum: `recursive`, `markdown_header`, `semantic` | Respects document structure (Markdown headers, PDF paragraphs) before falling back to sentence/char splits. |
| Indexing | `embedding_model` | `text-embedding-005` (768d) | any proxy embedding | Semantic space of retrieval. Changing it invalidates the whole index (dims aren't comparable). |
| Retrieval | `top_k` | 4 | 1–20 | Chunks actually handed to the LLM. The single most-tuned retrieval knob. |
| Retrieval | `candidate_k` | 20 | ≥ top_k | Wider candidate pool when reranker is on; reranker narrows to `top_k`. |
| Retrieval | `similarity_threshold` | 0.0 (off) | 0.0–1.0 | Minimum similarity to keep a chunk; primary driver of the "not found" path (F13). |
| Retrieval | `hybrid_search` | off | bool | Vector + BM25 keyword fusion. Big win for exact-match terms (names, IDs, error codes) that pure-vector search misses. |
| Retrieval | `reranker` | off | bool + model | Cross-encoder re-scores `candidate_k` → `top_k`. Highest-leverage accuracy lever per Anthropic's Contextual Retrieval results. |
| Query | `query_rewrite` | on | bool | Resolve follow-ups against chat history ("and X?") into a standalone retrieval query. |
| Generation | `llm_model` | `qwen3.8-max` | any proxy chat model | Answering model, via env/config. |
| Generation | `temperature` | 0.0 | 0–1 | Low temperature for factual, grounded answers. |

Deliberately deferred knobs (too complex for v1, noted for the roadmap):
contextual chunk pre-embedding (Anthropic Contextual Retrieval), HyDE
(hypothetical-document embeddings), multi-query fan-out, parent-child
("small-to-big") retrieval.

## 6. Technical requirements

- **T1:** Framework: **LangChain (Python)** with a **FastAPI** backend.
- **T2:** Vector store: **Chroma with on-disk persistence** (survives
  restarts; incremental adds; metadata used for F18 config fingerprints).
- **T3:** Embeddings: **one model per deployment, configurable** — default
  `text-embedding-005` (768 dims) via the class proxy's `/v1/embeddings`.
  Changing models requires re-indexing; the system should fail loudly rather
  than mix models.
- **T4:** Application database for users, document metadata, and chat
  history: **SQLite** (zero-setup; swappable later).
- **T5:** Single exposed port per workspace conventions: the frontend dev
  server serves the UI and proxies `/api` to the backend.
- **T6:** All per-user data (documents, chunks, chats) is strictly isolated;
  one user must never retrieve or see another user's content.

## 7. Success metrics & evaluation

### 7.1 Functional acceptance

- **M1:** Upload a document of each supported format → it appears as
  "indexed" within ~30 s (tens-of-docs corpus).
- **M2:** Two test users on the same deployment: user A's queries never
  surface user B's documents.
- **M3:** Restart the server → vector index, users, and chats all persist.
- **M4:** Changing a config value (e.g. `chunk_size`) and re-running the eval
  harness produces a report that can be compared side-by-side with the
  baseline run.

### 7.2 Golden dataset & eval harness

- **E1:** Maintain a golden QA set of ≥ 30 entries as
  `(question, expected answer, source document + passage)` triples in the
  repo (e.g. `eval/golden.jsonl`), including ≥ 5 **unanswerable** questions
  (answers not in the corpus).
- **E2:** An eval harness (CLI command) runs the full pipeline over the golden
  set and writes a report to `eval/runs/<timestamp>/` containing: per-question
  results, aggregate metrics, and the full config snapshot used (F19).
- **E3:** The harness supports re-running after a parameter change without
  manual bookkeeping; comparing two reports is a documented one-command step.

### 7.3 Measurable outcomes (v1 targets at default config)

| Metric | What it measures | Target |
|---|---|---|
| **Recall@k** | Fraction of golden questions whose source passage appears in the retrieved top-k | ≥ 0.80 |
| **MRR** (mean reciprocal rank) | How high up the correct passage ranks | ≥ 0.65 |
| **Answer correctness** | Semantic match between generated answer and golden answer (LLM-judged) | ≥ 0.80 |
| **Not-found rate on unanswerables** | Unanswerable questions correctly refused instead of hallucinated | ≥ 90% |
| **Faithfulness spot-check** | Sampled answers contain no facts outside retrieved context | 100% on 10-question manual sample |

### 7.4 Tuning workflow (definition of "better")

A parameter change counts as an improvement only if, on the golden set, it
raises Recall@k and/or answer correctness **without** lowering the not-found
rate. Every experiment lands as a committed report in `eval/runs/`, so the
baseline and each variant remain auditable.

## 8. Out of scope (deliberately deferred)

- Full RAGAS-style automated scoring (faithfulness/context-precision computed
  by LLM judge over every answer) — v1 ships deterministic retrieval metrics +
  LLM-judged answer correctness + one manual faithfulness sample.
- Contextual Retrieval, HyDE, multi-query, parent-child retrieval (§5).
- Tuning-parameter UI, document sharing, collections, search-over-docs UI.
- Scheduled re-crawling, webhooks, API keys for third parties.
- Horizontal scaling; per-user rate limiting beyond proxy limits.

## 9. Risks

- **Google OAuth setup cost:** needs a Google Cloud project, OAuth client ID,
  and a registered redirect URI; Google is strict about authorized redirect
  URIs, and the workspace's generated frontend URL must be added there.
  Fallback if it proves painful: simple username/password auth behind the
  same interface.
- **Re-index cost when tuning:** chunking/embedding changes rebuild the whole
  corpus. Trivial at tens of documents; expensive later. Mitigation: config
  fingerprints (F18) and targeted re-index (F17).
- **PDF parsing quality varies.** Scanned PDFs without a text layer won't
  work in v1; messy PDFs may need re-parsing later.
- **Retrieval quality is the product.** Poor chunking → bad answers. The eval
  harness (§7.2) exists precisely to iterate on this empirically rather than
  by feel.
- **Embedding model lock-in per index.** Changing the embedding model means
  rebuilding every user's index. Accepted for v1; noted in docs.
- **Proxy budget:** only `qwen3-coder` + embeddings draw the user's $50;
  `qwen3.8-max`/`deepseek-v4-pro` are class-subscription models. Embedding
  calls and LLM-judged eval draws the budget — keep the corpus and eval runs
  small during development.

## 10. Open questions

- **O1:** Streaming answers in v1 or deferred?
- **O2:** Golden set authoring: hand-written only for v1, or allow
  LLM-generated candidates with human review to reach 30+ faster?
- **O3:** Per-user storage quotas for uploads — needed for v1?

## 11. Milestones (suggested)

1. **Skeleton:** FastAPI + frontend scaffold, proxy wiring, health endpoint.
2. **Ingestion:** upload/URL/folder → chunk → embed → Chroma (per user),
   config-driven chunking (F16–F18).
3. **Chat:** retrieval + generation with citations; multi-turn; not-found path.
4. **Accounts:** Google OAuth login, per-user isolation, persistent chats.
5. **Eval harness:** golden set (E1), harness + reports (E2–E3), first
   baseline run + one parameter experiment proving the loop works.
6. **Hardening:** error handling, empty states, metrics check (§7).
