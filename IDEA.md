# RAGit — Idea

A multi-user web app where you chat with an LLM that answers **only from your
own documents** (PDFs, web pages, Markdown/text, CSV/JSON) and cites the exact
passages it used. Built as a portfolio-grade showcase, it is a genuine product:
guest-first access, per-user isolation, hybrid retrieval, live judging, and a
polished HUD aesthetic that reads as an instrument panel rather than a chat toy.

## The idea

- **Grounded answers** — every answer comes from your uploaded documents, never
  the model's general knowledge.
- **Citations** — answers link back to the exact source passage; clicking one
  opens the excerpt.
- **Multi-format ingestion** — upload files, paste a URL, or sync a folder
  (folders for signed-in users).
- **Tunable pipeline** — retrieval and generation are configured live in a
  Settings panel and in config, no code changes.
- **Measurable quality** — a live LLM judge scores every answer (similarity,
  faithfulness, relevancy, latency) against a real benchmark baseline;
  "Ungraded" when the judge fails, never a false claim.
- **Multi-user, isolated** — every account's documents and chats are fully
  separated; guests get a private, throwaway workspace of their own.
- **Built to grow** — clean module boundaries so a larger product can sit on
  top.

## How people start using it

**Guest-first, sign in to keep it.** Every anonymous visitor is automatically
provisioned their own private, writable workspace — not a shared demo account.
A small demo corpus (two documents: `helios_energy_handbook.md`,
`meridian_coffee_ops.md`) is embedded once and vector-copied per guest, so a
new visitor lands on an already-populated workspace with no latency and no
embedding cost. Guests can upload up to 3 documents / 5 MB and ask freely;
chats are per-visitor by construction. Idle guests are reaped after a couple
of hours. Signing in with Google — optional; the app works fully signed out as
a local account, and the button appears only when OAuth is configured —
*promotes* the guest's work into the permanent account: documents, chats and
chunks are re-pointed, nothing is re-embedded. A handful of write routes are
server-side gated for guests (eval config, benchmark runs, folder scans,
re-index) because those touch shared configuration or the server's filesystem,
or spend real LLM quota.

## How it works (in words)

The browser UI (vanilla JS + Vite, a landing page at `/` and the workspace at
`/app`) talks to a FastAPI backend behind a single Vercel function. Uploaded
documents are loaded, split into chunks, embedded, and stored in Neon
Postgres with pgvector (a Chroma backend is also supported). A question is
rewritten into a standalone query, retrieves the top chunks via vector search
plus BM25 keyword fusion (with an optional reranker), and generates an answer
grounded in those chunks. Sign-in uses Google OAuth (with a local
username/password fallback), carrying `itsdangerous`-signed session cookies —
no authlib, no session middleware. All embedding models render 768-dimension
vectors; changing the embedding model invalidates chunks and triggers a
re-index. Long jobs (re-index, folder scans, benchmark) run as sliced batches
to fit the platform's execution limit, and evaluation is fail-open: a broken
judge produces "Ungraded", never a confident wrong score.

## How it looks and feels

Dark is the default, light is a first-class toggle; both themes are verified
at every breakpoint. The acid-lime accent means *active / live*, never "good" —
passing answers show a neutral checkmark, red stays reserved for failure and
destructive actions. Type runs in two registers: a large, generous **content
register** for the answer, and a small mono, letterspaced **telemetry
register** for eval scores, latency, model ids and index status.

The workspace is three columns: **Sources** (card treatment with type, status
and citation-count pills), **Chat** (the largest share, with a Sessions
switcher, and the Excerpt pane flowing into the conversation column), and
**Evaluation** (collapsed by default; when deliberately opened, it is the one
in-app surface allowed display-scale benchmark numerals). Under each answer,
a quality readout — a dot, one word (`Grounded` / `Weak` / `Ungraded`) and a
tick bar against the benchmark baseline — expands on tap to the full four
metric rows. Settings are preset-first (`Fast` / `Balanced` / `High accuracy`
/ `Low cost`) with expert fields behind an "Advanced" toggle, and a ⌘K
command palette jumps across chats, sources and settings on desktop only.

Empty states walk the user through the four-beat loop — add a source, ask,
click a citation, read the quality readout — and jargon is explained inline on
tap, never in `title` tooltips (which don't exist on touch). Mobile is
deliberately a different product shape: a reader. One stacked column,
conversation primary, bulk corpus operations desktop-only, no horizontal
scroll, verified at 400px. The landing page at `/` is a near-literal homage to
the visual reference (Grok-UI): space backdrop, hero type, numbered section
labels, a pipeline diagram (ingest → retrieve → rerank → generate → judge) and
architecture notes — static, no API calls, so visitors who never enter cost
zero serverless invocations.

## Current state

Nearly all of the above is built and committed on `main`: themes, landing page
and routing, guest tiers with route guards, the two typographic registers,
the per-answer readout, settings presets, sequenced empty states, glossary,
persistent job status, the ⌘K palette, and the Google sign-in wiring. 62 tests
pass. `UI_UPGRADE_PLAN.md` and some of `PRODUCT_UX_PLAN.md` were planning docs;
the shipped behavior supersedes parts of each — `PRODUCT_UX_PLAN.md`,
`GOOGLE_OAUTH_SETUP.md` and `HANDOFF.md` hold the change log, the auth setup
steps and the exact commit list.

## Not yet built

- Offline golden-set eval as a standalone harness.
- Streaming answers.
- Retrieval upgrades: Contextual Retrieval, HyDE, multi-query, parent-child.
- Server-side per-source citation counts (client-side tallying already covers
  the visible count).
- The demo corpus is intentionally two files — landing copy must never promise
  ten.

## Current defaults

- Generation model: `models/gemma-4-26b-a4b-it`
- Embeddings model: `models/gemini-embedding-001` (768-dim)
- Both served through the class LLM proxy (OpenAI-compatible `/v1`)