# RAG-it — the idea, the decisions, and how to run it

A document-grounded RAG chat app: ask questions about your own documents and get
answers that cite the exact passage behind them. Built as a portfolio piece and
run as a real product — guest-first access, per-user isolation, hybrid
retrieval, a benchmark that ships with the app, and an instrument-panel
aesthetic that treats the answer as the payload and everything else as
telemetry.

**This file is the single design and operations record.** It absorbed
`PRODUCT_UX_PLAN.md`, `UI_UPGRADE_PLAN.md`, `GOOGLE_OAUTH_SETUP.md`,
`DEPLOY_VERCEL.md`, `HANDOFF.md` and `eval/EVAL_SPEC.md`, each of which had
drifted out of date in a different direction. Where they described something
that was never built, or was built differently, this file describes **what
actually shipped** — verified against the code, not copied forward.

Two other documents stay separate on purpose:

- **`CLAUDE.md`** — the constraints that bite when changing the code
  (serverless freeze, config precedence, the 768-dimension invariant, fail-open
  evaluation). Read it before touching retrieval or config handling.
- **`README.md`** — the public front door: stack table, architecture diagram,
  how to run it.

The narrative version of §16, written for a visitor rather than a maintainer,
is the live page at [`/built`](https://rag-gel.vercel.app/built).

> **Section numbers §1–§9 are load-bearing.** Roughly two dozen comments across
> `frontend/` and `ragchat/presets.py` cite them by number (`IDEA.md §6`).
> Renumber a section and those citations start pointing at the wrong rule.

---

## 0. Decisions locked

| Question | Decision |
|---|---|
| Audience | Portfolio / showcase piece that is also genuinely usable |
| Theming | Dark **and** light, toggleable — dark default, both verified at every breakpoint |
| Accent | Acid lime, meaning *active / live* — **never "good"** (§6) |
| Entry | Landing at `/`, workspace at `/app`, build write-up at `/built` |
| First run | **Guest-first** — a private workspace on arrival, no sign-in wall (§3) |
| Access model | Private writable guest workspace; sign in to keep it (§3) |
| Explanatory content | Inline in the panes — no About route |
| Settings | Presets first, expert fields behind Advanced (§7) |
| Visual hierarchy | Answer is hero, sources a clear second, eval is instrumentation (§1) |
| Evaluation pane | Ships with a scored result — **nobody runs a benchmark to see it** (§11) |
| Benchmark runs | Client-driven slices, never a background thread (§13) |
| Mobile | Genuinely good, and a **different product shape** from desktop (§9) |
| Deep search | Per-request, never a stored setting. The app reaches for it **by itself** when ranked search comes up short; the switch is an override (§10) |
| Grading | Runs **after** the answer is delivered, in its own request (§10) |

**Visual reference:** [joeynyc/Grok-UI](https://github.com/joeynyc/Grok-UI) —
its "Operator" (carbon black / signal lime) theme. Its vocabulary is kept in
full and its *proportion* deliberately inverted; §1 explains why.

---

## 1. The core principle — two visual registers

Everything else follows from this. The reference aesthetic is **not the skin for
the whole app** — it is the skin for the *instrumentation layer only*.

**Content register** — the answer, its citations, the conversation. Large sans
(~17–18px), `line-height: 1.65`, high contrast, generous spacing.
Reading-optimised. **Dominates the screen.**

**Telemetry register** — eval scores, latency, similarity, chunk counts, model
ids, index status. Mono, uppercase, `letter-spacing: 0.12em`, ~11px,
deliberately *low* contrast. This is where the HUD vocabulary lives: numbered
labels, hairline borders, notched corners, accent underline bars.

The contrast between them is the whole design. A mono micro-label reads as
instrumentation *because* it sits beside something large and humane. If
everything is HUD, nothing is — and the answer, which is the actual product,
drowns in its own dashboard.

**Scope.** The rule governs surfaces where an *answer* is present. Display-scale
(48–72px) numerals are therefore banned from the inline per-answer eval: those
are technical values sitting next to prose and must stay small. The rule does
**not** govern surfaces with no answer to protect — the landing page (§2) and
the deliberately-expanded benchmark view (§8), where telemetry *is* the content.

**Kept from the reference:** layered near-blacks, hairline borders, no shadows
(depth from value steps), mono uppercase letterspaced micro-labels, notched
corners, accent underline bars, lime as the single hot accent, multiple themes.

**Concentrated rather than dropped:** display numerals, big geometric hero type,
space backdrop, grid motifs — these live on the landing page and in the
benchmark view, where they cost the reader nothing.

**Dropped:** far-left numbered nav rail, breadcrumb strip, decorative
radar/grid inside the workspace.

---

## 2. Landing page, and the build write-up

Three pages, two of them static documents that ship **no app JS** and make **no
API calls** — CDN-served, costing zero serverless invocations for a visitor who
never enters.

- **`/` — the landing page.** Near-literal homage to the reference: space
  backdrop, large geometric hero type, numbered section labels, the five
  pipeline stages in one row, grid-lined `A1–A4` metric tiles with
  display-scale numerals. The tile figures are **structural facts** (56 golden
  questions, 27 corpus documents, 768 dimensions, 1 serverless function) and
  are asserted by `tests/test_landing_claims.py` — the page once claimed a
  10-document corpus for as long as it had 27, because nothing connected the
  sentence to the directory it described.
- **`/built` — how it was built.** What was measured, what only broke in
  production, and what was left undone. This is where the engineering story
  lives; §16 is its maintainer-facing counterpart.
- **`/app` — the workspace.**

Benchmark *scores* deliberately do not appear on the landing page. They live in
the app's Evaluation pane where a published run backs them, and would go stale
on a static page with nothing noticing.

All three read `data-theme` from the same `localStorage` key, so a light-theme
visitor never gets a dark pitch page followed by a light workspace. The theme
script is inline and blocking in each `<head>` for the same reason.

**Routing.** Two entry points in `frontend/vite.config.js`
(`index.html`, `app.html`, `built.html`) and rewrites in `vercel.json`. In local
dev the clean paths do not exist — use `/app.html` and `/built.html`, because
`/app` and `/built` are Vercel rewrites and the Vite dev server has no router.

---

## 3. Access model — guest-first

**Guest tier (no sign-in).** Every anonymous visitor is auto-provisioned their
**own private, writable, throwaway workspace** (`ragchat/guests.py`). Not a
shared account: the app scopes every query by user id, and one shared guest
account would let strangers read and delete each other's uploads — the exact
mixing per-user isolation exists to prevent.

- The demo corpus is embedded **once** under a template account and then
  **vector-copied** per guest. No embedding calls, no quota spend, no latency on
  arrival, and each visitor's copy is genuinely theirs.
- Guests can upload, capped at **3 documents / 2 MB** total
  (`GUEST_MAX_DOCUMENTS`, `GUEST_MAX_UPLOAD_BYTES`). Usage is reported by
  `GET /api/auth/status` → `guest_usage`, so the UI can show "1 of 3 files"
  rather than teaching the rule through a rejected upload. Read it from
  `guest_usage.documents`, **never** from `len(/api/documents)`, which counts
  the demo files and reads two too high.
- Guests are reaped after **30 minutes idle** (`GUEST_IDLE_TTL_SECONDS`).

**What a guest may not do** — enforced server-side by `require_account`, not
merely hidden in the UI. Exactly seven routes:

| Denied | Why |
|---|---|
| `PUT /api/eval/config`, `POST /api/eval/config/reset` | `config_overrides` is a **single row shared by the whole deployment**. A config write re-points the embedding model for *everyone* and invalidates their chunks. |
| `POST /api/eval/run`, `POST /api/eval/step` | Spends a full scored benchmark of real LLM quota. |
| `POST /api/folders`, `POST /api/folders/{id}/rescan` | A folder path names the **server's** filesystem, not the visitor's. Also bypasses the document cap — one scan ingests a tree. |
| `POST /api/documents/reindex` | Re-embeds from scratch, undoing the vector-copy saving. |

Everything else — upload, delete, ask, deep search, new chat, prune, theme — a
guest keeps. A guest workspace is a **trial, not a display case**.

**Private tier (signed in).** Google OAuth (§14) grants an uncapped workspace,
and signing in **promotes** the guest's work into the permanent account:
documents, folders, conversations and chunks are re-pointed, nothing is
re-embedded (`_promote_prior_guest`). This applies to every sign-in path.

**Identity has to paint before the server replies.** The session cookie is
`httponly`, so JavaScript cannot read it, and the topbar used to sit blank for
1.2s warm — about three seconds on a cold function — with neither guest badge
nor sign-in button. A second, non-secret `ragchat_kind` cookie carries only the
identity *kind* (`guest` / `account`); the boot script reads it synchronously
and paints the right topbar immediately, and `/api/auth/status` reconciles when
it lands. Being spoofable is harmless — every real authorisation decision stays
on the httpOnly session cookie. It must be cleared on logout and rewritten on
guest-login and promotion, or it outlives the session it describes.

**The messaging consequence.** The pitch is not "try a read-only demo". It is
**"start using it now, sign in when you want to keep it"** — which is why
guest-first has no downside if the visitor never signs in.

---

## 4. Demo corpus mechanics

`eval/corpus/` holds **27 documents**. **Exactly two are exposed to guests** —
`helios_energy_handbook.md` and `meridian_coffee_ops.md`
(`guests.DEMO_CORPUS_FILES`).

> **The two-file limit is deliberate and must not be "fixed".** The rest is
> business content that must never reach anonymous visitors. **Any landing copy
> promising more sample documents is wrong.**

- Embedded once into a `__demo_template__` account, then vector-copied per
  guest. This removes the 60s `maxDuration` problem from the visitor path
  entirely, along with the half-ingested-corpus-if-they-close-the-tab failure.
  Arriving just *opens* an already-populated workspace.
- Demo documents are **excluded from the guest cap** — they are the app's
  content, not the visitor's. A fresh guest has all 3 upload slots.
- `Document.source_text` must be set on the template **and** copied to each
  guest clone, or deep search (§10) is silently blind to the one corpus a
  first-time visitor will try it on.
- Re-run `python -m ragchat.demo_vectors` after any embedding-model or chunking
  change, or guest-login times out re-embedding on the visitor's request.

---

## 5. Guidance & inline help

No About page. Explanation lives where it is needed, layered.

- **Empty states are sequenced, not descriptive.** The app has a natural
  four-beat loop — *add a source → ask → click a citation → read the quality
  readout* — and nothing used to tell the user it existed. The empty states
  number those beats and light up as each completes.
- **Evaluation is never dead on arrival.** It used to show nothing until "Run
  benchmark" was pressed. A scored run now ships with the app (§11).
- **Glossary on demand.** Jargon gets a dotted-underline term with a tap/click
  definition — **not** a `title` attribute, which does not exist on touch and
  strands mobile users.
- **Long jobs get persistent status, not disappearing toasts.** Re-index, folder
  scan and benchmark announce themselves in a status row that stays until the
  job ends.
- **Wording avoids insider terms.** "Prune ghosts" became a plain-language
  command in the `/` palette; it was a glossary entry for a button, which is a
  sign the button was named for the implementation rather than the user.

---

## 6. Per-answer quality readout

One ~20px composite indicator under each answer, not the four labelled rows
with glosses it replaced — those carried roughly as much visual weight as the
answer itself, the exact inversion §1 exists to prevent.

```
● Grounded    0.55    compare
```

- **State + one word:** `Grounded` / `Weak` / `Ungraded` / `Grading…`.
- **Clicking it** puts that answer's readings on the benchmark bars in the
  Evaluation pane, so scrolling back through a conversation still works.

### The four states are not interchangeable

- **`Ungraded` means a grader broke** — a judge that 404s, times out or replies
  without a verdict. `evalData.faithful` / `.relevant` are nullable for exactly
  this reason, and rendering "FAIL" there would be a confident false claim about
  the user's own documents.
- **`Grading…` means the verdict has not arrived yet**, which is now the normal
  state of every answer for a few seconds (§10). Drawing it as `Ungraded` would
  report a fault on every single question.

### PASS is not green

Lime is the accent, and lime beside green is unreadable as two distinct
meanings. So:

- **PASS** → a checkmark glyph in neutral foreground colour, not a green pill.
- **FAIL** → stays red. Red is reserved exclusively for failure and destructive
  actions.
- **Lime** means *active / primary / live* — primary buttons, indexing in
  progress, grading in progress. It never means "good".

---

## 7. Settings — presets first

Fifteen controls with no inline explanation and no indication of a sensible
value was a wall, not a panel.

- **Default view:** four named presets that set the whole config at once —
  **Fast / Balanced / High accuracy / Low cost** (`ragchat/presets.py`) — each
  with a one-line description of the tradeoff it makes.
- **"Advanced" toggle** reveals the full field set, with a one-line description
  and range hint per field.
- **"Needs re-index" consequences stay visible.** The badge on the Chunking and
  Embedding fieldsets must survive any reorganisation — changing those
  invalidates every existing chunk.
- **Read-only for guests**, enforced server-side (§3).

> **Saving Settings writes one row that replaces `config.yaml` entirely, for the
> whole deployment.** `GET /api/health` reports the *effective* config and which
> keys a save is pinning. **Settings → Reset to shipped defaults** puts the file
> back in charge. This is the single most surprising behaviour in the app; see
> `CLAUDE.md` for why env vars are not the source of truth either.

---

## 8. Desktop layout

Three panes; weight redistributed toward the answer.

- **Chat (centre)** gets the largest share and the content register.
- **Sources (left)** is a clear second — card treatment with type pill, status
  pill and citation count, computed client-side from the loaded conversation.
  Substantial, but visibly subordinate.
- **Evaluation (right)** is reference material you visit deliberately; the
  per-answer indicator (§6) carries the everyday signal.
- **Excerpt** is the payoff for clicking a citation, and part of the content
  register.

Panes are drag-resizable, collapsible, and their widths persist. That behaviour
is what `layout_check.mjs` exists to verify — a screenshot cannot see it.

### The benchmark view

When the Evaluation pane is open there is no answer competing for attention, so
this one in-app surface gets display-scale numerals (§1). Each metric is drawn
as a bar with the published benchmark as a reference tick and the current
answer's reading as the fill, so "is this answer normal?" is a glance rather
than an inference. Latency is reported on its own line, because it is a speed
and not a quality and has no honest counterpart to be drawn against.

### Command palette

`⌘K` on a Mac, `Ctrl+K` elsewhere, and a `/` button for people who do not know
either. Jump to a chat or a source, open a setting, toggle the theme, start a
new chat, prune. **Desktop only** — it has no meaning on touch and must not
consume mobile screen space.

---

## 9. Mobile — a different product shape

Not a reflow of the desktop layout. The phone is a **reader**; the desktop is a
**workbench**.

**Primary:** the conversation. Ask, read, tap a citation, read the excerpt,
glance at the quality indicator.

**Available but demoted:** new chat, delete chat, delete a source, re-index a
single source.

**Desktop-only:** benchmark runs, folder scan, re-index all, advanced tuning
fields, the command palette. Presets stay reachable.

**Mechanics:**
- One pane at a time, chosen from a bottom tab bar — never three panes stacked
  into one endless scroll.
- 44×44px minimum tap targets. Hover-revealed actions **do not exist on touch**
  and must be persistent or behind an overflow menu.
- `100dvh`, not `100vh`; `env(safe-area-inset-bottom)` on the ask bar; 16px
  minimum on inputs or iOS Safari zooms on focus.
- Full-width composer. No horizontal scroll at any width — verified down to
  320px, which is where the topbar controls collided until the wordmark was
  dropped below 360px.

---

## 10. The pipeline

```
ingest:  load → chunk (512 tokens, 75 overlap, recursive) → embed (768-dim)
ask:     rewrite → retrieve (vector + keyword, RRF k=60) → rerank → generate
grade:   faithfulness + relevancy          ← a SEPARATE request
```

**Retrieval is two searches, not one.** Vector search finds meaning; keyword
search finds exact strings. Their rankings are fused with reciprocal rank
fusion, so an invoice number and a paraphrased policy are both findable. It is
unconditional — `hybrid_search` is hardcoded in `load_config()` because there is
no trade worth offering: it costs no model call (Postgres full-text on Neon,
in-process BM25 on Chroma).

**Reranking is one call for the whole pool.** Cohere `rerank-v3.5` via
OpenRouter. The old LLM cross-encoder scored one chat call *per passage* —
slower and dearer for no measured gain. *Whether* to rerank is a setting; which
vendor does it is not.

**The rewrite must leave a standalone question alone.** Told only to "resolve
references", the model resolves ones that were never there: after two turns
about a solar battery, *"What is the boiler pressure range for the espresso
machine?"* came back — deterministically — as *"…for the SunPak 5 espresso
machine"*, retrieval hunted for a product that does not exist, and the reader
was told the fact was not in their documents while it sat one paragraph from an
answer they had already been given. The prompt now forbids importing a subject
from an earlier turn. **This failure is invisible one question at a time**,
because rewriting is skipped entirely when there is no history.

**Deep search does not rank.** Ordinary search orders passages and takes the
best few, so anything below the cut is invisible to the answer. Deep search
(`ragchat/deepsearch.py`) reads every document the user owns word for word and
returns every literal match. It rides on the ask request (`AskIn.deep_search`)
and **nothing about it is ever written** — `config_overrides` is one row shared
by the whole deployment, so a stored toggle would be a deployment-wide switch
wearing the costume of a personal preference. That is precisely the bug the web
-augmentation toggle it replaced had.

Two details it would be easy to lose: Python's `\w` matches CJK, so a
whitespace tokenizer turns a whole Chinese clause into one "word" present in no
document and the feature looks like a correct "not in the corpus" rather than a
bug — hence character bigrams. And duplicate passages measurably *degrade*
answers, so `_drop_duplicates` runs over the merged pool.

**The answer is delivered before it is graded.** The two judges are two more
sequential model calls and, measured against the live provider, they cost more
than writing the answer did — 10.1s of grading against 7.8s of answering, 56% of
the wait, to score something the reader is already looking at. `POST /ask`
returns as soon as the answer exists; `POST /chats/{id}/messages/{id}/grade`
runs the judges a request later. Two things make that safe, both tested in
`tests/test_grade_split.py`:

- the judges see the passages the answer was **built from** — the context
  travels on the message, because a request later retrieval is gone and
  re-running it would grade the answer against passages the model never saw;
- grading is **idempotent** — a client retry after a timeout must not spend two
  more judge calls, and must never replace a verdict with a fresh failure.

**Refusal is measured, not incidental.** `ask()` refuses before generating when
every scored passage falls below `similarity_threshold` — and deliberately does
**not** refuse when no chunk carries a cosine at all, because a keyword-only
pool is the absence of evidence rather than evidence of absence. That is exactly
the part-number case fusion exists for.

### The escalation — the app reaching for a second tool by itself

The first place this pipeline **chooses an action** instead of executing a fixed
line. `ask()` is always handed the deep-search tool; `force_deep` only says
whether the visitor is holding the switch down. Left off, `ask()` decides.

It escalates at exactly the two points retrieval already knows it failed:

| Trigger | Where | What it does |
|---|---|---|
| `weak_retrieval` | nothing cleared `similarity_threshold`, about to refuse | scan, and if anything comes back, generate instead of refusing |
| `model_refused` | the model read the passages and replied `NOT_FOUND_ANSWER` | scan, and if anything comes back, rebuild the pool and answer **once** more |

Three properties make it safe, and all three are asserted in
`tests/test_escalation.py`:

- **Free on the happy path.** A question that was going to be answered never
  touches the tool. The scan itself costs *no model call* — it reads
  `Document.source_text` and matches literally — so the only real cost is one
  extra generation, spent solely on questions that were about to fail anyway.
- **Provably finite.** `scanned` can go true once, so the tool is used at most
  once and the answer is generated at most twice. A serverless function frozen
  the instant it responds, with 60 seconds to work in, cannot host an
  open-ended `while`: this is a ladder with one rung.
- **It degrades to what came before.** A failing scan, or a failing retry, keeps
  the refusal that already existed. Neither can turn a truthful answer into a
  500.

**It is deliberately not driven by the judges,** which is the counter-intuitive
part. `NOT_FOUND_ANSWER` is entirely faithful to its context and squarely
answers the question, so *both judges pass it* — the grader is blind to exactly
the failure deep search fixes. The judges also arrive a request later now, by
which time the reader is already reading. Retrieval confidence and the model's
own refusal are both free, both earlier, and both actually correlated with the
thing being fixed.

**And it says so.** An answer that exists because the machine decided to try a
second tool is not the same object as one that came straight back, and the
reader has no other way to tell. `eval_data.escalated` carries the reason and
the UI renders a line under the answer. A refusal *after* a scan also makes the
larger claim — "I also searched every document word for word" — because that is
a materially stronger statement than giving up after a ranked search.

### What is still missing

The trigger is a **rule**, not a judgement: the app decides *when* to look
harder, but the choice of tool is still hard-coded, because there is only one
tool to choose. Real tool selection needs a third option to select between.

The wiring is shaped for it: `ask(..., deep_search=<callable>)` takes a tool
rather than a boolean, every passage in the pool is chunk-shaped and carries
`similarity: None` when it has no measured distance, and `_drop_duplicates`
exists because any observe step needs it.

**Web search was deleted, not foreclosed.** It came back as an answer to "your
documents did not have it" by looking somewhere else instead of looking harder,
and it was a deployment-wide toggle wearing the costume of a personal one. When
it returns it is a **tool the model may call**, its passages are labelled in the
prompt and in the citation because they are the only ones not from the user's
own documents, and it does not go in `config_overrides`.

The constraint that shapes all of it: the function is frozen the instant it
responds and `maxDuration` is 60s, so a reason/act/observe loop cannot be an
open-ended `while`. Bound the iterations hard, or slice it across requests the
way the benchmark does.

---

## 11. Evaluation

**56 golden questions over 27 documents** — 53 answerable (6 of them spanning
two documents) and 3 deliberately unanswerable, because refusing to answer is a
behaviour worth measuring. Most of the corpus is confusable on purpose: same
firm, same vocabulary, different facts, so retrieval has to discriminate rather
than recognise a topic. Expanding it made the scores *fall*, which was the point.

**The scored result ships with the app.** `eval/published_run.json` is committed
and rendered on first paint; nobody runs a benchmark to see it, and the pane is
never empty. Republish with `python -m eval.published --from-run <ts>` — and
move it in the **same commit** as any model change, or the app displays scores
for a pipeline it is not running.

**`eval/build_golden_set.py` refuses to write** unless every golden passage is a
verbatim substring of a source document *and* absent from every distractor. A
passage present in two places gives the corpus two right answers, and the exact
metrics then score the same retrieval as a hit or a miss depending on which copy
came back.

### Targets

Used for the scorecard bars (`EVAL_TARGETS` in `frontend/app.js`).

| Metric | Target | Reads as |
|---|---|---|
| Context Recall | ≥ 0.80 | Found the right passages |
| Precision@k | ≥ 0.70 | Sent mostly relevant text |
| MRR | ≥ 0.65 | Best passage ranked high |
| NDCG@k | ≥ 0.70 | Good overall ordering |
| Hit rate@k | ≥ 0.80 | Right passage made the cut |
| Faithfulness | ≥ 0.90 | Stuck to the sources |
| Answer relevancy | ≥ 0.85 | Answered what was asked |
| Answer correctness | ≥ 0.80 | Matched the known answer |
| Not-found rate (unanswerables) | ≥ 0.90 | Refused what it should refuse |

A tuning change counts as an improvement only if retrieval **and** generation
rise without the not-found rate dropping.

### Two metric families, on purpose

Cosine matching has a **directional bias**: its false-positive rate rises with
corpus size, so adding filler documents makes scores go *up*. It is reported
because it is the RAGAS-comparable number, and it is **never** what CI gates on.
The deterministic `exact_*` metrics exist for that, and `eval/gate.py` watches
only those.

### Fail open, never closed

A judge that 404s, times out or replies without a verdict is a **broken grader**,
not a failed answer. `judges._parse_verdict` raises `JudgeError`;
`pipeline._eval_answer` and `run_eval._safe_judge` turn that into `None` plus a
reason; `aggregate()` excludes ungraded rows from the mean and counts them in
`n_ungraded`. The published run carries 8 of them and shows them rather than
quietly dropping them.

Judges are also thinking-model sensitive: keep `max_tokens` generous (reasoning
tokens bill against it) and strip `<thinking>` wrappers before parsing.

### Why a benchmark run is sliced

`POST /api/eval/run` creates a row and returns; `POST /api/eval/step` does one
bounded unit — index one corpus file, or score `EVAL_BATCH_DEFAULT` questions —
commits, and returns progress. The browser calls `/step` until the run is no
longer `running`. Closing the tab pauses it; reopening resumes from the last
committed slice.

**Slice size is latency-bound, not a preference.** One scored question costs
40–54s; a batch of 2 measured 83s and blew the 60s ceiling. That failure is
silent and self-perpetuating — an overrunning step is killed *before* it commits,
so the client retries the identical slice forever. If judge latency grows, slice
the judges across steps rather than raising the batch.

---

## 12. Running it locally

```bash
# Backend (port 8000)
.venv/Scripts/python -m uvicorn ragchat.app:app --reload --port 8000

# Frontend (port 5173, proxies /api to the backend)
cd frontend && npm install && npm run dev
```

In dev the pages are `/`, `/app.html` and `/built.html` (§2).

```bash
# Tests — no network, ~13s
.venv/Scripts/python -m pytest tests/ -q

# Retrieval metrics react to retrieval quality (standalone script, not pytest)
.venv/Scripts/python eval/check_metrics.py

# Screenshots: 3 pages x 5 breakpoints x 2 themes. Fails on horizontal overflow,
# any page error, a grid left with one item under a full row, or overlapping
# controls. Needs the root manifest: npm install
node shot.mjs http://localhost:5173

# Workspace layout behaviour a screenshot cannot see — drag-resize, collapse,
# persistence. Exits non-zero on the first failed assertion.
node layout_check.mjs http://localhost:5173

# Benchmark: retrieval only (free), or the full run (spends model calls)
.venv/Scripts/python -m eval.run_eval --retrieval-only
.venv/Scripts/python -m eval.run_eval --retrieval-only --with-rerank --ceiling
.venv/Scripts/python -m eval.run_eval

# Publish a completed run as the shipped result
.venv/Scripts/python -m eval.published --from-run latest

# Regenerate the CI baseline after ANY corpus / golden set / chunking / model
# change. Read the diff before committing — regenerating blindly launders a
# regression. --from-run avoids paying for the same measurement twice.
.venv/Scripts/python -m eval.baseline --from-run latest

# What CI runs: passes loudly when a provider is down, fails on a regression
# and on a baseline that no longer describes the pipeline.
.venv/Scripts/python -m eval.gate

# Compare two runs metric by metric
.venv/Scripts/python -m eval.compare --last-two

# Refresh demo vectors after an embedding-model or chunking change
.venv/Scripts/python -m ragchat.demo_vectors
```

**A stale local `config_overrides` row silently invalidates local measurement.**
One carrying `qwen3-embedding-8b` without the `qwen/` prefix changed the
fingerprint and made every local benchmark unrepresentative. If local and
deployed fingerprints differ, suspect this first.

**Tests can silently start hitting the network.** `test_retrieval_fixes.py`
monkeypatches `ProxyEmbeddings` at module level, and by the time the whole suite
has run, other modules have re-imported `ragchat.embeddings` and put the real one
back. Stub `retrieve` and `_eval_answer` rather than indexing a corpus. The suite
is fast with no network; if it takes a minute, something is calling out.

---

## 13. Deploying — Vercel + Neon

`api/index.py` does `from ragchat.app import app` and exposes that ASGI app for
`@vercel/python`; the FastAPI app is not rewritten. Dependencies come from
`requirements.txt` at the repository root. `vercel.json` sets `memory: 1024` and
`maxDuration: 60` — the Hobby default of 10s is not enough for a single scored
question. The frontend is built with `cd frontend && npm install && npm run
build`; the root `package.json` is for the Playwright checks only and is never
installed at deploy time.

### Run the function where the database is

`"regions": ["sin1"]` in `vercel.json`, because the Neon database is in
Singapore. Without it the function ran in US East while the data sat in
Singapore, and **every round trip crossed the Pacific twice** on a chunks table
holding 240 rows:

```
                     iad1 (before)     sin1 (after)
per round trip       ~420ms            ~3ms
guest sign-in        0.93s             0.05–0.31s
guest-seed           9.2s              0.46s
document list        0.94s             0.065s
document delete      2.74s             0.08s
cold start           25–31s            ~9s
```

Everything attempted before that — memoising the schema check, dropping
`pool_pre_ping`, sharing the engine, adding a delete index — was real work
shaving milliseconds off operations dominated by an ocean. Two rounds of
application-level reasoning about the delete path were both wrong for that
reason, and only instrumenting the path (`lookup 3ms, chunks 1335ms, row 6ms`)
found it.

**If the database moves, move this with it.** `GET /api/health` reports
`function_region`; `db_region` is null because the injected hostname carries no
region, so latency is the signal.

### Nothing periodic runs inside the app

A serverless function is frozen the instant it responds, so it cannot run a
timer, and Vercel's Hobby cron fires once per **day** — which cannot honour a
30-minute guest TTL. `.github/workflows/guest-sweeper.yml` calls
`POST /api/admin/sweep-guests` every 15 minutes. Work done in front of a waiting
visitor is a backstop only and stays tiny: `guests.INLINE_REAP_LIMIT` is 2, and
it was 20, which put 39.7s in the path of a first page load.

### Environment

Required: `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `DATABASE_URL` (or
`PG_DATABASE_URL`), `SESSION_SECRET`, `VECTOR_BACKEND=neon`.

Optional: `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` / `GOOGLE_REDIRECT_URI`
(§14), `PROXY_BASE_URL`, `RAG_LLM_MODEL`, `EMBEDDING_PROVIDER`,
`RERANKER_PROVIDER` — the last three are **boot defaults only**, overridden by
the Settings row (§7).

`GUEST_SWEEP_SECRET` is the one that needs setting in **two** places, or the
sweeper is a no-op and guest workspaces outlive their TTL. Unset means the route
is **disabled (404), not open** — the safe failure, but still a failure.

```bash
vercel env add GUEST_SWEEP_SECRET     # the deployment
gh secret set GUEST_SWEEP_SECRET      # the caller
```

Set the repo variable `APP_URL` if the deployment is not `rag-gel.vercel.app`.

### Check `/api/health` after every deploy

It reports the **effective** runtime config, not just which env vars are set.
Confirm three things:

1. `judge.model` **==** `effective_config.llm_model`. Diverged, the judge grades
   with a model the app is not answering with — and if the judge's id is not
   served, every judge call 404s and the whole scorecard comes back ungraded.
2. `embedding_models_by_provider.gemini` has **exactly one** entry
   (`models/gemini-embedding-001`). More means the 768 allowlist was bypassed,
   and a non-768 model reaching the UI fails at insert against the fixed
   `vector(768)` column.
3. `effective_config.embedding_provider` is the provider you intend. It comes
   from the DB override, not the env var.

---

## 14. Google sign-in setup

Optional. The app works fully signed out, and when `GOOGLE_CLIENT_ID` /
`GOOGLE_CLIENT_SECRET` are unset, `/api/auth/status` reports
`google_oauth: false`, the button stays hidden, and nothing breaks.

`ragchat/auth.py` implements the flow with `itsdangerous`-signed cookies — no
`authlib`, no Starlette `SessionMiddleware`.

1. **Google Cloud Console → APIs & Services → OAuth consent screen.** User type
   *External* (*Internal* for a Workspace-only app). The default scopes are
   enough: `openid`, `email`, `profile`. While in *Testing*, add your account
   under Test users, or publish the app.
2. **Credentials → Create Credentials → OAuth client ID → Web application.**
   Authorised redirect URIs — note the `/google/` segment:
   ```
   https://rag-gel.vercel.app/api/auth/google/callback
   http://localhost:5173/api/auth/google/callback
   http://localhost:4173/api/auth/google/callback
   ```
   (5173 = `vite dev`, 4173 = `vite preview`; both proxy `/api` to FastAPI.)
3. **Set the env vars** in the Vercel dashboard and in a local `.env`:
   `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`.

`GOOGLE_REDIRECT_URI` is **not** derived from the request — unset, the app sends
an empty `redirect_uri` and Google rejects the consent request. It must match
the registered URI *character for character*; `redirect_uri_mismatch` is almost
always a typo here.

Generate a session secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Without it `settings.session_secret` falls back to `"dev-session-secret"`, and
anyone who knows that placeholder can forge a session cookie.

**Keep production deployment protection OFF** (Project → Settings → Deployment
Protection → Vercel Authentication: *Disabled* for Production). Leaving it on
stacks a Vercel SSO login in front of the site and blocks Google's redirect back
to `/api/auth/google/callback` for anyone outside your Vercel team. Preview
protection is fine to keep — just register that preview URL's callback too.

---

## 15. Deliberately not built

- **The agentic tool loop** (§10) — the destination, not a gap.
- **Web search as a tool** — deleted as a fallback, expected back as a tool.
- **Streaming answers.**
- **Retrieval upgrades:** Contextual Retrieval, HyDE, multi-query, parent-child.
- **Server-side per-source citation counts.** Client-side tallying covers the
  visible count; a `n_citations` field on `GET /api/documents` would make it
  accurate across chats.
- **Deep search earns nothing on the sample corpus.** With six passages ordinary
  search already returns all of them, so there is nothing left to rescue — and
  the escalation therefore almost never fires for a guest. It pays off on real
  document sets, and that is an honest limitation rather than a bug to chase.
- **Cold start is ~9s** and most of what remains is the hosting plan. The page
  is built to stay honest while it waits.

---

## 16. Things learned the hard way

`CLAUDE.md` carries the constraints; this is the measurement record behind them.

**The ruler was broken before the search was.** `MATCH_THRESHOLD` was never
calibrated and at 0.6 scored **79.6% of true containments as misses**. The cause
is length, not language: a one-line passage against a ~500-token chunk scores low
even when verbatim inside it (Latin fared *worse* than CJK). Most of this repo's
historically low retrieval scores were a broken measurement.

**The harness measured the wrong list for months.** `retrieve()` returns
`candidate_k` chunks and `ask()` reranks to `top_k`; context recall was computed
over all of the former while precision/MRR/hit sliced the latter — two different
retrievals reported side by side. And `score_item` never reranked at all, so a
preset with `reranker: True` scored exactly as if it were False.

**The full benchmark had never run.** A rename left one stale reference on a line
only the full run reaches, and every full run had been dying there. 211 tests
missed it; running it once found it.

**A feature can make answers worse.** Deep search handed the model text the
ranked search had already given it, and one question went from right to "I
couldn't find this". Duplicates are not free.

**Two clever fixes that changed nothing.** Sharing the engine, adding an index —
both plausible, both irrelevant, because the cost was distance (§13). Instrument
before theorising.

**Grading cost more than answering** (§10), and the number the UI showed for
"Answered in" included it — making a graded answer look twice as slow as a
not-found one, which returned earlier and had always been honest.

**A bulk DELETE leaves the session's identity map intact.** `SessionLocal` sets
`expire_on_commit=False`, so after `query(...).delete()` a `db.get(User, id)`
still returns a live-looking object for a row that is gone. `purge_users`
expunges them — expunge and not expire, because an expired instance re-SELECTs on
the next attribute read and raises.

**Documentation goes stale silently, so it is tested.** The landing page claimed
a 10-document corpus for as long as it had 27; the README advertised
`gemini-3.5-flash-lite` one commit after the model was reverted. Both are now
asserted against the files they describe (`tests/test_landing_claims.py`), and a
failing test names the file to edit.

---

## Current defaults

| | |
|---|---|
| Generation + judges | `models/gemma-4-26b-a4b-it` (Google AI Studio) |
| Embeddings | `qwen/qwen3-embedding-8b` (OpenRouter), 768-dim |
| Reranker | Cohere `rerank-v3.5` (OpenRouter) |
| Chunking | recursive, 512 tokens, 75 overlap |
| Retrieval | `top_k` 4, `candidate_k` 20, threshold 0.0, hybrid + rerank on |

Gemma is kept over `gemini-3.5-flash-lite`, which was measured over the full
golden set and lost 6.5 points of answer correctness (0.8696 → 0.8049), 2 points
of faithfulness, and nearly doubled the answers it could not grade at all
(8 → 14 of 53) — for no measurable speed gain. Retrieval was identical either
way, so the whole difference was the model's own work.
