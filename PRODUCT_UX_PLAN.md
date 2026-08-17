# Product & UX Plan — agentic-RAG

**Status:** Proposal for review. No code changes yet. *(Revision 3 — calibrates
how closely we follow the visual reference; see §13 for the change log.)*
**Supersedes:** parts of `UI_UPGRADE_PLAN.md` (see §10).
**Visual reference:** [joeynyc/Grok-UI](https://github.com/joeynyc/Grok-UI) —
specifically its "Operator" (carbon black / signal lime) and "Event Horizon"
themes.

---

## 0. Decisions locked

| Question | Decision |
|---|---|
| Audience | **Portfolio / showcase piece** |
| Theming | Dark **and** light, toggleable — **dark is the default**, both verified at every step |
| Accent | **Acid lime**, and **PASS stops being green** (§6) |
| Entry | **Landing page at `/`**, app at `/app` |
| Landing content | Hero + value prop + CTA · pipeline diagram · tech/architecture notes |
| Landing style | **Near-literal Grok-UI homage** — full HUD, space backdrop, display numerals (§2) |
| Display numerals in app | **Expanded benchmark view only** (§6) |
| Extra HUD markers | **Command palette (⌘K)** only — no numbered nav, breadcrumb or grid motif |
| First-run | **Guest-first** — no sign-in wall; a private workspace is provisioned on arrival (§3) |
| Access model | **Private writable guest workspace; sign in to keep it** (§3, revision 4) |
| Explanatory content | **Inline in the panes** — no About route |
| Explanation depth | **Layered** — plain language by default, expandable detail |
| Settings | **Presets first**, expert fields behind an "Advanced" toggle |
| Visual hierarchy | **Answer is hero**, sources a clear second, eval is instrumentation |
| Evaluation pane | **Collapsed by default**, expandable (§8) |
| Answer typography | Large sans, generous leading |
| Per-answer eval | **Dot + word + benchmark tick bar**, expands on tap |
| Mobile | Genuinely good, and a **different product shape** from desktop |

---

## 1. The core principle — two visual registers

Everything below follows from this. The Grok-UI aesthetic is **not the skin for
the whole app** — it is the skin for the *instrumentation layer only*.

**Content register** — the answer, its citations, the conversation.
Large sans (~17–18px), `line-height: 1.65`, high contrast, generous spacing.
Reading-optimized. **Dominates the screen.**

**Telemetry register** — eval scores, latency, similarity, chunk counts, model
ids, index status. Mono, uppercase, `letter-spacing: 0.12em`, ~11px,
deliberately *low* contrast. This is where the reference's HUD vocabulary lives:
numbered labels, hairline borders, notched corners, accent underline bars.

The contrast between them is the whole design. A mono micro-label reads as
instrumentation *because* it sits beside something large and humane. If
everything is HUD, nothing is — and the answer, which is the actual product,
drowns in its own dashboard.

**Scope of this rule.** It governs surfaces where an *answer* is present — the
chat thread and anything rendered alongside it. Display-scale (48–72px) numerals
are therefore banned from the inline per-answer eval: those are technical values
sitting next to prose, and they must stay small.

The rule does **not** govern surfaces with no answer to protect. On the landing
page (§2) and in the deliberately-expanded benchmark view (§6), telemetry *is*
the content, and the reference's display numerals belong there. Revision 2
stated this ban too broadly; §13 records the correction.

### How closely we follow the reference

Grok-UI is a monitoring dashboard — roughly 100% telemetry, because the product
*is* telemetry and there is no prose to protect. This app's payload is a
paragraph someone reads. So we keep the reference's **vocabulary** in full and
deliberately invert its **proportion**.

- **Kept:** layered near-blacks, hairline borders, no shadows (depth from value
  steps), mono uppercase letterspaced micro-labels, notched corners, accent
  underline bars, acid lime as the single hot accent, multiple themes.
- **Concentrated rather than dropped:** display numerals, big geometric hero
  type, space backdrop, grid motifs — these move to the landing page (§2) and
  the benchmark view, where they cost the reader nothing.
- **Dropped:** far-left numbered nav rail, breadcrumb strip, decorative
  radar/grid inside the workspace, Privacy Mode.
- **Added, not in the reference's role:** command palette (§8) — Grok-UI has one
  as navigation across twelve areas; here it earns its place differently, as a
  fast jump across chats, sources and settings.

---

## 2. Landing page (`/`)

A separate static document. Ships **no app JS** and makes **no API calls** — it
is CDN-served, costing zero serverless invocations for visitors who never enter.

**Style: a near-literal Grok-UI homage.** This page has no long-form content to
protect, so it carries the reference's full visual force — and it is what makes
the app read as *that thing you liked* rather than as generic dark SaaS. Every
marker the workspace forgoes lives here:

- Space backdrop behind the hero.
- Large soft geometric hero type, the reference's "Your Grok, at a glance" move.
- Numbered section labels (`01 / 02 / 03`) in the telemetry register.
- Grid-lined feature card and `A1–A4` metric tiles with **display-scale
  numerals** — populated with real figures (corpus size, chunk count, benchmark
  faithfulness and relevancy rates), not invented ones. Static at build time or
  from a cached snapshot, so the page still makes no live API calls.
- Notched corners, hairline borders, accent underline bars, lime accent.

Because the landing and the app share the same tokens, type system and accent,
the two read as one product even though their proportions differ by design.

**Sections:**

1. **Hero** — product name, one-line value prop, two buttons:
   - Primary: **"Try it with sample documents"** → `/app?demo=1`
   - Secondary: "Sign in for your own workspace" → `/app`
2. **Pipeline diagram** — inline SVG walkthrough of
   `ingest → retrieve → rerank → generate → judge`. Carries most of the layered
   explanation and is the single most portfolio-valuable element on the page.
   Each stage labeled in the telemetry register, captioned in plain language.
3. **What makes it non-trivial** — hybrid BM25 + vector fusion, LLM
   cross-encoder reranking, RAGAS-style judging. These currently exist only
   inside `title` tooltips and the settings modal; they are the substance and
   are presently invisible.
4. **Architecture / stack** — FastAPI on a single Vercel function, Neon Postgres
   + pgvector, the sliced-job pattern for long runs, the 768-dimension
   invariant, fail-open evaluation.

**Theme:** the landing page respects the same `data-theme` value as the app and
reads it from the same `localStorage` key, so a light-theme user does not get a
dark pitch page followed by a light app.

**Route back to the story:** the app's help affordance links to `/` so the
pipeline diagram stays reachable from inside the workspace. This is what
replaces the About page we chose not to build.

### Routing mechanics

Two edits total. No router, no framework.

```javascript
// frontend/vite.config.js
build: { rollupOptions: { input: { main: 'index.html', app: 'app.html' } } }
```

```json
// vercel.json — add to "rewrites"
{ "source": "/app", "destination": "/app.html" }
```

Today's `frontend/index.html` moves to `frontend/app.html` unchanged; the new
`index.html` is the landing page.

---

## 3. Access model — **BUILT** (revision 4)

> Revisions 1–3 specified a *shared, read-only* demo corpus. That is **not what
> was built**, and the delivered design is better. This section now describes
> the shipped behaviour; §13 records what changed and why.

**Guest tier (no sign-in).** Every anonymous visitor is auto-provisioned their
**own private, writable, throwaway workspace** (`ragchat/guests.py`). Not a
shared account — the app scopes every query by user id, so one shared guest
account would let strangers read and delete each other's uploads, the exact
mixing per-user isolation exists to prevent.

- The demo corpus is embedded **once** under a template account and then
  **vector-copied** per guest. No embedding calls, no quota spend, no latency
  on arrival — and each visitor's copy is genuinely theirs.
- Guests **can upload**, capped at **3 documents / 5 MB**. Usage is reported by
  `GET /api/auth/status` → `guest_usage`, so the UI shows "1 of 3 files" rather
  than teaching the rule through a rejected upload.
- Chats are per-visitor by construction: each guest is already its own account,
  so the "ephemeral chats" problem revision 2 identified never arises.
- Guests are reaped after **2 hours idle**, opportunistically on guest creation
  (Vercel Hobby has no cron, and a background thread would be frozen — CLAUDE.md).

**What a guest may NOT do** — enforced server-side by `require_account`
(`ragchat/app.py`), not merely hidden in the UI:

| Denied | Why |
|---|---|
| `PUT /api/eval/config` | `config_overrides` is a **single shared row** (`db.py:121`). A config write re-points the embedding model for *everyone* and invalidates their chunks. |
| `POST /api/eval/hybrid-search`, `/web-augmentation` | Same shared row. |
| `POST /api/eval/run`, `/step` | Spends ~46 scored questions of real LLM quota. |
| `POST /api/folders`, `/rescan` | A folder path names the **server's** filesystem, not the visitor's. Also bypasses the document cap — one scan ingests a tree. |
| `POST /api/documents/reindex` | Re-embeds from scratch, undoing the vector-copy saving. |

Everything else — upload, delete, ask, new chat, prune, theme — a guest keeps.
A guest workspace is a **trial, not a display case**.

**Private tier (signed in).** Google OAuth grants an uncapped workspace. Signing
in **promotes** the guest's work into the permanent account: documents, folders,
conversations and chunks are re-pointed, nothing is re-embedded
(`app.py:_promote_prior_guest`). This applies to every sign-in path, not just
Google.

**The messaging consequence.** The CTA is no longer "try a read-only demo". It
is **"start using it now, sign in when you want to keep it"** — a materially
better pitch, and it is why guest-first has no downside if the visitor never
signs in.

---

## 4. Demo corpus mechanics — **BUILT**

`eval/corpus/` holds ten documents. **Exactly two are exposed to guests** —
`helios_energy_handbook.md` and `meridian_coffee_ops.md`.

> **The 2-file limit is deliberate and must not be "fixed".** `guests.py` states
> the other eight are real business content that must never reach anonymous
> visitors. **Any landing copy promising ten sample documents is wrong.**

- Embedded **once** into a `__demo_template__` account, then vector-copied per
  guest. This removes the 60s `maxDuration` problem from the visitor path
  entirely, along with the half-ingested-corpus-if-they-close-the-tab failure.
- Arriving at the app just *opens* an already-populated workspace. Instant.
- Demo documents are **excluded from the guest cap** — they are the app's
  content, not the visitor's. A fresh guest has all 3 upload slots.
  Consequently the UI must read usage from `guest_usage.documents`, **never**
  from `len(/api/documents)`, which reads two too high.
- **Three suggested questions** answerable from those two files render as
  clickable chips in the empty chat state. *(Still to do — step 10.)*

---

## 5. Guidance & inline help

No About page. Explanation lives where it is needed, layered.

- **Empty states become sequenced, not descriptive.** Current copy
  (`sources-empty`, `empty-hint`, `excerpt-empty`) is well written but passive.
  The app has a natural four-beat loop — *add a source → ask → click a citation
  → read the quality readout* — and nothing currently tells the user it exists.
  The empty states should number those beats and light up as each completes.
- **Evaluation stops being dead on arrival.** It currently shows nothing until
  "Run benchmark" is pressed, with no explanation of what a benchmark is, how
  long it takes, or that it spends API calls.
- **Glossary on demand.** Undefined jargon throughout: *Prune ghosts,
  Candidate-K, Keyword fusion, Re-index, Excerpt, Golden set, faithfulness,
  relevancy*. Each gets a dotted-underline term with a tap/click definition —
  **not** a `title` attribute, which does not exist on touch and would strand
  mobile users. The topbar toggles have exactly this bug today.
- **Long jobs get persistent status, not disappearing toasts.** Re-index all,
  folder scan and benchmark currently announce themselves through a toast that
  vanishes ("Re-indexing all sources (this may take a while)…") and then report
  nothing until completion.

---

## 6. Per-answer quality readout

Today `buildEvalBlock` (`frontend/app.js:786`) renders **four labeled rows with
explanatory glosses under every answer** — roughly as much visual weight as the
answer itself.

Replace with a single ~20px composite indicator:

```
● Grounded    ▏▔▔▔▔▔╱▔▔▔┃▔▔▔▔▔▏      ← tick = benchmark baseline
```

- **State + one word:** `Grounded` / `Weak` / `Ungraded`.
- **Tick bar:** this answer's position with the benchmark baseline marked.
  Targets already exist — `eval/run_eval.py:334` publishes faithfulness ≥ 0.90
  and answer relevancy ≥ 0.85 — so the baseline is real, not invented.
- **Tap/click expands** to today's four rows (top similarity, faithfulness,
  relevancy, latency) with their glosses.

### PASS is no longer green

Acid lime is the accent, and lime beside green is unreadable as two distinct
meanings. So the pass/fail vocabulary changes:

- **PASS** → a **checkmark glyph in neutral foreground color**, not a green pill.
- **FAIL** → stays red. Red remains reserved exclusively for failure and
  destructive actions.
- **Lime** means *active / primary / live* — primary buttons, indexing in
  progress, benchmark running. It never means "good".

**`Ungraded` remains a first-class state and must never render as failure.** A
judge that 404s or times out is a broken grader, not a bad answer.
`evalData.faithful` and `evalData.relevant` are nullable for exactly this
reason, and rendering "FAIL" there would be a confident false claim about the
user's own documents. This matches the fail-open rule in `CLAUDE.md`.

---

## 7. Settings — presets first

Fifteen controls today (chunk size, overlap, splitter, top-K, candidate-K,
similarity threshold, temperature, embedding provider/model, reranker
provider/toggle, keyword fusion, query rewrite) with **zero inline explanation**
and no indication of a sensible value.

- **Default view:** three or four named presets that set the whole config at
  once — e.g. *Fast* / *Balanced* / *High accuracy* / *Low cost* — each with a
  one-line description of the tradeoff it makes.
- **"Advanced" toggle** reveals today's full field set, now with a one-line
  description and range hint per field.
- **"Needs re-index" consequences stay visible.** The existing `index-badge` on
  the Chunking and Embedding fieldsets must survive the reorganization —
  changing those invalidates existing chunks.
- **Read-only in demo tier**, enforced server-side.

---

## 8. Desktop layout

Three panes stay; weight is redistributed.

- **Chat (center)** gets the largest share and the content register. Answers set
  large with generous leading; citations are obvious, tappable affordances.
- **Sources (left)** is a clear second — real card treatment with type pill,
  status pill, and citation count. Substantial, but visibly subordinate.
- **Evaluation (right)** is **collapsed by default**, reachable in one click.
  The per-answer indicator (§6) carries the everyday signal, so the pane is
  reference material you visit deliberately. This resolves the contradiction in
  revision 1, which called the pane "quiet" while giving it a third of the
  screen — a 344px pane *is* a focal point however small its type.
- **Excerpt (bottom)** unchanged in role — the payoff for clicking a citation,
  and part of the content register.

### The expanded benchmark view

When the Evaluation pane is deliberately opened, there is no answer competing
for attention — telemetry *is* the content. So this one surface inside the app
gets the reference's **display-scale numerals**: faithfulness rate, answer
relevancy rate and mean top similarity at 48–72px, each with its published
target marked (`eval/run_eval.py:334`) and an accent underline bar. Ungraded
counts are shown alongside, never folded into the rates.

This is the only place in the workspace where big numbers are permitted (§1).

### Command palette (⌘K)

The one HUD marker adopted beyond the token system. In the reference it is
navigation across twelve areas; here it earns its place differently:

- Jump to a chat, jump to a source, open a setting, toggle theme, start a new
  chat — all without leaving the keyboard.
- `⌘K` / `Ctrl+K`, with the affordance shown in the topbar in the telemetry
  register.
- Genuinely useful in a three-pane app, and it reinforces the instrument feel
  more than decorative markers would.
- **Desktop only.** It has no meaning on touch and must not consume mobile
  screen space.

Deliberately *not* adopted: the far-left numbered nav rail, the breadcrumb
strip, and decorative grid/radar motifs inside the workspace.

---

## 9. Mobile — a different product shape

Not a reflow of the desktop layout. The phone is a **reader**; the desktop is a
**workbench**.

**Primary:** the conversation. Ask, read the answer, tap a citation, read the
excerpt, glance at the quality indicator.

**Available but demoted** (per-item maintenance): new chat, delete chat, delete
a source, refresh / re-index a single source.

**Desktop-only** (bulk corpus operations): benchmark runs (last result shown
read-only), folder scan, re-index all, advanced tuning fields. Presets remain
reachable.

**Mechanics:**
- Bottom tab bar or sheet-based navigation — never three panes stacked into one
  endless scroll.
- 44×44px minimum tap targets. Hover-revealed actions (delete, rescan) **do not
  exist on touch** and must be persistent or behind an overflow menu.
- `100dvh`, not `100vh`.
- `env(safe-area-inset-bottom)` padding on the ask bar.
- 16px minimum on inputs, or iOS Safari zooms on focus.
- No horizontal scroll at any width. Verify at 400px.

---

## 10. Relationship to `UI_UPGRADE_PLAN.md`

**Still valid:** source cards with type/status/citation pills; the Sessions
switcher; the responsive breakpoint contract; the build-order discipline.

**Superseded:** the light-only assumption; the Evaluation → "Insights" pane
rename with browsable Answer cards. Under §1 the eval layer becomes *quieter*,
not a richer card feed, and per-answer quality collapses to §6's single
indicator.

---

## 11. Backend work

- **Access tiers (§3)** — demo user + read-only enforcement on write routes;
  anonymous session cookie for ephemeral demo chats; Google OAuth re-enabled for
  the private tier.
- **Demo corpus seeding (§4)** — administrative one-off, or sliced if it must go
  through the API.
- **Benchmark baseline for the tick bar** — confirm `GET /api/eval` returns the
  last run's aggregate (faithfulness rate, relevancy rate, mean top similarity).
  Needed before §6's tick bar is meaningful.
- *(Deferred)* per-source `n_citations` on `GET /api/documents`. Client-side
  tallying covers v1.

---

## 12. Build order

Each step independently shippable and reviewable. **Every visual step is
screenshotted and verified in both themes** at all four breakpoints before
merge. Bump the `?v=` cache-buster in `app.html` when `app.js` or `styles.css`
changes.

> **Gap found during step 0:** there is **no `_shot.mjs` in the repo.**
> `UI_UPGRADE_PLAN.md` refers to it and revisions 1–3 of this document repeated
> the claim, but only the output PNGs (`shot_desktop.png`, `shot_mobile.png`, …)
> were ever committed. Since every remaining visual step depends on
> screenshot verification in two themes at four breakpoints, **the capture
> script has to be written before step 2** — it is now step 1b below.

| # | Step | Risk | Notes |
|---|---|---|---|
| 0 | **Finish tokenization** — promote ~35 hardcoded colors (hover states, the `.src-icon[data-kind]` palette) to variables | none | Pure refactor, zero visual change. Blocks everything. |
| 1 | **Split CSS** into shared tokens + app styles | none | Needed for two pages *and* two themes. |
| 1b | **Screenshot capture script** — Playwright, four breakpoints × two themes | none | Does not exist today; every later visual step depends on it. |
| 2 | **Theme mechanism** — `data-theme` on `<html>`, dark default, localStorage, inline head script to avoid flash-of-light | low | |
| 3 | **Access tiers (§3)** — demo user, read-only enforcement, ephemeral demo chats, OAuth for private tier | **high** | Moved up from polish; steps 7–8 depend on it. Backend + security surface. |
| 4 | **Two registers** — type scale, mono micro-label system, content-register answer styling | low | Highest visual payoff per line changed. |
| 5 | **Dark HUD token set** — layered near-blacks, hairline borders, no shadows, lime accent, PASS→checkmark | med | The PASS change touches eval rendering in several places. |
| 6 | **Per-answer quality readout** (§6) | med | Touches the render path; keep expanded detail intact. |
| 7 | **Settings presets** (§7) | med | Must not regress the config save path. |
| 8 | **Landing page** + routing (§2) | low | Two config edits; `index.html` → `app.html`. |
| 9 | **Demo corpus seeding + CTA flow** (§4) | med | Depends on step 3. |
| 10 | **Inline help & sequenced empty states** (§5) | low | Mostly copy. |
| 11 | **Command palette (⌘K)** (§8) | med | Self-contained; desktop only. Can slip without blocking anything. |
| 12 | **Mobile pass** (§9) | high | Dedicated pass, both themes, all four breakpoints. |
| 13 | **Light theme repair** | low | Consistency sweep, not a first build — light is carried from step 2. |

---

## 13. Change log

### Revision 4 — §3/§4 rewritten to describe what was actually built

Revisions 1–3 specified a **shared, read-only** demo corpus. The backend built
something different and better — a private, writable, per-visitor workspace —
so those sections described a product that does not exist. Corrected in place.

Four other things resolved while turning guest mode on:

1. **Guest-first was a boot-policy change, not a rename.** `app.js` fell back to
   a guest only when Google OAuth was *unconfigured*. Production has it
   configured, so every visitor met a sign-in wall and the entire guest
   subsystem was unreachable there. Swapping the endpoint alone would have
   changed nothing.
2. **Four eval routes had no authentication at all** — `PUT /api/eval/config`,
   both toggles, and `POST /api/eval/run`/`step`. On the public deployment any
   anonymous caller could re-point the embedding model for every user or start a
   46-question benchmark. Closed by `require_account` before guest-first shipped;
   guest-first without it would have been a straight downgrade.
3. **The guest cap only covered uploads.** `POST /api/documents/url` was
   unguarded, so the 3-document limit was bypassable by pasting links.
4. **`prune_chunks` existed only in the Neon store**, so "Prune ghosts" 500'd on
   the Chroma backend. The two backends must expose the same surface —
   `vectordb.py` dispatches by name and cannot paper over a missing one.

### Revision 3 — calibrating distance from the reference

The question was whether the plan had drifted too far from Grok-UI. Audit found
that the *vocabulary* was faithful but every **memorable** marker had been cut —
display numerals, hero type, space backdrop, grid motifs — leaving something
that risked reading as generic dark SaaS. Resolved by **concentrating** those
markers rather than dropping them:

1. **Landing page becomes a near-literal homage** (§2) — full HUD, space
   backdrop, display numerals, grid cards. It has no prose to protect, so it can
   carry the reference at full strength.
2. **The display-numeral ban was too broad** (§1) — it now applies only where an
   answer is present. The expanded benchmark view gets big numbers (§8), because
   there telemetry *is* the content.
3. **Command palette (⌘K) adopted** (§8) as the single extra marker; numbered
   nav, breadcrumb and decorative grid motifs stay out of the workspace.
4. **Space backdrop resolved** — yes on landing, no in the workspace.

### Revision 2 — resolving conflicts found in review

Four conflicts surfaced in review and were resolved:

1. **Single-user mode vs. public demo** — the plan assumed per-user workspaces
   the app does not have. Resolved as a two-tier access model (§3); auth moved
   up to step 3. *This was a blocking issue, not a detail.*
2. **Evaluation pane contradiction** — revision 1 called it quiet telemetry and
   gave it a third of the screen. Now collapsed by default (§8).
3. **Accent collision** — crimson collides with FAIL-red, amber with
   indexing-amber, lime with PASS-green. Lime chosen; **PASS becomes a neutral
   checkmark** so lime can mean *active* without meaning *good* (§6).
4. **Landing theme + no route back to the story** — landing now shares the app's
   theme, and the in-app help affordance links to `/` (§2).

Also newly identified while resolving #1: **read-only documents do not fix
shared chat history.** Demo chats must be per-visitor and ephemeral, or each
visitor reads the previous one's questions (§3).

---

## 14. Open decisions

1. **Mono typeface.** Proposal: self-hosted **JetBrains Mono** (one woff2,
   ~30KB) over a CDN — keeps the frontend dependency-free and CSP-clean.
   Alternative is the system mono stack (`ui-monospace, SFMono-Regular, Menlo,
   Consolas`), which costs nothing but is less distinctive.
2. **Hero display typeface.** The landing hero (§2) wants a soft geometric sans
   in the reference's manner. Same self-hosted-vs-system tradeoff as above, and
   it is the one place a distinctive face pays for itself.
3. **Which space image**, and its licence. Must be self-hosted and compressed —
   it is the element most likely to look cheap if chosen badly. NASA imagery is
   public domain and a safe default.
4. **Preset names and values** (§7) — needs one pass of actual tuning to pick
   sensible chunk/top-K/reranker combinations per preset.
