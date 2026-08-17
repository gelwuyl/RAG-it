# UI Upgrade Plan — agentic-RAG

**Status:** Proposal for review (plan only, no code changes). Borrows the reference "Super Intelligence" notebook's visual strengths but re-maps them onto RAG semantics. References current `frontend/{app.js,index.html,styles.css}` (post Task A responsive layout).

**Guiding principle:** we already have Sources + Chat + Evaluation + Excerpt. Don't add product concepts we don't have (notebooks, hand-written notes). Adopt the *card system, pill vocabulary, session switcher, and color tokens* — skip the *notebook metaphor*.

---

## 1. Layout — adopt the 3-pane model?

**Decision: keep 3 panes, but do NOT add a 4th "Notes" column.** Adopt the reference's *middle "browsable cards" idea by re-purposing the Evaluation pane*, not by adding a column. A 4-column desktop layout would crush the chat thread and break Task A's tablet reflow (which already assumes sources|chat + eval-strip).

Current → proposed mapping:

| Reference pane | Our equivalent | Action |
|---|---|---|
| Sources | `sources-pane` (left) | Restyle to cards (§2). Stays left. |
| Notes / Insights | **`eval-pane` (right), renamed "Insights"** | Merge: eval scorecard + per-answer citations/eval become browsable **Answer cards** fed from `Message.eval_data` / `citations`. |
| Chat with Notebook | `chat-pane` (center) | Add Sessions switcher (§3). Otherwise as-is. |

**"Insights" pane content (right):** two stacked sections in the existing `pane-scroll`:
1. **Answers** (new) — one card per assistant message in the current chat: truncated answer, faithfulness/relevancy PASS/FAIL chips, top-sim, latency, and its citation chips. Clicking a card's citation opens the Excerpt pane (reuse `showExcerpt`). This is the direct analog of the reference's "note cards," sourced from data we already render inline.
2. **Benchmark** (existing) — the RAGAS scorecard + golden-question results, collapsed by default under an accordion header.

**Keep the right pane, merged — do not keep a separate eval tab.** The inline per-message eval block (`buildEvalBlock`) stays in the chat thread (it's useful in context); the Insights pane becomes the *cross-message* browsable view. This gives us the reference's "cards you can scan" without new backend data.

**Excerpt pane:** unchanged (bottom, full-width). It's our version of "click a citation → read the passage" and has no reference analog. Keep.

---

## 2. Source cards — link pill + N-insights pill + lightbulb

Restyle `.source-item` (rendered in `renderDocs` / `renderFolders`) into a rounded card matching the reference's source cards.

Card anatomy (adapt reference → our status model):
- **Title** (existing `src-title`).
- **Type/link pill** — reuse existing `sourceKind()`; a URL source shows a "link" pill (↗ + host), files show their type tag. This *is* the reference's "link" pill.
- **Status pill** — map `STATUS_META` to pill styling: `ready` (green), `indexing`/`pending` (amber/animated), `failed` (red). Reference has no status concept; this is our essential addition.
- **"N insights" pill** — map to **citation count**: how many times this source was cited across the current chat's answers. Yellow lightbulb icon (§5 `--insight`) + count.
  - **Data gap to flag:** `GET /api/documents` does not return a per-source citation count. Two options — (a) compute client-side by tallying `citations[].title`/source id across loaded messages (cheap, no backend change, but scoped to the open chat), or (b) add a `n_citations` field server-side (accurate, cross-chat, needs a backend change). **Recommend (a) for v1**, labeled "N cited here"; defer (b).
- **Right-aligned lightbulb** — decorative when count > 0, muted when 0.

Keep existing per-card actions (delete, folder rescan) as hover icon-buttons.

---

## 3. Sessions switcher

We already have the data: `GET /api/chats` (`state.chats`) and `renderChats()`. The reference's "Sessions (clock)" is purely a UI affordance we can add cheaply.

**Proposal:** the current inline `#chat-list` above the message thread works but eats vertical space. Replace with a **"Sessions" popover** anchored to a clock button in the chat pane head:
- Chat head gets: `Chat` title · **⏱ Sessions** button (opens popover list) · `+ New chat`.
- Popover reuses `renderChats()` markup (status dot, title, delete) — just moved into a dropdown container instead of an always-open strip.
- Active session title shown in the head.
- **Mobile:** popover becomes a full-width sheet; no horizontal scroll.

Low risk — `openChat` / `refreshChats` / delete logic are untouched; only the container and a toggle move.

---

## 4. Notebook header (Archive / Delete / "Add a description")

**Decision: SKIP.** Rationale:
- We have **sources and chats, not notebooks** — there is no single "document" entity to title, describe, archive, or delete. The reference's header describes *one notebook*; our app is a workspace over many sources + many chats.
- "Archive/Delete" at that level has no target in our model (delete already exists per-source and per-chat).
- Adopting it would imply a notebook data model we don't have and the task explicitly says we don't want.

**Adapt instead (small):** give the top bar a slightly stronger title treatment (already `RAG Chat` + subtitle) — no new header row. That's the only piece worth borrowing.

---

## 5. Visual tokens — blue / gray / red / yellow system

Our tokens already lean blue-primary / gray-neutral / red-error. The main additions are a **dedicated yellow "insight" accent** and card/pill tokens. Add to `:root` in `styles.css`:

```css
/* insight accent (reference's yellow lightbulb) */
--insight:        #d9a400;   /* icon / text */
--insight-soft:   #fff7e0;   /* pill bg */
--insight-border: #f2d98a;   /* pill border */

/* card system (rounded cards for sources + answers) */
--card-bg:        var(--panel);
--card-border:    var(--border);
--card-radius:    var(--r-lg);
--card-shadow:    var(--shadow-sm);
--card-gap:       10px;

/* pill system (link / status / insight / AI-generated) */
--pill-radius:    var(--r-pill);
--pill-pad:       2px 9px;
```

Keep existing `--primary` (blue), `--muted`/`--faint`/`--border` (gray), `--error`/`--fail` (red **destructive-only** — audit that red is never used decoratively). Demote `--accent` (purple) usage on the benchmark button to blue or keep as a distinct "run" accent — decide during build; reference uses blue for all primary actions, so **prefer folding purple into blue** for consistency unless we want eval visually separated.

New reusable classes to introduce (no new colors beyond above): `.card`, `.pill`, `.pill-link`, `.pill-status.{ready,working,error}`, `.pill-insight`, `.answer-card`.

---

## 6. Responsive — preserve Task A behavior

Every new element must collapse into the existing single-column stack; **no horizontal scroll** at any breakpoint. Task A's breakpoints (`1360`, `768–1099`, `≤767`, `≤400`) stay authoritative.

| New element | Desktop (≥1100) | Tablet (768–1099) | Mobile (≤767) |
|---|---|---|---|
| Source cards | left rail, vertical stack | same (narrower rail) | full-width stack in `sources` area |
| Insights pane (Answers + Benchmark) | right rail | **eval-strip below** (existing `"eval eval"` area) — Answers section collapsible, starts folded via existing `applyBreakpointDefaults` | stacked last, single column, `overflow: visible` |
| Sessions popover | dropdown anchored to clock btn | dropdown | **full-width sheet**, not a floating popover |

Rules to honor:
- Cards use `flex-direction: column` inside their pane's `pane-scroll`; never `flex-nowrap` rows that overflow.
- The Insights "Answers" section obeys the existing `.pane.collapsed` mechanism (`bindPaneToggle` + `applyBreakpointDefaults`) so it folds below 1100px like eval does today.
- Popover/sheet must not exceed viewport width; test at 400px.
- Re-verify with the existing shot scripts (`_shot.mjs`, `shot_desktop/tablet/mobile.png`) before merge.

---

## 7. Reuse vs adapt vs skip

**Copy (as-is intent):**
- Rounded card container + generous whitespace.
- Pill vocabulary (link pill, "N insights" pill, colored small pills).
- Yellow lightbulb insight accent.
- "Sessions" clock affordance.

**Adapt (re-mapped to our semantics):**
- "Notes" column → **Insights pane** built from `eval_data` + `citations` (Answer cards), not user-authored notes.
- "N insights" pill → **citation count** per source.
- Source card "link" pill → our existing `sourceKind()` type/URL tag.
- Header title treatment → light touch on existing top bar only.
- "AI Generated" pill → optional small pill on Answer cards ("Grounded" / "Web fallback used") sourced from answer metadata.

**Skip (no data model / conflicts with our app):**
- Notebook header with Archive / Delete / "Add a description" (§4).
- Far-left icon rail (our actions live in the top bar + panes; an icon rail adds chrome without new function).
- Hand-written Notes / "Write Note" (we generate answers, users don't author notes).
- Per-notebook "Created/Updated 2 months ago" metadata (no notebook entity).

---

## 8. File-level pointers & build order

**Files/regions to touch:**

`frontend/index.html`
- `.sources-pane` markup — no structural change (cards are CSS-driven off `renderDocs` output).
- `.chat-pane` `pane-head` — add ⏱ Sessions button + popover container; move `#chat-list` inside it.
- `.eval-pane` — rename title to "Insights"; add an `#answers-list` section above `#eval-scorecard`; wrap benchmark in a collapsible accordion.

`frontend/app.js`
- `renderDocs` / `renderFolders` (L221–284) — emit card markup + status/link/insight pills; compute per-source cited-count from loaded messages.
- New `renderAnswers()` — build Answer cards from the current chat's messages (reuse `buildEvalBlock` data + `showExcerpt`); call it at the end of `openChat` (L652) and after a new answer in `ask-form.onsubmit` (L804).
- `renderChats` (L582) — retarget into the Sessions popover container; add a toggle handler for the clock button.
- Excerpt (`showExcerpt`, L831) — unchanged; Answer-card citations call it.
- Collapsible logic (L1011–1042) — register the Insights "Answers" section with `bindPaneToggle` / `applyBreakpointDefaults`.

`frontend/styles.css`
- `:root` (L13–52) — add tokens from §5.
- New sections: `.card`, `.pill*`, `.answer-card`, `.sessions-popover`.
- `.source-item` (search L~420+) — restyle to `.card`.
- Media queries (L758–858) — add rules from §6 (Answers-section collapse below 1100px; Sessions sheet on mobile). Bump asset cache-bust `?v=6` → `?v=7` in `index.html`.

**Build order (each step independently shippable/reviewable):**
1. **Tokens** — add §5 CSS variables + base `.card`/`.pill` classes. No behavior change.
2. **Source cards** — restyle `.source-item`, add link/status pills. (No new JS data yet.)
3. **Insight count** — client-side cited-count → "N cited here" pill + lightbulb.
4. **Insights pane** — rename eval→Insights, add `renderAnswers()` Answer cards, make benchmark collapsible.
5. **Sessions switcher** — move chat list into clock popover/sheet.
6. **Responsive pass** — verify all four breakpoints with shot scripts; fix collapse/overflow.
7. **(Deferred)** backend `n_citations` on `GET /api/documents` for cross-chat accuracy.

**Risk notes:** steps 1–3 are pure CSS/low-risk. Step 4 touches `openChat`/`ask` render paths — keep inline eval block intact. Step 5 must not regress `openChat`/delete logic. Nothing here requires backend changes for v1 (step 7 is optional polish).
