# Handoff — frontend + backend sessions → combined session

**Written:** 2026-08-17 by the frontend session.
**Updated:** 2026-08-17 by the backend session (§0, §3, §4 corrections).
**Updated:** 2026-08-17 by the combined session — **§0 rewritten; read it first.**

> ## §0. Current status — read first
>
> Everything is committed on `main`. Phases 0–2 of the combined session are
> done; `git log --oneline` is the source of truth.
>
> | Commit | What |
> |---|---|
> | `4cde1a7` | Phase 0 — the frontend session's uncommitted tokens/theme/`shot.mjs` baseline |
> | `d4808d8` | Phase 1 — guest-first entry + the write-route guards it required |
> | `d29ed86` | Phase 2 — two registers, JetBrains Mono, lime in both themes, per-answer readout |
>
> **§4A, C, D, F, G are resolved. §4B and §4E remain open, deliberately** —
> both belong to the landing-page split and must ship together (see the ordering
> trap below).
>
> Two things this document previously got wrong:
>
> 1. **§4A was not a one-line change.** `app.js` fell back to a guest only when
>    Google OAuth was *unconfigured*. Production has it configured, so every
>    visitor met a sign-in wall and the guest subsystem was unreachable there.
>    Swapping the endpoint alone would have changed nothing. The fix was the
>    boot **policy**.
> 2. **§9's ordering breaks Google sign-in.** It puts "callback redirect →
>    `/app`" at position 3, before the landing page at position 8. `/app` does
>    not exist until the Vite multi-entry + `vercel.json` rewrite land, so doing
>    it in that order 404s every sign-in. **§4B and §4E must be one commit.**
>
> ### Found and fixed while turning guest mode on
>
> Turning guest-first on without these would have been a straight downgrade of
> production integrity:
>
> - **Four eval routes had no authentication dependency at all** —
>   `PUT /api/eval/config`, both toggles, and `POST /api/eval/run` / `/step`.
>   `config_overrides` is a **single shared row** (`db.py:121`), so on the public
>   deployment *any anonymous caller* could re-point the embedding model for
>   every user and invalidate their chunks, or start a 46-question benchmark
>   against the deployment's LLM quota. Now behind `require_account`.
> - **The guest cap only covered `/documents/upload`**, so the 3-document limit
>   was bypassable by pasting URLs.
> - **`prune_chunks` existed only in `store_neon.py`**, so "Prune ghosts" raised
>   `AttributeError` and 500'd on the Chroma backend.
>
> Tests: **62 passing** (was 43); `tests/test_guest_permissions.py` is new and
> asserts both directions — what a guest is refused *and* what a guest keeps.

This session did UI/UX planning and frontend steps 0–2 of
[`PRODUCT_UX_PLAN.md`](PRODUCT_UX_PLAN.md). It stopped at the point where
further work would collide with the backend session's changes.

---

## 1. Working-tree state

| File | Owner | State |
|---|---|---|
| `frontend/tokens.css` | **this session** | new — design tokens + dark theme |
| `frontend/styles.css` | **this session** | modified — de-hardcoded, imports tokens |
| `frontend/index.html` | **this session** | modified — no-flash theme script, theme toggle |
| `frontend/app.js` | **this session** | modified — +28 lines, theme toggle handler only |
| `shot.mjs` | **this session** | new — Playwright capture script |
| `PRODUCT_UX_PLAN.md` | **this session** | new — the plan (rev 3) |
| `ragchat/guests.py` | **backend session** | new — ephemeral guest accounts |
| `ragchat/app.py` | **backend session** | modified — guest login, promotion, account deletion |
| `ragchat/db.py` | **backend session** | modified — `User.last_seen_at`, `Document.size_bytes` |

The two sets do not overlap in files. They **do** overlap in contract — see §4.

---

## 2. What this session built (frontend)

### Step 0 — tokenization *(complete)*

Every hardcoded colour in `styles.css` was promoted to a variable in the new
`frontend/tokens.css`. Also tokenized the duplicated mono/sans font stacks as
`--font-mono` / `--font-sans`.

Verified: `grep` finds **zero** hex/rgba literals and zero raw font stacks left
in `styles.css`; **78 variable references, 78 definitions, none unresolved**
(the real risk — a typo'd token name fails silently as transparent).

### Step 1 — CSS split *(complete)*

`styles.css` now begins with `@import "./tokens.css"`. Vite inlines this at
build time, so production is still one CSS request (confirmed: no `@import`
survives in `dist/assets/*.css`). The landing page will link `tokens.css`
directly to get the palette without the app's rules.

### Step 1b — screenshot capture *(complete)*

`shot.mjs` at the repo root. **The `_shot.mjs` referenced by
`UI_UPGRADE_PLAN.md` never existed** — only its output PNGs were committed.
Playwright is installed at the repo root with no `package.json`, so run from
there:

```bash
node shot.mjs http://localhost:4173
```

Captures 5 breakpoints × 2 themes into `shots/<theme>/<bp>.png`, and **fails the
run** on horizontal overflow or any page error — both are silent killers a
screenshot alone would not reveal.

### Step 2 — theme mechanism + dark palette *(complete)*

- `data-theme` on `<html>`, dark default, persisted to `localStorage` under the
  key **`ragchat-theme`** (the landing page must reuse this exact key).
- No-flash inline `<head>` script in `index.html` — it must stay inline and
  blocking, or dark-theme users get a white flash every load.
- Toggle button `#theme-toggle` in the topbar; handler in `app.js`.
- Full dark palette in `tokens.css` under `[data-theme="dark"]`: four-step value
  depth, hairline borders, **no shadows** except the modal, **acid lime** as the
  single hot accent, purple `--accent` folded into lime.
- **PASS is neutral, not green** in dark, so lime can mean *active* without also
  meaning *good*.

Verified: build passes; all 10 captures clean, no overflow, no page errors.

**Known incomplete:** the PASS **checkmark glyph** is not done. Dark makes PASS
neutral by colour only; the `"PASS"` → `✓` text change lives in
`buildEvalBlock` (`frontend/app.js:786`), which plan step 6 rewrites wholesale.
Doing it now would be work thrown away.

---

## 3. What the backend session built

Read from the working tree, not from conversation — this is observed, and the
other session may have moved on since.

**`ragchat/guests.py`** — ephemeral per-visitor accounts:

- Each anonymous visitor gets their **own** `provider="guest"` account. This
  replaces the old behaviour where `local_login` signed *everyone* into one
  shared `local` account.
- **Demo corpus is embedded once** under a template account
  (`__demo_template__`) and then **vector-copied** per guest — no embedding
  calls, no quota spend, no latency on arrival.
- Guests **can upload**, capped at **3 documents / 5 MB total**.
- Guests are reaped after **2 hours idle**, opportunistically on guest creation
  (no cron on Vercel Hobby, no background threads).
- Signing in with Google **promotes** the guest's workspace into the permanent
  account — documents, folders, conversations and chunks are re-pointed, nothing
  is re-embedded.

**New endpoints:** `POST /api/auth/guest-login`, `DELETE /api/auth/account`.
**Schema:** `User.last_seen_at`, `Document.size_bytes`.

All `vectordb` helpers `guests.py` depends on (`copy_user_chunks`,
`reassign_user_chunks`, `delete_document_chunks`) **exist in both** `store.py`
and `store_neon.py` — checked.

---

## 4. Interface points — the actual handoff

Ordered by severity.

### A. `app.js` still calls a route the backend is replacing — **breaking**

`frontend/app.js:100` calls `POST /api/auth/local-login`. The backend added
`POST /api/auth/guest-login`. **If `local-login` is removed, the frontend fails
to boot.**

The new route returns `{id, name, guest: true}` and reuses an existing guest
session if the browser already has one.

*Action:* switch the call, and decide whether `local-login` stays as a
single-user dev convenience.

### B. OAuth callback redirect — **will break when the landing page ships**

`ragchat/app.py` still ends the Google callback with `RedirectResponse("/")`.
Once `/` is the landing page, **signing in dumps the user back on the marketing
page**. Must become `/app`, ideally with a `next` round-tripped through the
existing `oauth_state` cookie.

Still unfixed in the working tree.

*Good news:* the callback path itself (`/api/auth/google/callback`) is unchanged
by the landing split, so `GOOGLE_REDIRECT_URI` and the Google Console entry need
**no** changes. Given the trailing-slash bug documented at `ragchat/app.py:278`,
worth not disturbing.

### C + D. Guest identity and quota on `/api/auth/status` — **RESOLVED**

Both requests are implemented. `GET /api/auth/status` now returns:

```json
{
  "authenticated": true,
  "user": { "id": "…", "name": "Guest", "email": null },
  "google_oauth": true,
  "provider": "guest",
  "is_guest": true,
  "guest_usage": {
    "documents": 0, "max_documents": 3,
    "bytes": 0, "max_bytes": 5242880,
    "idle_ttl_seconds": 7200
  }
}
```

`guest_usage` is `null` for signed-in accounts, which are never capped. It
reports **usage as well as limits**, so the UI can show "2 of 3 documents"
before the visitor hits the wall rather than teaching the rule via a rejected
upload.

Note `documents` **excludes the seeded demo files**, matching how the cap counts
them. A guest with the 2 demo files still has all 3 upload slots — the demo
corpus is the app's content, not the visitor's. Do not display
`len(/api/documents)` as usage; it will read 2 too high.

### E. `vercel.json` — shared file, both sessions may touch it

The landing split needs one added rewrite. Coordinate so neither session
clobbers the other:

```json
{ "source": "/app", "destination": "/app.html" }
```

Plus, in `frontend/vite.config.js`:

```javascript
build: { rollupOptions: { input: { main: 'index.html', app: 'app.html' } } }
```

### F. Cache-buster collisions — **explained**

`?v=` moved 8 → 10 because the backend session edited `frontend/` too: the auth
repair (§4A's counterpart) and the **"RAG Chat" → "RAG-it" rename**. Both are
committed; `index.html` is at `v=10`. The advice still stands — re-read before
editing — but there is no unknown third writer.

**The rename is done and must not be reverted.** Every user-visible string is
now "RAG-it": `<title>`, the `#auth-view` heading, `.brand-name`, and the
FastAPI title on `/docs`. `frontend/dist/` still contains the old name; it is a
gitignored build artifact and regenerates.

### G. Session cookie lacks `secure`

`set_cookie(..., httponly=True)` throughout `ragchat/app.py` — no `secure=True`,
no explicit `samesite`. Worth hardening on an HTTPS deploy. Independent of this
work; noted because it now guards real Google identities.

---

## 5. Plan corrections required

`PRODUCT_UX_PLAN.md` **§3 and §4 are now wrong.** They were written before the
backend session's design existed and describe a *shared read-only demo corpus*.
The backend built something better:

| Plan says (rev 3) | Reality |
|---|---|
| Shared demo corpus, read-only | **Per-guest private copy, writable** |
| Guests cannot upload | Guests **can** upload, capped 3 docs / 5 MB |
| "Try it with 10 sample documents" | **2 files only** — `helios_energy_handbook.md`, `meridian_coffee_ops.md` |
| Sign in "for your own workspace" | Sign in **keeps the work you already did** — it is promoted, not discarded |
| Demo chats need separate ephemeral handling | Solved — each guest is already its own account |

**The 2-file limit is deliberate and must not be "fixed".** `guests.py` states
the other eight files in `eval/corpus/` are real business content that must
never be exposed to anonymous visitors. Any landing copy promising ten sample
documents is wrong.

The messaging shift matters for the landing page: the CTA is no longer "try a
read-only demo", it is **"start using it now, sign in when you want to keep
it"** — a materially better pitch.

---

## 6. Safe to continue without backend coordination

Frontend-only, no contract dependency:

- **Step 4 — two registers.** Type scale, mono micro-label system, large-sans
  content register for answers. Highest visual payoff remaining.
- **Step 11 — ⌘K command palette.** Self-contained, desktop only.
- **Step 12 — mobile pass.** Chat-first reader shape; bulk operations
  desktop-only. Note the guest quota UI will need adding afterwards.
- **Step 13 — light theme repair.** Light still uses blue `--primary` while dark
  uses lime; they should converge on the lime identity.

Blocked on §4 items: steps 3, 6, 7, 8, 9 of the plan.

---

## 7. Open decisions (unchanged from plan §14)

1. Mono typeface — self-hosted JetBrains Mono vs the system stack.
2. Hero display typeface for the landing page.
3. Which space image, and its licence. NASA imagery is public domain.
4. Preset names and values — needs a real tuning pass.

Plus four auth questions this session raised but never got answered, now largely
**overtaken** by the backend's design — re-derive them from §4 rather than from
the earlier conversation.

---

## 8. Backend session — everything else it shipped

Added 2026-08-17. All committed; `git log --oneline` for the full list.

### Auth is live end to end

Google sign-in **works** on both local and production. It had never worked
before: four frontend/backend contract mismatches, none of which could ever have
succeeded (`/api/auth/me` did not exist; the sign-in and sign-out buttons issued
GETs against POST-only routes; the status payload was read in an old shape).

Per-user isolation was **already fully implemented server-side** — every route is
gated by `get_current_user`, and documents, folders, chats and vector chunks all
filter on user id, with cross-user id access returning 404. Verified live with
two concurrent sessions. The only reason everyone shared one space was that
sign-in never worked, so every visitor was auto-signed into `local`.

### Production configuration is correct

`GET /api/health` now has an `auth` block. Current production state:

```
session_secret_is_default : false
google_oauth_configured   : true
google_redirect_uri_set   : true
```

Two gotchas worth not rediscovering:

- **`GOOGLE_REDIRECT_URI` must match the Google Console entry character for
  character.** Production is registered **without** a trailing slash, local
  **with** one. Both work, because the callback route is registered on both
  forms — but the env var must match its own environment's registration.
- **`SESSION_SECRET` must be set in production.** Without it the app signs
  sessions with a hardcoded default and anyone who knows it can forge a cookie
  for any account. `/api/health` → `auth.session_secret_is_default` reports this.

### Migration tool

`scripts/migrate_user_data.py` reassigns a whole workspace between accounts
(documents, folders, conversations, messages, vector chunks) with no
re-embedding. Dry-run by default, `--commit` to write, reversible by swapping
`--from`/`--to`. Used to move the owner's data from `local` to their Google
account. Handles both vector backends.

### Retrieval and eval fixes

- **Hybrid search was fusing two different corpora.** The vector half filtered on
  `fingerprint`, the FTS half did not, so stale chunks from a previous chunking
  config were retrieved by keyword and RRF-promoted. Fixed.
- **The benchmark could not survive Vercel.** One scored question takes 40–54s,
  so `EVAL_BATCH_DEFAULT = 2` took 83s against the 60s `maxDuration`. An
  overrunning step is killed *before* it commits, so the client retried the same
  slice forever. Now 1, with only ~6s headroom — **if judge latency grows, slice
  the judges rather than raising the batch.**
- **A truncated `<thought>` block was graded as a confident FAIL.** The judge
  model is thinking-capable; when reasoning hit `max_tokens` the wrapper was
  never closed, the stripper missed it, and a stray "FAIL" inside the trace was
  read as a verdict. Unterminated wrappers are now stripped → ungraded, not
  failed.
- **Settings could not be saved at all.** The stored embedding model was the
  legacy bare `qwen3-embedding-8b` while the allowlist held
  `qwen/qwen3-embedding-8b`; every save re-submits the stored model, so
  validation rejected all of them. Comparison now ignores the vendor prefix.
- Config values are bounds-checked (`chunk_size`, `top_k`, `similarity_threshold`
  …). Previously `top_k=0` or `threshold>1` saved fine and made every question
  answer "not found", which looks like a broken app rather than a bad setting.

### Known-good verification commands

```bash
.venv/Scripts/python -m pytest tests/ -q          # 43 passing
.venv/Scripts/python -m scripts.migrate_user_data --list
curl -s https://rag-gel.vercel.app/api/health | python -m json.tool
```

---

## 9. Remaining work, in order

Items 1–2, 4–5 of the previous list are **done** (see §0). What is left:

### Phase 3 — landing page + routing, as ONE commit

`index.html` → `app.html`, the Vite multi-entry input, the `vercel.json`
rewrite, **and** the §4B callback redirect together. Splitting them 404s
sign-in for however long the split lasts.

Blocked on open decisions §7.2 and §7.3 (hero typeface, space image). §7.1 is
**settled**: self-hosted JetBrains Mono, Regular + Bold, shipped in `d29ed86`.

Landing copy must say **two** sample documents, not ten, and pitch
"start now, sign in to keep it" rather than "try a read-only demo" (§5).

### Phase 4 — product depth

Settings presets (plan step 7) · inline help and sequenced empty states
(step 10) · ⌘K command palette (step 11).

> **Copy bug to fix in step 10:** the empty chat state says "Add sources on the
> left" — but a guest arrives with two demo documents already loaded, so it is
> wrong for every first-time visitor. It should offer the three suggested
> questions the plan §4 calls for instead.

### Phase 5 — mobile

Deliberately last: it re-lays-out everything above it. Scope (full separate
product shape vs. responsive-only) was left open pending Phase 2 and is now
ready to decide.

### Backend, unstarted

A **retrieval-only CI gate** running the 46 golden questions on every push at
zero LLM cost.

### Noted, not chased

Running locally against a scratch DB with **`config.yaml` defaults**, an answer
came back echoing the rewritten query rather than answering it, and the judge
returned a reasoning block with no verdict. Production runs a tuned
`config_overrides` row, not these defaults, so this may be an artefact of the
scratch environment rather than a live defect — but it is worth one deliberate
look, because both symptoms would be invisible from `/api/health`.
