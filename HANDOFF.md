# Handoff — agentic-RAG

Written 2026-08-18. Replaces the earlier frontend/backend handoff, which described
phases 0–4 and is now fully landed (see `git log`).

**Next session's first job is §1.** Everything else is queued work with decisions
already made.

---

## 0. Orientation

Read `CLAUDE.md` first — it carries the non-obvious constraints (serverless
freeze, config precedence, the 768-dim invariant, fail-open evaluation) and they
are all still accurate.

Everything this session did is in the git history with reasoning in the commit
bodies. Do not re-derive it from the code; read the messages:

```bash
git log --oneline -12
```

| commit | what |
|---|---|
| `3619d4d` | deterministic exact-containment metrics; gate baseline |
| `2f8a1f4` | eight distractor documents, 21 → 39 chunks |
| `665b8a6` | MATCH_THRESHOLD calibrated 0.6 → 0.45; baseline mechanism |
| `e2b2ecb` | synthetic corpus + golden set (then history-purged) |
| `215c064` | plain-language metric names + benchmark percentile |
| `f79f875` | demo vectors ship with the repo (cold-start 504 fix) |
| `c002445` | harness scores the passages the model is actually handed |
| `8e46ab2` | guest seeding: a document either has its vectors or is not there |
| `53cf838` | command palette moved to "/" in the composer |
| `acd599e` | Cohere reranks by default; two embedders instead of six |

Deployment: `https://rag-gel.vercel.app` · `GET /api/health` reports effective
config. Repo is **public**.

---

## 1. LIVE BUG — identity is blank for seconds after every load

**Reported as:** "the sign-in button at the top right is missing in the latest
deployment, and after a refresh it can't quickly tell guest from signed-in."

**It is not missing.** Measured on the deployment, the button renders correctly.
The real defect is timing, and the user's instinct about cookies is the right fix.

### Evidence (Resource Timing, warm function, deployed app)

```
domContentLoaded            73 ms
/api/auth/status      73 → 1301 ms      identity UNKNOWN for 1.2s
/api/chats           1302 → 2449 ms     ← all five serialise behind status
/api/eval            1302 → 2449 ms
/api/eval/config     1302 → 2456 ms
/api/documents       1302 → 2468 ms
/api/folders         1302 → 2499 ms
/api/auth/status     2500 → 3669 ms     ← SECOND call, re-renders the badge
```

Three separate problems:

1. **Topbar identity is blank until `/api/auth/status` resolves** — 1.2s warm.
   A cold Vercel function is ~3s (measured `/api/health` at 2.9–3.7s), so a
   first-time visitor stares at a topbar with neither guest badge nor sign-in
   button for about three seconds. That is the reported symptom.

2. **`/api/auth/status` is called twice on boot.** The second call is
   `refreshGuestUsage()` (`frontend/app.js` ~line 540), which re-renders the
   guest badge at 3669 ms — a visible second change after the first paint.
   Note the comment at `frontend/app.js` ~line 634 claims a duplicate call was
   already removed; this is a *different* call site and the comment is
   misleading as a result.

3. **Five API calls serialise behind status** rather than firing in parallel,
   so first meaningful content is ~2.5s.

### Why JS cannot tell today

The session cookie is `httponly=True` (`ragchat/app.py` ~line 84, set centrally
by one helper — good). So the page has no synchronous way to know who it is and
must wait for a round trip.

### Suggested fix (not yet implemented, not yet agreed with the user)

Set a **second, non-httpOnly hint cookie** beside the session cookie carrying
only the identity *kind* — e.g. `ragchat-kind=guest` or `=account`. The boot
script reads it synchronously and paints the correct topbar at ~73 ms, then
`/api/auth/status` reconciles when it lands.

It carries no secret and being spoofable is harmless: every real authorisation
decision stays server-side on the httpOnly session cookie. This is a *rendering*
hint only. Make sure it is cleared on logout and rewritten on
guest-login/promotion, or it will out-live the session it describes.

Also worth doing in the same pass:
- Delete the duplicate status call — `refreshGuestUsage()` should read the
  payload `applyAuthStatus` already stored, and fix the now-misleading comment.
- Fire the five independent calls in parallel with status instead of after it.
- Render a skeleton in `.topbar-identity` so it never occupies zero height
  (currently it collapses, which is why the layout appears to shift).

**Confirm the cookie approach with the user before building it** — they raised
it as a question, not an instruction.

---

## 2. Remaining queue, in order

All decisions below are already settled with the user. Do not re-ask.

### 2.1 Finish the corpus (~20 more chunks)
39 chunks today; the original haystack was ~59. Approved to continue *now that
the gate metric is deterministic*. Write same-domain **confusable** documents —
unrelated filler is separated trivially by embeddings and adds bulk without
difficulty. Regenerate the golden set and baseline afterwards (§4).

### 2.2 CI gate
GitHub Actions, **push to main only**, retrieval-only, fails on regression
against `eval/baseline.json`. Fork PRs never receive secrets on a public repo,
so PRs are skipped by design. Gate compares the `exact_*` metrics only — see
`eval/baseline.py:GATED_METRICS` and the test that pins it.

### 2.3 Scorecard reads the baseline
`frontend/app.js:EVAL_TARGETS` still hardcodes aspirational targets (0.80 etc.).
It must read `eval/baseline.json` so "red means regression" is true in the UI as
well as in CI. Both consume the same file — do it with §2.2.

### 2.4 Guest lifecycle — **fully designed, confirmed, not started**
- Idle TTL 2h → **30 min** (guests only; signed-in workspaces are permanent and
  already excluded by `provider == "guest"`, with a test).
- **Keepalive** while the tab is visible (~5 min) so an open tab is never reaped
  mid-read. Reuse `/api/auth/status`, which already calls `touch()`.
- **Close beacon**: `pagehide` → `sendBeacon` that *back-dates* `last_seen_at`
  so the next sweep collects it. It must **not** delete inline — close-and-reopen
  has to survive. **The OAuth redirect must suppress the beacon**, or signing in
  destroys the workspace during promotion.
- **Sweeper**: GitHub Actions every 15 min against an authenticated sweep
  endpoint (shared secret, constant-time compare). Vercel Hobby cron exists but
  is **once per day**, which cannot honour a 30-minute promise — verified, and
  the code comment in `ragchat/guests.py` saying Hobby has no cron is out of date.
- **Inline backstop**: `create_guest` clears at most **2** workspaces as one bulk
  statement, so cleanup never stops dead if Actions is disabled at 60 days.
- **Bulk set-based deletes**: `purge_user_data` currently costs ~6–8 round-trips
  per workspace and `create_guest` reaps 20 inline — that measured **39.7s** on a
  guest-login against 11.1s warm. Acceptance criterion: **guest-login under 10s**.

### 2.5 Phase 5 sweep
Tap-target audit (44×44 minimum) and a light-theme pass over the surfaces added
in phases 4–5. No decisions outstanding.

### 2.6 Deep search — planned, named, not started
Replaces web search entirely. Full design is in this conversation's history; the
essentials:
- `documents.source_text` (`ragchat/db.py` ~line 93) already stores full document
  text durably, so grep needs no new storage and no vector-backend dispatch.
- **Per-request flag, never a persisted config.** `config_overrides` is ONE row
  shared by the whole deployment — a persisted toggle would change retrieval for
  every user. The existing web-augmentation toggle has exactly this bug
  (`ragchat/app.py` ~line 1050 writes `save_config_override`).
- `seed_demo_corpus` does **not** copy `source_text` to guest clones — one line,
  and a prerequisite for guests to use deep search at all.
- UI name: **"Deep search"**. Delete web search in the same commit so there is
  never a build with two fallbacks.

---

## 3. Non-obvious things learned this session

Additions to what `CLAUDE.md` already documents.

**The eval harness measured the wrong list for months.** `retrieve()` returns
`candidate_k` chunks; `ask()` reranks to `top_k`. `context_recall` was computed
over all 40 while precision/MRR/hit sliced `[:6]` — two different retrievals
reported side by side. And `score_item` never reranked at all, so a preset with
`reranker: True` scored exactly as if it were False. Fixed in `c002445`.

**`MATCH_THRESHOLD` was never calibrated and was catastrophically wrong.** At
0.6 it scored **79.6% of true containments as misses**. Cause is length, not
language: a one-line passage against a ~500-token chunk scores low even when
verbatim inside it (Latin fared *worse* than CJK). Most of this repo's
historically low retrieval scores were a broken measurement.

**Cosine matching is biased with a direction.** Its false-positive rate *rises*
with corpus size (17.2% at 21 chunks → 19.5% at 39), so adding filler documents
makes scores go **up**. Never gate CI on it. The `exact_*` metrics exist for that
reason; cosine remains the reported RAGAS-comparable number by user decision.

**A `config_overrides` row masks `config.yaml` defaults forever.** Changing a
default does nothing on any deployment that ever saved Settings. `hybrid_search`
and now `reranker_provider` are **hardcoded** in `load_config()` for this reason.
Whether to rerank is still read from config; which vendor is not.

**A stale local override silently invalidates local measurement.** A local row
carried `qwen3-embedding-8b` without the `qwen/` prefix, which changed the
fingerprint and made every local benchmark unrepresentative. If local and
deployed fingerprints differ, suspect this first.

**`golden_passages` must be verbatim substrings.** If a passage is not
character-for-character present in a corpus document, `context_recall` measures
nothing — the cosine runs against text absent from the corpus. The generator
enforces this; keep it that way.

---

## 4. Verification

```bash
# Tests — 122 passing at time of writing
.venv\Scripts\python -m pytest tests/ -v

# Free retrieval benchmark (no LLM call). --with-rerank costs one Cohere
# call per question; --ceiling separates a ranking problem from a retrieval one.
.venv\Scripts\python -m eval.run_eval --retrieval-only
.venv\Scripts\python -m eval.run_eval --retrieval-only --with-rerank --ceiling

# Regenerate the baseline after ANY corpus / golden set / chunking / model change.
# Read the diff before committing — regenerating blindly launders a regression.
.venv\Scripts\python -m eval.baseline

# Demo vectors, after an embedding-model or chunking change (else guest-login 504s)
.venv\Scripts\python -m ragchat.demo_vectors

# Frontend: both pages x 5 breakpoints x 2 themes; fails on overflow or page error
node shot.mjs http://localhost:5173
node layout_check.mjs http://localhost:5173
```

The golden-set generator lives outside the repo (it was a scratchpad script). If
the corpus changes, it must be rewritten or recovered — it is what guarantees the
verbatim invariant. **Consider committing it under `eval/` as part of §2.1.**

---

## 5. Working agreements with this user

- **Interview with `AskUserQuestion` option cards**, not numbered questions in
  prose. They said so directly. Batch a round into one call, recommendation
  first, tradeoff in the description.
- **Explain commands in plain language** — purpose, outcome, and a choice. Never
  hand over a bare shell command. Prefer running it yourself and reporting.
- They read carefully and push back well. Twice they answered a question based on
  a misreading of *my* wording (they thought "lower the reap limit to 3" meant
  capping guest chat turns). **If an answer sounds like a yes to a different
  question, re-ask rather than proceeding.**
- They asked for speed, explicitly to shrink the exposure window. Prefer shipping
  verified tranches over long silent stretches.

---

## 6. Suggested skills

Call these via the `Skill` tool:

- **`anthropic-skills:grilling`** — for any multi-decision design work. This user
  responds very well to it; the guest-lifecycle design in §2.4 came out of it
  cleanly. Combine with the option-card rule in §5.
- **`code-review`** — before shipping §2.2 and §2.4. Both touch auth, deletion
  and CI, where a mistake is expensive and not obvious from tests.
- **`security-review`** — specifically for §1 (a new cookie) and §2.4 (a sweep
  endpoint with a shared secret and a deletion path). Worth one deliberate pass.

---

## 7. Open risk the user has accepted

The KFD documents were publicly fetchable for some time before the history purge
in `e2b2ecb`. The purge removed them from the repo and every commit (verified: no
blob in any of 95 commits contains the markers), and the paths now 404. It does
**not** undo any copy taken beforehand. The user has been told; treat the content
as disclosed rather than recovered.

A pre-purge backup bundle exists only in this session's scratchpad and will not
survive. If the originals matter, they must be recovered from elsewhere.
