# Handoff — agentic-RAG

Written 2026-08-18. The queue this document used to carry is cleared — see §2
for where each item landed.

**What is left is §1: the identity-blank timing bug**, plus two pieces of
configuration that live outside the repo (§2). Everything else is done and
pushed.

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
| `38bd468` | touch targets meet 44px |
| `a1c20cd` | **Deep search** replaces web fallback (web search deleted) |
| `0cad203` | guest TTL 30 min, sweeper, set-based deletes |
| `69ca56a` | CI gate + scorecard reading the same baseline |
| `df6cbfb` | confusable corpus, 39 -> 59 chunks, re-baselined |
| `26112b9` | Prune ghosts retired to the "/" palette |
| `3619d4d` | deterministic exact-containment metrics; gate baseline |
| `665b8a6` | MATCH_THRESHOLD calibrated 0.6 -> 0.45 |
| `c002445` | harness scores the passages the model is actually handed |

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

### Status

**Still open, and still unfixed.** The user has seen it, has said it can wait
until everything else is deployed, and has asked to revisit it then. Do not
treat it as agreed work until they say so.

### Suggested fix

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
it as a question, not an instruction, and later observed that it "does not seem
to improve it drastically". Worth knowing why that observation cannot be about
this fix: the hint cookie was never built. `grep -r ragchat-kind` finds nothing
outside this file. Whatever they were measuring, it was not this.

---

## 2. Queue — all cleared

Everything §2 listed is landed. Kept here as a map of where each decision lives.

| was | landed in | where it lives |
|---|---|---|
| 2.1 finish the corpus | `df6cbfb` | `eval/corpus/` (27 files, 59 chunks), `eval/build_golden_set.py` |
| 2.2 CI gate | `69ca56a` | `eval/gate.py`, `.github/workflows/retrieval-gate.yml` |
| 2.3 scorecard reads baseline | `69ca56a` | `GET /api/eval/baseline`, `frontend/app.js:scoreReference` |
| 2.4 guest lifecycle | `0cad203` | `ragchat/guests.py`, `.github/workflows/guest-sweeper.yml` |
| 2.5 phase 5 sweep | `38bd468` | `frontend/styles.css`, `frontend/landing.css` |
| 2.6 deep search | `a1c20cd` | `ragchat/deepsearch.py` |

### Measured on the deployment (2026-08-18, after everything landed)

```
guest sign-in     8.4-8.7s wire   (create 1.7s + seed 6.4s)   budget: <10s  MET
/api/auth/status  0.3-0.5s
/api/health       3.0-3.8s        (does live model discovery)
```

It was 11.1s until `_ensure_table` stopped running six DDL/catalog round trips
on every store call. There is more headroom in the remaining 8.1s and it has
not been chased: the numbers are suspiciously CONSTANT (1693-1697ms, then
6424-6448ms across seven runs), which is the signature of a fixed per-operation
cost rather than variable work — most likely a fresh Neon connection and TLS
handshake per `eng.begin()`. Connection reuse is the obvious next thing to
measure. `POST /api/auth/guest-login` returns `timings_ms`, so the next person
does not have to add instrumentation to find out.

### Two things that need doing OUTSIDE the repo

1. **`GUEST_SWEEP_SECRET`** must be set in *both* places or the sweeper is a
   no-op and guest workspaces outlive their TTL:

   ```
   vercel env add GUEST_SWEEP_SECRET
   gh secret set GUEST_SWEEP_SECRET
   ```

   The endpoint is DISABLED (404) without it, which is the safe failure — but it
   IS a failure. `DEPLOY_VERCEL.md` has the detail.

2. **`OPENROUTER_API_KEY` is over its spend limit.** Every embedding call answers
   403 "Key limit exceeded". That is why the re-baseline was written with
   `python -m eval.baseline --from-run latest` rather than a fresh scoring run,
   and why deep search has not been exercised against the live LLM path — its
   own scan needs no API and was verified end to end. The CI gate treats a 403 as
   "gate skipped", so it will pass, loudly, until the key is topped up.

### Behaviour change worth knowing about

`similarity_threshold` is live again. Its only previous use was gating web
augmentation, so deleting that feature would have left an exposed setting
decorating a meter. `ask()` now refuses before generating when the pool clears
nothing — and deliberately does NOT refuse when no chunk carries a cosine at
all, because a BM25-only pool is the absence of evidence rather than evidence of
absence (that is exactly the part-number case fusion exists for).

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

**A bulk DELETE leaves the session's identity map intact.** `SessionLocal` sets
`expire_on_commit=False`, so after `query(...).delete()` a `db.get(User, id)`
still returns a live-looking object for a row that is gone. `purge_users`
expunges them — expunge and not expire, because an expired instance re-SELECTs
on the next attribute read and raises, which would make reading `guest.id` off
the returned list an error.

**Python's `\w` matches CJK.** A plain tokenizer turns a whole Chinese clause
into one "word" that appears verbatim in no document, so a literal search finds
nothing and looks like a correct "not in the corpus" rather than a bug.
`deepsearch` uses character bigrams.

**Tests can silently start hitting the network.** `test_retrieval_fixes.py`
monkeypatches `ProxyEmbeddings` at module level, and by the time the whole suite
has run, other modules have re-imported `ragchat.embeddings` and put the real one
back. A test there that depends on live retrieval quietly becomes a test of
whether an API key works — and `ask()` also calls the LLM judge at the end, which
cost ~15s per test in retries. Stub `retrieve` and `_eval_answer` rather than
indexing a corpus. The suite is 178 tests in ~12s with no network; if it takes a
minute, something is calling out.

## 4. Verification

```bash
# Tests — 178 passing, ~12s, no network
.venv\Scripts\python -m pytest tests/ -v

# Free retrieval benchmark (no LLM call). --with-rerank costs one Cohere
# call per question; --ceiling separates a ranking problem from a retrieval one.
.venv\Scripts\python -m eval.run_eval --retrieval-only
.venv\Scripts\python -m eval.run_eval --retrieval-only --with-rerank --ceiling

# Regenerate the baseline after ANY corpus / golden set / chunking / model change.
# Read the diff before committing — regenerating blindly launders a regression.
# --from-run writes it from a run that already completed, instead of paying for
# the same measurement twice.
.venv\Scripts\python -m eval.baseline
.venv\Scripts\python -m eval.baseline --from-run latest

# What CI runs. Passes loudly when the provider is down; fails on a regression,
# and on a baseline that no longer describes the pipeline.
.venv\Scripts\python -m eval.gate

# Demo vectors, after an embedding-model or chunking change (else guest-login 504s)
.venv\Scripts\python -m ragchat.demo_vectors

# Frontend: both pages x 5 breakpoints x 2 themes; fails on overflow or page error
node shot.mjs http://localhost:5173
node layout_check.mjs http://localhost:5173
```

The golden-set generator is committed at `eval/build_golden_set.py`. It refuses
to write unless every golden passage is verbatim in a source document AND absent
from every distractor — a passage present in two places gives the corpus two
right answers, and the exact metrics then score the same retrieval as a hit or a
miss depending on which copy came back.

---

## 5. Suggested skills

Call these via the `Skill` tool:

- **`anthropic-skills:grilling`** — for any multi-decision design work. This user
  responds very well to it; the guest-lifecycle design came out of it cleanly.
- **`code-review`** — the guest sweeper and deep search both shipped without
  one. The sweeper deletes data behind a shared secret; deep search reads every
  document a user owns and puts the text in a model prompt. Both are worth a
  deliberate pass now that they are written rather than before the next change.
- **`security-review`** — for the same two, plus §1 if the hint cookie is built.

---

## 6. Open risk the user has accepted

The KFD documents were publicly fetchable for some time before the history purge
in `e2b2ecb`. The purge removed them from the repo and every commit (verified: no
blob in any of 95 commits contains the markers), and the paths now 404. It does
**not** undo any copy taken beforehand. The user has been told; treat the content
as disclosed rather than recovered.

A pre-purge backup bundle exists only in this session's scratchpad and will not
survive. If the originals matter, they must be recovered from elsewhere.
