// RAG-it frontend — vanilla JS, talks to the FastAPI backend via /api.

const $ = (id) => document.getElementById(id);

const state = {
  user: null,
  isGuest: false,    // anonymous visitor on a throwaway workspace (provider="guest")
  guestUsage: null,  // {documents, max_documents, bytes, max_bytes, …} or null when signed in
  googleOAuth: false, // whether this deployment has Google sign-in configured
  chats: [],
  currentChatId: null,
  currentCitations: [], // citations of the last assistant message, for the excerpt pane
  models: { chat: [], embedding: [] }, // proxy model catalog for the settings dropdowns
  evalData: null,     // last /api/eval payload, so an answer can place itself against the run
  evalBaseline: null, // eval/baseline.json — what a bar is measured against
  simThreshold: 0,    // live retrieval threshold, marked on the per-answer meter
  seeding: null,      // in-flight sample-document copy, so the sources pane can say so
  answerEval: null,   // the answer currently drawn on the benchmark bars
  bootUsageSkipped: false, // consumes the one duplicate status fetch at boot
  // Which tools the app is ALLOWED to reach for on the next question. They say
  // what exists for it to decide between, not what it must use. Per-question,
  // sent with ask, never stored — a stored toggle would be one row shared by
  // the whole deployment (see the ask route).
  //
  // Deep search on, web search OFF. Deep search reads the reader's own
  // documents, so using it is still answering from their material. The web is
  // not theirs, and an app whose claim is "grounded in your documents" cannot
  // quietly widen that when the documents fall short. Turning web search on is
  // the reader widening it deliberately.
  deepSearch: true,
  webSearch: false,
  answerEvalId: null,   // which answer's readings the benchmark bars show
  webAvailable: false,   // server-reported: needs a key AND an account
  // Documents + folders currently in the workspace. Only the COUNT is kept:
  // it is what the empty conversation needs to say something true, and holding
  // the full lists here would be a second copy of the DOM's own state.
  sourceCount: 0,
  // True when the whole seeded demo corpus is present (both files) — it decides
  // the empty-state copy, not the chips.
  demoDocs: false,
  // Suggested questions for the seeded documents actually in the workspace.
  demoQuestions: [],
  // Which beats of the four-beat loop this browser has completed (§5).
  loop: {},
  // Documents whose sliced-index loop is already being driven by this tab.
  // Guards against two loops racing the same document — they would both POST
  // /index-step, and the second would embed a slice the first had just done.
  indexing: new Set(),
};

// Human-friendly labels for known models. With live proxy discovery the
// catalog is dynamic, so any model not listed here simply shows its raw id.
const MODEL_LABELS = {
  "deepseek-v4-pro": "DeepSeek V4 Pro (class default)",
  "qwen3.8-max": "Qwen3.8 Max",
  "qwen3-coder": "Qwen3 Coder (metered)",
  // Embedding models. Every one is stored at 768 dims (the Neon `chunks`
  // table has a single fixed vector(768) column), so the label says so.
  "models/gemini-embedding-001": "gemini-embedding-001 (Gemini, 768d)",
  "openai/text-embedding-3-small": "text-embedding-3-small (OpenRouter, 768d)",
  "openai/text-embedding-3-large": "text-embedding-3-large (OpenRouter, 768d)",
  "qwen/qwen3-embedding-8b": "Qwen3 Embedding 8B (OpenRouter, 768d)",
  "qwen/qwen3-embedding-4b": "Qwen3 Embedding 4B (OpenRouter, 768d)",
  "perplexity/pplx-embed-v1-0.6b": "pplx-embed v1 0.6B (OpenRouter, 768d)",
  "google/gemini-embedding-001": "gemini-embedding-001 (OpenRouter, 768d)",
};

function modelLabel(id) {
  return MODEL_LABELS[id] || id;
}

// ---------- helpers ----------

async function api(path, options = {}) {
  const res = await fetch(path, {
    method: options.method || "GET",
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    credentials: "same-origin",
    ...options,
  });
  if (res.status === 401 && !path.startsWith("/api/auth")) {
    showAuth();
    throw new Error("Not authenticated");
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

let toastTimer = null;
function toast(message, isError = false) {
  const el = $("toast");
  el.textContent = message;
  el.className = `toast${isError ? " error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 4000);
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

// Shared markup for the "nothing here yet" panels. `extra` is trusted markup
// (beat lists, question chips) appended below the copy — callers pass literals,
// never anything from the server.
function emptyState(icon, title, text, extra = "") {
  return `<div class="empty-state">
      <span class="empty-icon" aria-hidden="true">${icon}</span>
      ${title ? `<p class="empty-title">${escapeHtml(title)}</p>` : ""}
      <p class="empty-text">${escapeHtml(text)}</p>
      ${extra}
    </div>`;
}

// ---------- the four-beat loop (IDEA.md §5) ----------
//
// The app has a natural loop — add a source → ask → click a citation → read the
// quality readout — and nothing used to tell the user it existed. Each empty
// state describes its own beat; the numbered list is rendered ONLY in the empty
// state for the beat you are actually on, so the guide sits next to the thing to
// do next instead of appearing three times at once.
//
// Beat 1 is derived from the live source count rather than remembered, so
// deleting every source honestly puts you back at the start. Beats 2–4 are
// per-browser learning state, which is why they persist in localStorage and not
// in the database: they say what this visitor has been shown, not what the
// workspace contains.
const LOOP_KEY = "ragchat-loop";
const BEATS = [
  "Add a source",
  "Ask a question",
  "Click a citation",
  "Read the quality readout",
];

function readLoop() {
  try { return JSON.parse(localStorage.getItem(LOOP_KEY)) || {}; }
  catch { return {}; }          // private mode / corrupt value — start over
}

// Read at module scope, not in boot(): refreshSources paints the beats and runs
// from initAuth, which starts before boot's own body finishes.
state.loop = readLoop();

function markBeat(name) {
  if (state.loop[name]) return;  // already done; don't churn storage or repaint
  state.loop[name] = true;
  try { localStorage.setItem(LOOP_KEY, JSON.stringify(state.loop)); }
  catch { /* storage disabled: the guide just restarts next visit */ }
  paintBeats();
}

// Which beat the user is on, or 0 once the loop is complete.
function currentBeat() {
  if (!state.sourceCount) return 1;
  if (!state.loop.asked) return 2;
  if (!state.loop.cited) return 3;
  if (!state.loop.readout) return 4;
  return 0;
}

// Returns the numbered list for `beat`, or "" when that beat is not the current
// one — which is what keeps a single strip on screen.
function beatsHtml(beat) {
  const now = currentBeat();
  if (now !== beat) return "";
  const rows = BEATS.map((label, i) => {
    const n = i + 1;
    const st = n < now ? "done" : n === now ? "current" : "todo";
    const pip = st === "done" ? "✓" : String(n);
    return `<li class="loop-beat" data-state="${st}">
        <span class="loop-pip" aria-hidden="true">${pip}</span>
        <span class="loop-label">${escapeHtml(label)}</span>
      </li>`;
  }).join("");
  return `<ol class="loop-list" aria-label="Getting started: step ${now} of 4">${rows}</ol>`;
}

// The two static empty states own their own beat container; the chat's is built
// with the rest of its markup. Called whenever a beat completes.
function paintBeats() {
  const s = $("sources-beats");
  if (s) s.innerHTML = beatsHtml(1);
  const x = $("excerpt-beats");
  if (x) x.innerHTML = beatsHtml(3);
  if ($("messages").querySelector(".empty-state")) {
    $("messages").innerHTML = chatEmptyState();
  }
  paintReadoutNudge();
}

// Beat 4's nudge is the one that cannot be rendered once and left alone: the
// beat becomes current when a citation is clicked, by which time the answer's
// readout chip is already on screen. It attaches to the LAST answer only —
// repeating it down a long conversation would be noise, not guidance.
function paintReadoutNudge() {
  const chips = [...$("messages").querySelectorAll(".eval-chip")];
  const want = currentBeat() === 4;
  chips.forEach((chip, i) => {
    const existing = chip.querySelector(".eval-nudge");
    const wanted = want && i === chips.length - 1;
    if (wanted && !existing) {
      const el = document.createElement("span");
      el.className = "eval-nudge";
      el.textContent = "what is this?";
      chip.insertBefore(el, chip.querySelector(".eval-caret"));
    } else if (existing && !wanted) {
      existing.remove();
    }
  });
}

// ---------- glossary (IDEA.md §5) ----------
//
// Every one of these was previously either undefined or explained in a `title`
// attribute, which does not exist on touch. A tap target with a real popover
// works for everyone; the trade is one shared element and a click-away handler.
const GLOSSARY = {
  "deep-search": [
    "Deep search",
    "Normal search RANKS: it picks the passages that look most like your " +
    "question, and a passage that ranks 21st out of 20 is simply not there. " +
    "Deep search does not rank. It reads every document you own word for " +
    "word and pulls out every literal occurrence of your terms, so if the " +
    "words are in your documents they reach the answer. You do not have to " +
    "ask for it: when ranked search comes up short, the app runs it by " +
    "itself and says so under the answer. Turn it on to force it on every " +
    "question. Either way it applies only to the question you send it with — " +
    "it is not a setting.",
  ],
  "web-search": [
    "Web search",
    "OFF by default, because this app answers from YOUR documents and the web " +
    "is not one of them. Turn it on and the app may look outside — but only " +
    "as a last resort: ranked search first, then reading every document you " +
    "own word for word, and only if both come up empty. Anything it finds is " +
    "labelled as a web source in the answer and in its citation, and your own " +
    "documents win any disagreement. Signed-in only, and it applies to the " +
    "question you send it with — it is not a setting.",
  ],
  "re-index": [
    "Re-index",
    "Chunks and embeds your sources again under the current settings. Needed " +
    "after changing chunk size, the splitter or the embedding model, because " +
    "vectors made by different settings are not comparable.",
  ],
  excerpt: [
    "Excerpt",
    "The exact passage an answer was built from, shown verbatim. Clicking a " +
    "citation marker in an answer opens the chunk that citation points at — " +
    "this is where you check the answer against the source.",
  ],
  "golden-set": [
    "Golden set",
    "46 questions with known answers, including ones the corpus deliberately " +
    "cannot answer. Replaying them through the live pipeline is what turns " +
    "“seems fine” into a number you can compare between configurations.",
  ],
  faithfulness: [
    "Faithfulness",
    "Whether every claim in the answer is supported by the retrieved passages. " +
    "A confident sentence with no passage behind it fails, however true it " +
    "happens to be — that is the hallucination check.",
  ],
  relevancy: [
    "Answer relevancy",
    "Whether the answer addresses the question that was asked. An answer can be " +
    "perfectly faithful to the sources and still not be a reply, which is why " +
    "this is scored separately.",
  ],
};

// Row labels in the per-answer readout that have a glossary entry behind them.
const EVAL_LABEL_TERMS = {
  Faithfulness: "faithfulness",
  Relevancy: "relevancy",
};

// Markup for an inline term: dotted underline, tap to define.
function termHtml(key, label) {
  return `<button class="term" type="button" data-term="${key}" aria-expanded="false">${
    escapeHtml(label ?? GLOSSARY[key][0])
  }</button>`;
}

let glossaryOpenFor = null;

function glossaryEl() {
  let el = $("glossary-pop");
  if (!el) {
    el = document.createElement("div");
    el.id = "glossary-pop";
    el.className = "glossary-pop hidden";
    el.setAttribute("role", "tooltip");
    document.body.appendChild(el);
  }
  return el;
}

function closeGlossary() {
  if (glossaryOpenFor) glossaryOpenFor.setAttribute("aria-expanded", "false");
  glossaryOpenFor = null;
  glossaryEl().classList.add("hidden");
}

function openGlossary(btn) {
  const entry = GLOSSARY[btn.dataset.term];
  if (!entry) return;
  const el = glossaryEl();
  el.innerHTML = `<p class="glossary-term">${escapeHtml(entry[0])}</p>
    <p class="glossary-def">${escapeHtml(entry[1])}</p>`;
  el.classList.remove("hidden");
  btn.setAttribute("aria-expanded", "true");
  glossaryOpenFor = btn;

  // Fixed positioning, clamped to the viewport: these terms sit in a topbar at
  // the right edge and in a pane at the bottom, so a popover anchored naively
  // would open off-screen on the very controls that needed explaining.
  const r = btn.getBoundingClientRect();
  const w = el.offsetWidth;
  const h = el.offsetHeight;
  const pad = 8;
  const left = Math.max(pad, Math.min(window.innerWidth - w - pad, r.left + r.width / 2 - w / 2));
  const below = r.bottom + 8;
  const top = below + h + pad > window.innerHeight ? Math.max(pad, r.top - h - 8) : below;
  el.style.left = `${Math.round(left)}px`;
  el.style.top = `${Math.round(top)}px`;
}

// Delegated: terms are rendered from three different places (static HTML, empty
// states, the eval pane) and re-rendered often, so binding per button would mean
// re-binding on every repaint.
document.addEventListener("click", (e) => {
  const btn = e.target.closest(".term, .term-mark");
  if (btn) {
    e.stopPropagation();
    if (glossaryOpenFor === btn) closeGlossary();
    else openGlossary(btn);
    return;
  }
  if (glossaryOpenFor && !e.target.closest("#glossary-pop")) closeGlossary();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && glossaryOpenFor) closeGlossary();
});

// A popover positioned in fixed coordinates cannot follow its anchor, so close
// rather than leave it stranded mid-page.
window.addEventListener("resize", closeGlossary);
window.addEventListener("scroll", closeGlossary, true);

// ---------- auth ----------

function showAuth() {
  $("app-view").classList.add("hidden");
  $("auth-view").classList.remove("hidden");
}

function showApp() {
  $("auth-view").classList.add("hidden");
  $("app-view").classList.remove("hidden");
}

async function initAuth() {
  try {
    let status = await api("/api/auth/status");

    // The landing page's secondary CTA is "Sign in with Google", and it links
    // to /app?signin=1 rather than straight to /api/auth/google/login. Going
    // direct would 400 on a deployment without OAuth configured, showing raw
    // JSON as the first thing a visitor sees. Bouncing through here means the
    // app can check first and fall back to its own sign-in card.
    // The guest check is NOT redundant with !authenticated: a guest IS
    // authenticated as far as the API is concerned, and under guest-first every
    // returning visitor already has a guest session. Testing only
    // !status.authenticated would make the landing page's "Sign in with Google"
    // button silently do nothing for everyone who had visited before.
    // Signing in from a guest session is also the good case — it promotes the
    // work they already did rather than discarding it.
    const params = new URLSearchParams(window.location.search);
    if (params.get("signin") && (!status.authenticated || status.is_guest)) {
      if (status.google_oauth) {
        goToOAuth();
        return;
      }
      renderAuthGate(status);
      showAuth();
      return;
    }

    // GUEST-FIRST: an unauthenticated visitor is given their own private,
    // throwaway workspace rather than a sign-in wall. This is the whole premise
    // of guest mode — "start using it now, sign in when you want to keep it" —
    // and it is safe to guess wrong, because signing in PROMOTES the guest's
    // documents and chats into the permanent account instead of discarding them
    // (app.py:_promote_prior_guest).
    //
    // The previous condition was `!status.authenticated && !status.google_oauth`,
    // which only fell back when OAuth was UNCONFIGURED. Production has OAuth
    // configured, so every visitor hit the gate and the entire guest subsystem
    // was unreachable there. It also called /api/auth/local-login, which signs
    // everyone into ONE shared account — the exact mixing per-user isolation
    // exists to prevent. guest-login gives each visitor their own.
    if (!status.authenticated) {
      try {
        const created = await api("/api/auth/guest-login", { method: "POST" });
        status = await api("/api/auth/status");
        // The sample documents are NOT in that workspace yet. Filling it is a
        // second request, deliberately: doing it inline measured 6.4s of
        // copying in front of an empty screen, and nobody asked for sample
        // documents before they had seen the app. Started here and pointedly
        // NOT awaited — the workspace is already usable, and this lands while
        // the visitor is still reading the page.
        if (created && created.seeded === false) state.seeding = seedGuestWorkspace();
      } catch (e) {
        // No guest workspace could be provisioned (DB down, seeding failed).
        // Fall through to the sign-in gate rather than booting into an app
        // whose every fetch would 401.
        console.error("guest login failed:", e);
      }
    }
    if (!status.authenticated) {
      renderAuthGate(status);
      showAuth();
      return;
    }
    applyAuthStatus(status);
  } catch (e) {
    console.error("auth failed:", e);
    // Stand the skeleton down even though nothing was resolved. It is sized to
    // the real button so the bar does not move when it is replaced — but if it
    // is never replaced it sits BESIDE the real one, and two sign-in shapes in
    // a 320px top bar collide with the controls next to them. An unanswered
    // status call is a reason to stop reserving space, not to reserve it
    // forever.
    document.documentElement.classList.add("identity-resolved");
  }
  showApp();
  try {
    await Promise.all([refreshSources(), refreshChats(), refreshLiveConfig(), loadEval()]);
  } catch (e) {
    console.error("boot fetch failed:", e);
  }
}

// Single place that reads the /api/auth/status payload into app state. Both the
// boot path and the identity badge need it; fetching it twice raced and could
// render a signed-in chip over a guest workspace.
function applyAuthStatus(status) {
  state.user = status.user;
  state.isGuest = !!status.is_guest;
  // Reconcile the boot hint with the truth and stand the skeleton down. The
  // cookie is only a hint — it can be stale (cleared session, expired guest)
  // or forged — so whatever it said, this is what the page renders from now on.
  const root = document.documentElement;
  root.setAttribute("data-kind", status.is_guest ? "guest" : "account");
  root.classList.add("identity-resolved");
  state.guestUsage = status.guest_usage || null;
  state.googleOAuth = !!status.google_oauth;
  // Whether the WEB tool exists for this caller. The server decides — it needs
  // a provider key and a signed-in account — so the switch can never offer
  // something the ask route would refuse.
  state.webAvailable = !!status.web_search_available;
  syncToolToggles();

  // A returning guest whose sample documents were copied under a superseded
  // config fingerprint. Their chunks are unreachable, so the workspace lists
  // two READY documents and answers every question with "I couldn't find this
  // in your documents" — looking perfect and being empty underneath. They
  // cannot re-index either; that is an account-only route.
  //
  // This branch exists because the seeding call above runs only for a visitor
  // who is NOT yet authenticated, and a returning guest already is. Nothing
  // else on the boot path would ever notice.
  if (status.demo_needs_reseed && !state.seeding) {
    state.seeding = seedGuestWorkspace().then(() => refreshSources());
  }

  const nameEl = $("user-name");
  if (nameEl) nameEl.textContent = state.isGuest ? "" : state.user?.name || "";
  renderGuestState();
  applyGuestLocks();
}

// The #auth-view card exists in index.html but nothing ever wired it up: both
// the Google link and the password form are `hidden` by default and no code
// revealed them, so hitting a 401 showed a dead card with no way to sign in.
function renderAuthGate(status) {
  const googleBtn = $("google-btn");
  const pwWrap = $("password-auth");
  if (googleBtn) googleBtn.classList.toggle("hidden", !status.google_oauth);
  // Keep the password fallback available when Google isn't configured, so a
  // deployment without OAuth is still usable with real per-user accounts.
  if (pwWrap) pwWrap.classList.toggle("hidden", !!status.google_oauth);

  const err = $("auth-error");
  const submit = async (path) => {
    const username = $("auth-username")?.value.trim();
    const password = $("auth-password")?.value || "";
    if (!username || !password) {
      if (err) err.textContent = "Enter a username and password.";
      return;
    }
    try {
      await api(path, { method: "POST", body: JSON.stringify({ username, password }) });
      window.location.reload();
    } catch (e) {
      if (err) err.textContent = e.message;
    }
  };
  const loginBtn = $("login-btn");
  const registerBtn = $("register-btn");
  if (loginBtn) loginBtn.onclick = () => submit("/api/auth/login");
  if (registerBtn) registerBtn.onclick = () => submit("/api/auth/register");
}

// ---------- guest state ----------

// Controls a guest may not use, each with the phrase that completes
// "Sign in with Google to ___". These are denied SERVER-side too
// (app.py:require_account); this only makes the refusal legible in advance
// instead of arriving as a 403 after the click.
//
// The list is deliberately short. Everything absent from it — upload, delete,
// ask, new chat, prune, theme — a guest can do, because a guest workspace is a
// real workspace, not a display case.
const GUEST_LOCKED = {
  "reindex-btn": "re-index every source",
  "add-folder-btn": "add a folder source",
  "settings-save": "save tuning settings",
};

// Reason these are global, not personal: config_overrides is a single shared
// row (db.py:121), so one visitor's "save" re-points the embedding model for
// everyone and invalidates their chunks. Benchmarks spend real LLM quota.
const GUEST_LOCK_WHY =
  "Those settings apply to the whole deployment, so they need an account.";

function applyGuestLocks() {
  for (const [id, what] of Object.entries(GUEST_LOCKED)) {
    const el = $(id);
    if (!el) continue;
    el.classList.toggle("guest-locked", state.isGuest);
    // aria-disabled rather than `disabled`: a disabled button dispatches no
    // click, so the visitor would get silence instead of a reason. Keeping it
    // clickable is what lets guestBlocked() explain itself — and explaining
    // beats hiding (IDEA.md §5).
    if (state.isGuest) el.setAttribute("aria-disabled", "true");
    else el.removeAttribute("aria-disabled");
    el.title = state.isGuest ? `Sign in with Google to ${what}.` : el.dataset.title || el.title;
  }
  const folderInput = $("folder-input");
  if (folderInput) {
    folderInput.disabled = state.isGuest;
    // Short enough not to truncate in the narrow sources pane — the previous
    // wording ended as "Folder sources need an accc".
    folderInput.placeholder = state.isGuest
      ? "Sign in to add folders"
      : "Add folder path (e.g. ~/documents)";
  }
}

// Call at the top of a handler for a locked control. Returns true if the action
// was refused, in which case the handler must return without doing anything.
function guestBlocked(elementId) {
  if (!state.isGuest) return false;
  const what = GUEST_LOCKED[elementId] || "do that";
  toast(`Sign in with Google to ${what}. ${GUEST_LOCK_WHY}`, true);
  return true;
}

// The badge states three things at once: that this is a guest workspace, how
// much of the allowance is spent, and that signing in KEEPS the work. That last
// clause is the whole pitch — without it "sign in" reads as a paywall rather
// than as a save button.
function renderGuestState() {
  const el = $("guest-badge");
  if (!el) return;
  el.classList.toggle("hidden", !state.isGuest);
  if (!state.isGuest) return;

  const u = state.guestUsage;
  // Usage comes from guest_usage, NOT from the length of /api/documents: the
  // seeded demo files are excluded from the cap, so counting documents reads
  // two too high and would show a fresh visitor as already 2/3 full.
  const used = u ? u.documents : 0;
  const max = u ? u.max_documents : 3;
  const full = u && used >= max;
  // The label carries what the phone hides: below 560px the "Guest" tag and the
  // word "files" are dropped for room, leaving a bare "0/3" that means nothing
  // to a screen reader without this.
  el.setAttribute("aria-label", `Guest workspace, ${used} of ${max} files used`);
  el.innerHTML = `<span class="guest-badge-tag">Guest</span>
    <span class="guest-badge-usage${full ? " is-full" : ""}">${used}/${max}<span class="usage-word"> files</span></span>
    <span class="guest-badge-note">Sign in to keep your work</span>`;
}

// Usage changes on every add and delete, so the badge has to be refreshed from
// the server rather than incremented locally — the server is the only thing
// that knows which documents count against the cap.
// The sources pane is empty for two different reasons and they need different
// words. "Upload a source to begin" under a workspace whose sample documents
// are still being copied reads as "there will never be anything here", which
// is how a visitor decides the app is broken and leaves.
function renderSourcesEmptyState() {
  const el = $("sources-empty");
  if (!el) return;
  const empty = state.sourceCount === 0;
  el.classList.toggle("hidden", !empty);
  if (!empty) return;
  const waiting = !!state.seeding;
  el.classList.toggle("is-waiting", waiting);
  const title = el.querySelector(".empty-title");
  const text = el.querySelector(".empty-text");
  if (waiting) {
    title.textContent = "Adding sample documents…";
    text.textContent =
      "Your workspace is ready — the samples are being copied into it. " +
      "You can upload your own files now without waiting.";
  } else {
    title.textContent = "Upload a source to begin";
    text.textContent =
      "Drop a file above, paste a URL, or point at a folder. " +
      "Everything you add is chunked, embedded and cited.";
  }
}

// Fill a new guest workspace with the sample documents, in the background.
//
// Failure here is not fatal and must not be shouted about: an empty workspace
// is a worse demo, not a broken app, and the visitor can upload their own
// files regardless. The sources pane says what is happening while it runs.
async function seedGuestWorkspace() {
  try {
    const r = await api("/api/auth/guest-seed", { method: "POST" });
    await refreshSources();
    return r;
  } catch (e) {
    console.error("sample documents could not be added:", e);
    return null;
  } finally {
    state.seeding = null;
    renderSourcesEmptyState();
  }
}

async function refreshGuestUsage() {
  if (!state.isGuest) return;
  // Skip exactly ONE fetch: the one refreshSources() triggers during boot,
  // moments after initAuth() already fetched status and applyAuthStatus stored
  // the usage from it. That duplicate landed at 3669ms and re-rendered the
  // badge well after first paint — a visible change with no explanation.
  //
  // A time window was the obvious guard and was wrong: uploading a file within
  // ten seconds of opening the page would have been skipped too, leaving the
  // "0 of 3 files" badge stale for exactly the visitor most likely to be
  // watching it. One shot, consumed at boot, is what was actually meant.
  if (state.bootUsageSkipped === false) {
    state.bootUsageSkipped = true;
    renderGuestState();
    return;
  }
  try {
    const status = await api("/api/auth/status");
    state.guestUsage = status.guest_usage || null;
    renderGuestState();
  } catch (e) {
    /* the badge going stale is not worth interrupting the user for */
  }
}

// ---------- optional Google sign-in ----------
//
// Identity badge in the top bar. When Google OAuth is NOT configured the app
// still boots signed-out into the shared `local` account; when it IS
// configured, signing in gives each account its own isolated space.
// The status payload carries no avatar URL, so the chip uses the initial.

const GOOGLE_G_SVG = `<svg class="google-g" viewBox="0 0 18 18" width="16" height="16" aria-hidden="true">
    <path fill="#4285F4" d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.92c1.7-1.57 2.68-3.88 2.68-6.62Z"/>
    <path fill="#34A853" d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.8.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.03-3.7H.96v2.33A9 9 0 0 0 9 18Z"/>
    <path fill="#FBBC05" d="M3.97 10.72a5.4 5.4 0 0 1 0-3.44V4.95H.96a9 9 0 0 0 0 8.1l3.01-2.33Z"/>
    <path fill="#EA4335" d="M9 3.58c1.32 0 2.5.46 3.44 1.35l2.58-2.58C13.46.9 11.43 0 9 0A9 9 0 0 0 .96 4.95l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58Z"/>
  </svg>`;

function renderSignedOut(slot, configured) {
  slot.innerHTML = `<button id="google-signin" class="btn btn-small google-btn" type="button"
      title="${configured === false
        ? "Google sign-in is not configured on this deployment"
        : "Optional — the app works signed out too"}">
      ${GOOGLE_G_SVG}<span class="google-btn-text">Sign in<span
        class="google-btn-long"> with Google</span></span>
    </button>`;
  slot.querySelector("#google-signin").onclick = () => {
    if (configured === false) {
      toast("Google sign-in isn't configured on this deployment.", true);
      return;
    }
    // /api/auth/login is POST-only (username+password). Navigating to it issued
    // a GET, which 405'd and never reached Google — this is why the button
    // "did nothing". The OAuth entry point is /api/auth/google/login.
    // Through goToOAuth(), which suppresses the close beacon: this navigation
    // is a round trip, not a departure, and the workspace must survive it.
    goToOAuth();
  };
}

function renderSignedIn(slot, me) {
  const name = me.name || me.email || "Signed in";
  const initial = (name.trim()[0] || "?").toUpperCase();
  const avatar = me.picture
    ? `<img class="user-avatar" src="${escapeHtml(me.picture)}" alt=""
         referrerpolicy="no-referrer" />`
    : `<span class="user-avatar user-avatar-initial" aria-hidden="true">${escapeHtml(initial)}</span>`;
  slot.innerHTML = `<div class="user-chip" title="${escapeHtml(me.email || name)}">
      ${avatar}
      <span class="user-chip-name">${escapeHtml(name)}</span>
      <button id="google-signout" class="user-chip-signout" type="button">Sign out</button>
    </div>`;
  // A broken avatar URL (Google sometimes 403s them) falls back to the initial.
  const img = slot.querySelector("img.user-avatar");
  if (img) {
    img.onerror = () => {
      img.replaceWith(
        Object.assign(document.createElement("span"), {
          className: "user-avatar user-avatar-initial",
          textContent: initial,
        })
      );
    };
  }
  // Logout is POST-only too, so navigating here 405'd and left you signed in.
  slot.querySelector("#google-signout").onclick = async () => {
    try {
      await api("/api/auth/logout", { method: "POST" });
    } catch (e) {
      /* clearing the cookie is best-effort; reload reflects the real state */
    }
    // The server clears both cookies together, but if that request failed the
    // identity hint would still say "account" and the next load would paint a
    // signed-in top bar over a signed-out session for a second.
    document.documentElement.removeAttribute("data-kind");
    window.location.reload();
  };
}

async function initGoogleAuth() {
  const slot = $("auth-slot");
  if (!slot) return;
  // Surface a failed round-trip once, then clean the URL.
  const params = new URLSearchParams(window.location.search);
  const authFlag = params.get("auth");
  if (authFlag) {
    toast(
      authFlag === "unavailable"
        ? "Google sign-in isn't configured on this deployment."
        : "Google sign-in failed — you can keep using the app signed out.",
      true
    );
    params.delete("auth");
    const qs = params.toString();
    window.history.replaceState({}, "", window.location.pathname + (qs ? `?${qs}` : ""));
  }

  // Reads the status initAuth() already fetched (applyAuthStatus stored it)
  // rather than fetching /api/auth/status a second time. The duplicate call
  // raced the guest-login above it: whichever landed first decided the badge, so
  // a freshly provisioned guest could render as signed-out.
  //
  // A guest IS authenticated as far as the API is concerned, but must still see
  // the sign-in button — that button is the only path to keeping their work.
  if (state.user && !state.isGuest) renderSignedIn(slot, state.user);
  else renderSignedOut(slot, state.googleOAuth);
}

// ---------- sources ----------

// Backend status -> human label + badge colour class.
const STATUS_META = {
  ready: { label: "ready", cls: "ready" },
  indexing: { label: "embedding…", cls: "working" },
  pending: { label: "queued", cls: "working" },
  failed: { label: "error", cls: "error" },
};

// File-type chips: a short uppercase tag in a type-coloured square, so a
// source list is scannable at a glance (PDF vs web page vs spreadsheet).
const EXT_KINDS = {
  pdf: ["pdf", "PDF"],
  md: ["doc", "MD"], markdown: ["doc", "MD"], rst: ["doc", "RST"],
  html: ["web", "HTML"], htm: ["web", "HTML"],
  txt: ["text", "TXT"], log: ["text", "LOG"],
  csv: ["data", "CSV"], json: ["data", "JSON"], yaml: ["data", "YML"], yml: ["data", "YML"],
  mp3: ["audio", "MP3"], wav: ["audio", "WAV"], m4a: ["audio", "M4A"],
  mp4: ["video", "MP4"], mov: ["video", "MOV"], webm: ["video", "WEBM"],
};

function sourceKind(doc) {
  if (doc.source_type === "url") return { kind: "web", tag: "WEB" };
  const name = String(doc.path_or_url || doc.title || "");
  const ext = name.includes(".") ? name.split(".").pop().toLowerCase() : "";
  const hit = EXT_KINDS[ext];
  if (hit) return { kind: hit[0], tag: hit[1] };
  return { kind: "text", tag: (ext || "file").slice(0, 4).toUpperCase() };
}

function renderFolders(folders) {
  const list = $("folder-list");
  list.innerHTML = "";
  for (const f of folders) {
    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `
      <span class="src-icon" data-kind="folder" aria-hidden="true">DIR</span>
      <span class="src-title" title="${escapeHtml(f.path)}">${escapeHtml(f.path)}</span>
      <span class="src-actions">
        <button class="icon-btn" data-act="rescan" title="Rescan folder">↻</button>
        <button class="icon-btn" data-act="remove" title="Remove folder source">✕</button>
      </span>
      <span class="src-sub">
        <span class="badge-status ready">folder</span>
        <span class="src-meta">${f.n_docs} docs</span>
      </span>`;
    item.querySelector('[data-act="rescan"]').onclick = async () => {
      try {
        // A folder scan walks the disk and can queue many documents, so it
        // reports in the persistent job line rather than in a toast that is gone
        // before the scan is.
        setJobStatus(`Scanning ${f.path}…`, { busy: true });
        const r = await api(`/api/folders/${f.id}/rescan`, { method: "POST" });
        setJobStatus(
          `Rescan: +${r.added} new, ${r.reindexed} updated, ${r.unchanged} unchanged${r.failed ? `, ${r.failed} failed` : ""}`,
          { sticky: true }
        );
        await refreshSources();
      } catch (e) {
        setJobStatus(`Rescan failed: ${e.message}`, { sticky: true });
        toast(e.message, true);
      }
    };
    item.querySelector('[data-act="remove"]').onclick = async () => {
      try {
        await api(`/api/folders/${f.id}`, { method: "DELETE" });
        await refreshSources();
      } catch (e) { toast(e.message, true); }
    };
    list.appendChild(item);
  }
}

function renderDocs(docs) {
  const list = $("doc-list");
  list.innerHTML = "";
  for (const d of docs) {
    const item = document.createElement("div");
    item.className = "source-item";
    item.dataset.docId = d.id;
    const { kind, tag } = sourceKind(d);
    const status = STATUS_META[d.status] || { label: d.status || "unknown", cls: "" };
    const sub = d.source_type === "url" ? d.path_or_url : (d.path_or_url || "upload");
    item.innerHTML = `
      <span class="src-icon" data-kind="${kind}" aria-hidden="true">${escapeHtml(tag)}</span>
      <span class="src-title" title="${escapeHtml(sub || "")}">${escapeHtml(d.title)}</span>
      <span class="src-actions">
        <button class="icon-btn" data-act="delete" title="Delete source">✕</button>
      </span>
      <span class="src-sub">
        <span class="badge-status ${status.cls}">${escapeHtml(status.label)}</span>
        ${d.n_chunks ? `<span class="src-meta">${d.n_chunks} chunks</span>` : ""}
      </span>
      ${indexProgressHtml(d)}
      ${d.error ? `<span class="src-error">${escapeHtml(d.error)}</span>` : ""}`;
    item.querySelector('[data-act="delete"]').onclick = async () => {
      try {
        await api(`/api/documents/${d.id}`, { method: "DELETE" });
        await refreshSources();
      } catch (e) { toast(e.message, true); }
    };
    list.appendChild(item);
  }
}

// The progress bar lives INSIDE the document card, so the file the user just
// dropped is the thing that reports on itself — rather than a detached toast
// that vanishes and leaves them wondering whether anything is happening.
function indexProgressHtml(d) {
  if (d.status !== "indexing") return "";
  const total = d.n_chunks || 0;
  const done = d.indexed_chunks || 0;
  // Before the first slice lands the width would be 0, which reads as stalled.
  // A small floor makes it obvious the work has started.
  const pct = total ? Math.max(4, Math.round((done / total) * 100)) : 8;
  return `<span class="src-progress" role="progressbar"
      aria-valuemin="0" aria-valuemax="${total}" aria-valuenow="${done}">
      <span class="src-progress-fill" style="width:${pct}%"></span>
    </span>
    <span class="src-progress-label">${total ? `${done} / ${total} chunks` : "reading…"}</span>`;
}

// Optimistic card, rendered from the File before any request is made, so the
// source list responds the instant a file is dropped. Replaced by the real
// document as soon as the upload returns.
function renderPendingDoc(file) {
  const item = document.createElement("div");
  item.className = "source-item is-pending";
  item.dataset.pendingName = file.name;
  const ext = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
  const hit = EXT_KINDS[ext];
  const kind = hit ? hit[0] : "text";
  const tag = hit ? hit[1] : (ext || "file").slice(0, 4).toUpperCase();
  item.innerHTML = `
    <span class="src-icon" data-kind="${kind}" aria-hidden="true">${escapeHtml(tag)}</span>
    <span class="src-title">${escapeHtml(file.name)}</span>
    <span class="src-actions"></span>
    <span class="src-sub">
      <span class="badge-status working">uploading…</span>
    </span>
    <span class="src-progress"><span class="src-progress-fill" style="width:6%"></span></span>`;
  $("doc-list").appendChild(item);
  $("sources-empty").classList.add("hidden");
  // Count the optimistic card too, or the header reads one short of what is
  // visibly on screen until the upload returns.
  const count = $("source-count");
  count.textContent = String((parseInt(count.textContent, 10) || 0) + 1);
  return item;
}

// Drive one document's sliced indexing to completion, repainting its card after
// every step. Each call does a bounded unit of work and commits server-side, so
// a refresh mid-run resumes rather than restarting.
async function driveIndexing(docId, onStep) {
  for (let guard = 0; guard < 2000; guard++) {
    let p;
    try {
      p = await api(`/api/documents/${docId}/index-step`, { method: "POST" });
    } catch (e) {
      toast(`Indexing failed: ${e.message}`, true);
      return null;
    }
    if (onStep) onStep(p);
    if (p.done) return p;
  }
  toast("Indexing did not finish — reload to resume.", true);
  return null;
}

// Repaint just one card's progress, so a running upload does not fight a full
// list re-render (which would drop the other cards' in-flight state).
function paintProgress(docId, p) {
  const el = document.querySelector(`.source-item[data-doc-id="${docId}"]`);
  if (!el) return;
  const bar = el.querySelector(".src-progress-fill");
  const label = el.querySelector(".src-progress-label");
  const total = p.n_chunks || 0;
  const pct = total ? Math.max(4, Math.round((p.indexed_chunks / total) * 100)) : 8;
  if (bar) bar.style.width = `${pct}%`;
  if (label) label.textContent = `${p.indexed_chunks} / ${total} chunks`;
}

async function refreshSources() {
  const [docs, folders] = await Promise.all([
    api("/api/documents"),
    api("/api/folders"),
  ]);
  renderFolders(folders);
  renderDocs(docs);
  // Resume any document left mid-index — after a reload, or a step that failed
  // to be driven because the tab was closed. Without this a half-indexed
  // document sits at "indexing" forever with no one advancing it.
  for (const d of docs) {
    if (d.status === "indexing" && !state.indexing.has(d.id)) {
      state.indexing.add(d.id);
      driveIndexing(d.id, (p) => paintProgress(d.id, p))
        .finally(() => { state.indexing.delete(d.id); refreshSources(); });
    }
  }
  state.sourceCount = docs.length + folders.length;
  $("source-count").textContent = String(state.sourceCount);
  renderSourcesEmptyState();
  // Suggested questions are only honest while the document they are about is
  // present, so they are collected per seeded file. Exactly two of the ten corpus
  // files are exposed to guests and that limit is deliberate (guests.py) — the
  // other eight are real business content that must not reach a visitor, which is
  // also why no landing copy may promise ten sample documents.
  // Matched on is_demo AND title: the flag alone doesn't say which questions
  // apply, and the title alone would fire on a user's own upload of the same name.
  const seeded = new Set(docs.filter((d) => d.is_demo).map((d) => d.title));
  state.demoQuestions = Object.entries(DEMO_QUESTIONS)
    .filter(([title]) => seeded.has(title))
    .flatMap(([, qs]) => qs);
  state.demoDocs = Object.keys(DEMO_QUESTIONS).every((t) => seeded.has(t));
  // A long job reports until it is actually finished, not until a toast fades.
  paintJobStatus(docs);
  paintBeats();
  paintTabCounts();
  // Sources and chats load in parallel, so the empty conversation may have
  // been painted before the count was known. Repaint it — but only while it IS
  // the empty state, never over a real conversation.
  if ($("messages").querySelector(".empty-state")) {
    $("messages").innerHTML = chatEmptyState();
  }
  // Every add and delete moves a guest's allowance. Not awaited: the badge is
  // secondary to the list it annotates, and blocking the render on a second
  // round-trip would make uploads feel slower than they are.
  refreshGuestUsage();
  return docs;
}

// Upload: click or drag-and-drop
const dropZone = $("drop-zone");
dropZone.onclick = () => $("file-input").click();
$("file-input").onchange = async (e) => {
  await uploadFiles(e.target.files);
  e.target.value = "";
};
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", async (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  await uploadFiles(e.dataTransfer.files);
});

async function uploadFiles(files) {
  // Each file gets a card in the sources list the moment it is dropped —
  // before any network call — so the app visibly reacts to the drop. The card
  // then carries its own progress bar through upload and indexing, which is
  // what stops a slow index from looking like a hung app.
  const jobs = [];
  for (const file of files) {
    const card = renderPendingDoc(file);
    jobs.push(
      (async () => {
        let doc;
        try {
          const form = new FormData();
          form.append("file", file);
          // Returns as soon as the text is extracted and staged; embedding is
          // sliced across the /index-step calls below.
          doc = await api("/api/documents/upload", { method: "POST", body: form });
        } catch (e) {
          card.querySelector(".badge-status").className = "badge-status error";
          card.querySelector(".badge-status").textContent = "failed";
          card.querySelector(".src-progress")?.remove();
          card.insertAdjacentHTML(
            "beforeend", `<span class="src-error">${escapeHtml(e.message)}</span>`
          );
          return;
        }
        // Hand the optimistic card its real identity so paintProgress can find
        // it, rather than re-rendering the whole list and losing the others.
        card.dataset.docId = doc.id;
        card.classList.remove("is-pending");
        delete card.dataset.pendingName;
        const badge = card.querySelector(".badge-status");
        if (badge) badge.textContent = "embedding…";
        if (!card.querySelector(".src-progress-label")) {
          card.insertAdjacentHTML(
            "beforeend", `<span class="src-progress-label">0 / ${doc.n_chunks || 0} chunks</span>`
          );
        }
        state.indexing.add(doc.id);
        try {
          await driveIndexing(doc.id, (p) => paintProgress(doc.id, p));
        } finally {
          state.indexing.delete(doc.id);
        }
      })()
    );
  }
  // Sequential would be safer for quota but makes a multi-file drop feel
  // serialised; the server caps each slice, so concurrency is bounded anyway.
  await Promise.all(jobs);
  await refreshSources();
}

$("add-url-btn").onclick = async () => {
  const url = $("url-input").value.trim();
  if (!url) return;
  try {
    toast("Fetching page…");
    $("add-url-btn").disabled = true;
    const doc = await api("/api/documents/url", {
      method: "POST",
      body: JSON.stringify({ url }),
    });
    $("url-input").value = "";
    // The card appears immediately in `indexing` state; refreshSources picks it
    // up and drives its slices, so the page shows the same progress bar an
    // uploaded file gets rather than a toast claiming it is already done.
    toast(`Added “${doc.title}” — indexing…`);
    await refreshSources();
  } catch (e) {
    toast(e.message, true);
  } finally {
    $("add-url-btn").disabled = false;
  }
};

$("add-folder-btn").onclick = async () => {
  if (guestBlocked("add-folder-btn")) return;
  const path = $("folder-input").value.trim();
  if (!path) return;
  try {
    toast("Scanning folder…");
    $("add-folder-btn").disabled = true;
    const r = await api("/api/folders", { method: "POST", body: JSON.stringify({ path }) });
    $("folder-input").value = "";
    toast(`Folder indexed: +${r.added} docs`);
    await refreshSources();
  } catch (e) {
    toast(e.message, true);
  } finally {
    $("add-folder-btn").disabled = false;
  }
};

/* ---------- settings ---------- */

// Reads the live pipeline config for the one value the UI needs outside the
// Settings modal: the retrieval threshold marked on the per-answer meter. Read
// rather than assumed, because config is hot-reloaded from the DB on every
// request and a hardcoded copy would drift the moment anyone tunes it.
//
// This used to also paint the keyword-fusion switch. Fusion is now simply on
// (config.yaml explains why), so there is no switch to paint.
async function refreshLiveConfig() {
  try {
    const cfg = await api("/api/eval/config");
    state.simThreshold = cfg.similarity_threshold ?? 0;
  } catch (e) {
    console.error("live config read failed:", e);
  }
}

// Deep search: a property of the NEXT QUESTION, held in the page and sent with
// the ask request. Not persisted anywhere, which is the point — the web toggle
// it replaces wrote config_overrides, a single row shared by the deployment, so
// one visitor flipping "their" switch flipped everyone's retrieval.
//
// It stays on once turned on, because someone who needed it for one question
// usually needs it for the follow-up too, and the button says so at a glance.
// Guests may use it: it reads their own documents and spends no model call.
//
// OFF no longer means "never". The tool is always handed to ask(), which
// reaches for it by itself when ranked retrieval comes up short (pipeline.ask,
// ESCALATION 1 and 2). This switch is the override: hold it down and the scan
// runs on every question, including the ones the ranker was confident about —
// which is exactly when a literal hit it missed goes unnoticed.
$("deep-toggle").onclick = () => {
  state.deepSearch = !state.deepSearch;
  syncToolToggles();
  toast(
    state.deepSearch
      ? "Deep search available — used when ranked search comes up short"
      : "Deep search off — your documents are searched by ranking only",
  );
};

$("web-toggle").onclick = () => {
  state.webSearch = !state.webSearch;
  syncToolToggles();
  toast(
    state.webSearch
      ? "Web search on — used only when your documents come up short, and always labelled"
      : "Web search off — answers come from your documents alone",
  );
};

// Lit means AVAILABLE, not "forced on". The app decides whether to use a tool;
// these switches decide whether it HAS one. That inversion is deliberate: the
// earlier version lit up to mean "run this on every question", which made an
// unlit button look like a disabled feature when it was the normal state.
function syncToolToggles() {
  for (const [id, on] of [
    ["deep-toggle", state.deepSearch],
    ["web-toggle", state.webSearch],
  ]) {
    const btn = $(id);
    if (!btn) continue;
    btn.classList.toggle("on", on);
    btn.setAttribute("aria-pressed", String(on));
  }
  // The web row is hidden rather than disabled when it is not on offer: a
  // guest has no way to make it work, and a dead control they cannot fix is
  // worse than one that is not there. `web_search_available` is the SERVER's
  // answer, so the control can never promise something the route refuses.
  const row = $("web-row");
  if (row) row.hidden = !state.webAvailable;
}

// Delete vector rows whose document is gone, plus any left under a superseded
// chunking/embedding config. Reached only from the "/" palette — it had a
// topbar button, which promised a decision the visitor had no way to make:
// nothing in the UI tells you whether orphans exist, so the usual outcome was
// "0 removed" and the usual reading of that was "this button is broken".
//
// The wording avoids "prune" and "ghost". Both were glossary terms, which is
// the tell: a command whose name needs a definition beside it is named wrong.
async function cleanUpOrphanChunks() {
  try {
    const r = await api("/api/documents/prune", { method: "POST" });
    toast(
      r.removed
        ? `Removed ${r.removed} leftover chunk(s) with no source behind them`
        : "Nothing to clean up — every chunk still belongs to a source",
    );
  } catch (e) {
    toast(e.message, true);
  }
}

/* Tuning parameters modal */
function openSettings() {
  $("settings-overlay").classList.remove("hidden");
  setAdvanced(false);   // presets first, every time
  $("settings-note").classList.add("hidden");
  // Say up front that nothing here will save, rather than letting a guest tune
  // ten fields and meet the refusal at the Save button.
  $("settings-guest-note").classList.toggle("hidden", !state.isGuest);
  setSettingsReadOnly(state.isGuest);
  loadSettingsIntoForm();
}

// "Read-only in the demo tier" (plan §7) means every control, not just Save.
// Leaving the fields editable while the save was refused let a guest tune ten
// numbers that could never take effect — the note at the top of the modal is
// only true if the controls agree with it. The preset cards stay clickable and
// aria-disabled instead, because they can still explain themselves.
function setSettingsReadOnly(readOnly) {
  for (const el of $("advanced-fields").querySelectorAll("input, select")) {
    el.disabled = readOnly;
  }
}

// ---------- presets (IDEA.md §7) ----------
//
// The default view is four named configurations, each stating what it trades
// away; the individual fields stay one click behind "Advanced".
//
// The VALUES are not here. They are served from GET /api/presets
// (ragchat/presets.py), because the eval harness scores those same values via
// `run_eval --preset <id>` — and a preset the benchmark measures has to be the
// preset the UI ships, or the published numbers describe a configuration nobody
// can select. Two copies of the table drift the first time one is tuned.
//
// Presets deliberately do NOT touch the model or provider fields. Those are
// deployment facts rather than tradeoffs — switching embedding provider
// re-points every vector in the store, and 422s outright when that provider has
// no key configured — so a card called "Fast" has no business deciding them.

// How each preset key maps to its control, and how to read it back. The ids are
// hardcoded in app.html either way, so this is the one place that knows both.
const FIELD_BY_KEY = {
  chunk_size: ["set-chunk-size", "int"],
  chunk_overlap: ["set-chunk-overlap", "int"],
  splitter: ["set-splitter", "str"],
  top_k: ["set-top-k", "int"],
  candidate_k: ["set-candidate-k", "int"],
  similarity_threshold: ["set-sim-threshold", "float"],
  reranker: ["set-reranker", "bool"],
  query_rewrite: ["set-query-rewrite", "bool"],
  temperature: ["set-temperature", "float"],
};

function readField(key) {
  const spec = FIELD_BY_KEY[key];
  if (!spec) return undefined;
  const raw = $(spec[0])?.value;
  if (raw == null) return undefined;
  if (spec[1] === "int") return parseInt(raw, 10);
  if (spec[1] === "float") return parseFloat(raw);
  if (spec[1] === "bool") return raw === "true";
  return raw;
}

function writeField(key, value) {
  const spec = FIELD_BY_KEY[key];
  const el = spec && $(spec[0]);
  if (!el) return;
  el.value = spec[1] === "bool" ? String(value) : value;
}

// Filled from GET /api/presets when the modal opens.
let presets = [];
let presetKeys = [];
// Local default, deliberately: this list only drives the "you should re-index"
// note, and losing the note because a fetch failed would be worse than the small
// duplication. The server value replaces it when the fetch succeeds.
let presetIndexKeys = ["chunk_size", "chunk_overlap", "splitter"];

async function loadPresets() {
  try {
    const r = await api("/api/presets");
    presets = Array.isArray(r.presets) ? r.presets : [];
    presetKeys = Array.isArray(r.keys) && r.keys.length ? r.keys : presetKeys;
    presetIndexKeys = Array.isArray(r.index_keys) && r.index_keys.length
      ? r.index_keys
      : presetIndexKeys;
  } catch (e) {
    // The advanced fields still work; only the shortcut is missing. Saying so
    // beats rendering an empty block that looks like a layout bug.
    presets = [];
    console.error("presets unavailable:", e);
  }
}

// Config as last read from (or written to) the server. Presets compare their
// chunking against this to decide whether to warn about a re-index.
let loadedConfig = null;

// Floats arrive as 0.0 from JSON and as "0" from a number input, so compare
// numerically with a tolerance and everything else by string.
function sameSetting(a, b) {
  if (typeof a === "number" || typeof b === "number") {
    return Math.abs(Number(a) - Number(b)) < 1e-9;
  }
  return String(a) === String(b);
}

function readPresetFieldsFromForm() {
  const out = {};
  for (const k of presetKeys) out[k] = readField(k);
  return out;
}

function matchPreset(vals) {
  return presets.find((p) => presetKeys.every((k) => sameSetting(p.values[k], vals[k]))) || null;
}

function presetNeedsReindex(preset) {
  if (!loadedConfig) return false;
  return presetIndexKeys.some((k) => !sameSetting(preset.values[k], loadedConfig[k]));
}

function renderPresets() {
  const list = $("preset-list");
  if (!list) return;
  if (!presets.length) {
    // Honest about the gap rather than rendering an empty block that reads as a
    // layout bug. Advanced still has every field.
    list.innerHTML = `<p class="preset-none muted small">Presets could not be
      loaded — the fields under Advanced still work.</p>`;
    return;
  }
  list.innerHTML = presets.map((p) => {
    const badge = presetNeedsReindex(p)
      ? `<span class="index-badge">needs re-index</span>`
      : "";
    // aria-disabled rather than `disabled` for guests, for the same reason as
    // the topbar controls: a disabled button dispatches no click, so the
    // visitor would get silence where applyPreset() gives them a reason.
    return `<button class="preset${state.isGuest ? " guest-locked" : ""}" type="button"
        data-preset="${p.id}" aria-pressed="false"${state.isGuest ? ' aria-disabled="true"' : ""}>
        <span class="preset-name">${escapeHtml(p.name)}${badge}</span>
        <span class="preset-desc">${escapeHtml(p.desc)}</span>
      </button>`;
  }).join("");
  for (const btn of list.querySelectorAll(".preset")) {
    btn.onclick = () => applyPreset(btn.dataset.preset);
  }
  updatePresetSelection();
}

// Fills the form; it does NOT save. Saving is one deliberate press of the same
// button every other change goes through, so a preset cannot re-point the
// pipeline (or trigger a re-index) on a stray click.
function applyPreset(id) {
  // The refusal belongs on the whole modal, not on Save alone — a guest filling
  // the form and then being told no has been misled by the intervening step.
  if (guestBlocked("settings-save")) return;
  const preset = presets.find((p) => p.id === id);
  if (!preset) return;
  for (const [k, v] of Object.entries(preset.values)) writeField(k, v);
  updatePresetSelection();
  const note = $("settings-note");
  note.textContent = presetNeedsReindex(preset)
    ? `${preset.name} filled in — press Save to apply. Its chunking differs from what is indexed, so sources will need a re-index.`
    : `${preset.name} filled in — press Save to apply.`;
  note.classList.remove("hidden");
}

// Reflects the form back onto the cards, so hand-editing any advanced field
// flips the readout to "Custom" instead of leaving a preset falsely lit.
function updatePresetSelection() {
  const active = matchPreset(readPresetFieldsFromForm());
  const label = $("preset-current");
  if (label) label.textContent = active ? active.name : "Custom";
  for (const btn of document.querySelectorAll(".preset")) {
    const on = !!active && btn.dataset.preset === active.id;
    btn.classList.toggle("is-active", on);
    btn.setAttribute("aria-pressed", String(on));
  }
}

// Advanced starts FOLDED every time the dialog opens.
//
// It used to persist across sessions, on the reasoning that this is a tuning
// app and collapsing it each time would make the fields feel hidden rather
// than folded. Reversed deliberately: the presets are the whole point of the
// dialog, and someone who opened the fields once to look at them then met a
// wall of fifteen inputs on every visit afterwards, with the four cards that
// answer the question pushed off the top. Anyone who wants the fields is one
// click away and that click is labelled.
function setAdvanced(open) {
  $("advanced-fields").classList.toggle("hidden", !open);
  $("advanced-toggle").setAttribute("aria-expanded", String(open));
}

$("advanced-toggle").onclick = () => {
  setAdvanced($("advanced-fields").classList.contains("hidden"));
};

setAdvanced(false);

$("settings-reset").onclick = async () => {
  if (guestBlocked("settings-save")) return;
  try {
    const r = await api("/api/eval/config/reset", { method: "POST" });
    // Says which of the two things happened. "Reset" when nothing was stored
    // would be a lie, and the reader would go looking for a change that never
    // needed to happen.
    toast(
      r.reset
        ? "Saved settings discarded — the app is using its shipped defaults again"
        : "Nothing was overridden; the shipped defaults were already live",
    );
    await refreshLiveConfig();
    loadSettingsIntoForm();
  } catch (e) {
    toast(e.message, true);
  }
};

// One delegated listener rather than fifteen: any edit inside the advanced
// fields re-derives which preset (if any) the form now describes.
$("advanced-fields").addEventListener("input", updatePresetSelection);
$("advanced-fields").addEventListener("change", updatePresetSelection);

// Value the embedding model had when the settings form was opened; used to
// decide whether saving needs a re-index prompt.
let loadedEmbeddingModel = null;
let loadedEmbeddingProvider = null;
// Last-known OpenRouter key presence (from /api/eval/config), so the provider
// dropdown change handler can refresh the warning without re-fetching config.
let openrouterConfigured = false;

function fillModelSelect(id, models, current) {
  const sel = $(id);
  const list = Array.isArray(models) ? models.filter(Boolean) : [];
  // Keep a hand-edited model that isn't in the catalog selectable.
  const all = current && !list.includes(current) ? [...list, current] : list;
  sel.innerHTML = all
    .map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(modelLabel(m))}</option>`)
    .join("");
  if (current) sel.value = current;
}

// Replace the embedding-model dropdown with exactly the models the given
// provider serves. Called on open AND on every provider change, so the list is
// always scoped to one provider (Gemini has a single embedder; the rest are
// OpenRouter's). If the previously-selected model doesn't belong to the new
// provider we fall back to that provider's default rather than leaving a stray
// cross-provider entry in the list.
async function refreshEmbeddingModels(provider, preferred) {
  const sel = $("set-embedding-model");
  if (!sel) return;
  let list;
  try {
    const m = await api(`/api/models?provider=${encodeURIComponent(provider)}`);
    list = m.embedding || [];
  } catch (e) {
    list = [];
  }
  if (!list.length) list = [defaultEmbedModelFor(provider)];
  // Trust `preferred` whenever the caller supplies it: it is passed ONLY when
  // the model is already known to belong to this provider (initial load, or
  // switching back to the provider it was saved under). On a genuine provider
  // switch the caller passes null, so the default still wins.
  //
  // It must be kept even when the allowlist lacks that exact spelling. A saved
  // id can be a legacy bare form (`qwen3-embedding-8b`) of a listed one
  // (`qwen/qwen3-embedding-8b`) — OpenRouter serves both. Requiring an exact
  // list match silently swapped the dropdown to the provider default, so
  // opening Settings and pressing Save WITHOUT TOUCHING the embedding fields
  // rewrote the saved model, changed the fingerprint, and invalidated the whole
  // index. fillModelSelect appends an unlisted current value, so it stays
  // visible and selected.
  const want = preferred || defaultEmbedModelFor(provider);
  fillModelSelect("set-embedding-model", list, want);
}

async function loadSettingsIntoForm() {
  try {
    const [cfg, models] = await Promise.all([
      api("/api/eval/config"),
      api("/api/models"),
      loadPresets(),
    ]);
    state.models = models;
    $("set-chunk-size").value = cfg.chunk_size;
    $("set-chunk-overlap").value = cfg.chunk_overlap;
    $("set-splitter").value = cfg.splitter;
    $("set-top-k").value = cfg.top_k;
    $("set-candidate-k").value = cfg.candidate_k;
    $("set-sim-threshold").value = cfg.similarity_threshold;
    $("set-reranker").value = String(cfg.reranker);
    $("set-query-rewrite").value = String(cfg.query_rewrite);
    fillModelSelect("set-llm-model", models.chat, cfg.llm_model);
    $("set-temperature").value = cfg.temperature;
    const savedProvider = (cfg.embedding_provider || "gemini").toLowerCase();
    $("set-embedding-provider").value = savedProvider;
    loadedEmbeddingModel = cfg.embedding_model;
    loadedEmbeddingProvider = savedProvider;
    openrouterConfigured = !!cfg.openrouter_configured;
    // Warn if the user is on (or picks) OpenRouter without a key.
    updateProviderWarnings(cfg.openrouter_configured);
    // Populate the embedding-model list for the SAVED provider only. There is
    // no unscoped first fetch any more — that was what briefly showed every
    // provider's models at once.
    await refreshEmbeddingModels(savedProvider, cfg.embedding_model);
    // Re-scope the list whenever the provider changes.
    const epSel = $("set-embedding-provider");
    if (epSel) {
      epSel.onchange = async () => {
        const p = epSel.value;
        // Only keep the current model if we're switching back to the provider
        // it belongs to; otherwise take that provider's default.
        const keep = p === loadedEmbeddingProvider ? loadedEmbeddingModel : null;
        await refreshEmbeddingModels(p, keep);
        updateProviderWarnings(openrouterConfigured);
      };
    }
    // The reranker switch now drives the OpenRouter-key warning, since reranking
    // is what consumes that key when embeddings are on Gemini.
    const rkSel = $("set-reranker");
    if (rkSel) {
      rkSel.onchange = () => updateProviderWarnings(openrouterConfigured);
    }
    // Presets are drawn AFTER the form is populated: each card's re-index badge
    // is computed against what is actually saved, so it needs the live config.
    loadedConfig = cfg;
    renderPresets();
  } catch (e) {
    toast(e.message, true);
  }
}

function defaultEmbedModelFor(provider) {
  return provider === "openrouter" ? "qwen/qwen3-embedding-8b" : "models/gemini-embedding-001";
}

function updateProviderWarnings(configured) {
  const warn = $("provider-warning");
  const ep = $("set-embedding-provider")?.value;
  // Reranking is Cohere via OpenRouter unconditionally now, so the key is needed
  // whenever the reranker is ON — not only when a provider dropdown says so. The
  // dropdown this used to read is gone; leaving the old condition here would have
  // stopped warning the one person who most needs it: reranker on, no key, every
  // answer silently falling back to vector order.
  const reranking = $("set-reranker")?.value === "true";
  if (configured || (ep !== "openrouter" && !reranking)) {
    if (warn) warn.classList.add("hidden");
    return;
  }
  if (warn) {
    warn.textContent = reranking && ep !== "openrouter"
      ? "Reranking needs OPENROUTER_API_KEY — without it, results keep their vector order."
      : "OpenRouter selected but no OPENROUTER_API_KEY found in .env — add it and restart the backend.";
    warn.classList.remove("hidden");
  }
}

function closeSettings() {
  $("settings-overlay").classList.add("hidden");
  $("settings-note").classList.add("hidden");
}

$("settings-btn").onclick = openSettings;

// ---------- theme ----------

// The initial attribute is set by the inline <head> script in index.html, not
// here — by the time app.js runs the page has already painted, so applying it
// at this point would flash. This only handles the toggle afterwards.
const THEME_KEY = "ragchat-theme";

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const btn = $("theme-toggle");
  const dark = theme === "dark";
  btn.textContent = dark ? "☾" : "☀";
  btn.setAttribute("aria-pressed", String(!dark));
  btn.title = dark ? "Switch to the light theme" : "Switch to the dark theme";
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch (e) {
    // Storage disabled: the toggle still works for this session, it just
    // won't be remembered. Not worth surfacing to the user.
  }
}

applyTheme(document.documentElement.getAttribute("data-theme") || "dark");
$("theme-toggle").onclick = () =>
  applyTheme(
    document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark",
  );
$("settings-close").onclick = closeSettings;
$("settings-overlay").onclick = (e) => {
  if (e.target === $("settings-overlay")) closeSettings();
};

$("settings-save").onclick = async () => {
  if (guestBlocked("settings-save")) return;
  try {
    const body = {
      chunk_size: parseInt($("set-chunk-size").value),
      chunk_overlap: parseInt($("set-chunk-overlap").value),
      splitter: $("set-splitter").value,
      top_k: parseInt($("set-top-k").value),
      candidate_k: parseInt($("set-candidate-k").value),
      similarity_threshold: parseFloat($("set-sim-threshold").value),
      reranker: $("set-reranker").value === "true",
        query_rewrite: $("set-query-rewrite").value === "true",
      llm_model: $("set-llm-model").value,
      temperature: parseFloat($("set-temperature").value),
      embedding_model: $("set-embedding-model").value,
      embedding_provider: $("set-embedding-provider").value,
    };
    const res = await api("/api/eval/config", {
      method: "PUT",
      body: JSON.stringify(body),
    });
    // Re-evaluate the OpenRouter-key warning from the live server response
    // (it reflects the actual env state, e.g. after a Vercel redeploy).
    // PUT /api/eval/config nests this under `config` — reading it off the top
    // level yielded undefined, so saving any change while OpenRouter was
    // selected raised "no OPENROUTER_API_KEY found" even with a valid key.
    updateProviderWarnings(res.config?.openrouter_configured);
    const embeddingChanged =
      body.embedding_model !== loadedEmbeddingModel ||
      body.embedding_provider !== loadedEmbeddingProvider;
    // The server reports needs_reindex from WHICH KEYS WERE SENT, and this form
    // always sends all of them — so it is true on every save, including one that
    // changed nothing index-affecting. Compare against the config we loaded
    // before believing it: a warning that fires every time teaches the user to
    // ignore the one time it matters.
    const chunkingChanged = !loadedConfig || presetIndexKeys.some(
      (k) => !sameSetting(body[k], loadedConfig[k])
    );
    if (res.needs_reindex && (embeddingChanged || chunkingChanged)) {
      $("settings-note").classList.remove("hidden");
      if (embeddingChanged) {
        // The index is unusable until re-embedded with the new model — offer
        // to do it right away rather than only leaving a note.
        $("settings-note").textContent =
          "Embedding model or provider changed — sources must be re-indexed before asking.";
        if (confirm("Embedding changed. Re-index all sources now?")) {
          await reindexAll();
        }
      } else {
        $("settings-note").textContent =
          "Index-affecting keys changed. You should re-index your sources.";
      }
    } else {
      $("settings-note").classList.add("hidden");
    }
    // What was persisted becomes the new baseline — for the preset re-index
    // badges and for the next save's comparison alike. Without this, saving the
    // same form twice re-announced (and re-prompted for) a re-index that had
    // already been done.
    if (res.config) {
      loadedConfig = res.config;
      loadedEmbeddingModel = res.config.embedding_model;
      loadedEmbeddingProvider = String(res.config.embedding_provider || "gemini").toLowerCase();
      renderPresets();
    }
    await refreshLiveConfig();
    toast("Settings saved — next ask will use the new config.");
  } catch (e) {
    toast("Save failed: " + e.message, true);
  }
};

// ---------- persistent job status (IDEA.md §5) ----------
//
// Re-index all and folder rescans announced themselves in a toast that vanished
// after four seconds and then reported nothing until they finished — which looks
// exactly like having failed. This line stays until the work is actually done.
//
// `sticky` marks a result worth leaving on screen; a plain progress message is
// replaced by the next refresh, and cleared when nothing is running.
let jobSticky = "";

function setJobStatus(text, { busy = false, sticky = false } = {}) {
  const el = $("job-status");
  if (!el) return;
  jobSticky = sticky ? text : "";
  el.textContent = text || "";
  el.classList.toggle("is-busy", !!busy && !!text);
  el.classList.toggle("hidden", !text);
}

// Derives the line from the documents themselves rather than from what we
// started, so a job resumed after a reload still reports.
function paintJobStatus(docs) {
  const working = docs.filter((d) => d.status === "indexing" || d.status === "pending");
  if (working.length) {
    const chunks = working.reduce((n, d) => n + (d.indexed_chunks || 0), 0);
    const total = working.reduce((n, d) => n + (d.n_chunks || 0), 0);
    const of = total ? ` · ${chunks}/${total} chunks` : "";
    setJobStatus(`Indexing ${working.length} source${working.length === 1 ? "" : "s"}${of}`, { busy: true });
    return;
  }
  // Nothing running: keep a result the user has not seen the end of yet, and
  // otherwise clear rather than leave a stale "indexing…" behind.
  setJobStatus(jobSticky, { sticky: !!jobSticky });
}

async function reindexAll() {
  if (guestBlocked("reindex-btn")) return;
  setJobStatus("Queueing a re-index of every source…", { busy: true });
  $("reindex-btn").disabled = true;
  try {
    // Queues rather than re-embeds: reindex returns at once and refreshSources
    // drives each document's slices, so a whole-workspace re-index cannot
    // overrun the function budget and shows per-document progress.
    const r = await api("/api/documents/reindex", { method: "POST" });
    const bad = r.unreadable ? `, ${r.unreadable} unreadable` : "";
    setJobStatus(`Re-indexing ${r.queued ?? r.reindexed} sources${bad}`, { busy: true });
    // refreshSources takes it from here: it drives each document's slices and
    // repaints this line from their real progress until they are all ready.
    await refreshSources();
  } catch (e) {
    setJobStatus(`Re-index failed: ${e.message}`, { sticky: true });
    toast(e.message, true);
  } finally {
    $("reindex-btn").disabled = false;
  }
}

$("reindex-btn").onclick = reindexAll;

// Empty the workspace: every document, every folder, every vector.
//
// The confirm names the count and says what SURVIVES, because "delete all" in
// a pane called Sources could reasonably be read as taking the chats with it.
// It does not: conversations are not embedded, they cost no vector storage, and
// an answer keeps its citations inline, so old chats stay readable after the
// documents behind them are gone.
async function deleteAllDocuments() {
  if (guestBlocked("delete-all-btn")) return;
  // Counted from what is on screen: the document list is rendered straight
  // from the fetch and never held in state, and a confirm for an
  // irreversible action should name the number it is about to destroy.
  const n = document.querySelectorAll('#doc-list .source-item').length;
  if (!n) {
    toast("Nothing to delete — this workspace has no documents.");
    return;
  }
  const ok = confirm(
    `Delete all ${n} document${n === 1 ? "" : "s"} and everything embedded from ` +
    `them?

This cannot be undone. Your chats are kept.`,
  );
  if (!ok) return;

  setJobStatus("Deleting every document and its vectors…", { busy: true });
  $("delete-all-btn").disabled = true;
  try {
    const r = await api("/api/documents", { method: "DELETE" });
    setJobStatus(
      `Deleted ${r.documents} document${r.documents === 1 ? "" : "s"}` +
      (r.folders ? ` and ${r.folders} folder source${r.folders === 1 ? "" : "s"}` : ""),
      { sticky: true },
    );
    await refreshSources();
  } catch (e) {
    setJobStatus(`Delete failed: ${e.message}`, { sticky: true });
    toast(e.message, true);
  } finally {
    $("delete-all-btn").disabled = false;
  }
}

$("delete-all-btn").onclick = deleteAllDocuments;

// ---------- chat ----------

// Local status overrides while a request is in flight; the backend status is
// the source of truth after each refresh.
const chatStatusOverride = new Map(); // chatId -> "pending" | "done"

// Three questions the seeded demo corpus can actually answer (plan §4). They
// exist because "ask anything about them" is useless advice about two documents
// the visitor has never read: the fastest way to show grounding is to hand them
// a question whose answer is a specific line in a specific file.
//
// Only ever shown when those documents are the ones present — see state.demoDocs.
// One per document, plus one that needs several facts from a single passage.
// Keyed by the document each question is answerable FROM, using the titles the
// backend stores (guests.DEMO_CORPUS_FILES). Per-document rather than one flat
// list because the two files are seeded independently: a visitor whose first
// paint has landed only one of them should still get the questions that file can
// answer, not a chipless empty state.
const DEMO_QUESTIONS = {
  "helios_energy_handbook.md": [
    "What warranty comes with the SunPak 5 battery?",
  ],
  "meridian_coffee_ops.md": [
    "What time do Meridian's stores open at the weekend?",
    "What does the espresso machine opening checklist require?",
  ],
};

function demoChipsHtml() {
  const chips = state.demoQuestions
    .map((q) => `<button class="ask-chip" type="button" data-q="${escapeHtml(q)}">${escapeHtml(q)}</button>`)
    .join("");
  return chips ? `<div class="ask-chips">${chips}</div>` : "";
}

// The empty conversation used to hard-code "Add sources on the left" — advice
// that is wrong for the visitor who most needs it, since a guest arrives with
// the demo corpus already loaded and nothing to add. Ask about what is there.
function chatEmptyState() {
  const n = state.sourceCount;
  // The "two sample documents" line is only true when they are the ONLY sources;
  // once the visitor adds their own, the count is what tells the truth. The chips
  // survive either way, because the files they ask about are still there.
  return n
    ? emptyState(
        "◈",
        "Ask a question to see results",
        state.demoDocs && n === Object.keys(DEMO_QUESTIONS).length
          ? "Two sample documents are already loaded. Try one of these — every answer cites the exact passage it came from."
          : `You have ${n} source${n === 1 ? "" : "s"} ready. Ask anything about ${
              n === 1 ? "it" : "them"
            } — every answer cites the exact passage it came from.`,
        demoChipsHtml() + beatsHtml(2)
      )
    : emptyState(
        "◈",
        "Add a source to begin",
        "Drop a file into Sources on the left, or paste a URL. Answers are grounded in your documents and cite them.",
        beatsHtml(2)
      );
}

function renderChats() {
  const list = $("chat-list");
  const count = $("chat-count");
  if (count) count.textContent = String(state.chats.length);
  paintTabCounts();
  const titleEl = $("chat-title");
  if (titleEl) {
    const open = state.chats.find((c) => c.id === state.currentChatId);
    titleEl.textContent = open ? open.title : "";
    titleEl.title = open ? open.title : "";
  }
  list.innerHTML = "";
  for (const c of state.chats) {
    const item = document.createElement("div");
    item.className = "chat-item" + (c.id === state.currentChatId ? " active" : "");
    const status = chatStatusOverride.get(c.id) || c.status || "done";
    item.innerHTML = `
      <span class="status-dot ${status}" title="${status === "pending" ? "Waiting for an answer" : "Answered"}"></span>
      <span class="title" title="${escapeHtml(c.title)}">${escapeHtml(c.title)}</span>
      <button class="icon-btn chat-delete" title="Delete conversation">✕</button>`;
    item.onclick = () => { if (c.id !== state.currentChatId) openChat(c.id); };
    item.querySelector(".chat-delete").onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete “${c.title}”?`)) return;
      try {
        // DELETE returns 204/200; only mutate local state on success.
        await api(`/api/chats/${c.id}`, { method: "DELETE" });
        // Remove the chat from the in-memory list so it disappears immediately
        // (the cached state.chats is the source of truth for renderChats).
        state.chats = state.chats.filter((x) => x.id !== c.id);
        chatStatusOverride.delete(c.id);
        const wasCurrent = state.currentChatId === c.id;
        if (wasCurrent) state.currentChatId = null;
        if (wasCurrent && state.chats.length) {
          // Jump to the newest remaining chat instead of the empty state.
          await openChat(state.chats[0].id);
        } else {
          renderChats();
        }
        if (!state.chats.length) {
          state.currentChatId = null;
          renderChats();
          $("messages").innerHTML = chatEmptyState();
        }
        toast("Conversation deleted");
      } catch (err) { toast(err.message, true); }
    };
    list.appendChild(item);
  }
}

async function refreshChats(selectId = null) {
  state.chats = await api("/api/chats");
  renderChats();
  const target = selectId || (state.chats[0] && state.chats[0].id);
  if (target) {
    await openChat(target);
  } else {
    state.currentChatId = null;
    renderChats();
    $("messages").innerHTML = chatEmptyState();
  }
}

$("new-chat-btn").onclick = async () => {
  try {
    const chat = await api("/api/chats", { method: "POST" });
    await refreshChats(chat.id);
  } catch (e) { toast(e.message, true); }
};

async function openChat(chatId) {
  state.currentChatId = chatId;
  renderChats();
  const chat = await api(`/api/chats/${chatId}`);
  const box = $("messages");
  box.innerHTML = "";
  if (chat.messages.length === 0) {
    // The same empty state as a fresh workspace, rather than a second variant of
    // it: an empty conversation is exactly where the suggested questions and the
    // loop guide are wanted, and this branch is what a new chat lands on.
    box.innerHTML = chatEmptyState();
    return;
  }
  // A conversation with messages in it is proof that beat 2 happened, including
  // on a later visit where the loop state was lost with the browser storage.
  markBeat("asked");
  for (const m of chat.messages) {
    const el = appendMessage(m.role, m.content, m.citations || [], false,
      m.eval_line || "", m.eval_data || null, m.id);
    // Restore the "these bars are yours" marker across a re-render. state holds
    // the truth; the class is only how it is drawn.
    if (m.id && m.id === state.answerEvalId) {
      el?.querySelector(".eval-chip")?.classList.add("is-shown");
    }
    // Reloaded into a thread whose last answer was still being graded: ask for
    // the verdict again rather than leaving it spinning. Grading is idempotent,
    // so this costs nothing when it has already finished.
    if (m.eval_data?.pending && m.id) fetchGrade(chatId, m.id, el);
  }
  box.scrollTop = box.scrollHeight;
}

// Render [1] markers as clickable spans; citations list becomes chips below.
function renderAssistantContent(el, content, citations) {
  // Models write "[2, 3, 4]" as readily as "[2]", and the old pattern read only
  // the second — so an answer resting on several sources at once had
  // references in the prose that could not be clicked, on an app whose whole
  // promise is that you can click the citation. Each number becomes its own
  // marker: "[2, 3, 4]" renders as [2][3][4], three separate targets, because a
  // single span over three numbers has no one passage to open.
  //
  // Must stay in step with pipeline.cited_in, which decides which sources get a
  // chip. If one reads a form the other does not, the reader sees a reference
  // with no chip or a chip nothing points at.
  const withMarkers = escapeHtml(content).replace(
    /\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]/g,
    (m, group) =>
      group
        .split(",")
        .map((n) => n.trim())
        .filter(Boolean)
        .map((n) => `<span class="cite-marker" data-cite="${n}">[${n}]</span>`)
        .join(""),
  );
  el.innerHTML = withMarkers;
  if (citations.length) {
    const wrap = document.createElement("div");
    wrap.className = "citations";
    const label = document.createElement("span");
    label.className = "citations-label";
    label.textContent = "Sources";
    wrap.appendChild(label);
    for (const c of citations) {
      const chip = document.createElement("button");
      chip.className = "cite-chip";
      chip.type = "button";
      // A web source is the only citation here that is not one of the user's
      // own documents. It gets a badge for the same reason the prompt gets a
      // marker: an answer that blends the two without saying which is which
      // breaks the one promise this app makes.
      if (c.is_web) chip.classList.add("is-web");
      chip.title = c.is_web
        ? `From the web — ${c.ref || c.title}`
        : `Show the passage from “${c.title}”`;
      chip.innerHTML =
        `<span class="cite-num">${c.number}</span>` +
        (c.is_web ? `<span class="cite-web">web</span>` : "") +
        `<span class="cite-name">${escapeHtml(c.title)}</span>`;
      chip.onclick = () => showExcerpt(c);
      wrap.appendChild(chip);
    }
    el.appendChild(wrap);
  }
  el.querySelectorAll(".cite-marker").forEach((span) => {
    span.onclick = () => {
      const n = parseInt(span.dataset.cite, 10);
      const c = citations.find((x) => x.number === n);
      if (c) showExcerpt(c);
    };
  });
}

// One word for the whole quality verdict (IDEA.md §6).
//
// UNGRADED IS NOT FAILURE and must never render as one. A judge that 404s or
// times out is a broken grader, not a bad answer — `faithful`/`relevant` are
// nullable for exactly this reason, and showing "Weak" there would be a
// confident false claim about the user's own documents. Null is checked BEFORE
// falsity so a partial grading can never be read as a verdict.
function evalVerdict(evalData) {
  const { faithful, relevant } = evalData;
  // Waiting on the judges is NOT the same as the judges having failed. The
  // answer is delivered before grading starts now, so every answer passes
  // through this state for a few seconds; showing "Ungraded" there would mean
  // the app reports a broken grader on every single question.
  if (evalData.pending) return { state: "pending", word: "Grading…" };
  if (faithful == null && relevant == null) return { state: "ungraded", word: "Ungraded" };
  if (faithful === false || relevant === false) return { state: "weak", word: "Weak" };
  if (faithful == null || relevant == null) return { state: "ungraded", word: "Partly graded" };
  return { state: "grounded", word: "Grounded" };
}

// Build the per-answer quality readout: one ~20px composite indicator that
// expands to the full detail on click.
//
// It replaces four always-visible labelled rows with glosses, which carried
// roughly as much visual weight as the answer itself — the exact inversion the
// two-register rule exists to prevent (§1). The detail is not deleted, only
// folded: it is genuinely useful, just not on every answer by default.
function buildEvalBlock(evalData, evalLine) {
  // One line under the answer, and the numbers themselves in the Evaluation
  // pane.
  //
  // Every answer used to carry its own expandable table of scores, and the pane
  // separately showed benchmark averages. Two sets of numbers, no relationship
  // drawn between them, and the reader left to hold both in their head. The
  // scores now sit on the benchmark's own bars, where "is this answer normal?"
  // is a glance rather than an inference.
  //
  // What stays here is a verdict and a way back: clicking any answer's chip
  // puts THAT answer's readings on the bars, so scrolling up through a
  // conversation still works. Without it the pane would only ever describe the
  // most recent answer and older ones would lose their scores entirely.
  const wrap = document.createElement("div");
  wrap.className = "eval-block";

  // Say when the app went and looked again on its own.
  //
  // An answer that exists because the machine decided to try a second tool is
  // not the same object as one that came straight back, and the reader has no
  // other way to tell. This app's whole argument is that you show what you
  // did, and a silent self-correction is the one place that would be easiest
  // to skip and worst to skip.
  const why = evalData && typeof evalData === "object" ? evalData.escalated : null;
  if (why) {
    const note = document.createElement("div");
    note.className = "escalation-note";
    const tools = evalData.tools_used || [];
    const did = [];
    if (tools.includes("deep")) did.push("every document was scanned word for word");
    if (tools.includes("web")) did.push("the web was searched");
    const what = did.length ? did.join(", then ") : "the app looked again";
    note.innerHTML =
      `<span class="escalation-mark" aria-hidden="true">⤷</span>` +
      `<span>${
        why === "weak_retrieval"
          ? `Ranked search found nothing close enough, so ${what}.`
          : `The first pass came up empty, so ${what} and the answer rewritten.`
      }</span>`;
    wrap.appendChild(note);
  }

  if (evalData && typeof evalData === "object") {
    const { state: verdict, word } = evalVerdict(evalData);
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "eval-chip";
    chip.title = "Show this answer's scores against the benchmark";

    const nudge = currentBeat() === 4
      ? `<span class="eval-nudge">what is this?</span>`
      : "";
    const sim = evalData.top_sim;
    const simText = sim != null
      ? `<span class="eval-meter-val">${sim.toFixed(2)}</span>`
      : "";
    chip.innerHTML = `<span class="eval-dot" data-state="${verdict}" aria-hidden="true"></span>
      <span class="eval-state">${escapeHtml(word)}</span>${simText}${nudge}
      <span class="eval-chip-cta">compare</span>`;

    chip.onclick = () => {
      showAnswerOnScorecard(evalData, chip);
      markBeat("readout");                       // beat 4 — the loop is complete
      chip.querySelector(".eval-nudge")?.remove();
    };
    wrap.appendChild(chip);
    return wrap;
  }
  // Fallback: the terse line as-is, for messages stored before eval data existed.
  if (evalLine) {
    const row = document.createElement("div");
    row.className = "eval-row";
    row.innerHTML = `<span class="eval-label">Eval</span>` +
      `<span class="eval-gloss">${escapeHtml(evalLine)}</span>`;
    wrap.appendChild(row);
  }
  return wrap;
}

// Put one answer's readings on the benchmark bars and mark which chip they came
// from, so the pane is never ambiguous about whose numbers are on screen.
function showAnswerOnScorecard(evalData, chip, messageId = null) {
  state.answerEval = evalData;
  // WHICH answer owns the bars, held in state rather than only as a class on a
  // chip. The class alone was the bug behind "grading…" sticking forever: the
  // chat list refreshes right after an answer lands, which re-renders the whole
  // thread and takes the marker with it — so eleven seconds later, when the
  // verdicts arrived, nothing believed the bars belonged to that answer and
  // they stayed waiting until a reload. Ownership has to outlive the DOM that
  // displays it.
  state.answerEvalId =
    messageId || chip?.closest(".msg")?.dataset.messageId || null;
  // Exactly one chip is marked, so it is never ambiguous which answer the bars
  // belong to once a conversation has several.
  for (const c of document.querySelectorAll(".eval-chip.is-shown")) {
    c.classList.remove("is-shown");
  }
  chip?.classList.add("is-shown");
  const data = state.evalData;
  if (data) renderScorecard(data.metrics || {}, data.mode);
  // On a phone the panes are tabs, so the bars this just updated are on a
  // screen the reader cannot see. Bring them to it.
  if (MOBILE.matches) setMobileTab("eval");
}

// Fetch the verdicts for an answer already on screen, and fold them in.
//
// The answer no longer waits for grading — two judge calls cost more than
// writing the answer did — so it lands ungraded and this fills it in a moment
// later. Everything here has to survive the reader not sitting still: they can
// ask another question, switch chats, or close the tab before it returns.
async function fetchGrade(chatId, messageId, msgEl) {
  try {
    const r = await api(`/api/chats/${chatId}/messages/${messageId}/grade`, {
      method: "POST",
    });
    const graded = r && r.eval;
    if (!graded) return;

    // Re-find the message rather than trusting the element we started with:
    // ten seconds is long enough for a chat refresh to have replaced it.
    const el =
      document.querySelector(`.msg[data-message-id="${messageId}"]`) || msgEl;
    if (!el || !el.isConnected) return;

    // Does this answer own the bars? Asked of STATE, not of the DOM. A class on
    // a chip does not survive the thread re-render that follows every answer,
    // and reading it here is what left the bars stuck on "grading…".
    const ownsBars = state.answerEvalId === messageId;

    // Rebuild this answer's chip in place. A :last-child lookup would attach
    // the verdict to whatever answer happens to be last by the time the judges
    // reply, which on a slow provider is often a different one.
    const block = el.querySelector(".eval-block");
    if (block) block.replaceWith(buildEvalBlock(graded, r.eval_line || ""));

    // Move the bars only if they were already showing this answer. If the
    // reader has since asked something else, updating the pane underneath them
    // would label a newer answer with an older answer's verdict.
    if (ownsBars) {
      showAnswerOnScorecard(graded, el.querySelector(".eval-chip"), messageId);
    }
  } catch (err) {
    // A failed grade leaves the answer ungraded, which is a state the UI
    // already draws honestly. It must never take the answer down with it.
    console.warn("grading failed", err);
  }
}


function appendMessage(role, content, citations = [], isPending = false, evalLine = "", evalData = null, messageId = null) {
  const box = $("messages");
  const hint = box.querySelector(".empty-state");
  if (hint) hint.remove();

  const el = document.createElement("div");
  el.className = `msg ${role}${isPending ? " pending" : ""}`;
  // Stamped so a grade arriving later can find this answer again. Holding the
  // element itself is not enough: a chat-list refresh re-renders the thread,
  // and the held node is then detached — updates land on a DOM nobody is
  // looking at, which is exactly how the chip stayed on "Grading…" while the
  // bars behind it filled in.
  if (messageId) el.dataset.messageId = messageId;
  if (role === "assistant") {
    if (isPending) {
      el.textContent = content;
    } else {
      if (content.startsWith("I couldn't find this in your documents")) {
        el.classList.add("not-found");
      }
      renderAssistantContent(el, content, citations);
      if (evalData || evalLine) {
        el.appendChild(buildEvalBlock(evalData, evalLine));
      }
    }
  } else {
    el.textContent = content;
  }
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
  return el;
}

$("ask-form").onsubmit = async (e) => {
  e.preventDefault();
  const input = $("question-input");
  const question = input.value.trim();
  if (!question) return;

  if (!state.currentChatId) {
    const chat = await api("/api/chats", { method: "POST" });
    await refreshChats(chat.id);
  }

  input.value = "";
  appendMessage("user", question);
  const pending = appendMessage("assistant", "Thinking…", [], true);
  $("ask-btn").disabled = true;
  chatStatusOverride.set(state.currentChatId, "pending"); // orange dot while we wait
  renderChats();

  try {
    const result = await api(`/api/chats/${state.currentChatId}/ask`, {
      method: "POST",
      body: JSON.stringify({
        question,
        deep_search: state.deepSearch,
        web_search: state.webSearch && state.webAvailable,
      }),
    });
    pending.remove();
    const msg = appendMessage(
      "assistant", result.answer, result.citations, false,
      result.eval_line || "", result.eval || null, result.message_id || null,
    );
    // The newest answer takes the bars without being asked. Watching them move
    // as each answer lands is the comparison; making the reader click first
    // would leave the pane describing an answer they have scrolled past.
    //
    // The chip comes from the element just appended rather than a :last-child
    // lookup, which would break the moment anything else is appended after it.
    if (result.eval) {
      showAnswerOnScorecard(
        result.eval, msg?.querySelector(".eval-chip"), result.message_id || null,
      );
    }
    // Not awaited: the answer is already readable, and the reader is free to
    // type the next question while the judges work.
    if (result.eval?.pending && result.message_id) {
      fetchGrade(state.currentChatId, result.message_id, msg);
    }
    markBeat("asked");                             // beat 2 — an answer exists
    chatStatusOverride.delete(state.currentChatId); // answered -> green dot
    // keep the chat list titles/statuses in sync
    const c = state.chats.find((x) => x.id === state.currentChatId);
    if (c) c.status = "done";
    if (c && c.title === "New chat") {
      await refreshChats(state.currentChatId);
    } else {
      renderChats();
    }
  } catch (err) {
    pending.remove();
    toast(err.message, true);
  } finally {
    $("ask-btn").disabled = false;
  }
};

// "/" on an empty composer opens the palette, the convention every chat box
// has trained people into. Only when it is the WHOLE value: a question may
// legitimately contain a slash, and swallowing that would be worse than having
// no shortcut at all. The character is removed, because it was a trigger rather
// than something the user meant to type.
$("question-input").addEventListener("input", (e) => {
  if (!PALETTE_DESKTOP.matches) return;
  if (e.target.value === "/") {
    e.target.value = "";
    openPalette();
  }
});

$("question-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("ask-form").requestSubmit();
  }
});

// Suggested questions. Delegated because the empty state is re-rendered whenever
// the source list changes, which would strip per-button handlers.
$("messages").addEventListener("click", (e) => {
  const chip = e.target.closest(".ask-chip");
  if (!chip) return;
  $("question-input").value = chip.dataset.q;
  $("ask-form").requestSubmit();
});

// ---------- excerpt pane (bottom) ----------

function showExcerpt(citation) {
  markBeat("cited");   // beat 3 of the loop — checking an answer against its source
  $("excerpt-content").classList.remove("hidden");
  $("excerpt-close").classList.remove("hidden");
  document.querySelector(".excerpt-empty").classList.add("hidden");
  $("excerpt-title").textContent = citation.title;
  $("excerpt-ref").textContent = citation.ref || "";
  $("excerpt-text").textContent = citation.excerpt;
  // On mobile the excerpt is a sheet over the conversation (plan §9), so the
  // passage arrives on top of the answer that cited it rather than requiring a
  // scroll — or, under the tabbed layout, a trip to another tab.
  if (MOBILE.matches) $("excerpt-pane").classList.add("is-sheet");
}

$("excerpt-close").onclick = () => {
  $("excerpt-content").classList.add("hidden");
  $("excerpt-close").classList.add("hidden");
  $("excerpt-title").textContent = "";
  $("excerpt-ref").textContent = "";
  document.querySelector(".excerpt-empty").classList.remove("hidden");
  $("excerpt-pane").classList.remove("is-sheet");   // closes the mobile sheet
};

// ---------- evaluation tab (right) ----------

// What the benchmark measures, in the order a question passes through it.
//
// GROUPED, because nine bars in a flat list invite the wrong reading — that
// they are nine views of one thing. They are not. Retrieval asks whether the
// right passage came back at all; ranking asks whether it came back near the
// TOP; generation asks whether the answer built from it was any good. A number
// can fall in one group and be fine in the others, and knowing which group
// moved is the whole diagnosis.
//
// The groups also make the framework honest, and `from` finishes the job by
// naming where each metric came from. Four are RAGAS's. The not-found rate is
// this app's own — nothing in RAGAS measures refusing correctly — and it says
// RAG-it, because a metric you invented is worth claiming rather than hiding
// among borrowed ones.
//
// Precision@k, MRR, NDCG@k and Hit rate@k carry no label. The last three are
// classic information-retrieval metrics that predate RAGAS by decades and
// belong to nobody in particular. Precision@k is the deliberate one: RAGAS
// does define a context precision, but it is rank-aware and this is plain
// precision@k — close enough to invite the label and not the same thing, which
// is the small overclaim this marking exists to stop making.
//
// Each row leads with what it MEASURES and keeps the formal name underneath.
// "Context Recall 49%" tells a visitor nothing; "Found the right passages"
// tells them what got worse. The formal name is the accented half: it is the
// searchable term, and at 11px mono it is the smallest text in the pane.
const EVAL_GROUPS = {
  retrieval: { label: "Retrieval", sub: "Did the right passage come back?" },
  ranking: { label: "Ranking", sub: "Did it come back near the top?" },
  generation: { label: "Generation", sub: "Was the answer any good?" },
};

const EVAL_TARGETS = {
  context_recall: { group: "retrieval", from: "RAGAS", label: "Found the right passages", sub: "Context Recall", target: 0.80, higher: true },
  precision_at_k: { group: "retrieval", label: "Sent mostly relevant text", sub: "Precision@k", target: 0.70, higher: true },
  mrr: { group: "ranking", label: "Best passage ranked high", sub: "MRR", target: 0.65, higher: true },
  ndcg_at_k: { group: "ranking", label: "Good overall ordering", sub: "NDCG@k", target: 0.70, higher: true },
  hit_rate_at_k: { group: "ranking", label: "Right passage made the cut", sub: "Hit rate@k", target: 0.80, higher: true },
  faithfulness: { group: "generation", from: "RAGAS", label: "Stuck to the sources", sub: "Faithfulness", target: 0.90, higher: true },
  answer_relevancy: { group: "generation", from: "RAGAS", label: "Answered what was asked", sub: "Answer relevancy", target: 0.85, higher: true },
  answer_correctness: { group: "generation", from: "RAGAS", label: "Matched the expected answer", sub: "Answer correctness", target: 0.80, higher: true },
  not_found_rate_unanswerables: { group: "generation", from: "RAG-it", label: "Admitted when it could not answer", sub: "Not-found rate", target: 0.90, higher: true },
};

function fmtPct(v) {
  if (v == null) return "—";
  return `${Math.round(v * 100)}%`;
}

// Which per-answer reading belongs on which benchmark bar.
//
// The Evaluation pane used to show benchmark averages, and every answer
// repeated its own scores underneath itself, and the two never met — so a
// reader had two sets of numbers and no way to relate them. They now share one
// axis: the benchmark is the reference tick, the answer just given is the bar.
//
// Only three of the readings have an honest counterpart. `latency` has none —
// it is a speed, not a quality — so it is reported on its own rather than
// drawn against a bar it has no relationship with. Inventing a fourth pairing
// to make the layout tidy would be the dishonest option.
const LIVE_TO_BENCHMARK = {
  top_sim: "context_recall",
  faithful: "faithfulness",
  relevant: "answer_relevancy",
};

// What the per-answer field means on a 0-1 axis. The judges answer yes/no for
// one answer while the benchmark reports a RATE across 53 questions, so the two
// are not the same statistic and the row says so in words: "this answer" vs
// "benchmark". Drawing a single pass at 100% against a 94% rate is only
// misleading if the labels pretend they are the same measurement.
function liveValue(evalData, field) {
  if (!evalData) return null;
  const raw = evalData[field];
  if (raw == null) return null;
  if (field === "top_sim") return Math.max(0, Math.min(1, raw));
  return raw ? 1 : 0;
}

function liveLabel(evalData, field) {
  const raw = evalData?.[field];
  if (raw == null) return null;
  if (field === "top_sim") return raw.toFixed(2);
  return raw ? "passed" : "failed";
}

function renderScorecard(metrics, runMode) {
  const el = $("eval-scorecard");
  el.innerHTML = "";
  const live = state.answerEval;
  let shown = 0;
  let anyLive = false;

  // Emitted as the group CHANGES rather than looped per group, so a metric the
  // published run does not carry cannot leave an empty heading behind it.
  let openGroup = null;
  for (const [key, t] of Object.entries(EVAL_TARGETS)) {
    const bench = metrics ? metrics[key] : null;
    if (bench == null) continue;
    shown++;

    if (t.group && t.group !== openGroup) {
      openGroup = t.group;
      const g = EVAL_GROUPS[t.group];
      const head = document.createElement("div");
      head.className = "score-group";
      head.innerHTML =
        `<span class="score-group-name">${g.label}</span>` +
        `<span class="score-group-sub">${g.sub}</span>`;
      el.appendChild(head);
    }

    const field = Object.keys(LIVE_TO_BENCHMARK).find(
      (f) => LIVE_TO_BENCHMARK[f] === key,
    );
    const v = field ? liveValue(live, field) : null;
    const hasLive = v != null;
    if (hasLive) anyLive = true;
    // Top similarity is known the moment the answer is; the two judged rows are
    // not, and they are what the second request is fetching. Only those wait.
    const waiting = !hasLive && !!(live && live.pending) && !!field;

    const benchPct = Math.round(bench * 100);
    const livePct = hasLive ? Math.round(v * 100) : 0;
    // Below the benchmark is not a failure — one answer against an average of
    // 53 is a comparison, not a verdict — so the bar is the accent when there
    // is a live reading and muted when it is only showing the benchmark.
    const below = hasLive && v < bench;

    const row = document.createElement("div");
    row.className =
      "score-row" + (hasLive ? " has-live" : "") + (waiting ? " is-waiting" : "");
    row.innerHTML = `
      <div class="score-head">
        <span class="score-name">${t.label}<span class="score-sub">${t.sub}${
          t.from ? `<span class="score-from">${t.from}</span>` : ""
        }</span></span>
        <span class="score-val ${hasLive ? (below ? "under" : "over") : "quiet"}">${
          hasLive ? escapeHtml(liveLabel(live, field)) : waiting ? "grading…" : `${benchPct}%`
        }</span>
      </div>
      <div class="score-bar">
        <div class="score-fill ${hasLive ? "live" : "bench"}"
             style="width:${Math.min(100, hasLive ? livePct : benchPct)}%"></div>
        <div class="score-target" style="left:${Math.min(100, benchPct)}%"
             title="benchmark ${benchPct}%"></div>
      </div>
      <div class="score-foot">${
        hasLive
          ? `this answer ${livePct}% · benchmark ${benchPct}%`
          : waiting
            ? `benchmark ${benchPct}% · waiting on the judge`
            : field
              ? `benchmark ${benchPct}%`
              // Six of the nine can never carry a live reading, and saying so
              // where the question arises beats leaving a row that looks
              // broken. They need a KNOWN answer to score against: precision,
              // MRR, NDCG and hit rate all ask "was the right passage
              // returned", which nobody can judge for a question the corpus
              // has no golden answer for. Correctness needs an expected
              // answer; the not-found rate needs a set of deliberately
              // unanswerable questions.
              : `benchmark ${benchPct}% · needs a known answer`
      }</div>`;
    el.appendChild(row);
  }

  if (!shown) {
    el.innerHTML = emptyState("◔", "", "No benchmark metrics available.");
    return;
  }

  // Latency has no bar because it has no counterpart in the benchmark. Given a
  // row of its own rather than squeezed onto an axis it does not belong on.
  if (live && live.latency_ms != null) {
    const t = document.createElement("div");
    t.className = "score-aside";
    // Two numbers, not one. `latency_ms` is the answer; the grading that fills
    // the bars above costs its own time and the reader waits for it too, so
    // folding it into "Answered in" overstated the answer and dropping it
    // understated the wait. Both are shown, and the second is the honest price
    // of the first.
    const secs = (ms) => `${(ms / 1000).toFixed(1)}s`;
    t.innerHTML =
      `<span class="score-aside-pair"><span>Answered in</span>` +
      `<strong>${secs(live.latency_ms)}</strong></span>` +
      (live.pending
        ? `<span class="score-aside-pair"><span>grading</span>` +
          `<strong class="score-aside-wait">…</strong></span>`
        : live.grade_ms != null && live.grade_ms > 0
          ? `<span class="score-aside-pair"><span>then graded in</span>` +
            `<strong>${secs(live.grade_ms)}</strong></span>`
          : "");
    el.appendChild(t);
  }

  const legend = document.createElement("p");
  legend.className = "score-legend";
  legend.textContent = anyLive
    ? "Three of these can be measured on a live answer; the rest need a question whose right answer is already known, so they show the benchmark alone. The tick is that benchmark — a reference, not a pass mark."
    : "Benchmark across the sample corpus. Ask a question and the three measurable rows show that answer against it.";
  el.appendChild(legend);
}


function renderEvalQuestions(results) {
  const el = $("eval-questions");
  el.innerHTML = "";
  if (!results || !results.length) {
    el.innerHTML = emptyState("◔", "", "No per-question results.");
    return;
  }
  for (const r of results) {
    const card = document.createElement("div");
    card.className = "eval-q";
    const fh = r.faithful == null ? "—" : (r.faithful ? "✓" : "✗");
    const cd = r.correct == null ? "—" : (r.correct ? "✓" : "✗");
    const fhClass = r.faithful == null ? "" : (r.faithful ? "pass" : "fail");
    const cdClass = r.correct == null ? "" : (r.correct ? "pass" : "fail");
    card.innerHTML = `
      <div class="eval-q-head">
        <span class="eval-q-text">${escapeHtml(r.question)}</span>
      </div>
      <div class="eval-q-metrics">
        <span class="badge ${fhClass}">faith ${fh}</span>
        <span class="badge ${cdClass}">correct ${cd}</span>
        <span class="badge">recall ${r.context_recall != null ? r.context_recall.toFixed(2) : "—"}</span>
      </div>
      <div class="eval-q-io">
        <div class="io-col"><span class="io-label">Expected</span><span class="io-text">${escapeHtml(r.expected || "—")}</span></div>
        <div class="io-col"><span class="io-label">Actual</span><span class="io-text">${escapeHtml(r.answer || "—")}</span></div>
      </div>`;
    el.appendChild(card);
  }
}

async function loadEval() {
  try {
    // In parallel, not in sequence. The baseline is a small static file and the
    // scorecard cannot be drawn correctly without it, but making it a second
    // round trip after /api/eval would add its whole latency to a boot the
    // topbar is already waiting on.
    //
    // Fetched once and kept: it changes only when someone commits a new
    // baseline, so re-requesting it on every render would be pure round trips.
    const [data, base] = await Promise.all([
      api("/api/eval"),
      state.evalBaseline
        ? Promise.resolve({ baseline: state.evalBaseline })
        : api("/api/eval/baseline").catch(() => ({ baseline: null })),
    ]);
    if (base && base.baseline) state.evalBaseline = base.baseline;
    renderEval(data);
    // Nothing resumes a run any more. There is no way to START one from the
    // app, so a row left in "running" is abandoned by definition — and the
    // client used to keep driving it on EVERY page load, which is what put
    // "Scoring golden questions... 0/56" and a retry countdown on the
    // deployment for minutes at a time, spending model calls the whole way.
    // The server now retires such a row and serves the published result.
  } catch (e) {
    console.error("eval load failed:", e);
  }
}

function renderEval(data) {
  // Kept so renderScorecard can draw each answer against the published run.
  // The Evaluation pane and the answer readout used to share no state at all,
  // which is why the two sets of numbers read as unrelated.
  state.evalData = data;
  // Two different jobs, and they used to share one element. `status` is
  // transient — indexing, scoring, failed — and belongs at the top where it
  // interrupts. `provenance` is a standing fact about the published run and
  // belongs with the evidence, folded away.
  const statusEl = $("eval-status");
  const provEl = $("eval-provenance");
  if (provEl) provEl.textContent = "";
  if (!data || data.status === "none") {
    statusEl.textContent = "";
    // The pane used to say only "no run yet", which explained neither what a
    // benchmark is, how long it takes, nor that it spends real model calls —
    // a button with unstated cost behind it (plan §5).
    $("eval-scorecard").innerHTML = emptyState(
      "◎",
      "No benchmark run yet",
      "A benchmark replays the golden set through the live pipeline and scores it " +
      "against published targets — retrieval quality first, then whether each " +
      "answer is faithful and on-topic. It runs in slices from this tab, takes " +
      "several minutes, and spends a handful of model calls per question.",
      `<p class="glossary-strip">${termHtml("golden-set")} · ${termHtml("faithfulness")} · ${termHtml("relevancy")}</p>`
    );
    $("eval-questions").innerHTML = "";
    return;
  }
  if (data.status === "running") {
    // Show real progress. The run advances one slice per request, so the user
    // sees which phase it's in rather than an indefinite spinner.
    const files = data.total_files || 0;
    if (files && data.indexed_files < files) {
      statusEl.textContent = `Indexing corpus… ${data.indexed_files}/${files} files`;
    } else {
      statusEl.textContent = `Scoring golden questions… ${data.completed}/${data.total}`;
    }
    // Partial results stream in as slices complete.
    renderScorecard(data.metrics || {}, data.mode);
    renderEvalQuestions(data.results || []);
    return;
  }
  if (data.status === "error") {
    statusEl.textContent = "Benchmark failed: " + (data.error || "unknown error");
    renderScorecard(data.metrics || {}, data.mode);
    renderEvalQuestions(data.results || []);
    return;
  }
  if (data.status === "cancelled") {
    statusEl.textContent = "Benchmark cancelled.";
    return;
  }
  // done
  const ts = data.timestamp ? ` · ${data.timestamp}` : "";
  const ungraded = (data.metrics || {}).n_ungraded;
  // Whose numbers these are is the first thing a reader needs. Presenting a
  // shipped result as "your benchmark" would be a quiet lie, and the reader's
  // very next thought would be "when did I run this?".
  // No "run your own" any more — nothing in the app starts a benchmark. The
  // model that produced these is named because the scores depend on it, and it
  // is the one number here that is NOT a property of the retrieval pipeline.
  const model = (data.config || {}).llm_model;
  const whose = data.published
    ? `Published run${ts} · ${data.n_corpus_files || "?"} sample documents` +
      (model ? ` · ${model.replace(/^models\//, "")}` : "")
    : "Latest benchmark" + ts;
  statusEl.textContent = "";
  if (provEl) {
    provEl.textContent =
      whose + (ungraded ? ` · ⚠ ${ungraded} ungraded (judge unavailable)` : "");
  }
  renderScorecard(data.metrics || {}, data.mode);
  renderEvalQuestions(data.results || []);
}

// The benchmark step retry helpers lived here and are gone with the loop that
// used them. Nothing in the app drives a run any more: the result ships with
// it (eval/published_run.json) and the CLI re-measures when the numbers should
// actually change.


// ---------- collapsible panes (tablet / mobile) ----------

// On narrow viewports the side panes stack, so Sources and Evaluation become
// collapsible sections; the chat stays expanded. CSS only honours `.collapsed`
// below 1100px, so a stale class can never hide a desktop pane.
const NARROW = window.matchMedia("(max-width: 1099px)");
const MOBILE = window.matchMedia("(max-width: 767px)");

function setCollapsed(paneId, btnId, collapsed) {
  $(paneId).classList.toggle("collapsed", collapsed);
  $(btnId).setAttribute("aria-expanded", String(!collapsed));
}

function bindPaneToggle(paneId, btnId) {
  $(btnId).onclick = () => {
    setCollapsed(paneId, btnId, !$(paneId).classList.contains("collapsed"));
  };
}

bindPaneToggle("eval-pane", "eval-toggle");
bindPaneToggle("sources-pane", "sources-toggle");

// Default state per breakpoint: below 1100px the Evaluation scorecard starts
// folded so Sources + Chat keep the room. Sources stays open — adding a source
// is the first thing you do — but can be folded by hand on mobile.
function applyBreakpointDefaults() {
  // Not on mobile: there the panes are TABS, and a collapsed pane would open to
  // nothing but its own heading. The tab bar is the show/hide control below 768px.
  setCollapsed("eval-pane", "eval-toggle", NARROW.matches && !MOBILE.matches);
  if (!MOBILE.matches) setCollapsed("sources-pane", "sources-toggle", false);
  else {
    setCollapsed("sources-pane", "sources-toggle", false);
    setCollapsed("eval-pane", "eval-toggle", false);
  }
}
applyBreakpointDefaults();
NARROW.addEventListener("change", applyBreakpointDefaults);
MOBILE.addEventListener("change", applyBreakpointDefaults);

// ---------- mobile tabs (IDEA.md §9) ----------
//
// Below 768px the panes stop being a stacked scroll and become four tabbed
// views, one at a time, conversation first. The phone is a reader: the answer
// should be on screen when you arrive, not three sections down.
//
// The excerpt is NOT one of the tabs. It opens as a sheet over the conversation
// (showExcerpt) and closes back to it, because making the reader navigate away
// from an answer to read the passage that answer came from is the wrong trade.
const TABS = {
  chat: ".chat-pane",
  sources: ".sources-pane",
  chats: ".chats-pane",
  eval: ".eval-pane",
};

// Chat every load rather than the last tab used: arriving on the file list
// because that is where you finished yesterday is not what a reader wants.
let mobileTab = "chat";

function applyMobileTab() {
  const mobile = MOBILE.matches;
  for (const [name, sel] of Object.entries(TABS)) {
    document.querySelector(sel)?.classList.toggle("is-tab", mobile && name === mobileTab);
  }
  for (const btn of document.querySelectorAll(".tab")) {
    btn.setAttribute("aria-selected", String(btn.dataset.tab === mobileTab));
  }
  // The pane you are looking at IS the active one here, so the accent follows the
  // tab rather than the scroll position.
  if (mobile) setActivePane(document.querySelector(TABS[mobileTab]));
}

function setMobileTab(name) {
  if (!TABS[name]) return;
  mobileTab = name;
  applyMobileTab();
}

for (const btn of document.querySelectorAll(".tab")) {
  btn.onclick = () => setMobileTab(btn.dataset.tab);
}

// Counts on the tabs, so switching away from Sources does not mean losing track
// of what is in it. Called from the same refreshes that paint the pane heads.
function paintTabCounts() {
  const pairs = [
    ["tab-count-sources", state.sourceCount],
    ["tab-count-chats", state.chats.length],
  ];
  for (const [id, n] of pairs) {
    const el = $(id);
    if (!el) continue;
    el.textContent = String(n);
    el.classList.toggle("hidden", !n);
  }
}

// ---------- active pane ----------
//
// The pane you are working in wears the accent on its title. Two different
// signals drive it, because "which pane am I in" means two different things:
//
//   Desktop — the columns are all visible at once, so it is the one you last
//   touched or focused. That is a real statement about where you are working.
//
//   Mobile — the panes are stacked and you move through them by SCROLLING, not
//   by tapping. Marking on interaction there would leave the accent on whatever
//   you last pressed while you read something else entirely, so the reading
//   follows the scroll position instead: whichever section owns the top of the
//   viewport is the one you are in.
//
// When the mobile pass lands a one-pane-at-a-time switcher, the visible pane IS
// the active one and the scroll driver becomes unnecessary.
let activePane = null;

function panes() {
  return [...document.querySelectorAll(".panes .pane")];
}

function setActivePane(pane) {
  if (!pane || pane === activePane) return;
  activePane = pane;
  for (const p of panes()) p.classList.toggle("is-active", p === pane);
}

const panesEl = document.querySelector(".panes");

for (const type of ["pointerdown", "focusin"]) {
  panesEl.addEventListener(type, (e) => {
    if (MOBILE.matches) return;          // mobile is driven by scroll, below
    setActivePane(e.target.closest(".pane"));
  });
}

// Chat is the default on desktop too: it is where the cursor goes and where the
// answer lands, so a title is lit from first paint rather than after a click.
//
// On mobile there is no scroll-position tracking to do — the tab bar decides
// which pane is on screen, and applyMobileTab() sets the accent from it.
function applyActivePaneDefault() {
  if (MOBILE.matches) applyMobileTab();
  else setActivePane(document.querySelector(".chat-pane"));
}
applyActivePaneDefault();
MOBILE.addEventListener("change", applyActivePaneDefault);

// ---------- resizable columns ----------
//
// The grid tracks are sized by three custom properties (tokens.css). Dragging
// a handle writes one of them onto <html> as an inline style, which is why
// nothing here measures or positions a pane: the track and the handle both
// read the same property, and the browser lays out the rest. Inline styles on
// :root also outrank every media-query default, so a width you chose by hand
// survives crossing a breakpoint — a resize that silently reverted your layout
// would be worse than not offering one.
const LAYOUT_KEY = "ragchat-layout";
const RESIZE_VARS = ["--sources-w", "--chats-w", "--eval-w"];

function readLayout() {
  try { return JSON.parse(localStorage.getItem(LAYOUT_KEY)) || {}; }
  catch { return {}; }        // private mode / corrupt value — use defaults
}

function saveLayout(patch) {
  try {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify({ ...readLayout(), ...patch }));
  } catch { /* storage disabled: widths just don't persist */ }
}

function setVar(name, px) {
  document.documentElement.style.setProperty(name, `${Math.round(px)}px`);
}

function currentVar(name) {
  return parseFloat(getComputedStyle(document.documentElement).getPropertyValue(name)) || 0;
}

function bindResizer(el) {
  const varName = el.dataset.var;
  const min = parseFloat(el.dataset.min);
  const max = parseFloat(el.dataset.max);
  // The Evaluation column is anchored to the right edge, so dragging right
  // must SHRINK it. Without this the handle would run away from the cursor.
  const dir = el.dataset.invert ? -1 : 1;
  const panes = document.querySelector(".panes");

  const clamp = (px) => Math.min(max, Math.max(min, px));

  el.addEventListener("pointerdown", (e) => {
    // Ignore secondary buttons: a right-click drag should open the context
    // menu, not resize.
    if (e.button !== 0) return;
    e.preventDefault();
    const startX = e.clientX;
    const startW = currentVar(varName);
    el.setPointerCapture(e.pointerId);
    el.classList.add("is-dragging");
    panes.classList.add("is-resizing");

    const onMove = (ev) => setVar(varName, clamp(startW + (ev.clientX - startX) * dir));
    const onUp = () => {
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onUp);
      el.classList.remove("is-dragging");
      panes.classList.remove("is-resizing");
      // Persist once, on release — not on every move, which would hammer
      // localStorage with a synchronous write per animation frame.
      saveLayout({ [varName]: currentVar(varName) });
    };
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);
  });

  // Keyboard: a separator that can only be dragged is unusable without a
  // mouse, and these are focusable (role="separator", tabindex=0) so they
  // already appear in the tab order.
  el.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 32 : 8;
    let next = null;
    if (e.key === "ArrowLeft") next = currentVar(varName) - step * dir;
    else if (e.key === "ArrowRight") next = currentVar(varName) + step * dir;
    else if (e.key === "Home") {                    // reset this column
      document.documentElement.style.removeProperty(varName);
      const l = readLayout(); delete l[varName];
      try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(l)); } catch {}
      e.preventDefault();
      return;
    } else return;
    e.preventDefault();
    setVar(varName, clamp(next));
    saveLayout({ [varName]: currentVar(varName) });
  });

  // Double-click resets, the convention every split-pane UI uses.
  el.addEventListener("dblclick", () => {
    document.documentElement.style.removeProperty(varName);
    const l = readLayout(); delete l[varName];
    try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(l)); } catch {}
  });
}

document.querySelectorAll(".resizer").forEach(bindResizer);

// Restore saved widths before first paint of the panes.
(function restoreLayout() {
  const l = readLayout();
  for (const v of RESIZE_VARS) {
    if (typeof l[v] === "number") setVar(v, l[v]);
  }
})();

// ---------- conversations column collapse ----------
//
// Auto-collapses below 1300px so four columns do not squeeze the conversation,
// but ONLY until you express a preference: an explicit collapse or expand is
// remembered and the breakpoint stops overriding it. Otherwise every window
// resize would undo a deliberate choice.
const CHATS_AUTO = window.matchMedia("(max-width: 1299px)");

function setChatsCollapsed(collapsed, remember) {
  document.querySelector(".panes").classList.toggle("chats-collapsed", collapsed);
  $("chats-collapse").setAttribute("aria-expanded", String(!collapsed));
  if (remember) saveLayout({ chatsCollapsed: collapsed });
}

function applyChatsDefault() {
  const saved = readLayout().chatsCollapsed;
  setChatsCollapsed(typeof saved === "boolean" ? saved : CHATS_AUTO.matches, false);
}

$("chats-collapse").onclick = () => setChatsCollapsed(true, true);
$("chats-expand").onclick = () => setChatsCollapsed(false, true);
applyChatsDefault();
CHATS_AUTO.addEventListener("change", applyChatsDefault);

// ---------- command palette (IDEA.md §8) ----------
//
// The one HUD marker adopted beyond the token system, and it earns its place by
// being useful rather than decorative: in a four-column app, jumping to a chat or
// a source without reaching for the mouse is the fastest path there is.
//
// DESKTOP ONLY — it has no meaning on touch. The gate is on the same 1100px
// breakpoint where the columns stop being columns, and it covers the shortcut as
// well as the button, so a narrow window has no hidden keyboard surface.
const PALETTE_DESKTOP = window.matchMedia("(min-width: 1100px)");
const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);

let paletteRows = [];
let paletteIndex = 0;

// The fields worth reaching by name. Not all fifteen: a palette that lists every
// input is a form, and the presets are the intended way in for most of them.
// Labels carry no "Setting:" prefix — the group column beside them already says
// it, and the duplication read as "Setting: Setting: chunk size".
const PALETTE_SETTINGS = {
  "set-chunk-size": "Chunk size",
  "set-top-k": "Top-K (chunks to LLM)",
  "set-candidate-k": "Candidate-K (pool before rerank)",
  "set-sim-threshold": "Similarity threshold",
  "set-llm-model": "LLM model",
  "set-embedding-model": "Embedding model",
  "set-reranker": "Reranker",
};

// Actions route through the REAL controls rather than re-implementing them: the
// guest locks, disabled states and confirmations all live on those handlers, and
// a palette that bypassed them would be a second, laxer way in.
function clickThrough(id) {
  return () => $(id)?.click();
}

function paletteCommands() {
  const rows = [
    { group: "Action", label: "New chat", run: clickThrough("new-chat-btn") },
    { group: "Action", label: "Open settings", run: () => openSettings() },
    { group: "Action", label: "Toggle theme (dark / light)", run: clickThrough("theme-toggle") },
    { group: "Action", label: "Toggle deep search (word-for-word scan)", run: clickThrough("deep-toggle") },
    { group: "Action", label: "Re-index all sources", run: clickThrough("reindex-btn"), lock: "reindex-btn" },
    // No `lock`: this one is NOT in GUEST_LOCKED and the endpoint answers 200
    // for a guest (tests/test_guest_permissions.py). It carried lock:
    // "prune-btn" and so advertised "needs an account" for something a guest
    // may do — the palette was contradicting the server.
    { group: "Action", label: "Clean up leftover chunks", run: cleanUpOrphanChunks },
  ];
  for (const c of state.chats) {
    rows.push({
      group: "Chat",
      label: c.title,
      run: () => { if (c.id !== state.currentChatId) openChat(c.id); },
    });
  }
  // Sources come from the DOM rather than from state: the card list is already
  // the authoritative rendering of them, and "jump to" means "show me that card".
  for (const el of document.querySelectorAll("#doc-list .source-item, #folder-list .source-item")) {
    const title = el.querySelector(".src-title")?.textContent?.trim();
    if (!title) continue;
    rows.push({ group: "Source", label: title, run: () => revealSource(el) });
  }
  // Settings fields, so "open a setting" is one keystroke and a name rather than
  // a modal plus a disclosure plus a scroll.
  for (const [id, label] of Object.entries(PALETTE_SETTINGS)) {
    rows.push({
      group: "Setting",
      label,
      run: () => {
        openSettings();
        setAdvanced(true);
        const field = $(id);
        field?.scrollIntoView({ block: "center" });
        field?.focus();
      },
    });
  }
  // Mark what a guest cannot do BEFORE they press it. The handler still explains
  // itself if they do — this only saves them the trip.
  for (const r of rows) {
    if (r.lock && state.isGuest) r.hint = "needs an account";
  }
  return rows;
}

function revealSource(el) {
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("is-revealed");
  setTimeout(() => el.classList.remove("is-revealed"), 1200);
}

// Substring match, with a prefix hit ranked above a mid-word one. Deliberately
// not fuzzy: with a handful of actions and a user's own chat titles, fuzzy
// matching mostly produces confident wrong answers.
function paletteFilter(q) {
  const all = paletteCommands();
  const needle = q.trim().toLowerCase();
  if (!needle) return all;
  return all
    .map((r) => {
      const hay = `${r.group} ${r.label}`.toLowerCase();
      const at = hay.indexOf(needle);
      const labelAt = r.label.toLowerCase().indexOf(needle);
      return at === -1 ? null : { ...r, score: labelAt === 0 ? 0 : labelAt > 0 ? 1 : 2 };
    })
    .filter(Boolean)
    .sort((a, b) => a.score - b.score);
}

function renderPalette() {
  const list = $("palette-list");
  if (!paletteRows.length) {
    list.innerHTML = `<li class="palette-none">No match</li>`;
    return;
  }
  list.innerHTML = paletteRows
    .map((r, i) => `<li class="palette-row${i === paletteIndex ? " is-active" : ""}"
        role="option" aria-selected="${i === paletteIndex}" data-i="${i}">
        <span class="palette-group">${escapeHtml(r.group)}</span>
        <span class="palette-label">${escapeHtml(r.label)}</span>
        ${r.hint ? `<span class="palette-lock">${escapeHtml(r.hint)}</span>` : ""}
      </li>`)
    .join("");
  list.querySelector(".is-active")?.scrollIntoView({ block: "nearest" });
}

function movePalette(delta) {
  if (!paletteRows.length) return;
  paletteIndex = (paletteIndex + delta + paletteRows.length) % paletteRows.length;
  renderPalette();
}

function openPalette() {
  if (!PALETTE_DESKTOP.matches) return;
  $("palette-overlay").classList.remove("hidden");
  const input = $("palette-input");
  input.value = "";
  paletteRows = paletteFilter("");
  paletteIndex = 0;
  renderPalette();
  input.focus();
}

function closePalette() {
  $("palette-overlay").classList.add("hidden");
  // The input is inside a display:none subtree once closed, and a focused
  // element in one keeps receiving keystrokes that go nowhere visible.
  if (document.activeElement === $("palette-input")) $("palette-input").blur();
}

function runPalette(i) {
  const row = paletteRows[i];
  if (!row) return;
  closePalette();
  row.run();
}

$("slash-btn").onclick = openPalette;

$("palette-input").addEventListener("input", (e) => {
  paletteRows = paletteFilter(e.target.value);
  paletteIndex = 0;
  renderPalette();
});

$("palette-input").addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown") { e.preventDefault(); movePalette(1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); movePalette(-1); }
  else if (e.key === "Enter") { e.preventDefault(); runPalette(paletteIndex); }
  else if (e.key === "Escape") { e.preventDefault(); closePalette(); }
});

$("palette-list").addEventListener("click", (e) => {
  const row = e.target.closest(".palette-row");
  if (row) runPalette(Number(row.dataset.i));
});

// Hover moves the selection so the mouse and the keyboard cannot disagree about
// which row Enter would run.
$("palette-list").addEventListener("mousemove", (e) => {
  const row = e.target.closest(".palette-row");
  if (!row) return;
  const i = Number(row.dataset.i);
  if (i !== paletteIndex) { paletteIndex = i; renderPalette(); }
});

$("palette-overlay").addEventListener("click", (e) => {
  if (e.target === $("palette-overlay")) closePalette();
});

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K")) {
    if (!PALETTE_DESKTOP.matches) return;
    e.preventDefault();     // Ctrl+K would otherwise reach the browser's omnibox
    if ($("palette-overlay").classList.contains("hidden")) openPalette();
    else closePalette();
  }
});

// Ctrl on Windows and Linux, ⌘ on a Mac. Getting this wrong advertises a
// shortcut for someone else's computer. The button reads "/" at every size now,
// so the key name lives in its tooltip rather than in its label.
$("slash-btn").title = `Commands — type / here, or press ${IS_MAC ? "⌘K" : "Ctrl+K"}`;

// Crossing into a narrow window closes it: the overlay is desktop-only, and a
// resize should not leave a dialog on screen that the layout no longer offers.
PALETTE_DESKTOP.addEventListener("change", (e) => { if (!e.matches) closePalette(); });

// ---------- guest workspace lifecycle ----------
//
// A guest workspace is destroyed after 30 minutes idle (ragchat/guests.py).
// "Idle" has to mean idle, not "open but quiet": reading a long answer for
// forty minutes must not race a sweeper. These two handlers are what make the
// server's `last_seen_at` mean what the TTL assumes it means.

// Comfortably inside both the TTL and the server's own 5-minute write throttle,
// so an open tab is always fresh without writing on every ping.
const KEEPALIVE_MS = 4 * 60 * 1000;

// Set immediately before any navigation we KNOW is coming back to us as the
// same person — the OAuth hop above all. `pagehide` cannot tell that departure
// apart from a close, and back-dating during sign-in would hand the sweeper a
// workspace the promotion is in the middle of preserving.
let leavingForAuth = false;

function goToOAuth() {
  leavingForAuth = true;
  window.location.href = "/api/auth/google/login";
}

let keepaliveTimer = null;

function startGuestKeepalive() {
  const ping = () => {
    // Only while the tab is actually on screen. A backgrounded tab left open
    // for a week is not someone reading, and pinging from it would make the
    // TTL unenforceable — the one workspace that never expires is the one
    // nobody is looking at.
    if (document.visibilityState !== "visible" || !state.isGuest) return;
    // /api/auth/status calls touch() server-side. The response is deliberately
    // ignored: re-rendering identity every four minutes is how the badge ended
    // up repainting mid-session, and nothing here needs the payload.
    api("/api/auth/status").catch(() => {});
  };
  if (keepaliveTimer) clearInterval(keepaliveTimer);
  keepaliveTimer = setInterval(ping, KEEPALIVE_MS);
  // Coming back to a tab that has been hidden a while is exactly when the
  // workspace is closest to expiry, so ping on the way in rather than waiting
  // out the rest of the interval.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") ping();
  });
}

// `pagehide`, not `beforeunload`: mobile Safari and Chrome frequently discard a
// backgrounded tab without ever firing beforeunload, and that is the single
// most common way a guest workspace is abandoned.
window.addEventListener("pagehide", () => {
  if (!state.isGuest || leavingForAuth) return;
  // sendBeacon, because a normal fetch is cancelled when the page goes away.
  // The server BACK-DATES on this rather than deleting: pagehide fires on a
  // reload too, and a visitor who reloads has to find their work still there.
  try {
    navigator.sendBeacon("/api/auth/leaving");
  } catch (e) {
    // Nothing to do and nobody to tell — the page is already going.
  }
});

// ---------- boot ----------

// Surface any script-load or runtime error captured by the inline collector
// in index.html, plus our own boot failures. Without this, a broken app.js
// looks like a "working" page where nothing responds.
(function boot() {
  const prev = window.__ragchat_errors || [];
  if (prev.length) toast("Script error: " + prev.join("; "), true);
  initAuth()
    .catch((e) => {
      console.error("boot failed:", e);
      toast("Boot failed: " + e.message, true);
    })
    // AFTER initAuth, not alongside it: the identity badge now reads the auth
    // state that initAuth resolved. Run in parallel it could render before
    // guest-login had provisioned anything, showing a signed-out topbar over a
    // working guest workspace. Chained off .catch() so it still runs when the
    // boot itself failed — a failed boot is exactly when a sign-in button is
    // most useful. Its own failure stays non-fatal: sign-in is optional.
    .then(() => initGoogleAuth())
    // After identity resolves, because it only runs for guests.
    .then(() => startGuestKeepalive())
    .catch((e) => console.warn("google auth init failed:", e));
})();
