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
  evalRunning: false, // true while the chunked benchmark loop is driving
  simThreshold: 0,    // live retrieval threshold, marked on the per-answer meter
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

// ---------- the four-beat loop (PRODUCT_UX_PLAN.md §5) ----------
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

// ---------- glossary (PRODUCT_UX_PLAN.md §5) ----------
//
// Every one of these was previously either undefined or explained in a `title`
// attribute, which does not exist on touch. A tap target with a real popover
// works for everyone; the trade is one shared element and a click-away handler.
const GLOSSARY = {
  "web-fallback": [
    "Web fallback",
    "When on, web results are appended as labelled [web] chunks — but only when " +
    "your own documents fail to clear the relevance threshold. It never " +
    "overrides a grounded answer, and it is off by default.",
  ],
  "prune-ghosts": [
    "Ghost chunks",
    "Vectors left in the store with no document behind them, usually after a " +
    "failed delete. Pruning removes those and nothing else — every chunk that " +
    "still belongs to one of your sources is kept.",
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
        window.location.href = "/api/auth/google/login";
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
        await api("/api/auth/guest-login", { method: "POST" });
        status = await api("/api/auth/status");
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
  state.guestUsage = status.guest_usage || null;
  state.googleOAuth = !!status.google_oauth;

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
  "eval-run-btn": "run the benchmark",
  "reindex-btn": "re-index every source",
  "add-folder-btn": "add a folder source",
  "web-toggle": "change the web fallback",
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
    // beats hiding (PRODUCT_UX_PLAN.md §5).
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
  el.innerHTML = `<span class="guest-badge-tag">Guest</span>
    <span class="guest-badge-usage${full ? " is-full" : ""}">${used}/${max} files</span>
    <span class="guest-badge-note">Sign in to keep your work</span>`;
}

// Usage changes on every add and delete, so the badge has to be refreshed from
// the server rather than incremented locally — the server is the only thing
// that knows which documents count against the cap.
async function refreshGuestUsage() {
  if (!state.isGuest) return;
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
    window.location.href = "/api/auth/google/login";
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
  $("sources-empty").classList.toggle("hidden", state.sourceCount > 0);
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

// Web augmentation fallback (DuckDuckGo [web] chunks only when documents don't answer).
$("web-toggle").onclick = async () => {
  if (guestBlocked("web-toggle")) return;
  try {
    const r = await api("/api/eval/web-augmentation", { method: "POST" });
    const btn = $("web-toggle");
    if (r.web_augmentation) {
      btn.textContent = "On";
      btn.classList.add("on");
      toast("Web fallback ON — web text used only when docs don't answer");
    } else {
      btn.textContent = "Off";
      btn.classList.remove("on");
      toast("Web fallback OFF — strictly grounded in your documents");
    }
  } catch (e) {
    toast(e.message, true);
  }
};

// Prune orphaned Neon vector chunks (no matching Document row) + stale fingerprints.
$("prune-btn").onclick = async () => {
  try {
    const r = await api("/api/documents/prune", { method: "POST" });
    toast(`Pruned ${r.removed} orphaned vector chunk(s)`);
  } catch (e) {
    toast(e.message, true);
  }
};

/* Tuning parameters modal */
function openSettings() {
  $("settings-overlay").classList.remove("hidden");
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

// ---------- presets (PRODUCT_UX_PLAN.md §7) ----------
//
// The default view is four named configurations, each stating what it trades
// away; the fifteen individual fields stay one click behind "Advanced".
//
// Presets deliberately do NOT touch the model or provider fields. Those are
// deployment facts rather than tradeoffs — switching embedding provider
// re-points every vector in the store, and 422s outright when that provider has
// no key configured — so a card called "Fast" has no business deciding them.
const PRESET_KEYS = [
  "chunk_size", "chunk_overlap", "splitter", "top_k", "candidate_k",
  "similarity_threshold", "reranker", "query_rewrite", "temperature",
];

// Keys whose change invalidates existing chunks — the same set the backend uses
// to decide `needs_reindex` (app.py:1129), minus the embedding fields no preset
// touches. A card's badge is computed against the SAVED config, not against
// another preset, so it promises a re-index only when one is really coming.
const PRESET_INDEX_KEYS = ["chunk_size", "chunk_overlap", "splitter"];

const PRESETS = [
  {
    id: "fast",
    name: "Fast",
    desc: "One model call per question. No rerank, no rewrite, a small pool — the quickest answer this pipeline can give.",
    values: {
      chunk_size: 512, chunk_overlap: 75, splitter: "recursive",
      top_k: 3, candidate_k: 10, similarity_threshold: 0.0,
      reranker: false, query_rewrite: false, temperature: 0.0,
    },
  },
  {
    id: "balanced",
    name: "Balanced",
    desc: "The shipped default. Follow-ups are rewritten so they still retrieve; everything else stays cheap.",
    values: {
      chunk_size: 512, chunk_overlap: 75, splitter: "recursive",
      top_k: 4, candidate_k: 20, similarity_threshold: 0.0,
      reranker: false, query_rewrite: true, temperature: 0.0,
    },
  },
  {
    id: "accurate",
    name: "High accuracy",
    desc: "Smaller chunks, a 40-wide candidate pool and an LLM rerank pass. One extra model call per question, and slower.",
    values: {
      chunk_size: 384, chunk_overlap: 96, splitter: "recursive",
      top_k: 6, candidate_k: 40, similarity_threshold: 0.0,
      reranker: true, query_rewrite: true, temperature: 0.0,
    },
  },
  {
    id: "low-cost",
    name: "Low cost",
    desc: "Bigger chunks means roughly a third fewer embedding calls to index, and the smallest prompt per question.",
    values: {
      chunk_size: 768, chunk_overlap: 64, splitter: "recursive",
      top_k: 3, candidate_k: 8, similarity_threshold: 0.0,
      reranker: false, query_rewrite: false, temperature: 0.0,
    },
  },
];

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
  return {
    chunk_size: parseInt($("set-chunk-size").value, 10),
    chunk_overlap: parseInt($("set-chunk-overlap").value, 10),
    splitter: $("set-splitter").value,
    top_k: parseInt($("set-top-k").value, 10),
    candidate_k: parseInt($("set-candidate-k").value, 10),
    similarity_threshold: parseFloat($("set-sim-threshold").value),
    reranker: $("set-reranker").value === "true",
    query_rewrite: $("set-query-rewrite").value === "true",
    temperature: parseFloat($("set-temperature").value),
  };
}

function matchPreset(vals) {
  return PRESETS.find((p) => PRESET_KEYS.every((k) => sameSetting(p.values[k], vals[k]))) || null;
}

function presetNeedsReindex(preset) {
  if (!loadedConfig) return false;
  return PRESET_INDEX_KEYS.some((k) => !sameSetting(preset.values[k], loadedConfig[k]));
}

function renderPresets() {
  const list = $("preset-list");
  if (!list) return;
  list.innerHTML = PRESETS.map((p) => {
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
  const preset = PRESETS.find((p) => p.id === id);
  if (!preset) return;
  const v = preset.values;
  $("set-chunk-size").value = v.chunk_size;
  $("set-chunk-overlap").value = v.chunk_overlap;
  $("set-splitter").value = v.splitter;
  $("set-top-k").value = v.top_k;
  $("set-candidate-k").value = v.candidate_k;
  $("set-sim-threshold").value = v.similarity_threshold;
  $("set-reranker").value = String(v.reranker);
  $("set-query-rewrite").value = String(v.query_rewrite);
  $("set-temperature").value = v.temperature;
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

// Advanced stays where you left it. Collapsing it on every open would make the
// full field set feel hidden rather than folded, and this is a tuning app.
const ADVANCED_KEY = "ragchat-settings-advanced";

function setAdvanced(open, remember) {
  $("advanced-fields").classList.toggle("hidden", !open);
  $("advanced-toggle").setAttribute("aria-expanded", String(open));
  if (remember) {
    try { localStorage.setItem(ADVANCED_KEY, open ? "1" : "0"); } catch { /* storage off */ }
  }
}

$("advanced-toggle").onclick = () => {
  setAdvanced($("advanced-fields").classList.contains("hidden"), true);
};

(function restoreAdvanced() {
  let open = false;
  try { open = localStorage.getItem(ADVANCED_KEY) === "1"; } catch { /* storage off */ }
  setAdvanced(open, false);
})();

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
    $("set-reranker-provider").value = cfg.reranker_provider || "gemini";
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
    const rpSel = $("set-reranker-provider");
    if (rpSel) {
      rpSel.onchange = () => updateProviderWarnings(openrouterConfigured);
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
  return provider === "openrouter" ? "openai/text-embedding-3-small" : "models/gemini-embedding-001";
}

function updateProviderWarnings(configured) {
  const warn = $("provider-warning");
  const ep = $("set-embedding-provider")?.value;
  const rp = $("set-reranker-provider")?.value;
  if (configured || (ep !== "openrouter" && rp !== "openrouter")) {
    if (warn) warn.classList.add("hidden");
    return;
  }
  if (warn) {
    warn.textContent = "OpenRouter selected but no OPENROUTER_API_KEY found in .env — add it and restart the backend.";
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
      reranker_provider: $("set-reranker-provider").value,
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
    const chunkingChanged = !loadedConfig || PRESET_INDEX_KEYS.some(
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

// ---------- persistent job status (PRODUCT_UX_PLAN.md §5) ----------
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
    appendMessage(m.role, m.content, m.citations || [], false, m.eval_line || "", m.eval_data || null);
  }
  box.scrollTop = box.scrollHeight;
}

// Render [1] markers as clickable spans; citations list becomes chips below.
function renderAssistantContent(el, content, citations) {
  const withMarkers = escapeHtml(content).replace(
    /\[(\d+)\]/g,
    (m, n) => `<span class="cite-marker" data-cite="${n}">[${n}]</span>`
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
      chip.title = `Show the passage from “${c.title}”`;
      chip.innerHTML =
        `<span class="cite-num">${c.number}</span>` +
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

// One word for the whole quality verdict (PRODUCT_UX_PLAN.md §6).
//
// UNGRADED IS NOT FAILURE and must never render as one. A judge that 404s or
// times out is a broken grader, not a bad answer — `faithful`/`relevant` are
// nullable for exactly this reason, and showing "Weak" there would be a
// confident false claim about the user's own documents. Null is checked BEFORE
// falsity so a partial grading can never be read as a verdict.
function evalVerdict(evalData) {
  const { faithful, relevant } = evalData;
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
  const wrap = document.createElement("div");
  wrap.className = "eval-block";

  if (evalData && typeof evalData === "object") {
    const rows = [];
    if (evalData.top_sim != null) {
      rows.push(["Top similarity", evalData.top_sim.toFixed(2), "how closely the best retrieved chunk matched your question (1.00 = exact)"]);
    }
    // PASS is a CHECKMARK in neutral foreground, not a green pill. Acid lime is
    // the accent, and lime beside green is unreadable as two distinct meanings,
    // so the verdict is carried by glyph and position. FAIL stays red — red is
    // reserved exclusively for failure and destructive actions.
    if (evalData.faithful != null) {
      rows.push(["Faithfulness", evalData.faithful ? "✓" : "FAIL", evalData.faithful_reason || "every claim is supported by the sources"]);
    }
    if (evalData.relevant != null) {
      rows.push(["Relevancy", evalData.relevant ? "✓" : "FAIL", evalData.relevant_reason || "the answer addresses your question"]);
    }
    // Grading unavailable is NOT a failure — say so explicitly rather than
    // leaving the row out (or, as before, rendering it as a confident FAIL).
    if (evalData.judge_error && (evalData.faithful == null || evalData.relevant == null)) {
      rows.push(["Grading", "unavailable", String(evalData.judge_error).slice(0, 160)]);
    }
    if (evalData.latency_ms != null) {
      rows.push(["Latency", `${(evalData.latency_ms / 1000).toFixed(1)} s`, "time to generate this answer"]);
    }
    if (rows.length) {
      const { state, word } = evalVerdict(evalData);
      const sim = evalData.top_sim;

      const summary = document.createElement("button");
      summary.type = "button";
      summary.className = "eval-chip";
      summary.setAttribute("aria-expanded", "false");

      let meter = "";
      if (sim != null) {
        // The bar is top similarity on 0–1, which is a real per-answer value.
        //
        // Plan §6 asked for a tick at "the benchmark baseline", citing the
        // published faithfulness ≥ 0.90 / relevancy ≥ 0.85 targets. Those are
        // PASS RATES ACROSS A RUN, not similarities — drawing one as a tick on
        // this axis would look precise and mean nothing. The retrieval
        // threshold is the real reference on this axis, so it is marked when
        // the config sets one above zero, and simply omitted otherwise rather
        // than invented.
        const pct = Math.max(0, Math.min(1, sim)) * 100;
        const thr = state.simThreshold;
        const tick = thr > 0 && thr < 1
          ? `<span class="eval-meter-tick" style="left:${(thr * 100).toFixed(1)}%"
               title="retrieval threshold ${thr.toFixed(2)}"></span>`
          : "";
        meter = `<span class="eval-meter" role="img"
            aria-label="top similarity ${sim.toFixed(2)} out of 1">
            <span class="eval-meter-fill" style="width:${pct.toFixed(1)}%"></span>${tick}
          </span><span class="eval-meter-val">${sim.toFixed(2)}</span>`;
      }
      // Beat 4 of the loop lives here rather than in an empty state, because by
      // the time it is reachable there is no empty state left on screen: the
      // readout only exists under an answer. The nudge disappears for good the
      // first time the detail is opened.
      const nudge = currentBeat() === 4
        ? `<span class="eval-nudge">what is this?</span>`
        : "";
      summary.innerHTML = `<span class="eval-dot" data-state="${state}" aria-hidden="true"></span>
        <span class="eval-state">${escapeHtml(word)}</span>${meter}${nudge}
        <span class="eval-caret" aria-hidden="true">›</span>`;

      const detail = document.createElement("div");
      detail.className = "eval-detail hidden";
      for (const [label, value, gloss] of rows) {
        const row = document.createElement("div");
        row.className = "eval-row";
        const vClass = value === "✓" ? "pass" : value === "FAIL" ? "fail" : "";
        // Two of these labels are the app's central jargon. Definitions reach a
        // touch user here; a `title` attribute would not.
        const term = EVAL_LABEL_TERMS[label];
        const labelHtml = term ? termHtml(term, label) : label;
        row.innerHTML = `<span class="eval-label">${labelHtml}</span>` +
          `<span class="eval-value ${vClass}">${escapeHtml(String(value))}</span>` +
          `<span class="eval-gloss">${escapeHtml(gloss)}</span>`;
        detail.appendChild(row);
      }
      summary.onclick = () => {
        const open = detail.classList.toggle("hidden");
        summary.setAttribute("aria-expanded", String(!open));
        markBeat("readout");                       // beat 4 — the loop is complete
        summary.querySelector(".eval-nudge")?.remove();
      };
      wrap.appendChild(summary);
      wrap.appendChild(detail);
      return wrap;
    }
  }
  // Fallback: show the terse line as-is (legacy messages before eval_data existed).
  if (evalLine) {
    const row = document.createElement("div");
    row.className = "eval-row";
    row.innerHTML = `<span class="eval-label">Eval</span>` +
      `<span class="eval-gloss">${escapeHtml(evalLine)}</span>`;
    wrap.appendChild(row);
  }
  return wrap;
}

function appendMessage(role, content, citations = [], isPending = false, evalLine = "", evalData = null) {
  const box = $("messages");
  const hint = box.querySelector(".empty-state");
  if (hint) hint.remove();

  const el = document.createElement("div");
  el.className = `msg ${role}${isPending ? " pending" : ""}`;
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
      body: JSON.stringify({ question }),
    });
    pending.remove();
    appendMessage("assistant", result.answer, result.citations, false, result.eval_line || "", result.eval || null);
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

// RAGAS-style metric targets (from eval/EVAL_SPEC.md). Used to render the
// scorecard bars and the pass/fail colouring.
const EVAL_TARGETS = {
  context_recall: { label: "Context Recall", target: 0.80, higher: true },
  precision_at_k: { label: "Precision@k", target: 0.70, higher: true },
  mrr: { label: "MRR", target: 0.65, higher: true },
  ndcg_at_k: { label: "NDCG@k", target: 0.70, higher: true },
  hit_rate_at_k: { label: "Hit Rate@k", target: 0.80, higher: true },
  faithfulness: { label: "Faithfulness", target: 0.90, higher: true },
  answer_relevancy: { label: "Answer Relevancy", target: 0.85, higher: true },
  answer_correctness: { label: "Answer Correctness", target: 0.80, higher: true },
  not_found_rate_unanswerables: { label: "Not-found rate (unanswerables)", target: 0.90, higher: true },
};

function fmtPct(v) {
  if (v == null) return "—";
  return `${Math.round(v * 100)}%`;
}

function renderScorecard(metrics) {
  const el = $("eval-scorecard");
  el.innerHTML = "";
  const keys = Object.keys(EVAL_TARGETS);
  let shown = 0;
  for (const k of keys) {
    const t = EVAL_TARGETS[k];
    const v = metrics[k];
    if (v == null) continue;
    shown++;
    const pct = Math.round(v * 100);
    const targetPct = Math.round(t.target * 100);
    const meets = v >= t.target;
    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML = `
      <div class="score-head">
        <span class="score-name">${t.label}</span>
        <span class="score-val ${meets ? "pass" : "fail"}">${pct}%</span>
      </div>
      <div class="score-bar">
        <div class="score-fill ${meets ? "pass" : "fail"}" style="width:${Math.min(100, pct)}%"></div>
        <div class="score-target" style="left:${Math.min(100, targetPct)}%" title="Target ${targetPct}%"></div>
      </div>
      <div class="score-foot">target ${targetPct}%</div>`;
    el.appendChild(row);
  }
  if (!shown) {
    el.innerHTML = emptyState("◔", "", "No generation metrics in this run yet.");
  }
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
    const data = await api("/api/eval");
    renderEval(data);
    // A run left in flight (tab closed, reload mid-benchmark) resumes from the
    // last committed slice instead of being stranded at "running" forever.
    if (data.status === "running" && !state.evalRunning) driveEvalRun();
  } catch (e) {
    console.error("eval load failed:", e);
  }
}

function renderEval(data) {
  const statusEl = $("eval-status");
  const runBtn = $("eval-run-btn");
  // `locked` is not "no run yet" — it is a feature that is not this visitor's
  // to run. Saying "No benchmark run yet" to a guest invites them to press a
  // button that will answer 403, so say what actually unblocks it. The
  // per-answer grades below still work for guests and are left alone.
  if (data && data.locked) {
    statusEl.textContent = "";
    $("eval-scorecard").innerHTML = emptyState(
      "◎",
      "Benchmark needs an account",
      "Sign in with Google to score retrieval and generation against the 46-question golden set. Every answer you ask below is still graded for faithfulness and relevance.",
      `<p class="glossary-strip">${termHtml("golden-set")} · ${termHtml("faithfulness")} · ${termHtml("relevancy")}</p>`
    );
    $("eval-questions").innerHTML = "";
    runBtn.disabled = true;
    runBtn.classList.add("guest-locked");
    return;
  }
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
    runBtn.disabled = false;
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
    runBtn.disabled = true;
    // Partial results stream in as slices complete.
    renderScorecard(data.metrics || {});
    renderEvalQuestions(data.results || []);
    return;
  }
  if (data.status === "error") {
    statusEl.textContent = "Benchmark failed: " + (data.error || "unknown error");
    runBtn.disabled = false;
    renderScorecard(data.metrics || {});
    renderEvalQuestions(data.results || []);
    return;
  }
  if (data.status === "cancelled") {
    statusEl.textContent = "Benchmark cancelled.";
    runBtn.disabled = false;
    return;
  }
  // done
  runBtn.disabled = false;
  const ts = data.timestamp ? ` · ${data.timestamp}` : "";
  const ungraded = (data.metrics || {}).n_ungraded;
  statusEl.textContent =
    "Latest benchmark" + ts + (ungraded ? ` · ⚠ ${ungraded} ungraded (judge unavailable)` : "");
  renderScorecard(data.metrics || {});
  renderEvalQuestions(data.results || []);
}

// Drive the run to completion, one slice per request. Each POST /api/eval/step
// does a bounded piece of work and commits it, so this loop is resumable: if
// the tab is closed mid-run, reopening it picks up from the last committed
// slice rather than starting over.
// A step that overruns the serverless time limit returns 504 (or 502/503) and,
// crucially, has NOT committed its slice — so re-issuing the request simply
// redoes that same slice. Aborting the whole benchmark on the first such blip
// threw away a run that was actually fine and resumable, which is what produced
// "Benchmark failed: Request failed (504)" near the end of a run: the free-tier
// quota depletes as the run proceeds, the server's own retry/backoff stretches a
// step past the limit, and one timeout killed everything. Waiting and retrying
// also gives the rate limit time to recover.
const EVAL_TRANSIENT_STATUS = /\((408|409|425|429|500|502|503|504)\)/;
// A function that hits its time limit does not always answer with a tidy 504 —
// it often just drops the connection, which surfaces as a fetch-level TypeError
// ("Failed to fetch" / "NetworkError" / "Load failed") with no status at all.
// Same cause, same safe response: the slice never committed, so retry it.
const EVAL_TRANSIENT_NETWORK = /failed to fetch|networkerror|network request failed|load failed/i;
const EVAL_STEP_MAX_RETRIES = 5;

function isTransientEvalError(err) {
  const msg = (err && err.message) || "";
  return EVAL_TRANSIENT_STATUS.test(msg) || EVAL_TRANSIENT_NETWORK.test(msg);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function driveEvalRun() {
  if (state.evalRunning) return;
  state.evalRunning = true;
  let failures = 0;
  try {
    for (;;) {
      let data;
      try {
        data = await api("/api/eval/step", { method: "POST" });
        failures = 0; // a slice landed — reset the budget
      } catch (e) {
        if (!isTransientEvalError(e) || ++failures > EVAL_STEP_MAX_RETRIES) throw e;
        const wait = Math.min(2000 * 2 ** (failures - 1), 15000);
        $("eval-status").textContent =
          `Step timed out — retrying in ${Math.round(wait / 1000)}s ` +
          `(attempt ${failures}/${EVAL_STEP_MAX_RETRIES}). Progress so far is saved.`;
        await sleep(wait);
        if (!state.evalRunning) break; // superseded while we were waiting
        continue;
      }
      renderEval(data);
      if (data.status !== "running") break;
      if (!state.evalRunning) break; // cancelled by a new run starting
    }
  } catch (e) {
    // Reopening the Evaluation tab resumes: loadEval() sees a run still marked
    // "running" and restarts the driver from the last committed slice. The Run
    // button deliberately does NOT resume — it supersedes the row and starts a
    // fresh run — so don't point the user at it here.
    $("eval-status").textContent =
      "Benchmark paused: " + e.message +
      " — progress is saved; reopen the Evaluation tab to resume.";
    $("eval-run-btn").disabled = false;
  } finally {
    state.evalRunning = false;
  }
}

$("eval-run-btn").onclick = async () => {
  // Guests still SEE the last run's scorecard — it is the most portfolio-legible
  // thing in the app — they just cannot spend 46 scored questions triggering a
  // new one.
  if (guestBlocked("eval-run-btn")) return;
  try {
    $("eval-run-btn").disabled = true;
    $("eval-status").textContent = "Starting benchmark…";
    state.evalRunning = false; // supersede any in-flight loop
    const r = await api("/api/eval/run", {
      method: "POST",
      body: JSON.stringify({ retrieval_only: false }),
    });
    renderEval(r);
    await driveEvalRun();
  } catch (e) {
    toast("Benchmark failed: " + e.message, true);
    $("eval-run-btn").disabled = false;
  }
};

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

// ---------- mobile tabs (PRODUCT_UX_PLAN.md §9) ----------
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

// ---------- command palette (PRODUCT_UX_PLAN.md §8) ----------
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
    { group: "Action", label: "Toggle web fallback", run: clickThrough("web-toggle"), lock: "web-toggle" },
    { group: "Action", label: "Re-index all sources", run: clickThrough("reindex-btn"), lock: "reindex-btn" },
    { group: "Action", label: "Run benchmark", run: clickThrough("eval-run-btn"), lock: "eval-run-btn" },
    { group: "Action", label: "Prune ghost chunks", run: clickThrough("prune-btn"), lock: "prune-btn" },
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
        setAdvanced(true, true);
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

$("palette-btn").onclick = openPalette;

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

// Ctrl on Windows and Linux, ⌘ on a Mac. Getting this wrong makes the hint read
// as a shortcut for someone else's computer.
$("palette-btn").textContent = IS_MAC ? "⌘K" : "Ctrl K";

// Crossing into a narrow window closes it: the overlay is desktop-only, and a
// resize should not leave a dialog on screen that the layout no longer offers.
PALETTE_DESKTOP.addEventListener("change", (e) => { if (!e.matches) closePalette(); });

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
    .catch((e) => console.warn("google auth init failed:", e));
})();
