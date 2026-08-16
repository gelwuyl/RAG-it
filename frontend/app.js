// RAG Chat frontend — vanilla JS, talks to the FastAPI backend via /api.

const $ = (id) => document.getElementById(id);

const state = {
  user: null,
  chats: [],
  currentChatId: null,
  currentCitations: [], // citations of the last assistant message, for the excerpt pane
  models: { chat: [], embedding: [] }, // proxy model catalog for the settings dropdowns
  evalPolling: null,
};

// Human-friendly labels for known models. With live proxy discovery the
// catalog is dynamic, so any model not listed here simply shows its raw id.
const MODEL_LABELS = {
  "deepseek-v4-pro": "DeepSeek V4 Pro (class default)",
  "qwen3.8-max": "Qwen3.8 Max",
  "qwen3-coder": "Qwen3 Coder (metered)",
  "text-embedding-005": "text-embedding-005 (768 dims)",
  "gemini-embedding": "gemini-embedding (3072 dims)",
};

function modelLabel(id) {
  return MODEL_LABELS[id] || id;
}

// ---------- helpers ----------

async function api(path, options = {}) {
  const res = await fetch(path, {
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

// Shared markup for the "nothing here yet" panels.
function emptyState(icon, title, text) {
  return `<div class="empty-state">
      <span class="empty-icon" aria-hidden="true">${icon}</span>
      ${title ? `<p class="empty-title">${escapeHtml(title)}</p>` : ""}
      <p class="empty-text">${escapeHtml(text)}</p>
    </div>`;
}

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
  // Single-user mode: sign in automatically as the built-in local account.
  try {
    const status = await api("/api/auth/status");
    if (!status.authenticated) {
      await api("/api/auth/local-login", { method: "POST" });
    }
    const refreshed = await api("/api/auth/status");
    state.user = refreshed.user;
    const nameEl = $("user-name");
    if (nameEl) nameEl.textContent = state.user?.name || "";
  } catch (e) {
    console.error("auth failed:", e);
  }
  showApp();
  try {
    await Promise.all([refreshSources(), refreshChats(), refreshHybridToggle(), loadEval()]);
  } catch (e) {
    console.error("boot fetch failed:", e);
  }
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
        toast("Rescanning…");
        const r = await api(`/api/folders/${f.id}/rescan`, { method: "POST" });
        toast(`Rescan: +${r.added} new, ${r.reindexed} updated, ${r.unchanged} unchanged${r.failed ? `, ${r.failed} failed` : ""}`);
        await refreshSources();
      } catch (e) { toast(e.message, true); }
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

async function refreshSources() {
  const [docs, folders] = await Promise.all([
    api("/api/documents"),
    api("/api/folders"),
  ]);
  renderFolders(folders);
  renderDocs(docs);
  $("source-count").textContent = String(docs.length + folders.length);
  $("sources-empty").classList.toggle("hidden", docs.length + folders.length > 0);
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
  for (const file of files) {
    try {
      toast(`Uploading ${file.name}…`);
      const form = new FormData();
      form.append("file", file);
      await api("/api/documents/upload", { method: "POST", body: form });
      toast(`${file.name} indexed`);
    } catch (e) {
      toast(`${file.name}: ${e.message}`, true);
    }
  }
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
    toast(`Indexed “${doc.title}”`);
    await refreshSources();
  } catch (e) {
    toast(e.message, true);
  } finally {
    $("add-url-btn").disabled = false;
  }
};

$("add-folder-btn").onclick = async () => {
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

async function refreshHybridToggle() {
  try {
    const cfg = await api("/api/eval/config");
    const btn = $("hybrid-toggle");
    if (cfg.hybrid_search) {
      btn.textContent = "On";
      btn.classList.add("on");
    } else {
      btn.textContent = "Off";
      btn.classList.remove("on");
    }
  } catch (e) {
    console.error("hybrid toggle failed:", e);
  }
}

// Footer toggle controls real BM25 keyword fusion (vector + keyword RRF).
// It is NOT web search — document grounding is unaffected. Web augmentation
// is a separate, default-off fallback exposed elsewhere.
$("hybrid-toggle").onclick = async () => {
  try {
    const r = await api("/api/eval/hybrid-search", { method: "POST" });
    const btn = $("hybrid-toggle");
    if (r.hybrid_search) {
      btn.textContent = "On";
      btn.classList.add("on");
      toast("Keyword fusion ON — exact IDs/names surface better");
    } else {
      btn.textContent = "Off";
      btn.classList.remove("on");
      toast("Keyword fusion OFF — vector-only retrieval");
    }
  } catch (e) {
    toast(e.message, true);
  }
};

// Web augmentation fallback (DuckDuckGo [web] chunks only when documents don't answer).
$("web-toggle").onclick = async () => {
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
  loadSettingsIntoForm();
}

// Value the embedding model had when the settings form was opened; used to
// decide whether saving needs a re-index prompt.
let loadedEmbeddingModel = null;

function fillModelSelect(id, models, current) {
  const sel = $(id);
  // Keep a hand-edited model that isn't in the catalog selectable.
  const all = models.includes(current) ? models : [...models, current];
  sel.innerHTML = all
    .map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(modelLabel(m))}</option>`)
    .join("");
  sel.value = current;
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
    $("set-hybrid-search").value = String(cfg.hybrid_search);
    $("set-query-rewrite").value = String(cfg.query_rewrite);
    fillModelSelect("set-llm-model", models.chat, cfg.llm_model);
    $("set-temperature").value = cfg.temperature;
    fillModelSelect("set-embedding-model", models.embedding, cfg.embedding_model);
    loadedEmbeddingModel = cfg.embedding_model;
  } catch (e) {
    toast(e.message, true);
  }
}

function closeSettings() {
  $("settings-overlay").classList.add("hidden");
  $("settings-note").classList.add("hidden");
}

$("settings-btn").onclick = openSettings;
$("settings-close").onclick = closeSettings;
$("settings-overlay").onclick = (e) => {
  if (e.target === $("settings-overlay")) closeSettings();
};

$("settings-save").onclick = async () => {
  try {
    const body = {
      chunk_size: parseInt($("set-chunk-size").value),
      chunk_overlap: parseInt($("set-chunk-overlap").value),
      splitter: $("set-splitter").value,
      top_k: parseInt($("set-top-k").value),
      candidate_k: parseInt($("set-candidate-k").value),
      similarity_threshold: parseFloat($("set-sim-threshold").value),
      reranker: $("set-reranker").value === "true",
      hybrid_search: $("set-hybrid-search").value === "true",
      query_rewrite: $("set-query-rewrite").value === "true",
      llm_model: $("set-llm-model").value,
      temperature: parseFloat($("set-temperature").value),
      embedding_model: $("set-embedding-model").value,
    };
    const r = await api("/api/eval/config", {
      method: "PUT",
      body: JSON.stringify(body),
    });
    if (r.needs_reindex) {
      $("settings-note").classList.remove("hidden");
      if (body.embedding_model !== loadedEmbeddingModel) {
        // The index is unusable until re-embedded with the new model — offer
        // to do it right away rather than only leaving a note.
        $("settings-note").textContent =
          "Embedding model changed — sources must be re-indexed before asking.";
        if (confirm("Embedding model changed. Re-index all sources now?")) {
          await reindexAll();
        }
      } else {
        $("settings-note").textContent =
          "Index-affecting keys changed. You should re-index your sources.";
      }
    } else {
      $("settings-note").classList.add("hidden");
    }
    await refreshHybridToggle();
    toast("Settings saved — next ask will use the new config.");
  } catch (e) {
    toast("Save failed: " + e.message, true);
  }
};

async function reindexAll() {
  toast("Re-indexing all sources (this may take a while)…");
  $("reindex-btn").disabled = true;
  try {
    const r = await api("/api/documents/reindex", { method: "POST" });
    toast(`Re-indexed ${r.reindexed} sources`);
    await refreshSources();
  } catch (e) {
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

function renderChats() {
  const list = $("chat-list");
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
        await api(`/api/chats/${c.id}`, { method: "DELETE" });
        chatStatusOverride.delete(c.id);
        if (state.currentChatId === c.id) state.currentChatId = null;
        await refreshChats(state.currentChatId);
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
    $("messages").innerHTML = emptyState(
      "◈",
      "Ask a question to see results",
      "Add sources on the left, then ask anything about them. Answers are grounded in your documents and cite their sources."
    );
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
    box.innerHTML = emptyState(
      "◈",
      "Ask a question to see results",
      "This conversation is empty. Ask something about your documents to get a cited, grounded answer."
    );
    return;
  }
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

// Build a readable evaluation block from the full eval dict (preferred) or the
// terse eval_line string (legacy). This is what makes "top sim" / "rel" / etc.
// understandable instead of cryptic abbreviations.
function buildEvalBlock(evalData, evalLine) {
  const wrap = document.createElement("div");
  wrap.className = "eval-block";

  if (evalData && typeof evalData === "object") {
    const rows = [];
    if (evalData.top_sim != null) {
      rows.push(["Top similarity", evalData.top_sim.toFixed(2), "how closely the best retrieved chunk matched your question (1.00 = exact)"]);
    }
    if (evalData.faithful != null) {
      rows.push(["Faithfulness", evalData.faithful ? "PASS" : "FAIL", evalData.faithful_reason || "every claim is supported by the sources"]);
    }
    if (evalData.relevant != null) {
      rows.push(["Relevancy", evalData.relevant ? "PASS" : "FAIL", evalData.relevant_reason || "the answer addresses your question"]);
    }
    if (evalData.latency_ms != null) {
      rows.push(["Latency", `${(evalData.latency_ms / 1000).toFixed(1)} s`, "time to generate this answer"]);
    }
    if (rows.length) {
      for (const [label, value, gloss] of rows) {
        const row = document.createElement("div");
        row.className = "eval-row";
        const vClass = value === "PASS" ? "pass" : value === "FAIL" ? "fail" : "";
        row.innerHTML = `<span class="eval-label">${label}</span>` +
          `<span class="eval-value ${vClass}">${escapeHtml(String(value))}</span>` +
          `<span class="eval-gloss">${escapeHtml(gloss)}</span>`;
        wrap.appendChild(row);
      }
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

// ---------- excerpt pane (bottom) ----------

function showExcerpt(citation) {
  $("excerpt-content").classList.remove("hidden");
  $("excerpt-close").classList.remove("hidden");
  document.querySelector(".excerpt-empty").classList.add("hidden");
  $("excerpt-title").textContent = citation.title;
  $("excerpt-ref").textContent = citation.ref || "";
  $("excerpt-text").textContent = citation.excerpt;
  // On mobile the excerpt sits at the very bottom of the page — bring it up.
  if (window.matchMedia("(max-width: 767px)").matches) {
    $("excerpt-pane").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

$("excerpt-close").onclick = () => {
  $("excerpt-content").classList.add("hidden");
  $("excerpt-close").classList.add("hidden");
  $("excerpt-title").textContent = "";
  $("excerpt-ref").textContent = "";
  document.querySelector(".excerpt-empty").classList.remove("hidden");
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
  } catch (e) {
    console.error("eval load failed:", e);
  }
}

function renderEval(data) {
  const statusEl = $("eval-status");
  const runBtn = $("eval-run-btn");
  if (!data || data.status === "none") {
    statusEl.textContent = "";
    $("eval-scorecard").innerHTML = emptyState(
      "◎",
      "No benchmark run yet",
      "Run the RAGAS-style benchmark to score retrieval and generation against the golden set."
    );
    $("eval-questions").innerHTML = "";
    runBtn.disabled = false;
    return;
  }
  if (data.status === "running") {
    statusEl.textContent = "Benchmark running… (indexing corpus + scoring golden questions)";
    runBtn.disabled = true;
    if (!state.evalPolling) startEvalPolling();
    return;
  }
  if (data.status === "error") {
    statusEl.textContent = "Benchmark failed: " + (data.error || "unknown error");
    runBtn.disabled = false;
    return;
  }
  // done
  runBtn.disabled = false;
  const ts = data.timestamp ? ` · ${data.timestamp}` : "";
  statusEl.textContent = "Latest benchmark" + ts;
  renderScorecard(data.metrics || {});
  renderEvalQuestions(data.results || []);
}

function startEvalPolling() {
  if (state.evalPolling) return;
  state.evalPolling = setInterval(async () => {
    try {
      const data = await api("/api/eval");
      renderEval(data);
      if (data.status !== "running") {
        clearInterval(state.evalPolling);
        state.evalPolling = null;
      }
    } catch (e) {
      clearInterval(state.evalPolling);
      state.evalPolling = null;
    }
  }, 2500);
}

$("eval-run-btn").onclick = async () => {
  try {
    $("eval-run-btn").disabled = true;
    $("eval-status").textContent = "Starting benchmark…";
    const r = await api("/api/eval/run", { method: "POST" });
    if (r.status === "running" || r.status === "started") {
      startEvalPolling();
    } else {
      await loadEval();
    }
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
  setCollapsed("eval-pane", "eval-toggle", NARROW.matches);
  if (!MOBILE.matches) setCollapsed("sources-pane", "sources-toggle", false);
}
applyBreakpointDefaults();
NARROW.addEventListener("change", applyBreakpointDefaults);
MOBILE.addEventListener("change", applyBreakpointDefaults);

// ---------- boot ----------

// Surface any script-load or runtime error captured by the inline collector
// in index.html, plus our own boot failures. Without this, a broken app.js
// looks like a "working" page where nothing responds.
(function boot() {
  const prev = window.__ragchat_errors || [];
  if (prev.length) toast("Script error: " + prev.join("; "), true);
  initAuth().catch((e) => {
    console.error("boot failed:", e);
    toast("Boot failed: " + e.message, true);
  });
})();
