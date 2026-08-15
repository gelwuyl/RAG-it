// RAG Chat frontend — vanilla JS, talks to the FastAPI backend via /api.

const $ = (id) => document.getElementById(id);

const state = {
  user: null,
  chats: [],
  currentChatId: null,
  currentCitations: [], // citations of the last assistant message, for the excerpt pane
  models: { chat: [], embedding: [] }, // proxy model catalog for the settings dropdowns
};

// Human-friendly labels for the model dropdowns.
const MODEL_LABELS = {
  "deepseek-v4-pro": "DeepSeek V4 Pro (class default)",
  "qwen3.8-max": "Qwen3.8 Max",
  "qwen3-coder": "Qwen3 Coder (metered)",
  "text-embedding-005": "text-embedding-005 (768 dims)",
  "gemini-embedding": "gemini-embedding (3072 dims)",
};

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
  // (Full auth — Google OAuth / local accounts — is deferred per PRD.)
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
    // Auth failed but we still show the app shell — the user will see
    // errors when they try to chat, rather than a blank screen.
    console.error("auth failed:", e);
  }
  showApp();
  try {
    await Promise.all([refreshSources(), refreshChats(), refreshHybridToggle()]);
  } catch (e) {
    console.error("boot fetch failed:", e);
  }
}

// ---------- sources ----------

const STATUS_ICON = { ready: "✓", indexing: "…", pending: "…", failed: "✗" };

function renderFolders(folders) {
  const list = $("folder-list");
  list.innerHTML = "";
  for (const f of folders) {
    const item = document.createElement("div");
    item.className = "source-item";
    item.innerHTML = `
      <div class="row">
        <span>📁</span>
        <span class="title" title="${escapeHtml(f.path)}">${escapeHtml(f.path)}</span>
      </div>
      <div class="row">
        <span class="status muted small">${f.n_docs} docs</span>
        <span style="flex:1"></span>
        <button class="icon-btn" data-act="rescan" title="Rescan folder">↻</button>
        <button class="icon-btn" data-act="remove" title="Remove folder source">✕</button>
      </div>`;
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
    const icon = d.source_type === "url" ? "🌐" : "📄";
    const sub = d.source_type === "url" ? d.path_or_url : (d.path_or_url || "upload");
    item.innerHTML = `
      <div class="row">
        <span>${icon}</span>
        <span class="title" title="${escapeHtml(sub || "")}">${escapeHtml(d.title)}</span>
        <button class="icon-btn" data-act="delete" title="Delete">✕</button>
      </div>
      <div class="row">
        <span class="status ${d.status}">${STATUS_ICON[d.status] || ""} ${d.status}${d.n_chunks ? ` · ${d.n_chunks} chunks` : ""}</span>
      </div>
      ${d.error ? `<div class="row error-text small">${escapeHtml(d.error)}</div>` : ""}`;
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

$("hybrid-toggle").onclick = async () => {
  try {
    const r = await api("/api/eval/hybrid-search", { method: "POST" });
    const btn = $("hybrid-toggle");
    if (r.hybrid_search) {
      btn.textContent = "On";
      btn.classList.add("on");
      toast("Web search ON — answers will include web results");
    } else {
      btn.textContent = "Off";
      btn.classList.remove("on");
      toast("Web search OFF — answers from your documents only");
    }
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
    .map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(MODEL_LABELS[m] || m)}</option>`)
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

async function refreshChats(selectId = null) {
  state.chats = await api("/api/chats");
  const sel = $("chat-select");
  sel.innerHTML = "";
  for (const c of state.chats) {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = c.title;
    sel.appendChild(opt);
  }
  const target = selectId || (state.chats[0] && state.chats[0].id);
  if (target) {
    sel.value = target;
    await openChat(target);
  } else {
    state.currentChatId = null;
    $("messages").innerHTML = `<div class="empty-hint" id="empty-hint">
      Add sources on the left, then ask anything about them.
      Answers are grounded in your documents and cite their sources.</div>`;
  }
}

$("chat-select").onchange = async (e) => {
  if (e.target.value) await openChat(e.target.value);
};

$("new-chat-btn").onclick = async () => {
  try {
    const chat = await api("/api/chats", { method: "POST" });
    await refreshChats(chat.id);
  } catch (e) { toast(e.message, true); }
};

async function openChat(chatId) {
  state.currentChatId = chatId;
  const chat = await api(`/api/chats/${chatId}`);
  const box = $("messages");
  box.innerHTML = "";
  if (chat.messages.length === 0) {
    box.innerHTML = `<div class="empty-hint">Start by asking a question about your documents.</div>`;
    return;
  }
  for (const m of chat.messages) {
    appendMessage(m.role, m.content, m.citations || [], false);
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
    for (const c of citations) {
      const chip = document.createElement("button");
      chip.className = "cite-chip";
      chip.textContent = `[${c.number}] ${c.title}`;
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

function appendMessage(role, content, citations = [], isPending = false) {
  const box = $("messages");
  const hint = box.querySelector(".empty-hint");
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

  try {
    const result = await api(`/api/chats/${state.currentChatId}/ask`, {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    pending.remove();
    appendMessage("assistant", result.answer, result.citations);
    // keep the chat list titles in sync
    if (state.chats.length) {
      const c = state.chats.find((x) => x.id === state.currentChatId);
      if (c && c.title === "New chat") await refreshChats(state.currentChatId);
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

// ---------- excerpt pane ----------

function showExcerpt(citation) {
  $("excerpt-content").classList.remove("hidden");
  document.querySelector(".excerpt-empty").classList.add("hidden");
  $("excerpt-title").textContent = citation.title;
  $("excerpt-ref").textContent = citation.ref || "";
  $("excerpt-text").textContent = citation.excerpt;
}

$("excerpt-close").onclick = () => {
  $("excerpt-content").classList.add("hidden");
  document.querySelector(".excerpt-empty").classList.remove("hidden");
};

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
