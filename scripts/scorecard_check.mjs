/**
 * Scorecard provenance check — the part of the Evaluation pane a screenshot
 * cannot prove. Eight answers with different grading provenance are served
 * through a mocked API (no backend, no model calls), and the check drives the
 * REAL path a reader does: open the app, click each answer's eval chip, and
 * assert what the two panels show.
 *
 * The pane now has TWO panels with two different jobs:
 *
 *   PRIMARY (#eval-scorecard, outside the expander) — exactly four rows in two
 *   groups: Context precision / Context recall (Retrieval) and Faithfulness /
 *   Answer relevancy (Generation). Per-answer, per-RAGAS: a dash before
 *   grading, "grading…" while the deferred request runs, then a percent from
 *   the judge's 0-1 score (or the binary passed/failed word when the judge
 *   omitted its score line). Retrieval rows carry the provenance tag derived
 *   from expected_source: "bank" → "known" (the demo bank's HUMAN reference),
 *   "draft" → "estimated" (a model-drafted reference). Generation rows need no
 *   reference and carry NO tag. There are no benchmark bars, ticks or legend
 *   here — the published run is not this panel's subject.
 *
 *   DETAIL (#eval-benchmark, behind "See the benchmark in detail") — the
 *   published run, minus the four metric families the primary panel carries
 *   (context recall, precision@k, faithfulness, answer relevancy): the same
 *   name above and below the fold read as the same row twice. What remains —
 *   answer correctness, not-found rate, MRR, NDCG, hit rate@k — is exactly
 *   what the primary panel does not show: stacked bars, one legend, pills,
 *   feet. Its only per-answer dimension is MEASURED: a bank-matched
 *   question's gold readings land on the surviving rows with a "known" tag
 *   and a golden foot (gold readings for the trimmed families no longer
 *   render anywhere — the primary panel's live grades superseded them).
 *   Judged readings of the selected answer no longer appear here at all — a
 *   verdict about one answer beside an average over 53 questions is the
 *   comparison the four-RAGAS pivot stopped making.
 *
 * Fixture states (one per answer):
 *   A. bank-referenced, graded  — primary: four scores, retrieval "known";
 *      detail: gold rows measured against known passages.
 *   B. drafted-reference, graded — primary: retrieval "estimated".
 *   C. pending + bank           — primary: rows wait ("grading…").
 *   D. pending + drafted, mixed — a sliced run: some readings landed while
 *      others wait, and the panel shows both.
 *   E. binary judge verdict WITHOUT a score — the 100/0 fallback renders as
 *      the passed/failed word, and the bar fills 100/0.
 *   I. graded DEFINITIONAL verdict — context recall 0% because the passages
 *      cannot yield a reference; the system reason is reachable on hover
 *      while healthy rows stay tooltip-quiet.
 *   F. known-unanswerable refusal — detail: the not-found row reads "refused".
 *   G. known-unanswerable, answered anyway — detail: "missed" (✗ polarity).
 *   H. answerable, hit rate 0 — detail: the ✗ "cut" pill, while MRR — a ratio
 *      that merely LANDS on 0 — keeps its bar.
 *
 * Usage:  node scorecard_check.mjs [baseUrl]   (default http://localhost:5173)
 * Exits non-zero on the first failed assertion or on any page error.
 */
import { chromium } from "playwright";

const BASE = process.argv[2] || "http://localhost:5173";
const fails = [];

function check(label, ok, detail = "") {
  console.log(`${ok ? "  ✓" : "  ✗"} ${label}${detail ? ` — ${detail}` : ""}`);
  if (!ok) fails.push(label);
}

// The published-run metrics every detail bar needs.
const METRICS = {
  context_recall: 0.49, precision_at_k: 0.71,
  faithfulness: 0.86, answer_relevancy: 0.83, answer_correctness: 0.80,
  not_found_rate_unanswerables: 0.92,
  mrr: 0.65, ndcg_at_k: 0.70, hit_rate_at_k: 0.80,
};

const GOLD_RETRIEVAL = {
  idx: 64, src: "demo", question: "What are the standard menu prices at Meridian Coffee?",
  unanswerable: false, refused: false,
  mrr: 1.0, ndcg_at_k: 0.9, hit_rate_at_k: 1, context_recall: 1.0, precision_at_k: 0.5,
};

const GRADED = {
  pending: false,
  faithful: true, faithful_score: 0.93,
  relevant: true, relevant_score: 0.90,
  context_precision: true, context_precision_score: 0.72,
  context_recall: false, context_recall_score: 0.55,
  latency_ms: 1200, top_sim: 0.62, deep_n: 0, cited_rank: 1, pool_n: 4,
  grade_ms: 3100,
};

const MESSAGES = [
  { id: "m0", role: "user", content: "What are the standard menu prices?", citations: [] },
  // A. bank-referenced, graded
  { id: "m1", role: "assistant", content: "Espresso is 3.20 … [1]", citations: [],
    eval_line: "top sim 0.62 - 1200 ms",
    eval_data: { ...GRADED, expected_answer: "Espresso is 3.20, cappuccino 4.50 …",
      expected_source: "bank", gold: { ...GOLD_RETRIEVAL } } },
  // B. drafted-reference, graded
  { id: "m2", role: "assistant", content: "Espresso is 3.20 … [1]", citations: [],
    eval_line: "top sim 0.55 - 1100 ms",
    eval_data: { ...GRADED, expected_answer: "Espresso is 3.20 …",
      expected_source: "draft", gold: undefined, top_sim: 0.55, cited_rank: 2 } },
  // C. pending, bank-referenced — nothing has graded yet
  { id: "m3", role: "assistant", content: "Espresso is 3.20 … [1]", citations: [],
    eval_line: "top sim 0.62 - 1300 ms",
    eval_data: { pending: true, faithful: null, relevant: null,
      context_precision: null, context_recall: null,
      expected_answer: "Espresso is 3.20, cappuccino 4.50 …",
      expected_source: "bank", latency_ms: 1300, top_sim: 0.62, deep_n: 0,
      retry_after_ms: 60_000 } },
  // D. pending, drafted, MIXED — the state a sliced judge run actually
  // produces: the generation pair landed, the retrieval pair is still out.
  { id: "m4", role: "assistant", content: "Espresso is 3.20 … [1]", citations: [],
    eval_line: "top sim 0.58 - 1400 ms",
    eval_data: { pending: true, faithful: true, faithful_score: 0.88,
      relevant: true, relevant_score: 0.91,
      context_precision: null, context_recall: null,
      expected_answer: "Espresso is 3.20 …", expected_source: "draft",
      cited_rank: 2, pool_n: 4,
      latency_ms: 1400, top_sim: 0.58, deep_n: 0, retry_after_ms: 60_000 } },
  // E. terminal partial result: generation judges returned scoreless binary
  // verdicts, while retrieval could not be graded. `grade_exhausted` makes the
  // terminal state explicit; without it this fixture looks like a completed
  // binary case even though the missing retrieval readings would be retried.
  { id: "m5", role: "assistant", content: "Espresso is 3.20 … [1]", citations: [],
    eval_line: "top sim 0.60 - 1000 ms",
    eval_data: { pending: false, grade_exhausted: true,
      judge_error: "retrieval readings unavailable after bounded retries",
      faithful: true, relevant: false,
      context_precision: null, context_recall: null,
      cited_rank: 1, pool_n: 4, latency_ms: 1000, top_sim: 0.60, deep_n: 0 } },
  // F. known-unanswerable, refused
  { id: "m6", role: "assistant", content: "I couldn't find an answer…",
    citations: [], eval_line: "top sim 0.31 - 900 ms",
    eval_data: { pending: false, latency_ms: 900, top_sim: 0.31, deep_n: 0,
      gold: { idx: 70, src: "demo",
        question: "Does Meridian sell gift cards?",
        unanswerable: true, refused: true } } },
  // G. known-unanswerable, answered anyway — the ✗ polarity of the refusal
  // pill.
  { id: "m7", role: "assistant", content: "Meridian offers…", citations: [],
    eval_line: "top sim 0.35 - 800 ms",
    eval_data: { pending: false, latency_ms: 800, top_sim: 0.35, deep_n: 0,
      gold: { idx: 71, src: "demo",
        question: "Does Meridian sell gift cards?",
        unanswerable: true, refused: false } } },
  // H. answerable, hit rate 0 — the ✗ polarity of the made-the-cut pill.
  // MRR and NDCG are real ratios that merely LAND on 0, so they keep bars.
  { id: "m8", role: "assistant", content: "Meridian roasts… [1]", citations: [],
    eval_line: "top sim 0.52 - 950 ms",
    eval_data: { ...GRADED, expected_answer: "Meridian offers light, medium…",
      expected_source: "bank",
      gold: { idx: 72, src: "demo",
        question: "What roast levels does Meridian offer?",
        unanswerable: false, refused: false,
        mrr: 0.0, ndcg_at_k: 0.0, hit_rate_at_k: 0,
        context_recall: 0.25, precision_at_k: 0.1 } } },
  // I. graded definitional verdict — context recall 0% because the passages
  // cannot yield a reference. A system reason rides the graded row and must
  // be reachable on hover; healthy rows stay tooltip-quiet.
  { id: "m9", role: "assistant", content: "Meridian roasts… [1]", citations: [],
    eval_line: "top sim 0.50 - 1050 ms",
    eval_data: { pending: false,
      faithful: true, faithful_score: 0.9,
      relevant: true, relevant_score: 0.9,
      context_precision: true, context_precision_score: 0.8,
      context_recall: false, context_recall_score: 0.0,
      expected_answer: "", expected_source: "draft",
      expected_reason: "no reference derivable: the passages do not answer this",
      latency_ms: 1050, top_sim: 0.50, deep_n: 0, pool_n: 4 } },
].map((m) => ({ ...m, eval_data: Object.fromEntries(Object.entries(m.eval_data || {}).filter(([, v]) => v !== undefined)) }));

const EVAL_DONE = {
  status: "done", mode: "demo", metrics: METRICS, results: [],
  timestamp: "2026-08-28T00:00:00Z", n_ungraded: 0,
};

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
const page = await ctx.newPage();
const pageErrors = [];
const consoleErrors = [];
page.on("pageerror", (e) => pageErrors.push(String(e)));
page.on("console", (m) => {
  // A dev server without a backend refuses /api calls at the network layer —
  // that is environmental, not an app error. Everything else counts.
  if (m.type() === "error" && !/Failed to load resource/i.test(m.text())) {
    consoleErrors.push(m.text());
  }
});

// Catch-all FIRST (playwright consults handlers last-registered-first), then
// the specific shapes.
await page.route("**/api/**", (route) => route.fulfill({ json: {} }));
await page.route("**/api/auth/status", (route) =>
  route.fulfill({ json: { authenticated: true, is_guest: false, google_oauth: false } }));
await page.route("**/api/documents", (route) => route.fulfill({ json: [] }));
await page.route("**/api/chats", (route) =>
  route.fulfill({ json: [{ id: "c1", title: "Demo chat" }] }));
await page.route("**/api/chats/c1/grade**", (route) =>
  route.fulfill({ json: { eval: null, eval_line: "" } }));
// Hold m10's settled grade at the route boundary until the test has selected
// Eval. This makes the active-pane behaviour deterministic without submitting
// through the hidden composer.
let releaseM10Settled;
const m10Settled = new Promise((resolve) => { releaseM10Settled = resolve; });
let releaseM10Partial;
const m10Partial = new Promise((resolve) => { releaseM10Partial = resolve; });
// The mobile pass below drives one answer through the same bounded retry states
// as the real grade endpoint. Keep the queue in the mock, rather than using a
// timing-dependent response, so each state is observable and reproducible.
const mobileGradeResponses = new Map();
const mobileAskResponses = [
  {
    answer: "The ordinary ask path returned this grounded answer. [1]",
    citations: [], eval_line: "top sim 0.61 - 900 ms", message_id: "m10",
    eval: {
      pending: true, faithful: null, relevant: null,
      context_precision: null, context_recall: null,
      expected_answer: "The ordinary ask path returned this grounded answer.",
      expected_source: "draft", latency_ms: 900, top_sim: 0.61, deep_n: 0,
      retry_after_ms: 250,
    },
  },
  {
    answer: "A second ordinary ask arrived after the scorecard check. [1]",
    citations: [], eval_line: "top sim 0.59 - 850 ms", message_id: "m11",
    eval: {
      pending: false, faithful: true, faithful_score: 0.91,
      relevant: true, relevant_score: 0.89,
      context_precision: true, context_precision_score: 0.74,
      context_recall: true, context_recall_score: 0.81,
      latency_ms: 850, top_sim: 0.59, deep_n: 0, grade_ms: 240,
    },
  },
  {
    answer: "The judge was unavailable for this answer. [1]",
    citations: [], eval_line: "top sim 0.57 - 700 ms", message_id: "m12",
    eval: {
      pending: false, grade_unavailable: "Judge disabled in the mock",
      faithful: null, relevant: null, context_precision: null,
      context_recall: null, latency_ms: 700, top_sim: 0.57, deep_n: 0,
    },
  },
  {
    answer: "This answer has no stored precision passages. [1]",
    citations: [], eval_line: "top sim 0.56 - 650 ms", message_id: "m13",
    eval: {
      pending: false, precision_data_gap: true,
      faithful: true, faithful_score: 0.87,
      relevant: true, relevant_score: 0.86,
      context_precision: null, context_recall: true, context_recall_score: 0.78,
      latency_ms: 650, top_sim: 0.56, deep_n: 0, grade_ms: 210,
    },
  },
  {
    answer: "The judge gave up on this answer after bounded retries. [1]",
    citations: [], eval_line: "top sim 0.54 - 630 ms", message_id: "m15",
    eval: {
      pending: false, grade_exhausted: true,
      judge_error: "retrieval readings unavailable after bounded retries",
      faithful: true, faithful_score: 0.84, relevant: true, relevant_score: 0.85,
      context_precision: null, context_recall: null,
      latency_ms: 630, top_sim: 0.54, deep_n: 0, grade_ms: 220,
    },
  },
  {
    answer: "A final ordinary ask from Chat created this settled grade. [1]",
    citations: [], eval_line: "top sim 0.55 - 640 ms", message_id: "m14",
    eval: {
      pending: false, faithful: true, faithful_score: 0.9,
      relevant: true, relevant_score: 0.88,
      context_precision: true, context_precision_score: 0.76,
      context_recall: true, context_recall_score: 0.8,
      latency_ms: 640, top_sim: 0.55, deep_n: 0, grade_ms: 230,
    },
  },
];
await page.route("**/api/chats/c1**", async (route) => {
  const url = route.request().url();
  if (url.includes("/grade")) {
    const mid = url.split("/messages/")[1]?.split("/")[0];
    const queue = mobileGradeResponses.get(mid);
    if (queue?.length) {
      const response = queue.shift();
      // Both m10 grade responses are held behind explicit test gates, so the
      // pending → partial → settled assertions observe exactly the state the
      // test intends and no wall-clock delay can reorder them.
      if (mid === "m10" && response.eval?.context_precision == null) {
        await m10Partial;
      }
      if (mid === "m10" && response.eval?.context_precision != null) {
        await m10Settled;
      }
      return route.fulfill({ json: response });
    }
    const m = MESSAGES.find((x) => x.id === mid);
    return route.fulfill({ json: { eval: m?.eval_data || {}, eval_line: m?.eval_line || "" } });
  }
  return route.fulfill({ json: { id: "c1", title: "Demo chat", messages: MESSAGES } });
});
const mobileAskCalls = [];
await page.route("**/api/chats/c1/ask", async (route) => {
  const call = {
    question: route.request().postDataJSON()?.question,
    response: null,
  };
  mobileAskCalls.push(call);
  const response = mobileAskResponses[Math.min(mobileAskCalls.length - 1, mobileAskResponses.length - 1)];
  await route.fulfill({ json: response });
  // Keep request and response accounting together so the mobile helper can
  // prove that one submit did not consume two responses or leave one pending.
  call.response = response;
});
await page.route("**/api/eval/baseline", (route) => route.fulfill({ json: { baseline: {} } }));
await page.route("**/api/eval", (route) => route.fulfill({ json: EVAL_DONE }));

await page.goto(`${BASE}/app.html`, { waitUntil: "networkidle" });
await page.waitForSelector("#eval-scorecard", { timeout: 15_000 });
await page.waitForTimeout(300);

// innerText returns RENDERED text and the pane styles tags/strip labels with
// text-transform: uppercase — so every comparison below is case-folded. The
// assertions are about provenance, not about a stylesheet.
const lower = (s) => s.trim().toLowerCase();

// Both panels use .score-row, so every locator is scoped to its panel: the
// primary four live in #eval-scorecard, the nine benchmark rows in
// #eval-benchmark (inside the expander).
const pRow = (label) => page.locator("#eval-scorecard .score-row", { hasText: label });
const dRow = (label) => page.locator("#eval-benchmark .score-row", { hasText: label });

async function chipOf(row, label) {
  return lower(await row(label).locator(".score-val").innerText());
}
async function tagOf(row, label) {
  return lower(await row(label).locator(".score-tag").innerText());
}
async function tagCount(row, label) {
  return row(label).locator(".score-tag").count();
}
async function footOf(row, label) {
  return lower(await row(label).locator(".score-foot").innerText());
}
async function fillPct(row, label) {
  return row(label).locator(".score-fill").evaluate((el) => el.style.width);
}
// The state pill a boolean gold reading renders as, plus the two things that
// must hold on its detail row: NO live fill (the pill is the state mark, not a
// magnitude), and the gray benchmark bar still present for it to ride.
async function pillInfo(label) {
  const row = dRow(label);
  if ((await row.locator(".score-pill").count()) === 0) return null;
  const pill = row.locator(".score-pill");
  return {
    text: lower(await pill.innerText()),
    fail: await pill.evaluate((el) => el.classList.contains("is-fail")),
    fills: await row.locator(".score-fill").count(),
    benchFills: await row.locator(".bench-fill").count(),
  };
}

async function clickChip(messageId) {
  await page.locator(`.msg[data-message-id="${messageId}"] .eval-chip`).click();
  await page.waitForTimeout(250);
}

// ---------- before any answer is selected: the ask-a-question state ----------
check("empty: the primary panel invites the first compare",
  (await page.locator("#eval-scorecard .empty-state").count()) === 1);

// ---------- A. bank-referenced: four readings, retrieval known ----------
await clickChip("m1");
check("A: exactly four primary rows in two groups",
  (await page.locator("#eval-scorecard .score-row").count()) === 4 &&
    (await page.locator("#eval-scorecard .score-group").count()) === 2,
  `rows=${await page.locator("#eval-scorecard .score-row").count()} groups=${await page.locator("#eval-scorecard .score-group").count()}`);
check("A: precision chip reads the score",
  (await chipOf(pRow, "Context precision")).includes("72%"), await chipOf(pRow, "Context precision"));
check("A: recall chip reads the score",
  (await chipOf(pRow, "Context recall")).includes("55%"), await chipOf(pRow, "Context recall"));
check("A: faithfulness chip reads the score",
  (await chipOf(pRow, "Stuck to the sources")).includes("93%"), await chipOf(pRow, "Stuck to the sources"));
check("A: relevancy chip reads the score",
  (await chipOf(pRow, "Answered what was asked")).includes("90%"), await chipOf(pRow, "Answered what was asked"));
check("A: precision bar fills to the score",
  (await fillPct(pRow, "Context precision")) === "72%", await fillPct(pRow, "Context precision"));
check("A: recall bar fills to the score",
  (await fillPct(pRow, "Context recall")) === "55%", await fillPct(pRow, "Context recall"));
check("A: retrieval tags are known on a bank match",
  (await tagOf(pRow, "Context precision")) === "known" &&
    (await tagOf(pRow, "Context recall")) === "known");
check("A: generation rows carry NO provenance tag",
  (await tagCount(pRow, "Stuck to the sources")) === 0 &&
    (await tagCount(pRow, "Answered what was asked")) === 0);
check("A: the primary panel has no benchmark bars, legend or pills",
  (await page.locator("#eval-scorecard .bench-fill").count()) === 0 &&
    (await page.locator("#eval-scorecard .score-legend").count()) === 0 &&
    (await page.locator("#eval-scorecard .score-pill").count()) === 0);

// The detail, behind the expander, keeps the measured dimension only.
await page.locator("#eval-details summary").click();
await page.waitForTimeout(200);
check("A: detail correctness row is benchmark-only (correctness left the live set)",
  !(await dRow("Matched the expected answer").evaluate((el) => el.classList.contains("has-live"))) &&
    (await tagOf(dRow, "Matched the expected answer")) === "bench" &&
    (await footOf(dRow, "Matched the expected answer")).includes("benchmark 80%"),
  await footOf(dRow, "Matched the expected answer"));
check("A: the four primary-panel families are trimmed from the detail",
  (await dRow("Found the right passages").count()) === 0 &&
    (await dRow("Sent mostly relevant text").count()) === 0 &&
    (await dRow("Stuck to the sources").count()) === 0 &&
    (await dRow("Answered what was asked").count()) === 0,
  `recall=${await dRow("Found the right passages").count()} ` +
    `precision=${await dRow("Sent mostly relevant text").count()} ` +
    `faith=${await dRow("Stuck to the sources").count()} ` +
    `rel=${await dRow("Answered what was asked").count()}`);
// Even with a bank match whose gold payload CARRIES recall/precision@k
// readings (fixture GOLD_RETRIEVAL), the trimmed rows must not resurrect —
// the trim is about the name reading twice, not about missing data.
// Boolean gold readings are states: hit rate (gold 1) renders as a pill on
// the benchmark track.
const hit = await pillInfo("Right passage made the cut");
check("A: hit-rate gold reading renders as a neutral pass pill",
  !!hit && !hit.fail && hit.text.includes("made the cut") && hit.text.includes("✓"),
  hit && hit.text);
check("A: hit-rate row has NO live fill and rides the gray benchmark bar",
  !!hit && hit.fills === 0 && hit.benchFills === 1,
  hit && `fills=${hit.fills} benchFills=${hit.benchFills}`);
await page.locator("#eval-details summary").click();
await page.waitForTimeout(200);

// ---------- B. drafted-reference: retrieval estimated ----------
await clickChip("m2");
check("B: retrieval tags are estimated on a drafted reference",
  (await tagOf(pRow, "Context precision")) === "estimated" &&
    (await tagOf(pRow, "Context recall")) === "estimated",
  `${await tagOf(pRow, "Context precision")} / ${await tagOf(pRow, "Context recall")}`);
check("B: generation rows still carry no tag",
  (await tagCount(pRow, "Stuck to the sources")) === 0);
check("B: the readings themselves still land",
  (await chipOf(pRow, "Context precision")).includes("72%"));

// ---------- C. pending + bank: the rows wait ----------
await clickChip("m3");
check("C: all four rows wait on the judge",
  (await chipOf(pRow, "Context precision")).includes("grading") &&
    (await chipOf(pRow, "Context recall")).includes("grading") &&
    (await chipOf(pRow, "Stuck to the sources")).includes("grading") &&
    (await chipOf(pRow, "Answered what was asked")).includes("grading"));
check("C: waiting rows still carry the bank tag (that is what they wait for)",
  (await tagOf(pRow, "Context precision")) === "known");
check("C: waiting rows show no phantom fill",
  (await fillPct(pRow, "Context precision")) === "0%");

// ---------- D. pending + drafted, mixed: a real sliced run ----------
await clickChip("m4");
check("D: landed generation readings show while the retrieval pair waits",
  (await chipOf(pRow, "Stuck to the sources")).includes("88%") &&
    (await chipOf(pRow, "Context precision")).includes("grading"),
  `${await chipOf(pRow, "Stuck to the sources")} / ${await chipOf(pRow, "Context precision")}`);
check("D: drafted provenance shows before the reading does",
  (await tagOf(pRow, "Context precision")) === "estimated");

// ---------- E. binary judge verdict without a score ----------
await clickChip("m5");
check("E: a scoreless pass reads the word, not a 100% claim",
  (await chipOf(pRow, "Stuck to the sources")).includes("passed") &&
    !(await chipOf(pRow, "Stuck to the sources")).includes("93%"),
  await chipOf(pRow, "Stuck to the sources"));
check("E: a scoreless pass fills the binary fallback",
  (await fillPct(pRow, "Stuck to the sources")) === "100%");
check("E: a scoreless fail reads the word and empties the bar",
  (await chipOf(pRow, "Answered what was asked")).includes("failed") &&
    (await fillPct(pRow, "Answered what was asked")) === "0%");
check("E: ungraded retrieval rows show a dash with no tag (no reference yet)",
  (await chipOf(pRow, "Context precision")).includes("—") &&
    (await tagCount(pRow, "Context precision")) === 0);
check("E: exhausted partial grading is explicitly unavailable, not pending",
  (await page.locator(".eval-judge-error").innerText()).toLowerCase().includes("unavailable") &&
    !(await chipOf(pRow, "Stuck to the sources")).includes("grading"));

// ---------- I. a graded definitional verdict explains itself ----------
await clickChip("m9");
check("I: the definitional 0% renders as a score, not a state word",
  (await chipOf(pRow, "Context recall")).includes("0%") &&
    !(await chipOf(pRow, "Context recall")).includes("failed"),
  await chipOf(pRow, "Context recall"));
check("I: the definitional verdict carries its reason on hover",
  (await pRow("Context recall").locator(".score-val")
    .getAttribute("title") || "").includes("no reference derivable"));
check("I: healthy rows stay tooltip-quiet",
  ((await pRow("Stuck to the sources").locator(".score-val")
    .getAttribute("title")) || "") === "");

// ---------- F–H. the gold dimension of the detail, behind the expander ----------
await page.locator("#eval-details summary").click();
await page.waitForTimeout(200);
await clickChip("m6");
check("F: not-found row reads the refusal verdict",
  (await chipOf(dRow, "Admitted when it could not answer")).includes("refused"),
  await chipOf(dRow, "Admitted when it could not answer"));
check("F: refusal row is known",
  (await tagOf(dRow, "Admitted when it could not answer")) === "known");
const refusedPill = await pillInfo("Admitted when it could not answer");
check("F: refusal renders as a NEUTRAL pass pill on the track",
  !!refusedPill && !refusedPill.fail && refusedPill.text.includes("refused") &&
    refusedPill.text.includes("✓"),
  refusedPill && refusedPill.text);
check("F: refusal row has NO live fill and rides the gray benchmark bar",
  !!refusedPill && refusedPill.fills === 0 && refusedPill.benchFills === 1,
  refusedPill && `fills=${refusedPill.fills} benchFills=${refusedPill.benchFills}`);

await clickChip("m7");
const missedPill = await pillInfo("Admitted when it could not answer");
check("G: a missed refusal renders as a FAIL pill",
  !!missedPill && missedPill.fail && missedPill.text.includes("missed") &&
    missedPill.text.includes("✗"),
  missedPill && missedPill.text);
check("G: foot says the answer should have refused",
  (await footOf(dRow, "Admitted when it could not answer")).includes("should have refused"),
  await footOf(dRow, "Admitted when it could not answer"));

await clickChip("m8");
const cutPill = await pillInfo("Right passage made the cut");
check("H: a cut hit rate renders as a FAIL pill",
  !!cutPill && cutPill.fail && cutPill.text.includes("cut") &&
    !cutPill.text.includes("made") && cutPill.text.includes("✗"),
  cutPill && cutPill.text);
check("H: cut chip says the verdict, not 0%",
  !(await chipOf(dRow, "Right passage made the cut")).includes("0%") &&
    (await chipOf(dRow, "Right passage made the cut")).includes("cut"),
  await chipOf(dRow, "Right passage made the cut"));
check("H: foot keeps the golden pointer (unchanged by design)",
  (await footOf(dRow, "Right passage made the cut")).includes("golden #72"),
  await footOf(dRow, "Right passage made the cut"));
const mrrRow = dRow("Best passage ranked high");
check("H: MRR at a true 0% keeps its bar (a ratio, not a state)",
  (await mrrRow.locator(".score-fill").count()) === 1 &&
    (await mrrRow.locator(".score-pill").count()) === 0,
  `fills=${await mrrRow.locator(".score-fill").count()} pills=${await mrrRow.locator(".score-pill").count()}`);

// ---------- colour legend (detail only) ----------
// Two swatches, named once: gray = benchmark, lime = this answer.
const legend = page.locator("#eval-benchmark .score-legend");
check("legend names the benchmark and this answer",
  lower(await legend.innerText()).includes("benchmark") &&
    lower(await legend.innerText()).includes("this answer"),
  await legend.innerText());
check("legend carries exactly two swatches with the two bar colours",
  (await legend.locator(".score-legend-swatch.bench").count()) === 1 &&
    (await legend.locator(".score-legend-swatch.live").count()) === 1,
  `bench=${await legend.locator(".score-legend-swatch.bench").count()} live=${await legend.locator(".score-legend-swatch.live").count()}`);
// The two colours, in BOTH themes — the legend is the only place the two bar
// colours are named, so it has to carry the right token in each theme.
// Light: --bench #98a2b3 / --primary #a8e01f. Dark: --bench #46536a /
// --primary #c2f53f.
for (const [theme, benchRGB, liveRGB] of [
  ["dark", "rgb(70, 83, 106)", "rgb(194, 245, 63)"],
  ["light", "rgb(152, 162, 179)", "rgb(168, 224, 31)"],
]) {
  const benchColor = await legend.locator(".score-legend-swatch.bench")
    .evaluate(async (el, t) => {
      document.documentElement.setAttribute("data-theme", t);
      await new Promise((r) => requestAnimationFrame(r));
      return getComputedStyle(el).backgroundColor;
    }, theme);
  const liveColor = await legend.locator(".score-legend-swatch.live")
    .evaluate((el) => getComputedStyle(el).backgroundColor);
  check(`legend swatches carry the right tokens in ${theme} theme`,
    benchColor === benchRGB && liveColor === liveRGB,
    `bench=${benchColor} live=${liveColor}`);
}
await page.evaluate(() =>
  document.documentElement.setAttribute("data-theme", "dark"));

// ---------- mobile grading lifecycle at the reader breakpoint ----------
// This is intentionally a second, isolated scenario: the desktop provenance
// assertions above use stored answers, while this pass drives the ordinary ask
// path and the deferred grade endpoint through every reader-visible state.
const mobileTab = (name) => page.locator(`.tab[data-tab="${name}"]`);
const evalBadge = () => page.locator("#tab-count-eval");
const selectedTab = async (name) =>
  (await mobileTab(name).getAttribute("aria-selected")) === "true";
const badgeHidden = async () => (await evalBadge().getAttribute("class"))?.includes("hidden");
await page.setViewportSize({ width: 375, height: 812 });
await page.waitForTimeout(350);
await page.locator('.msg[data-message-id="m5"] .eval-chip').click();
check("mobile grade_exhausted stays terminal and exposes the outage",
  await selectedTab("eval") &&
    (await page.locator(".eval-judge-error").innerText()).toLowerCase().includes("unavailable"));
await mobileTab("chat").click();
mobileGradeResponses.set("m10", [
  // /ask already returned the pending state. The separate /grade queue starts
  // with a partial response, then settles on the next bounded retry.
  { eval: {
    pending: false, faithful: true, faithful_score: 0.88,
    relevant: true, relevant_score: 0.91,
    // Both retrieval readings are absent in this partial response. It is not a
    // settled grade and must remain retryable until the bounded run settles.
    context_precision: null, context_recall: null,
    expected_source: "draft", latency_ms: 900, top_sim: 0.61,
    retry_after_ms: 250,
  } },
  { eval: {
    pending: false, faithful: true, faithful_score: 0.88,
    relevant: true, relevant_score: 0.91,
    context_precision: true, context_precision_score: 0.73,
    context_recall: false, context_recall_score: 0.52,
    expected_source: "draft", latency_ms: 900, top_sim: 0.61, deep_n: 0,
    grade_ms: 260,
  } },
]);

async function askOnMobile(id, question) {
  // Keep both mobile submission branches honest. Chat is visible for the
  // native requestSubmit path; when Eval owns the phone, invoke the existing
  // submit handler directly with a real event because Playwright rejects clicks
  // on the hidden chat pane.
  const evalVisible = await selectedTab("eval");
  const before = mobileAskCalls.length;
  if (evalVisible) {
    // fill({ force: true }) silently no-ops on the display:none chat-pane
    // input, and the submit handler early-returns on an empty question. Set
    // the value directly so the handler sees the real question, then invoke
    // the same submit event a reader's Enter key would produce.
    await page.locator("#question-input").evaluate((el, q) => {
      el.value = q;
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, question);
    await page.locator("#ask-form").evaluate((form) => {
      form.dispatchEvent(new SubmitEvent("submit", {
        bubbles: true,
        cancelable: true,
        submitter: form.querySelector("#ask-btn"),
      }));
    });
  } else {
    await page.locator("#question-input").fill(question);
    await page.locator("#ask-form").evaluate((form) => form.requestSubmit());
  }
  await page.locator(`.msg[data-message-id="${id}"]`).waitFor({ state: "attached" });
  await page.locator(`.msg[data-message-id="${id}"] .eval-block`).waitFor({ state: "attached" });

  const calls = mobileAskCalls.slice(before);
  check(`mobile ${id}: ${evalVisible ? "hidden Eval fallback" : "visible Chat requestSubmit"} makes exactly one request and response`,
    calls.length === 1 && calls[0].response?.message_id === id,
    `requests=${calls.length} responses=${calls.filter((call) => call.response).length}`);
}

await askOnMobile("m10", "What does the ordinary ask path return?");
check("mobile pending: ordinary ask keeps the reader on CHAT",
  await selectedTab("chat") && !(await selectedTab("eval")));
check("mobile pending: no attention marker before a settled reading",
  await badgeHidden(), await evalBadge().getAttribute("class"));
const m10Chip = page.locator('.msg[data-message-id="m10"] .eval-state');
check("mobile pending: answer chip says grading",
  (await m10Chip.innerText()).toLowerCase().includes("grading"), await m10Chip.innerText());

// The pending chip is asserted while the partial grade is still gated above;
// only now is it allowed to arrive.
releaseM10Partial();

await m10Chip.waitFor({ state: "visible" });
// The partial response can already say "grounded" because its generation
// readings landed first. Retrieval dashes are the observable boundary that
// distinguishes that retryable state from the later settled grade.
await page.waitForFunction(() => {
  const chip = document.querySelector('.msg[data-message-id="m10"] .eval-state');
  const rows = [...document.querySelectorAll("#eval-scorecard .score-row")];
  const reading = (label) => rows.find((row) => row.textContent.includes(label))
    ?.querySelector(".score-val")?.textContent;
  return chip?.textContent.toLowerCase().includes("grounded") &&
    reading("Context precision")?.includes("—") &&
    reading("Context recall")?.includes("—");
});
check("mobile partial: faithful/relevant state remains honest without a partial label",
  (await m10Chip.innerText()).toLowerCase().includes("grounded") &&
    await selectedTab("chat") && await badgeHidden(), await m10Chip.innerText());
check("mobile partial: missing precision and recall are not settled",
  (await chipOf(pRow, "Context precision")).includes("—") &&
    (await chipOf(pRow, "Context recall")).includes("—") &&
    await badgeHidden(),
  `${await chipOf(pRow, "Context precision")} / ${await chipOf(pRow, "Context recall")}`);

// Release the queued settled response while the reader is STILL on CHAT. The
// gate is explicit, so this cannot race the tab. This is the pin for the
// delayed grade path: a settled grade arriving mid-read must not yank the
// reader to Eval — it must raise exactly one unread marker instead.
releaseM10Settled();
// Do not key this wait off the verdict chip: "grounded" is also the partial
// generation verdict. A non-dash retrieval reading proves the queued grade
// has settled.
await page.waitForFunction(() => {
  const rows = [...document.querySelectorAll("#eval-scorecard .score-row")];
  const reading = (label) => rows.find((row) => row.textContent.includes(label))
    ?.querySelector(".score-val")?.textContent;
  const precision = reading("Context precision");
  const recall = reading("Context recall");
  return precision && recall && !precision.includes("—") && !recall.includes("—");
});
check("mobile settled arrival: reader stays on CHAT with one Eval marker",
  await selectedTab("chat") && !(await badgeHidden()) &&
    (await evalBadge().innerText()) === "1",
  `tab=${await selectedTab("chat")} badge=${await evalBadge().innerText()}`);

// Opening Eval consumes the marker; the settled readings are already on the
// scorecard under the selected answer.
await mobileTab("eval").click();
await page.waitForTimeout(100);
check("mobile Eval opening clears its attention marker",
  await selectedTab("eval") && await badgeHidden());
check("mobile settled: retrieval readings arrive under the selected answer",
  (await chipOf(pRow, "Context precision")).includes("73%") &&
    (await chipOf(pRow, "Context recall")).includes("52%"),
  `${await chipOf(pRow, "Context precision")} / ${await chipOf(pRow, "Context recall")}`);

// A settled reading that arrives while Eval is ALREADY selected is consumed in
// place: it must not create a marker for a pane the reader is looking at. This
// ask drives the hidden-composer fallback submission path while it is at it.
await askOnMobile("m11", "Ask again after the scorecard check.");
check("mobile settled while Eval selected: consumed in place, no marker",
  await selectedTab("eval") && await badgeHidden(),
  `tab=${await selectedTab("eval")} badge=${await evalBadge().innerText()}`);

// Terminal backend states are not retryable and are shown as data/status gaps,
// never as a false binary verdict or an endless pending chip. They are asked
// from CHAT deliberately: on EVAL the unread reset would mask a broken
// notification predicate, while from CHAT a false positive would raise the
// marker and fail these checks. The hidden-composer fallback is already
// exercised above by m11.
await mobileTab("chat").click();
await askOnMobile("m12", "Show an unavailable grading result.");
check("mobile grade_unavailable is terminal, explained, and unannounced",
  (await page.locator('.msg[data-message-id="m12"] .eval-judge-error').innerText())
    .toLowerCase().includes("judge disabled") && await badgeHidden());
await askOnMobile("m13", "Show a precision data gap.");
check("mobile precision_data_gap is terminal without a retry marker",
  (await page.locator('.msg[data-message-id="m13"] .eval-state').innerText())
    .toLowerCase().includes("grounded") &&
    (await chipOf(pRow, "Context precision")).includes("—") &&
    await badgeHidden());
await askOnMobile("m15", "Show an exhausted grading run.");
check("mobile grade_exhausted is terminal, explained, and unannounced",
  (await page.locator('.msg[data-message-id="m15"] .eval-judge-error').innerText())
    .toLowerCase().includes("unavailable") && await badgeHidden());
// One final ordinary ask. Its visible composer and settled response must
// create exactly one unread Eval marker.
await askOnMobile("m14", "Ask once more from Chat for the unread marker.");
check("mobile ordinary ask: settled arrival creates exactly one Eval marker",
  await selectedTab("chat") && !(await badgeHidden()) &&
    (await evalBadge().innerText()) === "1",
  `tab=${await selectedTab("chat")} badge=${await evalBadge().innerText()}`);
check("mobile asks consume exactly one response each",
  mobileAskCalls.length === 6 && mobileAskCalls.every((call, i) =>
    call.response?.message_id === ["m10", "m11", "m12", "m13", "m15", "m14"][i]),
  mobileAskCalls.map((call) => call.response?.message_id || "no response").join(", "));

// Chip navigation is the intentional compare affordance: it may move from the
// reader tab to Eval, unlike a deferred response which must never do so.
await page.locator('.msg[data-message-id="m14"] .eval-chip').click();
await page.waitForTimeout(100);
check("mobile direct scorecard chip navigation opens Eval",
  await selectedTab("eval") &&
    (await chipOf(pRow, "Context precision")).includes("76%"));

// ---------- layout: no horizontal overflow at desktop and 375px ----------
for (const [w, h, name] of [[1600, 900, "desktop"], [375, 812, "375px"]]) {
  await page.setViewportSize({ width: w, height: h });
  await page.waitForTimeout(400);
  const over = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  check(`${name}: no horizontal overflow`, over <= 0, `overflow=${over}px`);
}

check("no page errors", pageErrors.length === 0, pageErrors.join(" | "));
check("no console errors", consoleErrors.length === 0, consoleErrors.join(" | "));

await browser.close();
if (fails.length) {
  console.log(`\nFAILED (${fails.length}): ${fails.join("; ")}`);
  process.exit(1);
}
console.log("\nAll scorecard provenance checks passed.");
