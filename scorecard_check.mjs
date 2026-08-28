/**
 * Scorecard provenance check — the part of the Evaluation pane a screenshot
 * cannot prove. Eight answers with different grading provenance are served
 * through a mocked API (no backend, no model calls), and the check drives the
 * REAL path a reader does: open the app, click each answer's eval chip, and
 * assert where its correctness reading landed.
 *
 * The state under test:
 *   A. bank-referenced  — expected_source: "bank": the correctness judge
 *      graded against the question's HUMAN known answer, so the reading is
 *      GOLD: it fills the Answer correctness bar, names the golden entry, and
 *      must NOT appear in the estimates strip as "vs drafted ref".
 *   B. drafted-reference — the old behaviour, which must survive untouched:
 *      the reading stays in the strip, the bar stays benchmark-only.
 *   C. pending + bank    — the bar waits for its gold reading; the strip
 *      neither claims it nor waits for it.
 *   D. pending + drafted — the strip's grading indicator covers correctness,
 *      as before.
 *   E. known-unanswerable refusal — the not-found row reads "refused".
 *   F. judge graded binary WITHOUT a score — the old 100/0 fill is now a
 *      state pill: faithfulness ✓ passed, relevancy ✗ failed.
 *   G. known-unanswerable, NOT refused — the ✗ polarity: "missed".
 *   H. answerable, hit rate 0 — the ✗ polarity of the made-the-cut pill;
 *      the other retrieval readings are real ratios that merely LAND on 0
 *      and keep their bars.
 *
 * Boolean live readings (refused/missed, made-the-cut/cut, and the binary
 * judge fallback) render as a pill riding the GRAY benchmark bar: their rows
 * carry a .bench-fill (the benchmark rate) but NO .score-fill (no live
 * magnitude). Ratio rows carry BOTH bars — gray benchmark above lime live. A
 * ratio that lands on 0 or 100 (context recall at 100%, MRR at 0%) is a real
 * magnitude and keeps its live bar — the fixtures pin that distinction.
 * Chips read the VERDICT WORD and tag provenance "known" where they used to
 * read "gold". The legend at the panel bottom names the two bar colours once.
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

// The published-run metrics every bar needs. Values chosen so live readings
// land both above and below their benchmark.
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
  context_relevance: true, context_relevance_score: 0.72,
  context_sufficiency: false, context_sufficiency_score: 0.55,
  correct: true, correct_score: 0.81,
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
    eval_data: { ...GRADED, correct: true, correct_score: undefined,
      gold: undefined, expected_answer: undefined, expected_source: undefined,
      top_sim: 0.55, cited_rank: 2 } },
  // C. pending, bank-referenced
  { id: "m3", role: "assistant", content: "Espresso is 3.20 … [1]", citations: [],
    eval_line: "top sim 0.62 - 1300 ms",
    eval_data: { pending: true, faithful: null, relevant: null,
      context_relevance: null, context_sufficiency: null, correct: null,
      expected_answer: "Espresso is 3.20, cappuccino 4.50 …",
      expected_source: "bank", latency_ms: 1300, top_sim: 0.62, deep_n: 0,
      retry_after_ms: 60_000 } },
  // D. pending, drafted — correctness landed (so the strip carries
  // "vs drafted ref") while a context estimate is still grading (so the
  // strip also carries the grading indicator), which is the mixed state a
  // sliced judge run actually produces.
  { id: "m4", role: "assistant", content: "Espresso is 3.20 … [1]", citations: [],
    eval_line: "top sim 0.58 - 1400 ms",
    eval_data: { pending: true, faithful: null, relevant: null,
      context_relevance: null, context_sufficiency: null,
      correct: true, correct_score: 0.66, cited_rank: 2, pool_n: 4,
      latency_ms: 1400, top_sim: 0.58, deep_n: 0, retry_after_ms: 60_000 } },
  // E. known-unanswerable, refused
  { id: "m5", role: "assistant", content: "I couldn't find an answer…",
    citations: [], eval_line: "top sim 0.31 - 900 ms",
    eval_data: { pending: false, latency_ms: 900, top_sim: 0.31, deep_n: 0,
      gold: { idx: 70, src: "demo",
        question: "Does Meridian sell gift cards?",
        unanswerable: true, refused: true } } },
  // F. judge graded binary WITHOUT a score — the fallback that used to draw
  // a 100/0 fill. Faithfulness passed; relevancy failed.
  { id: "m6", role: "assistant", content: "Espresso is 3.20 … [1]", citations: [],
    eval_line: "top sim 0.60 - 1000 ms",
    eval_data: { pending: false, faithful: true, relevant: false,
      correct: null, context_relevance: null, context_sufficiency: null,
      cited_rank: 1, pool_n: 4, latency_ms: 1000, top_sim: 0.60, deep_n: 0 } },
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
await page.route("**/api/chats/c1**", (route) => {
  const url = route.request().url();
  if (url.includes("/grade")) {
    const mid = url.split("/messages/")[1]?.split("/")[0];
    const m = MESSAGES.find((x) => x.id === mid);
    return route.fulfill({ json: { eval: m?.eval_data || {}, eval_line: m?.eval_line || "" } });
  }
  return route.fulfill({ json: { id: "c1", title: "Demo chat", messages: MESSAGES } });
});
await page.route("**/api/eval/baseline", (route) => route.fulfill({ json: { baseline: {} } }));
await page.route("**/api/eval", (route) => route.fulfill({ json: EVAL_DONE }));

await page.goto(`${BASE}/app.html`, { waitUntil: "networkidle" });
await page.waitForSelector(".score-row", { timeout: 15_000 });
await page.waitForTimeout(300);

const rowText = (label) =>
  page.locator(".score-row", { hasText: label });

// innerText returns RENDERED text and the pane styles tags/strip labels with
// text-transform: uppercase — so every comparison below is case-folded. The
// assertions are about provenance, not about a stylesheet.
const lower = (s) => s.trim().toLowerCase();

async function chipOf(label) {
  return lower(await rowText(label).locator(".score-val").innerText());
}
async function tagOf(label) {
  return lower(await rowText(label).locator(".score-tag").innerText());
}
async function footOf(label) {
  return lower(await rowText(label).locator(".score-foot").innerText());
}
async function fillPct(label) {
  return rowText(label).locator(".score-fill").evaluate((el) => el.style.width);
}
// The state pill a boolean live reading renders as, plus the two things that
// must hold on its row: NO live fill (the pill is the state mark, not a
// magnitude), and the gray benchmark bar still present for it to ride.
async function pillInfo(label) {
  const row = rowText(label);
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

const rowA = rowText("Matched the expected answer");

// ---------- A. bank-referenced: gold on the bar, out of the strip ----------
await clickChip("m1");
check("A: correctness bar fills with the judge score",
  (await fillPct("Matched the expected answer")) === "81%",
  `width=${await fillPct("Matched the expected answer")}`);
check("A: correctness row is live", await rowA.evaluate((el) => el.classList.contains("has-live")));
check("A: chip reads the percentage", (await chipOf("Matched the expected answer")).includes("81%"),
  await chipOf("Matched the expected answer"));
check("A: tag is known", (await tagOf("Matched the expected answer")) === "known",
  await tagOf("Matched the expected answer"));
const footA = await footOf("Matched the expected answer");
check("A: footer names the known answer and the golden entry",
  footA.includes("known answer") && footA.includes("golden #64"), footA);
check("A: footer does not claim passage measurement",
  !footA.includes("known passages"), footA);
const estA = lower(await page.locator(".score-est").innerText().catch(() => ""));
check("A: strip has no 'vs drafted ref'", !estA.includes("vs drafted ref"), estA);
check("A: strip still carries the cited-rank estimate", estA.includes("cited"), estA);
// Gold retrieval rows still measured, from the same answer's gold entry.
check("A: retrieval recall row stays gold-measured",
  (await footOf("Found the right passages")).includes("golden #64") &&
    (await footOf("Found the right passages")).includes("known passages"),
  await footOf("Found the right passages"));
check("A: faithfulness row is judged",
  (await tagOf("Stuck to the sources")) === "judged", await tagOf("Stuck to the sources"));
// Boolean gold readings are states: hit rate (gold 1) renders as a pill on
// the benchmark track; a ratio that merely LANDS on 100% keeps its bar.
const hit = await pillInfo("Right passage made the cut");
check("A: hit-rate gold reading renders as a neutral pass pill",
  !!hit && !hit.fail && hit.text.includes("made the cut") && hit.text.includes("✓"),
  hit && hit.text);
check("A: hit-rate row has NO live fill and rides the gray benchmark bar",
  !!hit && hit.fills === 0 && hit.benchFills === 1,
  hit && `fills=${hit.fills} benchFills=${hit.benchFills}`);
check("A: hit-rate chip says the verdict, not 100%",
  !(await chipOf("Right passage made the cut")).includes("100%") &&
    (await chipOf("Right passage made the cut")).includes("made the cut"),
  await chipOf("Right passage made the cut"));
check("A: hit-rate foot keeps the golden pointer (unchanged by design)",
  (await footOf("Right passage made the cut")).includes("golden #64"),
  await footOf("Right passage made the cut"));
const recallRow = rowText("Found the right passages");
check("A: recall at a true 100% keeps its live bar (a ratio, not a state)",
  (await recallRow.locator(".score-fill").count()) === 1 &&
    (await recallRow.locator(".score-pill").count()) === 0 &&
    (await recallRow.locator(".bench-fill").count()) === 1 &&
    (await fillPct("Found the right passages")) === "100%",
  `fills=${await recallRow.locator(".score-fill").count()} pills=${await recallRow.locator(".score-pill").count()} benchFills=${await recallRow.locator(".bench-fill").count()}`);

// ---------- B. drafted-reference: the strip keeps it, the bar stays benchmark ----------
await clickChip("m2");
check("B: correctness bar does NOT fill",
  !(await rowA.evaluate((el) => el.classList.contains("has-live"))));
check("B: correctness tag is bench", (await tagOf("Matched the expected answer")) === "bench",
  await tagOf("Matched the expected answer"));
check("B: correctness chip is the benchmark", (await chipOf("Matched the expected answer")).includes("80%"),
  await chipOf("Matched the expected answer"));
// No live reading: the row shows the gray benchmark bar alone, dimmed.
check("B: bench-only row dims its single gray bar",
  (await rowA.locator(".bench-fill").count()) === 1 &&
    (await rowA.locator(".score-fill").count()) === 0 &&
    (await rowA.locator(".score-bar.no-live").count()) === 1,
  `benchFills=${await rowA.locator(".bench-fill").count()} fills=${await rowA.locator(".score-fill").count()} noLive=${await rowA.locator(".score-bar.no-live").count()}`);
const estB = lower(await page.locator(".score-est").innerText());
check("B: strip carries 'vs drafted ref'", estB.includes("vs drafted ref"), estB);
check("B: drafted binary verdict renders as 100%", estB.includes("100%"), estB);

// ---------- C. pending + bank: the bar waits, the strip does not ----------
await clickChip("m3");
check("C: correctness waits on the judge",
  (await chipOf("Matched the expected answer")).includes("grading"),
  await chipOf("Matched the expected answer"));
check("C: waiting row is tagged known (that is the reading it waits for)",
  (await tagOf("Matched the expected answer")) === "known",
  await tagOf("Matched the expected answer"));
const estC = lower(await page.locator(".score-est").innerText());
check("C: strip has no 'vs drafted ref' while waiting", !estC.includes("vs drafted ref"), estC);

// ---------- D. pending + drafted: unchanged behaviour ----------
await clickChip("m4");
check("D: correctness row still shows benchmark while drafted grading pends",
  (await chipOf("Matched the expected answer")).includes("80%"),
  await chipOf("Matched the expected answer"));
const estD = lower(await page.locator(".score-est").innerText());
check("D: strip carries 'vs drafted ref' and the grading indicator",
  estD.includes("vs drafted ref") && estD.includes("grading"), estD);

// ---------- E. known-unanswerable refusal ----------
await clickChip("m5");
check("E: not-found row reads the refusal verdict",
  (await chipOf("Admitted when it could not answer")).includes("refused"),
  await chipOf("Admitted when it could not answer"));
check("E: refusal row is known",
  (await tagOf("Admitted when it could not answer")) === "known",
  await tagOf("Admitted when it could not answer"));
const refusedPill = await pillInfo("Admitted when it could not answer");
check("E: refusal renders as a NEUTRAL pass pill on the track",
  !!refusedPill && !refusedPill.fail && refusedPill.text.includes("refused") &&
    refusedPill.text.includes("✓"),
  refusedPill && refusedPill.text);
check("E: refusal row has NO live fill and rides the gray benchmark bar",
  !!refusedPill && refusedPill.fills === 0 && refusedPill.benchFills === 1,
  refusedPill && `fills=${refusedPill.fills} benchFills=${refusedPill.benchFills}`);

// ---------- F. judge graded binary without a score: pills, not 100/0 fills ----------
await clickChip("m6");
const faithPill = await pillInfo("Stuck to the sources");
check("F: faithfulness binary verdict renders as a pass pill",
  !!faithPill && !faithPill.fail && faithPill.text.includes("passed") &&
    faithPill.text.includes("✓"),
  faithPill && faithPill.text);
check("F: faithfulness row has NO live fill and rides the gray benchmark bar",
  !!faithPill && faithPill.fills === 0 && faithPill.benchFills === 1,
  faithPill && `fills=${faithPill.fills} benchFills=${faithPill.benchFills}`);
check("F: faithfulness tag is judged",
  (await tagOf("Stuck to the sources")) === "judged", await tagOf("Stuck to the sources"));
const relPill = await pillInfo("Answered what was asked");
check("F: relevancy binary verdict renders as a FAIL pill",
  !!relPill && relPill.fail && relPill.text.includes("failed") && relPill.text.includes("✗"),
  relPill && relPill.text);
check("F: relevancy row has NO live fill and rides the gray benchmark bar",
  !!relPill && relPill.fills === 0 && relPill.benchFills === 1,
  relPill && `fills=${relPill.fills} benchFills=${relPill.benchFills}`);
check("F: foot names the benchmark, not a phantom verdict",
  (await footOf("Answered what was asked")).includes("graded against the benchmark"),
  await footOf("Answered what was asked"));

// ---------- G. known-unanswerable, missed: the ✗ polarity ----------
await clickChip("m7");
const missedPill = await pillInfo("Admitted when it could not answer");
check("G: a missed refusal renders as a FAIL pill",
  !!missedPill && missedPill.fail && missedPill.text.includes("missed") &&
    missedPill.text.includes("✗"),
  missedPill && missedPill.text);
check("G: missed row has NO live fill and rides the gray benchmark bar",
  !!missedPill && missedPill.fills === 0 && missedPill.benchFills === 1,
  missedPill && `fills=${missedPill.fills} benchFills=${missedPill.benchFills}`);
check("G: tag is known",
  (await tagOf("Admitted when it could not answer")) === "known",
  await tagOf("Admitted when it could not answer"));
check("G: foot says the answer should have refused",
  (await footOf("Admitted when it could not answer")).includes("should have refused"),
  await footOf("Admitted when it could not answer"));

// ---------- H. hit-rate 0: the ✗ polarity, and a ratio that lands on 0 ----------
await clickChip("m8");
const cutPill = await pillInfo("Right passage made the cut");
check("H: a cut hit rate renders as a FAIL pill",
  !!cutPill && cutPill.fail && cutPill.text.includes("cut") &&
    !cutPill.text.includes("made") && cutPill.text.includes("✗"),
  cutPill && cutPill.text);
check("H: cut row has NO live fill and rides the gray benchmark bar",
  !!cutPill && cutPill.fills === 0 && cutPill.benchFills === 1,
  cutPill && `fills=${cutPill.fills} benchFills=${cutPill.benchFills}`);
check("H: cut chip says the verdict, not 0%",
  !(await chipOf("Right passage made the cut")).includes("0%") &&
    (await chipOf("Right passage made the cut")).includes("cut"),
  await chipOf("Right passage made the cut"));
check("H: foot keeps the provenance — the pill owns the verdict, not the foot",
  (await footOf("Right passage made the cut")).includes("golden #72") &&
    (await footOf("Right passage made the cut")).includes("known passages") &&
    !/\bthis answer\b/.test(await footOf("Right passage made the cut")),
  await footOf("Right passage made the cut"));
check("H: tag is known", (await tagOf("Right passage made the cut")) === "known",
  await tagOf("Right passage made the cut"));
const mrrRow = rowText("Best passage ranked high");
check("H: MRR at a true 0% keeps its bar (a ratio, not a state)",
  (await mrrRow.locator(".score-fill").count()) === 1 &&
    (await mrrRow.locator(".score-pill").count()) === 0,
  `fills=${await mrrRow.locator(".score-fill").count()} pills=${await mrrRow.locator(".score-pill").count()}`);

// ---------- colour legend ----------
// Two swatches, named once: gray = benchmark, lime = this answer. They
// replaced both the per-row "· benchmark N%" foot suffixes and the old prose
// provenance paragraph — the feet and the tags still carry provenance words.
const legend = page.locator(".score-legend");
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
