/**
 * Workspace layout behaviour check — the part shot.mjs cannot see.
 *
 * shot.mjs proves the panes RENDER; this proves they still RESPOND. The column
 * widths are custom properties shared by the grid tracks and the drag handles,
 * which is what makes the handles follow the columns for free — and also what
 * makes it silently breakable: rename a property, or move a handle out of
 * .panes' containing block, and the layout still looks perfect in a screenshot
 * while the handle no longer sits on the border it resizes.
 *
 * The inverted handle earns its own assertion. The Evaluation column is
 * anchored to the right edge, so dragging right must SHRINK it; get the sign
 * wrong and the handle runs away from the cursor, which no static check sees.
 *
 * Usage:  node layout_check.mjs [baseUrl]     (default http://localhost:5173)
 * Exits non-zero on the first failed assertion.
 */
import { chromium } from "playwright";

const BASE = process.argv[2] || "http://localhost:5173";
const fails = [];

function check(label, ok, detail = "") {
  console.log(`${ok ? "  ✓" : "  ✗"} ${label}${detail ? ` — ${detail}` : ""}`);
  if (!ok) fails.push(label);
}

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1600, height: 900 } });
const page = await ctx.newPage();
const pageErrors = [];
page.on("pageerror", (e) => pageErrors.push(String(e)));

await page.goto(`${BASE}/app.html`, { waitUntil: "networkidle" });
await page.waitForTimeout(900);

const geom = () =>
  page.evaluate(() => {
    const q = (s) => {
      const r = document.querySelector(s).getBoundingClientRect();
      return { x: Math.round(r.x), w: Math.round(r.width) };
    };
    return {
      sources: q(".sources-pane"), chats: q(".chats-pane"),
      chat: q(".chat-pane"), evalp: q(".eval-pane"),
      hSources: q(".resizer-sources"), hChats: q(".resizer-chats"),
      hEval: q(".resizer-eval"),
    };
  });

const drag = async (fromX, dx) => {
  await page.mouse.move(fromX, 400);
  await page.mouse.down();
  await page.mouse.move(fromX + dx, 400, { steps: 10 });
  await page.mouse.up();
};

console.log(`\nlayout @1600  ${BASE}`);
const a = await geom();

check("four columns, left to right",
  a.sources.x < a.chats.x && a.chats.x < a.chat.x && a.chat.x < a.evalp.x,
  `${a.sources.w} | ${a.chats.w} | ${a.chat.w} | ${a.evalp.w}`);

// +4 is the handle's half-width: the 9px hit area straddles the 1px border.
check("sources handle sits on the sources|chats border",
  Math.abs(a.hSources.x + 4 - (a.sources.x + a.sources.w)) <= 1);
check("chats handle sits on the chats|chat border",
  Math.abs(a.hChats.x + 4 - (a.chats.x + a.chats.w)) <= 1);
check("eval handle sits on the chat|eval border",
  Math.abs(a.hEval.x + 4 - a.evalp.x) <= 1);

await drag(a.hSources.x + 4, 100);
const b = await geom();
check("dragging right widens Sources by the drag distance",
  b.sources.w - a.sources.w === 100, `${a.sources.w} -> ${b.sources.w}`);
check("the handle travels with the column it resizes",
  b.hSources.x - a.hSources.x === 100);

await drag(b.hEval.x + 4, 80);
const c = await geom();
check("dragging the inverted eval handle right SHRINKS Evaluation",
  c.evalp.w === a.evalp.w - 80, `${a.evalp.w} -> ${c.evalp.w}`);

await drag(c.hSources.x + 4, 5000);
check("width clamps at its maximum", (await geom()).sources.w === 460);
await drag((await geom()).hSources.x + 4, -5000);
check("width clamps at its minimum", (await geom()).sources.w === 200);

await page.click("#chats-collapse");
await page.waitForTimeout(150);
const collapsed = await geom();
check("collapsing leaves a rail, not a void", collapsed.chats.w === 40);
check("the rail is clickable to get the column back",
  await page.locator("#chats-expand").isVisible());
await page.click("#chats-expand");
await page.waitForTimeout(150);
check("expanding restores the column", (await geom()).chats.w === 216);

const widths = await geom();
await page.reload({ waitUntil: "networkidle" });
await page.waitForTimeout(900);
const after = await geom();
check("widths survive a reload",
  after.sources.w === widths.sources.w && after.evalp.w === widths.evalp.w,
  `${after.sources.w} / ${after.evalp.w}`);

// Below 1100px the columns stack, so there is nothing to drag and the handles
// must be gone rather than floating over a stacked layout.
await page.setViewportSize({ width: 900, height: 900 });
await page.waitForTimeout(200);
check("handles are absent once the columns stack",
  (await page.locator(".resizer-sources").isVisible()) === false);

check("no page errors", pageErrors.length === 0, pageErrors.join("; ") || "none");

await browser.close();
console.log(fails.length ? `\nFAILED: ${fails.join(", ")}\n` : "\nAll clean.\n");
process.exit(fails.length ? 1 : 0);
