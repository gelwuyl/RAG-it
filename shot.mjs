/**
 * Screenshot capture for the frontend — both pages × five breakpoints × both themes.
 *
 * Replaces the throwaway script that produced the committed shot_*.png files
 * but was never itself committed. Every visual step in PRODUCT_UX_PLAN.md is
 * meant to be verified this way, so it needs to live in the repo.
 *
 * Playwright is installed at the repo root (no package.json — it was added
 * ad hoc), so run this from the repo root:
 *
 *   node shot.mjs                      # against the Vite dev server
 *   node shot.mjs http://localhost:4173  # against `npm run preview` (built dist)
 *
 * Output lands in shots/<page>/<theme>/<breakpoint>.png, where <page> is
 * "landing" (/) or "app" (/app).
 *
 * The theme is forced by writing localStorage before the app boots, which is
 * the same key the no-flash inline script in index.html reads. Reloading after
 * the write is required — the theme is applied once at document start.
 */
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";

const BASE = process.argv[2] || "http://localhost:5173";
const THEME_KEY = "ragchat-theme";

// Three documents to capture since the routing split: the landing page at "/"
// and the workspace at "/app". In DEV there is no /app rewrite — that lives in
// vercel.json — so the app is served at /app.html. Against `npm run preview`
// or a real deploy, pass the base and both still resolve.
const PAGES = [
  { name: "landing", path: "/" },
  { name: "app", path: "/app.html" },
  // The engineering write-up. Included because it is a real page a visitor can
  // land on, and a page nobody screenshots is a page that breaks quietly.
  { name: "built", path: "/built.html" },
];

const BREAKPOINTS = [
  { name: "desktop", width: 1440, height: 1000 },
  { name: "laptop", width: 1280, height: 860 },
  { name: "tablet", width: 900, height: 1000 },
  { name: "mobile", width: 390, height: 844 },
  { name: "mobile-narrow", width: 360, height: 800 },
];

const THEMES = ["dark", "light"];

const browser = await chromium.launch();
let failures = 0;

for (const page_ of PAGES) {
for (const theme of THEMES) {
  await mkdir(`shots/${page_.name}/${theme}`, { recursive: true });

  for (const bp of BREAKPOINTS) {
    const context = await browser.newContext({
      viewport: { width: bp.width, height: bp.height },
      deviceScaleFactor: 2,
    });
    const page = await context.newPage();

    // Console errors are worth surfacing here: a broken app.js renders as a
    // plausible-looking empty shell, and a screenshot alone would not show it.
    const errors = [];
    page.on("pageerror", (e) => errors.push(String(e)));

    // A CSS syntax error is served as a 500 by Vite and produces NO page
    // error — the document renders, unstyled, and every other check here
    // passes. That shipped an entirely unstyled landing page once: no
    // overflow (nothing is laid out), correct data-theme (set by inline JS),
    // "All clean". Failed responses are now failures.
    page.on("response", (r) => {
      if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`);
    });

    try {
      await page.goto(BASE + page_.path, { waitUntil: "domcontentloaded" });
      await page.evaluate(
        ([key, value]) => localStorage.setItem(key, value),
        [THEME_KEY, theme],
      );
      await page.reload({ waitUntil: "networkidle" });

      const applied = await page.evaluate(() =>
        document.documentElement.getAttribute("data-theme"),
      );
      if (applied !== theme) {
        console.warn(`  ! ${page_.name} ${theme}/${bp.name}: data-theme is "${applied}"`);
        failures++;
      }

      await page.screenshot({
        path: `shots/${page_.name}/${theme}/${bp.name}.png`,
        fullPage: bp.width < 768, // mobile stacks and scrolls; capture all of it
      });

      // Belt and braces on the same failure: assert the stylesheet actually
      // took effect. Vite injects CSS as a JS module in dev, so there is no
      // <link> to inspect — but an unstyled body has a transparent background
      // and the default 16px, and a styled one never does.
      const styled = await page.evaluate(() => {
        const s = getComputedStyle(document.body);
        return s.backgroundColor !== "rgba(0, 0, 0, 0)" && s.backgroundColor !== "";
      });
      if (!styled) {
        console.warn(`  ! ${page_.name} ${theme}/${bp.name}: body has no background — stylesheet did not apply`);
        failures++;
      }

      // A LONE CARD UNDER A FULL ROW.
      //
      // Not a visual bug a screenshot review catches reliably — the page looks
      // fine, it just reads as "and one more thing" instead of as a set. It is
      // what `repeat(auto-fit, minmax(...))` produces whenever the item count
      // and the fitted column count are coprime, which is most of the time:
      // five stage notes fitted four across, four cards fitted three.
      //
      // Rows are derived from actual top offsets rather than from the CSS, so
      // this tests what rendered.
      const orphans = await page.evaluate(() => {
        const bad = [];
        for (const grid of document.querySelectorAll(
          ".stage-notes, .cards, .lessons, .tiles, .platform",
        )) {
          const kids = [...grid.children];
          if (kids.length < 3) continue;
          const rows = new Map();
          for (const k of kids) {
            const top = Math.round(k.getBoundingClientRect().top);
            rows.set(top, (rows.get(top) || 0) + 1);
          }
          const counts = [...rows.values()];
          if (counts.length < 2) continue;
          const last = counts[counts.length - 1];
          const prev = counts[counts.length - 2];
          if (last === 1 && prev >= 3) {
            bad.push(`${grid.className} -> ${counts.join("+")}`);
          }
        }
        return bad;
      });
      for (const o of orphans) {
        console.warn(`  ! ${page_.name} ${theme}/${bp.name}: lone item under a full row — ${o}`);
        failures++;
      }

      // No horizontal scroll at any breakpoint is a hard rule in the plan.
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      );
      const overflowNote = overflow > 0 ? `  ⚠ overflows by ${overflow}px` : "";
      console.log(`  ✓ ${page_.name} ${theme}/${bp.name}${overflowNote}`);
      if (overflow > 0) failures++;

      for (const e of errors) {
        console.warn(`  ! ${page_.name} ${theme}/${bp.name} page error: ${e}`);
        failures++;
      }
    } catch (e) {
      console.error(`  ✗ ${page_.name} ${theme}/${bp.name}: ${e.message}`);
      failures++;
    } finally {
      await context.close();
    }
  }
}
}

await browser.close();
console.log(failures ? `\n${failures} problem(s) found.` : "\nAll clean.");
process.exit(failures ? 1 : 0);
