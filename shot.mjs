/**
 * Screenshot capture for the frontend — four breakpoints × both themes.
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
 * Output lands in shots/<theme>/<breakpoint>.png.
 *
 * The theme is forced by writing localStorage before the app boots, which is
 * the same key the no-flash inline script in index.html reads. Reloading after
 * the write is required — the theme is applied once at document start.
 */
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";

const BASE = process.argv[2] || "http://localhost:5173";
const THEME_KEY = "ragchat-theme";

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

for (const theme of THEMES) {
  await mkdir(`shots/${theme}`, { recursive: true });

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

    try {
      await page.goto(BASE, { waitUntil: "domcontentloaded" });
      await page.evaluate(
        ([key, value]) => localStorage.setItem(key, value),
        [THEME_KEY, theme],
      );
      await page.reload({ waitUntil: "networkidle" });

      const applied = await page.evaluate(() =>
        document.documentElement.getAttribute("data-theme"),
      );
      if (applied !== theme) {
        console.warn(`  ! ${theme}/${bp.name}: data-theme is "${applied}"`);
        failures++;
      }

      await page.screenshot({
        path: `shots/${theme}/${bp.name}.png`,
        fullPage: bp.width < 768, // mobile stacks and scrolls; capture all of it
      });

      // No horizontal scroll at any breakpoint is a hard rule in the plan.
      const overflow = await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      );
      const overflowNote = overflow > 0 ? `  ⚠ overflows by ${overflow}px` : "";
      console.log(`  ✓ ${theme}/${bp.name}${overflowNote}`);
      if (overflow > 0) failures++;

      for (const e of errors) {
        console.warn(`  ! ${theme}/${bp.name} page error: ${e}`);
        failures++;
      }
    } catch (e) {
      console.error(`  ✗ ${theme}/${bp.name}: ${e.message}`);
      failures++;
    } finally {
      await context.close();
    }
  }
}

await browser.close();
console.log(failures ? `\n${failures} problem(s) found.` : "\nAll clean.");
process.exit(failures ? 1 : 0);
