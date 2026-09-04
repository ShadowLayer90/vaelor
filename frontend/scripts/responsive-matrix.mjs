/**
 * What gets visited, at what size, and at what zoom.
 *
 * The matrix is the harness's claim about coverage, and it is the part that has
 * most often been wrong in a way nothing could see: a hardcoded breakpoint that
 * disagreed with the stylesheet, a hardcoded destination table that went stale,
 * and a "zoom" pass that never shrank a CSS pixel. Each of those made the run
 * green by not looking, so the derivations are kept here together where they
 * can be read against each other.
 */

import { readFile } from 'node:fs/promises';
import { assert, variantLabel } from './responsive-report.mjs';

export const viewports = [
  { name: 'phone', width: 375, height: 812 },
  { name: 'tablet-portrait', width: 768, height: 1024 },
  { name: 'compact-desktop', width: 1024, height: 768 },
  { name: 'desktop-short', width: 1280, height: 720 },
  { name: 'desktop', width: 1440, height: 900 },
  { name: 'desktop-wide', width: 1920, height: 1080 },
];

/*
 * The width at and below which the rail swaps to the mobile bar and folds the
 * last five destinations behind "More" (`@media (max-width: 720px)` in
 * `src/styles/dialogs-responsive.css`).
 *
 * This was hardcoded as 680, which disagreed with the stylesheet for every
 * width in (680, 720]. Nothing caught it because no viewport in the matrix
 * landed in that band until browser zoom started producing them: at 720x450 the
 * rail had already collapsed, so "Cluster" was inside a closed popover — out of
 * the accessibility tree entirely — while the harness looked for it in the bar.
 */
export const MOBILE_RAIL_MAX_WIDTH = 720;

/*
 * Real browser zoom shrinks the CSS viewport; `deviceScaleFactor` only raises
 * the device pixel ratio and leaves every CSS pixel, media query and viewport
 * unit exactly where it was at 100%. The old zoom pass was therefore
 * structurally unable to reach a `dvh` clamp that inverted below 560 CSS px of
 * height, which is what put the Ask Vaelor composer behind a hidden overflow
 * at 150% zoom while this script reported green. Zoom is emulated here the way
 * the browser does it: the window stays the same, the CSS viewport shrinks.
 */
/*
 * 175% is here because a defect was found at it that neither neighbour
 * produces. The Home hero's fact row collided between 721 and 768 CSS pixels,
 * and this matrix visited 640, 720, 853, 960 and 1280 — over the band on one
 * side and under it on the other. Chrome, Edge and Firefox all offer 175% as a
 * step, so it is a width real readers sit at and the only one of the three that
 * reaches that band from a 1280 window.
 *
 * The general point, since this is the second finding of its kind: zoom levels
 * are how this harness reaches the widths between breakpoints, and a gap
 * between two of them is a gap in coverage that reports as a pass.
 */
const zoomLevels = [1.5, 1.75, 2];
const zoomWindows = ['desktop-short', 'desktop', 'desktop-wide'];
export const zoomVariants = zoomLevels.flatMap((zoom) => viewports
  .filter((viewport) => zoomWindows.includes(viewport.name))
  .map((window) => ({
    window,
    zoom,
    viewport: {
      name: `${window.name}-zoom${Math.round(zoom * 100)}`,
      width: Math.round(window.width / zoom),
      height: Math.round(window.height / zoom),
    },
  })));

/*
 * Read the destination names from the registry the app itself renders, rather
 * than keeping a second copy here. The hardcoded table silently went stale the
 * moment the canonical names landed, and a harness that waits for a heading
 * nothing renders fails for a reason that has nothing to do with layout.
 */
const destinationsSource = await readFile(
  new URL('../src/lib/destinations.ts', import.meta.url), 'utf8',
);
export const routes = [...destinationsSource.matchAll(
  /^\s{2}"?([a-z-]+)"?:\s*\{\s*\n\s*page:[^\n]*\n\s*name:\s*"([^"]+)"/gm,
)].map(([, page, name]) => [page, name]);
if (routes.length !== 9) {
  throw new Error(
    `expected 9 destinations from destinations.ts, parsed ${routes.length}`,
  );
}

export function destinationName(page) {
  const entry = routes.find(([candidate]) => candidate === page);
  if (!entry) throw new Error(`unknown destination ${page}`);
  return entry[1];
}

export async function setViewport(page, variant) {
  await page.setViewportSize({ width: variant.viewport.width, height: variant.viewport.height });
  const probe = await page.evaluate(() => ({
    density: Number.parseFloat(window.getComputedStyle(document.documentElement).zoom),
    devicePixelRatio: window.devicePixelRatio,
    height: window.innerHeight,
    width: window.innerWidth,
  }));
  assert(Math.abs(probe.width - variant.viewport.width) <= 2 && Math.abs(probe.height - variant.viewport.height) <= 2, `Expected ${variantLabel(variant.viewport, variant.zoom)} CSS pixels, got ${probe.width}x${probe.height}.`);
  assert(Math.abs(probe.density - 1) < 0.01, `CSS zoom must remain 1 at ${variantLabel(variant.viewport, variant.zoom)}, got ${probe.density}.`);
  if (variant.zoom) {
    assert(probe.devicePixelRatio > 1, `Zoomed rendering density was not applied; devicePixelRatio was ${probe.devicePixelRatio}.`);
    assert(probe.width < variant.window.width && probe.height < variant.window.height, `Browser zoom must shrink the CSS viewport below the ${variant.window.width}x${variant.window.height} window; got ${probe.width}x${probe.height}.`);
  }
  return probe;
}
