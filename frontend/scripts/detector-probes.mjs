/**
 * Synthetic pages that each trip exactly one of the responsive harness's
 * detectors, plus one page that must trip none of them.
 *
 * These live in their own file for two reasons. `responsive-acceptance.mjs`
 * was already past the 1,000-line ceiling `CLAUDE.md` sets, and fixture data
 * is not harness logic — keeping the two apart means the probes can be read as
 * what they are: the smallest page that reproduces each defect this harness
 * exists to find.
 *
 * Why they exist at all: a harness that reports green is making two claims,
 * "the app is fine" and "I looked", and only the first was ever checked.
 * VLR-074 shipped twice behind a green run. A detector that never fires is
 * indistinguishable from a page with no defects, and nothing in the harness
 * could tell those apart.
 */

const filler = "a".repeat(64);

export const detectorProbes = [
  {
    /*
     * VLR-074 in miniature: a control inside a box that hides what does not
     * fit, with no scrollbar to recover it. The document is not wider, is not
     * taller, and nothing paints over the button.
     */
    name: "clippedControls",
    field: "clippedControls",
    html: `<main><div style="height:40px;overflow:hidden"><div style="height:300px"></div><button style="width:120px;height:44px">Send</button></div></main>`,
  },
  {
    // VLR-076: `.color-presets` paints after the colour grid and took the
    // pointer for two thirds of the Blue field without widening anything.
    name: "coveredControls",
    field: "coveredControls",
    html: `<main><button style="width:200px;height:44px">Save</button><div style="position:relative;width:200px;height:44px;margin-top:-44px;background:#fff">presets</div></main>`,
  },
  {
    name: "undersizedControls",
    field: "undersizedControls",
    html: `<main><button style="width:20px;height:20px">x</button></main>`,
  },
  {
    name: "unlabeledControls",
    field: "unlabeledControls",
    html: `<main><input type="text" style="width:120px;height:44px"></main>`,
  },
  {
    name: "duplicateIds",
    field: "duplicateIds",
    html: `<main><span id="twice">one</span><span id="twice">two</span></main>`,
  },
  {
    name: "textProblems",
    field: "textProblems",
    html: `<main><p style="font-size:8px;line-height:1.4">unreadable</p></main>`,
  },
  {
    name: "feedbackIssues",
    field: "feedbackIssues",
    html: `<main><div class="notice" style="width:200px;height:40px">Saved, silently</div></main>`,
  },
  {
    // VLR-077 in kind: a value with no room and no way to reveal the rest.
    name: "overflowIssues",
    field: "overflowIssues",
    html: `<main><div style="width:120px"><code style="display:block;white-space:pre;overflow:hidden">${filler}</code></div></main>`,
  },
  {
    name: "verticalSpillIssues",
    field: "verticalSpillIssues",
    html: `<main><div style="height:20px;overflow:visible"><div style="height:200px;display:block">spilling</div></div></main>`,
  },
  {
    name: "shreddedTextControls",
    field: "shreddedTextControls",
    html: `<main><button style="width:20px;height:200px">Cluster</button></main>`,
  },
  {
    /*
     * The Home hero at 150% and 175% zoom, in miniature: four fact columns in
     * a grid whose tracks are capped at `minmax(0, 1fr)` while the items keep
     * `min-width: auto`, so each item overruns its own track and paints over
     * the one beside it. The document never widens, nothing escapes the
     * viewport, and there is not a single control involved — which is why the
     * whole existing detector set was silent on it.
     */
    name: "textCollisionIssues",
    field: "textCollisionIssues",
    html: `<main><dl style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));width:240px">`
      + `<div><dt>UPTIME</dt><dd>3 days</dd></div>`
      + `<div><dt>VERSION</dt><dd>2.1.0a19</dd></div>`
      + `<div><dt>ACCESS</dt><dd>Administrator</dd></div>`
      + `<div><dt>OPERATING SYSTEM</dt><dd>Ubuntu 24.04</dd></div>`
      + `</dl></main>`,
  },
];

/** An element painted past the right edge of the window. */
export const wideProbe = `<main><div style="width:2000px;height:20px">wide</div></main>`;

/**
 * A named control that hit-tests to the element covering it — the shape
 * VLR-074 takes once the panel stops clipping and starts being painted over.
 */
export const unreachableProbe = `<main><button class="probe" style="width:200px;height:44px">Send</button><div style="position:relative;width:200px;height:44px;margin-top:-44px;background:#fff">over</div></main>`;

/**
 * None of the above. An over-eager detector has to be caught the same way a
 * dead one is, so every check must stay silent on this page.
 */
export const cleanProbe = `<main style="width:700px;font-size:14px;line-height:1.5">`
  + `<p style="font-size:16px;line-height:1.5">Everything on this page fits.</p>`
  + `<div style="height:120px"><button style="width:140px;height:44px;font-size:14px;line-height:1.5">Continue</button></div>`
  + `<div class="notice" role="status" style="width:240px;height:40px">Saved</div>`
  /*
   * The same four-column fact row, laid out so it does not collide: the grid
   * items may shrink and their words may break. An over-eager collision
   * detector would report this, and the clean-page assertion catches that.
   */
  + `<dl style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));width:680px;margin:0">`
  + `<div style="min-width:0;overflow-wrap:anywhere"><dt>UPTIME</dt><dd style="margin:0">3 days</dd></div>`
  + `<div style="min-width:0;overflow-wrap:anywhere"><dt>VERSION</dt><dd style="margin:0">2.1.0a19</dd></div>`
  + `<div style="min-width:0;overflow-wrap:anywhere"><dt>ACCESS</dt><dd style="margin:0">Administrator</dd></div>`
  + `<div style="min-width:0;overflow-wrap:anywhere"><dt>SYSTEM</dt><dd style="margin:0">Ubuntu</dd></div>`
  + `</dl>`
  + `</main>`;
