/**
 * The Performance dashboard's layout contract (VD-147 spec 5.1 and the owner's
 * "readable charts" decision), measured in a real browser at a 1920 x 1080
 * viewport, for EVERY state the backend builders produce
 * (`src/test/dashboardPayloads.json`, kept byte-equal to them by
 * `tests/test_dashboard_payload_contract.py`).
 *
 * Easy mode, every state (VD-200, the ClusterPerformance boards: "one size
 * for everything on the page"):
 *   - the six tiles have equal height and equal width;
 *   - the chart cards sit four to a row, and every card is the same height
 *     and the same width;
 *   - the tiles and the FIRST chart row end FOLD_SLACK (32 px) above the fold
 *     (1080 px): the font is not bundled, so the margin is what another
 *     machine's wrapping may spend;
 *     the second row may run below it;
 *   - every plot is at least 124 px tall;
 *   - the gridlines of every plot are at least 20 px apart;
 *   - the page has no horizontal overflow;
 *   - a name the fixed regions shorten (a tile's title, a legend's machine, a
 *     chip, a card's title) keeps its full text as its title (the cut is
 *     visual only).
 *
 * Advanced mode, every state: the three Advanced panels (package power, GTT
 * used, GPU edge temperature) are there, each plot they draw is at least
 * 124 px tall with gridlines at least 20 px apart, every card is the height
 * of the others, the page has no horizontal overflow and a shortened name
 * keeps its full text as its title. The fold does not apply in Advanced.
 *
 * Every clause is held in every state: there is no recorded deviation.
 *
 * Fleet machine tiles (`scripts/dashboard-contract/fleet.html`, the real Fleet
 * tab in Advanced with the worker's card - a `.cf-card` - opened by its foot
 * button, "Cluster connection, trends and machine actions"), at 1920, 1440,
 * 1100, 900, 768 and 375 px wide, for each power sensor word the backend sends
 * ("graphics engine", "board", none) and once more under a long machine name:
 * the opened card draws seven telemetry tiles (StatTile, `.ui-stat-tile` in
 * `.cf-telemetry__tiles`); every tile's trend (`.ui-trend`) keeps its full
 * TREND_HEIGHT (22 px) with every bar at least 1 px wide, never squeezed by the
 * row beside it - the StatTile draws no icon badge, so this is the clause the
 * old round 38 px badge carried; no tile title runs past two lines; no reading
 * runs under its trend; and the page has no horizontal overflow.
 *
 * Deployments row (the same page on its Deployments tab, `deploy=1`: a split
 * with both fence notes and a long LAN endpoint), at 1920, 1440, 1280, 1100,
 * 1024, 900, 768, 600, 506, 480 and 375 px wide. The row is the first row of
 * the model's group in the deployments table (`tbody.cl-dep-group`,
 * ClusterDeployments.tsx) and its notes are Notice banners in the rows beneath
 * it. The endpoint cell is one line and carries its full text as its title,
 * the type tag and the status pill are one line each, the name keeps at least
 * 160 px (a row once left it a few pixels and broke it letter by letter), both
 * notes show uncut, and the page has no horizontal overflow. The table scrolls
 * inside its own box, which the overflow clause skips, so no scroll box in the
 * tab may hide content sideways either (a single-line value in a `code` box,
 * shown beside its own Copy button, is the one kind allowed to). The same
 * clauses hold at 1220 and 1200 px with a long IPv6 endpoint
 * (`endpoint=long`), where the endpoint track once kept its whole width and
 * left the name less than 160 px. A table 900 px or wider keeps the row on one
 * line (the endpoint to the right of the name, on the same line). And at 1440,
 * 1024, 768, 600, 506, 480 and 375 px the same clauses hold for a live-like tab
 * (`live=1`: the row serving, the endpoint cards filled), where the tab once
 * scrolled sideways (W4-D2); a failure names the boxes past the edge.
 *
 * API keys (B1, LESSONS 3): the endpoint cards and their keys show under the
 * Deployments tab's Models filter, so at every live width the contract also
 * picks Models and measures again: no horizontal overflow, no scroll box hiding
 * content sideways, two key rows (one line per key, `ul.cl-ep-keys > li`,
 * EndpointsPanel.tsx) each no taller than KEY_ROW_MAX px, and - on both
 * filters - no button breaking a word across lines. The keys table once passed
 * every clause above at 375 px while its Rotate and Revoke buttons stood one
 * letter per line in 305 px rows, scrolled out of sight inside the table's box.
 *
 * Cluster tab strip (B2): at 768, 600 and 375 px, opened on each of its last
 * two tabs ("Activity", "Agents & tools"), all six tabs are drawn and the
 * selected tab is wholly inside the strip's visible box. Above 720 px the
 * strip scrolls, and a strip with tabs out of sight says so with a cue at that
 * edge (`data-more-start` / `data-more-end` on the list, drawn as a fade); at
 * 720 px and below it wraps onto more rows (styles/cluster.css), which hides
 * nothing, so there no tab may be out of sight at all - and a tab out of sight
 * at an edge with no cue fails at every width. The strip once cut "Activity"
 * and "Agents & tools" at 768 px and below with no cue, and opened on the last
 * tab at 375 px with it out of sight.
 *
 * Confirmation dialog (review round 1): the cluster-link confirmation (the
 * Cluster dialog, role "alertdialog" with the eyebrow "Cluster link"), with its
 * shared-card note, opened from Setup at 375 x 667 and 320 x 667 (short phones)
 * and at 1440 x 900: the note is in the dialog, the dialog lies inside the
 * viewport, and both of its buttons are wholly on screen and at least 44 px
 * tall. The dialog once ran to 912 px on a 667 px screen, with Cancel below
 * the bottom edge.
 *
 * The W4d UI audit's findings (W5-S4, `scripts/dashboard-contract/a11y-pass.mjs`,
 * which lists its clauses): Fleet reflow at 320-390 px, compact controls'
 * height, focus never under the sticky topbar, danger-button contrast, the
 * Assistant's accessible names and its aria references (its own test page,
 * `scripts/dashboard-contract/assistant.html`). `--only=a11y` runs just these.
 * `--only=` takes any list of passes (easy, advanced, fleet, deployments, tabs,
 * dialogs, a11y) and `--a11y=` a list of audit clauses, for a quicker loop
 * while a layout is being fixed (chooseRun); a partial run says so.
 *
 * It starts its own Vite server on the first free local port from 5173 (Vite
 * reads the `port: 0` below as "use the default", not "any port"), serves the
 * test page in `scripts/dashboard-contract/` (the real Cluster page fed by the
 * payloads), measures, stops the server and exits 1 when any held clause
 * fails. Every page is opened through `dashboard-contract/page-load.mjs`: a
 * page left blank by a module request the browser could not complete is
 * loaded once more and the run prints PAGE LOADED AGAIN; a second failure, or
 * a server error, fails the run naming the request (HARNESS-1).
 *
 * The browser (chooseBrowser): VAELOR_ACCEPTANCE_CHROME names a Chrome or
 * Chromium executable; without it the usual install paths are tried. A named
 * file that does not exist fails the run. When no browser is found the run
 * FAILS, unless VAELOR_SKIP_DASHBOARD_CONTRACT=1 is set: then it prints a loud
 * SKIPPED notice and exits 0 (the Linux appliance build sets it when it has no
 * browser). The script measures only when run directly, so a test can import
 * chooseBrowser.
 */

import path from 'node:path';
import process from 'node:process';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { A11Y_CLAUSES, a11yPass, printA11y } from './dashboard-contract/a11y-pass.mjs';
import { LOAD_TIMEOUT_MS, loadNotes, openPage, withLoadNotes } from './dashboard-contract/page-load.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const frontend = path.resolve(here, '..');
const repository = path.resolve(frontend, '..');
const VIEWPORT = { width: 1920, height: 1080 };
const PLOT_MIN = 124;
const GRID_GAP_MIN = 20;
const TOLERANCE = 0.5;
/** Easy mode draws eight panels; Advanced adds three after them. */
const EASY_PANELS = 8;
const ADVANCED_PANELS = 3;
/**
 * How far above the fold row 1 must end (W5 review). The font is the
 * system's (Inter, Segoe UI, system-ui - none bundled), so a line can wrap
 * differently on another machine; row 1 once ended at 1079 of 1080 px, a pass
 * no other font would keep. Two 16 px line pitches is the margin a different
 * wrap of one caption or legend can spend.
 */
const FOLD_SLACK = 32;
/** What a failed fold clause prints. */
const FOLD_REASON = 'LESSONS 13 / W5 review: row 1 must end FOLD_SLACK px above the fold, because the font is not '
  + 'bundled and another machine wraps differently. Do not lower FOLD_SLACK - win the room back in what row 1 says.';

export const CHROME_PATHS = [
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
  '/snap/bin/chromium',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
];

/**
 * Which browser the contract runs on, or why it does not run:
 * `{kind: 'run', path}`, `{kind: 'fail', message}` or `{kind: 'skip', notice}`.
 */
export function chooseBrowser(env, exists, candidates = CHROME_PATHS) {
  const named = env.VAELOR_ACCEPTANCE_CHROME;
  if (named) {
    if (exists(named)) return { kind: 'run', path: named };
    return {
      kind: 'fail',
      message: `Dashboard layout contract: VAELOR_ACCEPTANCE_CHROME is set to ${named}, which does not exist. `
        + 'Point it at a Chrome or Chromium executable.',
    };
  }
  const found = candidates.find((candidate) => exists(candidate));
  if (found) return { kind: 'run', path: found };
  if (env.VAELOR_SKIP_DASHBOARD_CONTRACT === '1') {
    const bar = '='.repeat(78);
    return {
      kind: 'skip',
      notice: [bar,
        'DASHBOARD LAYOUT CONTRACT SKIPPED: no Chrome or Chromium was found and',
        'VAELOR_SKIP_DASHBOARD_CONTRACT=1 is set. The dashboard layout was NOT checked.',
        'Set VAELOR_ACCEPTANCE_CHROME to a Chrome or Chromium executable to run it,',
        'e.g. VAELOR_ACCEPTANCE_CHROME=/usr/bin/chromium npm run dashboard:contract',
        bar].join('\n'),
    };
  }
  return {
    kind: 'fail',
    message: 'Dashboard layout contract: no Chrome or Chromium was found. Set VAELOR_ACCEPTANCE_CHROME to a '
      + 'Chrome or Chromium executable, or set VAELOR_SKIP_DASHBOARD_CONTRACT=1 to skip the contract on a machine '
      + 'without a browser.',
  };
}

const round = (value) => (value === null ? null : Math.round(value * 10) / 10);

/** The Fleet tile clauses' widths and cases (see the module docstring). */
const FLEET_WIDTHS = [1920, 1440, 1100, 900, 768, 375];
const FLEET_CASES = [
  { sensor: 'graphics engine', long: false },
  { sensor: 'board', long: false },
  { sensor: 'none', long: false },
  { sensor: 'board', long: true },
];
/**
 * The height styles/cluster-fleet.css gives an opened card's trend
 * (`.cf-telemetry__tiles .ui-trend`). Written here, not read from the
 * stylesheet: a trend squeezed by its tile reads its own squeezed height.
 */
const TREND_HEIGHT = 22;
/** The narrowest a trend bar may be drawn and still be seen as a bar. */
const TREND_BAR_MIN_WIDTH = 1;
const TITLE_LINES_MAX = 2;
const OPEN_CARD_LABEL = 'Cluster connection, trends and machine actions';
const FLEET_REASON = 'LESSONS 2 / VD-200: the opened machine card\'s telemetry tiles (StatTile, `.ui-stat-tile` with '
  + '`.ui-trend`, MachineMetrics.tsx). A trend squeezed below its height or to bars under a pixel, a title past two '
  + 'lines or a reading under its trend is the tile the board does not draw. Do not drop a width or a case, and do '
  + 'not lower the counts - fit the tile row (it once squeezed the icon badge, VD-147).';

/** What one Fleet page measures: each open machine tile's trend, title lines and reading against its trend. */
function measureFleetTiles() {
  const lines = (element) => {
    const style = getComputedStyle(element);
    const height = parseFloat(style.lineHeight) || parseFloat(style.fontSize) * 1.2;
    return Math.round(element.getBoundingClientRect().height / height);
  };
  return {
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    // Only the opened card's tiles: the closed cards and the Fleet headline draw no telemetry tiles.
    tiles: [...document.querySelectorAll('.cf-card.is-open .cf-telemetry__tiles > .ui-stat-tile')].map((tile) => {
      const title = tile.querySelector('.ui-stat-tile__top > span');
      const reading = tile.querySelector('.ui-stat-tile__value');
      const trend = tile.querySelector('.ui-trend');
      const bars = trend ? [...trend.children].map((bar) => bar.getBoundingClientRect().width) : [];
      return {
        title: title ? title.textContent.trim() : '(no title)',
        titleLines: title ? lines(title) : 0,
        trend: trend ? [trend.getBoundingClientRect().width, trend.getBoundingClientRect().height] : null,
        bars: bars.length,
        narrowestBar: bars.length ? Math.min(...bars) : null,
        underTrend: Boolean(reading && trend) && reading.getBoundingClientRect().bottom > trend.getBoundingClientRect().top + 0.5,
        hasReading: Boolean(reading),
      };
    }),
  };
}

async function fleetPass(browser, fleetUrl) {
  const results = [];
  for (const width of FLEET_WIDTHS) {
    for (const { sensor, long } of FLEET_CASES) {
      if (long && ![1440, 900, 375].includes(width)) continue;
      // The worker's card, opened by its foot button (its accessible name ends "for <machine name>").
      const openerOf = (page) => page.locator('.cf-card', { hasText: long ? 'Workstation' : 'ZBook' })
        .getByRole('button', { name: new RegExp(`^${OPEN_CARD_LABEL}`) });
      const page = await openPage(browser, `${fleetUrl}?cluster=fleet&sensor=${encodeURIComponent(sensor)}${long ? '&long=1' : ''}`,
        { viewport: { width, height: width < 600 ? 812 : 1000 }, ready: (opened) => openerOf(opened).waitFor({ timeout: LOAD_TIMEOUT_MS }) });
      await openerOf(page).click();
      await page.waitForFunction(() => document.querySelectorAll('.cf-card.is-open .cf-telemetry__tiles > .ui-stat-tile .ui-trend').length >= 7,
        null, { timeout: LOAD_TIMEOUT_MS });
      await page.waitForTimeout(300);
      const measured = await page.evaluate(measureFleetTiles);
      await page.close();
      // Where the old tile drew a round 38 px icon badge, the StatTile draws no
      // icon: its one fixed-size graphic is the trend, so the clause that the
      // badge is never squeezed by the row beside it becomes the trend's.
      const squeezed = measured.tiles.filter((tile) => !tile.trend || Math.abs(tile.trend[1] - TREND_HEIGHT) > TOLERANCE
        || tile.bars === 0 || tile.narrowestBar < TREND_BAR_MIN_WIDTH);
      const tall = measured.tiles.filter((tile) => tile.titleLines < 1 || tile.titleLines > TITLE_LINES_MAX);
      const under = measured.tiles.filter((tile) => !tile.hasReading || tile.underTrend);
      const clauses = {
        seven_tiles: measured.tiles.length === 7,
        trends_full_size: squeezed.length === 0,
        titles_two_lines_at_most: tall.length === 0,
        readings_clear_of_trend: under.length === 0,
        no_horizontal_overflow: measured.overflow <= 0,
      };
      results.push({
        case: `${width}-${sensor.replace(/ /g, '_')}${long ? '-long' : ''}`,
        tiles: measured.tiles.length,
        squeezed: squeezed.map((tile) => `${tile.title} ${tile.trend ? tile.trend.map(round).join('x') : 'no trend'}`
          + `, ${tile.bars} bars, narrowest ${round(tile.narrowestBar)} px`),
        tall: tall.map((tile) => `${tile.title} (${tile.titleLines} lines)`),
        under: under.map((tile) => tile.title),
        overflow: measured.overflow,
        clauses,
        pass: Object.values(clauses).every(Boolean),
      });
    }
  }
  return results;
}

function printFleet(results) {
  console.log('FLEET MACHINE TILES');
  for (const row of results) {
    const failed = Object.entries(row.clauses).filter(([, held]) => !held).map(([clause]) => clause);
    console.log([row.case.padEnd(26), row.pass ? 'yes' : 'NO', failed.join(' | '), `${row.tiles} tiles, overflow ${row.overflow} px`,
      row.squeezed.length ? 'trends: ' + row.squeezed.join(' | ') : '',
      row.tall.length ? 'titles: ' + row.tall.join(' | ') : '',
      row.under.length ? 'under trend: ' + row.under.join(' | ') : ''].join(' '));
    if (!row.pass) console.log(`  ${FLEET_REASON}`);
  }
}

const DEPLOYMENT_WIDTHS = [1920, 1440, 1280, 1100, 1024, 900, 768, 600, 506, 480, 375];
/** W4-D2: the widths a live-like Deployments tab (a serving row, the endpoint
 *  cards filled) is held at, where it once scrolled sideways (491-506 px). */
const LIVE_DEPLOYMENT_WIDTHS = [1440, 1024, 768, 600, 506, 480, 375];
/** B1: the tallest a key row (one line per key, `ul.cl-ep-keys > li`, VD-200) may be. */
const KEY_ROW_MAX = 220;
/** What a failed key clause prints, so the reader learns the rule here. */
const KEYS_REASON = 'LESSONS 3 / B1, VD-200: the overflow clause skips anything inside a scroll box, so the keys '
  + 'table once passed at 375 px with Rotate and Revoke one letter per line in 305 px rows, hidden in its own '
  + 'scrolling box. The keys are now one line per key (EndpointsPanel.tsx). Do not raise KEY_ROW_MAX or exempt a '
  + 'box - make the key rows fit.';
/** The widths a long (IPv6) endpoint is measured at: where the row is still
 *  one line and its endpoint track once took the name's room (merge review). */
const LONG_ENDPOINT_WIDTHS = [1220, 1200];
const NAME_MIN_WIDTH = 160;

/** What a failed Deployments row clause prints (VD-200 port). */
const DEPLOYMENTS_REASON = 'LESSONS 2, 3 / VD-200: the Deployments row is now a table (`tbody.cl-dep-group`, ClusterDeployments.tsx) '
  + 'inside a box that scrolls sideways on its own, so the page-overflow clause alone would pass a row scrolled out '
  + 'of sight. Do not drop a width or a case, and do not exempt the table box - fit the row (a block per row on a phone).';

/** The tab panel every Deployments measurement is scoped to. */
const PANEL = '[role="tabpanel"].fleet-tab-panel';

/**
 * The Deployments row, in the page: line counts come from the text's own line
 * boxes. The row is the model group's first row; its notes are the two fixture
 * fence notes, drawn as Notice banners in the rows beneath it.
 */
function measureDeploymentRow(panelSelector) {
  const panel = document.querySelector(panelSelector);
  const group = panel.querySelector('tbody.cl-dep-group');
  const row = group.querySelector('tr:not(.cl-dep-sub)');
  // Text lines only: a decoration inside the element (the status dot) has a
  // box of its own at another height and is not a line of text.
  const lines = (element) => {
    if (!element) return 0;
    const tops = new Set();
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      if (!node.textContent.trim() || node.parentElement.closest('.sr-only')) continue;
      const range = document.createRange();
      range.selectNodeContents(node);
      for (const rect of range.getClientRects()) if (rect.width > 0) tops.add(Math.round(rect.bottom));
    }
    return tops.size;
  };
  const endpoint = row.querySelector('td.cl-dep-address');
  const name = row.querySelector('td.cl-dep-cell--name');
  const box = panel.querySelector('.ui-table-scroll');
  const notes = [...group.querySelectorAll('tr.cl-dep-sub .ui-notice')]
    .filter((note) => note.textContent.trim().startsWith('Fixture'));
  return {
    endpointLines: lines(endpoint),
    endpointText: endpoint.textContent,
    endpointTitle: endpoint.getAttribute('title'),
    badgeLines: lines(row.querySelector('.cl-tag')),
    statusLines: lines(row.querySelector('.status-pill')),
    nameWidth: name.getBoundingClientRect().width,
    nameRight: name.getBoundingClientRect().right,
    nameBottom: name.getBoundingClientRect().bottom,
    endpointLeft: endpoint.getBoundingClientRect().left,
    endpointTop: endpoint.getBoundingClientRect().top,
    listWidth: box.getBoundingClientRect().width,
    notesCut: notes.map((note) => note.querySelector('.ui-notice__content > span'))
      .filter((text) => !text || text.scrollWidth > text.clientWidth + 1 || text.scrollHeight > text.clientHeight + 1
        || text.getBoundingClientRect().right > box.getBoundingClientRect().right + 1).length,
    notes: notes.map((note) => note.dataset.severity).sort().join(','),
  };
}

/**
 * What the whole Deployments panel shows, in the page: the page's overflow,
 * the boxes past the edge, what a scroll box hides sideways, the key rows and
 * any button that breaks a word.
 */
function measureDeploymentsPanel(panelSelector) {
  const panel = document.querySelector(panelSelector);
  const lines = (element) => {
    const tops = new Set();
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      if (!node.textContent.trim() || node.parentElement.closest('.sr-only')) continue;
      const range = document.createRange();
      range.selectNodeContents(node);
      for (const rect of range.getClientRects()) if (rect.width > 0) tops.add(Math.round(rect.bottom));
    }
    return tops.size;
  };
  const visibleText = (element) => [...element.childNodes].map((node) => (node.nodeType === Node.TEXT_NODE
    ? node.textContent : node.nodeType === Node.ELEMENT_NODE && !node.classList.contains('sr-only') ? visibleText(node) : '')).join('');
  return {
    overflow: document.documentElement.scrollWidth - window.innerWidth,
    // The innermost boxes past the edge (none of their children is), which
    // is where a too-wide thing is, not the containers it stretched. A box
    // inside a scroll box of its own scrolls there, not the page, and is not
    // reported here - `hiddenInScrollBox` reads those.
    wider: [...panel.querySelectorAll('*')]
      .filter((element) => {
        const past = (box) => box.getBoundingClientRect().right > window.innerWidth + 1;
        let parent = element.parentElement;
        while (parent) {
          if (['auto', 'scroll', 'hidden'].includes(getComputedStyle(parent).overflowX)) return false;
          parent = parent.parentElement;
        }
        return past(element) && ![...element.children].some(past);
      })
      .slice(0, 4).map((element) => `${element.tagName.toLowerCase()}.${[...element.classList].join('.')}`),
    // B1: inside the scroll boxes - the deployments table's own box among
    // them. A `code` value box is allowed: it is one line beside its own Copy button.
    hiddenInScrollBox: [...panel.querySelectorAll('*')]
      .filter((element) => element.tagName !== 'CODE'
        && ['auto', 'scroll'].includes(getComputedStyle(element).overflowX)
        && element.scrollWidth > element.clientWidth + 1)
      .map((element) => `${element.tagName.toLowerCase()}.${[...element.classList].join('.')} hides ${element.scrollWidth - element.clientWidth} px`),
    // The endpoint cards' key rows: one line per key (EndpointsPanel.tsx).
    keyRows: [...panel.querySelectorAll('ul.cl-ep-keys > li')].map((row) => Math.round(row.getBoundingClientRect().height)),
    // A button whose label takes more lines than it has words has broken a word.
    brokenButtons: [...panel.querySelectorAll('button')]
      .filter((button) => {
        const words = visibleText(button).trim();
        return words && button.getBoundingClientRect().width > 0 && lines(button) > words.split(/\s+/).length;
      })
      .map((button) => `${visibleText(button).trim()} (${lines(button)} lines)`),
  };
}

async function deploymentsPass(browser, fleetUrl) {
  const results = [];
  const cases = [...DEPLOYMENT_WIDTHS.map((width) => [width, '']),
    ...LONG_ENDPOINT_WIDTHS.map((width) => [width, 'long']),
    ...LIVE_DEPLOYMENT_WIDTHS.map((width) => [width, 'live'])];
  for (const [width, endpoint] of cases) {
    const query = endpoint === 'live' ? '&live=1' : endpoint ? `&endpoint=${endpoint}` : '';
    const page = await openPage(browser, `${fleetUrl}?cluster=deployments&deploy=1${query}`,
      { viewport: { width, height: width < 600 ? 812 : 1000 }, ready: 'tbody.cl-dep-group tr.cl-dep-sub .ui-notice--info' });
    await page.waitForTimeout(300);
    // Two evaluations: each function is sent to the page on its own.
    const measured = { ...(await page.evaluate(measureDeploymentRow, PANEL)), ...(await page.evaluate(measureDeploymentsPanel, PANEL)) };
    // The endpoint cards and their API keys show under the Models filter
    // (ClusterDeploymentsTab.tsx), so the live tab is measured there as well.
    let keys = null;
    if (endpoint === 'live') {
      await page.locator(`${PANEL} .cl-chips button`, { hasText: /^Models/ }).click();
      await page.waitForFunction((selector) => document.querySelectorAll(`${selector} ul.cl-ep-keys > li`).length >= 2,
        PANEL, { timeout: LOAD_TIMEOUT_MS });
      await page.waitForTimeout(300);
      keys = await page.evaluate(measureDeploymentsPanel, PANEL);
    }
    await page.close();
    const clauses = {
      endpoint_one_line: measured.endpointLines === 1,
      endpoint_titled: measured.endpointTitle === measured.endpointText,
      badge_and_status_one_line: measured.badgeLines === 1 && measured.statusLines === 1,
      name_not_crushed: measured.nameWidth >= NAME_MIN_WIDTH,
      // A list 900 px or wider keeps the row on one line (the endpoint to the
      // right of the name, on the same line): a table that stacks its cells
      // passes every other clause.
      one_line_when_wide: measured.listWidth < 900
        || (measured.endpointLeft >= measured.nameRight && measured.endpointTop < measured.nameBottom),
      both_notes_uncut: measured.notes === 'info,warning' && measured.notesCut === 0,
      no_horizontal_overflow: measured.overflow <= 0,
      // VD-200: the table scrolls inside its own box, which the overflow clause skips.
      nothing_hidden_in_scroll_box: measured.hiddenInScrollBox.length === 0,
      ...(keys ? {
        keys_no_horizontal_overflow: keys.overflow <= 0,
        keys_nothing_hidden_in_scroll_box: keys.hiddenInScrollBox.length === 0,
        key_rows_capped: keys.keyRows.length === 2 && keys.keyRows.every((height) => height <= KEY_ROW_MAX),
        buttons_whole_words: measured.brokenButtons.length === 0 && keys.brokenButtons.length === 0,
      } : {}),
    };
    const suffix = { long: '-long-endpoint', live: '-live' }[endpoint] ?? '';
    results.push({ case: `${width}-deployments${suffix}`, measured, keys, clauses,
      pass: Object.values(clauses).every(Boolean) });
  }
  return results;
}

function printDeployments(results) {
  console.log('DEPLOYMENTS ROW');
  for (const row of results) {
    const failed = Object.entries(row.clauses).filter(([, held]) => !held).map(([clause]) => clause);
    const { measured, keys } = row;
    console.log([row.case.padEnd(26), row.pass ? 'yes' : 'NO', failed.join(' | '),
      `endpoint ${measured.endpointLines} line(s), name ${round(measured.nameWidth)} px, notes [${measured.notes}] ${measured.notesCut} cut`,
      measured.wider.length ? `past the edge: ${measured.wider.join(', ')}` : '',
      measured.hiddenInScrollBox.length ? `hidden in a scroll box: ${measured.hiddenInScrollBox.join(', ')}` : '',
      keys?.wider.length ? `keys past the edge: ${keys.wider.join(', ')}` : ''].join(' '));
    if (failed.some((clause) => !clause.startsWith('key') && clause !== 'buttons_whole_words')) console.log(`  ${DEPLOYMENTS_REASON}`);
    if (failed.some((clause) => clause.startsWith('key') || clause === 'buttons_whole_words')) {
      console.log(`  ${KEYS_REASON}\n  hidden in a scroll box: ${keys?.hiddenInScrollBox.join(', ') || 'none'}; `
        + `key rows: ${keys?.keyRows.join(', ')} px; broken buttons: `
        + `${[...measured.brokenButtons, ...(keys?.brokenButtons ?? [])].join(', ') || 'none'}`);
    }
  }
}
const TAB_STRIP_WIDTHS = [768, 600, 375];
const TAB_STRIP_SECTIONS = ['activity', 'agents'];
/** The Cluster page's tabs (FleetCenter.tsx): a strip that shows fewer has lost one. */
const TAB_COUNT = 6;
const TAB_STRIP_REASON = 'LESSONS 3 / B2, VD-200: a scrolling strip with no cue reads as a strip with nothing more in it, '
  + 'and a selected tab out of sight reads as no tab selected. At 720 px and below the strip now wraps onto two rows '
  + '(styles/cluster.css), which hides nothing - so every tab must be in sight there; a strip that scrolls must still '
  + 'cue each end it hides. Do not drop the width or the case - keep the cue attributes in step with the scroll '
  + 'position and scroll the selected tab into the strip.';

/** The Cluster tab strip, in the page. */
function measureTabStrip() {
  const list = document.querySelector('[role="tablist"][aria-label="Cluster views"]');
  const selected = list.querySelector('[aria-selected="true"]');
  const strip = list.getBoundingClientRect();
  const tab = selected.getBoundingClientRect();
  const hiddenStart = list.scrollLeft > 1;
  const hiddenEnd = list.scrollLeft + list.clientWidth < list.scrollWidth - 1;
  const tabs = [...list.querySelectorAll('[role="tab"]')];
  // A tab out of sight: outside the strip's visible box or past the viewport.
  const outOfSight = tabs.filter((item) => {
    const box = item.getBoundingClientRect();
    return box.left < Math.max(strip.left, 0) - 1 || box.right > Math.min(strip.right, innerWidth) + 1
      || box.top < strip.top - 1 || box.bottom > strip.bottom + 1;
  });
  return {
    selected: selected.textContent.trim(),
    selectedInView: tab.left >= strip.left - 1 && tab.right <= strip.right + 1,
    tabs: tabs.length,
    rows: new Set(tabs.map((item) => Math.round(item.getBoundingClientRect().top))).size,
    scrolls: list.scrollWidth > list.clientWidth + 1,
    outOfSight: outOfSight.map((item) => item.textContent.trim()),
    // Each tab out of sight lies at an end the strip cues.
    outOfSightUncued: outOfSight.filter((item) => {
      const box = item.getBoundingClientRect();
      const atStart = box.left < strip.left - 1;
      const atEnd = box.right > strip.right + 1;
      return !((atStart && list.hasAttribute('data-more-start')) || (atEnd && list.hasAttribute('data-more-end')));
    }).map((item) => item.textContent.trim()),
    hiddenStart,
    hiddenEnd,
    cueStart: list.hasAttribute('data-more-start'),
    cueEnd: list.hasAttribute('data-more-end'),
  };
}

async function tabStripPass(browser, fleetUrl) {
  const results = [];
  for (const width of TAB_STRIP_WIDTHS) {
    for (const section of TAB_STRIP_SECTIONS) {
      const page = await openPage(browser, `${fleetUrl}?cluster=${section}`,
        { viewport: { width, height: 812 }, ready: '[aria-label="Cluster views"] [aria-selected="true"]' });
      await page.waitForTimeout(400);
      const measured = await page.evaluate(measureTabStrip);
      await page.close();
      const clauses = {
        all_tabs_drawn: measured.tabs === TAB_COUNT,
        selected_tab_in_view: measured.selectedInView,
        // A wrapped strip scrolls nowhere, so it may cue nothing and hide nothing:
        // "no tab out of sight". A scrolling strip cues each end it hides.
        cue_where_tabs_are_hidden: measured.cueStart === measured.hiddenStart && measured.cueEnd === measured.hiddenEnd,
        no_tab_out_of_sight_uncued: measured.outOfSightUncued.length === 0
          && (measured.scrolls || measured.outOfSight.length === 0),
      };
      results.push({ case: `${width}-tabs-${section}`, measured, clauses, pass: Object.values(clauses).every(Boolean) });
    }
  }
  return results;
}

function printTabStrip(results) {
  console.log('CLUSTER TAB STRIP');
  for (const row of results) {
    const failed = Object.entries(row.clauses).filter(([, held]) => !held).map(([clause]) => clause);
    const { measured } = row;
    console.log([row.case.padEnd(26), row.pass ? 'yes' : 'NO', failed.join(' | '),
      `${measured.tabs} tabs on ${measured.rows} row(s), ${measured.scrolls ? 'scrolling' : 'not scrolling'}; `
      + `selected "${measured.selected}"; hidden start/end ${measured.hiddenStart}/${measured.hiddenEnd}; `
      + `cue start/end ${measured.cueStart}/${measured.cueEnd}; out of sight: ${measured.outOfSight.join(', ') || 'none'}`].join(' '));
    if (!row.pass) console.log(`  ${TAB_STRIP_REASON}`);
  }
}

const DIALOG_VIEWPORTS = [[1440, 900], [375, 667], [320, 667]];
const DIALOG_REASON = 'LESSONS 13 / review round 1, VD-200: a confirmation taller than a short phone put Cancel below the '
  + 'screen, where it cannot be tapped. The cluster-link confirmation is now the Cluster dialog (role "alertdialog", eyebrow '
  + '"Cluster link", HostSettingsPanel.tsx). Do not drop the viewport or its note - give the dialog a max-height inside '
  + 'the viewport and scroll its body, with the buttons kept on screen.';
/** The cluster-link confirmation, in the page: the one dialog whose eyebrow is "Cluster link". */
function measureLinkDialog() {
  const dialogs = [...document.querySelectorAll('[role="alertdialog"]')]
    .filter((element) => element.querySelector('.cl-eyebrow')?.textContent.trim() === 'Cluster link');
  if (dialogs.length !== 1) return { found: dialogs.length, top: 0, bottom: 0, viewport: innerHeight, buttons: [], note: false };
  const [dialog] = dialogs;
  const box = dialog.getBoundingClientRect();
  const buttons = [...dialog.querySelectorAll('button')].map((button) => {
    const rect = button.getBoundingClientRect();
    return { label: button.textContent.trim(), top: rect.top, bottom: rect.bottom, height: rect.height };
  });
  // The case is the confirmation WITH its shared-card note: without it the dialog is short and passes anyway.
  const note = [...dialog.querySelectorAll('.ui-notice')].some((element) => element.textContent.includes('Fixture note'));
  return { found: 1, top: box.top, bottom: box.bottom, viewport: innerHeight, buttons, note };
}

async function dialogPass(browser, fleetUrl) {
  const results = [];
  for (const [width, height] of DIALOG_VIEWPORTS) {
    const pickerOf = (page) => page.getByRole('combobox', { name: 'Link for split-model traffic' });
    const page = await openPage(browser, `${fleetUrl}?cluster=setup&hostlink=1`,
      { viewport: { width, height }, ready: (opened) => pickerOf(opened).waitFor({ timeout: LOAD_TIMEOUT_MS }) });
    const picker = pickerOf(page);
    await picker.selectOption('enp1s0');
    await page.getByRole('button', { name: 'Review link change' }).click();
    // A confirmation is an alertdialog (ClusterDialog's `role`). Not finding it is a failed clause with
    // the count it found, not a crash that stops every pass after it (LESSONS 3).
    await page.getByRole('alertdialog').filter({ has: page.locator('.cl-eyebrow', { hasText: 'Cluster link' }) })
      .waitFor({ timeout: LOAD_TIMEOUT_MS }).catch(() => {});
    await page.waitForTimeout(300);
    const measured = await page.evaluate(measureLinkDialog);
    await page.close();
    const clauses = {
      dialog_with_its_note: measured.found === 1 && measured.note,
      dialog_inside_viewport: measured.found === 1 && measured.top >= -0.5 && measured.bottom <= measured.viewport + 0.5,
      buttons_on_screen: measured.buttons.length === 2
        && measured.buttons.every((button) => button.top >= 0 && button.bottom <= measured.viewport + 0.5),
      buttons_44_tall: measured.buttons.length === 2 && measured.buttons.every((button) => button.height >= 44 - TOLERANCE),
    };
    results.push({ case: `${width}x${height}-link-confirm`, measured, clauses, pass: Object.values(clauses).every(Boolean) });
  }
  return results;
}

function printDialogs(results) {
  console.log('CONFIRMATION DIALOG');
  for (const row of results) {
    const failed = Object.entries(row.clauses).filter(([, held]) => !held).map(([clause]) => clause);
    console.log([row.case.padEnd(26), row.pass ? 'yes' : 'NO', failed.join(' | '),
      `dialog ${round(row.measured.top)}-${round(row.measured.bottom)} of ${row.measured.viewport}; `
      + row.measured.buttons.map((button) => `${button.label} ${round(button.top)}-${round(button.bottom)}`).join(', ')].join(' '));
    if (!row.pass) console.log(`  ${DIALOG_REASON}`);
  }
}

/** Load one state in one mode and wait until its panels are drawn. */
async function open(browser, pageUrl, state, mode, panels) {
  // The first page compiles the app on a cold Vite server; on a loaded machine
  // (a full test run beside it) that took more than the default 30 s. A page
  // left blank by a module request that failed is loaded once more by
  // openPage, which says so (HARNESS-1); a page that is merely slow is waited for.
  const page = await openPage(browser, `${pageUrl}?cluster=performance&s=${state}&mode=${mode}`,
    { viewport: VIEWPORT, ready: '.perf-tiles .perf-tile' });
  await page.waitForFunction((count) => document.querySelectorAll('.perf-panels > figure').length >= count, panels,
    { timeout: LOAD_TIMEOUT_MS });
  await page.waitForTimeout(300);
  return page;
}

/** What one loaded page measures, in the page. `from` is the first panel whose plots are read. */
function measure(from) {
  const box = (element) => {
    const rect = element.getBoundingClientRect();
    return { top: rect.top + window.scrollY, bottom: rect.bottom + window.scrollY, width: rect.width, height: rect.height, left: rect.left };
  };
  const cut = (element) => element.scrollWidth > element.clientWidth + 1;
  const figures = [...document.querySelectorAll('.perf-panels > figure')];
  const read = figures.slice(from);
  return {
    count: figures.length,
    titles: read.map((figure) => figure.querySelector('.perf-chart__title')?.textContent.trim() ?? ''),
    tiles: [...document.querySelectorAll('.perf-tiles > .perf-tile')].map(box),
    panels: figures.map(box),
    plots: read.flatMap((figure) => [...figure.querySelectorAll('.chart-plot')]).map((plot) => plot.getBoundingClientRect().height),
    // The rendered gridlines of each plot, top to bottom; the smallest gap between neighbours.
    gaps: read.flatMap((figure) => [...figure.querySelectorAll('.perf-chart__svg')]).map((svg) => {
      const ys = [...svg.querySelectorAll('line.perf-grid')].map((line) => line.getBoundingClientRect().top).sort((a, b) => a - b);
      const steps = ys.slice(1).map((y, index) => y - ys[index]);
      return steps.length ? Math.min(...steps) : null;
    }).filter((gap) => gap !== null),
    overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    // A name the screen shows shortened with no full text in its title to fall back on: never allowed.
    shortened: [...document.querySelectorAll('.perf-panels .perf-legend__toggle, .perf-panels .perf-chip, .perf-tiles .perf-tile__title, .perf-panels .perf-chart__title')]
      .filter((element) => cut(element.querySelector('.perf-legend__name') ?? element))
      .filter((element) => (element.getAttribute('title') ?? '').trim() !== element.textContent.trim())
      .map((element) => element.textContent.trim()),
  };
}

const lowest = (values) => (values.length ? Math.min(...values) : null);
const plotsTall = (plots) => plots.every((height) => height >= PLOT_MIN - TOLERANCE);
const gridApart = (gap) => gap === null || gap >= GRID_GAP_MIN - TOLERANCE;

async function easyPass(browser, pageUrl, states) {
  const results = [];
  for (const state of states) {
    const page = await open(browser, pageUrl, state, 'easy', EASY_PANELS);
    const measured = await page.evaluate(measure, 0);
    await page.close();
    const heights = measured.tiles.map((tile) => tile.height);
    const widths = measured.tiles.map((tile) => tile.width);
    const spread = (values) => Math.max(...values) - Math.min(...values);
    const rowOneCards = measured.panels.slice(0, 4);
    const rowOne = Math.max(...rowOneCards.map((card) => card.bottom), ...measured.tiles.map((tile) => tile.bottom));
    const rowTwo = Math.max(...measured.panels.slice(4, 8).map((card) => card.bottom));
    const plotMin = lowest(measured.plots);
    const gapMin = lowest(measured.gaps);
    const clauses = {
      six_tiles: measured.tiles.length === 6,
      tiles_equal_height: spread(heights) <= TOLERANCE,
      tiles_equal_width: spread(widths) <= TOLERANCE,
      // One size for every chart card (owner, approving the boards), four to a row.
      cards_one_size: spread(measured.panels.map((card) => card.height)) <= TOLERANCE && spread(measured.panels.map((card) => card.width)) <= TOLERANCE,
      four_columns: rowOneCards.every((card, at) => Math.abs(card.top - rowOneCards[0].top) <= TOLERANCE && (at === 0 || card.left > rowOneCards[at - 1].left)),
      first_row_above_fold: rowOne <= VIEWPORT.height - FOLD_SLACK,
      // Every plot drawn (a panel that says why it has nothing to draw has none) keeps its full height.
      plots_tall_enough: plotsTall(measured.plots),
      gridlines_apart: gridApart(gapMin),
      no_horizontal_overflow: measured.overflow <= 0,
      shortened_names_keep_their_title: measured.shortened.length === 0,
    };
    const pass = Object.values(clauses).every(Boolean);
    results.push({
      state,
      tile_height: round(heights[0]),
      tile_width: round(widths[0]),
      panel_width: round(measured.panels[0].width),
      first_row_bottom: round(rowOne),
      second_row_bottom: round(rowTwo),
      plot_min_height: round(plotMin),
      gridline_gap_min: round(gapMin),
      horizontal_overflow: measured.overflow,
      shortened: measured.shortened,
      clauses,
      pass,
    });
  }
  return results;
}

async function advancedPass(browser, pageUrl, states) {
  const results = [];
  for (const state of states) {
    const page = await open(browser, pageUrl, state, 'advanced', EASY_PANELS + ADVANCED_PANELS);
    const measured = await page.evaluate(measure, EASY_PANELS);
    await page.close();
    const gapMin = lowest(measured.gaps);
    const spread = (values) => Math.max(...values) - Math.min(...values);
    const clauses = {
      cards_one_size: spread(measured.panels.map((card) => card.height)) <= TOLERANCE,
      advanced_panels_shown: measured.count === EASY_PANELS + ADVANCED_PANELS && measured.titles.length === ADVANCED_PANELS,
      plots_tall_enough: plotsTall(measured.plots),
      gridlines_apart: gridApart(gapMin),
      no_horizontal_overflow: measured.overflow <= 0,
      shortened_names_keep_their_title: measured.shortened.length === 0,
    };
    results.push({
      state,
      panels: measured.titles,
      plots_drawn: measured.plots.length,
      plot_min_height: round(lowest(measured.plots)),
      gridline_gap_min: round(gapMin),
      horizontal_overflow: measured.overflow,
      shortened: measured.shortened,
      clauses,
      pass: Object.values(clauses).every(Boolean),
    });
  }
  return results;
}

function printEasy(results) {
  console.log('EASY');
  console.log('state                    tiles (h x w)   row-1 bottom  row-2 bottom  plot min  grid gap  overflow  fold   names  cards  pass');
  for (const row of results) {
    console.log([
      row.state.padEnd(24), `${row.tile_height} x ${row.tile_width}`.padEnd(15),
      String(row.first_row_bottom).padEnd(13), String(row.second_row_bottom).padEnd(13),
      String(row.plot_min_height).padEnd(9), String(row.gridline_gap_min).padEnd(9), String(row.horizontal_overflow).padEnd(9),
      String(row.clauses.first_row_above_fold).padEnd(6), String(row.clauses.shortened_names_keep_their_title).padEnd(6),
      String(row.clauses.cards_one_size).padEnd(6),
      row.pass ? 'yes' : 'NO',
      row.shortened.length ? 'shortened without a title: ' + row.shortened.join(' | ') : '',
    ].join(' '));
    if (!row.clauses.first_row_above_fold) console.log(`  ${FOLD_REASON} (row 1 ends at ${row.first_row_bottom}, limit ${VIEWPORT.height - FOLD_SLACK})`);
  }
}

function printAdvanced(results) {
  console.log('ADVANCED');
  console.log('state                    panels  plots  plot min  grid gap  overflow  names  pass');
  for (const row of results) {
    console.log([
      row.state.padEnd(24), String(row.panels.length).padEnd(7), String(row.plots_drawn).padEnd(6),
      String(row.plot_min_height).padEnd(9), String(row.gridline_gap_min).padEnd(9), String(row.horizontal_overflow).padEnd(9),
      String(row.clauses.shortened_names_keep_their_title).padEnd(6), row.pass ? 'yes' : 'NO',
      row.shortened.length ? 'shortened: ' + row.shortened.join(' | ') : '',
    ].join(' '));
  }
}

/** Every pass, by the name `--only=` takes. */
export const PASSES = ['easy', 'advanced', 'fleet', 'deployments', 'tabs', 'dialogs', 'a11y'];

/**
 * Which passes a run measures. No argument runs every pass. `--only=a,b` runs
 * the named passes and `--a11y=clause,...` only the named audit clauses, for a
 * quicker loop while a layout is being fixed. A name that is not a pass or a
 * clause fails the run rather than measure nothing and pass (LESSONS 3), and a
 * partial run says so on its last line. `npm run dashboard:contract` runs it all.
 */
export function chooseRun(argv) {
  const listed = (flag) => {
    const argument = argv.find((item) => item.startsWith(flag));
    return argument ? argument.slice(flag.length).split(',').map((item) => item.trim()).filter(Boolean) : null;
  };
  const only = listed('--only=');
  const clauses = listed('--a11y=');
  const unknown = [...(only ?? []).filter((name) => !PASSES.includes(name)),
    ...(clauses ?? []).filter((name) => !A11Y_CLAUSES.includes(name))];
  if (unknown.length || (only && !only.length) || (clauses && !clauses.length)) {
    return { error: `Dashboard layout contract (LESSONS 3 / VD-200: a run that names nothing real would measure nothing and pass): not a pass or an audit clause: ${unknown.join(', ') || '(an empty list)'}. `
      + `Passes: ${PASSES.join(', ')}. Audit clauses: ${A11Y_CLAUSES.join(', ')}.` };
  }
  // Audit clauses named alone run the audit alone.
  const passes = new Set(only ?? (clauses ? [] : PASSES));
  if (clauses) passes.add('a11y');
  const partial = only || clauses
    ? `PARTIAL RUN: only ${[...passes].join(', ')}${clauses ? ` (audit clauses ${clauses.join(', ')})` : ''}; `
      + 'the contract was not run in full.'
    : '';
  return { passes, clauses: clauses ? new Set(clauses) : null, partial };
}

/** The run's JSON line: every pass's rows, the reloads it made (HARNESS-1) and the verdict. */
export function contractSummary(results) {
  const { easy, advanced, fleet, deployments, tabStrip, dialogs, audit, pass } = results;
  return { viewport: VIEWPORT, easy, advanced, fleet, deployments, tabStrip, dialogs, audit, reloads: [...loadNotes], pass };
}

async function main() {
  // Before any server starts: a name that is not a pass fails at once.
  const selection = chooseRun(process.argv.slice(2));
  if (selection.error) {
    console.error(selection.error);
    return 1;
  }
  const choice = chooseBrowser(process.env, existsSync);
  if (choice.kind === 'fail') {
    console.error(choice.message);
    return 1;
  }
  if (choice.kind === 'skip') {
    console.warn(choice.notice);
    return 0;
  }
  const { chromium } = await import('playwright-core');
  const { createServer } = await import('vite');
  const payloads = JSON.parse(await readFile(path.join(frontend, 'src/test/dashboardPayloads.json'), 'utf8'));
  const states = Object.keys(payloads).sort();

  const server = await createServer({
    root: frontend,
    configFile: path.join(frontend, 'vite.config.ts'),
    logLevel: 'error',
    server: { host: '127.0.0.1', port: 0, strictPort: false, fs: { allow: [repository] } },
  });
  await server.listen();
  const address = server.httpServer?.address();
  const port = typeof address === 'object' && address ? address.port : null;
  if (!port) throw new Error('The contract server did not start.');
  const pageUrl = `http://127.0.0.1:${port}/v2/scripts/dashboard-contract/page.html`;
  const fleetUrl = `http://127.0.0.1:${port}/v2/scripts/dashboard-contract/fleet.html`;
  const assistantUrl = `http://127.0.0.1:${port}/v2/scripts/dashboard-contract/assistant.html`;
  const run = (name) => selection.passes.has(name);

  let easy = [];
  let advanced = [];
  let fleet = [];
  let deployments = [];
  let tabStrip = [];
  let dialogs = [];
  let audit = [];
  const browser = await chromium.launch({ executablePath: choice.path, headless: true, args: ['--disable-gpu', '--no-first-run'] });
  try {
    if (run('easy')) easy = await easyPass(browser, pageUrl, states);
    if (run('advanced')) advanced = await advancedPass(browser, pageUrl, states);
    if (run('fleet')) fleet = await fleetPass(browser, fleetUrl);
    if (run('deployments')) deployments = await deploymentsPass(browser, fleetUrl);
    if (run('tabs')) tabStrip = await tabStripPass(browser, fleetUrl);
    if (run('dialogs')) dialogs = await dialogPass(browser, fleetUrl);
    if (run('a11y')) audit = await a11yPass(browser, fleetUrl, assistantUrl, selection.clauses);
  } finally {
    await browser.close();
    await server.close();
  }
  if (run('easy')) printEasy(easy);
  if (run('advanced')) printAdvanced(advanced);
  if (run('fleet')) printFleet(fleet);
  if (run('deployments')) printDeployments(deployments);
  if (run('tabs')) printTabStrip(tabStrip);
  if (run('dialogs')) printDialogs(dialogs);
  if (run('a11y')) printA11y(audit);
  if (selection.partial) console.log(selection.partial);
  const pass = [...easy, ...advanced, ...fleet, ...deployments, ...tabStrip, ...dialogs, ...audit].every((row) => row.pass);
  console.log(JSON.stringify(contractSummary({ easy, advanced, fleet, deployments, tabStrip, dialogs, audit, pass })));
  return pass ? 0 : 1;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  // The reloads are said even when a pass throws: a reload is never silent (HARNESS-1).
  process.exit(await withLoadNotes(main));
}
