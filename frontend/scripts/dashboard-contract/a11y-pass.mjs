/**
 * The W5-S4 clauses of the layout contract (scripts/dashboard-layout-contract.mjs
 * runs them): what the W4d UI audit measured as broken, held in a real browser
 * so the fix cannot quietly come undone. Each clause names its audit finding.
 *
 * On the Cluster page (fleet.html):
 *   - fleet_reflow (UX-A8, WCAG 1.4.10): the Fleet tab, short and long machine
 *     name, at 320, 375 and 390 px, never scrolls sideways, and no machine
 *     name (each `.cf-card` title) or the capacity headline (the Free memory
 *     tile's figure) is broken mid-word to fit; all three are found.
 *   - compact_controls (UX-A2, WCAG 2.5.8): the Easy/Advanced toggle and the
 *     Deployments filter chips are at least 24 px tall at 1280 px and at least
 *     44 px tall at 390 and 320 px (a phone).
 *   - focus_not_obscured (UX-A9, WCAG 2.4.11): tabbing twice round the
 *     live-like Deployments tab, on its All filter and again on Models (where
 *     the endpoint cards and keys are), no focused control has a sticky or
 *     fixed element on top of it, at 1280 x 900 and 390 x 812; a walk that
 *     reaches no control fails.
 * On the Assistant page (assistant.html):
 *   - danger_contrast (UX-A1, WCAG 1.4.3): "Delete chat" (in the saved chat's
 *     More menu, VD-200) and every danger control on Ask reads at 4.5:1 or
 *     better, at rest and under the pointer.
 *   - accessible_names (UX-A3, UX-A6): the problem-area select is named
 *     "Problem area"; each information-access checkbox in the custom-agent
 *     editor is named by its title alone (its description is its description),
 *     and no button or checkbox in the editor is unnamed.
 *   - idrefs_resolve (UX-A4): on Ask and on Routines, every aria-controls,
 *     aria-labelledby and aria-describedby names an element that exists.
 * Back on the Cluster page:
 *   - setup_alignment (UX-A12): on Setup at 1440 and 1024 px, the field labels
 *     of the add-machine form and of the machine settings card start on one
 *     left edge (they once stood 1 px apart, one card bordered, one not). The
 *     add form draws its fields two-up, so each card's first column is its
 *     labels within 40 px of its leftmost one, and a settings card label must
 *     be among them.
 *   - pool_alignment (W5-D4, W6-2): at 1512 and 1280 px, with a short and a
 *     long worker name, the two GPU memory pool cards stand side by side and
 *     each row (the readings, the new-size label, the buttons) of one starts at
 *     the height of the same row of the other - before a Recheck and again
 *     once its outcome line shows (FE-W7-4).
 *   - retained_data (W4d-D20's panel, live=1 keeps two volumes): at 1440, 768
 *     and 375 px both entries show, the page does not scroll sideways, each
 *     "Delete data" is at least 44 px tall, on one line, and reads at 4.5:1.
 *   - audit_wrap (FE-W7-1, FE-W7-3; audit=long): at 1280, 390 and 320 px the
 *     Security audit trail, with a long chat title, a URL, two IPv6 addresses
 *     and a 216-character token, scrolls neither the page nor its own table -
 *     on the top-level Activity page (page=activity) AND on Cluster > Activity
 *     (the `.cl-act__audit` card, which is about 480 px wide at 1280 px).
 *     The second sits in a tab panel whose own word-break wraps it, so it alone
 *     stayed green with the audit wrap CSS deleted at 390 and 320.
 *     At 390 and 320 px the stacked audit card gives its content the width: the
 *     action cell takes at least 60% of the card (W7-D4: the Time cell kept
 *     ~150 px beside an ~80 px content column that broke addresses mid-word).
 *   - nav_labels (W7-D3; nav=1 renders the real navigation): at 320 and 375 px
 *     the phone's bottom bar shows every destination's whole name - no two
 *     labels overlap, none is cut, none is broken mid-word - and (W8-3) the
 *     same under a browser font raised to 20 and 24 px, where each item stays
 *     inside the bar and the page's bottom padding is at least the bar. W8-D2:
 *     at every font every name is VISIBLE - an icon-only bar (the name kept
 *     only for screen readers) fails, because the owner who raised the text
 *     size is the one who needs the words (WCAG 1.4.4).
 *   - facts_whole_words (W8-D2; page=home-facts renders Home's real facts):
 *     at 320 and 375 px, at the default text size and at 125% and 150%, no
 *     headline fact ("administrator", the version, the OS) breaks mid-word.
 *   - machine_card (VD-194 P1, FE-4): the Fleet tab with the worker's card
 *     open and a populated Worker software row, at 1280, 390 and 320 px, with a
 *     long machine name: no sideways scroll, nothing past the edge, nothing
 *     below 12 px.
 *   - type_floor (UX-A11): on Performance, Fleet, Deployments (All and Models), Setup, Activity,
 *     Agents & tools and the Assistant, at 1280 and 375 px, no visible text
 *     renders below 12 px (styles/typeFloor.test.ts reads every stylesheet).
 */

import { LOAD_TIMEOUT_MS, openPage } from './page-load.mjs';

const REASONS = {
  fleet_reflow: 'LESSONS 13 / UX-A8: a fixed min-width plus padding scrolled a 320 px phone 42 px sideways, and '
    + 'the first fix broke the controller name letter by letter. Do not drop the width - let the box shrink to its '
    + 'row (min(…, 100%)) and let what is beside a name wrap under it. LESSONS 2 / VD-200: the names are each `.cf-card` '
    + 'title and the Free memory figure - three of them, or the clause measured nothing.',
  compact_controls: 'LESSONS 13 / UX-A2: the Easy/Advanced toggle was 21 px tall and the chips 29 px. '
    + 'Do not lower the floor - size them from --control-height-compact. VD-200: the two Detail options and '
    + 'the four filter chips (`.cl-chips .cl-chip`).',
  focus_not_obscured: 'LESSONS 13 / UX-A9: keyboard focus scrolled controls under the sticky topbar (and, on a '
    + 'phone, under the fixed bottom navigation bar). Do not drop the case - keep scroll-padding-top and, at 720 px '
    + 'and below, scroll-padding-bottom on the page, each the bar plus a gap. LESSONS 2 / VD-200: the endpoint cards '
    + 'show under the Models filter, so both filters are walked, and a walk that reaches no control fails.',
  danger_contrast: 'LESSONS 5 / UX-A1: "Delete chat" drew --danger-text on --danger-strong at 3.97:1. '
    + 'Do not restyle the one button - every foreground token must pass on every danger fill.',
  accessible_names: 'LESSONS 19 / UX-A3, UX-A6: a control a screen reader cannot name is not reachable. '
    + 'Name each control by its visible title (aria-labelledby) and describe it with its help text.',
  idrefs_resolve: 'LESSONS 19 / UX-A4: an aria reference to an element that is not rendered points at nothing. '
    + 'Set it only while its target is in the page.',
  setup_alignment: 'LESSONS 13 / UX-A12: form rows one pixel apart read as misaligned (the owner has caught this '
    + 'before). Do not loosen the tolerance - give sibling cards on Setup the same border and padding. LESSONS 2 / '
    + 'VD-200: a settings card label must be among them, or the clause compared the add-machine form with itself; '
    + 'the second column of the add form (40 px past its first) is not compared. Do not widen the band.',
  pool_alignment: 'LESSONS 13 / W5-D4, FE-W7-4: the worker GPU pool card had a "Read ... / Recheck" row the controller card '
    + 'lacked, so every row below stood 46 px lower; a long worker name then wrapped "Recheck <name>" and put them '
    + '26 px apart (W6-2). Do not drop the case - give both cards the same rows, each the same height, and keep the '
    + 'button label short (the machine is named in its aria-label). A Recheck outcome line goes below the measured '
    + 'rows, never inside the read row (FE-W7-4). VD-200: the cards are `article.cl-setup__pool`.',
  retained_data: "LESSONS 19 / W4d-D20: the retained-data panel is the only way to reach a removed app's kept "
    + 'volume. Do not drop the width - an entry and its Delete (one line, 44 px) must stay whole and on screen. '
    + 'VD-200: the entries are `ul.cl-dep-retained > li` in the Retained data card.',
  audit_wrap: 'LESSONS 13 / FE-W7-1, FE-W7-3: one long chat title widened the audit table to ~1,900 px at 1280, '
    + 'and a URL, an IPv6 address or a long token scrolled a phone sideways. Do not drop the case - let the From and '
    + 'Target cells and the disclosure summary wrap (white-space: normal; overflow-wrap: anywhere). VD-200: on Cluster > '
    + 'Activity the trail is the `.cl-act__audit` card; its table must not scroll sideways in its own box either.',
  audit_card: 'LESSONS 13 / W7-D4: at 320 px the audit card left its action and address ~80 px and its time '
    + '~150 px, so "::ffff:10.20.30.72" broke mid-address. Do not drop the width - on a phone the time goes under '
    + 'the content, never beside it.',
  nav_labels: 'LESSONS 13 / W7-D3: at 320 px the bottom bar printed "HomeRemote consoleSystemApps and A" - each '
    + 'label wider than its column ran into the next. Do not shorten the canonical names or drop the width - let '
    + 'each column shrink (minmax(0, 1fr)) and let a long name take a second line between its words. W8-3: under '
    + 'a raised browser font the fixed 72 px bar clipped them - let the bar grow with its text, with the page '
    + 'padding following it; never a fixed px height. W8-D2: the next fix hid the names (icons only) at 125% and '
    + '150% - do not hide a name at any font; give the bar fewer columns (a second row) when a column is too narrow.',
  facts_whole_words: 'LESSONS 13 / W8-D2: at 125% text on 320 px Home printed "Administrat/or" - overflow-wrap: '
    + 'anywhere broke a word that had room on a line of its own. Do not shorten the role - let the facts take fewer '
    + 'columns when a column is narrower than a word, and break a word only when it is wider than the whole line.',
  machine_card: 'LESSONS 13 / VD-194 P1 (FE-4): no clause rendered an open machine card, so the Worker software '
    + 'row (long reasons, twelve service names) was never measured at phone width. Do not drop the row from the '
    + 'fixture - wrap its text (overflow-wrap: anywhere) and stack its columns at 720 px. VD-200: the card is '
    + '`.cf-card.is-open`, its software `.cf-software`.',
  type_floor: 'LESSONS 13 / UX-A11 - owner decision 2026-10-03: compact density stays, 12 px floor. '
    + 'Do not raise the root size or patch one component - floor the type token (max(12px, ...)) or use one.',
};

/** The pages type_floor reads, relative to the fixture directory. */
const TYPE_FLOOR_PAGES = [
  ['performance', 'page.html?s=replicated&mode=advanced'],
  // FE-4: `cluster=fleet`, or the page opens on Setup and Fleet is never read.
  // VD-200: the worker's card (the last) opened by its foot button.
  ['fleet', 'fleet.html?cluster=fleet&live=1&software=1', '.cf-card .cf-open'],
  ['deployments', 'fleet.html?cluster=deployments&deploy=1&live=1'],
  // VD-200: the endpoint cards and their keys show under the Models filter.
  ['deployments-models', 'fleet.html?cluster=deployments&deploy=1&live=1', '.cl-chips .cl-chip:has-text("Models")'],
  ['setup', 'fleet.html?cluster=setup&hostlink=1'],
  ['activity', 'fleet.html?cluster=activity'],
  ['agents', 'fleet.html?cluster=agents'],
  ['assistant', 'assistant.html'],
];

/** Visible text below 12 px in the page, as "size tag.class" with a count. */
function textBelowFloor() {
  const below = new Map();
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (!node.textContent.trim()) continue;
    const element = node.parentElement;
    const box = element.getBoundingClientRect();
    if (!box.width || !box.height || element.closest('.sr-only, [aria-hidden="true"]')) continue;
    const style = getComputedStyle(element);
    if (style.visibility === 'hidden') continue;
    const size = parseFloat(style.fontSize);
    if (size >= 12 - 0.01) continue;
    const key = `${size.toFixed(2)} px ${element.tagName.toLowerCase()}.${String(element.className).split(' ')[0]}`;
    below.set(key, (below.get(key) ?? 0) + 1);
  }
  return [...below].map(([key, count]) => `${key} x${count}`);
}

const SCOPE_TITLES = ['Hardware and system facts', 'Cooling', 'Workloads', 'Jobs', 'Assistant', 'Fleet and cluster facts'];
/**
 * Opens a fixture page and waits until `ready` is drawn - by default anything
 * in the app root, so a page whose modules never ran cannot pass a clause by
 * measuring nothing (HARNESS-1: `type_floor` waited a fixed 900 ms and would
 * have read "none below 12 px" off a blank page). page-load.mjs reloads a page
 * a failed module request left blank, once, and says so.
 */
async function load(browser, url, viewport, ready = '#root *') {
  const page = await openPage(browser, url, { viewport, ready });
  await page.waitForTimeout(600);
  return page;
}

async function axNodes(page) {
  const cdp = await page.context().newCDPSession(page);
  const { nodes } = await cdp.send('Accessibility.getFullAXTree');
  return nodes.filter((node) => !node.ignored);
}

/**
 * The focused control, and the sticky or fixed element that hides it, if one
 * does: 2.4.11 fails when the part of the control inside the viewport is
 * ENTIRELY covered, so every sampled point of that part must be covered.
 */
function focusedCover() {
  const focused = document.activeElement;
  if (!focused || focused === document.body) return null;
  const box = focused.getBoundingClientRect();
  const left = Math.max(0, box.left);
  const right = Math.min(innerWidth, box.right);
  const top = Math.max(0, box.top);
  const bottom = Math.min(innerHeight, box.bottom);
  const name = (focused.getAttribute('aria-label') || focused.textContent || '').trim().slice(0, 30);
  if (right - left < 1 || bottom - top < 1) return { name, cover: 'the viewport edge' };
  const inset = (low, high) => [low + Math.min(2, (high - low) / 2), (low + high) / 2, high - Math.min(2, (high - low) / 2)];
  const coverAt = (x, y) => {
    for (let node = document.elementFromPoint(x, y); node && node !== document.documentElement; node = node.parentElement) {
      if (node === focused || node.contains(focused) || focused.contains(node)) return null;
      if (['sticky', 'fixed'].includes(getComputedStyle(node).position)) return node;
    }
    return null;
  };
  const covers = inset(left, right).flatMap((x) => inset(top, bottom).map((y) => coverAt(x, y)));
  const cover = covers.every(Boolean) ? covers[0] : null;
  return { name, cover: cover ? `${cover.tagName.toLowerCase()}.${[...cover.classList].join('.')}` : null };
}

/** Contrast of an element's text on its own (opaque) fill. */
function contrastOf(element) {
  const parse = (value) => (value.match(/[\d.]+/g) || []).map(Number);
  const lum = ([r, g, b]) => [r, g, b].map((v) => { const c = v / 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; })
    .reduce((sum, c, i) => sum + c * [0.2126, 0.7152, 0.0722][i], 0);
  const style = getComputedStyle(element);
  const fg = lum(parse(style.color));
  const bg = lum(parse(style.backgroundColor));
  return Math.round(((Math.max(fg, bg) + 0.05) / (Math.min(fg, bg) + 0.05)) * 100) / 100;
}

function missingIdrefs() {
  return [...document.querySelectorAll('[aria-controls],[aria-labelledby],[aria-describedby]')].flatMap((element) =>
    ['aria-controls', 'aria-labelledby', 'aria-describedby'].flatMap((attribute) =>
      (element.getAttribute(attribute) || '').split(/\s+/).filter(Boolean)
        .filter((id) => !document.getElementById(id))
        .map((id) => `${attribute}=${id}`)));
}

/** Every clause this pass can hold, by the name `--a11y=` takes. */
export const A11Y_CLAUSES = Object.keys(REASONS);

/**
 * The redesign's Fleet DOM (VD-200): machine cards are `.cf-card`, each closed
 * card opened by its foot button, whose accessible name starts with this.
 */
const OPEN_CARD = /^Cluster connection, trends and machine actions/;

/**
 * `clauses` is null for every clause, or the set `--a11y=` named (the
 * contract's quicker loop); a block runs when any clause it holds is named.
 */
export async function a11yPass(browser, fleetUrl, assistantUrl, clauses = null) {
  const results = [];
  const push = (name, clause, held, detail) => results.push({ case: name, clause, pass: held, detail });
  const want = (...names) => !clauses || names.some((name) => clauses.has(name));

  if (want('fleet_reflow')) for (const width of [320, 375, 390]) {
    for (const long of [false, true]) {
      const page = await load(browser, `${fleetUrl}?cluster=fleet${long ? '&long=1' : ''}`, { width, height: 812 }, '.cf-tiles .ui-stat-tile');
      const measured = await page.evaluate(() => {
        // The machine names (each card's title) and the fleet's capacity
        // headline (the Free memory tile's figure, the first tile).
        const names = [...document.querySelectorAll('.cf-card .ui-card__titles h2, .cf-tiles > .ui-stat-tile:first-child .ui-stat-tile__value')];
        return {
          names: names.length,
          // A name taking more lines than it has words has been broken mid-word.
          brokenNames: names.filter((element) => {
            const tops = new Set();
            const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
            for (let node = walker.nextNode(); node; node = walker.nextNode()) {
              const range = document.createRange();
              range.selectNodeContents(node);
              for (const rect of range.getClientRects()) if (rect.width > 0) tops.add(Math.round(rect.bottom));
            }
            return tops.size > element.textContent.trim().split(/\s+/).length;
          }).map((element) => element.textContent.trim()),
          overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
          past: [...document.querySelectorAll('.cf-tab *')].filter((element) => element.getBoundingClientRect().right > innerWidth + 1)
            .slice(0, 3).map((element) => `${element.tagName.toLowerCase()}.${[...element.classList].join('.')}`),
        };
      });
      await page.close();
      // Two machine cards and the headline: fewer means the selectors found nothing to measure (LESSONS 2).
      push(`${width}-fleet${long ? '-long' : ''}`, 'fleet_reflow',
        measured.names === 3 && measured.overflow <= 0 && measured.brokenNames.length === 0,
        `${measured.names} names measured; overflow ${measured.overflow} px${measured.past.length ? `; past the edge: ${measured.past.join(', ')}` : ''}`
        + `${measured.brokenNames.length ? `; broken mid-word: ${measured.brokenNames.join(', ')}` : ''}`);
    }
  }

  if (want('machine_card')) for (const width of [1280, 390, 320]) {
    const page = await load(browser, `${fleetUrl}?cluster=fleet&software=1&long=1`, { width, height: 812 }, '.cf-card .cf-open');
    // The worker's card is the last one; the controller's comes first.
    await page.locator('.cf-card').last().getByRole('button', { name: OPEN_CARD }).click();
    await page.waitForSelector('.cf-card.is-open .cf-software', { timeout: LOAD_TIMEOUT_MS });
    await page.locator('.cf-card.is-open').last().scrollIntoViewIfNeeded();
    const measure = () => page.evaluate(() => ({
      rows: document.querySelectorAll('.cf-card.is-open .cf-software__component').length,
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      past: [...document.querySelectorAll('.cf-card.is-open *')].filter((element) => element.getBoundingClientRect().right > innerWidth + 1)
        .slice(0, 3).map((element) => `${element.tagName.toLowerCase()}.${[...element.classList].join('.')}`),
    }));
    const measured = await measure();
    const small = (await page.evaluate(textBelowFloor)).filter(Boolean);
    // VD-194 P1 review round 2: the profile codes sit under a disclosure that
    // starts closed, so the card is measured again with it open.
    const codes = page.locator('.cf-card.is-open .cf-software__codes summary');
    const hasCodes = (await codes.count()) === 1;
    if (hasCodes) await codes.click();
    const opened = await measure();
    const openSmall = (await page.evaluate(textBelowFloor)).filter(Boolean);
    const shown = await page.evaluate(() => document.querySelectorAll('.cf-card.is-open .cf-software__codes[open] dd').length);
    await page.close();
    const ok = (state) => state.overflow <= 0 && state.past.length === 0;
    push(`${width}-machine-card`, 'machine_card',
      measured.rows === 4 && ok(measured) && small.length === 0
        && hasCodes && shown === 3 && ok(opened) && openSmall.length === 0,
      `${measured.rows} component rows; overflow ${measured.overflow} px`
      + `${measured.past.length ? `; past the edge: ${measured.past.join(', ')}` : ''}`
      + `${small.length ? `; below 12 px: ${small.join(', ')}` : ''}`
      + `; codes open: ${hasCodes ? `${shown} shown, overflow ${opened.overflow} px` : 'no disclosure found'}`
      + `${opened.past.length ? `, past the edge: ${opened.past.join(', ')}` : ''}`
      + `${openSmall.length ? `, below 12 px: ${openSmall.join(', ')}` : ''}`);
  }

  // VD-200: the Easy/Advanced toggle is the Detail segmented control (two
  // options) and the filter chips are `.cl-chip` (four): six controls.
  if (want('compact_controls')) for (const width of [1280, 390, 320]) {
    const page = await load(browser, `${fleetUrl}?cluster=deployments&deploy=1`, { width, height: 812 }, '.cl-chips .cl-chip');
    const heights = await page.evaluate(() => [...document.querySelectorAll('.cl-detail .ui-segmented__option, .cl-chips .cl-chip')]
      .map((button) => ({ label: button.textContent.trim(), height: Math.round(button.getBoundingClientRect().height * 10) / 10 })));
    await page.close();
    const floor = width <= 720 ? 44 : 24;
    const short = heights.filter((item) => item.height < floor - 0.5);
    push(`${width}-compact-controls`, 'compact_controls', heights.length >= 6 && short.length === 0,
      `floor ${floor} px; ${heights.length} controls; ${short.map((item) => `${item.label} ${item.height}`).join(', ') || 'none short'}`);
  }

  if (want('focus_not_obscured')) for (const [width, height] of [[1280, 900], [390, 812]]) {
    const page = await load(browser, `${fleetUrl}?cluster=deployments&deploy=1&live=1`, { width, height }, '.cl-chips .cl-chip');
    const covered = [];
    const reached = new Set();
    // Twice round the tab order, so the second pass reaches each control
    // scrolling UP from the bottom - the direction the topbar covers. VD-200:
    // the endpoint cards and their keys show under the Models filter, so the
    // tab is walked on All and again on Models.
    const sweep = async (filter) => {
      const stops = await page.evaluate(() => [...document.querySelectorAll(
        'a[href], button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), summary, [tabindex="0"]')]
        .filter((element) => element.getBoundingClientRect().width > 0).length);
      for (let index = 0; index < 2 * stops + 2; index += 1) {
        await page.keyboard.press('Tab');
        const stop = await page.evaluate(focusedCover);
        if (stop) reached.add(`${filter}:${stop.name}`);
        if (stop?.cover) covered.push(`${filter}: ${stop.name} under ${stop.cover}`);
      }
    };
    await sweep('All');
    await page.locator('.cl-chips .cl-chip', { hasText: /^Models/ }).click();
    await page.waitForSelector('ul.cl-ep-keys > li', { timeout: LOAD_TIMEOUT_MS });
    await sweep('Models');
    await page.close();
    // A sweep that reached no control measured nothing (LESSONS 2).
    const onModels = [...reached].filter((name) => name.startsWith('Models:')).length;
    push(`${width}x${height}-focus`, 'focus_not_obscured', covered.length === 0 && reached.size > onModels && onModels > 0,
      `${reached.size} controls reached (${onModels} on Models); `
      + (covered.length ? [...new Set(covered)].slice(0, 5).join('; ') : 'no focused control covered'));
  }

  if (want('danger_contrast', 'idrefs_resolve', 'accessible_names')) {
    // The saved chat's More menu (VD-200, the AssistChats board) holds Delete chat.
    const page = await load(browser, assistantUrl, { width: 1280, height: 900 }, '.as-convo-bar button[aria-haspopup="menu"]');
    await page.locator('.as-convo-bar button[aria-haspopup="menu"]').click();
    const deleteChat = page.getByRole('menuitem', { name: 'Delete chat' });
    await deleteChat.waitFor({ timeout: LOAD_TIMEOUT_MS });
    // contrastOf runs in the page, so its source is handed over as text.
    const restRatios = await page.evaluate((source) => {
      const contrast = new Function(`return (${source})`)();
      return [...document.querySelectorAll('.ui-button--danger, .as-menu__danger, .as-btn-danger')]
        .filter((button) => !button.disabled && button.getBoundingClientRect().width > 0)
        .map((button) => ({ label: button.textContent.trim(), ratio: contrast(button) }));
    }, contrastOf.toString());
    await deleteChat.hover();
    await page.waitForTimeout(300);
    const hover = await page.evaluate((source) => {
      const contrast = new Function(`return (${source})`)();
      const button = [...document.querySelectorAll('button')].find((item) => item.textContent.trim() === 'Delete chat');
      return contrast(button);
    }, contrastOf.toString());
    const low = restRatios.filter((item) => item.ratio < 4.5);
    push('1280-assistant-danger', 'danger_contrast', restRatios.length > 0 && low.length === 0 && hover >= 4.5,
      `${restRatios.map((item) => `${item.label} ${item.ratio}`).join(', ')}; Delete chat under the pointer ${hover}`);

    await page.keyboard.press('Escape');
    await page.locator('.as-refinement > summary').click();
    await page.waitForTimeout(300);
    const askNames = (await axNodes(page)).filter((node) => node.role?.value === 'combobox').map((node) => node.name?.value ?? '');
    const askRefs = await page.evaluate(missingIdrefs);

    await page.getByRole('tab', { name: 'Routines' }).click();
    await page.waitForTimeout(600);
    const routinesRefs = await page.evaluate(missingIdrefs);
    push('1280-assistant-idrefs', 'idrefs_resolve', askRefs.length === 0 && routinesRefs.length === 0,
      `Ask: ${askRefs.join(', ') || 'none missing'}; Routines: ${routinesRefs.join(', ') || 'none missing'}`);

    await page.getByRole('button', { name: /New agent/ }).first().click();
    await page.locator('summary', { hasText: 'Choose information access' }).click();
    await page.waitForTimeout(300);
    const nodes = await axNodes(page);
    const checkboxes = nodes.filter((node) => node.role?.value === 'checkbox').map((node) => node.name?.value ?? '');
    const unnamed = nodes.filter((node) => ['button', 'checkbox'].includes(node.role?.value) && !(node.name?.value ?? '').trim()).length;
    const scopeNames = checkboxes.slice(0, SCOPE_TITLES.length);
    await page.close();
    const named = askNames.includes('Problem area')
      && SCOPE_TITLES.every((title, index) => scopeNames[index] === title)
      && unnamed === 0;
    push('1280-assistant-names', 'accessible_names', named,
      `select: ${askNames.join(' | ') || 'none'}; access checkboxes: ${scopeNames.join(' | ')}; unnamed: ${unnamed}`);
  }
  // VD-200: the add-machine form is `.cl-setup__add`; the machine settings are
  // the Cluster link card (`.cl-setup__pair`) and the GPU pool cards. The add
  // form now draws its fields two-up, so a label in its second column is not
  // a misaligned first-column label: the first column is each card's labels
  // within FIRST_COLUMN_BAND of its leftmost one - a label one pixel off stays
  // in it, a second column (hundreds of pixels on) does not.
  if (want('setup_alignment')) for (const width of [1440, 1024]) {
    const page = await load(browser, `${fleetUrl}?cluster=setup&hostlink=1`, { width, height: 900 }, '.cl-setup__add label');
    const lefts = await page.evaluate(() => [...document.querySelectorAll('.cl-setup__add label, .cl-setup__pair label, .cl-setup__pool label')]
      .filter((label) => label.getBoundingClientRect().width > 0)
      .map((label) => ({ text: label.textContent.trim().slice(0, 24), left: Math.round(label.getBoundingClientRect().left * 10) / 10,
        card: label.closest('.cl-setup__add') ? 'add' : label.closest('.cl-setup__pool') ? 'pool' : 'link',
        half: label.getBoundingClientRect().left < innerWidth / 2 })));
    await page.close();
    const FIRST_COLUMN_BAND = 40;
    const cardLeft = (card) => Math.min(...lefts.filter((item) => item.card === card).map((item) => item.left));
    const firstColumn = lefts.filter((item) => item.half && item.left <= cardLeft(item.card) + FIRST_COLUMN_BAND);
    const spread = firstColumn.length ? Math.max(...firstColumn.map((item) => item.left)) - Math.min(...firstColumn.map((item) => item.left)) : 0;
    // Both cards' labels: a first column of the add-machine form's alone holds nothing about the two cards.
    const settings = firstColumn.filter((item) => item.card !== 'add').length;
    push(`${width}-setup-labels`, 'setup_alignment', firstColumn.length >= 3 && settings >= 1 && spread <= 0.5,
      `${firstColumn.length} first-column labels (${settings} on a settings card); `
      + `${firstColumn.map((item) => `${item.text} ${item.left}`).join(', ')}`);
  }

  // W5-D4: the two GPU pool cards side by side - each row of one at the height
  // of the same row of the other.
  // W6-2: and with a long worker name, which once wrapped the worker's read row
  // ("Recheck <name>") to two lines and put every row below it 26 px apart.
  // FE-W7-4: and after a Recheck, whose outcome line once rendered inside the
  // worker's read row and put the cards 26 px apart - so each case is measured
  // before the click and again once the outcome line is on screen.
  const measurePoolRows = () => {
    // VD-200: each machine's pool card is `article.cl-setup__pool`; its readings are the `.cl-kv` list.
    const cards = [...document.querySelectorAll('article.cl-setup__pool')].filter((card) => card.querySelector('.cl-kv'));
    const top = (card, selector) => {
      const element = card.querySelector(selector);
      return element ? Math.round(element.getBoundingClientRect().top * 10) / 10 : null;
    };
    return cards.map((card) => ({
      left: card.getBoundingClientRect().left,
      facts: top(card, '.cl-kv'),
      label: top(card, '.cl-setup__pool-edit label'),
      actions: top(card, '.cl-setup__pool-edit .cl-setup__pool-buttons'),
    }));
  };
  if (want('pool_alignment')) for (const [width, long] of [[1512, false], [1280, false], [1512, true], [1280, true]]) {
    const page = await load(browser, `${fleetUrl}?cluster=setup&hostlink=1&pool=1${long ? '&long=1' : ''}`, { width, height: 900 }, 'article.cl-setup__pool');
    const phases = [['', await page.evaluate(measurePoolRows)]];
    await page.locator('article.cl-setup__pool').getByRole('button', { name: /^Recheck/ }).first().click();
    const shown = await page.waitForSelector('article.cl-setup__pool .recheck-outcome', { timeout: 10_000 }).then(() => true, () => false);
    phases.push(['-after-recheck', shown ? await page.evaluate(measurePoolRows) : []]);
    await page.close();
    for (const [phase, rows] of phases) {
      const sideBySide = rows.length === 2 && Math.abs(rows[0].left - rows[1].left) > 1;
      const apart = ['facts', 'label', 'actions'].filter((key) => rows.length === 2
        && (rows[0][key] === null || rows[1][key] === null || Math.abs(rows[0][key] - rows[1][key]) > 0.5));
      push(`${width}-gpu-pool-rows${long ? '-long' : ''}${phase}`, 'pool_alignment', sideBySide && apart.length === 0,
        `${phase && !rows.length ? 'no Recheck outcome appeared; ' : ''}${rows.length} cards${sideBySide ? ' side by side' : ' NOT side by side'}; `
        + ['facts', 'label', 'actions'].map((key) => `${key} ${rows.map((row) => row[key]).join('/')}`).join(', '));
    }
  }

  // VD-200: the Retained data card lists each kept volume as `ul.cl-dep-retained > li`.
  if (want('retained_data')) for (const width of [1440, 768, 375]) {
    const page = await load(browser, `${fleetUrl}?cluster=deployments&deploy=1&live=1`, { width, height: 812 }, 'ul.cl-dep-retained > li');
    const measured = await page.evaluate((source) => {
      const contrast = new Function(`return (${source})`)();
      const panel = document.querySelector('ul.cl-dep-retained').closest('.cl-card');
      const lines = (element) => {
        const tops = new Set();
        const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
        for (let node = walker.nextNode(); node; node = walker.nextNode()) {
          const range = document.createRange();
          range.selectNodeContents(node);
          for (const rect of range.getClientRects()) if (rect.width > 0) tops.add(Math.round(rect.bottom));
        }
        return tops.size;
      };
      return {
        entries: panel.querySelectorAll('ul.cl-dep-retained > li').length,
        overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        panelRight: panel.getBoundingClientRect().right,
        buttons: [...panel.querySelectorAll('button')].filter((button) => button.textContent.trim() === 'Delete data').map((button) => ({
          height: Math.round(button.getBoundingClientRect().height), lines: lines(button), ratio: contrast(button),
          right: button.getBoundingClientRect().right })),
      };
    }, contrastOf.toString());
    await page.close();
    const held = measured.entries === 2 && measured.overflow <= 0 && measured.panelRight <= width + 0.5
      && measured.buttons.length === 2
      && measured.buttons.every((button) => button.height >= 44 && button.lines === 1 && button.ratio >= 4.5 && button.right <= width + 0.5);
    push(`${width}-retained-data`, 'retained_data', held,
      `${measured.entries} entries; overflow ${measured.overflow} px; Delete data ${measured.buttons
        .map((button) => `${button.height} px, ${button.lines} line(s), ${button.ratio}:1`).join(' / ') || 'none'}`);
  }
  // The Activity page's Security audit tab draws its table in `.acty-table-wrap`;
  // VD-200 drew Cluster > Activity's as the Security audit trail card
  // (`.cl-act__audit`), a table in a `.ui-table-scroll` box.
  const AUDIT_TABLES = { activity: ['.acty-audit table', '.acty-table-wrap'], cluster: ['.cl-act__audit table', '.ui-table-scroll'] };
  if (want('audit_wrap', 'audit_card')) for (const [host, query] of [['activity', 'page=activity'], ['cluster', 'cluster=activity']]) for (const width of [1280, 390, 320]) {
    const [tableSelector, wrapSelector] = AUDIT_TABLES[host];
    const page = await load(browser, `${fleetUrl}?${query}&audit=long`, { width, height: 812 }, `${tableSelector} tbody tr`);
    const measured = await page.evaluate(([tableAt, wrapAt]) => {
      const table = document.querySelector(tableAt);
      const wrap = table.closest(wrapAt);
      return {
        rows: table.querySelectorAll('tbody tr').length,
        overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        tableOverflow: Math.round(wrap.scrollWidth - wrap.clientWidth),
        // Text spilling out of its cell counts even where the cell's own box
        // stays put: a range over each text node is what the reader sees.
        past: [...table.querySelectorAll('td')].filter((cell) => {
          const range = document.createRange();
          range.selectNodeContents(cell);
          const right = Math.max(...[...range.getClientRects()].map((rect) => rect.right), cell.getBoundingClientRect().right);
          const edge = Math.min(innerWidth, wrap.getBoundingClientRect().right);
          return right > edge + 1 || cell.scrollWidth > cell.clientWidth + 1;
        }).slice(0, 3).map((cell) => cell.textContent.trim().slice(0, 24)),
        // W7-D4: the share of the stacked card the action column gets.
        actionShare: Math.min(...[...table.querySelectorAll('tbody tr')].map((row) =>
          row.querySelector('td').getBoundingClientRect().width / row.getBoundingClientRect().width)),
      };
    }, [tableSelector, wrapSelector]);
    await page.close();
    if (width <= 390) {
      push(`${width}-audit-card-${host}`, 'audit_card', measured.actionShare >= 0.6,
        `the action column takes ${Math.round(measured.actionShare * 100)}% of the narrowest card`);
    }
    push(`${width}-audit-wrap-${host}`, 'audit_wrap',
      measured.rows === 4 && measured.overflow <= 0 && measured.tableOverflow <= 0 && measured.past.length === 0,
      `${measured.rows} rows; page overflow ${measured.overflow} px; table overflow ${measured.tableOverflow} px`
      + `${measured.past.length ? `; past the edge: ${measured.past.join(', ')}` : ''}`);
  }
  // W8-3: the same bar under a raised browser font (the setting, through CDP,
  // not a page style): at 20 and 24 px the two-line labels made items 70-79 px
  // tall inside a fixed 72 px bar with overflow hidden, and clipped them.
  // W8-D2: the retest raised the root size through the page's own style
  // (`page`), not the browser setting - both must keep every name visible.
  if (want('nav_labels')) for (const [width, font, via = 'setting'] of [[320, 16], [375, 16], [320, 20], [375, 20], [320, 24], [375, 24],
    [320, 20, 'page'], [375, 24, 'page']]) {
    const before = async (opening) => {
      if (via !== 'setting') return;
      const cdp = await opening.context().newCDPSession(opening);
      await cdp.send('Page.enable');
      await cdp.send('Page.setFontSizes', { fontSizes: { standard: font, fixed: Math.round(font * 13 / 16) } });
    };
    const page = await openPage(browser, `${fleetUrl}?nav=1`, { viewport: { width, height: 812 }, before });
    if (via === 'page') await page.evaluate((size) => { document.documentElement.style.fontSize = `${size}px`; }, font);
    await page.waitForTimeout(600);
    await page.waitForSelector('.sidebar__nav--mobile .nav-item__label', { timeout: LOAD_TIMEOUT_MS });
    const measured = await page.evaluate(() => {
      const allLabels = [...document.querySelectorAll('.sidebar__nav--mobile > .nav-item .nav-item__label, .sidebar__nav--mobile > details > summary .nav-item__label')];
      // W8-D2: a label the bar hides (icons only, the name kept for screen
      // readers) is reported and fails - every name must be visible.
      const hidden = (label) => label.getBoundingClientRect().width <= 1 || label.getBoundingClientRect().height <= 1
        || getComputedStyle(label).visibility === 'hidden' || getComputedStyle(label).display === 'none';
      const iconOnly = allLabels.filter(hidden).map((label) => label.textContent.trim());
      const labels = allLabels.filter((label) => !hidden(label));
      const textBoxes = (element) => {
        const range = document.createRange();
        range.selectNodeContents(element);
        return [...range.getClientRects()].filter((rect) => rect.width > 0);
      };
      const bar = document.querySelector('.sidebar').getBoundingClientRect();
      const items = labels.map((label) => ({ text: label.textContent.trim(), boxes: textBoxes(label),
        cut: label.scrollWidth > label.clientWidth + 1 || label.scrollHeight > label.clientHeight + 1
          // The bar clips what runs past it, so a label below its bottom edge is cut too.
          || textBoxes(label).some((box) => box.bottom > bar.bottom + 0.5 || box.top < bar.top - 0.5) }));
      const overlaps = [];
      items.forEach((item, index) => items.slice(index + 1).forEach((other) => {
        if (item.boxes.some((a) => other.boxes.some((b) => a.left < b.right - 0.5 && b.left < a.right - 0.5
          && a.top < b.bottom - 0.5 && b.top < a.bottom - 0.5))) overlaps.push(`${item.text} / ${other.text}`);
      }));
      // Each destination's whole box (icon and name) inside the bar, and the
      // page's bottom padding at least the bar's real height, so the bar never
      // covers the end of the page.
      const itemsOut = [...document.querySelectorAll('.sidebar__nav--mobile > .nav-item, .sidebar__nav--mobile > details > summary')]
        .filter((item) => { const box = item.getBoundingClientRect(); return box.bottom > bar.bottom + 0.5 || box.top < bar.top - 0.5; })
        .map((item) => item.textContent.trim());
      const workspace = document.querySelector('.workspace');
      const padding = workspace ? parseFloat(getComputedStyle(workspace).paddingBottom) : 0;
      return {
        rootFont: parseFloat(getComputedStyle(document.documentElement).fontSize),
        iconOnly,
        unnamed: iconOnly.filter((text) => !text).length,
        total: allLabels.length,
        barHeight: Math.round(bar.height),
        itemsOut,
        paddingShort: padding < bar.height - 0.5,
        count: items.length,
        overlaps,
        cut: items.filter((item) => item.cut).map((item) => item.text),
        broken: items.filter((item) => new Set(item.boxes.map((box) => Math.round(box.bottom))).size > item.text.split(/\s+/).length)
          .map((item) => item.text),
      };
    });
    await page.close();
    push(`${width}-nav-labels${font === 16 ? '' : `-font-${font}`}${via === 'page' ? '-page-style' : ''}`, 'nav_labels',
      measured.rootFont === font && measured.total === 5 && measured.unnamed === 0
        && measured.iconOnly.length === 0 && !measured.overlaps.length && !measured.cut.length
        && !measured.broken.length && !measured.itemsOut.length && !measured.paddingShort,
      `root ${measured.rootFont} px; bar ${measured.barHeight} px; ${measured.count} labels shown`
      + `${measured.iconOnly.length ? `, name hidden (icon only): ${measured.iconOnly.join(', ')}` : ''}; `
      + `overlapping: ${measured.overlaps.join(', ') || 'none'}; cut: ${measured.cut.join(', ') || 'none'}; `
      + `broken mid-word: ${measured.broken.join(', ') || 'none'}; past the bar: ${measured.itemsOut.join(', ') || 'none'}`
      + `${measured.paddingShort ? '; the page bottom padding is shorter than the bar' : ''}`);
  }
  // W8-D2: Home's headline facts at a phone width under a raised text size:
  // "Administrator" broke mid-word ("Administrat/or") at 125% on 320 px.
  if (want('facts_whole_words')) for (const [width, font] of [[320, 16], [375, 16], [320, 20], [375, 20], [320, 24], [375, 24]]) {
    const before = async (opening) => {
      const cdp = await opening.context().newCDPSession(opening);
      await cdp.send('Page.enable');
      await cdp.send('Page.setFontSizes', { fontSizes: { standard: font, fixed: Math.round(font * 13 / 16) } });
    };
    const page = await openPage(browser, `${fleetUrl}?page=home-facts`,
      { viewport: { width, height: 812 }, before, ready: '.system-strip__facts dd' });
    await page.waitForTimeout(300);
    const measured = await page.evaluate(() => {
      const broken = [...document.querySelectorAll('.system-strip__facts dd')].filter((element) => {
        const range = document.createRange();
        range.selectNodeContents(element);
        const lines = new Set([...range.getClientRects()].filter((rect) => rect.width > 0).map((rect) => Math.round(rect.bottom)));
        return lines.size > element.textContent.trim().split(/\s+/).length;
      }).map((element) => element.textContent.trim());
      return { broken, overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        count: document.querySelectorAll('.system-strip__facts dd').length };
    });
    await page.close();
    push(`${width}-home-facts${font === 16 ? '' : `-font-${font}`}`, 'facts_whole_words',
      measured.count === 4 && !measured.broken.length && measured.overflow <= 0,
      `${measured.count} facts; broken mid-word: ${measured.broken.join(', ') || 'none'}; overflow ${measured.overflow} px`);
  }
  const fixtureBase = fleetUrl.replace(/fleet\.html$/, '');
  if (want('type_floor')) for (const width of [1280, 375]) {
    const found = [];
    for (const [name, query, open] of TYPE_FLOOR_PAGES) {
      const page = await load(browser, fixtureBase + query, { width, height: 900 });
      await page.waitForTimeout(900);
      if (open) {
        await page.locator(open).last().click();
        await page.waitForTimeout(300);
      }
      found.push(...(await page.evaluate(textBelowFloor)).map((item) => `${name}: ${item}`));
      await page.close();
    }
    push(`${width}-type-floor`, 'type_floor', found.length === 0,
      found.length ? found.slice(0, 8).join('; ') + (found.length > 8 ? `; and ${found.length - 8} more` : '') : `${TYPE_FLOOR_PAGES.length} pages, none below 12 px`);
  }
  return results;
}

export function printA11y(results) {
  console.log('W5-S4 AUDIT CLAUSES');
  for (const row of results) {
    console.log([row.case.padEnd(26), row.clause.padEnd(20), row.pass ? 'yes' : 'NO', row.detail].join(' '));
    if (!row.pass) console.log(`  ${REASONS[row.clause]}`);
  }
}
