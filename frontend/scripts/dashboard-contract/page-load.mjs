/**
 * How every page of the layout contract is opened (HARNESS-1, LESSONS 3 and 23).
 *
 * A contract page is an ES-module graph of a few hundred requests to the
 * contract's own Vite server. When one of those requests fails in the browser
 * (seen as `net::ERR_NO_BUFFER_SPACE` on `/@react-refresh` with the machine
 * loaded), the graph never runs, the page stays blank for good, and a wait for
 * anything on it used to sit out its whole 180 s and then name only the
 * selector. A solo re-run passed, so it read as a flake.
 *
 * `openPage` watches the page's own requests while it loads:
 *   - a request that got no answer at all (a transport failure), or the 504
 *     Vite answers for a dependency it has re-optimised, loads the page once
 *     more, at once, and the reload is recorded in `loadNotes` so the run says
 *     it happened; a second failure fails the run, naming the request;
 *   - any other answer of 400 or above is the server refusing or failing to
 *     build a file (a missing module is a 404, which Chrome also reports as a
 *     failed request): the run fails at once, naming the status and the file,
 *     because loading it again gets the same answer;
 *   - requests to any other origin never count: only the page's own server
 *     decides whether the page loads again;
 *   - a page whose requests all completed but which never shows `ready` fails
 *     with what the page does show, not just the selector.
 * Nothing here waits longer than the wait it replaced.
 */

export const LOAD_TIMEOUT_MS = 180_000;

/** Every reload `openPage` made in this run, for the run's own output. */
export const loadNotes = [];

/** Says every reload this run made; called whether the run passed or threw. */
export function printLoadNotes(write = console.log) {
  for (const note of loadNotes) write(`PAGE LOADED AGAIN ${note}`);
}

/** Runs `run` and says the reloads afterwards, whether it returned or threw. */
export async function withLoadNotes(run, write = console.log) {
  try {
    return await run();
  } finally {
    printLoadNotes(write);
  }
}

const REASON = 'LESSONS 3 / HARNESS-1';
/** The request kinds a page cannot draw without. */
const NEEDED = new Set(['document', 'script', 'stylesheet']);

/**
 * Which way a failed load goes: `reload` (the browser could not complete a
 * request), `fail` (the server answered one with an error) or `wait` (nothing
 * failed). Pure, so it is tested without a browser.
 */
export function classifyLoad(failures) {
  if (failures.some((failure) => failure.kind === 'server')) return 'fail';
  if (failures.some((failure) => failure.kind === 'transport')) return 'reload';
  return 'wait';
}

/** Records the requests a page cannot draw without that did not complete. */
export function watchLoad(page, origin) {
  const failures = [];
  let notify = () => undefined;
  const failed = new Promise((resolve) => { notify = resolve; });
  const ours = (request) => request.url().startsWith(origin) && NEEDED.has(request.resourceType());
  const answered = new Set();
  const onAnswer = (url, status) => {
    if (answered.has(url)) return;
    answered.add(url);
    // 504 is Vite's "outdated optimized dependency": the page is meant to load again.
    failures.push({ kind: status === 504 ? 'transport' : 'server', url, why: `HTTP ${status}` });
    notify();
  };
  const onResponse = (response) => {
    if (response.status() < 400 || !ours(response.request())) return;
    onAnswer(response.url(), response.status());
  };
  const onFailed = async (request) => {
    if (!ours(request)) return;
    // Chrome reports a module answered 404 as a failed request (net::ERR_ABORTED)
    // too; it had an answer, so it is the file that is missing, not the network.
    const response = await request.response().catch(() => null);
    if (response && response.status() >= 400) {
      onAnswer(request.url(), response.status());
      return;
    }
    failures.push({ kind: 'transport', url: request.url(), why: request.failure()?.errorText ?? 'failed' });
    notify();
  };
  page.on('requestfailed', onFailed);
  page.on('response', onResponse);
  return {
    failures,
    failed,
    stop() {
      page.off('requestfailed', onFailed);
      page.off('response', onResponse);
    },
  };
}

const describe = (failures) => failures.map((failure) => `${failure.url} (${failure.why})`).join(', ');

/**
 * Opens `url` in a new page and waits until `ready` (a selector, or a function
 * of the page that resolves when it is drawn) - see the module docstring.
 * `before(page)` runs before each load, for settings the page must load with.
 */
export async function openPage(browser, url, { viewport, ready = '#root *', before, timeout = LOAD_TIMEOUT_MS }) {
  const origin = new URL(url).origin;
  const earlier = [];
  for (let attempt = 1; ; attempt += 1) {
    const page = await browser.newPage({ viewport });
    const watch = watchLoad(page, origin);
    const drawn = (async () => {
      if (before) await before(page);
      await page.goto(url, { waitUntil: 'networkidle', timeout });
      if (typeof ready === 'function') await ready(page);
      else await page.waitForSelector(ready, { timeout });
      return 'drawn';
    })();
    // A rejection after the race is decided is the closed page's, not news.
    drawn.catch(() => undefined);
    let outcome;
    let waitError = null;
    try {
      outcome = await Promise.race([drawn, watch.failed.then(() => 'failed')]);
    } catch (error) {
      outcome = 'error';
      waitError = error;
    }
    watch.stop();
    if (outcome === 'drawn') {
      if (earlier.length) loadNotes.push(`${url}: loaded once more after ${describe(earlier)}`);
      return page;
    }
    const shown = await page.evaluate(() => document.body?.innerText.trim().slice(0, 160) ?? '').catch(() => '');
    await page.close();
    const route = classifyLoad(watch.failures);
    if (route === 'reload' && attempt === 1) {
      earlier.push(...watch.failures);
      continue;
    }
    if (route === 'fail') {
      throw new Error(`${REASON}: ${url} did not draw - the contract's Vite server answered ${describe(watch.failures)}. `
        + 'The file is missing or failed to build; loading it again gets the same answer, so fix the file or its import.');
    }
    if (route === 'reload') {
      throw new Error(`${REASON}: ${url} did not draw - ${describe(earlier)} failed in the browser, and on the `
        + `one reload ${describe(watch.failures)} failed too. Without its modules the page stays blank, so a longer `
        + 'wait cannot help; find why the browser cannot reach the contract\'s own server.');
    }
    throw new Error(`${REASON}: ${url} loaded every module, but what the contract waits for never appeared `
      + `(${waitError?.message.split('\n')[0] ?? 'no error'}). The page shows: "${shown || '(nothing)'}". `
      + 'Raising the timeout is not the fix; find what the page is waiting on.');
  }
}
