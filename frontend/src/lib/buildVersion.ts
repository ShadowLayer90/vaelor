import { useEffect, useState } from "react";

/**
 * Is this open page still the version installed on the appliance? (ACC-167)
 *
 * A page left open across an update keeps running the JavaScript it loaded:
 * the ten-second refresh fetches data, never code. So after a deploy the owner
 * could be reading a fixed screen through the unfixed bundle with nothing to
 * say so.
 *
 * The two builds are told apart by what each already carries, so nothing new
 * is built or stored: the build gives its entry script a content-hashed name
 * (`/v2/assets/entry-<hash>.js`), the open document holds the name it was
 * loaded with, and the appliance's `index.html` (served `no-cache`) names the
 * one installed now. When they differ, the page says so and offers a reload.
 * It never reloads by itself: a reload drops whatever the owner was doing.
 *
 * A check that could not be made (offline, an error page, the dev server,
 * where there is no hashed entry) claims nothing.
 *
 * Only the entry script's name is compared, and that is enough: the name is
 * a hash of the entry's content, and the entry imports every other chunk by
 * its own content-hashed name, so any change to any shipped chunk changes the
 * entry's name. The shell runs this once for every screen (`BuildReloadNotice`).
 */

/** How often the installed version is looked at while the page is visible. */
export const BUILD_CHECK_MS = 60_000;

const ENTRY_FILE = /assets\/entry-[A-Za-z0-9_-]+\.js$/;

/** The built entry script named by an `index.html` body, or null when it names none. */
export function entryOf(html: string): string | null {
  const parsed = new DOMParser().parseFromString(html, "text/html");
  return runningEntry(parsed);
}

/** The built entry script a document was loaded with, or null (the dev server has none). */
export function runningEntry(doc: Document = document): string | null {
  for (const script of Array.from(doc.querySelectorAll('script[type="module"][src]'))) {
    const source = script.getAttribute("src") ?? "";
    if (ENTRY_FILE.test(source)) return source;
  }
  return null;
}

/**
 * True once the appliance is found to serve a different build from the one
 * this page is running. It stays true: a page that was out of date does not
 * become current again without a reload.
 */
export function useBuildFreshness(): { stale: boolean } {
  const [stale, setStale] = useState(false);

  useEffect(() => {
    const running = runningEntry();
    if (running === null || stale) return undefined;
    // The page the entry was loaded from: the entry's own address without its
    // file name ("/v2/assets/entry-x.js" is served by "/v2/").
    const index = running.replace(ENTRY_FILE, "");
    let cancelled = false;
    const check = () => {
      if (document.hidden) return;
      fetch(index, { cache: "no-store", credentials: "same-origin" })
        .then((response) => (response.ok ? response.text() : null))
        .then((html) => {
          const installed = html === null ? null : entryOf(html);
          if (!cancelled && installed !== null && installed !== running) setStale(true);
        })
        // Could not ask: nothing is known, so nothing is said.
        .catch(() => undefined);
    };
    check();
    const timer = globalThis.setInterval(check, BUILD_CHECK_MS);
    document.addEventListener("visibilitychange", check);
    return () => {
      cancelled = true;
      globalThis.clearInterval(timer);
      document.removeEventListener("visibilitychange", check);
    };
  }, [stale]);

  return { stale };
}
