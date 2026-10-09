import "@testing-library/jest-dom/vitest";
import { cleanup, configure } from "@testing-library/react";
import { afterEach } from "vitest";
import { clearApiRequestCache } from "../lib/api";
import { enforceSuiteSentinel, installSuiteSentinel } from "./modalErrorSentinel";

/*
 * How long `findBy*` and `waitFor` are allowed to wait for a condition.
 *
 * Testing Library's default is one second of wall-clock time, and this suite
 * runs a hundred jsdom environments in parallel on a machine that is often
 * also building or running the backend tests. One second of *contended* time
 * is not one second of work: `FleetCenter` timed out waiting for its first
 * render roughly one full run in six, with nothing wrong and a different test
 * named each time. An intermittently red suite is worse than a slow one,
 * because it teaches everyone reading it to re-run rather than to look.
 *
 * This weakens nothing. A condition that never becomes true still fails; it
 * fails later, and only on runs that were going to fail anyway. Vitest's
 * `testTimeout` (vite.config.ts) is at least twice this, so even a test that
 * waits twice has its genuine miss reported by Testing Library, which names
 * the element, rather than as a bare test timeout that does not
 * (src/test/timeoutBudget.test.ts holds the ratio).
 */
configure({ asyncUtilTimeout: 4_000 });

/*
 * VD-189: a message that appears on the page under an open modal fails the
 * test that caused it (see modalErrorSentinel.ts). Node-environment test
 * files share this setup and have no document.
 */
const sentinel = typeof document !== "undefined" ? installSuiteSentinel() : null;

afterEach(() => {
  const behindModal = sentinel?.take() ?? [];
  cleanup();
  clearApiRequestCache();
  // Drafts persist in sessionStorage on purpose (#150); between tests that
  // persistence is state leaking from one test into the next. Guarded
  // because node-environment test files share this setup and have none.
  if (typeof sessionStorage !== "undefined") sessionStorage.clear();
  // localStorage is the same hazard: the application-research resume key
  // (vaelor.application-research.admin) and lighting state live there, and a
  // test that sets one leaked into the next under full-suite ordering, so a
  // container/lighting test failed only when it ran after another - the
  // ordering flake reported independently during the a101 feature work. Clear
  // it too so every test starts from empty storage.
  if (typeof localStorage !== "undefined") localStorage.clear();
  enforceSuiteSentinel(behindModal);
});
