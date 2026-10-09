/**
 * The guard for the error-behind-the-modal class (VD-189; W4d-D16, D18 and the
 * W5 review's audit of ~30 dialogs). Installed for EVERY test by setup.ts, so
 * every dialog test is held by it without being written for it.
 *
 * While a modal (`[aria-modal="true"]`) is open, the page under it is inert and
 * aria-hidden: a message that appears there is neither seen nor announced. So
 * a message that APPEARS (is added, or has its text changed) outside the
 * topmost open modal while one is open is recorded, and the test that caused it
 * fails in afterEach. "A message" is an alert (role="alert"): a danger or
 * warning Notice, an error OperationFeedback, a form error. An info or success
 * notice (role="status") is not one - page content that renders while a dialog
 * is open is not a refusal, and the first version, which counted them, flagged
 * the Assistant's model sentence on every dialog test. So a refusal shown as an
 * INFO notice is invisible here: a refusal is an alert, or it is a defect of
 * its own.
 *
 * What it cannot see: a dialog action that no test ever fails. Each dialog
 * whose action can fail carries a test that fails it; this sentinel is what
 * makes that test red if the refusal moves back onto the page.
 */
const MESSAGE = '[role="alert"]';

export const SENTINEL_REASON = "LESSONS 19 / VD-189: a message appeared on the page UNDER an open modal, where the page "
  + "is inert and aria-hidden - the owner sees a dialog that did nothing. Pass the refusal to the dialog "
  + "(ModalShell/ConfirmDialog `error`, useModalAction) or close the dialog on failure. Do not silence this check.";

const MODAL = '[aria-modal="true"]';

function dialogName(modal: Element): string {
  const labelledBy = modal.getAttribute("aria-labelledby");
  // Read inside the dialog first: a dialog just removed is out of the document.
  const label = labelledBy
    ? ([...modal.querySelectorAll("[id]")].find((node) => node.id === labelledBy) ?? document.getElementById(labelledBy))?.textContent
    : modal.getAttribute("aria-label");
  return (label ?? "an unnamed dialog").trim().slice(0, 80);
}

function messagesAround(node: Node): Element[] {
  const element = node instanceof Element ? node : node.parentElement;
  if (!element) return [];
  const own = element.closest(MESSAGE);
  return [...(own ? [own] : []), ...(node instanceof Element ? element.querySelectorAll(MESSAGE) : [])];
}

let suiteSentinel: { take(): string[] } | null = null;

/**
 * For the sentinel's OWN tests, which make the class happen on purpose: drops
 * what the suite-wide sentinel recorded so far. Nothing else may call it.
 */
export function discardSuiteRecords() {
  suiteSentinel?.take();
}

let armed = false;
const caught: string[] = [];
let enforcements = 0;

/**
 * What setup.ts's afterEach does with the sentinel's records: throw, failing
 * the test that put a message behind a modal. Here rather than inline in
 * setup.ts so the suite can prove the hook runs and throws
 * (modalErrorSentinelEnforced.test.tsx); a guard nobody can see fail is not one.
 */
export function enforceSuiteSentinel(behindModal: string[]) {
  enforcements += 1;
  if (!behindModal.length) return;
  const error = new Error(`${SENTINEL_REASON}\n  shown behind the modal:\n    ${behindModal.join("\n    ")}`);
  if (armed) {
    armed = false;
    caught.push(error.message);
    return;
  }
  throw error;
}

/**
 * For the enforcement test only: the NEXT enforcement records its failure
 * instead of throwing it, so a following test can read that it happened.
 */
export function expectSuiteSentinelFailure() {
  armed = true;
}

/** For the enforcement test only: what an armed enforcement caught, and how many ran. */
export function takeSuiteSentinelFailures() {
  return { caught: caught.splice(0), enforcements };
}

/** The one sentinel setup.ts installs for the whole suite. */
export function installSuiteSentinel() {
  suiteSentinel = installModalErrorSentinel();
  return suiteSentinel;
}

export function installModalErrorSentinel() {
  const found: string[] = [];
  /** The modals open at the end of the last batch read, in the order they opened. */
  let open: Element[] = [...document.querySelectorAll(MODAL)];

  /*
   * The observer reports in batches after the DOM has moved on: a test's
   * synchronous render and click land in one batch, and by the time it is read
   * the first record's container already holds the dialog the second record
   * added. So every node is dated by the LAST record that inserted it or an
   * ancestor of it, and the modals open at each date are replayed from those
   * dates - never read off the document as it is now.
   */
  const record = (records: MutationRecord[]) => {
    if (!records.length) return;
    // Text nodes count: an always-mounted alert region whose text goes from
    // empty to a refusal adds a text node, not an element (review round 2).
    const inserted = records.map((change) => [...change.addedNodes].filter((node) => node instanceof Element || node instanceof Text));
    const deleted = records.map((change) => [...change.removedNodes].filter((node): node is Element => node instanceof Element));
    const dated = (node: Node, roots: Node[][]) => {
      let date = -1;
      roots.forEach((list, index) => { if (list.some((root) => root === node || root.contains(node))) date = index; });
      return date;
    };
    const candidates = new Map<Element, number>();
    const modalsAdded = new Map<Element, number>();
    inserted.forEach((list) => list.forEach((root) => {
      if (root instanceof Element) {
        [...(root.matches(MODAL) ? [root] : []), ...root.querySelectorAll(MODAL)]
          .forEach((modal) => modalsAdded.set(modal, dated(modal, inserted)));
      }
      messagesAround(root).forEach((message) => candidates.set(message, Math.max(candidates.get(message) ?? -1, dated(root, inserted))));
    }));
    records.forEach((change, index) => {
      if (change.type !== "characterData") return;
      messagesAround(change.target).forEach((message) => candidates.set(message, Math.max(candidates.get(message) ?? -1, index)));
    });
    const modalsRemoved = new Map<Element, number>();
    [...open, ...modalsAdded.keys()].forEach((modal) => {
      const date = dated(modal, deleted);
      if (date >= 0) modalsRemoved.set(modal, date);
    });
    // A dialog re-mounted (by `key`) in this batch is still open to the owner:
    // the removed one is paired with the added one of the same name, which
    // stands in for it from the moment it was removed (review round 2).
    const replacedBy = new Map<Element, Element>();
    for (const [removed, removedAt] of modalsRemoved) {
      const successor = [...modalsAdded].find(([added, addedAt]) => added !== removed && addedAt >= removedAt
        && !modalsRemoved.has(added) && dialogName(added) === dialogName(removed));
      if (successor) replacedBy.set(removed, successor[0]);
    }
    const remounted = new Set(deleted.flat().flatMap((root) => messagesAround(root)).map((message) => (message.textContent ?? "").trim()));
    const openAt = (date: number) => [
      ...open.map((modal) => ({ modal, from: -1 })),
      ...[...modalsAdded].map(([modal, from]) => ({ modal, from })).sort((a, b) => a.from - b.from),
    ].flatMap(({ modal, from }) => {
      if (from > date) return [];
      const removedAt = modalsRemoved.get(modal) ?? Infinity;
      if (removedAt <= date && removedAt > from) return replacedBy.has(modal) ? [replacedBy.get(modal)!] : [];
      return [replacedBy.get(modal) ?? modal];
    });
    /** The node this batch inserted that holds `node`, at that date. */
    const insertedRoot = (node: Node) => inserted.flat().find((root) => root === node || root.contains(node));

    for (const [message, date] of candidates) {
      const text = (message.textContent ?? "").trim();
      const top = openAt(date).at(-1);
      if (!text || !top || top.contains(message)) continue;
      // A dialog closed in the same batch (closed on failure, its refusal then
      // on the page) is fine; a message re-mounted unchanged is not new.
      if (modalsRemoved.has(top) || remounted.has(text)) continue;
      // A page that mounts with its alert and a dialog in one insertion (a
      // first render) put nothing behind the dialog (review round 2).
      const root = modalsAdded.has(top) ? insertedRoot(top) : undefined;
      if (root && root !== top && root.contains(message)) continue;
      found.push(`"${text.slice(0, 160)}" behind "${dialogName(top)}"`);
    }
    open = [...open, ...modalsAdded.keys()].filter((modal) => modal.isConnected);
  };
  const observer = new MutationObserver(record);
  observer.observe(document.body, { childList: true, subtree: true, characterData: true });
  return {
    /** The messages recorded since the last take, and the record cleared. */
    take(): string[] {
      record(observer.takeRecords());
      return found.splice(0);
    },
    disconnect: () => observer.disconnect(),
  };
}
