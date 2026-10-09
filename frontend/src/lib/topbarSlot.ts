import { useEffect, useState, useSyncExternalStore, type ReactNode } from "react";
import { createPortal } from "react-dom";

/*
 * The console's top bar (VD-200): the breadcrumb on the left, and on the
 * right the open page's status pill and its one or two actions, then the
 * shell's own connection pill, search and account. The shell owns the bar;
 * a page fills two parts of it from here.
 */

/**
 * The id of the empty element at the start of the top bar's actions. The
 * shell renders it and nothing inside it; each page puts its pill and Reload
 * there through `TopbarPageActions`.
 */
export const TOPBAR_PAGE_SLOT_ID = "topbar-page-slot";

/**
 * A page's top-bar items, drawn in the shell's slot. The items stay in the
 * component whose state they act on. Until the slot is found this renders
 * nothing, never a second copy in the page body.
 */
export function TopbarPageActions({ children }: { children: ReactNode }) {
  const [slot, setSlot] = useState<HTMLElement | null>(null);
  useEffect(() => {
    const found = document.getElementById(TOPBAR_PAGE_SLOT_ID);
    if (found) { setSlot(found); return undefined; }
    // A slot drawn after this page mounted (a shell that renders later) is
    // still found: watch for it once, then stop watching.
    const watch = new MutationObserver(() => {
      const late = document.getElementById(TOPBAR_PAGE_SLOT_ID);
      if (!late) return;
      watch.disconnect();
      setSlot(late);
    });
    watch.observe(document.body, { childList: true, subtree: true });
    return () => watch.disconnect();
  }, []);
  return slot ? createPortal(children, slot) : null;
}

/*
 * The breadcrumb's second part, reported by the page that owns its words: the
 * open app in Apps and AI, the Assistant's tab and view, a Settings tab, a
 * Remote console view. Some of these are not in the address at all, and the
 * rest move with `pushState`/`replaceState`, which tell the shell nothing.
 * Cluster and System are read from the address by the shell instead, on the
 * events those two pages send. The newest report is the one shown; when its
 * page unmounts, the report before it (a page still mounted, such as a view
 * that drew a dialog over itself) is shown again rather than nothing.
 */
const reports: Array<{ owner: object; parts: readonly string[] }> = [];
let reported: { owner: object; parts: readonly string[] } | null = null;
const listeners = new Set<() => void>();
const PART_SEPARATOR = "\u001f";

function announce() {
  for (const listener of listeners) listener();
}

/** Says where on its page the reader is, as the parts after the page's name. */
export function usePagePlace(parts: readonly string[] | null | undefined): void {
  const key = parts && parts.length > 0 ? parts.join(PART_SEPARATOR) : "";
  useEffect(() => {
    if (!key) return undefined;
    const owner = {};
    reports.push({ owner, parts: key.split(PART_SEPARATOR) });
    reported = reports[reports.length - 1];
    announce();
    return () => {
      const index = reports.findIndex((report) => report.owner === owner);
      if (index >= 0) reports.splice(index, 1);
      const next = reports[reports.length - 1] ?? null;
      if (next === reported) return;
      reported = next;
      announce();
    };
  }, [key]);
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

/** The shell's side: what the shown page last said about its place, or null. */
export function useReportedPagePlace(): readonly string[] | null {
  return useSyncExternalStore(subscribe, () => reported?.parts ?? null);
}

