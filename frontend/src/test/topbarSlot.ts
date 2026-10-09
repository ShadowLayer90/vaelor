import { afterEach, beforeEach } from "vitest";
import { TOPBAR_PAGE_SLOT_ID } from "../lib/topbarSlot";

/**
 * The shell's top-bar page slot, drawn in the document around every test in
 * the file that calls this, as Overview draws it around every page. A page
 * rendered alone puts its pill and actions there (`TopbarPageActions`), and
 * without it they are not drawn at all. Returns a reader for the slot.
 */
export function withTopbarSlot(): () => HTMLElement {
  let slot: HTMLElement | null = null;
  beforeEach(() => {
    slot = document.createElement("div");
    slot.id = TOPBAR_PAGE_SLOT_ID;
    slot.className = "topbar__page";
    slot.style.display = "contents";
    document.body.prepend(slot);
  });
  afterEach(() => {
    slot?.remove();
    slot = null;
  });
  return () => {
    if (!slot) throw new Error("withTopbarSlot(): the slot exists only while a test runs");
    return slot;
  };
}

/** The reason a page's top-bar test gives when it fails. */
export const TOPBAR_WHY = "LESSONS 6 / VD-200: the boards draw each page's pill and actions in the console's top bar, "
  + "once. A copy in the page body is a second answer that can disagree; do not move them back.";
