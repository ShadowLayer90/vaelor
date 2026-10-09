import { type RefObject, useEffect, useRef } from "react";

const FOCUSABLE_SELECTOR = [
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[href]",
  "[tabindex]:not([tabindex='-1'])",
  "[contenteditable='true']",
].join(", ");
type InertSnapshot = {
  element: HTMLElement;
  ariaHidden: string | null;
  inertAttribute: string | null;
  inertProperty: boolean | undefined;
  hasInertProperty: boolean;
};

function focusableElements(root: HTMLElement | null) {
  return Array.from(
    root?.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR) ?? [],
  ).filter((element) =>
    !element.hidden
    && element.getAttribute("aria-hidden") !== "true"
    && element.getAttribute("inert") === null
  );
}

function isolationTargets(root: HTMLElement) {
  const targets: HTMLElement[] = [];
  let current: Element | null = root;
  while (current?.parentElement) {
    const parentElement: HTMLElement = current.parentElement;
    for (const sibling of Array.from(parentElement.children)) {
      if (
        sibling !== current
        && sibling instanceof HTMLElement
        && !["SCRIPT", "STYLE", "LINK", "META", "TITLE"].includes(sibling.tagName)
        && !sibling.hasAttribute("aria-modal")
        && !sibling.querySelector("[aria-modal='true']")
      ) {
        targets.push(sibling);
      }
    }
    if (parentElement === document.body) break;
    current = parentElement;
  }
  return targets;
}

/*
 * How many open dialogs hide each element, and what the page had set on it
 * before the first one did. A dialog that opens before the previous one closes
 * (an approval inside a wizard, a confirm replacing a form) hides the same
 * elements twice; restoring a per-dialog snapshot then put "hidden" back, or
 * took it away while the newer dialog was still open, and the page beneath
 * stayed inert once every dialog was gone (VD-200 merge, LESSONS 22). The
 * element is given back only when the last dialog hiding it closes.
 */
const hiddenBy = new Map<HTMLElement, { count: number; snapshot: InertSnapshot }>();

function hide(element: HTMLElement) {
  const held = hiddenBy.get(element);
  if (held) {
    held.count += 1;
  } else {
    const inertElement = element as HTMLElement & { inert?: boolean };
    const hasInertProperty = "inert" in inertElement;
    hiddenBy.set(element, {
      count: 1,
      snapshot: {
        element,
        ariaHidden: element.getAttribute("aria-hidden"),
        inertAttribute: element.getAttribute("inert"),
        inertProperty: hasInertProperty ? inertElement.inert : undefined,
        hasInertProperty,
      },
    });
  }
  element.setAttribute("aria-hidden", "true");
  element.setAttribute("inert", "");
  if ("inert" in element) (element as HTMLElement & { inert?: boolean }).inert = true;
}

function release(element: HTMLElement) {
  const held = hiddenBy.get(element);
  if (!held) return;
  held.count -= 1;
  if (held.count > 0) return;
  hiddenBy.delete(element);
  const { snapshot } = held;
  if (snapshot.ariaHidden === null) element.removeAttribute("aria-hidden");
  else element.setAttribute("aria-hidden", snapshot.ariaHidden);

  if (snapshot.inertAttribute === null) element.removeAttribute("inert");
  else element.setAttribute("inert", snapshot.inertAttribute);

  if (snapshot.hasInertProperty) {
    (element as HTMLElement & { inert?: boolean }).inert = Boolean(snapshot.inertProperty);
  }
}

/* The page's scroll lock, counted the same way: held while any dialog is open. */
let scrollLocks = 0;
let overflowBeforeLock = "";

function lockScroll() {
  if (scrollLocks === 0) overflowBeforeLock = document.body.style.overflow;
  scrollLocks += 1;
  document.body.style.overflow = "hidden";
  let released = false;
  return () => {
    if (released) return;
    released = true;
    scrollLocks -= 1;
    if (scrollLocks === 0) document.body.style.overflow = overflowBeforeLock;
  };
}

function isolateBackground(root: HTMLElement) {
  const targets = isolationTargets(root);
  targets.forEach(hide);
  return () => targets.forEach(release);
}

function isTopmostDialog(root: HTMLElement) {
  const dialogs = Array.from(
    document.querySelectorAll<HTMLElement>("[aria-modal='true']"),
  );
  return dialogs.at(-1) === root;
}

export function useDialogFocus({
  active = true,
  containerRef,
  initialFocusRef,
  onEscape,
}: {
  active?: boolean;
  containerRef: RefObject<HTMLElement | null>;
  initialFocusRef?: RefObject<HTMLElement | null>;
  onEscape: () => void;
}) {
  const onEscapeRef = useRef(onEscape);
  onEscapeRef.current = onEscape;

  useEffect(() => {
    if (!active) return;
    const root = containerRef.current;
    if (!root) return;
    const topmostAtMount = isTopmostDialog(root);

    const previous = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const focusable = () => focusableElements(root);
    const initial = initialFocusRef?.current;
    const initialFocus = initial
      && root.contains(initial)
      && !initial.hidden
      && initial.getAttribute("disabled") === null
      ? initial
      : focusable()[0] ?? root;

    if (topmostAtMount) initialFocus.focus();

    const keydown = (event: KeyboardEvent) => {
      if (!isTopmostDialog(root)) return;
      if (event.key === "Escape") {
        const nestedModal = Array.from(
          root.querySelectorAll<HTMLElement>("[role='dialog'][aria-modal='true']"),
        ).find((modal) => modal.contains(document.activeElement));
        if (nestedModal || event.defaultPrevented) return;
        event.preventDefault();
        onEscapeRef.current();
        return;
      }
      if (event.key !== "Tab") return;

      const items = focusable();
      if (!items.length) {
        event.preventDefault();
        root.focus();
        return;
      }

      const first = items[0];
      const last = items[items.length - 1];
      const activeElement = document.activeElement;
      if (!root.contains(activeElement) || activeElement === root) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    const restoreBackground = topmostAtMount ? isolateBackground(root) : () => undefined;
    const unlockScroll = lockScroll();
    document.addEventListener("keydown", keydown);
    return () => {
      document.removeEventListener("keydown", keydown);
      restoreBackground();
      unlockScroll();
      if (previous?.isConnected) previous.focus();
    };
  }, [active, containerRef, initialFocusRef]);
}
