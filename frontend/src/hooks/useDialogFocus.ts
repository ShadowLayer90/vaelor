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

function isolateBackground(root: HTMLElement) {
  const snapshots: InertSnapshot[] = isolationTargets(root).map((element) => {
    const inertElement = element as HTMLElement & { inert?: boolean };
    const hasInertProperty = "inert" in inertElement;
    const snapshot: InertSnapshot = {
      element,
      ariaHidden: element.getAttribute("aria-hidden"),
      inertAttribute: element.getAttribute("inert"),
      inertProperty: hasInertProperty ? inertElement.inert : undefined,
      hasInertProperty,
    };
    element.setAttribute("aria-hidden", "true");
    element.setAttribute("inert", "");
    if (hasInertProperty) inertElement.inert = true;
    return snapshot;
  });

  return () => {
    for (const snapshot of snapshots) {
      if (snapshot.ariaHidden === null) snapshot.element.removeAttribute("aria-hidden");
      else snapshot.element.setAttribute("aria-hidden", snapshot.ariaHidden);

      if (snapshot.inertAttribute === null) snapshot.element.removeAttribute("inert");
      else snapshot.element.setAttribute("inert", snapshot.inertAttribute);

      if (snapshot.hasInertProperty) {
        (snapshot.element as HTMLElement & { inert?: boolean }).inert = Boolean(snapshot.inertProperty);
      }
    }
  };
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

    const overflow = document.body.style.overflow;
    const restoreBackground = topmostAtMount ? isolateBackground(root) : () => undefined;
    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", keydown);
    return () => {
      document.removeEventListener("keydown", keydown);
      restoreBackground();
      document.body.style.overflow = overflow;
      if (previous?.isConnected) previous.focus();
    };
  }, [active, containerRef, initialFocusRef]);
}
