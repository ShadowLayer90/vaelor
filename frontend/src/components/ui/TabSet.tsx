import { Fragment, useCallback, useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { joinClassNames } from "./field";

export type TabItem = {
  id: string;
  label: ReactNode;
  disabled?: boolean;
  disabledReason?: string;
};

export type TabSetProps = {
  label: string;
  items: readonly TabItem[];
  selectedId: string;
  /**
   * `keyboard` is true for an arrow, Home or End move: a page that keeps its
   * tab in the address replaces the entry for those and pushes one for a click,
   * so walking the strip by key does not fill Back with every tab passed.
   */
  onSelect: (id: string, how?: { keyboard: boolean }) => void;
  children: ReactNode;
  className?: string;
  /**
   * Appearance hooks for the strip and the panel. A surface adopting the
   * primitive keeps whatever look it already had — including its responsive
   * rules — instead of that CSS having to be rewritten against the primitive's
   * own class names.
   */
  listClassName?: string;
  panelClassName?: string;
};

/**
 * How far past a tab's edge the strip scrolls to show it: the width of the
 * fade that marks hidden tabs (`--tab-strip-cue` in shared-primitives.css), so
 * the tab brought into sight is not left under the cue.
 */
const CUE_WIDTH_PX = 40;

/** Which ends of a scrolling strip hide tabs. */
type HiddenEnds = { start: boolean; end: boolean };

export function TabSet({
  label,
  items,
  selectedId,
  onSelect,
  children,
  className,
  listClassName,
  panelClassName,
}: TabSetProps) {
  const instanceId = useId().replaceAll(":", "");
  const enabledItems = items.filter((item) => !item.disabled && !item.disabledReason);
  const activeItem = enabledItems.find((item) => item.id === selectedId) ?? enabledItems[0];
  const panelId = "ui-tab-panel-" + instanceId;
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const listRef = useRef<HTMLDivElement | null>(null);
  // B2: a strip narrower than its tabs scrolls. It says which end hides tabs
  // (`data-more-start` / `data-more-end`, drawn as a fade), and the selected
  // tab is scrolled into it - at 375 px the last tab once opened out of sight.
  const [hidden, setHidden] = useState<HiddenEnds>({ start: false, end: false });
  const measureHidden = useCallback(() => {
    const list = listRef.current;
    if (!list) return;
    const start = list.scrollLeft > 1;
    const end = list.scrollLeft + list.clientWidth < list.scrollWidth - 1;
    setHidden((current) => (current.start === start && current.end === end ? current : { start, end }));
  }, []);
  const activeId = activeItem?.id;

  useLayoutEffect(() => {
    const list = listRef.current;
    const tab = activeId ? tabRefs.current[activeId] : null;
    if (list && tab) {
      // Only the strip scrolls, never the page (scrollIntoView would move it).
      const strip = list.getBoundingClientRect();
      const box = tab.getBoundingClientRect();
      if (box.left < strip.left) list.scrollLeft -= strip.left - box.left + CUE_WIDTH_PX;
      else if (box.right > strip.right) list.scrollLeft += box.right - strip.right + CUE_WIDTH_PX;
    }
    measureHidden();
  }, [activeId, measureHidden]);

  useEffect(() => {
    const list = listRef.current;
    if (!list || typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", measureHidden);
      return () => window.removeEventListener("resize", measureHidden);
    }
    const observer = new ResizeObserver(measureHidden);
    observer.observe(list);
    return () => observer.disconnect();
  }, [measureHidden]);

  const moveFocus = (itemId: string, direction: "next" | "previous" | "first" | "last") => {
    if (!enabledItems.length) return;
    const currentIndex = Math.max(0, enabledItems.findIndex((item) => item.id === itemId));
    const nextIndex = direction === "first"
      ? 0
      : direction === "last"
        ? enabledItems.length - 1
        : (currentIndex + (direction === "next" ? 1 : -1) + enabledItems.length) % enabledItems.length;
    const nextItemId = enabledItems[nextIndex].id;
    onSelect(nextItemId, { keyboard: true });
    tabRefs.current[nextItemId]?.focus();
    queueMicrotask(() => tabRefs.current[nextItemId]?.focus());
  };

  const onTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>, itemId: string) => {
    // A modified key is not a tab move: Alt+Left is the browser's Back, and
    // Ctrl or Cmd with an arrow or Home/End belongs to the browser too.
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      event.preventDefault();
      moveFocus(itemId, "next");
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      event.preventDefault();
      moveFocus(itemId, "previous");
    } else if (event.key === "Home") {
      event.preventDefault();
      moveFocus(itemId, "first");
    } else if (event.key === "End") {
      event.preventDefault();
      moveFocus(itemId, "last");
    }
  };

  return (
    <div className={joinClassNames("tab-set", "ui-tab-set", className)}>
      <div
        aria-label={label}
        className={joinClassNames("ui-tab-set__list", listClassName)}
        data-more-end={hidden.end || undefined}
        data-more-start={hidden.start || undefined}
        onScroll={measureHidden}
        ref={listRef}
        role="tablist"
      >
        {items.map((item) => {
          const isDisabled = Boolean(item.disabled || item.disabledReason);
          const isSelected = item.id === activeItem?.id;
          const tabId = "ui-tab-" + instanceId + "-" + item.id;
          const disabledReasonId = item.disabledReason ? tabId + "-disabled-reason" : undefined;
          return (
            <Fragment key={item.id}>
              <button
                aria-controls={panelId}
                aria-describedby={disabledReasonId}
                aria-disabled={isDisabled || undefined}
                aria-selected={isSelected}
                className={joinClassNames("tab-set__tab", "ui-tab", isSelected && "ui-tab--selected")}
                disabled={isDisabled}
                id={tabId}
                ref={(button) => {
                  if (button) tabRefs.current[item.id] = button;
                  else delete tabRefs.current[item.id];
                }}
                onClick={() => onSelect(item.id, { keyboard: false })}
                onKeyDown={(event) => onTabKeyDown(event, item.id)}
                tabIndex={isSelected ? 0 : -1}
                type="button"
                role="tab"
              >
                {item.label}
              </button>
              {item.disabledReason && (
                <span className="ui-tab__disabled-reason sr-only" id={disabledReasonId}>
                  {item.disabledReason}
                </span>
              )}
            </Fragment>
          );
        })}
      </div>
      <div
        aria-labelledby={activeItem ? "ui-tab-" + instanceId + "-" + activeItem.id : undefined}
        className={joinClassNames("ui-tab-set__panel", panelClassName)}
        id={panelId}
        role="tabpanel"
        tabIndex={0}
      >
        {children}
      </div>
    </div>
  );
}
