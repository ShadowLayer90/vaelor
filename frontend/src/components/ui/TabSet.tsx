import { Fragment, useId, useRef, type KeyboardEvent, type ReactNode } from "react";
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
  onSelect: (id: string) => void;
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

  const moveFocus = (itemId: string, direction: "next" | "previous" | "first" | "last") => {
    if (!enabledItems.length) return;
    const currentIndex = Math.max(0, enabledItems.findIndex((item) => item.id === itemId));
    const nextIndex = direction === "first"
      ? 0
      : direction === "last"
        ? enabledItems.length - 1
        : (currentIndex + (direction === "next" ? 1 : -1) + enabledItems.length) % enabledItems.length;
    const nextItemId = enabledItems[nextIndex].id;
    onSelect(nextItemId);
    tabRefs.current[nextItemId]?.focus();
    queueMicrotask(() => tabRefs.current[nextItemId]?.focus());
  };

  const onTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>, itemId: string) => {
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
      <div aria-label={label} className={joinClassNames("ui-tab-set__list", listClassName)} role="tablist">
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
                onClick={() => onSelect(item.id)}
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
