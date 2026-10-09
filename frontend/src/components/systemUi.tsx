import { useEffect, useId, useRef, type ReactNode, type RefObject } from "react";
import { ModalShell } from "./ModalShell";
import { Button } from "./ui";
import { ModalError } from "./ui/ModalError";
import { joinClassNames } from "./ui/field";

/*
 * The System and Remote console streams' own building blocks (VD-200, the
 * System, Cooling, Console and DialogsSystem boards). They compose the shared
 * card classes (page-primitives.css) and add only what those boards draw and
 * the shared primitives do not: a card whose title carries a small uppercase
 * eyebrow, the key/value cell grid, and a dialog with the boards' header,
 * body and footer. Their look lives in styles/system.css.
 */

/**
 * Repeats `tick` every `everyMs` while the page is visible and `enabled`, and
 * once more as the page comes back into view. The first read stays with the
 * caller. A hidden tab asks the appliance nothing: the System polls (fans
 * every 3 s, lighting every 5 s, inventory every 15 s) once ran in a
 * background tab all day.
 */
export function useVisiblePoll(tick: () => void, everyMs: number, enabled = true) {
  const latest = useRef(tick);
  useEffect(() => { latest.current = tick; });
  useEffect(() => {
    if (!enabled) return undefined;
    const run = () => { if (!document.hidden) latest.current(); };
    const interval = window.setInterval(run, everyMs);
    document.addEventListener("visibilitychange", run);
    return () => {
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", run);
    };
  }, [enabled, everyMs]);
}

/**
 * A card whose title may carry an eyebrow ("THIS MACHINE · UBUNTU 26.04",
 * "KEEP VAELOR CURRENT"). The eyebrow sits outside the heading, so the
 * heading's accessible name stays the title alone.
 */
export function SectionCard({
  actions,
  as: Element = "section",
  children,
  className,
  description,
  eyebrow,
  flush = false,
  footer,
  id,
  title,
  titleId,
}: {
  actions?: ReactNode;
  as?: "section" | "article";
  children?: ReactNode;
  className?: string;
  description?: ReactNode;
  eyebrow?: ReactNode;
  flush?: boolean;
  footer?: ReactNode;
  id?: string;
  title?: ReactNode;
  titleId?: string;
}) {
  const generated = "sys-card-" + useId().replaceAll(":", "");
  const headingId = titleId ?? generated;
  return (
    <Element
      aria-labelledby={title ? headingId : undefined}
      className={joinClassNames("card", "ui-card", "sys-card", className)}
      id={id}
      tabIndex={id ? -1 : undefined}
    >
      {(title || description || actions || eyebrow) && (
        <header className="ui-card__header">
          <div className="ui-card__titles">
            {eyebrow && <span className="sys-eyebrow">{eyebrow}</span>}
            {title && <h2 id={headingId}>{title}</h2>}
            {description && <p>{description}</p>}
          </div>
          {actions && <div className="ui-card__actions">{actions}</div>}
        </header>
      )}
      {children !== undefined && children !== null && children !== false && (
        <div className={joinClassNames("ui-card__body", flush && "ui-card__body--flush")}>{children}</div>
      )}
      {footer && <footer className="ui-card__footer">{footer}</footer>}
    </Element>
  );
}

export interface KvItem {
  key?: string;
  label: ReactNode;
  value: ReactNode;
  detail?: ReactNode;
  /** The value in the data face (a version, a size); words stay in the UI face. */
  mono?: boolean;
}

/** The boards' key/value grid: hairline cells, a quiet label over the value. */
export function KvGrid({ items, label, className }: { items: KvItem[]; label?: string; className?: string }) {
  return (
    <dl aria-label={label} className={joinClassNames("sys-kv", className)}>
      {items.map((item, index) => (
        <div className="sys-kv__cell" key={item.key ?? (typeof item.label === "string" ? item.label : index)}>
          <dt>{item.label}</dt>
          <dd className={item.mono ? "sys-kv__value sys-kv__value--mono" : "sys-kv__value"}>{item.value}</dd>
          {item.detail && <dd className="sys-kv__detail">{item.detail}</dd>}
        </div>
      ))}
    </dl>
  );
}

/**
 * The DialogsSystem board's dialog: an eyebrow and a title over a hairline,
 * the body, and a footer of actions. A failure is said inside it (the shell's
 * `error`), never on the page under it.
 */
export function SystemDialog({
  children,
  closeLabel = "Close",
  error,
  eyebrow,
  eyebrowTone = "quiet",
  footer,
  initialFocusRef,
  onClose,
  role,
  size = "standard",
  className,
  description,
  showClose = true,
  title,
  titleId,
}: {
  children: ReactNode;
  closeLabel?: string;
  error?: ReactNode;
  eyebrow?: ReactNode;
  /** `warning` for a dialog whose action restarts something ("Vaelor restarts"). */
  eyebrowTone?: "quiet" | "warning";
  footer?: ReactNode;
  initialFocusRef?: RefObject<HTMLElement | null>;
  onClose: () => void;
  /** `alertdialog` for a confirmation (the update and reinstall dialogs). */
  role?: "dialog" | "alertdialog";
  size?: "standard" | "wide";
  className?: string;
  description?: ReactNode;
  showClose?: boolean;
  title: ReactNode;
  titleId: string;
}) {
  return (
    <ModalShell
      className={joinClassNames("sys-dialog", className)}
      initialFocusRef={initialFocusRef}
      labelledBy={titleId}
      onClose={onClose}
      role={role}
      size={size}
    >
      <header className="sys-dialog__header">
        <div className="sys-dialog__titles">
          {eyebrow && <span className={joinClassNames("sys-eyebrow", eyebrowTone === "warning" && "sys-eyebrow--warning")}>{eyebrow}</span>}
          <h2 id={titleId}>{title}</h2>
          {description && <p>{description}</p>}
        </div>
        {showClose && <Button onClick={onClose} type="button" variant="quiet">{closeLabel}</Button>}
      </header>
      <div className="sys-dialog__body">
        {children}
        {/* The failure sits above the actions that would retry it, as the board draws it. */}
        <ModalError error={error} />
      </div>
      {footer && <footer className="sys-dialog__footer">{footer}</footer>}
    </ModalShell>
  );
}
