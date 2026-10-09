import { useEffect, useRef, type ReactNode, type RefObject } from "react";
import "../styles/apps.css";
import { useDialogFocus } from "../hooks/useDialogFocus";
import { Icon, ICON_SIZE, type IconName } from "./Icon";
import { Button, Notice } from "./ui";
import { joinClassNames } from "./ui/field";

/*
 * The Apps and AI area's building blocks (VD-200, the Apps* and Manage*
 * boards). Every dialog in the area is the boards' one dialog: an eyebrow over
 * the title with Close at the right, a body, and a footer whose buttons sit at
 * the right. A refusal of the dialog's own action is shown at the top of the
 * body, inside the dialog, because the page under it is inert.
 *
 * These are area pieces, not `ui/*` primitives: the shared primitives are
 * frozen for the redesign, and Claude consolidates after the merge.
 */

export function AppsDialog({
  busy = false,
  children,
  className,
  closeLabel = "Close",
  describedBy,
  error,
  eyebrow,
  eyebrowTone,
  footer,
  footerStart,
  headerActions,
  initialFocusRef,
  onClose,
  role = "dialog",
  showClose = true,
  size = "standard",
  title,
  titleId,
}: {
  /** While the dialog's action is submitting, Close and Escape are held. */
  busy?: boolean;
  children?: ReactNode;
  className?: string;
  closeLabel?: string;
  /** The id of the sentence that says what this dialog does (its aria-describedby). */
  describedBy?: string;
  /** The refusal of this dialog's own action, shown at the top of the body. */
  error?: ReactNode;
  eyebrow?: ReactNode;
  /** `danger` for a dialog that deletes something (the removal and delete boards). */
  eyebrowTone?: "danger";
  /** The footer's buttons, at the right. */
  footer?: ReactNode;
  /** What sits at the footer's left: a shield line, a ghost Cancel. */
  footerStart?: ReactNode;
  /** What sits in the header before Close: a state pill, at most one quiet control. */
  headerActions?: ReactNode;
  initialFocusRef?: RefObject<HTMLElement | null>;
  onClose: () => void;
  /** `alertdialog` for a confirmation (a delete, a dependency-aware removal). */
  role?: "dialog" | "alertdialog";
  /** False for a dialog the board draws without a header Close (its footer Cancel closes it). */
  showClose?: boolean;
  size?: "narrow" | "standard" | "wide";
  title: ReactNode;
  titleId: string;
}) {
  const dialog = useRef<HTMLDivElement>(null);
  const close = () => { if (!busy) onClose(); };
  useDialogFocus({ containerRef: dialog, initialFocusRef, onEscape: close });
  return (
    <div
      className="modal-shell apps-dialog-backdrop"
      data-modal-shell="true"
      onMouseDown={(event) => { if (event.target === event.currentTarget) close(); }}
      role="presentation"
    >
      <div
        aria-describedby={describedBy}
        aria-labelledby={titleId}
        aria-modal="true"
        className={joinClassNames("modal-shell__dialog", "apps-dialog", "apps-dialog--" + size, className)}
        ref={dialog}
        role={role}
        tabIndex={-1}
      >
        <header className="apps-dialog__header">
          <div className="apps-dialog__titles">
            {eyebrow && <span className={joinClassNames("apps-eyebrow", eyebrowTone === "danger" && "apps-eyebrow--danger")}>{eyebrow}</span>}
            <h2 id={titleId}>{title}</h2>
          </div>
          <div className="apps-dialog__header-actions">
            {headerActions}
            {showClose && <Button className="apps-dialog__close" disabled={busy} onClick={close} variant="quiet">{closeLabel}</Button>}
          </div>
        </header>
        <div className="apps-dialog__body">
          <DialogRefusal error={error} />
          {children}
        </div>
        {(footer || footerStart) && (
          <footer className="apps-dialog__footer">
            {footerStart && <div className="apps-dialog__footer-start">{footerStart}</div>}
            {footer && <div className="apps-dialog__footer-actions">{footer}</div>}
          </footer>
        )}
      </div>
    </div>
  );
}

/** A dialog's refusal: red, announced, scrolled into view when it appears. */
function DialogRefusal({ error }: { error?: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const shown = Boolean(error);
  useEffect(() => {
    if (shown) ref.current?.scrollIntoView?.({ block: "nearest" });
  }, [shown, error]);
  if (!shown) return null;
  return <div className="apps-dialog__refusal" ref={ref}><Notice severity="danger">{error}</Notice></div>;
}

/** The boards' feature chips ("Known configuration", "Policy scan"): outline, never a status. */
export function AppsTags({ items, label }: { items: string[]; label?: string }) {
  return (
    <ul aria-label={label} className="apps-tags">
      {items.map((item) => <li className="apps-tag" key={item}>{item}</li>)}
    </ul>
  );
}

/** The 34 px icon tile; `accent` for the one tile the board draws orange. */
export function AppsIconTile({ accent = false, name }: { accent?: boolean; name: IconName }) {
  return (
    <span aria-hidden="true" className={joinClassNames("apps-ico", accent && "apps-ico--accent")}>
      <Icon name={name} size={ICON_SIZE.nav} />
    </span>
  );
}

/**
 * Label and value pairs as the boards draw them in a card ("Memory limit
 * 512 MB"): a grey label column and a white value column. `mono` for values
 * that are names the machine reads (a repository, a file, a digest).
 */
export function AppsFacts({
  className,
  rows,
}: {
  className?: string;
  rows: Array<{ label: ReactNode; value: ReactNode; mono?: boolean; key?: string }>;
}) {
  return (
    <dl className={joinClassNames("apps-facts", className)}>
      {rows.map((row, index) => (
        <div className="apps-facts__row" key={row.key ?? (typeof row.label === "string" ? row.label : index)}>
          <dt>{row.label}</dt>
          <dd className={row.mono ? "apps-mono" : undefined}>{row.value}</dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * The boards' key/value grid (`.v-kv`): hairline-separated cells, a grey label
 * over a mono value. Used for readings a reader compares (CPU, Memory, Port).
 */
export function AppsKv({ cells, label }: { cells: Array<{ label: ReactNode; value: ReactNode; key?: string }>; label?: string }) {
  return (
    <dl aria-label={label} className="apps-kv">
      {cells.map((cell, index) => (
        <div key={cell.key ?? (typeof cell.label === "string" ? cell.label : index)}>
          <dt>{cell.label}</dt>
          <dd>{cell.value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** A quiet bordered block inside a dialog or panel: a title line, a detail line, what follows. */
export function AppsInset({
  children,
  className,
  detail,
  icon,
  title,
  tone,
}: {
  children?: ReactNode;
  className?: string;
  detail?: ReactNode;
  icon?: IconName;
  title?: ReactNode;
  /** `accent` is the boards' recommended (orange-outlined) card; `danger` a failed block. */
  tone?: "accent" | "danger";
}) {
  return (
    <div className={joinClassNames("apps-inset", tone && "apps-inset--" + tone, className)}>
      {(icon || title || detail) && (
        <div className="apps-inset__head">
          {icon && <Icon aria-hidden="true" name={icon} size={ICON_SIZE.inline} />}
          <div className="apps-inset__text">
            {title && <strong>{title}</strong>}
            {detail && <span>{detail}</span>}
          </div>
        </div>
      )}
      {children}
    </div>
  );
}

/** A progress bar as the boards draw it: a 6 px track, an orange fill, a percent when one was read. */
export function AppsProgress({ fraction, label, stale = false }: { fraction: number | null; label: string; stale?: boolean }) {
  const percent = fraction === null ? null : Math.round(Math.max(0, Math.min(1, fraction)) * 100);
  return (
    <div
      aria-label={label}
      aria-valuemax={100}
      aria-valuemin={0}
      aria-valuenow={percent ?? undefined}
      className={joinClassNames("apps-progress", stale && "apps-progress--stale")}
      role="progressbar"
    >
      {percent !== null && <i style={{ width: `${percent}%` }} />}
    </div>
  );
}
