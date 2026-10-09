import { useId, useRef, type ReactNode, type RefObject } from "react";
import { useDialogFocus } from "../hooks/useDialogFocus";
import { Button } from "./ui";
import { ModalError } from "./ui/ModalError";
import "../styles/records.css";

/*
 * The pieces the Activity and Settings boards draw that the shared primitives
 * do not have yet (VD-200, the DialogsRecords board and its pages): a dialog
 * with an eyebrow over its title, a key/value grid, and the quiet pager under
 * a list. They live with the admin pages until the shared kit takes them in.
 */

/**
 * "2026-10-06 18:02" in this browser's time zone, from epoch seconds: the
 * boards' record time - sortable, and the same width on every row.
 */
export function shortStamp(seconds: number): string {
  const at = new Date(seconds * 1000);
  const two = (value: number) => String(value).padStart(2, "0");
  return `${at.getFullYear()}-${two(at.getMonth() + 1)}-${two(at.getDate())} ${two(at.getHours())}:${two(at.getMinutes())}`;
}

export type RecordDialogTone = "danger" | "warning" | "plain";

/**
 * A dialog drawn as the board draws one: an eyebrow (red for something that
 * cannot be undone), the title, a body, and the actions on the right. A
 * confirm (`alert`) focuses Cancel first, so Enter never removes anything.
 */
export function RecordDialog({
  actions,
  alert = true,
  children,
  error,
  eyebrow,
  eyebrowTone = "danger",
  initialFocusRef,
  onClose,
  title,
  wide = false,
  headerAction,
  footerStart,
  description,
}: {
  actions: ReactNode;
  /** An alertdialog (a confirm) rather than a plain dialog. */
  alert?: boolean;
  children: ReactNode;
  /** Why the dialog's own action was refused; shown inside the dialog. */
  error?: string;
  eyebrow: ReactNode;
  eyebrowTone?: RecordDialogTone;
  initialFocusRef?: RefObject<HTMLElement | null>;
  onClose: () => void;
  title: ReactNode;
  wide?: boolean;
  /** A control at the right of the header (a dialog's own Close). */
  headerAction?: ReactNode;
  /** What sits at the left of the actions row (a support reference). */
  footerStart?: ReactNode;
  /** A quiet line under the title. */
  description?: ReactNode;
}) {
  const ids = useId().replaceAll(":", "");
  const titleId = `record-dialog-${ids}-title`;
  const bodyId = `record-dialog-${ids}-body`;
  const container = useRef<HTMLElement>(null);
  useDialogFocus({ active: true, containerRef: container, initialFocusRef, onEscape: onClose });
  return (
    <div className="dialog-backdrop record-dialog-backdrop" onMouseDown={onClose} role="presentation">
      <section
        aria-describedby={bodyId}
        aria-labelledby={titleId}
        aria-modal="true"
        className={wide ? "record-dialog record-dialog--wide" : "record-dialog"}
        onMouseDown={(event) => event.stopPropagation()}
        ref={container}
        role={alert ? "alertdialog" : "dialog"}
        tabIndex={-1}
      >
        <header className="record-dialog__head">
          <div>
            <div className={`record-eyebrow record-eyebrow--${eyebrowTone}`}>{eyebrow}</div>
            <h2 id={titleId}>{title}</h2>
            {description && <p className="record-dialog__description">{description}</p>}
          </div>
          {headerAction}
        </header>
        <div className="record-dialog__body" id={bodyId}>
          {children}
          {error ? <ModalError error={error} /> : null}
        </div>
        <footer className={footerStart ? "record-dialog__actions record-dialog__actions--split" : "record-dialog__actions"}>
          {footerStart}
          <div className="record-dialog__buttons">{actions}</div>
        </footer>
      </section>
    </div>
  );
}

/**
 * The confirm the boards use for anything that removes or replaces: Cancel
 * (focused first) and a solid danger button. While `blockedReason` is set the
 * button stays disabled and says why beside it.
 */
export function RecordConfirm({
  blockedReason,
  busy,
  children,
  confirmLabel,
  error,
  eyebrow,
  eyebrowTone = "danger",
  onCancel,
  onConfirm,
  primary = false,
  title,
}: {
  blockedReason?: string;
  busy: boolean;
  children: ReactNode;
  confirmLabel: string;
  error?: string;
  eyebrow: ReactNode;
  eyebrowTone?: RecordDialogTone;
  onCancel: () => void;
  onConfirm: () => void;
  /** The orange primary instead of danger, for a confirm that only queues work. */
  primary?: boolean;
  title: ReactNode;
}) {
  const cancel = useRef<HTMLButtonElement>(null);
  return (
    <RecordDialog
      actions={<>
        <Button disabled={busy} onClick={onCancel} ref={cancel} type="button">Cancel</Button>
        <Button
          busy={busy}
          disabledReason={busy ? undefined : blockedReason}
          onClick={onConfirm}
          type="button"
          variant={primary ? "primary" : "danger"}
        >
          {confirmLabel}
        </Button>
      </>}
      error={error}
      eyebrow={eyebrow}
      eyebrowTone={eyebrowTone}
      initialFocusRef={cancel}
      onClose={() => { if (!busy) onCancel(); }}
      title={title}
    >
      {children}
    </RecordDialog>
  );
}

/** A key/value grid: hairline gaps, each cell a label over its value. */
export function RecordFacts({ facts, className }: {
  facts: Array<{ key: string; label: ReactNode; value: ReactNode; mono?: boolean; field?: string }>;
  className?: string;
}) {
  return (
    <dl className={className ? `record-facts ${className}` : "record-facts"}>
      {facts.map((fact) => (
        <div key={fact.key}>
          <dt>{fact.label}</dt>
          <dd className={fact.mono ? "record-mono" : undefined} data-field={fact.field}>{fact.value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** The quiet pager under a list: what is shown on the left, Previous · Page n of m · Next on the right. */
export function RecordPager({
  label,
  note,
  page,
  setPage,
  totalPages,
}: {
  label: string;
  note?: ReactNode;
  page: number;
  setPage: (next: number) => void;
  totalPages: number;
}) {
  return (
    <div className="record-pager">
      <span className="record-pager__note">{note}</span>
      <nav aria-label={`${label} pages`} className="record-pager__nav">
        <Button className="record-pager__step" disabled={page <= 1} onClick={() => setPage(Math.max(1, page - 1))} type="button" variant="quiet">Previous</Button>
        <span>Page {page} of {totalPages}</span>
        <Button className="record-pager__step" disabled={page >= totalPages} onClick={() => setPage(Math.min(totalPages, page + 1))} type="button" variant="quiet">Next</Button>
      </nav>
    </div>
  );
}
