import { useId, useRef } from "react";
import { useDialogFocus } from "../hooks/useDialogFocus";
import { Button, Notice } from "./ui";
import { ModalError } from "./ui/ModalError";

/** The red eyebrow the boards put over a confirmation whose action cannot be reversed. */
export const IRREVERSIBLE_EYEBROW = "Can't be undone";

/**
 * The console's confirmation, drawn as the boards draw one (VD-200: the
 * ChatDialogs, AssistAskDialogs and AssistRunDialogs boards): a header with
 * the question, a body saying what it does, and a footer with Cancel beside
 * the action. It is an `alertdialog`; Cancel takes the focus first, and
 * Escape and the backdrop cancel unless the request is under way.
 *
 * `irreversible` adds the red "Can't be undone" eyebrow. It is set only by a
 * caller whose action really cannot be reversed (a permanent delete); a
 * confirmation of something the owner can change back carries no eyebrow.
 */
export function ConfirmDialog({
  open,
  title,
  description,
  note = "",
  error = "",
  confirmLabel,
  busy,
  irreversible = false,
  onCancel,
  onConfirm,
}: {
  open: boolean;
  title: string;
  description: string;
  /** A second fact the owner should weigh, shown as its own notice (B8: the backend's sentence). */
  note?: string;
  /** Why the confirmed action was refused (W4d-D18): shown here, never on the inert page beneath. */
  error?: string;
  confirmLabel: string;
  busy: boolean;
  /** The action cannot be reversed: the header says so in red. */
  irreversible?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const dialogRef = useRef<HTMLElement>(null);
  const ids = useId().replaceAll(":", "");
  const titleId = `confirm-title-${ids}`;
  const descriptionId = `confirm-description-${ids}`;
  const noteId = `confirm-note-${ids}`;
  const cancelRef = useRef<HTMLButtonElement>(null);
  useDialogFocus({
    active: open,
    containerRef: dialogRef,
    initialFocusRef: cancelRef,
    onEscape: () => {
      if (!busy) onCancel();
    },
  });

  if (!open) return null;
  return (
    <div
      className="dialog-backdrop"
      role="presentation"
      onMouseDown={() => {
        // The backdrop cancels like Escape: not while the request is under way.
        if (!busy) onCancel();
      }}
    >
      <section
        aria-describedby={note ? `${descriptionId} ${noteId}` : descriptionId}
        aria-labelledby={titleId}
        aria-modal="true"
        className="dialog dialog--confirm dialog-sectioned"
        ref={dialogRef}
        role="alertdialog"
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="dialog-sectioned__header">
          {irreversible && (
            <span className="dialog-sectioned__eyebrow dialog-sectioned__eyebrow--danger">{IRREVERSIBLE_EYEBROW}</span>
          )}
          <h2 id={titleId}>{title}</h2>
        </header>
        {/* Review round 1: the body scrolls inside a dialog held within the
            viewport, so the actions below stay on a short phone's screen. */}
        <div className="dialog__body dialog-sectioned__body">
          <p id={descriptionId}>{description}</p>
          {note ? <div id={noteId}><Notice severity="info">{note}</Notice></div> : null}
          <ModalError error={error} />
        </div>
        <footer className="dialog__actions dialog-sectioned__footer">
          <Button
            disabled={busy}
            onClick={onCancel}
            ref={cancelRef}
            type="button"
          >
            Cancel
          </Button>
          <Button variant="danger"
            disabled={busy}
            onClick={onConfirm}
          >
            {busy ? "Sending…" : confirmLabel}
          </Button>
        </footer>
      </section>
    </div>
  );
}
