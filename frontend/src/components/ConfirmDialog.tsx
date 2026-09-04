import { useRef } from "react";
import { useDialogFocus } from "../hooks/useDialogFocus";
import { Icon } from "./Icon";
import { Button } from "./ui";

export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel,
  busy,
  onCancel,
  onConfirm,
}: {
  open: boolean;
  title: string;
  description: string;
  confirmLabel: string;
  busy: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const dialogRef = useRef<HTMLElement>(null);
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
    <div className="dialog-backdrop" role="presentation" onMouseDown={onCancel}>
      <section
        aria-labelledby="confirm-title"
        aria-modal="true"
        className="dialog"
        ref={dialogRef}
        role="alertdialog"
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="dialog__icon">
          <Icon name="alert" size={22} />
        </div>
        <h2 id="confirm-title">{title}</h2>
        <p>{description}</p>
        <div className="dialog__actions">
          <Button variant="quiet"
            className="ui-control ui-control--button"
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
        </div>
      </section>
    </div>
  );
}
