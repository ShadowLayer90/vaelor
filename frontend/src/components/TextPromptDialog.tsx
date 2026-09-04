import { useEffect, useRef } from "react";
import { useDialogFocus } from "../hooks/useDialogFocus";
import { Button, Input } from "./ui";

export function TextPromptDialog({
  open,
  title,
  description,
  label,
  value,
  busy,
  onChange,
  onCancel,
  onSubmit,
}: {
  open: boolean;
  title: string;
  description: string;
  label: string;
  value: string;
  busy: boolean;
  onChange: (value: string) => void;
  onCancel: () => void;
  onSubmit: () => void;
}) {
  const dialogRef = useRef<HTMLFormElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus, Escape, and the Tab trap are centralised in useDialogFocus (same
  // hook ConfirmDialog and ModalShell use). This effect only preserves the
  // select-all convenience, which the hook does not do.
  useDialogFocus({
    active: open,
    containerRef: dialogRef,
    initialFocusRef: inputRef,
    onEscape: () => {
      if (!busy) onCancel();
    },
  });

  useEffect(() => {
    if (!open) return;
    inputRef.current?.select();
  }, [open]);

  if (!open) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onCancel}>
      <form
        aria-describedby="text-prompt-description"
        aria-labelledby="text-prompt-title"
        aria-modal="true"
        className="dialog"
        onMouseDown={(event) => event.stopPropagation()}
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
        }}
        ref={dialogRef}
        role="dialog"
        tabIndex={-1}
      >
        <h2 id="text-prompt-title">{title}</h2>
        <p id="text-prompt-description">{description}</p>
        <Input
          disabled={busy}
          label={label}
          maxLength={120}
          onChange={(event) => onChange(event.target.value)}
          ref={inputRef}
          required
          value={value}
        />
        <div className="dialog__actions">
          <Button disabled={busy} onClick={onCancel} variant="quiet">
            Cancel
          </Button>
          <Button disabled={busy || !value.trim()} type="submit" variant="primary">
            {busy ? "Saving…" : "Save name"}
          </Button>
        </div>
      </form>
    </div>
  );
}
