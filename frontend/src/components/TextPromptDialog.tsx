import { useEffect, useRef } from "react";
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
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    inputRef.current?.focus();
    inputRef.current?.select();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onCancel();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [busy, onCancel, open]);

  if (!open) return null;
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={onCancel}>
      <form
        aria-labelledby="text-prompt-title"
        aria-modal="true"
        className="dialog"
        onMouseDown={(event) => event.stopPropagation()}
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
        }}
        role="dialog"
      >
        <h2 id="text-prompt-title">{title}</h2>
        <p>{description}</p>
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
