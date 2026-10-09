import { type ReactNode, type RefObject, useRef } from "react";
import { useDialogFocus } from "../hooks/useDialogFocus";
import { ModalError } from "./ui/ModalError";

export function ModalShell({
  children,
  className,
  backdropClassName,
  describedBy,
  error,
  initialFocusRef,
  labelledBy,
  onClose,
  role = "dialog",
  size = "standard",
}: {
  children: ReactNode;
  className?: string;
  backdropClassName?: string;
  describedBy?: string;
  /**
   * The refusal of this dialog's own action (W4d-D16, D18). It renders inside
   * the dialog; the page beneath is inert while the dialog is open.
   */
  error?: ReactNode;
  initialFocusRef?: RefObject<HTMLElement | null>;
  labelledBy: string;
  onClose: () => void;
  /**
   * `alertdialog` for a confirmation: something the owner must answer before
   * going on (a restart, a removal, a change that takes effect at once). A
   * form or a reading stays a plain `dialog`.
   */
  role?: "dialog" | "alertdialog";
  size?: "standard" | "wide";
}) {
  const dialog = useRef<HTMLDivElement>(null);
  useDialogFocus({ containerRef: dialog, initialFocusRef, onEscape: onClose });

  return (
    <div
      className={["modal-shell", backdropClassName].filter(Boolean).join(" ")}
      data-modal-shell="true"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
      role="presentation"
    >
      <div
        aria-describedby={describedBy}
        aria-labelledby={labelledBy}
        aria-modal="true"
        className={[
          "modal-shell__dialog",
          "modal-shell__dialog--" + size,
          className,
        ].filter(Boolean).join(" ")}
        ref={dialog}
        role={role}
        tabIndex={-1}
      >
        {children}
        <ModalError error={error} />
      </div>
    </div>
  );
}
