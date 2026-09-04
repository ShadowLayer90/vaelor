import { type ReactNode, type RefObject, useRef } from "react";
import { useDialogFocus } from "../hooks/useDialogFocus";

export function ModalShell({
  children,
  className,
  backdropClassName,
  describedBy,
  initialFocusRef,
  labelledBy,
  onClose,
  size = "standard",
}: {
  children: ReactNode;
  className?: string;
  backdropClassName?: string;
  describedBy?: string;
  initialFocusRef?: RefObject<HTMLElement | null>;
  labelledBy: string;
  onClose: () => void;
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
        role="dialog"
        tabIndex={-1}
      >
        {children}
      </div>
    </div>
  );
}
