import { useEffect, useRef, type ReactNode } from "react";
import { Notice } from "./Notice";

/**
 * Where a modal shows the refusal of its own action (W4d-D16, D18): inside
 * the dialog, as an alert, scrolled into view when it appears. The page under
 * an open dialog is inert and aria-hidden, so an error rendered there is
 * neither seen nor announced. Renders nothing while there is no error.
 */
export function ModalError({ error }: { error?: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const shown = Boolean(error);
  useEffect(() => {
    // A long form scrolls inside its dialog; the refusal must not appear
    // below the fold of the dialog the owner is looking at.
    if (shown) ref.current?.scrollIntoView?.({ block: "nearest" });
  }, [shown, error]);
  if (!shown) return null;
  return (
    <div className="modal-error" ref={ref}>
      <Notice severity="danger">{error}</Notice>
    </div>
  );
}
