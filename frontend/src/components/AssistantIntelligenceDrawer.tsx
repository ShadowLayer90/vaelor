import { useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useDialogFocus } from "../hooks/useDialogFocus";
import { Button } from "./ui";

/**
 * Change intelligence, as a drawer over Ask (VD-200 decision 6, the
 * AssistIntelligence board). It used to open as a section pushed into the page
 * above the conversation, so changing the model moved the question box away
 * from the reader; a drawer leaves the conversation where it was.
 *
 * The drawer is a modal dialog: focus moves in and is held, Escape and the
 * dimmed page close it, and the page beneath is inert while it is open (the
 * same `useDialogFocus` every dialog here uses). `active` is the Active
 * intelligence card; `setup` is CopilotSetup with `showHeader={false}`: the
 * drawer's own head names it and closes it.
 */
export function AssistantIntelligenceDrawer({
  active,
  onClose,
  setup,
}: {
  active: ReactNode;
  onClose: () => void;
  setup: ReactNode;
}) {
  const panel = useRef<HTMLElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  useDialogFocus({ containerRef: panel, initialFocusRef: closeButton, onEscape: onClose });
  // On the document body, like every other modal layer, so the page's own
  // styles (its 36 px buttons) do not reach the setup embedded in it.
  return createPortal(
    <div
      className="as-drawer-backdrop"
      onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}
      role="presentation"
    >
      <aside
        aria-labelledby="assistant-intelligence-title"
        aria-modal="true"
        className="as-drawer as-intel"
        ref={panel}
        role="dialog"
        tabIndex={-1}
      >
        <div className="as-dialog__head">
          <div>
            <span className="as-label">Assistant setup</span>
            <h2 id="assistant-intelligence-title">Choose how Vaelor Assistant thinks</h2>
            <p>We checked this device and narrowed the choices down for you.</p>
          </div>
          <Button className="as-btn-ghost" onClick={onClose} ref={closeButton} type="button">Close</Button>
        </div>
        <div className="as-drawer__body">
          <div className="as-drawer__column">{active}</div>
          <div className="as-drawer__column as-intel__setup">{setup}</div>
        </div>
      </aside>
    </div>,
    document.body,
  );
}
