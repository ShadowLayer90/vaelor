import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Session } from "../types";
import { apiRequest } from "../lib/api";
import { ConfirmDialog } from "./ConfirmDialog";
import { Icon, ICON_SIZE } from "./Icon";
import { Button } from "./ui";

export type PowerAction = "restart_service" | "reboot" | "shutdown";
export type PowerCapabilities = {
  actions: Record<PowerAction, { available: boolean; reason: string }>;
};

const actionCopy: Record<
  PowerAction,
  {
    title: string;
    description: string;
    label: string;
    confirmation: string;
    /** Shown on the control itself, before the confirmation dialog opens. */
    consequence: string;
  }
> = {
  restart_service: {
    // The control and its confirmation have to be the same action. A card
    // labelled "Restart service" that opens "Restart hardware service?" reads
    // as a second, broader thing happening.
    title: "Restart service?",
    description:
      "Live telemetry will disconnect briefly. The device will remain powered on.",
    label: "Restart service",
    confirmation: "restart-service",
    consequence: "Interrupts live telemetry for a few seconds",
  },
  reboot: {
    title: "Reboot this device?",
    description:
      "All active sessions and services will stop while the device restarts.",
    label: "Reboot device",
    confirmation: "reboot-device",
    consequence: "Stops every session and running app until it restarts",
  },
  shutdown: {
    title: "Shut down this device?",
    description:
      "Remote access will be lost and physical power may be required to start it again.",
    label: "Shut down",
    confirmation: "shutdown-device",
    consequence: "Ends remote access; restarting may need physical power",
  },
};

/**
 * The System page's Power button (VD-200, the SystemCompute board): a menu of
 * the three power actions, each with its consequence in words. Choosing one
 * opens the same confirmation as before, with the same words. The
 * confirmation lives and dies with this menu, so it can never follow the
 * owner to another page (W4d-D30): leaving System unmounts it.
 */
export function PowerMenu({
  capabilities,
  onNotice,
  session,
}: {
  capabilities: PowerCapabilities | null;
  onNotice: (message: string) => void;
  session: Session;
}) {
  const [open, setOpen] = useState(false);
  const [selectedAction, setSelectedAction] = useState<PowerAction | null>(null);
  const [actionBusy, setActionBusy] = useState(false);
  const menu = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  /** Set when the confirmation closes, so focus goes back to Power rather than to the page (the menu pattern). */
  const returnFocus = useRef(false);

  // After the dialog has unmounted and released the page, not before: its own
  // clean-up would otherwise move focus after this did.
  useEffect(() => {
    if (selectedAction !== null || !returnFocus.current) return;
    returnFocus.current = false;
    trigger.current?.focus();
  }, [selectedAction]);

  useEffect(() => {
    if (!open) return;
    menu.current?.querySelector<HTMLButtonElement>("[role=menuitem]:not(:disabled)")?.focus();
    const away = (event: MouseEvent) => {
      if (!menu.current?.contains(event.target as Node) && !trigger.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, [open]);

  const performAction = async () => {
    if (!selectedAction) return;
    const action = selectedAction;
    setActionBusy(true);
    try {
      await apiRequest(
        "/power/actions",
        {
          method: "POST",
          body: JSON.stringify({
            action,
            confirmation: actionCopy[action].confirmation,
          }),
        },
        session.csrf_token,
      );
      onNotice(`${actionCopy[action].label} request accepted.`);
      returnFocus.current = true;
      setSelectedAction(null);
      window.setTimeout(() => onNotice(""), 5000);
    } catch (caught) {
      // W4d-D30: a rejected request left the confirmation mounted; it
      // resurfaced over another page. It ends with its request, either way.
      returnFocus.current = true;
      setSelectedAction(null);
      onNotice(
        action === "restart_service"
          ? "The restart request did not get an answer. The control plane may already be restarting; reload in a moment before trying again."
          : caught instanceof Error && caught.message
            ? caught.message
            : `${actionCopy[action].label} could not be requested.`,
      );
    } finally {
      setActionBusy(false);
    }
  };

  const viewer = session.user.role === "viewer";
  return (
    <div className="power-menu">
      {/* One trigger: it opens a menu (view state only); a viewer's is disabled, with the reason beside it.
          The reason hangs under the trigger, out of the bar's row (`detached`): in the row it lifted Power
          about 14 px off the bar's centre line and pushed the bar 84 px past a 768 px screen (polish
          audit, cause 8). It stays visible text (the owner's rule) and the button's description. */}
      <Button
        aria-controls={viewer ? undefined : "power-menu-list"}
        aria-expanded={viewer ? undefined : open}
        aria-haspopup={viewer ? undefined : "menu"}
        className="power-menu__trigger"
        disabledReason={viewer ? "Operator access is required." : undefined}
        disabledReasonLayout="detached"
        onClick={() => setOpen((value) => !value)}
        ref={trigger}
        variant="secondary"
      >
        <Icon name="power" size={ICON_SIZE.inline} /><span className="power-menu__label">Power</span>
      </Button>
      {open && !viewer && (
        <div
          aria-label="Power"
          className="power-menu__list"
          id="power-menu-list"
          onKeyDown={(event) => {
            const items = [...(menu.current?.querySelectorAll<HTMLButtonElement>("[role=menuitem]:not(:disabled)") ?? [])];
            const index = items.indexOf(document.activeElement as HTMLButtonElement);
            if (event.key === "Escape") { event.preventDefault(); setOpen(false); trigger.current?.focus(); }
            // Tab leaves a menu: it closes and focus returns to Power, never into a menu that is gone.
            if (event.key === "Tab") { event.preventDefault(); setOpen(false); trigger.current?.focus(); }
            if (event.key === "ArrowDown") { event.preventDefault(); items[(index + 1) % items.length]?.focus(); }
            if (event.key === "ArrowUp") { event.preventDefault(); items[(index - 1 + items.length) % items.length]?.focus(); }
          }}
          ref={menu}
          role="menu"
        >
          {(Object.keys(actionCopy) as PowerAction[]).map((action) => {
            const available = Boolean(capabilities?.actions[action]?.available);
            // A destructive action must not look like a link: the danger word,
            // the consequence in plain words, and a confirmation step.
            return (
              <Button
                aria-haspopup="dialog"
                className="power-menu__item"
                disabled={!available}
                key={action}
                onClick={() => { setOpen(false); setSelectedAction(action); }}
                role="menuitem"
                title={capabilities?.actions[action]?.reason || undefined}
                variant="danger"
              >
                <strong>{actionCopy[action].label}</strong>
                <small>
                  {available
                    ? actionCopy[action].consequence
                    : capabilities?.actions[action]?.reason || "Checking platform support"}
                </small>
              </Button>
            );
          })}
        </div>
      )}
      {/* On the body, not in the header: the top bar's backdrop-filter traps a fixed
          backdrop inside its 64 px, and the confirmation's title rose off the screen. */}
      {selectedAction && createPortal(
        <ConfirmDialog
          busy={actionBusy}
          confirmLabel={actionCopy[selectedAction].label}
          description={actionCopy[selectedAction].description}
          onCancel={() => {
            if (actionBusy) return;
            returnFocus.current = true;
            setSelectedAction(null);
          }}
          onConfirm={() => void performAction()}
          open
          title={actionCopy[selectedAction].title}
        />,
        document.body,
      )}
    </div>
  );
}
