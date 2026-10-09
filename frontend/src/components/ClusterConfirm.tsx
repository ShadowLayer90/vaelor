import { useId, useRef, type ReactNode } from "react";
import { useDialogFocus } from "../hooks/useDialogFocus";
import { Button } from "./ui/Button";
import { ModalError } from "./ui/ModalError";
import type { ConfirmRequest } from "./ClusterPlanDialogs";
import type { ClusterPlanRequest } from "../lib/clusterJobs";
import { MODE_B_REMOVAL_NOTE, isModeBDeployment } from "../lib/gpuServingMode";
import type { PooledDeployment } from "./ClusterDeployments";
import "../styles/cluster.css";

/**
 * The Cluster boards' confirmation (VD-200, the ClusterDialogsConfirm board):
 * an eyebrow ("Confirm removal", "API keys", "Can't be undone") over the
 * question, what it does, and Cancel beside the action. It is an
 * `alertdialog`: Cancel takes the focus first, Escape and the backdrop cancel
 * unless a request is under way, and the refusal of the action is shown here,
 * never on the inert page beneath.
 *
 * It is its own module, the cluster's counterpart of ConfirmDialog, so the
 * sweep inventory (tools/ui_inventory.py) can name each use by its `title`
 * and read the two buttons it draws from its props.
 */
export function ClusterConfirm({
  busy,
  busyLabel = "Removing…",
  confirmLabel,
  confirmVariant = "danger",
  description,
  error,
  eyebrow = "Confirm removal",
  eyebrowTone,
  onCancel,
  onConfirm,
  open,
  title,
}: {
  busy: boolean;
  busyLabel?: string;
  confirmLabel: string;
  /** A confirmation that is not destructive (Load, Use this link) carries the primary action. */
  confirmVariant?: "danger" | "primary";
  description: ReactNode;
  error?: ReactNode;
  eyebrow?: string;
  /**
   * The eyebrow's colour. By default it follows the action: red over a
   * destructive one, neutral over a constructive one (Load), so "Confirm load"
   * is never drawn as a warning (VD-200 review).
   */
  eyebrowTone?: "danger" | "warning" | "accent" | "neutral";
  onCancel: () => void;
  onConfirm: () => void;
  open: boolean;
  title: string;
}) {
  const tone = eyebrowTone ?? (confirmVariant === "primary" ? "neutral" : "danger");
  const ids = useId().replaceAll(":", "");
  const titleId = "cl-confirm-title-" + ids;
  const descriptionId = "cl-confirm-description-" + ids;
  const dialogRef = useRef<HTMLDivElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  useDialogFocus({
    active: open,
    containerRef: dialogRef,
    initialFocusRef: cancelRef,
    onEscape: () => { if (!busy) onCancel(); },
  });
  if (!open) return null;
  return (
    <div
      className="modal-shell"
      data-modal-shell="true"
      onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) onCancel(); }}
      role="presentation"
    >
      <div
        aria-describedby={descriptionId}
        aria-labelledby={titleId}
        aria-modal="true"
        className="modal-shell__dialog modal-shell__dialog--standard cl-dialog cl-dialog--standard"
        ref={dialogRef}
        role="alertdialog"
        tabIndex={-1}
      >
        <header className="cl-dialog__header">
          <div className="cl-dialog__titles">
            <span className={tone === "neutral" ? "cl-eyebrow" : "cl-eyebrow cl-eyebrow--" + tone}>{eyebrow}</span>
            <h2 id={titleId}>{title}</h2>
          </div>
        </header>
        <div className="cl-dialog__body">
          <div className="cl-confirm__text" id={descriptionId}>{description}</div>
          <ModalError error={error} />
        </div>
        <footer className="cl-dialog__footer">
          <Button disabled={busy} onClick={onCancel} ref={cancelRef}>Cancel</Button>
          <Button disabled={busy} onClick={onConfirm} variant={confirmVariant}>{busy ? busyLabel : confirmLabel}</Button>
        </footer>
      </div>
    </div>
  );
}

/** The quoted name a confirmation is about. */
const named = (payload: Record<string, unknown>) => String(payload.name ?? "");

/**
 * The confirmation a Deployments row action asks for when it has no
 * server-side plan builder (GPU remove, unload and load, agent remove), or
 * null when the action goes through the reviewed `/cluster/plan` step instead.
 * Moved out of FleetCenter, which is over the module size warning.
 */
export function serviceConfirmRequest(
  action: string,
  payload: Record<string, unknown>,
  pooled: PooledDeployment[],
): (ConfirmRequest & { request: ClusterPlanRequest }) | null {
  const request = { action, payload };
  // VD-127: removing or unloading a Mode B deployment changes what serves AI Chat.
  const clustered = pooled.some((entry) => entry.name === payload.name && isModeBDeployment(entry));
  if (action === "remove-gpu") {
    // VD-127: removing a Mode B deployment is also the switch back. The
    // confirmation has to say what comes back, because that is the half of
    // the change the row itself does not show.
    return {
      title: "Remove GPU deployment",
      body: `This stops the vLLM server "${named(payload)}" and revokes its gateway credential. The cached model weights stay on the workers.${clustered ? ` ${MODE_B_REMOVAL_NOTE}` : ""}`,
      request,
    };
  }
  if (action === "unload-gpu") {
    // G3a: Unload warns that the model stops serving and, in Mode B, that AI
    // Chat is unavailable until it is loaded again IF it is still using the
    // cluster - the owner may have chosen another connection (VD-210).
    return {
      title: "Unload GPU deployment",
      eyebrow: "Confirm unload",
      confirmLabel: "Unload",
      busyLabel: "Unloading…",
      body: `This stops the vLLM server "${named(payload)}" to free the GPU, but keeps the deployment, its cached model weights and its gateway credential so it can be loaded again warm.${clustered ? " If AI Chat is using it, AI Chat will be unavailable until it is loaded again." : " The model stops serving until it is loaded again."}`,
      request,
    };
  }
  if (action === "load-gpu") {
    // W4-D7: Load also starts a FAILED row again; say what that does.
    const failed = pooled.some((entry) => entry.name === payload.name && entry.state === "failed");
    return {
      title: "Load GPU deployment",
      eyebrow: "Confirm load",
      confirmLabel: "Load",
      busyLabel: "Loading…",
      constructive: true,
      body: `This re-serves the vLLM model "${named(payload)}" on the GPU from its cached weights and repoints AI Chat back at it when it backs AI Chat. The weights are cached, so it should be serving again shortly.${failed ? " It failed earlier: whatever is left of it is stopped first, and when it runs on this controller its own AI Chat model stops while it loads, as for a deploy." : ""}`,
      request,
    };
  }
  if (action === "remove-agent") {
    // F6c-1: removing an agent has no server-side plan builder either.
    return {
      title: "Remove agent",
      body: `This stops the deployed agent "${named(payload)}" and revokes its API keys. The backing model deployment keeps running.`,
      request,
    };
  }
  return null;
}
