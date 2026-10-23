import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { apiRequest } from "../lib/api";
import { jobLabel } from "../lib/jobPresentation";
import {
  RESTART_INTERRUPTS_CODE,
  acknowledgement,
  needsAcknowledgement,
  readRestartImpact,
  restartImpactPath,
  type RestartImpact,
} from "../lib/restartImpact";
import { ClusterDialog } from "./ClusterDialog";
import { Button, Checkbox, Notice } from "./ui";
import "../styles/restart-dialog.css";

/**
 * The console's restart (and shut down) confirmation, for this controller
 * and for a worker (VD-133 1.5.5 #5).
 *
 * In 1.5.4 the owner restarted the controller while a worker software update
 * ran; the update was cut off and then showed as failed, and the dialog had
 * said nothing about it. This dialog reads what is running when it opens
 * (`GET /power/restart-impact`) and lists it. When the restart would cut work
 * off, the owner ticks that they understand before the restart can be sent,
 * and the power route refuses a confirmation that did not see everything -
 * work that started after the read comes back here, listed, to confirm again.
 *
 * A worker's restart cuts no job off: Vaelor runs one job at a time and the
 * restart is itself one, so it waits behind what runs now. That is said too,
 * with how long it may wait.
 *
 * The read covers jobs only. An empty list says no job is running and never
 * that nothing is: the models and apps a machine serves stop with it whatever
 * the list holds, and each caller's `description` says so for its action.
 */
export function RestartDialog({
  machine,
  title,
  description,
  confirmLabel,
  eyebrow = "Can't be undone while it restarts",
  onCancel,
  onConfirm,
  onFailed,
}: {
  /** `controller`, or the worker's node id. */
  machine: string;
  title: string;
  description: ReactNode;
  confirmLabel: string;
  eyebrow?: string;
  onCancel: () => void;
  /** Sends the restart with what the owner saw. A "work started since" refusal keeps the dialog open with the new list. */
  onConfirm: (seen: ReturnType<typeof acknowledgement>) => Promise<void>;
  /**
   * Any other refusal (a 502 from the bridge, an unreachable appliance): the
   * dialog closes and the caller says why on its page, as the Power menu did
   * before (W4d-D30: the request ends the confirmation, either way).
   */
  onFailed: (message: string) => void;
}) {
  const [impact, setImpact] = useState<RestartImpact | null>(null);
  const [readFailed, setReadFailed] = useState("");
  const [understood, setUnderstood] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const cancelRef = useRef<HTMLButtonElement>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const read = useCallback(async () => {
    try {
      const answer = readRestartImpact(await apiRequest<unknown>(restartImpactPath(machine)));
      if (!mounted.current) return;
      setImpact(answer);
      setReadFailed(answer ? "" : "the answer was not in the form this dialog reads");
    } catch (caught) {
      if (mounted.current) setReadFailed(caught instanceof Error ? caught.message : "The request failed.");
    }
  }, [machine]);

  useEffect(() => { void read(); }, [read]);

  const checking = impact === null && !readFailed;
  const mustAcknowledge = needsAcknowledgement(impact);
  const blocked = checking || (mustAcknowledge && !understood);
  const cancel = () => { if (!busy) onCancel(); };

  const confirm = async () => {
    if (blocked) return;
    setBusy(true);
    setError("");
    try {
      await onConfirm(acknowledgement(impact));
    } catch (caught) {
      if (!mounted.current) return;
      // Read by its code, not by class: an ApiError carries `code` and the
      // envelope's `details` (lib/api.ts).
      const refusal = caught as { code?: unknown; details?: { impact?: unknown } } | null;
      if (refusal?.code === RESTART_INTERRUPTS_CODE) {
        // Something started after the list was read: show the new list and ask again.
        const fresh = readRestartImpact(refusal.details?.impact);
        setUnderstood(false);
        if (fresh) setImpact(fresh);
        else void read();
        setError(caught instanceof Error && caught.message ? caught.message : "The restart could not be requested.");
        return;
      }
      onFailed(caught instanceof Error && caught.message ? caught.message : "The restart could not be requested.");
    } finally {
      if (mounted.current) setBusy(false);
    }
  };

  const interrupts = impact?.effect === "interrupts";
  const count = (impact?.items.length ?? 0) + (impact?.others ?? 0);
  const waitMinutes = Math.round((impact?.wait_limit_seconds ?? 0) / 60);

  return (
    <ClusterDialog
      error={error || undefined}
      eyebrow={eyebrow}
      footer={(
        <>
          <Button disabled={busy} onClick={cancel} ref={cancelRef}>Cancel</Button>
          <Button
            disabled={busy || blocked}
            disabledReason={!busy && blocked
              ? checking ? "Checking what is running first." : "Tick that you understand what this cuts off."
              : undefined}
            onClick={() => void confirm()}
            variant="danger"
          >
            {busy ? "Sending…" : confirmLabel}
          </Button>
        </>
      )}
      initialFocusRef={cancelRef}
      onClose={cancel}
      role="alertdialog"
      title={title}
      tone="warning"
    >
      <p>{description}</p>
      {checking && <p className="cl-meta" role="status">Checking what is running…</p>}
      {readFailed && (
        <Notice severity="warning">
          Vaelor could not check what is running ({readFailed}). A restart may cut off work in
          progress; check Activity first.
        </Notice>
      )}
      {impact && !impact.readable && <Notice severity="warning">{impact.reason}</Notice>}
      {/* What was checked is jobs, so that is all this line may clear; what stops is the
          caller's `description` (found live 2026-10-21: "Nothing Vaelor is running would be cut
          off" while a 30B model served AI Chat on both machines; LESSONS 5 and 8). */}
      {impact && impact.readable && count === 0 && (
        <p className="cl-meta restart-impact__none">
          {interrupts ? "No Vaelor job is running now, so none is cut off." : "No other Vaelor job is running now."}
        </p>
      )}
      {impact && count > 0 && (
        <div className="restart-impact" role="group" aria-label={interrupts ? "Work this cuts off" : "Work this waits for"}>
          <p className={interrupts ? "cl-warn-text restart-impact__lead" : "restart-impact__lead"}>
            {interrupts
              ? `This cuts off ${count === 1 ? "1 job" : `${count} jobs`} running now. ${count === 1 ? "It does" : "They do"} not carry on when this controller is back: a job ends failed, or cancelled if it was already being cancelled, and has to be started again.`
              : `${count === 1 ? "1 job is" : `${count} jobs are`} running now. Vaelor runs one job at a time, so this restart starts when ${count === 1 ? "it finishes" : "they finish"}; if that takes more than ${waitMinutes} minutes, the restart is not carried out and you confirm it again.`}
          </p>
          <ul className="restart-impact__list">
            {impact.items.map((item) => (
              <li key={item.id}>
                <strong>{jobLabel(item.type)}</strong>
                <span>{item.message || "In progress"}</span>
              </li>
            ))}
            {impact.others > 0 && (
              <li>
                <strong>{impact.others === 1 ? "1 job" : `${impact.others} jobs`}</strong>
                <span>Started by other accounts.</span>
              </li>
            )}
          </ul>
          {interrupts && (
            <Checkbox
              checked={understood}
              disabled={busy}
              label={count === 1 ? "I understand this job is cut off." : "I understand these jobs are cut off."}
              onChange={(event) => setUnderstood(event.target.checked)}
            />
          )}
        </div>
      )}
      {impact?.reason && impact.readable && <p className="cl-meta">{impact.reason}</p>}
    </ClusterDialog>
  );
}
