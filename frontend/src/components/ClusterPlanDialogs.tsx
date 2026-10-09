import { Button, Input } from "./ui";
import { ClusterDialog } from "./ClusterDialog";
import { jobLabel, jobStateLabel } from "../lib/jobPresentation";
import type { ActionPlan, FleetJob } from "./fleetTypes";

/*
 * The Cluster page's own dialogs (VD-200, the ClusterDialogsChange and
 * ClusterDialogsConfirm boards): the body of a reviewed change plan, the
 * words of a confirmation for a destructive action that has no server-side
 * plan, and the current cluster operation. FleetCenter owns their state and
 * every request, and a refusal of their own action is shown inside them.
 */

/** Why a plan that destroys data cannot be approved yet, or undefined once the exact acknowledgement is typed. */
export function planAckReason(plan: ActionPlan, ack: string): string | undefined {
  return plan.data_loss_ack && ack !== plan.data_loss_ack
    ? "Type the exact acknowledgement to accept the data loss and proceed."
    : undefined;
}

/**
 * The body of the "Change plan" dialog: the plan's steps, its expected impact
 * and, for a plan that destroys data, the typed acknowledgement. FleetCenter
 * draws the dialog and its two buttons itself, so the sweep inventory can
 * follow Approve to the operation it starts.
 */
export function ChangePlanBody({ ack, onAck, plan }: { ack: string; onAck: (value: string) => void; plan: ActionPlan }) {
  return (
    <>
      <ol className="cl-ordered">{plan.steps.map((step) => <li key={step}>{step}</li>)}</ol>
      <div className="cl-panel"><strong>Expected impact</strong><p>{plan.impact}</p></div>
      {plan.terminology && <p className="cl-meta">{plan.terminology}</p>}
      {plan.data_loss_ack && (
        <Input
          autoComplete="off"
          id="cluster-data-loss-ack"
          label={plan.ack_prompt
            // R7: the backend's prompt already names the token and asks for
            // it; saying it again doubled the sentence.
            ? <span>{plan.ack_prompt}</span>
            : <span>This is irreversible data loss. Type <strong>{plan.data_loss_ack}</strong> to confirm you accept it.</span>}
          onChange={(event) => onAck(event.target.value)}
          value={ack}
        />
      )}
    </>
  );
}

export interface ConfirmRequest {
  title: string;
  body: string;
  /** The eyebrow; a removal is the default ("Confirm removal"). */
  eyebrow?: string;
  confirmLabel?: string;
  busyLabel?: string;
  /** Load is not destructive: its button is the primary action rather than a red one. */
  constructive?: boolean;
}

/** What a confirmation without a server-side plan says it does. */
export function confirmDescription(body: string) {
  return <p><strong className="cl-strong">What this does.</strong> {body}</p>;
}

/** "Current cluster operation": what it is doing, how far it got, and what can be done about it. */
export function OperationDialog({
  busy,
  canCancel,
  canFinish,
  canRetry,
  error,
  job,
  onBackground,
  onCancelJob,
  onDone,
  onForcedRemoval,
  onRetry,
}: {
  busy: boolean;
  canCancel: boolean;
  canFinish: boolean;
  canRetry: boolean;
  error?: string;
  job: FleetJob;
  onBackground: () => void;
  onCancelJob: () => void;
  onDone: () => void;
  /** Offered when the job is a removal that failed in a way a forced removal can finish. */
  onForcedRemoval?: () => void;
  onRetry: () => void;
}) {
  return (
    <ClusterDialog
      error={error}
      eyebrow="Current cluster operation"
      footer={<>
        <Button onClick={onBackground} variant="quiet">Continue in background</Button>
        {canCancel && <Button disabled={busy} onClick={onCancelJob}>Cancel operation</Button>}
        {canRetry && <Button disabled={busy} onClick={onRetry} variant="primary">Retry safely</Button>}
        {onForcedRemoval && <Button onClick={onForcedRemoval} variant="danger">Review forced removal</Button>}
        {canFinish && <Button onClick={onDone} variant="primary">Done</Button>}
      </>}
      onClose={onBackground}
      title={jobLabel(job.type)}
      titleId="cluster-operation-title"
    >
      <p>{job.message || jobStateLabel(job)}</p>
      <progress className="cl-progress" max="100" value={job.progress}>{job.progress}%</progress>
      <div className="cl-panel"><strong>Current state</strong><p>{jobStateLabel(job)}</p></div>
    </ClusterDialog>
  );
}
