import { useEffect, useId, useState, type ChangeEvent, type FormEvent, type ReactNode } from "react";
import type { UseOperationOwnerResult } from "../hooks/useOperationOwner";
import {
  OPERATION_SCHEMA,
  operationIsTerminal,
  type OperationAction,
  type OperationArtifact,
  type OperationProjection,
} from "../lib/operationOwner";
import { timeAgo } from "../lib/format";
import { Button, Card, Input, Notice, OperationFeedback, Textarea, type OperationState as FeedbackState } from "./ui";
import { OperationAuditDialog } from "./OperationAuditDialog";

type MaybePromise = void | Promise<unknown>;
type OwnerAction = OperationAction | "correct";

export type OperationOwnerMode = "owner" | "history" | "interactive";
export type OperationOwnerOperation = OperationProjection;

/** A presentation owner accepts the lifecycle controller returned by useOperationOwner. */
export type OperationOwnerController = Partial<Pick<
  UseOperationOwnerResult,
  "cancel" | "retry" | "resume" | "discardSavedState" | "done"
>> & Partial<Pick<
  UseOperationOwnerResult,
  "actionPending" | "error" | "actionError" | "resourceRefreshError" | "hasSavedState" | "connection"
>>;

export interface OperationOwnerCorrectionField {
  id: string;
  label: string;
  value?: string;
  type?: "text" | "textarea";
  required?: boolean;
  placeholder?: string;
  hint?: string;
  error?: string;
}

export interface OperationOwnerProps {
  operation: OperationOwnerOperation | null;
  controller?: OperationOwnerController;
  /** Correction is deliberately separate from lifecycle actions in the shared controller. */
  onCorrection?: (values: Record<string, string>) => MaybePromise;
  mode?: OperationOwnerMode;
  title?: ReactNode;
  description?: ReactNode;
  className?: string;
  children?: ReactNode;
}

type Feedback = {
  state: FeedbackState;
  message: string;
};

const actionPendingCopy: Record<OwnerAction, string> = {
  cancel: "Cancellation requested. Waiting for the operation service to confirm it.",
  correct: "Correction submitted. The operation owner will update when it is accepted.",
  retry: "Retry requested. Waiting for the new attempt to start.",
  resume: "Resume requested. Waiting for the saved operation to continue.",
  discard: "Saved draft discarded.",
  done: "Operation closed.",
};

function safeId(value: string): string {
  return value.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "operation";
}

function humanize(value: string | null | undefined): string {
  if (!value) return "Operation state not reported";
  return value.replaceAll("_", " ").replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function humanizePhase(value: string): string {
  const known: Record<string, string> = {
    needs_input: "Waiting for your input",
    needs_approval: "Waiting for approval",
    ready_for_review: "Ready for review",
  };
  if (known[value]) return known[value];
  return /[_-]/.test(value) && !/\s/.test(value) ? humanize(value) : value;
}

function operationStateLabel(value: string): string {
  const known: Record<string, string> = {
    queued: "Queued", running: "Running", waiting: "Waiting", paused: "Paused",
    needs_approval: "Waiting for approval", completed: "Finished", healthy: "Finished",
    failed: "Failed", rejected: "Failed", blocked: "Needs attention",
    interrupted: "Needs attention", cancelled: "Cancelled", superseded: "Superseded",
  };
  return known[value] ?? humanize(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function textValue(value: unknown): string | undefined {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return undefined;
}

function correctionFields(operation: OperationOwnerOperation): OperationOwnerCorrectionField[] {
  const correction = operation.correction;
  if (!correction) return [];
  const rawFields = correction["fields"];
  if (Array.isArray(rawFields)) {
    return rawFields.flatMap((rawField, index) => {
      if (!isRecord(rawField)) return [];
      const id = textValue(rawField.id) ?? `correction-${index + 1}`;
      return [{
        id,
        label: textValue(rawField.label) ?? humanize(id),
        value: textValue(rawField.value),
        type: rawField.type === "textarea" ? "textarea" : "text",
        required: rawField.required === true,
        placeholder: textValue(rawField.placeholder),
        hint: textValue(rawField.hint),
        error: textValue(rawField.error),
      }];
    });
  }
  if (isRecord(rawFields)) {
    return Object.entries(rawFields).map(([id, rawField]) => {
      if (!isRecord(rawField)) return { id, label: humanize(id), value: textValue(rawField) };
      return {
        id,
        label: textValue(rawField.label) ?? humanize(id),
        value: textValue(rawField.value),
        type: rawField.type === "textarea" ? "textarea" as const : "text" as const,
        required: rawField.required === true,
        placeholder: textValue(rawField.placeholder),
        hint: textValue(rawField.hint),
        error: textValue(rawField.error),
      };
    });
  }
  if (isRecord(correction.values)) {
    return Object.entries(correction.values).map(([id, value]) => ({ id, label: humanize(id), value: textValue(value) }));
  }
  return correction.field ? [{
    id: correction.field,
    label: humanize(correction.field),
    value: textValue(correction.values?.[correction.field]),
  }] : [];
}

function formatTimestamp(value: string | number | null | undefined): string | null {
  if (value === null || value === undefined || value === "") return null;
  const numeric = typeof value === "number" ? value : Number(value);
  const date = Number.isFinite(numeric)
    ? new Date(numeric < 100000000000 ? numeric * 1000 : numeric)
    : new Date(String(value));
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/**
 * How long an unfinished operation may go without an event before its card
 * stops claiming to be working on it.
 *
 * Seen on the appliance 2026-08-11: a row whose last event was **eight days
 * earlier** still animated its progress bar and read "Another local
 * specialist is running. Retry when it finishes." A moving bar is the
 * strongest claim a screen can make that something is happening right now,
 * and for a job abandoned a week ago it is simply false — the same family as
 * the green dot beside "Paused" (#52) and the failed card that still showed a
 * percentage (#148).
 *
 * **One hour, and the width is deliberate.** Fifteen minutes was tried first
 * and is the wrong instinct: a step that legitimately reports nothing while
 * it works - a large image pull onto a Raspberry Pi's SD card, a model
 * download, a queued specialist waiting its turn - would be publicly accused
 * of having stopped while it was busy. That is the same defect as the
 * animation, pointed the other way, and it is worse because it invites the
 * owner to intervene in something healthy.
 *
 * An hour cannot be reached by any single step this appliance performs, and
 * still calls the eight-day case what it is. The card never states this
 * number: it states the age of the last event, so the reader judges rather
 * than being handed a threshold.
 */
export const OPERATION_STALL_AFTER_MS = 60 * 60 * 1000;

function timestampMs(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const numeric = typeof value === "number" ? value : Number(value);
  const date = Number.isFinite(numeric)
    ? new Date(numeric < 100000000000 ? numeric * 1000 : numeric)
    : new Date(String(value));
  return Number.isNaN(date.getTime()) ? null : date.getTime();
}

/** Whether an unfinished operation has gone quiet for long enough to say so. */
/**
 * The states where silence means Vaelor has stopped reporting, rather than
 * that it is correctly waiting.
 *
 * **Excluding only terminal states was wrong.** `needs_approval` exists to
 * wait for a human, and `paused` exists because someone asked it to stop — so
 * an owner who left a plan pending overnight was told the next morning that
 * it "stopped reporting… it may have been interrupted", about an operation
 * waiting for them exactly as designed. That is #52's family (a green dot
 * beside "Paused") re-committed inside the fix for it.
 *
 * Listed positively so a state added to the vocabulary has to declare which
 * side it falls on, rather than defaulting into being accused.
 */
const REPORTING_OPERATION_STATES: ReadonlySet<string> = new Set([
  "running",
]);

/**
 * States where the appliance is waiting rather than working.
 *
 * **This is the backend's own list, copied deliberately.**
 * `vaelor/operation_projection.py:_progress` gives exactly
 * `queued/draft/ready/waiting/needs_approval/paused` no determinate progress,
 * and `job_projection.py` calls the same family liveness `waiting`. Deriving
 * a second vocabulary here is how two mechanisms end up answering one
 * question differently, which is the failure mode behind #52 and #74 — and
 * the first version of this fix did exactly that, keeping `ready` on the
 * reporting side while the backend had it on this one. `claim_next` leaves a
 * `ready` task untouched while its dependency runs, so a task queued behind a
 * model download sits there for hours by design.
 *
 * If this list and the Python one ever diverge, the Python one is right: it
 * is the thing that computes what the card is drawn from.
 */
const WAITING_OPERATION_STATES: ReadonlySet<string> = new Set([
  "queued",
  "draft",
  "ready",
  "waiting",
  "needs_approval",
  "paused",
]);

/** Whether this operation is waiting on something rather than working. */
export function operationIsWaiting(state: string | null | undefined): boolean {
  return WAITING_OPERATION_STATES.has(String(state ?? ""));
}

export function operationHasStalled(
  operation: OperationOwnerOperation, now = Date.now(),
): boolean {
  // #180: the server owns this verdict now. When the projection carries a
  // `staleness` block, render it rather than recomputing here - the client held
  // its own hour threshold and its own set of states that can stall, and two
  // mechanisms answering one question is exactly how the `ready` case drifted
  // (#201). The computation below survives only for pre-projection fixtures
  // that have no `staleness` block.
  const verdict = operation.staleness;
  if (verdict && typeof verdict.stale === "boolean") return verdict.stale;
  if (!REPORTING_OPERATION_STATES.has(String(operation.state))) return false;
  const last = timestampMs(operation.timestamps?.updated_at)
    ?? timestampMs(operation.timestamps?.started_at);
  // No timestamp at all is not evidence of a stall. Claiming one would be the
  // same invention in the other direction.
  if (last === null) return false;
  return now - last > OPERATION_STALL_AFTER_MS;
}

function timingContext(operation: OperationOwnerOperation): string {
  const started = formatTimestamp(operation.timestamps?.started_at);
  const updated = formatTimestamp(operation.timestamps?.updated_at ?? operation.timestamps?.completed_at);
  if (started && updated) return `Started ${started} · updated ${updated}`;
  if (started) return `Started ${started}`;
  if (updated) return `Updated ${updated}`;
  return "Timing not reported";
}

function artifactName(artifact: OperationArtifact, index: number): string {
  return artifact.name ?? artifact.kind ?? `Artifact ${index + 1}`;
}

function operationResultSummary(operation: OperationOwnerOperation): string | null {
  // The backend builds a failed operation's `message` and `result_summary`
  // from the same record, so a summary string that equals the error is the
  // error arriving by a second door — the attention Notice already shows it
  // (#148 review). Skip it and fall through to the state-appropriate line.
  const errorMessage = operation.recoverable_error?.message?.trim();
  const fresh = (value: unknown): value is string =>
    typeof value === "string" && Boolean(value.trim()) && value.trim() !== errorMessage;
  const direct = operation.result?.summary;
  if (fresh(direct)) return direct;
  const data = operation.result_summary?.data;
  if (fresh(data)) return data;
  if (isRecord(data)) {
    for (const key of ["summary", "message", "title", "dismissal"]) {
      const value = data[key];
      if (fresh(value)) return value;
    }
    if (!operationIsTerminal(operation.state)) {
      return "The operation returned technical details while it continues.";
    }
    // #148: this sentence read "The operation completed." for every terminal
    // state, so a card whose status said Failed announced completion two
    // lines below it. The fallback must agree with the state beside it, and
    // the failure endings promise no location — the reason lives in the
    // attention notice, and the technical block only renders when there is
    // something distinct in it.
    const endings: Record<string, string> = {
      failed: "The operation did not complete.",
      rejected: "The operation was rejected.",
      cancelled: "The operation was cancelled.",
      superseded: "A newer attempt replaced this operation.",
    };
    return endings[operation.state]
      ?? "The operation completed. Technical details are available below.";
  }
  return null;
}

function operationTechnicalResult(operation: OperationOwnerOperation): string | null {
  const data = operation.result_summary?.data;
  if (!isRecord(data) || ["summary", "message", "title", "dismissal"].some((key) => typeof data[key] === "string" && data[key].trim())) return null;
  return JSON.stringify(data, null, 2);
}

function actionLabel(action: OwnerAction): string {
  return {
    cancel: "Cancel operation",
    correct: "Apply correction",
    retry: "Retry operation",
    resume: "Resume operation",
    discard: "Discard saved draft",
    done: "Done",
  }[action];
}

export function OperationOwner({
  operation,
  controller,
  onCorrection,
  mode = "owner",
  title = "Operation",
  description,
  className,
  children,
}: OperationOwnerProps) {
  const generatedId = useId().replaceAll(":", "");
  const [correctionValues, setCorrectionValues] = useState<Record<string, string>>({});
  const [discardPending, setDiscardPending] = useState(false);
  const [busyAction, setBusyAction] = useState<OwnerAction | null>(null);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [auditOpen, setAuditOpen] = useState(false);

  useEffect(() => {
    if (!operation) {
      setCorrectionValues({});
      setDiscardPending(false);
      return;
    }
    const nextValues: Record<string, string> = {};
    for (const field of correctionFields(operation)) nextValues[field.id] = field.value ?? "";
    setCorrectionValues(nextValues);
    setDiscardPending(false);
  }, [operation?.operation_key]);

  if (!operation || operation.schema !== OPERATION_SCHEMA) return null;

  const readOnly = mode === "history";
  const titleId = `operation-owner-${generatedId}-title`;
  const descriptionId = `operation-owner-${generatedId}-description`;
  const correctionId = `operation-owner-${generatedId}-correction`;
  const resultId = `operation-owner-${generatedId}-result`;
  const fields = correctionFields(operation);
  const error = operation.recoverable_error;
  const correctionMessage = textValue(operation.correction?.message);
  const currentStep = operation.phase ? humanizePhase(operation.phase) : null;
  const progress = operation.progress;
  const progressValue = progress.value;
  const hasValidProgress = typeof progressValue === "number" && Number.isFinite(progressValue) && progressValue >= 0 && progressValue <= 100;
  const determinate = progress.determinate && hasValidProgress;
  const timing = timingContext(operation);
  const stalled = operationHasStalled(operation);
  const waiting = operationIsWaiting(operation.state);
  const lastEventMs = timestampMs(operation.timestamps?.updated_at)
    ?? timestampMs(operation.timestamps?.started_at);
  const stalledSince = lastEventMs === null ? "its last event" : timeAgo(lastEventMs);
  const waitingSince = lastEventMs === null ? null : timeAgo(lastEventMs);
  const resultSummary = operationResultSummary(operation);
  const technicalResult = operationTechnicalResult(operation);
  const artifacts = operation.artifacts ?? [];
  const resourceEndpoint = operation.result?.open_url ?? operation.result?.resource_url ?? null;
  const actionPending = controller?.actionPending ?? busyAction;
  const controllerError = controller?.actionError ?? controller?.resourceRefreshError ?? controller?.error ?? null;
  const feedbackState = feedback?.state ?? (controllerError ? "error" : "idle");
  // #148: one failed run printed the same error sentence four times in one
  // card, because `operation.message` mirrors `recoverable_error.message` and
  // three surfaces each rendered their copy. The attention Notice owns the
  // error; everything else shows a text only when it says something different.
  const statusMessage = operation.message && operation.message !== error?.message
    ? operation.message
    : undefined;
  const feedbackMessage = feedback?.message ?? controllerError ?? statusMessage;
  const correctionInstruction = correctionMessage && correctionMessage !== error?.message
    ? correctionMessage
    : undefined;
  const canCancel = operation.permissions?.cancel === true;
  const canRetry = operation.permissions?.retry === true;
  const canResume = controller?.hasSavedState === true && controller.connection === "reconnecting";
  const canDiscard = operation.permissions?.discard === true;
  const canDone = operationIsTerminal(operation.state);
  const correctionAvailable = operation.correction?.["available"] !== false;
  const canCorrect = !readOnly && Boolean(onCorrection) && correctionAvailable && fields.length > 0;
  const anyBusy = Boolean(actionPending);

  const run = async (action: OwnerAction, callback: (() => MaybePromise) | undefined) => {
    if (!callback || anyBusy) return;
    setBusyAction(action);
    setFeedback({ state: "pending", message: actionPendingCopy[action] });
    try {
      await callback();
      if (action === "discard" || action === "done") setFeedback({ state: "success", message: actionPendingCopy[action] });
      else setFeedback(null);
    } catch (caught) {
      setFeedback({
        state: "error",
        message: caught instanceof Error ? caught.message : "The operation request could not be completed. Try again.",
      });
    } finally {
      setBusyAction(null);
    }
  };

  const submitCorrection = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!onCorrection || anyBusy) return;
    await run("correct", () => onCorrection(correctionValues));
  };

  const actionButton = (
    action: OperationAction,
    available: boolean,
    callback: (() => Promise<OperationProjection | null>) | undefined,
    variant: "primary" | "secondary" | "danger" = "secondary",
  ) => available && (
    <Button
      busy={actionPending === action}
      disabled={anyBusy || !callback}
      disabledReason={!callback ? "The operation controller did not provide this action." : undefined}
      onClick={() => void run(action, callback)}
      variant={variant}
    >
      {actionLabel(action)}
    </Button>
  );

  return (
    <Card
      aria-describedby={description ? descriptionId : undefined}
      className={["operation-owner", readOnly ? "operation-owner--history" : "operation-owner--interactive", className].filter(Boolean).join(" ")}
      data-mode={readOnly ? "history" : "owner"}
      data-operation-id={operation.operation_id}
      heading={<span id={titleId}>{title}</span>}
      description={description ? <span id={descriptionId}>{description}</span> : undefined}
    >
      <div className="operation-owner__header">
        {currentStep && <div className="operation-owner__step">
          <span className="operation-owner__eyebrow">Current step</span>
          <strong>{currentStep}</strong>
        </div>}
        <span className="operation-owner__state" data-state={operation.state}>{operationStateLabel(operation.state)}</span>
      </div>

      {operationIsTerminal(operation.state) ? (
        /* #148: an ended operation kept its progress bar and stage label, so
           a Failed card also said "In progress" beside a partly-filled bar.
           Once the state is terminal there is no progress to report — only
           when it ran. */
        /* aria-live stays: a polled operation ending swaps this region's
           content, and the always-live progress region was what announced
           the transition for successes, which raise no alert. */
        <div className="operation-owner__progress" aria-live="polite">
          <div className="operation-owner__progress-heading">
            <span>No further progress — this operation has ended.</span>
            <span>{timing}</span>
          </div>
        </div>
      ) : (
        <div className="operation-owner__progress" aria-live="polite">
          <div className="operation-owner__progress-heading">
            <span>{stalled
              ? `Stopped reporting — nothing has been recorded since ${stalledSince}.`
              : waiting
                ? (waitingSince
                  ? `Waiting — nothing has run against this operation for ${waitingSince}.`
                  : "Waiting — this operation has not started yet.")
                : determinate ? `${progressValue}% complete` : progress.label || "Progress is not available"}</span>
            <span>{timing}</span>
          </div>
          {/*
            * A stalled operation renders no bar at all — not a frozen one, not
            * an animating one. An indeterminate `<progress>` sweeps forever,
            * which is a claim that work is happening now; on this appliance
            * one swept for eight days. The heading above carries the age, so
            * the reader gets the fact instead of the animation.
            *
            * **A waiting operation gets the same treatment, and review is why.**
            * The first version of this fix stopped calling `needs_approval` and
            * `paused` stalled, and left them sweeping — so an operation the
            * owner had paused animated under the words "In progress", which is
            * #52's contradiction rebuilt one component over. The bar is a claim
            * about *now*, and "now" is when nothing is running.
            */}
          {stalled || waiting ? null : determinate ? (
            <progress aria-label="Operation progress" className="operation-owner__progress-bar" max={100} value={progressValue}>{progressValue}%</progress>
          ) : (
            <progress aria-label="Operation progress" className="operation-owner__progress-bar" />
          )}
          {stalled && <small>Nothing has reported against this operation since the time above. It may have been interrupted; the actions below still apply.</small>}
          {waiting && <small>This operation is waiting rather than running, so there is nothing to report against it yet. The actions below still apply.</small>}
          {!stalled && !waiting && !determinate && <small>The service has not supplied a reliable percentage for this step.</small>}
        </div>
      )}

      <OperationFeedback className="operation-owner__feedback" message={feedbackMessage} state={feedbackState} />

      {error?.message && (
        <Notice className="operation-owner__error" heading="This operation needs attention." severity={error.recoverable === false ? "danger" : "warning"}>
          <span>{error.message}</span>
          {/* The raw error code (`operation_failed`) was shown to the owner
              here; the support reference at the foot of the card is the
              identifier meant for that job (#148). The correction sentence
              renders with its fields when there are any. */}
          {correctionInstruction && fields.length === 0 && <small>{correctionInstruction}</small>}
        </Notice>
      )}

      {canCorrect && (
        <form aria-labelledby={correctionId} className="operation-owner__correction" onSubmit={(event) => void submitCorrection(event)}>
          <div>
            <h3 id={correctionId}>Correct and continue</h3>
            <p>{correctionInstruction ?? "Update the requested value here; the operation will keep its identity and history."}</p>
          </div>
          <div className="operation-owner__fields">
            {fields.map((field) => {
              const fieldId = `operation-owner-${generatedId}-${safeId(field.id)}`;
              const updateValue = (event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => setCorrectionValues((current) => ({ ...current, [field.id]: event.target.value }));
              const shared = {
                id: fieldId,
                label: field.label,
                hint: field.hint,
                error: field.error,
                placeholder: field.placeholder,
                required: field.required,
                value: correctionValues[field.id] ?? "",
              };
              return field.type === "textarea"
                ? <Textarea key={field.id} {...shared} onChange={(event) => updateValue(event)} />
                : <Input key={field.id} {...shared} onChange={(event) => updateValue(event)} />;
            })}
          </div>
          <Button busy={busyAction === "correct"} disabled={anyBusy || !onCorrection} disabledReason={!onCorrection ? "The owner did not provide a correction callback." : undefined} type="submit" variant="primary">Apply correction</Button>
        </form>
      )}

      {fields.length > 0 && !canCorrect && (
        <section aria-labelledby={`${correctionId}-summary`} className="operation-owner__correction operation-owner__correction--readonly">
          <div>
            <h3 id={`${correctionId}-summary`}>Correction fields</h3>
            <p>{correctionInstruction ?? "The correction values recorded for this operation."}</p>
          </div>
          <dl className="operation-owner__correction-values">
            {fields.map((field) => <div key={field.id}><dt>{field.label}</dt><dd>{field.value || "Value not supplied"}</dd></div>)}
          </dl>
        </section>
      )}

      {resultSummary || artifacts.length > 0 ? (
        <section aria-labelledby={resultId} className="operation-owner__result">
          <h3 id={resultId}>Result</h3>
          {resultSummary && <p>{resultSummary}</p>}
          {technicalResult && (
            <details>
              <summary>Technical result details</summary>
              <pre>{technicalResult}</pre>
            </details>
          )}
          {artifacts.length > 0 && (
            <ul className="operation-owner__artifacts">
              {artifacts.map((artifact, index) => {
                const href = artifact.url;
                return <li key={artifact.id ?? `${artifactName(artifact, index)}-${index}`}>{href ? <a href={href}>{artifactName(artifact, index)}</a> : <span>{artifactName(artifact, index)}</span>}{artifact.kind && <small>{artifact.kind}</small>}</li>;
              })}
            </ul>
          )}
        </section>
      ) : null}

      {children}

      <div className="operation-owner__links">
        {resourceEndpoint && <a href={resourceEndpoint} rel="noopener noreferrer" target="_blank">Open installed app</a>}
        {operation.audit_link && <Button onClick={() => setAuditOpen(true)} type="button" variant="quiet">View Activity evidence</Button>}
        <span className="operation-owner__identity">Support reference <code>{operation.operation_id.includes(":") ? operation.operation_id.split(":").slice(1).join(":").slice(-8) : operation.operation_id.slice(-8)}</code></span>
      </div>

      {!readOnly && (
        <footer className="operation-owner__actions" aria-label="Operation actions">
          {actionButton("cancel", canCancel, controller?.cancel)}
          {actionButton("retry", canRetry, controller?.retry)}
          {actionButton("resume", canResume, controller?.resume)}
          {canDiscard && (discardPending ? (
            <span className="operation-owner__discard-confirmation" role="group" aria-label="Confirm discard saved state">
              <span>Discard this saved operation state? This cannot resume the saved draft.</span>
              <Button busy={actionPending === "discard"} disabled={anyBusy || !controller?.discardSavedState} disabledReason={!controller?.discardSavedState ? "The operation controller did not provide discard saved draft." : undefined} onClick={() => { setDiscardPending(false); void run("discard", controller?.discardSavedState); }} variant="danger">Confirm discard saved draft</Button>
              <Button disabled={anyBusy} onClick={() => setDiscardPending(false)}>Keep saved state</Button>
            </span>
          ) : <Button disabled={anyBusy || !controller?.discardSavedState} disabledReason={!controller?.discardSavedState ? "The operation controller did not provide discard saved draft." : undefined} onClick={() => setDiscardPending(true)} variant="danger">Discard saved draft</Button>)}
          {actionButton("done", canDone, controller?.done, "primary")}
        </footer>
      )}
      {auditOpen && operation.audit_link && (
        <OperationAuditDialog auditLink={operation.audit_link} onClose={() => setAuditOpen(false)} operationId={operation.operation_id} />
      )}
    </Card>
  );
}
