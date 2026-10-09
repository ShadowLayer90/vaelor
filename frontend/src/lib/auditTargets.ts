import type { AuditEvent } from "../types";
import { auditActionLabel } from "./auditLabels";
import { jobLabel } from "./jobPresentation";

/**
 * The words for an audit row's target (W6-D1).
 *
 * The backend names each target from the store that owns its kind, chosen by
 * the action that wrote it (`vaelor/audit_targets.py`); this never guesses a
 * type from an id prefix (LESSONS 6). An operation's type is named here
 * through `jobLabel`, the one owner of job words, rather than a second copy.
 */
export function auditTargetText(event: AuditEvent): string {
  const view = event.target_view;
  if (!view) return event.target || "Not about one item";
  if (view.kind === "operation" && view.job_type) return `Operation: ${jobLabel(view.job_type)}`;
  return view.label || event.target || "Not about one item";
}

/** Where a row came from, as the From cell shows it. */
export function auditFromText(event: Pick<AuditEvent, "remote_addr">): string {
  return event.remote_addr || "This appliance";
}

/**
 * Every word an audit row SHOWS, in the order it shows them (FE-W7-2).
 *
 * The cell and the search read this one function, so a row is found by what
 * the owner can see - "Download AI model", "Approved an operation" - and not
 * by the job.create slug or the raw id that sits behind the disclosure
 * (LESSONS 6: two derivations of one row's text drifted apart).
 */
export function auditDisplayedText(event: AuditEvent): string {
  return [
    auditActionLabel(event.action, event.result),
    event.actor,
    auditFromText(event),
    auditTargetText(event),
    event.result,
  ].join(" ");
}

/** Whether the raw id is worth a disclosure: only when the words replaced it. */
export function auditTargetHasRawId(event: AuditEvent): boolean {
  return Boolean(event.target) && auditTargetText(event) !== event.target;
}
