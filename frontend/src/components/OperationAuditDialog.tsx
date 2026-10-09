import { useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { auditActionLabel } from "../lib/auditLabels";
import { jobLabel } from "../lib/jobPresentation";
import type { AuditEvent } from "../types";
import { RecordDialog, RecordFacts } from "./RecordKit";
import { StatusPill } from "./StatusPill";
import { Button, LoadingLines, Notice, type StatusTone } from "./ui";

interface OperationAuditPayload {
  events: AuditEvent[];
  operation_id: string;
  schema: string;
}

/*
 * W5-D6: this dialog title-cased every identifier it was given - "Job Create",
 * "Type Host Memory Optimize", "Resource Id" - after the card titles had been
 * fixed (W4d-D2). The words now come from their one owner: an action from
 * auditLabels, an operation type from jobLabels. A detail this dialog has no
 * name for is shown as recorded, under its key, never title-cased into
 * something that looks like a sentence (LESSONS 5).
 */

/** Detail keys an owner reads, in the owner's words. */
const DETAIL_NAMES: Record<string, string> = {
  type: "Operation",
  deduplicated: "Joined an operation already queued",
  reason: "Reason",
  role: "Role",
};

/** Detail keys that identify records rather than describe the action. */
const TECHNICAL_DETAILS = new Set([
  "resource_id", "display_identity", "job_id", "operation_id", "endpoint_id",
  "ledger", "source_operation_id", "error_code",
]);

/** What happened, for each result the audit writes. */
const RESULT_WORDS: Record<string, string> = {
  success: "Succeeded",
  failure: "Failed",
  accepted: "Accepted",
  refused: "Refused",
  denied: "Refused",
};

/** The pill tone for each result: green only for a success, red only for a failure. */
const RESULT_TONES: Record<string, StatusTone> = {
  success: "success",
  failure: "danger",
  accepted: "info",
};

function resultWords(result: string): string {
  return RESULT_WORDS[result] ?? (result ? result[0].toUpperCase() + result.slice(1) : "Recorded");
}

function displayValue(key: string, value: unknown): string | null {
  if (TECHNICAL_DETAILS.has(key)) return null;
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return key === "type" ? jobLabel(value) : value;
  if (Array.isArray(value) && value.every((item) => ["string", "number", "boolean"].includes(typeof item))) {
    return value.map((item) => String(item)).join(", ");
  }
  return null;
}

function eventTime(value: number): string {
  return new Date(value < 100000000000 ? value * 1000 : value).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function OperationAuditDialog({
  auditLink,
  onClose,
  operationId,
}: {
  auditLink: string;
  onClose: () => void;
  operationId: string;
}) {
  const [payload, setPayload] = useState<OperationAuditPayload | null>(null);
  const [error, setError] = useState("");
  const done = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    let active = true;
    const path = auditLink.startsWith("/api/v2") ? auditLink.slice(7) : auditLink;
    void apiRequest<OperationAuditPayload>(path, { cache: "no-store" })
      .then((result) => { if (active) setPayload(result); })
      .catch((caught) => { if (active) setError(caught instanceof Error ? caught.message : "Activity evidence is unavailable."); });
    return () => { active = false; };
  }, [auditLink]);

  const supportReference = operationId.includes(":")
    ? operationId.split(":").slice(1).join(":").slice(-8)
    : operationId.slice(-8);

  return (
    <RecordDialog
      actions={<Button onClick={onClose} ref={done} type="button" variant="primary">Done</Button>}
      alert={false}
      description="A readable history of the recorded actions and outcomes for this operation."
      eyebrow="Operation evidence"
      eyebrowTone="plain"
      footerStart={<span className="ui-muted ui-small">Support reference <code className="record-mono">{supportReference}</code></span>}
      headerAction={<Button aria-label="Close activity evidence" className="record-ghost" onClick={onClose} type="button" variant="quiet">Close</Button>}
      initialFocusRef={done}
      onClose={onClose}
      title="Activity evidence"
      wide
    >
      {error && <Notice heading="Evidence could not be loaded." severity="danger">{error}</Notice>}
      {!payload && !error && <LoadingLines label="Loading activity evidence..." lines={1} />}
      {payload && payload.events.length === 0 && <Notice severity="info">No audit record names this operation. Actions taken through the console are listed here when they record the operation they started.</Notice>}
      {payload?.events.map((event) => {
        const readableDetails = Object.entries(event.details ?? {}).flatMap(([key, value]) => {
          const displayed = displayValue(key, value);
          return displayed === null ? [] : [{ key, value: displayed }];
        });
        const technicalDetails = Object.fromEntries(
          Object.entries(event.details ?? {}).filter(([key, value]) => displayValue(key, value) === null),
        );
        return (
          <article className="record-evidence" key={event.id}>
            <div className="record-evidence__heading">
              <div><span className="ui-muted ui-small">Recorded action</span><h3>{auditActionLabel(event.action, event.result)}</h3></div>
              <StatusPill label={resultWords(event.result)} tone={RESULT_TONES[event.result] ?? "neutral"} />
            </div>
            <RecordFacts facts={[
              { key: "when", label: "When", value: eventTime(event.created_at) },
              { key: "operator", label: "Operator", value: event.actor || "System" },
              ...readableDetails.map((detail) => ({ key: detail.key, label: DETAIL_NAMES[detail.key] ?? detail.key, value: detail.value })),
            ]} />
            {(event.remote_addr || Object.keys(technicalDetails).length > 0) && (
              <details className="record-evidence__technical">
                <summary>Technical details</summary>
                {event.remote_addr && <p className="ui-muted ui-small">Recorded from <code className="record-mono">{event.remote_addr}</code></p>}
                {Object.keys(technicalDetails).length > 0 && <pre className="record-mono">{JSON.stringify(technicalDetails, null, 2)}</pre>}
              </details>
            )}
          </article>
        );
      })}
    </RecordDialog>
  );
}
