import { useEffect, useId, useState } from "react";
import { apiRequest } from "../lib/api";
import type { AuditEvent } from "../types";
import { ModalShell } from "./ModalShell";
import { Button, Notice } from "./ui";

interface OperationAuditPayload {
  events: AuditEvent[];
  operation_id: string;
  schema: string;
}

function humanize(value: string): string {
  return value.replaceAll("_", " ").replaceAll("-", " ").replaceAll(".", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function displayValue(value: unknown): string | null {
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return humanize(value);
  if (Array.isArray(value) && value.every((item) => ["string", "number", "boolean"].includes(typeof item))) {
    return value.map((item) => displayValue(item)).join(", ");
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
  const generatedId = useId().replaceAll(":", "");
  const titleId = `operation-audit-${generatedId}-title`;
  const descriptionId = `operation-audit-${generatedId}-description`;
  const [payload, setPayload] = useState<OperationAuditPayload | null>(null);
  const [error, setError] = useState("");

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
    <ModalShell describedBy={descriptionId} labelledBy={titleId} onClose={onClose} size="wide">
      <header className="operation-audit__header">
        <div>
          <span className="page-eyebrow">Operation evidence</span>
          <h2 id={titleId}>Activity evidence</h2>
          <p id={descriptionId}>A readable history of the recorded actions and outcomes for this operation.</p>
        </div>
        <Button aria-label="Close activity evidence" onClick={onClose} type="button" variant="quiet">Close</Button>
      </header>
      <div className="operation-audit__body">
        {error && <Notice heading="Evidence could not be loaded" severity="danger">{error}</Notice>}
        {!payload && !error && <p className="operation-audit__loading" role="status">Loading activity evidence...</p>}
        {payload && payload.events.length === 0 && <Notice severity="info">No activity evidence has been recorded for this operation yet.</Notice>}
        {payload?.events.map((event) => {
          const readableDetails = Object.entries(event.details ?? {}).flatMap(([key, value]) => {
            const displayed = displayValue(value);
            return displayed === null ? [] : [{ key, value: displayed }];
          });
          const technicalDetails = Object.fromEntries(
            Object.entries(event.details ?? {}).filter(([, value]) => displayValue(value) === null),
          );
          return (
            <article className="operation-audit__event" key={event.id}>
              <div className="operation-audit__event-heading">
                <div><span>Recorded action</span><h3>{humanize(event.action)}</h3></div>
                <strong data-result={event.result}>{event.result === "success" ? "Succeeded" : "Failed"}</strong>
              </div>
              <dl className="operation-audit__facts">
                <div><dt>When</dt><dd>{eventTime(event.created_at)}</dd></div>
                <div><dt>Operator</dt><dd>{event.actor || "System"}</dd></div>
                {readableDetails.map((detail) => <div key={detail.key}><dt>{humanize(detail.key)}</dt><dd>{detail.value}</dd></div>)}
              </dl>
              {(event.remote_addr || Object.keys(technicalDetails).length > 0) && (
                <details className="operation-audit__technical">
                  <summary>Technical details</summary>
                  {event.remote_addr && <p>Recorded from <code>{event.remote_addr}</code></p>}
                  {Object.keys(technicalDetails).length > 0 && <pre>{JSON.stringify(technicalDetails, null, 2)}</pre>}
                </details>
              )}
            </article>
          );
        })}
      </div>
      <footer className="operation-audit__footer">
        <span>Support reference <code>{supportReference}</code></span>
        <Button onClick={onClose} type="button" variant="primary">Done</Button>
      </footer>
    </ModalShell>
  );
}
