import { useEffect, useMemo, useState } from "react";
import { apiRequest } from "../lib/api";
import { auditMatches } from "../lib/auditSearch";
import { auditActionLabel, auditResultView, type AuditResultTone } from "../lib/auditLabels";
import { auditFromText, auditTargetHasRawId, auditTargetText } from "../lib/auditTargets";
import type { AuditEvent } from "../types";
import { Icon } from "./Icon";
import { usePagination } from "./PaginatedItems";
import { RecordPager, shortStamp } from "./RecordKit";
import { StatusPill } from "./StatusPill";
import { Card, Input, LoadingLines, Notice, type StatusTone } from "./ui";
import "../styles/activity.css";

/** How many audit events the trail reads, and how many a page shows (the ActivityAudit board). */
const AUDIT_WINDOW = 200;
const EVENTS_PER_PAGE = 12;

const RESULT_TONES: Record<AuditResultTone, StatusTone> = {
  success: "success",
  failure: "danger",
  warning: "warning",
  info: "info",
  neutral: "neutral",
};

/**
 * Activity › Security audit (VD-200, the ActivityAudit board): every
 * authenticated change and access event, newest first, searchable by action,
 * user and result.
 */
export function ActivityAudit() {
  const [events, setEvents] = useState<AuditEvent[] | null>(null);
  const [loadError, setLoadError] = useState("");
  const [query, setQuery] = useState("");

  useEffect(() => {
    let active = true;
    apiRequest<AuditEvent[]>(`/audit?limit=${AUDIT_WINDOW}`)
      .then((next) => { if (active) { setEvents(next); setLoadError(""); } })
      .catch((error: unknown) => {
        if (active) setLoadError(error instanceof Error && error.message ? error.message : "Operational history could not be loaded.");
      });
    return () => { active = false; };
  }, []);

  const filtered = useMemo(() => (events ?? []).filter((event) => auditMatches(event, query)), [events, query]);
  const auditPage = usePagination(filtered, EVENTS_PER_PAGE);
  const trimmed = query.trim();
  const note = events === null
    ? null
    : trimmed
      ? `${filtered.length} of ${events.length} events match “${trimmed}”`
      : `${EVENTS_PER_PAGE} events a page.`;

  return (
    <Card
      actions={(
        <div className="acty-search">
          <Icon aria-hidden="true" className="acty-search__icon" name="search" size={16} />
          <Input
            className="acty-search__input"
            label={<span className="sr-only">Search audit events</span>}
            onChange={(event) => { setQuery(event.target.value); auditPage.setPage(1); }}
            placeholder="Search action, user, or result"
            type="search"
            value={query}
          />
        </div>
      )}
      as="section"
      className="acty-audit"
      description={`Authenticated changes and access events · the newest ${AUDIT_WINDOW}`}
      flush
      footer={events !== null && events.length > 0
        ? <RecordPager label="Audit events" note={note} page={auditPage.page} setPage={auditPage.setPage} totalPages={auditPage.totalPages} />
        : undefined}
      heading="Security audit trail"
    >
      {loadError && <div className="acty-list__notice"><Notice severity="danger"><span>{loadError}</span></Notice></div>}
      {events === null && !loadError && <LoadingLines label="Loading the audit trail…" />}
      {events !== null && events.length === 0 && <p className="acty-list__empty">No audit events are recorded yet.</p>}
      {events !== null && events.length > 0 && filtered.length === 0 && (
        <p className="acty-list__empty">No audit event matches “{trimmed}”. Search covers action, user and result.</p>
      )}
      {filtered.length > 0 && (
        <div className="acty-table-wrap">
          <table className="acty-table">
            <caption className="sr-only">Security audit trail</caption>
            <thead><tr><th scope="col">Action</th><th scope="col">User</th><th scope="col">From</th><th scope="col">Target</th><th scope="col">Time</th><th scope="col">Result</th></tr></thead>
            <tbody>
              {auditPage.visible.map((event) => {
                const result = auditResultView(event.result);
                return (
                  <tr key={event.id}>
                    <td data-label="Action">{auditActionLabel(event.action, event.result)}</td>
                    <td data-label="User">{event.actor}</td>
                    <td className="record-mono acty-table__wrap" data-label="From">{auditFromText(event)}</td>
                    <td className="acty-table__wrap ui-muted" data-label="Target">{auditTargetHasRawId(event)
                      ? <details className="audit-target"><summary>{auditTargetText(event)}</summary><code>{event.target}</code></details>
                      : auditTargetText(event)}</td>
                    <td className="ui-muted" data-label="Time"><time dateTime={new Date(event.created_at * 1000).toISOString()}>{shortStamp(event.created_at)}</time></td>
                    <td data-label="Result"><StatusPill label={result.label} tone={RESULT_TONES[result.tone]} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
