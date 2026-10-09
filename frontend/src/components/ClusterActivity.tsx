import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ApiError, apiRequest } from "../lib/api";
import { auditMatches } from "../lib/auditSearch";
import { auditFromText, auditTargetHasRawId, auditTargetText } from "../lib/auditTargets";
import { auditActionLabel, auditResultView } from "../lib/auditLabels";
import { exactTime } from "../lib/format";
import { jobLabel } from "../lib/jobPresentation";
import { routeHref } from "../lib/navigation";
import {
  isOperationProjection,
  operationIsTerminal,
  operationRevisionKey,
  type OperationProjection,
} from "../lib/operationOwner";
import type { AuditEvent, Session } from "../types";
import { activityHashForSection, readOperationSummary } from "./ActivityCenter";
import { ClusterAlerts } from "./ClusterAlerts";
import { ClusterCard, FilterChips, IconTile } from "./ClusterPrimitives";
import { Icon, type IconName } from "./Icon";
import { ClusterConfirm } from "./ClusterConfirm";
import { OperationOwner } from "./OperationOwner";
import { usePagination } from "./PaginatedItems";
import { StatusPill } from "./StatusPill";
import { Button, DataTable, Input, LoadingLines, Notice } from "./ui";
import type { StatusTone } from "./ui";
import "../styles/cluster-activity.css";

/**
 * The Cluster page's Activity tab (VD-200, the ClusterActivity and
 * ClusterActivityAlerts boards): the operation record and the security audit
 * trail side by side, a pointer to the recovery checkpoints, then the alert
 * rules, their delivery channels and the schedules.
 *
 * It reads the same routes the top-level Activity page reads - `/operations`
 * with the server's partition (`summary`, ACC-125) and `?bucket=`, `/audit`,
 * `/checkpoints` - and labels them through the same `lib` helpers, so a row
 * reads the same words on both screens. The feed is the whole appliance's
 * operation record, not a cluster-only slice; enrolment, model and app deploy,
 * telemetry install and drain appear here among the rest. Managing restore
 * points lives on the Activity page; this tab says how many there are and
 * links there.
 */

type OperationFilter = "all" | "in_progress" | "attention" | "finished";

interface OperationPage {
  filter: OperationFilter;
  operations: OperationProjection[];
  matched: number | null;
}

const OPERATION_WINDOW = 50;
const OPERATION_PAGE_SIZE = 6;
const AUDIT_PAGE_SIZE = 5;
const POLL_MS = 3000;

/** What each chip lists, named for the notice when that list cannot be read. */
const FILTER_LIST_NAMES: Record<OperationFilter, string> = {
  all: "All operations",
  in_progress: "Operations in progress",
  attention: "Operations that need attention",
  finished: "Finished operations",
};

function operationsPath(filter: OperationFilter): string {
  return filter === "all"
    ? `/operations?limit=${OPERATION_WINDOW}`
    : `/operations?limit=${OPERATION_WINDOW}&bucket=${filter}`;
}

function readOperations(payload: unknown): OperationProjection[] {
  const record = payload && typeof payload === "object" ? (payload as Record<string, unknown>) : {};
  const rows = Array.isArray(payload) ? payload : Array.isArray(record.operations) ? record.operations : [];
  const seen = new Set<string>();
  return rows.filter((row): row is OperationProjection => {
    if (!isOperationProjection(row)) return false;
    const key = operationRevisionKey(row);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function readMatched(payload: unknown): number | null {
  if (!payload || typeof payload !== "object") return null;
  const matched = (payload as Record<string, unknown>).matched;
  return typeof matched === "number" && Number.isInteger(matched) && matched >= 0 ? matched : null;
}

/** An operation timestamp (epoch seconds, milliseconds or ISO text) in milliseconds. */
function toMillis(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value < 1e12 ? value * 1000 : value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Date.parse(value);
    return Number.isNaN(parsed) ? null : parsed;
  }
  return null;
}

/** "now", "22m", "2h", "1d": the row's age at a glance; the exact time is on hover. */
function shortAge(at: number, now = Date.now()): string {
  const seconds = Math.max(0, Math.floor((now - at) / 1000));
  if (seconds < 60) return "now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}

/** A clock time today, "yesterday", or the date: how the audit table names a moment. */
function clockText(at: number, now = new Date()): string {
  const when = new Date(at);
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (at >= startOfToday) return when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (at >= startOfToday - 86_400_000) return "yesterday";
  return when.toLocaleDateString();
}

const TASK_ORIGINS: Record<string, string> = {
  alert_rule: "Alert rule fired",
  schedule: "Scheduled run",
  person: "Agent task",
};

/** What an operation was and where it came from (ACC-127: a fired alert names its rule). */
function operationHeading(operation: OperationProjection): { origin: string; title: string } {
  if (operation.ledger !== "agent_tasks") return { origin: "Workload operation", title: jobLabel(operation.type) };
  return {
    origin: TASK_ORIGINS[operation.origin ?? ""] ?? "Agent task",
    title: operation.title?.trim() || jobLabel(operation.type),
  };
}

function operationIcon(operation: OperationProjection): IconName {
  if (operation.ledger === "agent_tasks") return "assistant";
  const type = operation.type;
  if (/backup|checkpoint|restore|volume/.test(type)) return "database";
  if (/model|llm|gpu|inference/.test(type)) return "gpu";
  if (/enrol|enroll|join|worker|node|fleet|cluster\./.test(type)) return "cluster";
  if (/compose|app|deploy/.test(type)) return "apps";
  return "activity";
}

function operationMessage(operation: OperationProjection): string {
  return operation.message || operation.recoverable_error?.message || "";
}

/**
 * The row's state pill. Green only for a finished success; a failure says
 * what failed; a running one carries its percentage only when it reported one.
 */
function operationPill(operation: OperationProjection): { label: string; tone: StatusTone } {
  const state = operation.state;
  // #180: an operation the server says has stopped reporting is not shown as
  // running; its last word is old, so the pill is grey and says so.
  if (!operationIsTerminal(state) && operation.staleness?.stale) return { label: "Stopped reporting", tone: "neutral" };
  if (state === "completed" || state === "healthy") return { label: "Done", tone: "success" };
  if (state === "failed") {
    const why = operation.recoverable_error?.message || operation.message;
    return { label: why ? `Failed: ${why}` : "Failed", tone: "danger" };
  }
  if (state === "running") {
    const value = operation.progress.determinate ? operation.progress.value : null;
    return { label: value !== null ? `Running · ${value}%` : "Running", tone: "info" };
  }
  const named: Partial<Record<string, { label: string; tone: StatusTone }>> = {
    draft: { label: "Draft", tone: "neutral" },
    queued: { label: "Queued", tone: "info" },
    waiting: { label: "Waiting", tone: "info" },
    ready: { label: "Ready", tone: "info" },
    paused: { label: "Paused", tone: "neutral" },
    needs_approval: { label: "Waiting for approval", tone: "warning" },
    rejected: { label: "Rejected", tone: "neutral" },
    cancelled: { label: "Cancelled", tone: "neutral" },
    superseded: { label: "Superseded", tone: "neutral" },
  };
  return named[state] ?? { label: state.replaceAll("_", " "), tone: "neutral" };
}

function auditTone(tone: string): StatusTone {
  if (tone === "failure") return "danger";
  if (tone === "success" || tone === "warning" || tone === "info") return tone;
  return "neutral";
}

/** The foot of a paged card: a line on the left, Previous and Next on the right. */
function Pager({ label, page, setPage, summary, totalPages }: {
  label: string;
  page: number;
  setPage: (next: number) => void;
  summary: ReactNode;
  totalPages: number;
}) {
  return (
    <nav aria-label={`${label} pages`} className="cl-act__pager">
      <span>{summary}</span>
      <Button disabled={page <= 1} onClick={() => setPage(page - 1)} variant="quiet">Previous</Button>
      <Button disabled={page >= totalPages} onClick={() => setPage(page + 1)} variant="quiet">Next</Button>
    </nav>
  );
}

function OperationRow({ onDismissed, operation, session }: {
  onDismissed: () => void;
  operation: OperationProjection;
  session: Session;
}) {
  const [open, setOpen] = useState(false);
  const [dismissing, setDismissing] = useState(false);
  const [confirmDismiss, setConfirmDismiss] = useState(false);
  const [dismissError, setDismissError] = useState("");
  const heading = operationHeading(operation);
  const pill = operationPill(operation);
  const at = toMillis(operation.timestamps.updated_at) ?? toMillis(operation.timestamps.created_at);
  const startedAt = toMillis(operation.timestamps.created_at);
  const route = operation.owner_route ?? operation.owner?.route ?? null;
  const resource = operation.owner_resource ?? operation.owner?.resource ?? null;
  // VD-139: Dismiss is offered only for what the server says needs attention
  // now, and only to the roles its route accepts.
  const dismissable = operation.needs_attention === true
    && (session.user.role === "operator" || session.user.role === "administrator");
  const progress = operation.progress.determinate && operation.progress.value !== null
    ? `${String(operation.progress.value)}%`
    : operationIsTerminal(operation.state) ? "—" : "Not reported";
  const detailId = `cl-op-${operation.operation_id.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  const message = operationMessage(operation);

  const dismiss = async () => {
    setDismissing(true);
    setDismissError("");
    try {
      await apiRequest(`/operations/${operation.operation_id}/dismiss`, { method: "POST" }, session.csrf_token);
      setConfirmDismiss(false);
      onDismissed();
    } catch (error) {
      if (!(error instanceof ApiError)) console.error("Dismissing an operation failed", error);
      setDismissError(error instanceof ApiError && error.message ? error.message : "The dismissal was not saved.");
    } finally {
      setDismissing(false);
    }
  };

  return (
    <li className={"cl-act__op" + (open ? " is-open" : "")} data-operation-key={operation.operation_key}>
      <Button
        aria-controls={detailId}
        aria-expanded={open}
        className="cl-act__op-row"
        data-testid={`activity-operation-${operation.operation_id}`}
        onClick={() => setOpen((current) => !current)}
        variant="quiet"
      >
        <IconTile accent={operation.state === "running"} name={operationIcon(operation)} />
        <span className="cl-act__op-text">
          <strong>{heading.title}</strong>
          <span>{[heading.origin, resource, startedAt !== null ? clockText(startedAt) : null].filter(Boolean).join(" · ")}</span>
        </span>
        <span className="cl-act__op-side">
          <StatusPill label={pill.label} tone={pill.tone} />
          {at !== null && <time dateTime={new Date(at).toISOString()} title={exactTime(at)}>{shortAge(at)}</time>}
        </span>
      </Button>
      {open && (
        <div className="cl-act__op-detail" id={detailId}>
          <dl className="cl-facts">
            <dt>Status</dt><dd data-field="state">{pill.label}</dd>
            <dt>Progress</dt><dd data-field="progress">{progress}</dd>
            <dt>Message</dt><dd data-field="message">{message || "—"}</dd>
            {operation.dismissal && (
              <>
                <dt>Dismissed</dt>
                <dd data-field="dismissal">By {operation.dismissal.dismissed_by || "an operator"}, {exactTime(operation.dismissal.dismissed_at * 1000)}</dd>
              </>
            )}
          </dl>
          {dismissError && !confirmDismiss && <Notice severity="danger">Could not dismiss this: {dismissError}</Notice>}
          <div className="cl-actions">
            {route
              ? <a className="ui-button ui-button--secondary" href={routeHref(route)}>Open</a>
              : <span className="cl-meta">Where it ran is not recorded</span>}
            {dismissable && (
              <Button onClick={() => { setDismissError(""); setConfirmDismiss(true); }} variant="quiet">Dismiss — I've dealt with this</Button>
            )}
          </div>
          <details className="cl-act__technical">
            <summary>Technical details</summary>
            <dl className="cl-facts">
              <dt>Operation ID</dt><dd><code>{operation.operation_id}</code></dd>
              <dt>Source ID</dt><dd><code>{operation.source_id}</code></dd>
              <dt>Revision</dt><dd>{String(operation.revision)}</dd>
              <dt>Retry lineage</dt>
              <dd>Attempt {operation.retry_lineage.attempt}, depth {operation.retry_lineage.depth}{operation.retry_lineage.parent_operation_id ? `, retried from ${operation.retry_lineage.parent_operation_id}` : ""}</dd>
            </dl>
          </details>
          <div className="cl-act__owner" data-testid={`activity-owner-${operation.operation_id}`}>
            <OperationOwner mode="history" operation={operation} />
          </div>
        </div>
      )}
      {/* A dismissal has no undo, so it is confirmed. It clears this item only
          while it stays as it is now; any later change shows it again. */}
      <ClusterConfirm
        busy={dismissing}
        busyLabel="Dismissing…"
        confirmLabel="Dismiss"
        description="It leaves Need attention and every count of it. The record and its history stay, and there is no undo; if the operation changes again it comes back."
        error={dismissError}
        eyebrow="Can't be undone"
        onCancel={() => { if (!dismissing) { setConfirmDismiss(false); setDismissError(""); } }}
        onConfirm={() => void dismiss()}
        open={confirmDismiss}
        title={`Dismiss this item? ${heading.title}`}
      />
    </li>
  );
}

function OperationsCard({ onDismissed, session }: { onDismissed: () => void; session: Session }) {
  const [filter, setFilter] = useState<OperationFilter>("all");
  const [page, setPage] = useState<OperationPage | null>(null);
  const [summary, setSummary] = useState<ReturnType<typeof readOperationSummary>>(null);
  const [readError, setReadError] = useState<{ filter: OperationFilter; message: string } | null>(null);
  const [pageNumber, setPageNumber] = useState(1);
  // The poll and a chip click can both be in flight; an answer is applied
  // only while the filter it was asked for is still the one on screen.
  const filterRef = useRef<OperationFilter>("all");

  const apply = useCallback((asked: OperationFilter, payload: unknown) => {
    if (asked !== filterRef.current) return;
    setPage({ filter: asked, operations: readOperations(payload), matched: readMatched(payload) });
    setSummary(readOperationSummary(payload));
    setReadError(null);
  }, []);
  // A failed read is said, not left as a spinner; the poll keeps retrying it.
  const report = useCallback((asked: OperationFilter, error: unknown) => {
    if (asked !== filterRef.current) return;
    if (!(error instanceof ApiError)) console.error("Reading operations failed", error);
    setReadError({ filter: asked, message: error instanceof ApiError && error.message ? error.message : "The request failed." });
  }, []);
  const read = useCallback((asked: OperationFilter) => apiRequest<unknown>(operationsPath(asked))
    .then((payload) => apply(asked, payload))
    .catch((error: unknown) => report(asked, error)), [apply, report]);

  useEffect(() => {
    void read(filterRef.current);
    let polling = false;
    const poll = () => {
      if (document.hidden || polling) return;
      polling = true;
      void read(filterRef.current).finally(() => { polling = false; });
    };
    const interval = window.setInterval(poll, POLL_MS);
    const visibilityChanged = () => { if (!document.hidden) poll(); };
    document.addEventListener("visibilitychange", visibilityChanged);
    return () => {
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, [read]);

  const choose = (next: OperationFilter) => {
    filterRef.current = next;
    setFilter(next);
    setPageNumber(1);
    void read(next);
  };

  const current = page !== null && page.filter === filter ? page : null;
  const currentError = readError?.filter === filter ? readError : null;
  const operations = current?.operations ?? [];
  const totalPages = Math.max(1, Math.ceil(operations.length / OPERATION_PAGE_SIZE));
  const shownPage = Math.min(pageNumber, totalPages);
  const visible = operations.slice((shownPage - 1) * OPERATION_PAGE_SIZE, shownPage * OPERATION_PAGE_SIZE);
  const total = current?.matched ?? operations.length;
  const first = (shownPage - 1) * OPERATION_PAGE_SIZE + 1;
  const pagerText = shownPage === 1
    ? `Showing the newest ${visible.length} of ${total}.`
    : `Showing ${first}-${first + visible.length - 1} of ${total}.`;
  const windowText = total > operations.length ? ` Only the newest ${operations.length} can be paged here.` : "";
  // A count the server did not report is left off the chip, never shown as 0.
  const count = (value: number | undefined) => (summary ? value : undefined);

  return (
    <ClusterCard
      actions={<span className={"cl-meta" + (readError ? " cl-warn-text" : "")}>{readError ? "Not read just now · retrying every 3 s" : "Live · read every 3 s"}</span>}
      className="cl-act__ops"
      description="A read-only record of every operation, newest first, with where each one ran"
      flush
      title="Operations"
    >
      <div className="cl-act__chips">
        <FilterChips
          label="Show operations"
          onChange={choose}
          options={[
            { value: "all", label: "All operations", count: count(summary?.total) },
            { value: "in_progress", label: "In progress", count: count(summary?.in_progress) },
            { value: "attention", label: "Need attention", count: count(summary?.attention) },
            { value: "finished", label: "Finished", count: count(summary?.finished) },
          ]}
          value={filter}
        />
      </div>
      {currentError && (
        <div className="cl-act__note">
          <Notice severity="warning">
            {FILTER_LIST_NAMES[currentError.filter]} could not be read ({currentError.message}). Vaelor keeps retrying every few seconds.
          </Notice>
        </div>
      )}
      {current === null && !currentError && <div className="cl-act__note"><LoadingLines label="Reading operations" /></div>}
      {current !== null && operations.length === 0 && <p className="cl-act__empty">No operations here.</p>}
      {visible.length > 0 && (
        <ul className="cl-act__op-list">
          {visible.map((operation) => (
            <OperationRow key={operationRevisionKey(operation)} onDismissed={() => { void read(filterRef.current); onDismissed(); }} operation={operation} session={session} />
          ))}
        </ul>
      )}
      {current !== null && operations.length > 0 && (
        <Pager label="Operations" page={shownPage} setPage={setPageNumber} summary={pagerText + windowText} totalPages={totalPages} />
      )}
    </ClusterCard>
  );
}

/**
 * The security audit trail. `reloadKey` re-reads it: a Dismiss writes an audit
 * row, so the trail beside it is read again rather than left a step behind
 * (VD-200 review, as at 68ccdfb).
 */
function AuditCard({ reloadKey }: { reloadKey: number }) {
  const [events, setEvents] = useState<AuditEvent[] | null>(null);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  useEffect(() => {
    let live = true;
    apiRequest<AuditEvent[]>("/audit?limit=200")
      .then((rows) => { if (live) { setEvents(Array.isArray(rows) ? rows : []); setError(""); } })
      .catch((reason: unknown) => {
        if (!live) return;
        if (!(reason instanceof ApiError)) console.error("Reading the audit trail failed", reason);
        setError(reason instanceof ApiError && reason.message ? reason.message : "The audit trail could not be read.");
      });
    return () => { live = false; };
  }, [reloadKey]);
  const filtered = useMemo(() => (events ?? []).filter((event) => auditMatches(event, query)), [events, query]);
  const paging = usePagination(filtered, AUDIT_PAGE_SIZE);

  return (
    <ClusterCard
      actions={(
        <div className="cl-act__search">
          <Icon name="search" />
          <Input
            label={<span className="sr-only">Search audit events</span>}
            onChange={(event) => { setQuery(event.target.value); paging.setPage(1); }}
            placeholder="Search action, user, or result"
            type="search"
            value={query}
          />
        </div>
      )}
      className="cl-act__audit"
      description="Authenticated changes and access events"
      flush
      title="Security audit trail"
    >
      {error && <div className="cl-act__note"><Notice severity="danger">The audit trail could not be read. {error}</Notice></div>}
      {events === null && !error && <div className="cl-act__note"><LoadingLines label="Reading the audit trail" /></div>}
      {events !== null && filtered.length === 0 && (
        <p className="cl-act__empty">{query ? "No audit events match that search." : "No audit events yet."}</p>
      )}
      {filtered.length > 0 && (
        <>
          <DataTable
            caption="Security audit trail"
            columns={[
              { key: "action", label: "Action · user · from" },
              { key: "target", label: "Target" },
              { key: "time", label: "Time" },
              { key: "result", label: "Result" },
            ]}
            // No minimum: a card narrower than the four columns draws each
            // event as a block (styles/cluster-activity.css), never a table
            // scrolled sideways inside the card.
            minWidth={0}
            rows={paging.visible.map((event) => {
              const result = auditResultView(event.result);
              const at = event.created_at * 1000;
              return {
                key: String(event.id),
                cells: {
                  action: (
                    <span className="cl-act__audit-action">
                      <strong>{auditActionLabel(event.action, event.result)}</strong>
                      <span>{event.actor} · from {auditFromText(event)}</span>
                    </span>
                  ),
                  target: auditTargetHasRawId(event)
                    ? <details className="cl-act__target"><summary>{auditTargetText(event)}</summary><code>{event.target}</code></details>
                    : <span className="cl-act__target">{auditTargetText(event)}</span>,
                  time: <time dateTime={new Date(at).toISOString()} title={exactTime(at)}>{clockText(at)}</time>,
                  result: <StatusPill label={result.label} tone={auditTone(result.tone)} />,
                },
              };
            })}
          />
          <Pager
            label="Audit events"
            page={paging.page}
            setPage={paging.setPage}
            summary={`Page ${paging.page} of ${paging.totalPages}`}
            totalPages={paging.totalPages}
          />
        </>
      )}
    </ClusterCard>
  );
}

/** How many restore points exist, said honestly: a list read at its cap says "or more". */
function RecoveryCheckpointsRow() {
  const [count, setCount] = useState<number | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let live = true;
    apiRequest<unknown[]>("/checkpoints?limit=50")
      .then((rows) => { if (live) setCount(Array.isArray(rows) ? rows.length : null); })
      .catch((reason: unknown) => {
        if (!(reason instanceof ApiError)) console.error("Reading the restore points failed", reason);
        if (live) setFailed(true);
      });
    return () => { live = false; };
  }, []);
  const counted = count === null
    ? (failed ? " The number of restore points could not be read." : "")
    : count >= 50 ? " 50 or more restore points."
      : ` ${count} restore ${count === 1 ? "point" : "points"}.`;
  return (
    <ClusterCard
      actions={<a className="cl-act__link" href={activityHashForSection("restore-points")}><Icon name="chevron" />Open on the Activity page</a>}
      className="cl-act__recovery"
      description={`Inspect, verify, restore, or remove configuration restore points for managed apps.${counted}`}
      icon="restore"
      title="Recovery checkpoints"
    />
  );
}

export function ClusterActivity({ session, reloadKey }: { session: Session; reloadKey?: number }) {
  const [dismissals, setDismissals] = useState(0);
  return (
    <div className="cl-stack cl-act">
      <p className="cl-intro">
        What has happened across the fleet: enrolments, model and app deployments, worker telemetry
        installs and drains appear here among the appliance's other operations. Alert rules, where
        alerts go, and schedules are further down this tab.
      </p>
      <div className="cl-act__grid">
        <OperationsCard onDismissed={() => setDismissals((value) => value + 1)} session={session} />
        <AuditCard reloadKey={(reloadKey ?? 0) + dismissals} />
      </div>
      <RecoveryCheckpointsRow />
      <ClusterAlerts reloadKey={reloadKey} session={session} />
    </div>
  );
}
