import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { timeAgo } from "../lib/format";
import { jobLabel } from "../lib/jobPresentation";
import { routeHref } from "../lib/navigation";
import {
  isOperationProjection,
  operationIsTerminal,
  operationRevisionKey,
  type OperationProjection,
} from "../lib/operationOwner";
import type { Session } from "../types";
import { Icon, type IconName } from "./Icon";
import { OperationOwner } from "./OperationOwner";
import { RecordConfirm, RecordFacts, RecordPager } from "./RecordKit";
import { StatusPill } from "./StatusPill";
import { Button, Card, LoadingLines, Notice, type StatusTone } from "./ui";
import "../styles/activity.css";

/**
 * ACC-125. The filters and their counts are the server's partition. They used
 * to be computed here over the newest fifty operations, so a failure older
 * than the fiftieth dropped out of "need attention" while it still needed the
 * owner. `GET /api/v2/operations` now returns `summary`, counted over every
 * operation both ledgers hold by `OperationProjection.page` in
 * `vaelor/operation_projection.py` (which owns the rule), and `?bucket=`
 * returns the newest operations of one filter rather than filtering a window.
 */
export type OperationFilter = "all" | "in_progress" | "attention" | "finished";

export interface OperationSummary {
  total: number;
  in_progress: number;
  attention: number;
  finished: number;
}

interface OperationPage {
  filter: OperationFilter;
  operations: OperationProjection[];
  matched: number | null;
}

const OPERATION_WINDOW = 50;
/** Rows a page of the list shows (the Activity board). */
const ROWS_PER_PAGE = 6;

/** What each filter lists, named for the notice when that list cannot be read. */
const FILTER_LIST_NAMES: Record<OperationFilter, string> = {
  all: "All operations",
  in_progress: "Operations in progress",
  attention: "Operations that need attention",
  finished: "Finished operations",
};

const FILTERS: Array<{ filter: OperationFilter; label: string; count: (summary: OperationSummary) => number }> = [
  { filter: "all", label: "All operations", count: (summary) => summary.total },
  { filter: "in_progress", label: "In progress", count: (summary) => summary.in_progress },
  { filter: "attention", label: "Need attention", count: (summary) => summary.attention },
  { filter: "finished", label: "Finished", count: (summary) => summary.finished },
];

function operationsPath(filter: OperationFilter): string {
  return filter === "all"
    ? `/operations?limit=${OPERATION_WINDOW}`
    : `/operations?limit=${OPERATION_WINDOW}&bucket=${filter}`;
}

function isCount(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

/** The server's partition, or `null` when the response did not carry one. */
export function readOperationSummary(payload: unknown): OperationSummary | null {
  if (!payload || typeof payload !== "object") return null;
  const summary = (payload as Record<string, unknown>).summary;
  if (!summary || typeof summary !== "object") return null;
  const record = summary as Record<string, unknown>;
  const { total, in_progress: inProgress, attention, finished } = record;
  if (!isCount(total) || !isCount(inProgress) || !isCount(attention) || !isCount(finished)) return null;
  return { total, in_progress: inProgress, attention, finished };
}

function readMatched(payload: unknown): number | null {
  if (!payload || typeof payload !== "object") return null;
  const matched = (payload as Record<string, unknown>).matched;
  return isCount(matched) ? matched : null;
}

/*
 * Job names come from `jobPresentation`, the one table every surface reads.
 * `rejected` said "Failed" once, which put rows labelled Failed under the
 * "finished" filter and made the filters look like they overlapped. A
 * rejection is a decision, not a failure.
 */
function friendlyState(state: string): string {
  const known: Record<string, string> = { draft: "Draft", queued: "Queued", running: "Running", waiting: "Waiting", paused: "Paused", ready: "Ready", needs_approval: "Waiting for approval", completed: "Finished", healthy: "Finished", failed: "Failed", rejected: "Rejected", cancelled: "Cancelled", superseded: "Superseded", blocked: "Needs attention", interrupted: "Needs attention" };
  return known[state] ?? state.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

/** The row's pill: the server's attention verdict first, then the state in words. */
function rowPill(operation: OperationProjection): { label: string; tone: StatusTone } {
  if (operation.needs_attention === true) return { label: "Needs attention", tone: "warning" };
  const tones: Record<string, StatusTone> = { running: "info", queued: "info", failed: "danger", blocked: "warning", interrupted: "warning" };
  return { label: friendlyState(operation.state), tone: tones[operation.state] ?? "neutral" };
}

function collectOperations(payload: unknown): OperationProjection[] {
  const queue: unknown[] = [payload];
  const visited = new Set<object>();
  const candidates: unknown[] = [];
  while (queue.length > 0 && candidates.length < 200) {
    const candidate = queue.shift();
    if (isOperationProjection(candidate)) {
      candidates.push(candidate);
      continue;
    }
    if (Array.isArray(candidate)) {
      queue.push(...candidate);
      continue;
    }
    if (!candidate || typeof candidate !== "object" || visited.has(candidate)) continue;
    visited.add(candidate);
    const record = candidate as Record<string, unknown>;
    for (const key of ["operation", "projection", "operations", "items", "data"]) {
      if (key in record) queue.push(record[key]);
    }
  }
  const seen = new Set<string>();
  return candidates.filter((candidate): candidate is OperationProjection => {
    if (!isOperationProjection(candidate)) return false;
    const key = operationRevisionKey(candidate);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function ownerRoute(operation: OperationProjection): string | null {
  return operation.owner_route ?? operation.owner?.route ?? null;
}

function operationMessage(operation: OperationProjection): string {
  return operation.message ?? operation.recoverable_error?.message ?? "—";
}

const TASK_ORIGIN_EYEBROWS: Record<string, string> = {
  alert_rule: "Alert rule fired",
  schedule: "Scheduled run",
  person: "Agent task",
};

const TASK_ORIGIN_ICONS: Record<string, IconName> = { alert_rule: "alert", schedule: "refresh", person: "assistant" };

/**
 * What an agent task was and who started it. A fired alert read "Agent task /
 * Run custom agent" (ACC-127); its own title names the rule and the machine.
 */
function operationHeading(operation: OperationProjection): { eyebrow: string; title: string; icon: IconName } {
  if (operation.ledger !== "agent_tasks") {
    return { eyebrow: "Workload operation", title: jobLabel(operation.type), icon: "package" };
  }
  return {
    eyebrow: TASK_ORIGIN_EYEBROWS[operation.origin ?? ""] ?? "Agent task",
    title: operation.title?.trim() || jobLabel(operation.type),
    icon: TASK_ORIGIN_ICONS[operation.origin ?? ""] ?? "assistant",
  };
}

/** Milliseconds from the projection's timestamp, which may be seconds, milliseconds or ISO text. */
function stampMs(value: string | number | null | undefined): number | null {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) return value < 100_000_000_000 ? value * 1000 : value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Date.parse(value);
    return Number.isNaN(parsed) ? null : parsed;
  }
  return null;
}

/** "14 min ago" from the operation's last recorded change, or nothing when none was recorded. */
function operationAge(operation: OperationProjection): string | null {
  const stamps = operation.timestamps;
  const at = stampMs(stamps?.updated_at) ?? stampMs(stamps?.completed_at) ?? stampMs(stamps?.started_at) ?? stampMs(stamps?.created_at);
  return at === null ? null : timeAgo(at);
}

/**
 * Whether this session may dismiss an attention item: the same role the
 * server's dismiss route requires (operator or administrator), so a viewer is
 * never shown a button the server would refuse.
 */
function canDismiss(session: Session): boolean {
  return session.user.role === "operator" || session.user.role === "administrator";
}

/** One row of the list: the operation's name, where it came from and when, and its state. */
function OperationRow({ operation, selected, onSelect }: {
  operation: OperationProjection;
  selected: boolean;
  /** Absent in the list-only feed, where a row opens nothing beside it. */
  onSelect?: () => void;
}) {
  const heading = operationHeading(operation);
  const age = operationAge(operation);
  const pill = rowPill(operation);
  if (!onSelect) {
    return (
      <div className="acty-row acty-row--static" data-operation-key={operation.operation_key}>
        <span aria-hidden="true" className="ui-row__icon"><Icon name={heading.icon} size={18} /></span>
        <span className="acty-row__text">
          <strong>{heading.title}</strong>
          <small>{age ? `${heading.eyebrow} · ${age}` : heading.eyebrow}</small>
        </span>
        <StatusPill label={pill.label} tone={pill.tone} />
      </div>
    );
  }
  return (
    <Button
      aria-pressed={selected}
      className={selected ? "acty-row acty-row--selected" : "acty-row"}
      data-operation-key={operation.operation_key}
      onClick={onSelect}
      type="button"
    >
      <span aria-hidden="true" className={selected ? "ui-row__icon ui-row__icon--accent" : "ui-row__icon"}><Icon name={heading.icon} size={18} /></span>
      <span className="acty-row__text">
        <strong>{heading.title}</strong>
        <small>{age ? `${heading.eyebrow} · ${age}` : heading.eyebrow}</small>
      </span>
      <StatusPill label={pill.label} tone={pill.tone} />
    </Button>
  );
}

/**
 * The selected operation, read-only: its name and where to act on it, then
 * the shared owner view in history mode with this record's own fields. The
 * owner view is the same one the app's page shows, so the two cannot drift.
 */
function OperationDetail({ operation, session, onDismissed }: {
  operation: OperationProjection;
  session: Session;
  onDismissed: () => void;
}) {
  const [dismissing, setDismissing] = useState(false);
  const [confirmDismiss, setConfirmDismiss] = useState(false);
  const [dismissError, setDismissError] = useState("");
  const route = ownerRoute(operation);
  const retryLineage = operation.retry_lineage ?? null;
  // #148: an ended operation has no progress to report, and a running one
  // without a percentage says so in words the owner uses.
  const progress = operation.progress.determinate && operation.progress.value !== null
    ? `${String(operation.progress.value)}%`
    : operationIsTerminal(operation.state) ? "—" : "Not reported";
  const heading = operationHeading(operation);
  // VD-139: offered only for what the server says needs attention now.
  const dismissable = operation.needs_attention === true && canDismiss(session);
  const dismiss = async () => {
    setDismissing(true);
    setDismissError("");
    try {
      await apiRequest(`/operations/${operation.operation_id}/dismiss`, { method: "POST" }, session.csrf_token);
      setConfirmDismiss(false);
      onDismissed();
    } catch (error) {
      setDismissError(error instanceof Error && error.message ? error.message : "The dismissal was not saved.");
      setConfirmDismiss(false);
    } finally {
      setDismissing(false);
    }
  };

  return (
    <Card
      actions={<>
        {dismissable && <Button className="record-ghost" onClick={() => setConfirmDismiss(true)} type="button" variant="quiet">Dismiss — I've dealt with this</Button>}
        {route ? <a className="ui-button ui-button--secondary" href={routeHref(route)}>Open</a> : <span className="ui-muted ui-small">Source unavailable</span>}
      </>}
      as="section"
      className="acty-detail"
      data-testid={`activity-operation-${operation.operation_id}`}
      // The eyebrow and the icon ride in the description so the heading's
      // name is the operation's own; the card lays them out above and beside it.
      description={<>
        <span aria-hidden="true" className="ui-row__icon ui-row__icon--accent acty-detail__icon"><Icon name={heading.icon} size={18} /></span>
        <span className="record-eyebrow acty-detail__eyebrow">{heading.eyebrow}</span>
      </>}
      heading={heading.title}
    >
      {dismissError && <Notice severity="danger"><span>Could not dismiss this: {dismissError}</span></Notice>}
      {confirmDismiss && (
        // A dismissal has no undo, so it is confirmed. It clears this item
        // only while it stays as it is now; any later change shows again.
        <RecordConfirm
          busy={dismissing}
          confirmLabel="Dismiss"
          eyebrow="No undo"
          onCancel={() => setConfirmDismiss(false)}
          onConfirm={() => void dismiss()}
          title={`Dismiss this item? ${heading.title}`}
        >
          <p>It leaves Need attention and every count of it. The record and its history stay, and there is no undo; if the operation changes again it comes back.</p>
        </RecordConfirm>
      )}
      <div
        className="acty-detail__owner"
        data-owner-message={operationMessage(operation)}
        data-owner-mode="history"
        data-owner-operation-id={operation.operation_id}
        data-owner-progress={progress}
        data-owner-retry-lineage={JSON.stringify(retryLineage)}
        data-owner-revision={String(operation.revision)}
        data-owner-state={operation.state}
        data-testid={`activity-owner-${operation.operation_id}`}
      >
        <OperationOwner mode="history" operation={operation} title={null}>
          <RecordFacts facts={[
            { key: "state", label: "Status", value: friendlyState(operation.state), mono: true, field: "state" },
            { key: "progress", label: "Progress", value: progress, field: "progress" },
            { key: "message", label: "Message", value: operationMessage(operation), field: "message" },
            ...(operation.dismissal ? [{ key: "dismissal", label: "Dismissed", value: `By ${operation.dismissal.dismissed_by || "an operator"}, ${new Date(operation.dismissal.dismissed_at * 1000).toLocaleString()}`, field: "dismissal" }] : []),
          ]} />
          <p className="ui-muted ui-small acty-detail__note">Activity is a read-only record. Correct and continue happens where the operation runs: Open takes you there.</p>
          <details className="acty-detail__technical">
            <summary>Technical details</summary>
            <RecordFacts facts={[
              { key: "operation-id", label: "Operation ID", value: <code>{operation.operation_id}</code>, mono: true, field: "operation-id" },
              { key: "source-id", label: "Source ID", value: <code>{operation.source_id}</code>, mono: true, field: "source-id" },
              { key: "revision", label: "Revision", value: String(operation.revision), mono: true, field: "revision" },
              { key: "retry-lineage", label: "Retry lineage", value: `Attempt ${retryLineage.attempt}, depth ${retryLineage.depth}${retryLineage.parent_operation_id ? `, retried from ${retryLineage.parent_operation_id}` : ""}`, mono: true, field: "retry-lineage" },
            ]} />
          </details>
        </OperationOwner>
      </div>
    </Card>
  );
}

/**
 * Activity › Operations (VD-200, the ActivityOperation board): the server's
 * four filters with their counts, the list of operations, and the one you
 * picked beside it. It polls every three seconds while the tab is visible.
 */
export function ActivityOperations({ session, detail = true }: {
  session: Session;
  /** False for the list alone (the Cluster page's feed): no selected operation beside it. */
  detail?: boolean;
}) {
  const [page, setPage] = useState<OperationPage | null>(null);
  const [summary, setSummary] = useState<OperationSummary | null>(null);
  const [operationFilter, setOperationFilter] = useState<OperationFilter>("all");
  const [loadError, setLoadError] = useState("");
  // Why the list a filter asked for could not be read, until it is read. A
  // filter whose request failed used to sit on "Loading operations..." for good.
  const [filterError, setFilterError] = useState<{ filter: OperationFilter; message: string } | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [listPage, setListPage] = useState(1);
  // The poll and a filter click can both be in flight; a response is applied
  // only while the filter it was asked for is still the one on screen.
  const filterRef = useRef<OperationFilter>("all");

  const applyOperations = useCallback((filter: OperationFilter, payload: unknown) => {
    if (filter !== filterRef.current) return;
    setPage({ filter, operations: collectOperations(payload), matched: readMatched(payload) });
    setSummary(readOperationSummary(payload));
    setFilterError(null);
    setLoadError("");
  }, []);

  // A failed read of the list on screen is said, not left as a spinner; the
  // 3-second poll keeps retrying it, and the first answer clears the notice.
  const reportFilterError = useCallback((filter: OperationFilter, error: unknown) => {
    if (filter !== filterRef.current) return;
    setFilterError({
      filter,
      message: error instanceof Error && error.message ? error.message : "The request failed.",
    });
  }, []);

  const refresh = useCallback(async () => {
    const filter = filterRef.current;
    try {
      applyOperations(filter, await apiRequest<unknown>(operationsPath(filter)));
    } catch (error) {
      setLoadError(error instanceof Error && error.message ? error.message : "Operational history could not be loaded.");
    }
  }, [applyOperations]);

  const chooseFilter = (filter: OperationFilter) => {
    filterRef.current = filter;
    setOperationFilter(filter);
    setListPage(1);
    void apiRequest<unknown>(operationsPath(filter))
      .then((payload) => applyOperations(filter, payload))
      .catch((error: unknown) => reportFilterError(filter, error));
  };

  useEffect(() => {
    void refresh();
    let polling = false;
    const pollOperations = () => {
      if (!document.hidden && !polling) {
        polling = true;
        const filter = filterRef.current;
        void apiRequest<unknown>(operationsPath(filter))
          .then((payload) => applyOperations(filter, payload))
          .catch((error: unknown) => reportFilterError(filter, error))
          .finally(() => { polling = false; });
      }
    };
    const interval = window.setInterval(pollOperations, 3000);
    const visibilityChanged = () => {
      if (!document.hidden) pollOperations();
    };
    document.addEventListener("visibilitychange", visibilityChanged);
    return () => {
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, [refresh, applyOperations, reportFilterError]);

  const current = page !== null && page.filter === operationFilter ? page : null;
  const currentFilterError = current === null && filterError?.filter === operationFilter ? filterError : null;
  const operations = current?.operations ?? [];
  const totalPages = Math.max(1, Math.ceil(operations.length / ROWS_PER_PAGE));
  const shownPage = Math.min(listPage, totalPages);
  const visible = operations.slice((shownPage - 1) * ROWS_PER_PAGE, shownPage * ROWS_PER_PAGE);
  const selected = visible.find((operation) => operation.operation_id === selectedId) ?? visible[0] ?? null;
  const windowNote = current !== null && current.matched !== null && current.matched > operations.length
    ? `Showing the newest ${operations.length} of ${current.matched}.`
    : null;

  return (
    <div className="acty-operations">
      {loadError && <Notice severity="danger"><span>{loadError}</span></Notice>}
      <div aria-label="Filter operations" className="acty-filters" role="group">
        {FILTERS.map(({ filter, label, count }) => {
          // A count the server did not report is shown as missing, never as 0.
          const value = summary ? count(summary) : null;
          return (
            <Button
              aria-label={value !== null ? `${label} ${value}` : page === null ? `${label}: loading` : `${label}: count not reported`}
              aria-pressed={operationFilter === filter}
              className={operationFilter === filter ? "acty-filter acty-filter--on" : "acty-filter"}
              key={filter}
              onClick={() => chooseFilter(filter)}
              type="button"
            >
              {label}<span className="acty-filter__count">{value ?? "—"}</span>
            </Button>
          );
        })}
      </div>
      <div className={detail ? "acty-split" : "acty-split acty-split--list"}>
        <Card
          as="section"
          className="acty-list"
          description="Agent tasks and background jobs together · newest first"
          flush
          footer={current !== null && operations.length > 0
            ? <RecordPager label="Operations" note={windowNote && <span data-testid="activity-operation-window">{windowNote}</span>} page={shownPage} setPage={setListPage} totalPages={totalPages} />
            : undefined}
          heading="Operations"
        >
          {currentFilterError && (
            <div className="acty-list__notice">
              <Notice severity="warning">
                <span>
                  {FILTER_LIST_NAMES[currentFilterError.filter]} could not be loaded ({currentFilterError.message}). Vaelor keeps retrying every few seconds.
                </span>
              </Notice>
              <p className="ui-muted ui-small">Counts that were not read show “—”, never 0.</p>
            </div>
          )}
          {current === null && !loadError && !currentFilterError && <LoadingLines label="Loading operations…" />}
          {current !== null && operations.length === 0 && <p className="acty-list__empty">No operations here.</p>}
          {visible.map((operation) => (
            <OperationRow
              key={operationRevisionKey(operation)}
              onSelect={detail ? () => setSelectedId(operation.operation_id) : undefined}
              operation={operation}
              selected={detail && selected?.operation_id === operation.operation_id}
            />
          ))}
        </Card>
        {detail && selected && <OperationDetail key={selected.operation_id} onDismissed={() => void refresh()} operation={selected} session={session} />}
      </div>
    </div>
  );
}
