import { useCallback, useEffect, useMemo, useState } from "react";
import { apiRequest } from "../lib/api";
import { auditMatches } from "../lib/auditSearch";
import { auditActionLabel } from "../lib/auditLabels";
import { jobLabel } from "../lib/jobPresentation";
import {
  isOperationProjection,
  operationIsTerminal,
  operationRevisionKey,
  type OperationProjection,
} from "../lib/operationOwner";
import type { AuditEvent, Session } from "../types";
import { Icon } from "./Icon";
import { OperationOwner } from "./OperationOwner";
import { PaginatedItems, PaginationControls, usePagination } from "./PaginatedItems";
import { RecoveryPointList } from "./RecoveryPointList";
import { StatusPill } from "./StatusPill";
import { Button, Input, Notice } from "./ui";
import { destinations } from "../lib/destinations";

type OperationFilter = "all" | "active" | "attention" | "completed";

/*
 * Job names come from `jobPresentation`, the one table every surface reads.
 * This screen used to keep its own copy, so the same job appeared as "Create
 * checkpoint" here and "Back up app configuration" one screen away.
 */
function friendlyState(state: string): string {
  // `rejected` said "Failed", which put rows labelled Failed under the
  // "finished" tile and made the tiles look like they overlapped. A rejection
  // is a decision, not a failure.
  const known: Record<string, string> = { draft: "Draft", queued: "Queued", running: "Running", waiting: "Waiting", paused: "Paused", ready: "Ready", needs_approval: "Waiting for approval", completed: "Finished", healthy: "Finished", failed: "Failed", rejected: "Rejected", cancelled: "Cancelled", superseded: "Superseded", blocked: "Needs attention", interrupted: "Needs attention" };
  return known[state] ?? state.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function operationNeedsAttention(operation: OperationProjection): boolean {
  return ["failed", "blocked", "interrupted", "needs_attention"].includes(operation.state);
}

type OperationBucket = "active" | "attention" | "completed";

/**
 * Sort one operation into exactly one of the three summary tiles.
 *
 * The tiles are read as a partition of "all operations", and they were not
 * one. A failure that had already been retried was subtracted from "need
 * attention" because a later attempt names it in its lineage, and was still
 * excluded from "finished" because it is a failure - so it was counted in the
 * total and nowhere else. A `draft` operation fell through the same gap from
 * the other side: neither an active state nor a terminal one. That is how
 * 50 all / 1 in progress / 14 need attention / 32 finished added up to 47.
 *
 * Assigning a bucket here, once, is what makes the arithmetic hold: a
 * superseded failure is finished, because the attempt that replaced it is the
 * one still open, and anything not terminal is in progress whether or not this
 * client recognises its state.
 */
function operationBucket(
  operation: OperationProjection,
  supersededOperationIds: ReadonlySet<string>,
): OperationBucket {
  if (operationNeedsAttention(operation) && !supersededOperationIds.has(operation.operation_id)) {
    return "attention";
  }
  return operationIsTerminal(operation.state) ? "completed" : "active";
}

export function bucketOperations(operations: readonly OperationProjection[]): Record<OperationBucket, OperationProjection[]> {
  const supersededOperationIds = new Set(operations.flatMap((operation) => [
    operation.retry_lineage.parent_operation_id,
    ...operation.retry_lineage.ancestry,
  ].filter((item): item is string => Boolean(item))));
  const buckets: Record<OperationBucket, OperationProjection[]> = { active: [], attention: [], completed: [] };
  for (const operation of operations) {
    buckets[operationBucket(operation, supersededOperationIds)].push(operation);
  }
  return buckets;
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

function renderOwner(operation: OperationProjection) {
  return <OperationOwner operation={operation} mode="history" />;
}

function OperationHistoryCard({ operation }: { operation: OperationProjection }) {
  const route = ownerRoute(operation);
  const retryLineage = operation.retry_lineage ?? null;
  // #148: this rendered the raw word "indeterminate" beside Status, so a
  // Failed card also carried a progress reading that contradicted it. An
  // ended operation has no progress to report, and a running one without a
  // percentage says so in words the owner uses.
  const progress = operation.progress.determinate && operation.progress.value !== null
    ? `${String(operation.progress.value)}%`
    : operationIsTerminal(operation.state) ? "—" : "Not reported";

  return (
    <article className="activity-operation" data-operation-key={operation.operation_key}>
      <div className="activity-operation__projection" data-testid={`activity-operation-${operation.operation_id}`}>
        <div className="activity-operation__heading">
          <div>
            <span className="page-eyebrow">{operation.ledger === "agent_tasks" ? "Agent task" : "Workload operation"}</span>
            <h3>{jobLabel(operation.type)}</h3>
          </div>
          {route ? <a className="ui-button ui-button--quiet" href={route}>Open</a> : <span className="event-state">Source unavailable</span>}
        </div>
        <dl className="activity-operation__fields">
          <div><dt>Status</dt><dd data-field="state">{friendlyState(operation.state)}</dd></div>
          <div><dt>Progress</dt><dd data-field="progress">{progress}</dd></div>
          <div><dt>Message</dt><dd data-field="message">{operationMessage(operation)}</dd></div>
        </dl>
        <details className="activity-operation__technical">
          <summary>Technical details</summary>
          <dl>
            <div><dt>Operation ID</dt><dd data-field="operation-id"><code>{operation.operation_id}</code></dd></div>
            <div><dt>Source ID</dt><dd data-field="source-id"><code>{operation.source_id}</code></dd></div>
            <div><dt>Revision</dt><dd data-field="revision">{String(operation.revision)}</dd></div>
            <div><dt>Retry lineage</dt><dd data-field="retry-lineage">Attempt {retryLineage.attempt}, depth {retryLineage.depth}{retryLineage.parent_operation_id ? `, retried from ${retryLineage.parent_operation_id}` : ""}</dd></div>
          </dl>
        </details>
      </div>
      <div
        className="activity-operation__owner"
        data-owner-message={operationMessage(operation)}
        data-owner-mode="history"
        data-owner-operation-id={operation.operation_id}
        data-owner-progress={progress}
        data-owner-retry-lineage={JSON.stringify(retryLineage)}
        data-owner-revision={String(operation.revision)}
        data-owner-state={operation.state}
        data-testid={`activity-owner-${operation.operation_id}`}
      >
        {renderOwner(operation)}
      </div>
    </article>
  );
}

export function ActivityCenter({ session }: { session: Session }) {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [operations, setOperations] = useState<OperationProjection[]>([]);
  const [query, setQuery] = useState("");
  const [operationFilter, setOperationFilter] = useState<OperationFilter>("all");
  const [loadError, setLoadError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [nextEvents, nextOperations] = await Promise.all([
        apiRequest<AuditEvent[]>("/audit?limit=200"),
        apiRequest<unknown>("/operations?limit=50"),
      ]);
      setEvents(nextEvents);
      setOperations(collectOperations(nextOperations));
      setLoadError("");
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Operational history could not be loaded.");
    }
  }, []);

  useEffect(() => {
    void refresh();
    let polling = false;
    const pollOperations = () => {
      if (!document.hidden && !polling) {
        polling = true;
        void apiRequest<unknown>("/operations?limit=50")
          .then((payload) => setOperations(collectOperations(payload)))
          .catch(() => undefined)
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
  }, [refresh]);

  const filtered = useMemo(() => events.filter((event) => auditMatches(event, query)), [events, query]);
  const { active: activeOperations, attention: attentionOperations, completed: completedOperations } = bucketOperations(operations);
  const visibleOperations = operationFilter === "active" ? activeOperations : operationFilter === "attention" ? attentionOperations : operationFilter === "completed" ? completedOperations : operations;
  const auditPage = usePagination(filtered, 12);

  return (
    <div className="activity-page">
      <div className="page-heading">
        <div>
          <h1>{destinations.activity.name}</h1>
          <p>See who changed what, and jump to where each operation ran.</p>
        </div>
        <StatusPill label="Auditing active" status="healthy" />
      </div>
      {loadError && <Notice severity="danger"><span>{loadError}</span></Notice>}
      <section className="activity-summary">
        <Button aria-label={`${operations.length} all operations`} aria-pressed={operationFilter === "all"} onClick={() => setOperationFilter("all")}><Icon name="activity" /><strong>{operations.length}</strong><small>all operations</small></Button>
        <Button aria-label={`${activeOperations.length} in progress`} aria-pressed={operationFilter === "active"} onClick={() => setOperationFilter("active")}><Icon name="bolt" /><strong>{activeOperations.length}</strong><small>in progress</small></Button>
        <Button aria-label={`${attentionOperations.length} need attention`} aria-pressed={operationFilter === "attention"} onClick={() => setOperationFilter("attention")}><Icon name="shield" /><strong>{attentionOperations.length}</strong><small>need attention</small></Button>
        <Button aria-label={`${completedOperations.length} finished`} aria-pressed={operationFilter === "completed"} onClick={() => setOperationFilter("completed")}><Icon name="database" /><strong>{completedOperations.length}</strong><small>finished</small></Button>
      </section>
      <section className="data-panel admin-wide" aria-labelledby="activity-operations-heading">
        <div className="panel-heading"><div><h2 id="activity-operations-heading">Operations</h2><p>A read-only record of every operation. Agent tasks and background jobs appear here together.</p></div><Icon name="database" /></div>
        <div className="activity-operation-list">
          <PaginatedItems items={visibleOperations} label="Operations" pageSize={6} render={(operation) => <OperationHistoryCard key={operationRevisionKey(operation)} operation={operation} />} />
        </div>
      </section>
      <section className="data-panel admin-wide">
        <div className="panel-heading"><div><h2>Recovery checkpoints</h2><p>Inspect, verify, restore, or remove configuration restore points for managed apps.</p></div><Icon name="shield" /></div>
        <RecoveryPointList session={session} />
      </section>
      <section className="data-panel admin-wide">
        <div className="panel-heading"><div><h2>Security audit trail</h2><p>Authenticated changes and access events.</p></div><div className="activity-search"><Input className="activity-search__input" label={<span className="sr-only">Search audit events</span>} onChange={(event) => { setQuery(event.target.value); auditPage.setPage(1); }} placeholder="Search action, user, or result" type="search" value={query} /></div></div>
        <div className="activity-table-wrap"><table className="activity-table"><thead><tr><th>Action</th><th>User</th><th>Target</th><th>Time</th><th>Result</th></tr></thead><tbody>{auditPage.visible.map((event) => <tr key={event.id}><td title={event.action}>{auditActionLabel(event.action)}</td><td>{event.actor}</td><td>{event.target || "—"}</td><td>{new Date(event.created_at * 1000).toLocaleString()}</td><td><span className={`event-state event-state--${event.result}`}>{event.result}</span></td></tr>)}</tbody></table></div>
        <PaginationControls label="Audit events" page={auditPage.page} setPage={auditPage.setPage} totalItems={filtered.length} totalPages={auditPage.totalPages} />
      </section>
    </div>
  );
}
