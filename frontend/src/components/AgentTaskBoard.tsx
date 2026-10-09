import { useState, type ReactNode } from "react";
import { apiRequest } from "../lib/api";
import { timeAgo } from "../lib/format";
import { AgentWriteReviewDialog } from "./AgentWriteReviewDialog";
import { AssistantNextStep } from "./AssistantNextStep";
import { ListPager } from "./assistantListPager";
import { Icon } from "./Icon";
import { OperationOwner } from "./OperationOwner";
import { usePagination } from "./PaginatedItems";
import { StatusPill } from "./StatusPill";
import { Button, EmptyState, Notice, Select } from "./ui";
import { useModalAction } from "../hooks/useModalAction";
import { useOperationOwner } from "../hooks/useOperationOwner";
import { leakedListItems } from "../lib/leakedList";
import type { AgentTask, AgentWriteProposal, HandoffTarget } from "./agentTypes";
import { evidenceSourceLabel } from "../lib/evidenceSourceLabels";

/** Runs per page of the Evidence card; the rest page client-side. */
const RUNS_PER_PAGE = 6;

export interface AgentTaskBoardProps {
  busy: boolean;
  csrfToken: string;
  /**
   * Names what the reader is looking at when the board is filtered. Without it
   * an "Agent runs" filter still announced "Recent appliance checks", which is
   * the wrong noun for the rows underneath it.
   */
  heading?: string;
  /**
   * Overrides the eyebrow above `heading`. A caller that already owns the
   * section this board is the whole content of must be able to name it once:
   * "EVIDENCE / What Vaelor has run here" sat 90px above "APPLIANCE EVIDENCE /
   * Every run", two heading pairs for one list.
   */
  eyebrow?: string;
  /** Extra controls for the single control band, before the archive actions. */
  controls?: ReactNode;
  /** Rendered between the control band and the Evidence card. */
  intro?: ReactNode;
  emptyTitle?: string;
  emptyDetail?: string;
  handoffSelections: Record<string, string>;
  handoffTargets: HandoffTarget[];
  onArchiveFinished: () => void;
  onDiscuss: (task: AgentTask) => void;
  onHandoff: (task: AgentTask) => void;
  onHandoffSelection: (taskId: string, username: string) => void;
  onUpdated: (notice: string) => void;
  /** Re-reads the runs; the durable run owner calls it when a run finishes. */
  onRefresh: () => void;
  onRetry: (task: AgentTask) => void;
  /** Re-run the SAME task on the more capable GPU model (shared with AI Chat). */
  onEscalate: (task: AgentTask) => void;
  onTaskViewChange: () => void;
  onTransition: (task: AgentTask, state: "ready" | "cancelled") => void;
  tasks: AgentTask[];
  taskView: "recent" | "archive";
}

const APPLIANCE_PROFILE_LABELS: Record<string, string> = {
  docker: "Docker and app troubleshooter",
  local_ai: "Local AI troubleshooter",
  remote_access: "Remote access troubleshooter",
  security: "Appliance security check",
  setup: "Setup evidence check",
  system: "System health troubleshooter",
};

function applianceCopy(value: string): string {
  return value
    .replace(/specialist review/gi, "appliance check")
    .replace(/specialist task/gi, "appliance check")
    .replace(/specialist/gi, "appliance troubleshooter");
}

/**
 * A profile id in the reader's language.
 *
 * Shared so that every surface naming a profile names it the same way. A
 * schedule card printed the raw `custom_stock` beside cards that read "System
 * health troubleshooter", which asked the reader to know the wire format to
 * follow their own automation.
 */
export function applianceProfileName(profile: string, fallbackName?: string): string {
  return APPLIANCE_PROFILE_LABELS[profile]
    || (fallbackName ? applianceCopy(fallbackName) : "")
    || profile.replaceAll("custom_", "").replaceAll("_", " ")
    || "Appliance check";
}

function applianceProfileLabel(task: AgentTask): string {
  return APPLIANCE_PROFILE_LABELS[task.profile]
    || applianceCopy(task.approval_context?.profile_name || "Appliance check");
}

const FINISHED_STATES = ["completed", "failed", "cancelled"];
const RESULT_STATES = ["completed", "archived"];

function AgentTaskOwner({ task, csrfToken, onRefresh }: { task: AgentTask; csrfToken: string; onRefresh: () => void }) {
  const owner = useOperationOwner({
    operationKey: `agent_tasks/${task.id}`,
    csrfToken,
    pollIntervalMs: 1500,
    onResourceRefresh: onRefresh,
  });
  if (!owner.operation) {
    // The owner's own words come from the server's error envelope; a poll
    // repeats every 1.5 s, so the refusal is a standing status, not an alert
    // re-announced on every read.
    return owner.error
      ? <Notice className="ah-owner-state" severity="danger" standing>Durable run owner unavailable: {owner.error}</Notice>
      : (
        <div className="as-box ah-owner-state" role="status">
          <span className="as-small as-muted">Loading the durable run owner…</span>
          <span aria-hidden="true" className="ui-skeleton" />
        </div>
      );
  }
  return (
    <OperationOwner
      className="ah-owner"
      controller={owner}
      description="Server-owned progress, actions, retry lineage, and parsed result for this durable run."
      operation={owner.operation}
      title={`${applianceCopy(task.title)} · durable run owner`}
    >
      {task.result.capability_audit?.execution_binding && (
        <dl className="as-kv ah-owner__binding">
          <div><dt>Execution binding</dt><dd className="as-mono">{task.result.capability_audit.execution_binding.mode || "active"}</dd></div>
          <div><dt>Provider</dt><dd>{task.result.capability_audit.execution_binding.provider || "Runtime selected"}</dd></div>
          <div><dt>Model</dt><dd className="as-mono">{task.result.capability_audit.execution_binding.model || "Runtime selected"}</dd></div>
        </dl>
      )}
    </OperationOwner>
  );
}

/** One labelled list in the run's result grid; nothing when the list is empty. */
function ResultList({ label, children }: { label: string; children: ReactNode[] | undefined }) {
  if (!children?.length) return null;
  return (
    <section className="ah-result">
      <h4 className="as-label">{label}</h4>
      <ul>{children}</ul>
    </section>
  );
}

/**
 * Everything the run produced, in the board's three-column grid. It was a
 * closed "View full appliance check" disclosure beside a separate "View full
 * run" toggle for the durable owner; the History board opens both with one
 * "View full run", and this is what it opens.
 */
function RunResult({ task }: { task: AgentTask }) {
  const { result } = task;
  const evidence = [...(result.evidence ?? []), ...(result.sources ?? [])];
  return (
    <div className="ah-result-grid">
      <ResultList label="Findings">{result.findings?.map((item) => <li key={item}>{item}</li>)}</ResultList>
      <ResultList label="Recommendations">{result.recommendations?.map((item) => <li key={item}>{item}</li>)}</ResultList>
      <ResultList label="Next actions">{result.next_actions?.map((item) => <li key={item}><AssistantNextStep action={item} /></li>)}</ResultList>
      {result.answer && <section className="ah-result"><h4 className="as-label">Answer</h4><p>{result.answer}</p></section>}
      <ResultList label="Evidence and sources">{evidence.map((item, index) => <li key={`${item.source}-${index}`}><span title={item.source}>{evidenceSourceLabel(item.source)}</span>{item.summary && <> · {item.summary}</>}</li>)}</ResultList>
      <ResultList label="Warnings">{result.warnings?.map((item) => <li key={item}>{item}</li>)}</ResultList>
      <ResultList label="Model notes">{result.errors?.map((item) => <li key={item}>{item}</li>)}</ResultList>
      <ResultList label="Knowledge sources">{result.knowledge_sources?.map((item, index) => <li key={`${item.document}-${index}`}>{item.document} · {item.collection} · chunk {item.chunk}</li>)}</ResultList>
    </div>
  );
}

function ProposedWrites({ task, onReview }: { task: AgentTask; onReview: (proposal: AgentWriteProposal, index: number) => void }) {
  const proposals = task.result.proposed_writes;
  if (!proposals?.length) return null;
  return (
    <section aria-label="Proposed knowledge writes" className="ah-writes">
      <span className="as-label">Proposed knowledge writes</span>
      {proposals.map((proposal, index) => (
        <div className="ah-writes__row" key={`${proposal.collection_id}-${index}`}>
          <div>
            <div className="ah-writes__name">{proposal.name}</div>
            <div className="as-small as-muted">{proposal.executed ? `Written by ${proposal.approved_by}` : "Pending operator approval · nothing written"}</div>
          </div>
          <Button disabled={Boolean(proposal.executed)} onClick={() => onReview(proposal, index)} type="button" variant={proposal.executed ? "secondary" : "primary"}>
            {proposal.executed ? "Completed" : "Review exact write"}
          </Button>
        </div>
      ))}
    </section>
  );
}

function RunRow({
  expanded,
  onReviewWrite,
  onToggle,
  props,
  task,
}: {
  expanded: boolean;
  onReviewWrite: (proposal: AgentWriteProposal, index: number) => void;
  onToggle: () => void;
  props: AgentTaskBoardProps;
  task: AgentTask;
}) {
  const { busy, handoffSelections, handoffTargets } = props;
  const primary = applianceCopy(task.result.summary || task.error || task.description);
  const asked = applianceCopy(task.description || "");
  // #247w: a run that finished but could not verify an answer travels
  // as state "completed" with outcome "needs_input". It is not healthy
  // and must not wear the green pill - it needs the owner to add detail.
  const needsInput = task.state === "completed" && task.result.outcome === "needs_input";
  const pillTone = needsInput
    ? "warning"
    : task.state === "completed" ? "success" : task.state === "failed" ? "danger" : "warning";
  // Every underscore, not only the first: `needs_approval_review` read as
  // "needs approval_review".
  const pillLabel = needsInput ? "needs input" : task.state.replaceAll("_", " ");
  const hasResult = RESULT_STATES.includes(task.state);
  const canHandOff = !["running", "completed", "archived"].includes(task.state) && handoffTargets.length > 0;
  const listed = leakedListItems(primary);
  return (
    <article className={expanded ? "ah-run ah-run--open" : "ah-run"} id={`agent-task-${task.id}`}>
      <div className="ah-run__main">
        <span className="as-small as-muted ah-run__meta">{applianceProfileLabel(task)} · run {task.id.slice(-8)} · {task.kind}</span>
        <h3 className="ah-run__title">{applianceCopy(task.title)}</h3>
        {/* A result that arrived as a Python list literal is a list, not a
            sentence with brackets and quotes in it. */}
        {listed
          ? <ul className="ah-run__summary">{listed.map((item) => <li key={item}>{item}</li>)}</ul>
          : <p className="ah-run__summary">{primary}</p>}
        {/*
          * Once a run completed, the summary replaced the request and every
          * finished card read the same generic title, so the symptom the
          * reader actually typed was gone and four runs were distinguishable
          * only by their run hash.
          */}
        {asked && asked !== primary && asked !== applianceCopy(task.title) && (
          <p className="as-small as-muted ah-run__asked"><span>You asked</span> · <span>{asked}</span></p>
        )}
        {/*
          * Every completed card carried the same boilerplate summary, so the
          * board gave no way to tell one run from another, and a run that
          * never reached the model looked identical to one that did.
          */}
        {task.result.degraded && (
          <p className="as-box ah-run__note">
            Answered with built-in read-only diagnostics — the selected AI model was not reachable.
          </p>
        )}
        {/* The first finding as a lead while the run is closed; open, the
            whole Findings list is in the grid below. */}
        {!expanded && task.result.findings?.length ? (
          <p className="ah-run__lead">{applianceCopy(task.result.findings[0])}</p>
        ) : null}
        {/*
          * One history carries appliance checks and custom-agent runs side by
          * side, so the approval summary can no longer assume the built-in
          * case: it labelled a versioned agent run as a built-in check and
          * printed "External integrations: none" over a run that had been
          * granted one.
          */}
        {task.approval_context && (
          <div className="as-box ah-run__access">
            <strong>{task.profile.startsWith("custom_") ? `Reviewed agent version ${task.approval_context.profile_version ?? task.profile_version ?? 1}` : "Reviewed built-in appliance check"}</strong>
            <small>Read-only appliance access: {task.approval_context.capabilities?.join(" · ") || "none"}</small>
            <small>External integrations: {task.approval_context.integrations?.join(" · ") || "none"}</small>
          </div>
        )}
        {expanded && hasResult && <RunResult task={task} />}
        {expanded && hasResult && <ProposedWrites onReview={onReviewWrite} task={task} />}
        {expanded && <AgentTaskOwner csrfToken={props.csrfToken} onRefresh={props.onRefresh} task={task} />}
        {canHandOff && (
          <div className="ah-run__handoff">
            {/* With one operator there is nobody to reassign to: the button
                says so beside itself, and a one-option select would only
                repeat it. */}
            {handoffTargets.length >= 2 && (
              <Select
                id={"handoff-" + task.id}
                label="Assigned operator"
                onChange={(event) => props.onHandoffSelection(task.id, event.target.value)}
                value={handoffSelections[task.id] || task.assigned_to || task.actor}
              >
                {handoffTargets.map((target) => <option key={target.username} value={target.username}>{target.username} · {target.role}</option>)}
              </Select>
            )}
            <Button
              disabled={busy}
              disabledReason={handoffTargets.length < 2 ? "Only one operator exists." : undefined}
              onClick={() => props.onHandoff(task)}
              type="button"
            >
              Reassign
            </Button>
          </div>
        )}
        <div className="ah-run__actions">
          {task.state === "needs_approval" && (
            <>
              <Button onClick={() => props.onTransition(task, "ready")} type="button" variant="primary">Approve and run</Button>
              <Button onClick={() => props.onTransition(task, "cancelled")} type="button">Cancel</Button>
            </>
          )}
          {["failed", "cancelled", "blocked"].includes(task.state) && (
            <Button disabled={busy} onClick={() => props.onRetry(task)} type="button" variant="primary">Retry as a new task</Button>
          )}
          {/* A run that underperformed on the default NPU model and has a GPU
              model available can be re-run on it. Only the model changes - the
              same approval and read-only envelope still apply. Shown only when
              escalation is available; never on a clean success. */}
          {task.result.escalation_available && (
            <Button disabled={busy} onClick={() => props.onEscalate(task)} type="button">Retry on the more capable model</Button>
          )}
          {hasResult && (
            <Button onClick={() => props.onDiscuss(task)} type="button">Discuss this result</Button>
          )}
          <Button aria-expanded={expanded} className="as-btn-ghost" onClick={onToggle} type="button" variant="quiet">
            {expanded ? "Hide full run details" : "View full run"}
          </Button>
          {task.result.escalation_available && (
            <span className="as-small as-muted">Runs the same task on the graphics model (shared with AI Chat).</span>
          )}
        </div>
        <small className="as-small as-muted">Updated {timeAgo(task.updated_at * 1000)} · no changes were made</small>
      </div>
      <StatusPill tone={pillTone} label={pillLabel} />
    </article>
  );
}

/**
 * The History tab's control band and its Evidence card: every run, one row
 * each, with the run's evidence one "View full run" deeper.
 */
export function AgentTaskBoard(props: AgentTaskBoardProps) {
  const { busy, tasks, taskView } = props;
  const [writeReview, setWriteReview] = useState<{
    task: AgentTask; proposal: AgentWriteProposal; index: number;
  } | null>(null);
  // The review's own refusal stays in the review (VD-189), never on the page under it.
  const writeAction = useModalAction();
  const [ownerTaskId, setOwnerTaskId] = useState("");
  const { page, setPage, totalPages, visible } = usePagination(tasks, RUNS_PER_PAGE);
  const canArchive = tasks.some((task) => FINISHED_STATES.includes(task.state));
  const approveWrite = async () => {
    if (!writeReview) return;
    const written = await writeAction.run(() => apiRequest(
      `/assistant/tasks/${writeReview.task.id}/writes/${writeReview.index}/approve`,
      { method: "POST", body: "{}" },
      props.csrfToken,
    ));
    if (!written) return;
    setWriteReview(null);
    props.onUpdated("Knowledge document written and recorded in the audit log.");
  };
  const heading = props.heading ?? (taskView === "archive" ? "Archived appliance checks" : "Recent appliance checks");
  const emptyTitle = props.emptyTitle ?? (taskView === "archive" ? "Archive is empty" : "No recent appliance checks");
  const emptyDetail = props.emptyDetail ?? (taskView === "archive" ? "Finished appliance checks moved off the main screen will appear here." : "Choose a problem area on the composer to run the first appliance check.");
  return (
    <>
      {/* One control band: the filters and the archive actions together. */}
      <div className="ah-toolbar">
        {props.controls}
        <div className="ah-toolbar__actions">
          <Button aria-pressed={taskView === "archive"} disabled={busy} onClick={props.onTaskViewChange} type="button">{taskView === "archive" ? "Back to recent" : "View archive"}</Button>
          {taskView === "recent" && (
            <Button
              disabled={busy}
              disabledReason={canArchive ? undefined : "No finished runs to archive."}
              onClick={props.onArchiveFinished}
              type="button"
            >
              Archive finished
            </Button>
          )}
        </div>
      </div>
      {props.intro}
      <section aria-labelledby="agent-task-title" className="card ui-card ah-evidence" id="specialist-runs">
        <header className="ui-card__header">
          <div className="ui-card__titles">
            <span className="as-label">{props.eyebrow ?? "Appliance evidence"}</span>
            <h2 id="agent-task-title">{heading}</h2>
          </div>
          <span className="as-small as-muted">{tasks.length === 1 ? "1 run" : `${tasks.length} runs`}</span>
        </header>
        {tasks.length === 0 ? (
          <EmptyState
            action={taskView === "archive"
              ? <Button disabled={busy} onClick={props.onTaskViewChange} type="button">Back to recent</Button>
              : undefined}
            icon={<Icon name={taskView === "archive" ? "restore" : "activity"} size={18} />}
            text={emptyDetail}
            title={emptyTitle}
          />
        ) : (
          <div className="ah-runs">
            {visible.map((task) => (
              <RunRow
                expanded={ownerTaskId === task.id}
                key={task.id}
                onReviewWrite={(proposal, index) => setWriteReview({ task, proposal, index })}
                onToggle={() => setOwnerTaskId((current) => current === task.id ? "" : task.id)}
                props={props}
                task={task}
              />
            ))}
            <ListPager label="Runs" page={page} setPage={setPage} totalItems={tasks.length} totalPages={totalPages} />
          </div>
        )}
      </section>
      <AgentWriteReviewDialog
        busy={writeAction.busy}
        error={writeAction.error}
        onApprove={() => void approveWrite()}
        onClose={() => { if (!writeAction.busy) { writeAction.clear(); setWriteReview(null); } }}
        proposal={writeReview?.proposal ?? null}
      />
    </>
  );
}
