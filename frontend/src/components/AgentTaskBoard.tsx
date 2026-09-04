import { useState, type ReactNode } from "react";
import { apiRequest } from "../lib/api";
import { timeAgo } from "../lib/format";
import { AgentWriteReviewDialog } from "./AgentWriteReviewDialog";
import { AssistantNextStep } from "./AssistantNextStep";
import { Icon } from "./Icon";
import { OperationOwner } from "./OperationOwner";
import { PaginatedItems } from "./PaginatedItems";
import { StatusPill } from "./StatusPill";
import { Button, Select } from "./ui";
import { useOperationOwner } from "../hooks/useOperationOwner";
import { leakedListItems } from "../lib/leakedList";
import type { AgentTask, AgentWriteProposal, HandoffTarget } from "./agentTypes";

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
  /** Extra controls for the single heading row, before the archive actions. */
  controls?: ReactNode;
  /** Rendered between the heading and the list, inside the same section. */
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

function AgentTaskOwner({ task, csrfToken, onRefresh }: { task: AgentTask; csrfToken: string; onRefresh: () => void }) {
  const owner = useOperationOwner({
    operationKey: `agent_tasks/${task.id}`,
    csrfToken,
    pollIntervalMs: 1500,
    onResourceRefresh: onRefresh,
  });
  if (!owner.operation) {
    return owner.error
      ? <p className="agent-task-owner__error" role="status">Durable run owner unavailable: {owner.error}</p>
      : <p className="agent-task-owner__loading" role="status">Loading the durable run owner…</p>;
  }
  return (
    <OperationOwner
      className="agent-task-owner"
      controller={owner}
      description="Server-owned progress, actions, retry lineage, and parsed result for this durable run."
      operation={owner.operation}
      title={`${applianceCopy(task.title)} · durable run owner`}
    >
      {task.result.capability_audit?.execution_binding && (
        <dl className="agent-task-owner__binding">
          <div><dt>Execution binding</dt><dd>{task.result.capability_audit.execution_binding.mode || "active"}</dd></div>
          <div><dt>Provider</dt><dd>{task.result.capability_audit.execution_binding.provider || "Runtime selected"}</dd></div>
          <div><dt>Model</dt><dd>{task.result.capability_audit.execution_binding.model || "Runtime selected"}</dd></div>
        </dl>
      )}
    </OperationOwner>
  );
}

export function AgentTaskBoard(props: AgentTaskBoardProps) {
  const { busy, handoffSelections, handoffTargets, tasks, taskView } = props;
  const [writeReview, setWriteReview] = useState<{
    task: AgentTask; proposal: AgentWriteProposal; index: number;
  } | null>(null);
  const [writeBusy, setWriteBusy] = useState(false);
  const [ownerTaskId, setOwnerTaskId] = useState("");
  const canArchive = tasks.some((task) => ["completed", "failed", "cancelled"].includes(task.state));
  const approveWrite = async () => {
    if (!writeReview) return;
    setWriteBusy(true);
    try {
      await apiRequest(
        `/assistant/tasks/${writeReview.task.id}/writes/${writeReview.index}/approve`,
        { method: "POST", body: "{}" },
        props.csrfToken,
      );
      setWriteReview(null);
      props.onUpdated("Knowledge document written and recorded in the audit log.");
    } finally { setWriteBusy(false); }
  };
  return (
    <section className="agent-task-board" aria-labelledby="agent-task-title" id="specialist-runs">
      <div className="section-heading">
        <div><span className="page-eyebrow">{props.eyebrow ?? "Appliance evidence"}</span><h2 id="agent-task-title">{props.heading ?? (taskView === "archive" ? "Archived appliance checks" : "Recent appliance checks")}</h2></div>
        {/* One control band. The filters used to sit in a second row 90px above
            these, so the same list had two sets of controls that never met. */}
        <div className="agent-task-board__actions">
          {props.controls}
          <Button aria-pressed={taskView === "archive"} disabled={busy} onClick={props.onTaskViewChange} type="button" variant="quiet">{taskView === "archive" ? "Back to recent" : "View archive"}</Button>
          {taskView === "recent" && <Button disabled={busy || !canArchive} onClick={props.onArchiveFinished} type="button" variant="quiet">Archive finished</Button>}
          <Button onClick={props.onRefresh} type="button" variant="quiet">Reload</Button>
        </div>
      </div>
      {props.intro}
      {tasks.length === 0 ? (
        <div className="empty-state"><Icon name="activity" /><h3>{props.emptyTitle ?? (taskView === "archive" ? "Archive is empty" : "No recent appliance checks")}</h3><p>{props.emptyDetail ?? (taskView === "archive" ? "Finished appliance checks moved off the main screen will appear here." : "Choose a problem area on the composer to run the first appliance check.")}</p></div>
      ) : (
        <div className="agent-task-list">
          <PaginatedItems items={tasks} label="Appliance checks" pageSize={6} render={(task) => {
            const primary = applianceCopy(task.result.summary || task.error || task.description);
            const asked = applianceCopy(task.description || "");
            // #247w: a run that finished but could not verify an answer travels
            // as state "completed" with outcome "needs_input". It is not healthy
            // and must not wear the green pill - it needs the owner to add detail.
            const needsInput = task.state === "completed" && task.result.outcome === "needs_input";
            const pillTone = needsInput
              ? "warning"
              : task.state === "completed" ? "success" : task.state === "failed" ? "danger" : "warning";
            const pillLabel = needsInput ? "needs input" : task.state.replaceAll("_", " ");
            return (
            <article className="agent-task-card" id={`agent-task-${task.id}`} key={task.id}>
              <div className="agent-task-card__header">
                <div><small>{applianceProfileLabel(task)} · run {task.id.slice(-8)} · {task.kind}</small><h3>{applianceCopy(task.title)}</h3></div>
                {/* Every underscore, not only the first: `needs_approval_review`
                    read as "needs approval_review". */}
                <StatusPill tone={pillTone} label={pillLabel} />
              </div>
              {/* A result that arrived as a Python list literal is a list, not
                  a sentence with brackets and quotes in it. */}
              {leakedListItems(primary)
                ? <ul className="agent-task-card__list">{leakedListItems(primary)!.map((item) => <li key={item}>{item}</li>)}</ul>
                : <p>{primary}</p>}
              {/*
                * Once a run completed, the summary replaced the request and
                * every finished card read the same generic title, so the symptom
                * the reader actually typed was gone and four runs were
                * distinguishable only by their run hash.
                */}
              {asked && asked !== primary && asked !== applianceCopy(task.title) && (
                <p className="agent-task-card__asked"><small>You asked</small>{asked}</p>
              )}
              {/*
                * Every completed card carried the same boilerplate summary, so
                * the board gave no way to tell one run from another, and a run
                * that never reached the model looked identical to one that did.
                */}
              {task.result.degraded && (
                <p className="agent-task-card__degraded">
                  Answered with built-in read-only diagnostics — the selected AI model was not reachable.
                </p>
              )}
              {task.result.findings?.length ? (
                <p className="agent-task-card__lead-finding">{applianceCopy(task.result.findings[0])}</p>
              ) : null}
              {/*
                * One history now carries appliance checks and custom-agent runs
                * side by side, so the approval summary can no longer assume the
                * built-in case: it labelled a versioned agent run as a built-in
                * check and printed "External integrations: none" over a run that
                * had been granted one.
                */}
              {task.approval_context && <div className="agent-approval-summary"><strong>{task.profile.startsWith("custom_") ? `Reviewed agent version ${task.approval_context.profile_version ?? task.profile_version ?? 1}` : "Reviewed built-in appliance check"}</strong><small>Read-only appliance access: {task.approval_context.capabilities?.join(" · ") || "none"}</small><small>External integrations: {task.approval_context.integrations?.join(" · ") || "none"}</small></div>}
              <div className="agent-task-card__actions agent-task-card__actions--owner">
                <Button aria-expanded={ownerTaskId === task.id} onClick={() => setOwnerTaskId((current) => current === task.id ? "" : task.id)} type="button" variant="quiet">
                  {ownerTaskId === task.id ? "Hide full run details" : "View full run"}
                </Button>
              </div>
              {ownerTaskId === task.id && <AgentTaskOwner csrfToken={props.csrfToken} onRefresh={props.onRefresh} task={task} />}
              {["completed", "archived"].includes(task.state) && (
                <details className="agent-task-result">
                  <summary>View full appliance check</summary>
                  {task.result.findings?.length ? <section><h4>Findings</h4><ul>{task.result.findings.map((item) => <li key={item}>{item}</li>)}</ul></section> : null}
                  {task.result.recommendations?.length ? <section><h4>Recommendations</h4><ul>{task.result.recommendations.map((item) => <li key={item}>{item}</li>)}</ul></section> : null}
                  {task.result.next_actions?.length ? <section><h4>Next actions</h4><ul>{task.result.next_actions.map((item) => <li key={item}><AssistantNextStep action={item} /></li>)}</ul></section> : null}
                  {task.result.answer && <section><h4>Answer</h4><p>{task.result.answer}</p></section>}
                  {task.result.evidence?.length || task.result.sources?.length ? <section><h4>Evidence and sources</h4><ul>{[...(task.result.evidence ?? []), ...(task.result.sources ?? [])].map((item, index) => <li key={`${item.source}-${index}`}><strong>{item.source}</strong>{item.summary && <> · {item.summary}</>}</li>)}</ul></section> : null}
                  {task.result.warnings?.length ? <section><h4>Warnings</h4><ul>{task.result.warnings.map((item) => <li key={item}>{item}</li>)}</ul></section> : null}
                  {task.result.errors?.length ? <section><h4>Model notes</h4><ul>{task.result.errors.map((item) => <li key={item}>{item}</li>)}</ul></section> : null}
                  {task.result.knowledge_sources?.length ? <section><h4>Knowledge sources</h4><ul>{task.result.knowledge_sources.map((item, index) => <li key={`${item.document}-${index}`}><strong>{item.document}</strong> · {item.collection} · chunk {item.chunk}</li>)}</ul></section> : null}
                  {task.result.proposed_writes?.length ? <section className="agent-write-proposals"><h4>Proposed knowledge writes</h4>{task.result.proposed_writes.map((proposal, index) => <article key={`${proposal.collection_id}-${index}`}><div><strong>{proposal.name}</strong><small>{proposal.executed ? `Written by ${proposal.approved_by}` : "Pending operator approval · nothing written"}</small></div><Button disabled={Boolean(proposal.executed)} onClick={() => setWriteReview({ task, proposal, index })} type="button" variant="quiet">{proposal.executed ? "Completed" : "Review exact write"}</Button></article>)}</section> : null}
                  <div className="agent-task-card__actions"><Button onClick={() => props.onDiscuss(task)} type="button" variant="quiet">Discuss this result</Button></div>
                </details>
              )}
              {task.state === "needs_approval" && <div className="agent-task-card__actions"><Button onClick={() => props.onTransition(task, "ready")} type="button" variant="primary">Approve and run</Button><Button onClick={() => props.onTransition(task, "cancelled")} type="button" variant="quiet">Cancel</Button></div>}
              {["failed", "cancelled", "blocked"].includes(task.state) && <div className="agent-task-card__actions"><Button disabled={busy} onClick={() => props.onRetry(task)} type="button" variant="primary">Retry as a new task</Button></div>}
              {/* A run that underperformed on the default NPU model and has a GPU
                  model available can be re-run on it. Only the model changes -
                  the same approval and read-only envelope still apply. Shown only
                  when escalation is available; never on a clean success. */}
              {task.result.escalation_available && (
                <div className="agent-task-card__escalate">
                  <div className="agent-task-card__actions">
                    <Button disabled={busy} onClick={() => props.onEscalate(task)} type="button" variant="primary">Retry on the more capable model</Button>
                  </div>
                  <small>Runs the same task on the graphics model (shared with AI Chat).</small>
                </div>
              )}
              {!["running", "completed", "archived"].includes(task.state) && handoffTargets.length > 0 && (
                <div className="agent-task-card__handoff">
                  <div className="agent-task-card__actions">
                    <Select
                      disabled={handoffTargets.length < 2}
                      disabledReason={handoffTargets.length < 2 ? "Only one operator exists." : undefined}
                      id={"handoff-" + task.id}
                      label="Assigned operator"
                      onChange={(event) => props.onHandoffSelection(task.id, event.target.value)}
                      value={handoffSelections[task.id] || task.assigned_to || task.actor}
                    >
              {handoffTargets.map((target) => <option key={target.username} value={target.username}>{target.username} · {target.role}</option>)}
                    </Select>
                    <Button disabled={busy || handoffTargets.length < 2} disabledReason={handoffTargets.length < 2 ? "Only one operator exists." : undefined} onClick={() => props.onHandoff(task)} type="button" variant="quiet">Reassign</Button>
                  </div>
                </div>
              )}
              <small>Updated {timeAgo(task.updated_at * 1000)} · no changes were made</small>
            </article>
            );
          }} />
        </div>
      )}
      <AgentWriteReviewDialog
        busy={writeBusy}
        onApprove={() => void approveWrite().catch((error) => props.onUpdated(error instanceof Error ? error.message : "The knowledge write could not be approved."))}
        onClose={() => !writeBusy && setWriteReview(null)}
        proposal={writeReview?.proposal ?? null}
      />
    </section>
  );
}
