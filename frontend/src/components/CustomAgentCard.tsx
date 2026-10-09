import { useEffect, useRef, useState } from "react";
import { Button, Notice, type StatusTone } from "./ui";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import type { AgentProfile, AgentTask } from "./agentTypes";
import { agentTaskNeedsInput, agentTaskStateLabel, NEEDS_INPUT_LABEL } from "./AgentRunWorkspace";
import { agentGrantsLine } from "./customAgentDraft";
import { evidenceSourceLabel } from "../lib/evidenceSourceLabels";

export type AgentRevision = { version: number; action: string; created_at: number };

/** A run's state as a pill: amber while it waits on you, blue while it works, green only once it finished. */
const RUN_TONE: Record<string, StatusTone> = {
  needs_approval: "warning",
  ready: "info",
  running: "info",
  completed: "success",
  failed: "danger",
  blocked: "danger",
};

/**
 * One custom agent: a row in the "Your agents" card, with two actions in the
 * open and the rest one click deeper.
 *
 * Every card used to carry seven equally weighted buttons — six cards meant
 * forty-two — wrapping into an uneven block that read as a wall rather than as
 * a choice. Run and Edit are what an operator does day to day; App access,
 * automation, versions, archiving and deletion are administration, so they sit
 * in the More menu. Nothing was removed.
 *
 * This is also where the "creating an agent does not create a schedule" notice
 * used to be repeated verbatim once per agent. The list states that once, above;
 * the row carries a "Not scheduled" pill instead.
 */
export function CustomAgentCard({
  agent,
  appAccessOpen,
  busy,
  modelReady,
  onApprove,
  onAutomate,
  onCancelRun,
  onDelete,
  onEdit,
  onRetry,
  onRun,
  onToggleAppAccess,
  onToggleEnabled,
  onToggleRuns,
  onToggleVersions,
  revisions,
  runBlockedReason,
  runtimeManaged = true,
  runs,
  runsOpen,
  scheduleCount,
  triggerCount,
}: {
  agent: AgentProfile;
  appAccessOpen: boolean;
  busy: boolean;
  modelReady: boolean;
  onApprove: (task: AgentTask) => void;
  onAutomate: () => void;
  onCancelRun: (task: AgentTask) => void;
  onDelete: () => void;
  onEdit: () => void;
  onRetry: (task: AgentTask) => void;
  onRun: () => void;
  onToggleAppAccess: () => void;
  onToggleEnabled: () => void;
  onToggleRuns: (open: boolean) => void;
  onToggleVersions: () => void;
  revisions?: AgentRevision[];
  /** Why Run is off when the model cannot take it, said beside the button. */
  runBlockedReason?: string;
  /** Whether this row's agent runs in the Assistant runtime. False for a
      read-only inference agent, whose Run, automation and app-access actions do
      not apply, so they are omitted. */
  runtimeManaged?: boolean;
  runs: AgentTask[];
  runsOpen: boolean;
  scheduleCount: number;
  triggerCount: number;
}) {
  const unattended = scheduleCount + triggerCount;
  return (
    <article className={appAccessOpen ? "ar-agent is-selected" : "ar-agent"}>
      <span aria-hidden="true" className={appAccessOpen ? "ar-icon ar-icon--accent" : "ar-icon"}><Icon name="assistant" size={18} /></span>
      <div className="ar-agent__body">
        <span className="ar-meta">Custom agent · version {agent.version} · {agent.enabled ? "active" : "archived"}</span>
        <h3 className="ar-agent__name">{agent.name}</h3>
        <p className="ar-agent__purpose">{agent.description}</p>
        <span className="ar-meta">{agentGrantsLine(agent)}</span>
        {/*
          * The schedule state is a pill, not a paragraph. Six agents meant six
          * copies of the same twenty-three-word notice, which is a list-level
          * fact stated once per row. It counts only schedules and alert rules
          * that will still act on their own: a paused or finished one is not
          * "scheduled" (ACC-140).
          */}
        {runtimeManaged && (
          <span className="ar-pills">
            <StatusPill
              status={unattended ? "healthy" : "neutral"}
              label={unattended
                ? `${scheduleCount} active schedule${scheduleCount === 1 ? "" : "s"} · ${triggerCount} active alert rule${triggerCount === 1 ? "" : "s"}`
                : "Not scheduled"}
            />
          </span>
        )}
        {/* An inference agent is served over the cluster and never has an
            Assistant run, so a run history would only ever read "(0)" (ACC-079). */}
        {runtimeManaged && (
          <details className="ar-runs" onToggle={(event) => onToggleRuns(event.currentTarget.open)} open={runsOpen}>
            <summary className="ar-disc ar-disc--compact">
              <span>Run history ({runs.length})</span>
              <Icon className="ar-disc__chevron" name="chevron" size={16} />
            </summary>
            {runs.length ? (
              <ul className="ar-run-list">
                {runs.slice(0, 5).map((task) => (
                  <CustomAgentRunRow
                    agentVersion={agent.version}
                    busy={busy}
                    key={task.id}
                    onApprove={() => onApprove(task)}
                    onCancel={() => onCancelRun(task)}
                    onRetry={() => onRetry(task)}
                    task={task}
                  />
                ))}
              </ul>
            ) : <p className="ar-meta">No runs yet.</p>}
          </details>
        )}
        {revisions && (
          <div className="as-box ar-versions">
            <strong>Version history</strong>
            <ul>
              {revisions.map((revision) => (
                <li key={revision.version}>
                  Version {revision.version} · {revision.action} · {new Date(revision.created_at * 1000).toLocaleString()}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <div className="ar-agent__actions">
        <div className="ar-actions">
          {agent.enabled && runtimeManaged && (
            <Button disabled={busy} disabledReason={modelReady ? undefined : runBlockedReason ?? "The Assistant model is not answering."} onClick={onRun} variant="primary">Run</Button>
          )}
          <Button onClick={onEdit}>Edit</Button>
          <MoreMenu
            agent={agent}
            appAccessOpen={appAccessOpen}
            busy={busy}
            modelReady={modelReady}
            onAutomate={onAutomate}
            onDelete={onDelete}
            onToggleAppAccess={onToggleAppAccess}
            onToggleEnabled={onToggleEnabled}
            onToggleVersions={onToggleVersions}
            revisionsOpen={Boolean(revisions)}
            runtimeManaged={runtimeManaged}
          />
        </div>
        {!agent.enabled && <span className="ar-meta">Archived agents don't run. Restore it from More.</span>}
      </div>
    </article>
  );
}

/**
 * The row's More popover: administration one click deeper. It closes on a
 * choice, on Escape (returning focus to More) and on a click elsewhere.
 */
function MoreMenu({ agent, appAccessOpen, busy, modelReady, onAutomate, onDelete, onToggleAppAccess, onToggleEnabled, onToggleVersions, revisionsOpen, runtimeManaged }: {
  agent: AgentProfile;
  appAccessOpen: boolean;
  busy: boolean;
  modelReady: boolean;
  onAutomate: () => void;
  onDelete: () => void;
  onToggleAppAccess: () => void;
  onToggleEnabled: () => void;
  onToggleVersions: () => void;
  revisionsOpen: boolean;
  runtimeManaged: boolean;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const outside = (event: MouseEvent) => {
      if (root.current && !root.current.contains(event.target as Node)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") { setOpen(false); trigger.current?.focus(); }
    };
    document.addEventListener("mousedown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("mousedown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);
  // The chosen item unmounts with the menu, so focus goes to More first: a
  // dialog the action opens then returns focus there when it closes, not to
  // the page body (VD-200 assist review).
  const choose = (action: () => void) => () => { trigger.current?.focus(); setOpen(false); action(); };
  const menuId = `ar-more-${agent.id}`;
  return (
    <div className="as-popover" ref={root}>
      <Button
        aria-controls={open ? menuId : undefined}
        aria-expanded={open}
        aria-label={`More actions for ${agent.name}`}
        className={open ? "as-btn-on" : "as-btn-ghost"}
        onClick={() => setOpen((current) => !current)}
        ref={trigger}
        variant="quiet"
      >
        More
      </Button>
      {open && (
        <div aria-label={`More actions for ${agent.name}`} className="as-menu ar-menu" id={menuId} role="group">
          {agent.enabled && runtimeManaged && (
            <Button disabled={!modelReady || busy} onClick={choose(onAutomate)} variant="quiet">Add automation</Button>
          )}
          {runtimeManaged && (
            <Button aria-pressed={appAccessOpen} className={appAccessOpen ? "as-btn-on" : undefined} onClick={choose(onToggleAppAccess)} variant="quiet">
              {appAccessOpen ? "Hide app access" : "App access"}
            </Button>
          )}
          <Button disabled={busy} onClick={choose(onToggleVersions)} variant="quiet">
            {revisionsOpen ? "Hide versions" : "View versions"}
          </Button>
          <Button onClick={choose(onToggleEnabled)} variant="quiet">{agent.enabled ? "Archive" : "Restore"}</Button>
          <hr />
          <Button className="as-menu__danger" onClick={choose(onDelete)} variant="quiet">Delete</Button>
        </div>
      )}
    </div>
  );
}

function CustomAgentRunRow({
  agentVersion,
  busy,
  onApprove,
  onCancel,
  onRetry,
  task,
}: {
  agentVersion: number | undefined;
  busy: boolean;
  onApprove: () => void;
  onCancel: () => void;
  onRetry: () => void;
  task: AgentTask;
}) {
  const reviewedVersion = task.approval_context?.profile_version ?? task.profile_version;
  return (
    <li className="ar-run">
      <div className="ar-run__head">
        <strong>{task.title}</strong>
        {agentTaskNeedsInput(task)
          ? <StatusPill label={NEEDS_INPUT_LABEL} tone="warning" />
          : <StatusPill label={agentTaskStateLabel(task.state)} tone={RUN_TONE[task.state] ?? "neutral"} />}
      </div>
      {["ready", "running"].includes(task.state) && (
        <p className="ar-meta" role="status">
          {task.state === "running"
            ? "It is working on it — this can take a minute."
            : "Approved. Waiting for the runner to pick it up."}
        </p>
      )}
      {(task.result.summary || task.error || task.description) && <p className="ar-run__summary">{task.result.summary || task.error || task.description}</p>}
      {task.approval_context && (
        <p className="ar-meta">
          {reviewedVersion === agentVersion
            ? `Reviewed agent version ${reviewedVersion}`
            : `Prepared against version ${reviewedVersion}; this agent is now version ${agentVersion}`}
          {" · "}Granted capabilities: {task.approval_context.capabilities?.join(" · ") || "none"}
          {" · "}API integrations: {task.approval_context.integrations?.join(" · ") || "none"}
        </p>
      )}
      {["completed", "archived"].includes(task.state) && (
        <details className="ar-output">
          <summary>View full custom-agent output</summary>
          {task.result.answer && <section><h4>Answer</h4><p>{task.result.answer}</p></section>}
          {task.result.findings?.length ? <section><h4>Findings</h4><ul>{task.result.findings.map((item) => <li key={item}>{item}</li>)}</ul></section> : null}
          {task.result.recommendations?.length ? <section><h4>Recommendations</h4><ul>{task.result.recommendations.map((item) => <li key={item}>{item}</li>)}</ul></section> : null}
          {task.result.next_actions?.length ? <section><h4>Next actions</h4><ul>{task.result.next_actions.map((item) => <li key={item}>{item}</li>)}</ul></section> : null}
          {task.result.evidence?.length || task.result.sources?.length ? (
            <section>
              <h4>Evidence and sources</h4>
              <ul>
                {[...(task.result.evidence ?? []), ...(task.result.sources ?? [])].map((evidence, index) => (
                  <li key={`${evidence.source}-${index}`}><strong title={evidence.source}>{evidenceSourceLabel(evidence.source)}</strong>{evidence.summary && <> · {evidence.summary}</>}</li>
                ))}
              </ul>
            </section>
          ) : null}
          {task.result.warnings?.length ? <section><h4>Warnings</h4><ul>{task.result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></section> : null}
        </details>
      )}
      {task.state === "needs_approval" && (
        <div className="ar-actions">
          <Button disabled={busy} onClick={onApprove} variant="primary">Approve and run</Button>
          <Button disabled={busy} onClick={onCancel}>Cancel</Button>
        </div>
      )}
      {["failed", "cancelled", "blocked"].includes(task.state) && (
        <div className="ar-actions">
          <Button disabled={busy} onClick={onRetry}>Retry as a new run</Button>
        </div>
      )}
    </li>
  );
}

/**
 * The one place the list says that creating an agent does not schedule it.
 *
 * Rendering it per row produced six identical twenty-three-word notices on one
 * screen; the fact is about the list, and the rows carry a pill instead.
 */
export function UnscheduledAgentsNotice({ agents }: { agents: AgentProfile[] }) {
  if (agents.length === 0) return null;
  return (
    <Notice severity="info">
      <span>
        <strong>{agents.length === 1
          ? `${agents[0].name} describes recurring work but has no schedule.`
          : `${agents.length} agents describe recurring work but have no schedule.`}</strong>{" "}
        Creating an agent does not create one. Open <strong>More · Add automation</strong> on the
        agent to choose when it runs.
      </span>
    </Notice>
  );
}
