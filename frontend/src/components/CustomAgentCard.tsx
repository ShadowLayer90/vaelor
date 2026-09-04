import { Button, Notice } from "./ui";
import { StatusPill } from "./StatusPill";
import type { AgentProfile, AgentTask } from "./agentTypes";

export type AgentRevision = { version: number; action: string; created_at: number };

/**
 * One custom agent, with two actions in the open and the rest one click deeper.
 *
 * Every card used to carry seven equally weighted buttons — six cards meant
 * forty-two — wrapping into an uneven block that read as a wall rather than as
 * a choice. Run and Edit are what an operator does day to day; App access,
 * automation, versions, archiving and deletion are administration, so they move
 * into an overflow disclosure. Nothing was removed.
 *
 * This is also where the "creating an agent does not create a schedule" notice
 * used to be repeated verbatim once per agent. The list states that once, above;
 * the card carries a "Not scheduled" chip instead.
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
  runs: AgentTask[];
  runsOpen: boolean;
  scheduleCount: number;
  triggerCount: number;
}) {
  const context = agent.scopes.length
    ? agent.scopes.map((scope) => scope.replace(":read", "")).join(" · ")
    : "No Vaelor context";
  const unattended = scheduleCount + triggerCount;
  return (
    <article className="custom-agent-card">
      <div className="custom-agent-card__body">
        <small>Custom agent · version {agent.version} · {agent.enabled ? "active" : "archived"}</small>
        <h3>{agent.name}</h3>
        <p>{agent.description}</p>
        <small>{context}</small>
        {agent.web_access?.enabled && (
          <small> · guarded web{agent.web_access.allowed_domains.length ? ` (${agent.web_access.allowed_domains.length} domains)` : " search"}</small>
        )}
        {agent.permissions?.length ? (
          <small> · {agent.permissions.map((permission) => permission.replace(":", " ")).join(" · ")}</small>
        ) : null}
        <div className="custom-agent-card__chips">
          {/*
            * The schedule state is a chip, not a paragraph. Six agents meant six
            * copies of the same twenty-three-word notice, which is a list-level
            * fact stated once per row.
            */}
          <StatusPill
            status={unattended ? "healthy" : "neutral"}
            label={unattended
              ? `${scheduleCount} schedule${scheduleCount === 1 ? "" : "s"} · ${triggerCount} trigger${triggerCount === 1 ? "" : "s"}`
              : "Not scheduled"}
          />
        </div>
        <details
          open={runsOpen}
          onToggle={(event) => onToggleRuns(event.currentTarget.open)}
        >
          <summary>Run history ({runs.length})</summary>
          {runs.length ? (
            <ul>
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
          ) : <p>No runs yet.</p>}
        </details>
        {revisions && (
          <div className="custom-agent-versions">
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
      <div className="custom-agent-card__actions">
        {agent.enabled && (
          <Button variant="primary" disabled={!modelReady || busy} onClick={onRun}>Run</Button>
        )}
        <Button variant="quiet" onClick={onEdit}>Edit</Button>
        <details className="custom-agent-card__menu">
          <summary aria-label={`More actions for ${agent.name}`} className="ui-button ui-button--quiet">More</summary>
          <div className="custom-agent-card__menu-items">
            {agent.enabled && (
              <Button variant="quiet" disabled={!modelReady || busy} onClick={onAutomate}>Add automation</Button>
            )}
            <Button variant="quiet" aria-pressed={appAccessOpen} onClick={onToggleAppAccess}>
              {appAccessOpen ? "Hide app access" : "App access"}
            </Button>
            <Button variant="quiet" disabled={busy} onClick={onToggleVersions}>
              {revisions ? "Hide versions" : "View versions"}
            </Button>
            <Button variant="quiet" onClick={onToggleEnabled}>{agent.enabled ? "Archive" : "Restore"}</Button>
            <Button variant="danger" onClick={onDelete}>Delete</Button>
          </div>
        </details>
      </div>
    </article>
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
    <li>
      <div><strong>{task.title}</strong> · {task.state}</div>
      {["ready", "running"].includes(task.state) && (
        <p className="custom-agent-run-progress" role="status">
          {task.state === "running"
            ? "Running now — this can take a minute."
            : "Approved. Waiting for the runner to pick it up."}
        </p>
      )}
      <p>{task.result.summary || task.error || task.description}</p>
      {task.approval_context && (
        <div className="agent-approval-summary">
          <strong>{reviewedVersion === agentVersion
            ? `Reviewed agent version ${reviewedVersion}`
            : `Prepared against version ${reviewedVersion}; this agent is now version ${agentVersion}`}</strong>
          <small>Granted capabilities: {task.approval_context.capabilities?.join(" · ") || "none"}</small>
          <small>API integrations: {task.approval_context.integrations?.join(" · ") || "none"}</small>
        </div>
      )}
      {["completed", "archived"].includes(task.state) && (
        <details>
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
                  <li key={`${evidence.source}-${index}`}><strong>{evidence.source}</strong>{evidence.summary && <> · {evidence.summary}</>}</li>
                ))}
              </ul>
            </section>
          ) : null}
          {task.result.warnings?.length ? <section><h4>Warnings</h4><ul>{task.result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></section> : null}
        </details>
      )}
      {task.state === "needs_approval" && (
        <div className="agent-task-card__actions">
          <Button variant="primary" disabled={busy} onClick={onApprove} type="button">Approve and run</Button>
          <Button variant="quiet" disabled={busy} onClick={onCancel} type="button">Cancel</Button>
        </div>
      )}
      {["failed", "cancelled", "blocked"].includes(task.state) && (
        <Button variant="quiet" disabled={busy} onClick={onRetry} type="button">Retry as a new run</Button>
      )}
    </li>
  );
}

/**
 * The one place the list says that creating an agent does not schedule it.
 *
 * Rendering it per card produced six identical twenty-three-word notices on one
 * screen; the fact is about the list, and the cards carry a chip instead.
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
