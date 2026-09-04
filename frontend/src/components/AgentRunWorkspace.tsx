import type { AgentTask } from "./agentTypes";
import { leakedListItems } from "../lib/leakedList";
import { Button, Notice } from "./ui";

/**
 * The result of running an agent, shown where the run was started.
 *
 * Alpha 12 scattered one run across four places: you described it in a dialog,
 * the dialog closed, a banner told you to look "below in this agent's run
 * history", that history was a closed disclosure, and the output itself sat
 * inside a second disclosure nested in the first. Nothing about the run
 * appeared where you launched it, which is why running an agent felt like it
 * produced no outcome at all.
 *
 * This keeps one surface open from request to answer.
 */

const STAGES = ["needs_approval", "ready", "running", "completed"] as const;

const STAGE_LABEL: Record<string, string> = {
  needs_approval: "Waiting for your approval",
  ready: "Approved — waiting for the runner",
  running: "Running now",
  completed: "Finished",
};

/** Terminal states that are not "completed" get their own presentation. */
const STOPPED: Record<string, string> = {
  failed: "This run did not finish",
  cancelled: "You cancelled this run",
  blocked: "This run is blocked",
  archived: "Finished",
};

export function AgentRunProgress({ task }: { task: AgentTask }) {
  const stopped = STOPPED[task.state];
  if (stopped) {
    return (
      <p className="agent-run__stage" role="status">
        {stopped}
      </p>
    );
  }
  const reached = STAGES.indexOf(task.state as (typeof STAGES)[number]);
  return (
    <ol aria-label="Run progress" className="agent-run__stages">
      {STAGES.map((stage, index) => (
        <li
          className={index <= reached ? "is-done" : undefined}
          data-current={index === reached ? "true" : undefined}
          key={stage}
        >
          {STAGE_LABEL[stage]}
        </li>
      ))}
    </ol>
  );
}

/**
 * Free text from a run, never as source syntax.
 *
 * A run's answer arrived as `["...", '...', '...']` and was printed with its
 * brackets and quotes intact. When the text is a list, it is shown as one.
 */
function RunProse({ text }: { text: string }) {
  const items = leakedListItems(text);
  return items
    ? <ul>{items.map((item) => <li key={item}>{item}</li>)}</ul>
    : <p>{text}</p>;
}

export function AgentRunResult({ task }: { task: AgentTask }) {
  const result = task.result ?? {};
  const sources = [...(result.evidence ?? []), ...(result.sources ?? [])];
  const nothingToShow = !result.answer
    && !result.summary
    && !result.findings?.length
    && !result.recommendations?.length
    && !result.next_actions?.length
    && !result.clarifications?.length;

  if (task.state === "failed" || task.state === "blocked") {
    return (
      <Notice severity="warning">
        <span>{task.error || result.summary || "This run did not produce a result."}</span>
      </Notice>
    );
  }

  return (
    <div className="agent-run__result">
      {result.degraded && (
        <Notice severity="warning">
          <span>
            Answered with built-in read-only diagnostics — the selected AI model was not
            reachable, so this did not use the model.
          </span>
        </Notice>
      )}
      {/* #247w: when the run could not verify an answer it asks the user
          specific questions instead of guessing. Those lead the result. */}
      {result.clarifications?.length ? (
        <section className="agent-run__clarifications">
          <h4>I need a bit more to answer this</h4>
          <ul>{result.clarifications.map((item) => <li key={item}>{item}</li>)}</ul>
          <p className="agent-run__clarifications-hint">
            Re-run this task with the detail above and I will try again.
          </p>
        </section>
      ) : null}
      {/* The reply comes first. It is the thing the user asked for. */}
      {result.answer && (
        <section className="agent-run__answer">
          <h4>Answer</h4>
          <RunProse text={result.answer} />
        </section>
      )}
      {result.summary && !result.answer && (
        <section className="agent-run__answer">
          <h4>Result</h4>
          <RunProse text={result.summary} />
        </section>
      )}
      {result.findings?.length ? (
        <section><h4>What it found</h4><ul>{result.findings.map((item) => <li key={item}>{item}</li>)}</ul></section>
      ) : null}
      {result.recommendations?.length ? (
        <section><h4>Recommended</h4><ul>{result.recommendations.map((item) => <li key={item}>{item}</li>)}</ul></section>
      ) : null}
      {result.next_actions?.length ? (
        <section><h4>Next steps</h4><ul>{result.next_actions.map((item) => <li key={item}>{item}</li>)}</ul></section>
      ) : null}
      {sources.length ? (
        <section>
          <h4>Sources it used</h4>
          <ul>
            {sources.map((item, index) => (
              <li key={`${item.source}-${index}`}>
                <strong>{item.source}</strong>{item.summary && <> · {item.summary}</>}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
      {result.warnings?.length ? (
        <section className="agent-run__warnings">
          <h4>Check before relying on this</h4>
          <ul>{result.warnings.map((item) => <li key={item}>{item}</li>)}</ul>
        </section>
      ) : null}
      {nothingToShow && (
        <Notice severity="info">
          <span>This run finished without producing any output.</span>
        </Notice>
      )}
    </div>
  );
}

/**
 * The user-triggered re-run on the more capable model.
 *
 * A custom agent runs on the small NPU model by default. When that run
 * underperforms - it fails, falls back to built-in diagnostics, or produces no
 * usable answer - and a GPU model is available, the run carries
 * `escalation_available` and the user can re-run the SAME task on the graphics
 * model that powers AI Chat. It is never shown on a clean success. The action
 * changes only the model, not what the run is allowed to do.
 */
export function EscalationAction({
  task,
  busy,
  onEscalate,
}: {
  task: AgentTask;
  busy: boolean;
  onEscalate: () => void;
}) {
  const escalation = task.result?.escalation_available;
  if (!escalation) return null;
  return (
    <div className="agent-run__escalate">
      <Button disabled={busy} onClick={onEscalate} type="button" variant="primary">
        Retry on the more capable model
      </Button>
      <p className="agent-run__escalate-hint">
        Runs the same task on the graphics model (shared with AI Chat).
      </p>
    </div>
  );
}

export function AgentRunWorkspace({
  task,
  busy,
  onApprove,
  onCancel,
  onRetry,
  onClose,
  onRunAgain,
  onEscalate,
}: {
  task: AgentTask;
  busy: boolean;
  onApprove: () => void;
  onCancel: () => void;
  onRetry: () => void;
  onClose: () => void;
  onRunAgain: () => void;
  onEscalate: () => void;
}) {
  const finished = ["completed", "archived", "failed", "cancelled", "blocked"].includes(task.state);
  return (
    <div className="agent-run">
      <p className="agent-run__request"><strong>You asked:</strong> {task.description}</p>
      <AgentRunProgress task={task} />
      {finished && <AgentRunResult task={task} />}
      {finished && <EscalationAction busy={busy} onEscalate={onEscalate} task={task} />}
      <div className="dialog__actions">
        {task.state === "needs_approval" && (
          <>
            <Button disabled={busy} onClick={onCancel} type="button" variant="quiet">Cancel run</Button>
            <Button busy={busy} disabled={busy} onClick={onApprove} type="button" variant="primary">
              Approve and run
            </Button>
          </>
        )}
        {["failed", "cancelled", "blocked"].includes(task.state) && (
          <Button disabled={busy} onClick={onRetry} type="button" variant="primary">Try again</Button>
        )}
        {["completed", "archived"].includes(task.state) && (
          <Button disabled={busy} onClick={onRunAgain} type="button" variant="quiet">Ask something else</Button>
        )}
        <Button onClick={onClose} type="button" variant={finished ? "primary" : "quiet"}>
          {finished ? "Done" : "Close"}
        </Button>
      </div>
    </div>
  );
}
