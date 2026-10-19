import type { AgentTask } from "./agentTypes";
import { leakedListItems } from "../lib/leakedList";
import { Button, Notice } from "./ui";
import { evidenceSourceLabel } from "../lib/evidenceSourceLabels";
import { AiChatUnavailableNotice, aiChatUnavailableSentence } from "./customAgentModel";

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

/**
 * A run's state in the owner's words, for a one-line row. The same words the
 * progress view uses, so a run reads the same wherever it is listed; the raw
 * state (`needs_approval`, `ready`) never reaches the screen (ACC-140).
 */
export function agentTaskStateLabel(state: string): string {
  if (state === "triage") return "Being prepared";
  return STOPPED[state] ?? STAGE_LABEL[state] ?? "Status unknown";
}

/**
 * A run that finished without a verified answer travels as state "completed"
 * with outcome "needs_input" (#247w). It asks the owner for detail, so it is
 * never drawn as a green "Finished" (VD-200 assist review).
 */
export function agentTaskNeedsInput(task: AgentTask): boolean {
  return task.state === "completed" && task.result?.outcome === "needs_input";
}

export const NEEDS_INPUT_LABEL = "Finished · needs your input";

/** The tone of a run that has stopped moving: green only for one that finished. */
const ENDED_TONE: Record<string, string> = {
  completed: "success",
  archived: "success",
  failed: "danger",
  blocked: "danger",
  cancelled: "neutral",
};

/**
 * Where a run is: a four-step strip while it moves (the step it is on carries
 * the accent), and one line in words once it has stopped.
 */
export function AgentRunProgress({ task }: { task: AgentTask }) {
  const needsInput = agentTaskNeedsInput(task);
  const ended = needsInput ? NEEDS_INPUT_LABEL : STOPPED[task.state] ?? (task.state === "completed" ? STAGE_LABEL.completed : undefined);
  if (ended) {
    return (
      <p className={`ar-run-state ar-run-state--${needsInput ? "warning" : ENDED_TONE[task.state] ?? "neutral"}`} role="status">
        {ended}
      </p>
    );
  }
  const reached = STAGES.indexOf(task.state as (typeof STAGES)[number]);
  return (
    <ol aria-label="Run progress" className="ar-stages">
      {STAGES.map((stage, index) => (
        <li
          className={index < reached ? "is-done" : index === reached ? "is-current" : undefined}
          data-current={index === reached ? "true" : undefined}
          key={stage}
        >
          <span aria-hidden="true" className="ar-stages__bar" />
          <span>{STAGE_LABEL[stage]}</span>
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
    ? <ul className="ar-list">{items.map((item) => <li key={item}>{item}</li>)}</ul>
    : <p className="ar-run-answer">{text}</p>;
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
    // VD-220: AI Chat's model could not take the run - said plainly, with the way there.
    const unavailable = aiChatUnavailableSentence(task.error);
    if (unavailable) return <AiChatUnavailableNotice sentence={unavailable} />;
    return (
      <Notice severity="danger">
        <span>{task.error || result.summary || "This run did not produce a result."}</span>
      </Notice>
    );
  }

  return (
    <div className="ar-run-result">
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
        <section>
          <h4 className="as-label">I need a bit more to answer this</h4>
          <ul className="ar-list">{result.clarifications.map((item) => <li key={item}>{item}</li>)}</ul>
          <p className="ar-meta">
            Re-run this task with the detail above and I will try again.
          </p>
        </section>
      ) : null}
      {/* The reply comes first. It is the thing the user asked for. */}
      {result.answer && (
        <section>
          <h4 className="as-label">Answer</h4>
          <RunProse text={result.answer} />
        </section>
      )}
      {result.summary && !result.answer && (
        <section>
          <h4 className="as-label">Result</h4>
          <RunProse text={result.summary} />
        </section>
      )}
      {result.findings?.length ? (
        <section><h4 className="as-label">What it found</h4><ul className="ar-list">{result.findings.map((item) => <li key={item}>{item}</li>)}</ul></section>
      ) : null}
      {result.recommendations?.length ? (
        <section><h4 className="as-label">Recommended</h4><ul className="ar-list">{result.recommendations.map((item) => <li key={item}>{item}</li>)}</ul></section>
      ) : null}
      {result.next_actions?.length ? (
        <section><h4 className="as-label">Next steps</h4><ul className="ar-list">{result.next_actions.map((item) => <li key={item}>{item}</li>)}</ul></section>
      ) : null}
      {sources.length ? (
        <section>
          <h4 className="as-label">Sources it used</h4>
          <ul className="ar-list">
            {sources.map((item, index) => (
              <li key={`${item.source}-${index}`}>
                <strong title={item.source}>{evidenceSourceLabel(item.source)}</strong>{item.summary && <> · {item.summary}</>}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
      {result.warnings?.length ? (
        <section>
          <h4 className="as-label">Check before relying on this</h4>
          <ul className="ar-list">{result.warnings.map((item) => <li key={item}>{item}</li>)}</ul>
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

/*
 * This surface is the custom-agent run dialog's only. A custom agent runs on
 * AI Chat's model (VD-220), so it carries no "more capable model" re-run: an
 * old run that still has `escalation_available` is shown without it. Built-in
 * appliance checks keep theirs on the History board (AgentTaskBoard).
 */
export function AgentRunWorkspace({
  task,
  busy,
  onApprove,
  onCancel,
  onRetry,
  onClose,
  onRunAgain,
}: {
  task: AgentTask;
  busy: boolean;
  onApprove: () => void;
  onCancel: () => void;
  onRetry: () => void;
  onClose: () => void;
  onRunAgain: () => void;
}) {
  const finished = ["completed", "archived", "failed", "cancelled", "blocked"].includes(task.state);
  // Stopped short of an answer: failed, cancelled or blocked (STOPPED, less a finished archive).
  const stopped = task.state !== "archived" && task.state in STOPPED;
  return (
    <>
      <div className="as-dialog__body ar-run-body">
        <p className="ar-run-request"><strong>You asked:</strong> {task.description}</p>
        <AgentRunProgress task={task} />
        {finished && <AgentRunResult task={task} />}
      </div>
      <div className="as-dialog__foot">
        {task.state === "needs_approval" && (
          <>
            <Button disabled={busy} onClick={onCancel}>Cancel run</Button>
            <Button busy={busy} disabled={busy} onClick={onApprove} variant="primary">
              Approve and run
            </Button>
          </>
        )}
        {stopped && (
          <Button disabled={busy} onClick={onRetry} variant="primary">Try again</Button>
        )}
        {["completed", "archived"].includes(task.state) && (
          <Button disabled={busy} onClick={onRunAgain}>Ask something else</Button>
        )}
        {finished
          ? <Button onClick={onClose} variant={stopped ? "secondary" : "primary"}>Done</Button>
          : <Button className="as-btn-ghost" onClick={onClose} variant="quiet">Close</Button>}
      </div>
    </>
  );
}
