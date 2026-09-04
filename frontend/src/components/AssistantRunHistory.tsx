import { timeAgo, timeUntil } from "../lib/format";
import { AgentTaskBoard, applianceProfileName, type AgentTaskBoardProps } from "./AgentTaskBoard";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button } from "./ui";
import type { AgentTask, Automation, Trigger } from "./agentTypes";

export type RunHistoryFilter = "all" | "checks" | "agents" | "automatic";

/**
 * A run is "automatic" when nobody typed it, and only the server knows that.
 *
 * The server records every scheduled run in `automation_runs`, so those task
 * ids are authoritative and they are the whole test.
 *
 * A `Scheduled: ` / `Alert: ` title prefix used to stand in for the ids. It
 * cannot: a durable appliance check is titled with the operator's own words, so
 * anyone who typed "Scheduled: check my fans tonight" had their own check
 * classified as something the appliance did by itself and lost it out of the
 * Checks filter. Both authors are covered by ids now: the scheduler records
 * runs in `automation_runs` and the trigger evaluator records alert-rule runs
 * in `automation_trigger_runs`, and `/assistant/automations` returns both, so
 * `automaticTaskIds` carries every automatic run regardless of which started
 * it. Showing a run in the wrong filter is recoverable; telling the reader
 * their own question ran on its own is not.
 */
export function isAutomaticRun(task: AgentTask, automaticTaskIds: ReadonlySet<string>): boolean {
  return automaticTaskIds.has(task.id);
}

export function isAgentRun(task: AgentTask): boolean {
  return task.profile.startsWith("custom_");
}

export function filterRuns(
  tasks: AgentTask[],
  filter: RunHistoryFilter,
  automaticTaskIds: ReadonlySet<string>,
): AgentTask[] {
  if (filter === "all") return tasks;
  if (filter === "automatic") return tasks.filter((task) => isAutomaticRun(task, automaticTaskIds));
  if (filter === "agents") {
    return tasks.filter((task) => isAgentRun(task) && !isAutomaticRun(task, automaticTaskIds));
  }
  return tasks.filter((task) => !isAgentRun(task) && !isAutomaticRun(task, automaticTaskIds));
}

const headings: Record<RunHistoryFilter, { title: string; empty: string; detail: string }> = {
  all: {
    title: "Every run",
    empty: "No runs yet",
    detail: "Appliance checks, agent runs, and automatic runs all appear here.",
  },
  checks: {
    title: "Appliance checks",
    empty: "No appliance checks yet",
    detail: "Choose a problem area on the composer to run the first appliance check.",
  },
  agents: {
    title: "Agent runs",
    empty: "No agent runs yet",
    detail: "Runs started from one of your custom agents appear here.",
  },
  automatic: {
    title: "Automatic runs",
    empty: "Nothing has run automatically yet",
    detail: "Schedules and alert rules create runs here without being asked each time.",
  },
};

interface AssistantRunHistoryProps extends Omit<
  AgentTaskBoardProps, "heading" | "eyebrow" | "controls" | "intro" | "emptyTitle" | "emptyDetail"
> {
  automations: Automation[];
  automaticTaskIds: ReadonlySet<string>;
  /** Whether this reader has the administrator-only Agents tab to be sent to. */
  canManageAgents: boolean;
  filter: RunHistoryFilter;
  onFilterChange: (filter: RunHistoryFilter) => void;
  triggers: Trigger[];
}

/**
 * One history for every kind of run this destination can produce.
 *
 * The old shape split the same evidence across three tabs by how it happened to
 * be started, so a reader looking for "what has this thing done" had to know
 * the implementation first. Schedules and alert rules are listed beside the
 * runs they produce, under Automatic, because a run with no obvious author is
 * exactly when the reader needs to see what authored it.
 */
export function AssistantRunHistory({
  automations,
  automaticTaskIds,
  canManageAgents,
  filter,
  onFilterChange,
  triggers,
  ...board
}: AssistantRunHistoryProps) {
  const counts: Record<RunHistoryFilter, number> = {
    all: board.tasks.length,
    checks: filterRuns(board.tasks, "checks", automaticTaskIds).length,
    agents: filterRuns(board.tasks, "agents", automaticTaskIds).length,
    automatic: filterRuns(board.tasks, "automatic", automaticTaskIds).length,
  };
  const labels: Record<RunHistoryFilter, string> = {
    all: "All",
    checks: "Checks",
    agents: "Agent runs",
    automatic: "Automatic",
  };
  const heading = headings[filter];
  /*
   * One heading pair and one control band.
   *
   * "EVIDENCE / What Vaelor has run here" used to sit 90px above "APPLIANCE
   * EVIDENCE / Every run", with the filters in one row and the archive controls
   * in another 90px below them: two names and two control bands for one list.
   * The filters are handed to the board so both live in its single heading row.
   */
  const filters = (
    <div aria-label="Filter run history" className="assistant-run-history__filters" role="group">
      {(Object.keys(labels) as RunHistoryFilter[]).map((item) => (
        <Button
          aria-pressed={filter === item}
          className={filter === item ? "run-filter run-filter--active" : "run-filter"}
          key={item}
          onClick={() => onFilterChange(item)}
          type="button"
          variant="quiet"
        >
          {labels[item]} ({counts[item]})
        </Button>
      ))}
    </div>
  );

  const automationIntro = filter === "automatic" ? (
    <div className="assistant-run-history__automation">
      <p>
        These are the schedules and alert rules that create the runs below. Every run they
        start is read-only, and any change it proposes still needs a separate approval.
      </p>
      <div className="assistant-run-history__automation-list">
        {automations.map((automation) => (
          <article key={automation.id}>
            <div>
              <small>Schedule · {applianceProfileName(automation.profile)}</small>
              <strong>{automation.name}</strong>
              <small>
                {automation.schedule_text}
                {automation.next_run_at ? ` · next ${timeUntil(automation.next_run_at * 1000)}` : ""}
              </small>
            </div>
            <StatusPill
              label={automation.enabled ? "enabled" : "paused"}
              status={automation.enabled ? "healthy" : "neutral"}
            />
          </article>
        ))}
        {triggers.map((trigger) => (
          <article key={trigger.id}>
            <div>
              <small>Alert rule · {trigger.source.replaceAll("_", " ")} {trigger.operator} {trigger.threshold}</small>
              <strong>{trigger.name}</strong>
              <small>
                {trigger.last_value == null ? "Waiting for first sample" : `Latest value ${trigger.last_value}`}
                {trigger.last_triggered_at ? ` · fired ${timeAgo(trigger.last_triggered_at * 1000)}` : ""}
              </small>
            </div>
            <StatusPill
              label={trigger.enabled ? "watching" : "paused"}
              status={trigger.enabled ? "healthy" : "neutral"}
            />
          </article>
        ))}
        {automations.length === 0 && triggers.length === 0 && (
          <div className="empty-state">
            <Icon name="activity" />
            <h3>Nothing is set to run on its own</h3>
            {/* Routines is administrator-only, so naming it to anyone else
                points at a tab that is not in their tablist. */}
            <p>
              {canManageAgents
                ? "Schedules and alert rules are created alongside the agent they run, under Routines."
                : "Schedules and alert rules are created alongside the agent they run. An administrator sets them up."}
            </p>
          </div>
        )}
      </div>
    </div>
  ) : null;

  return (
    <div className="assistant-run-history">
      <AgentTaskBoard
        {...board}
        controls={filters}
        emptyDetail={heading.detail}
        emptyTitle={heading.empty}
        eyebrow="Evidence"
        heading={board.taskView === "archive" ? `Archived · ${heading.title}` : heading.title}
        intro={automationIntro}
        tasks={filterRuns(board.tasks, filter, automaticTaskIds)}
      />
    </div>
  );
}
