import { automationPill, operatorLabel, showsNextRun, signalLabel } from "../lib/automationStatus";
import { timeAgo, timeUntil } from "../lib/format";
import { AgentTaskBoard, applianceProfileName, type AgentTaskBoardProps } from "./AgentTaskBoard";
import type { ReactNode } from "react";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { EmptyState, SegmentedControl } from "./ui";
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
    // The "Automatic runs" card above names the schedules and rules; this
    // card holds what they started, so it is not given the same name twice.
    title: "Every automatic run",
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
  /** The page's last outcome, drawn under the control band as the board places it. */
  notice?: ReactNode;
  onFilterChange: (filter: RunHistoryFilter) => void;
  triggers: Trigger[];
}

/**
 * The schedules and alert rules behind the Automatic filter, as their own card
 * above the runs they produced: a run with no obvious author is exactly when
 * the reader needs to see what authored it.
 */
function AutomaticRunsCard({
  automations,
  canManageAgents,
  count,
  triggers,
}: {
  automations: Automation[];
  canManageAgents: boolean;
  count: number;
  triggers: Trigger[];
}) {
  if (automations.length === 0 && triggers.length === 0) {
    return (
      <div className="card ui-card ah-automatic ah-automatic--empty">
        <EmptyState
          icon={<Icon name="activity" size={18} />}
          // Routines is administrator-only, so naming it to anyone else points
          // at a tab that is not in their tablist.
          text={canManageAgents
            ? "Schedules and alert rules are created alongside the agent they run, under Routines."
            : "Schedules and alert rules are created alongside the agent they run. An administrator sets them up."}
          title="Nothing is set to run on its own"
        />
      </div>
    );
  }
  return (
    <section aria-labelledby="ah-automatic-title" className="card ui-card ah-automatic">
      <header className="ui-card__header">
        <div className="ui-card__titles"><h2 id="ah-automatic-title">Automatic runs</h2></div>
        <StatusPill label={`Automatic (${count})`} tone="neutral" />
      </header>
      <p className="as-small as-muted ah-automatic__intro">
        These are the schedules and alert rules that create the runs below. Every run they
        start is read-only, and any change it proposes still needs a separate approval.
      </p>
      {automations.map((automation) => {
        const pill = automationPill(automation.status);
        return (
          <article className="ah-automatic__row" key={automation.id}>
            <div>
              {/* The pinned definition's own name, so a custom agent is not
                  shown by its profile id (ACC-140). */}
              <div className="as-small as-muted">Schedule · {applianceProfileName(automation.profile, automation.capability_disclosure?.agent)}</div>
              <div className="ah-automatic__name">{automation.name}</div>
              <div className="as-small as-muted">
                {automation.schedule_text}
                {showsNextRun(automation.status) && automation.next_run_at ? ` · next ${timeUntil(automation.next_run_at * 1000)}` : ""}
              </div>
              {automation.status?.detail && <div className="as-small as-muted">{automation.status.detail}</div>}
            </div>
            <StatusPill label={pill.label} reading={pill.reading} tone={pill.tone} />
          </article>
        );
      })}
      {triggers.map((trigger) => {
        const pill = automationPill(trigger.status);
        return (
          <article className="ah-automatic__row" key={trigger.id}>
            <div>
              <div className="as-small as-muted">Alert rule · {signalLabel(trigger)} {operatorLabel(trigger.operator)} {trigger.threshold}</div>
              <div className="ah-automatic__name">{trigger.name}</div>
              <div className="as-small as-muted">
                {trigger.last_value == null
                  ? "No reading yet"
                  : `Latest value ${trigger.last_value}${trigger.last_value_at ? `, read ${timeAgo(trigger.last_value_at * 1000)}` : ""}`}
                {trigger.last_triggered_at ? ` · fired ${timeAgo(trigger.last_triggered_at * 1000)}` : ""}
              </div>
              {trigger.status?.detail && <div className="as-small as-muted">{trigger.status.detail}</div>}
            </div>
            <StatusPill label={pill.label} reading={pill.reading} tone={pill.tone} />
          </article>
        );
      })}
    </section>
  );
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
  notice,
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
   * The filters are handed to the board so both live in its single band.
   */
  const filters = (
    <SegmentedControl
      label="Filter run history"
      onChange={onFilterChange}
      options={(Object.keys(labels) as RunHistoryFilter[]).map((item) => ({
        value: item,
        label: `${labels[item]} (${counts[item]})`,
      }))}
      value={filter}
    />
  );

  const intro = (
    <>
      {notice}
      {filter === "automatic" && (
        <AutomaticRunsCard
          automations={automations}
          canManageAgents={canManageAgents}
          count={counts.automatic}
          triggers={triggers}
        />
      )}
    </>
  );

  return (
    <div className="ah-history">
      <AgentTaskBoard
        {...board}
        controls={filters}
        emptyDetail={heading.detail}
        emptyTitle={board.taskView === "archive" ? `Archived · ${heading.empty}` : heading.empty}
        eyebrow="Evidence"
        heading={board.taskView === "archive" ? `Archived · ${heading.title}` : heading.title}
        intro={intro}
        tasks={filterRuns(board.tasks, filter, automaticTaskIds)}
      />
    </div>
  );
}
