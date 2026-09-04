import { destinations } from "../lib/destinations";
import { AssistantRunHistory, type RunHistoryFilter } from "./AssistantRunHistory";
import { Notice, type NoticeSeverity } from "./ui";
import type { AgentTask, Automation, HandoffTarget, Trigger } from "./agentTypes";
import type { Session } from "../types";

/**
 * What has already run: the whole archive, and nothing else.
 *
 * This was the second half of Ask. A live chat and a forty-five-row audit
 * archive on one screen is not one job — the archive measured 1,883px against
 * the answer's 146, a 13:1 ratio against the thing the screen exists for — and
 * the two are in different tenses. Approval is unchanged: a run that needs
 * review still needs review, and it is reviewed here.
 */
export function AssistantHistoryPanel({
  automaticTaskIds,
  automations,
  busy,
  filter,
  handoffSelections,
  handoffTargets,
  notice,
  noticeSeverity,
  onArchiveFinishedTasks,
  onDiscussTask,
  onFilterChange,
  onHandoffSelection,
  onHandoffTask,
  onRefresh,
  onRetryTask,
  onEscalateTask,
  onTaskViewChange,
  onTransitionTask,
  onUpdated,
  session,
  tasks,
  taskView,
  triggers,
}: {
  automaticTaskIds: ReadonlySet<string>;
  automations: Automation[];
  busy: boolean;
  filter: RunHistoryFilter;
  handoffSelections: Record<string, string>;
  handoffTargets: HandoffTarget[];
  /**
   * The last thing that happened, for a reader standing on this tab.
   *
   * Every outcome this page can produce — approving a run, retrying one,
   * handing one off, clearing finished ones, and the prepared-run handover that
   * navigates here from Ask — was written to a banner that only Ask rendered.
   * The tab that performs the action is the tab that has to report it.
   */
  notice: string;
  noticeSeverity: NoticeSeverity;
  onArchiveFinishedTasks: () => void;
  onDiscussTask: (task: AgentTask) => void;
  onFilterChange: (filter: RunHistoryFilter) => void;
  onHandoffSelection: (taskId: string, username: string) => void;
  onHandoffTask: (task: AgentTask) => void;
  onRefresh: () => void;
  onRetryTask: (task: AgentTask) => void;
  /** Re-run the SAME task on the more capable GPU model. */
  onEscalateTask: (task: AgentTask) => void;
  onTaskViewChange: () => void;
  onTransitionTask: (task: AgentTask, state: "ready" | "cancelled") => void;
  onUpdated: (message: string) => void;
  session: Session;
  tasks: AgentTask[];
  taskView: "recent" | "archive";
  triggers: Trigger[];
}) {
  return (
    <div id="history-panel" role="tabpanel">
      <div className="page-heading agent-heading">
        <div>
          <h1>{destinations.assistant.name}</h1>
          <p>History · Every appliance check, agent run, and automatic run this appliance has produced, with the evidence it used.</p>
        </div>
      </div>
      {notice && <Notice severity={noticeSeverity}>{notice}</Notice>}
      <AssistantRunHistory
        automaticTaskIds={automaticTaskIds}
        automations={automations}
        busy={busy}
        canManageAgents={session.user.role === "administrator"}
        csrfToken={session.csrf_token}
        filter={filter}
        handoffSelections={handoffSelections}
        handoffTargets={handoffTargets}
        onArchiveFinished={onArchiveFinishedTasks}
        onDiscuss={onDiscussTask}
        onFilterChange={onFilterChange}
        onHandoff={onHandoffTask}
        onHandoffSelection={onHandoffSelection}
        onRefresh={onRefresh}
        onRetry={onRetryTask}
        onEscalate={onEscalateTask}
        onTaskViewChange={onTaskViewChange}
        onTransition={onTransitionTask}
        onUpdated={onUpdated}
        tasks={tasks}
        taskView={taskView}
        triggers={triggers}
      />
    </div>
  );
}
