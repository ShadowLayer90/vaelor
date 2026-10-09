import { type ReactNode, useState } from "react";
import type { Session } from "../types";
import { AUTOMATIONS_LEAD, AUTOMATIONS_TITLE } from "./agent-center-automations-panel";
import { useAssistantPlace } from "./assistantBar";
import { ASSISTANT_TAB_PLACES } from "./AssistantNavigationTabs";
import { CustomAgentManager } from "./CustomAgentManager";
import { RoutinesCreateIn } from "./routinesCreateIn";
import { StatusPill } from "./StatusPill";
import { SegmentedControl } from "./ui";
import type { AgentProfile, AgentTask, Automation, Trigger } from "./agentTypes";
import { destinations } from "../lib/destinations";

type RoutinesView = "agents" | "schedules";

/** The two views' words, in the segmented control and the top bar breadcrumb ("Routines / Agents"). */
const ROUTINES_VIEW_LABELS: Readonly<Record<RoutinesView, string>> = {
  agents: "Agents",
  schedules: "Schedules and alerts",
};

/**
 * The Assistant's Routines tab (VD-200 decision 5): two views chosen by the
 * segmented control beside the heading - Agents (the AssistRoutines board)
 * and Schedules and alerts (the AssistAutomations board). Only the shown view
 * is mounted, so only its pill and primary action are in the top bar. The
 * schedule, alert-rule and channel create forms open as dialogs here; the
 * provider below says so to the panels the page hands in.
 */
export function CustomAgentsPanel({
  automations,
  automationsPanel,
  clusterModelServing,
  modelLabel,
  modelNotAnsweringReason,
  modelReady,
  modelStatusResolved,
  onChanged,
  onViewRun,
  profiles,
  profilesRead = true,
  profilesReadError = "",
  session,
  tasks,
  triggers,
}: {
  automations: Automation[];
  /** Schedules, alert rules and where alerts go: the Schedules and alerts view. */
  automationsPanel: ReactNode;
  /** A cluster inference model is serving, so a cluster agent can be authored
      and deployed even when the single-node Assistant model is not ready. */
  clusterModelServing: boolean;
  modelLabel: string;
  /** Non-empty when a configured model is not answering (from the shared
      `assistantModelReadiness` projection Ask also reads). */
  modelNotAnsweringReason: string;
  modelReady: boolean;
  /** False until `/agent/status` has answered at least once. */
  modelStatusResolved: boolean;
  onChanged: () => void;
  onViewRun: (task: AgentTask) => void;
  profiles: AgentProfile[];
  /** False until the agent list has been read once: no count and no "none yet" before then. */
  profilesRead?: boolean;
  /** Why the agent list could not be read, when it could not. */
  profilesReadError?: string;
  session: Session;
  tasks: AgentTask[];
  triggers: Trigger[];
}) {
  const [view, setView] = useState<RoutinesView>("agents");
  const agentCount = profiles.filter((item) => item.custom).length;
  const agentsView = view === "agents";
  useAssistantPlace([ASSISTANT_TAB_PLACES.routines, ROUTINES_VIEW_LABELS[view]]);
  return (
    <div className="as-panel ar-routines" id="routines-panel" role="tabpanel">
      {/* The destination's name for assistive technology; the view's own
          title is the large line a reader sees. */}
      <h1 className="sr-only">{destinations.assistant.name}</h1>
      <div className="ar-routines__head">
        <div className="ar-routines__titles">
          <h2 className="ar-routines__title">{agentsView ? "What runs without you" : AUTOMATIONS_TITLE}</h2>
          <p className="ar-routines__lead">
            {agentsView
              ? "Author an agent for your own domain, data, and workflows, and choose when it runs. Every capability is denied until you grant it."
              : AUTOMATIONS_LEAD}
          </p>
        </div>
        <SegmentedControl
          label="Routines views"
          onChange={setView}
          options={[
            // No count before the list is read: "Agents 0" there was a false zero (VD-200 review).
            { value: "agents", label: profilesRead ? `${ROUTINES_VIEW_LABELS.agents} ${agentCount}` : ROUTINES_VIEW_LABELS.agents },
            { value: "schedules", label: ROUTINES_VIEW_LABELS.schedules },
          ]}
          value={view}
        />
      </div>
      {/* The whole Assistant is administrator-only (the rail turns it away for
          anybody else), so this does not promise other roles a view they never get. */}
      {agentsView && <span className="ar-meta">Administrators only, like the rest of the Assistant.</span>}
      {agentsView ? (
        <CustomAgentManager
          automations={automations}
          barStatus={(
            /* Nothing is claimed about the model before `/agent/status` answers:
               an amber MODEL REQUIRED that turns green two seconds later is a
               false alarm about the reader's own machine. */
            <StatusPill
              /* A failed readiness read is "Not read", never "Checking…" for ever (VD-200). */
              reading={!modelStatusResolved && profilesReadError ? "unread" : undefined}
              status={!modelStatusResolved ? "neutral" : modelNotAnsweringReason ? "degraded" : modelReady || clusterModelServing ? "healthy" : "degraded"}
              label={!modelStatusResolved ? (profilesReadError ? "Not read" : "Checking…") : modelNotAnsweringReason ? "Model unreachable" : modelReady ? modelLabel : clusterModelServing ? "Cluster model" : "Model required"}
            />
          )}
          clusterModelServing={clusterModelServing}
          csrfToken={session.csrf_token}
          modelLabel={modelLabel}
          modelNotAnsweringReason={modelNotAnsweringReason}
          modelReady={modelReady}
          modelStatusResolved={modelStatusResolved}
          onChanged={onChanged}
          onViewRun={onViewRun}
          profiles={profiles}
          profilesRead={profilesRead}
          profilesReadError={profilesReadError}
          tasks={tasks}
          triggers={triggers}
        />
      ) : (
        <RoutinesCreateIn.Provider value="dialog">{automationsPanel}</RoutinesCreateIn.Provider>
      )}
    </div>
  );
}
