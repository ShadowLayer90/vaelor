import type { ReactNode } from "react";
import type { Session } from "../types";
import { CustomAgentManager } from "./CustomAgentManager";
import { StatusPill } from "./StatusPill";
import type { AgentProfile, AgentTask, Automation, Trigger } from "./agentTypes";
import { destinations } from "../lib/destinations";

export function CustomAgentsPanel({
  automations,
  automationsPanel,
  modelLabel,
  modelReady,
  modelStatusResolved,
  onChanged,
  onViewRun,
  profiles,
  session,
  tasks,
  triggers,
}: {
  automations: Automation[];
  /** Schedules and alert rules, rendered with the agents they run. */
  automationsPanel: ReactNode;
  modelLabel: string;
  modelReady: boolean;
  /** False until `/agent/status` has answered at least once. */
  modelStatusResolved: boolean;
  onChanged: () => void;
  onViewRun: (task: AgentTask) => void;
  profiles: AgentProfile[];
  session: Session;
  tasks: AgentTask[];
  triggers: Trigger[];
}) {
  return (
    <div id="routines-panel" role="tabpanel">
      <div className="page-heading agent-heading">
        <div>
          <h1>{destinations.assistant.name}</h1>
          <p>Routines · What runs without you. Author an agent for your own domain, data, and workflows, and choose when it runs. Every capability is denied until you grant it.</p>
        </div>
        {/* Nothing is claimed about the model before `/agent/status` answers:
            an amber MODEL REQUIRED that turns green two seconds later is a
            false alarm about the reader's own machine. */}
        <StatusPill
          status={!modelStatusResolved ? "neutral" : modelReady ? "healthy" : "degraded"}
          label={!modelStatusResolved ? "Checking…" : modelReady ? modelLabel : "Model required"}
        />
      </div>
      <CustomAgentManager
        automations={automations}
        csrfToken={session.csrf_token}
        modelLabel={modelLabel}
        modelReady={modelReady}
        onChanged={onChanged}
        onViewRun={onViewRun}
        profiles={profiles}
        tasks={tasks}
        triggers={triggers}
      />
      {automationsPanel}
    </div>
  );
}
