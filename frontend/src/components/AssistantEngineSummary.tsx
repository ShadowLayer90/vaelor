import { Icon } from "./Icon";
import { Button } from "./ui";
import type { AgentStatus } from "./agentTypes";

export function AssistantEngineSummary({
  onToggle,
  settingsOpen,
  status,
}: {
  onToggle: () => void;
  settingsOpen: boolean;
  status: AgentStatus | null;
}) {
  const loading = status === null;
  return (
    <section className="assistant-engine">
      <span className="assistant-engine__icon"><Icon name={loading || status.configured ? "network" : "cpu"} /></span>
      <div>
        <small>ACTIVE INTELLIGENCE</small>
        <strong>{loading ? "Checking active intelligence" : status.configured ? status.provider : "Built-in appliance help"}</strong>
        <p>{loading ? "Reading the current Assistant connection from this node…" : status.configured ? `${status.model || "Auto-detected model"} · live hardware context included` : "Basic answers and live diagnostics work now. Add a model for broader reasoning."}</p>
        {status?.capability && (
          <div className={`assistant-engine__capability assistant-engine__capability--${status.capability.tier}`}>
            <span>{status.capability.label}</span>
            <p>{status.capability.description}</p>
            {status.capability.limitations.length > 0 && (
              <small>{status.capability.limitations.join(" ")}</small>
            )}
          </div>
        )}
      </div>
      <Button disabled={loading} onClick={onToggle} type="button" variant="quiet">
        {settingsOpen ? "Close settings" : "Change intelligence"}
      </Button>
    </section>
  );
}
