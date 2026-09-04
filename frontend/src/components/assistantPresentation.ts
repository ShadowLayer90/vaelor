import type { AgentStatus } from "./agentTypes";

const SOURCE_LABELS: Record<string, string> = {
  "built-in-capability": "Policy-verified catalog",
  "built-in-planner": "Built-in planner",
};

export function assistantSourceLabel(source: string, status: AgentStatus | null) {
  if (source === "connected-model") {
    return status?.capability?.label || "Connected model";
  }
  return SOURCE_LABELS[source] || source.replaceAll("-", " ");
}
