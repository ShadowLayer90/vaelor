import type { StatusTone } from "../components/ui/status";

/**
 * Every word a deployed cluster agent's `runtime.state` may carry. The owner is
 * `vaelor/agent_runtime_state.py` `AGENT_RUNTIME_STATES`; this is the one
 * TypeScript copy, marked for `tests/test_wire_vocabularies.py`, so a word
 * added on one side alone fails the suite instead of reaching the badge as an
 * unknown state.
 */
export const AGENT_RUNTIME_STATES = [ // vocabulary: agent-runtime-state
  "serving",
  "starting",
  "failed",
  "removing",
  "model-missing",
  "model-unloaded",
  "model-paused",
  "unknown",
  "not-running",
  "no-keys",
  "applying-keys",
  "applying-changes",
  "model-not-answering",
] as const;

export type AgentRuntimeState = (typeof AGENT_RUNTIME_STATES)[number];

/**
 * What a deployed agent is actually doing, read by the backend from its row,
 * its backing model deployment, the bridge's unit/gate/key reading and the
 * runtime's own health check. `state` is kept as the raw wire word so a word
 * this console has no copy for is shown as unknown rather than dropped;
 * `detail` is the backend's sentence for every state but `serving`.
 */
export interface AgentRuntime {
  state: string;
  reason: string;
  detail: string;
}

export interface AgentRuntimeBadge {
  label: string;
  tone: StatusTone;
}

const BADGES: Record<AgentRuntimeState, AgentRuntimeBadge> = {
  serving: { label: "Serving", tone: "success" },
  starting: { label: "Starting", tone: "info" },
  failed: { label: "Failed", tone: "danger" },
  removing: { label: "Removing", tone: "neutral" },
  "model-missing": { label: "Model removed", tone: "danger" },
  "model-unloaded": { label: "Model unloaded", tone: "warning" },
  "model-paused": { label: "Paused (idle)", tone: "info" },
  unknown: { label: "Unknown", tone: "neutral" },
  "not-running": { label: "Not running", tone: "danger" },
  "no-keys": { label: "No keys", tone: "warning" },
  "applying-keys": { label: "Applying key change", tone: "info" },
  "applying-changes": { label: "Applying changes", tone: "info" },
  "model-not-answering": { label: "Model not answering", tone: "danger" },
};

const UNKNOWN_BADGE: AgentRuntimeBadge = BADGES.unknown;

function isAgentRuntimeState(value: string): value is AgentRuntimeState {
  return (AGENT_RUNTIME_STATES as readonly string[]).includes(value);
}

/**
 * The agent's badge, from what it is DOING rather than from what its deploy
 * row says. "Serving" is said for the `serving` reading and nothing else: the
 * card once said Serving with one active key after every key was revoked
 * (ACC-071), because it read the stored row.
 *
 * A backend older than the runtime reading sends no `runtime`; then nothing
 * optimistic is derived - the stored row word is shown as recorded, neutral,
 * and never as "Serving".
 */
export function agentRuntimeBadge(runtime?: AgentRuntime | null, storedState = ""): AgentRuntimeBadge {
  if (!runtime) {
    const word = storedState.trim();
    if (!word) return UNKNOWN_BADGE;
    return { label: `Recorded as ${word}`, tone: "neutral" };
  }
  return isAgentRuntimeState(runtime.state) ? BADGES[runtime.state] : UNKNOWN_BADGE;
}

/** The backend's own sentence for an agent that is not serving, or "". */
export function agentRuntimeNote(runtime?: AgentRuntime | null): string {
  if (!runtime || runtime.state === "serving") return "";
  return runtime.detail;
}
