/**
 * Names for the evidence sources an answer or a proposal cites.
 *
 * "Evidence used" printed `source.replaceAll(".", " ")`, so the owner read
 * "system telemetry" and "assistant machine-brief" - an identifier with its
 * dots removed (W6 retest, LESSONS 5). The ids have one owner,
 * `vaelor/answer_evidence.py` EVIDENCE_SOURCES and EVIDENCE_SOURCE_FAMILIES;
 * `evidenceSourceLabels.test.ts` reads that tuple and requires a name here for
 * each (LESSONS 6).
 *
 * A source the appliance does not name - a URL a custom agent read, a source a
 * model put in its own answer - is shown exactly as recorded: rewriting it
 * would invent a name for something nobody named.
 */

const labels: Record<string, string> = {
  "ai-chat.connections": "AI Chat connections",
  "apps.catalog": "App catalog",
  "assistant.capability": "What the Assistant can do",
  "assistant.fallback": "The Assistant's fallback answer",
  "assistant.machine-brief": "This machine's standing facts",
  "assistant.machine-events": "What happened on a machine",
  "assistant.memory": "Reviewed Assistant memory",
  "assistant.policy": "The Assistant's request rules",
  "assistant.response-guard": "The Assistant's reply check",
  "assistant.scope": "What the Assistant answers",
  "assistant.scope-guard": "The Assistant's scope check",
  "assistant.time-scope": "The time range asked about",
  "cluster.digest": "The cluster's machines",
  "cluster.serving-mode": "What serves AI Chat",
  "cooling.status": "Cooling readings",
  "display.status": "Display settings",
  "gpu.status": "GPU readings",
  "health.status": "Health check",
  "inference.status": "Which model each engine runs",
  "jobs.recent": "Recent operations",
  "lighting.status": "Lighting settings",
  "llm-server.status": "LLM Server status",
  "logs.service": "A service's recent log",
  "metrics.history": "Telemetry history",
  "network.status": "Network interfaces",
  "npu.status": "NPU readings",
  "recovery.checkpoints": "Recovery checkpoints",
  "services.status": "Service status",
  "storage.status": "Storage readings",
  "system.identity": "This machine's identity",
  "system.telemetry": "Live telemetry",
  "updates.status": "Operating-system updates",
  "workloads.capabilities": "App and model capabilities",
  "workloads.inventory": "Installed apps and models",
  // Sources the console writes itself, in its own review dialogs (FE-W7-6).
  // The backend never sends these; evidenceSourceLabels.test.ts reads every
  // `{ source, summary }` the frontend writes and requires a name for each.
  "compose.policy": "The Docker stack safety rules",
  "setup-assistant.plan": "The setup assistant's plan",
  "web-research.image": "The pinned search image",
  "web-research.endpoint": "The private search address",
};

/** A fixed prefix and the name the owner chose after it. */
const families: ReadonlyArray<[prefix: string, label: string]> = [
  ["custom-agent.", "Custom agent"],
  ["skill.", "Skill"],
];

export function evidenceSourceIsNamed(source: string): boolean {
  const id = source.trim();
  return Boolean(labels[id]) || families.some(([prefix]) => id.startsWith(prefix) && id.length > prefix.length);
}

/** The words for one evidence source: its name, or the source as recorded. */
export function evidenceSourceLabel(source: string): string {
  const id = (source ?? "").trim();
  if (!id) return "Source not recorded";
  const named = labels[id];
  if (named) return named;
  const family = families.find(([prefix]) => id.startsWith(prefix) && id.length > prefix.length);
  if (family) return `${family[1]}: ${id.slice(family[0].length)}`;
  return id;
}
