import type { AgentConnector, AgentProfile } from "./agentTypes";

/*
 * The custom-agent definition as the editor holds it, and the words the
 * workshop uses for its grants. Shared by the manager (which saves it), the
 * editor dialog (which edits it) and the agent rows (which summarise it), so a
 * scope is named the same way in all three.
 */

export const CAPABILITIES = [
  ["system:read", "Hardware and system facts", "Telemetry, storage, network, services, and OS facts"],
  ["cooling:read", "Cooling", "Fan and thermal state, where this machine reports it"],
  ["workloads:read", "Workloads", "Docker apps, local models, and capabilities"],
  ["jobs:read", "Jobs", "Deployment state and audited history"],
  ["assistant:read", "Assistant", "Reviewed memory and assistant status"],
  ["cluster:read", "Fleet and cluster facts", "Controller, worker, service, and placement health"],
] as const;

export type Draft = {
  id?: string;
  name: string;
  description: string;
  instructions: string;
  scopes: string[];
  permissions: string[];
  read_collection_ids: string[];
  write_collection_id: string;
  web_access: { enabled: boolean; allowed_domains: string[] };
  connectors: AgentConnector[];
};

export type KnowledgeCollection = {
  id: string;
  name: string;
  description: string;
  document_count: number;
};

export const EMPTY_DRAFT: Draft = {
  name: "",
  description: "",
  instructions: "",
  scopes: [],
  permissions: [],
  read_collection_ids: [],
  write_collection_id: "",
  web_access: { enabled: false, allowed_domains: [] },
  connectors: [],
};

/**
 * What an agent may read and do, in one line of words (the row under its
 * purpose). Appliance scopes by their editor names; research is said as
 * "guarded web", which is what the research scope grants.
 */
export function agentGrantsLine(agent: AgentProfile): string {
  const appliance = agent.scopes
    .filter((scope) => scope !== "research:read")
    .map((scope) => CAPABILITIES.find(([id]) => id === scope)?.[1] ?? scope.replace(":read", ""));
  const parts = [appliance.length ? appliance.join(" · ") : "No appliance access"];
  if (agent.web_access?.enabled) {
    const domains = agent.web_access.allowed_domains.length;
    parts.push(domains ? `guarded web (${domains} domain${domains === 1 ? "" : "s"})` : "guarded web search");
  }
  for (const permission of agent.permissions ?? []) parts.push(permission.replace(":", " "));
  return parts.join(" · ");
}
