import { apiRequest } from "./api";
import type { AiChatAgentProposal } from "../components/aiChatTypes";

/**
 * Save an agent proposal from AI Chat as an Assistant task awaiting approval.
 * Moved out of `AiChat` (VD-221 review round 1, item 13) to give the page room
 * under the 1,000-line ceiling.
 */
export async function saveAgentRun(proposal: AiChatAgentProposal, csrfToken: string): Promise<void> {
  await apiRequest<{ id?: string }>(
    "/assistant/tasks",
    {
      method: "POST",
      body: JSON.stringify({
        title: `${proposal.profile_name} chat request`,
        description: proposal.task,
        profile: proposal.profile_id,
        profile_version: proposal.profile_version,
        approval_required: true,
        idempotency_key: globalThis.crypto?.randomUUID?.() ?? `ai-chat-${Date.now()}`,
      }),
    },
    csrfToken,
  );
}

/*
 * Assistant › Agents is administrator-only. Naming it to anyone else sent
 * them looking for a tab that is not in their tablist; the run is in the
 * Assistant's own run history either way, under the Agent runs filter.
 * `task.id` used to pick between two identical strings.
 */
export function agentRunSavedNotice(role: string): string {
  return role === "administrator"
    ? "Saved. Open it under Assistant › Agents, in this agent's run history, to approve and run it."
    : "Saved. Open it under Assistant, in the Agent runs filter, to approve and run it.";
}

/*
 * This used to offer the "Assistant task ledger" and land on the Assistant's
 * chat tab. No ledger exists in this product, and the run the user was sent
 * to find was on a different tab behind a collapsed disclosure - so the one
 * button offered after a matched agent request led nowhere useful.
 */
export function openAgentRun(): void {
  window.dispatchEvent(new CustomEvent("pironman:navigate", { detail: "assistant" }));
  window.dispatchEvent(new CustomEvent("vaelor:assistant-tab", { detail: "customAgents" }));
}
