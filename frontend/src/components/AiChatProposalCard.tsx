import type { AiChatAgentProposal } from "./aiChatTypes";
import { Icon, ICON_SIZE } from "./Icon";
import { Button } from "./ui";

/**
 * A custom-agent run AI Chat matched a request to, shown for review (the
 * ChatFocus board).
 *
 * Nothing has run when this appears: the card names the agent, its pinned
 * version, what it may touch, and offers to save the run for approval on the
 * Assistant.
 */
export function AiChatProposalCard({
  busy,
  onOpenRun,
  onReview,
  proposal,
  submitted,
}: {
  busy: boolean;
  onOpenRun: () => void;
  onReview: () => void;
  proposal: AiChatAgentProposal;
  submitted: boolean;
}) {
  return (
    <section className="ai-chat-proposal" aria-label="Custom agent run proposal">
      <span className="ai-chat-icon-tile ai-chat-icon-tile--accent"><Icon name="assistant" size={ICON_SIZE.nav} /></span>
      <div className="ai-chat-proposal__text">
        <small>Custom agent · version {proposal.profile_version} · approval required</small>
        <strong>{proposal.profile_name}</strong>
        <p>{proposal.task}</p>
        <small>
          {proposal.capabilities.length
            ? `Capabilities: ${proposal.capabilities.join(", ")}`
            : "Capabilities: none listed"}
        </small>
        <small>
          {proposal.app_grants.length
            ? proposal.app_grants.map((grant) => {
              const operations = grant.operations.map((operation) => operation.name).join(", ");
              return `${grant.app_name}: ${operations || "no operations listed"}`;
            }).join(" · ")
            : "App access: none granted"}
        </small>
        <small>
          {proposal.integrations.length
            ? `Integrations: ${proposal.integrations.join(", ")}`
            : "Integrations: none"}
        </small>
      </div>
      <div className="ai-chat-proposal__actions">
        <Button disabled={busy || submitted} onClick={onReview} type="button" variant={submitted ? "secondary" : "primary"}>{submitted ? "Saved for review" : "Review agent run"}</Button>
        {submitted && (
          <Button className="ai-chat-ghost" onClick={onOpenRun} type="button" variant="quiet">Open this run</Button>
        )}
      </div>
    </section>
  );
}
