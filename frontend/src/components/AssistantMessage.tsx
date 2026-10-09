import { continueProposedWorkload } from "../lib/workloadHandoff";
import { evidenceSourceLabel } from "../lib/evidenceSourceLabels";
import { performanceLine } from "../lib/performanceLine";
import type { ProposalReview } from "./AgentAssistantPanel";
import type { AgentRunProposal, ChatMessage } from "./agentTypes";
import { AssistantApplicationHandoff } from "./AssistantApplicationHandoff";
import { AssistantAnswerDestinations, AssistantNextStep } from "./AssistantNextStep";
import { AssistantProposalCard } from "./AssistantProposalCard";
import { assistantSourceLabel } from "./assistantPresentation";
import { Icon } from "./Icon";
import { Button } from "./ui";

/**
 * One turn of the conversation, drawn to the AssistAnswer board: the reader's
 * question on the right; Vaelor's byline with who answered and how fast, the
 * answer, the evidence it used, suggested next steps, and whichever extra the
 * answer carries - an action to review, a deployment to continue, an app to
 * research, an agent run to review - then where to go next.
 */
export function AssistantMessage({
  busy,
  item,
  onPrepareAgentRun,
  onReview,
}: {
  busy: boolean;
  item: ChatMessage;
  onPrepareAgentRun: (proposal: AgentRunProposal) => void;
  onReview: (review: ProposalReview) => void;
}) {
  const metadata = item.metadata;
  if (item.role === "user") {
    return (
      <article className="as-msg as-msg--user">
        <span className="sr-only">You</span>
        <div className="as-bubble as-bubble--me">
          {item.content.split("\n").filter(Boolean).map((paragraph, index) => <p key={index}>{paragraph}</p>)}
        </div>
      </article>
    );
  }
  const performance = performanceLine(metadata?.performance);
  const source = metadata?.source ? assistantSourceLabel(metadata.source, metadata.model_label) : "";
  const evidence = metadata?.evidence ?? [];
  return (
    <article className={`as-msg as-msg--assistant${metadata?.stopped ? " as-msg--stopped" : ""}`}>
      <div className="as-msg__byline">
        {/*
          * A stopped response was not Vaelor speaking, so it is not signed as
          * if it were. The byline is the terminal state.
          */}
        {metadata?.stopped
          ? <span className="as-stopped-pill">Response stopped</span>
          : <><span aria-hidden="true" className="as-msg__mark">V</span><strong>Vaelor</strong></>}
        {(source || performance) && (
          <span className="as-small as-muted">{[source, performance].filter(Boolean).join(" · ")}</span>
        )}
      </div>
      <div className="as-bubble as-bubble--ai">
        {item.content.split("\n").filter(Boolean).map((paragraph, index) => <p key={index}>{paragraph}</p>)}
      </div>
      {evidence.length > 0 && (
        // Open as it arrives (the board): the evidence is half of the answer.
        <details className="as-evidence" open>
          <summary>
            <span className="as-small as-muted">Evidence used · {evidence.length} source{evidence.length === 1 ? "" : "s"}</span>
            <Icon name="chevron" size={16} />
          </summary>
          <ul>
            {evidence.map((entry) => (
              <li key={`${entry.source}-${entry.summary}`}>
                <span title={entry.source}>{evidenceSourceLabel(entry.source)}</span>
                <span className="as-mono as-muted">{entry.summary}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
      {metadata?.suggested_actions?.length ? (
        <div className="as-next-steps">
          <span className="as-label">Suggested next steps</span>
          <ul>{metadata.suggested_actions.map((action) => <li key={action}><AssistantNextStep action={action} /></li>)}</ul>
        </div>
      ) : null}
      {metadata?.proposed_job && !metadata.application_intent ? (
        <AssistantProposalCard
          busy={busy}
          job={metadata.proposed_job}
          onContinue={() => continueProposedWorkload(metadata.proposed_job!)}
          onReview={() => onReview({
            job: metadata.proposed_job!,
            summary: item.content,
            evidence,
            suggestedActions: metadata.suggested_actions ?? [],
          })}
        />
      ) : null}
      {metadata?.application_intent ? <AssistantApplicationHandoff intent={metadata.application_intent} /> : null}
      {metadata?.proposed_agent_task ? (
        <section aria-label="Custom agent run proposal" className="as-extra as-extra--column">
          <span className="as-small as-muted">Custom agent · version {metadata.proposed_agent_task.profile_version}</span>
          <strong>{metadata.proposed_agent_task.profile_name}</strong>
          <p>{metadata.proposed_agent_task.task}</p>
          <span className="as-small as-muted">
            {metadata.proposed_agent_task.capabilities.length
              ? `Granted capabilities: ${metadata.proposed_agent_task.capabilities.join(", ")}`
              : "No appliance capabilities granted"}
          </span>
          <span className="as-small as-muted">
            {metadata.proposed_agent_task.integrations?.length
              ? `API integrations: ${metadata.proposed_agent_task.integrations.join(", ")}`
              : "No API integrations granted"}
          </span>
          <Button disabled={busy} onClick={() => onPrepareAgentRun(metadata.proposed_agent_task!)} type="button" variant="primary">
            Review agent run
          </Button>
        </section>
      ) : null}
      {/*
        * The routes the answer's destinations resolve to. The suggested steps
        * above link a place they happen to name in a sentence; this is the
        * answer's own machine-readable list, so a redirect that names AI Chat
        * only in its prose - which is every model-authored refusal - still
        * reaches it.
        */}
      <AssistantAnswerDestinations steps={metadata?.next_steps} />
    </article>
  );
}
