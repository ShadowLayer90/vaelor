import { AiChatModelPicker } from "./AiChatModelPicker";
import type { AiChatAgent, AiChatConnection } from "./aiChatTypes";
import { Button, Select } from "./ui";

/**
 * The row above the transcript: what this chat is, what answers it, and how it
 * is being viewed.
 *
 * It has its own file because `AiChat.tsx` stood at 995 lines against the
 * 1,000-line limit in `CLAUDE.md`, and because the defect this row had is a
 * layout defect that no amount of care inside a 995-line component would have
 * been readable next to.
 *
 * **What was wrong (owner screenshot, Z2).** *Model* carried a sentence under
 * it — *"This chat uses …"* — and *Agent (optional)* carried nothing, so the
 * two columns were different heights, the selects did not line up, and the
 * Focus view / Details buttons aligned to neither. Three controls in one row,
 * three different baselines.
 *
 * Two things fix it, and both are needed. Every column now has the same three
 * parts — a label, a control, and a sentence about what that control is
 * currently doing — so the row is symmetrical by construction rather than by
 * coincidence; and the columns are top-aligned in `ai-chat-picker.css`, so a
 * sentence that wraps to two lines grows downwards instead of pushing its own
 * control off the line.
 */

/**
 * What the Agent select is doing right now, in one sentence.
 *
 * `resolved` is the same distinction the model picker draws between "we have
 * not asked" and "we asked and the answer is none". An empty list before
 * `/assistant/profiles` answers is not evidence that this appliance has no
 * agents, and saying so would be a settled claim about data nobody has read.
 */
export function agentSelectionHint(
  agents: AiChatAgent[],
  selectedAgentId: string,
  resolved: boolean,
): string {
  if (!resolved) return "Vaelor is still reading which agents this appliance has.";
  if (!agents.length) {
    return "No custom agents are set up on this appliance, so answers come from the model alone.";
  }
  const chosen = agents.find((agent) => agent.id === selectedAgentId);
  if (!chosen) {
    return `No agent. The model answers directly; ${agents.length} agent${agents.length === 1 ? " is" : "s are"} available.`;
  }
  return `${chosen.name} is offered this chat's questions. Anything it proposes still needs your approval.`;
}

export function AiChatToolbar({
  agents,
  agentsResolved,
  busy,
  connection,
  detailsOpen,
  focusMode,
  modelFailures,
  models,
  onChooseModel,
  onSelectAgent,
  onToggleDetails,
  onToggleFocus,
  selectedAgentId,
  selectedModel,
  subtitle,
  title,
}: {
  agents: AiChatAgent[];
  /** False until `/assistant/profiles` has answered once, however it answered. */
  agentsResolved: boolean;
  busy: boolean;
  connection: AiChatConnection | null | undefined;
  detailsOpen: boolean;
  focusMode: boolean;
  modelFailures: Record<string, string>;
  models: string[];
  onChooseModel: (model: string) => void;
  onSelectAgent: (id: string) => void;
  onToggleDetails: () => void;
  onToggleFocus: () => void;
  selectedAgentId: string;
  selectedModel: string;
  subtitle: string;
  title: string;
}) {
  return (
    <header className="ai-chat-toolbar">
      <div>
        <strong title={title}>{title}</strong>
        <span>{subtitle}</span>
      </div>
      <AiChatModelPicker
        connection={connection}
        modelFailures={modelFailures}
        models={models}
        onChoose={onChooseModel}
        value={selectedModel}
      />
      <Select
        disabled={busy}
        hint={agentSelectionHint(agents, selectedAgentId, agentsResolved)}
        id="ai-chat-agent"
        label="Agent (optional)"
        onChange={(event) => onSelectAgent(event.target.value)}
        value={selectedAgentId}
      >
        <option value="">No agent</option>
        {agents.map((agent) => (
          <option disabled={agent.operational === false} key={agent.id} value={agent.id}>
            {agent.name}{agent.operational === false ? " (unavailable)" : ""}
          </option>
        ))}
      </Select>
      {/*
        * A labelled column like the two beside it, rather than two loose
        * buttons hanging in the row. The label is what puts these controls on
        * the same baseline as the selects: without it the group had no top
        * row to align to, which is why it aligned to neither.
        */}
      <div className="ai-chat-toolbar__actions" role="group" aria-labelledby="ai-chat-view-label">
        <span className="ui-field__label" id="ai-chat-view-label">View</span>
        <div>
          <Button className="ai-chat-toolbar__focus" onClick={onToggleFocus} type="button" variant="quiet">
            {focusMode ? "Exit focus" : "Focus view"}
          </Button>
          <Button
            className={detailsOpen ? "ai-chat-toolbar__details is-active" : "ai-chat-toolbar__details"}
            onClick={onToggleDetails}
            type="button"
            variant="quiet"
          >
            Details
          </Button>
        </div>
      </div>
    </header>
  );
}
