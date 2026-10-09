import { useId } from "react";
import { openAiChatConnectionForm } from "../lib/connectionFormHandoff";
import { destinations } from "../lib/destinations";
import type { AiChatThinkingState } from "../hooks/useAiChatThinking";
import { AiChatModelPicker, moveOptionFocus, useChatPopover } from "./AiChatModelPicker";
import { AiChatThinkingControl } from "./AiChatThinking";
import type { AiChatAgent, AiChatClustering, AiChatConnection } from "./aiChatTypes";
import { Icon, ICON_SIZE } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button } from "./ui";

/**
 * The row above the transcript (the Chat, ChatTemporary and ChatFocus
 * boards): what this chat is, what answers it, and how it is being viewed.
 * The chat's name and one line under it on the left; Model, Agent, Focus view
 * and Details on the right.
 */

/**
 * What the Agent selector is doing right now, in one sentence.
 *
 * `resolved` is the same distinction the model picker draws between "we have
 * not asked" and "we asked and the answer is none". An empty list before
 * `/assistant/profiles` answers is not evidence that this appliance has no
 * agents, and saying so would be a settled claim about data nobody has read.
 */
export interface AgentsReading {
  /** `/assistant/profiles` could not be read: unknown, never "none". */
  failed: boolean;
  /** The Assistant's custom agents that exist but are not active. */
  inactive: number;
  /**
   * Whether this viewer can open Assistant > Routines, which is
   * administrator-only (R-F10): an operator is told who can activate one.
   */
  administrator?: boolean;
}

/** The six states the ChatDialogs board draws for the agent selector. */
export type AgentSelectorState = "reading" | "not-read" | "none-active" | "none" | "unchosen" | "chosen";

export function agentSelectorState(
  agents: AiChatAgent[],
  selectedAgentId: string,
  resolved: boolean,
  reading: AgentsReading = { failed: false, inactive: 0 },
): AgentSelectorState {
  if (!resolved) return "reading";
  if (reading.failed) return "not-read";
  if (!agents.length) return reading.inactive > 0 ? "none-active" : "none";
  return agents.some((agent) => agent.id === selectedAgentId) ? "chosen" : "unchosen";
}

/** Why the Agent button is held while an answer is on its way, shown beside it. */
const AGENT_HELD_REASON = "Changes after this answer";

/** What the Agent button shows after its key, per state. */
const AGENT_BUTTON_VALUE: Record<Exclude<AgentSelectorState, "chosen">, string> = {
  reading: "Reading…",
  "not-read": "Not read",
  "none-active": "None active",
  none: "No agent",
  unchosen: "No agent",
};

/**
 * W4d-D17 (LESSONS 6, 5): AI Chat offers the ASSISTANT's custom agents only -
 * the list `chat_agent_proposals.custom_profiles` matches against; a cluster
 * agent serves its own endpoint and would dead-end here. "No custom agents are
 * set up on this appliance" was false beside Cluster > Agents & tools, so the
 * sentence names the list it read and why the others are not in it.
 */
const CLUSTER_AGENTS_ELSEWHERE = "Cluster agents answer on their own endpoints and are not offered here.";

export function agentSelectionHint(
  agents: AiChatAgent[],
  selectedAgentId: string,
  resolved: boolean,
  reading: AgentsReading = { failed: false, inactive: 0 },
): string {
  const state = agentSelectorState(agents, selectedAgentId, resolved, reading);
  if (state === "reading") return "Vaelor is still reading which agents this appliance has.";
  if (state === "not-read") {
    return "Vaelor could not read the Assistant's custom agents, so none is offered; answers come from the model alone.";
  }
  if (state === "none-active") {
    const count = reading.inactive === 1 ? "1 custom agent" : `${reading.inactive} custom agents`;
    const activate = reading.administrator
      ? "Activate one in Assistant > Routines to offer it here."
      : "An administrator can activate one to offer it here.";
    return `The Assistant has ${count}, not active, so answers come from the model alone. ${activate}`;
  }
  if (state === "none") {
    return `AI Chat can hand a question to the Assistant's custom agents, and the Assistant has none active, so answers come from the model alone. ${CLUSTER_AGENTS_ELSEWHERE}`;
  }
  if (state === "unchosen") {
    // An agent the Assistant reports as unavailable is listed but cannot be
    // chosen, so it is not counted as available.
    const available = agents.filter((agent) => agent.operational !== false).length;
    return `No agent. The model answers directly; ${available} agent${available === 1 ? " is" : "s are"} available.`;
  }
  const chosen = agents.find((agent) => agent.id === selectedAgentId)!;
  return `${chosen.name} is offered this chat's questions. Anything it proposes still needs your approval.`;
}

function AgentPicker({
  agents,
  busy,
  hint,
  onSelect,
  selectedAgentId,
  state,
}: {
  agents: AiChatAgent[];
  busy: boolean;
  hint: string;
  onSelect: (id: string) => void;
  selectedAgentId: string;
  state: AgentSelectorState;
}) {
  const popover = useChatPopover();
  const ids = useId().replaceAll(":", "");
  const chosen = agents.find((agent) => agent.id === selectedAgentId);
  const value = chosen ? chosen.name : AGENT_BUTTON_VALUE[state as keyof typeof AGENT_BUTTON_VALUE];
  const choose = (id: string) => {
    onSelect(id);
    popover.close();
  };
  const option = (id: string, name: string, line: string, unavailable = false) => (
    <Button
      aria-selected={id === selectedAgentId}
      className={id === selectedAgentId ? "ai-chat-option is-selected" : "ai-chat-option"}
      disabled={unavailable}
      key={id || "none"}
      onClick={() => choose(id)}
      role="option"
      type="button"
      variant="quiet"
    >
      <span className="ai-chat-option__text"><strong>{name}</strong><small>{line}</small></span>
      {id === selectedAgentId && <Icon className="ai-chat-option__check" name="done" size={16} />}
    </Button>
  );
  return (
    <div className="ai-chat-popover-anchor ai-chat-agent-picker" data-agent-state={state} ref={popover.rootRef}>
      <Button
        aria-controls={popover.open ? `ai-chat-agent-list-${ids}` : undefined}
        aria-describedby={busy ? `ai-chat-agent-busy-${ids}` : undefined}
        aria-expanded={popover.open}
        aria-haspopup="listbox"
        aria-label={`Agent ${value}`}
        className={popover.open ? "ai-chat-menu-button is-open" : "ai-chat-menu-button"}
        disabled={busy}
        id="ai-chat-agent"
        onClick={() => popover.setOpen((open) => !open)}
        ref={popover.buttonRef}
        type="button"
      >
        <span className="ai-chat-menu-button__key">Agent</span>
        <span className="ai-chat-menu-button__value">{value}</span>
        <Icon className="ai-chat-menu-button__chevron" name="chevron" size={ICON_SIZE.inline} />
      </Button>
      {/* The reason is drawn beside the held button, not left to a tooltip (VD-200 assist review). */}
      {busy && <span className="ui-button__disabled-reason" id={`ai-chat-agent-busy-${ids}`}>{AGENT_HELD_REASON}</span>}
      {popover.open && (
        <div className="ai-chat-popover ai-chat-popover--agents">
          <h3 className="ai-chat-popover__eyebrow" id={`ai-chat-agent-label-${ids}`}>Agent (optional)</h3>
          <div
            aria-labelledby={`ai-chat-agent-label-${ids}`}
            id={`ai-chat-agent-list-${ids}`}
            onKeyDown={moveOptionFocus}
            role="listbox"
          >
            {option("", "No agent", "The model answers directly")}
            {agents.map((agent) => option(
              agent.id,
              agent.operational === false ? `${agent.name} (unavailable)` : agent.name,
              agent.version ? `Custom agent · version ${agent.version}` : "Custom agent",
              agent.operational === false,
            ))}
          </div>
          <p className="ai-chat-popover__note">{hint}</p>
        </div>
      )}
    </div>
  );
}

export function AiChatToolbar({
  agents,
  agentsResolved,
  agentsReading,
  archived = false,
  busy,
  connection,
  connectionBusy = false,
  connections = [],
  detailsOpen,
  focusMode,
  modelFailures,
  models,
  onActivateConnection,
  onChooseModel,
  onSelectAgent,
  onToggleDetails,
  onToggleFocus,
  clustering,
  selectedAgentId,
  selectedModel,
  subtitle,
  temporary = false,
  thinking,
  title,
}: {
  agents: AiChatAgent[];
  /** False until `/assistant/profiles` has answered once, however it answered. */
  agentsResolved: boolean;
  /** How the agent list read, for the sentence (W4d-D17). */
  agentsReading?: AgentsReading;
  /** The open chat is in the archive. */
  archived?: boolean;
  busy: boolean;
  connection: AiChatConnection | null | undefined;
  /** A connection is being activated. */
  connectionBusy?: boolean;
  connections?: AiChatConnection[];
  detailsOpen: boolean;
  focusMode: boolean;
  modelFailures: Record<string, string>;
  models: string[];
  onActivateConnection?: (connection: AiChatConnection) => void;
  onChooseModel: (model: string) => void;
  onSelectAgent: (id: string) => void;
  onToggleDetails: () => void;
  onToggleFocus: () => void;
  /** VD-210: which row is the cluster, and which rows AI Chat cannot take now. */
  clustering?: AiChatClustering;
  selectedAgentId: string;
  selectedModel: string;
  subtitle: string;
  temporary?: boolean;
  /** VD-209: the thinking control's reading and setter; hidden when absent. */
  thinking?: AiChatThinkingState;
  title: string;
}) {
  const administrator = Boolean(agentsReading?.administrator);
  const state = agentSelectorState(agents, selectedAgentId, agentsResolved, agentsReading);
  const hint = agentSelectionHint(agents, selectedAgentId, agentsResolved, agentsReading);
  return (
    <>
      <header className="ai-chat-toolbar">
        <div className="ai-chat-toolbar__title">
          {/* The page's name for the reader who navigates by headings; the
              chat's own name is the large line a reader sees (the Chat boards). */}
          <h1 className="sr-only">{destinations["ai-chat"].name}</h1>
          <h2 title={title}>
            <span>{title}</span>
            {temporary && <StatusPill className="ai-chat-toolbar__pill" label="Not saved" reading="stale" tone="neutral" />}
            {archived && !temporary && <StatusPill className="ai-chat-toolbar__pill" label="Archived" tone="neutral" />}
          </h2>
          <p>{subtitle}</p>
        </div>
        <div className="ai-chat-toolbar__controls">
          <AiChatModelPicker
            administrator={administrator}
            busy={connectionBusy}
            connection={connection}
            connections={connections}
            modelFailures={modelFailures}
            models={models}
            onActivateConnection={onActivateConnection}
            onChoose={onChooseModel}
            // Only an administrator can add a connection (`POST /credentials`).
            onConnect={administrator ? openAiChatConnectionForm : undefined}
            onManageConnections={administrator ? () => { window.location.hash = "#/admin/connections"; } : undefined}
            clustering={clustering}
            value={selectedModel}
          />
          <AiChatThinkingControl thinking={thinking} />
          <AgentPicker
            agents={agents}
            busy={busy}
            hint={hint}
            onSelect={onSelectAgent}
            selectedAgentId={selectedAgentId}
            state={state}
          />
          <Button
            aria-pressed={focusMode}
            className={focusMode ? "ai-chat-toolbar__focus is-selected" : "ai-chat-toolbar__focus ai-chat-ghost"}
            onClick={onToggleFocus}
            type="button"
            variant="quiet"
          >
            {focusMode ? "Exit focus" : "Focus view"}
          </Button>
          <Button
            aria-pressed={detailsOpen}
            className={detailsOpen ? "ai-chat-toolbar__details is-selected" : "ai-chat-toolbar__details"}
            onClick={onToggleDetails}
            type="button"
          >
            Details
          </Button>
        </div>
      </header>
      {state === "chosen" && <p className="ai-chat-agent-line">{hint}</p>}
    </>
  );
}
