import { type FormEvent, type ReactNode, useRef, useState } from "react";
import { useAssistantChat } from "../hooks/useAssistantChat";
import { useLatestInView } from "../hooks/useLatestInView";
import { continueProposedWorkload } from "../lib/workloadHandoff";
import { destinations } from "../lib/destinations";
import { timeAgo } from "../lib/format";
import type { Session } from "../types";
import { ActionReviewDialog, type ProposedJob } from "./ActionReviewDialog";
import { AssistantApplicationHandoff } from "./AssistantApplicationHandoff";
import { AssistantCapabilityStrip } from "./AssistantCapabilityStrip";
import { AssistantChatComposer, AssistantResponseStatus } from "./AssistantChatComposer";
import { AssistantEngineSummary } from "./AssistantEngineSummary";
import { AssistantModelState } from "./AssistantModelState";
import { AssistantAnswerDestinations, AssistantNextStep } from "./AssistantNextStep";
import { AssistantProposalCard } from "./AssistantProposalCard";
import { ConfirmDialog } from "./ConfirmDialog";
import { CopilotSetup, type CopilotSetupData } from "./CopilotSetup";
import { useMachineProfile } from "../hooks/useMachineProfile";
import { unknownMachine } from "../lib/machine";
import { suggestedAssistantPrompts } from "../lib/assistantPrompts";
import { Icon } from "./Icon";
import { Button, Checkbox, Notice, Select, type NoticeSeverity } from "./ui";
import { StatusPill } from "./StatusPill";
import { TextPromptDialog } from "./TextPromptDialog";
import { assistantSourceLabel } from "./assistantPresentation";
import type {
  AgentProfile,
  AgentRunProposal,
  AgentStatus,
  AssistantEvidence,
} from "./agentTypes";

type IntelligenceChoice = "" | "basic" | "local" | "provider" | null;

export interface ProposalReview {
  job: ProposedJob;
  summary: string;
  evidence: AssistantEvidence[];
  suggestedActions: string[];
}

/** The composer's problem-area refinement defaults to letting Vaelor decide. */
export const AUTOMATIC_PROBLEM_AREA = "";

interface AgentAssistantPanelProps {
  agentStatus: AgentStatus | null;
  busy: boolean;
  chat: ReturnType<typeof useAssistantChat>;
  /** True only while an appliance check is in flight, not for every busy state. */
  checkRunning: boolean;
  /** Epoch ms the in-flight appliance check started, so its counter survives a tab change. */
  checkStartedAt: number;
  durable: boolean;
  intelligenceChoice: IntelligenceChoice;
  /** Reviewed memories in the appliance-wide store, or null when not readable. */
  memoryCount: number | null;
  modelReady: boolean;
  /** False until `/agent/status` has answered at least once. */
  modelStatusResolved: boolean;
  /** Non-empty when a model is configured but its endpoint did not answer. */
  modelUnreachableReason?: string;
  notice: string;
  /** Travels with `notice`: a blocked run must not read like a clean pass. */
  noticeSeverity: NoticeSeverity;
  problemArea: string;
  profiles: AgentProfile[];
  proposalReview: ProposalReview | null;
  session: Session;
  setupData: CopilotSetupData | null;
  showIntelligenceSetup: boolean;
  showSkills: boolean;
  skillCount: number | null;
  skillsPanel: ReactNode;
  onApproveProposal: () => void;
  /** Stops waiting for an appliance check; the run itself continues server-side. */
  onCancelCheck: () => void;
  onChooseIntelligence: (choice: "basic" | "local" | "provider", openSetup?: boolean) => void;
  onCloseIntelligenceSetup: () => void;
  onPrepareAgentRun: (proposal: AgentRunProposal) => void;
  onRefresh: () => void;
  onSubmit: (event: FormEvent) => void;
  onToggleSkills: () => void;
  setDurable: (durable: boolean) => void;
  setProblemArea: (profile: string) => void;
  setProposalReview: (review: ProposalReview | null) => void;
  setShowIntelligenceSetup: (open: boolean | ((current: boolean) => boolean)) => void;
}

/**
 * Ask: one question box, and the answer to it.
 *
 * Ask Vaelor and Troubleshoot were the same question sent to two endpoints, and
 * choosing between them meant knowing which endpoint you wanted. They are one
 * surface now: the problem area is an optional refinement that defaults to
 * automatic, and keeping a run as a re-runnable check is a checkbox rather than
 * a second screen. Nothing routes by keyword — the reader's explicit choice is
 * the only thing that changes which engine answers.
 *
 * The run history left with the History tab. Carrying a live chat and a
 * forty-five-row audit archive on one screen gave the answer 146px against the
 * archive's 1,883, and the refinements moved below the question box because
 * asking somebody to categorise a problem before they have stated it is the
 * wrong order.
 */
export function AgentAssistantPanel(props: AgentAssistantPanelProps) {
  const chat = props.chat;
  const latestRef = useRef<HTMLDivElement>(null);
  const machine = useMachineProfile() ?? unknownMachine;
  /*
   * Only suggest questions this machine can answer, and suggest the ones its
   * class makes worth asking. The list is per-class rather than a single
   * capability filter — see `lib/assistantPrompts.ts`.
   */
  const suggestedPrompts = suggestedAssistantPrompts(machine);
  /*
   * Whether the refinement is showing. Collapsing it used to hide a ticked
   * "Keep this as a check I can re-run" while leaving the submit button reading
   * "Run this check", so the next ordinary question silently became an
   * approval-gated run. A control the reader cannot see must not be the thing
   * that decides what the button does: while the disclosure is closed and the
   * mode is armed, the mode says so in the open, with a way out of it.
   */
  const [refinementOpen, setRefinementOpen] = useState(false);
  // Every append path must reveal itself: the optimistic user echo and the
  // final answer both change the message count, the pending row and the
  // failure notice toggle independently.
  const { following, scrollToLatest } = useLatestInView(
    latestRef,
    [
      chat.chatMessages.length,
      chat.chatMessages.at(-1)?.content,
      chat.chatRequestActive,
      chat.chatNotice,
      chat.conversationId,
    ],
    { enabled: chat.chatMessages.length > 0 },
  );
  const currentConversation = chat.conversations.find((item) => item.id === chat.conversationId);
  const visibleConversations = chat.conversations.filter(
    (conversation) => conversation.archived === (chat.conversationView === "archive"),
  );
  const applianceProfiles = props.profiles.filter((item) => !item.custom);
  const selectedArea = applianceProfiles.find((item) => item.id === props.problemArea);
  const runsCheck = props.durable || props.problemArea !== AUTOMATIC_PROBLEM_AREA;
  const questionTooLong = chat.chatInput.trim().length > 4000;
  const areaUnavailable = Boolean(props.problemArea) && selectedArea?.operational === false;
  const composerBlocked = questionTooLong || areaUnavailable;

  return (
    <div id="ask-panel" role="tabpanel">
      <div className="page-heading agent-heading">
        <div>
          {/*
            * One canonical name per destination: the heading names the place,
            * and what you do here is subordinate to it. Three competing names
            * for the same screen is what made this unnavigable for beginners.
            */}
          <h1>{destinations.assistant.name}</h1>
          <p>{destinations.assistant.descriptor}.</p>
        </div>
        {/*
          * Nothing is claimed about the appliance until the appliance has
          * answered. `/agent/status` resolves about two seconds after paint, so
          * an unconditional pill opened with an amber MODEL REQUIRED and then
          * flipped to green: the first thing the product said to a beginner was
          * a false alarm about their own machine.
          */}
        {/*
          * The pill NAMES the model that backs answers rather than making a
          * generic "Evidence-backed" claim: an owner asked to be told which LLM
          * is answering, and a named model is a truthful, checkable statement
          * where the badge was an assertion about the product. The raw catalog
          * tag leads (e.g. "qwen3.5:4b"); the friendlier capability label is the
          * fallback only when the tag is unknown. When ready but neither is
          * known the pill says "Model connected", never "Evidence-backed" - the
          * generic claim the owner rejected must not return through the
          * fallback. With no model connected the pill says exactly that,
          * degraded. "Checking…" and "Model unreachable" stay: they are honest
          * states the model name must not paper over.
          */}
        <StatusPill
          status={!props.modelStatusResolved ? "neutral" : props.modelUnreachableReason ? "degraded" : props.modelReady ? "healthy" : "degraded"}
          label={
            !props.modelStatusResolved
              ? "Checking…"
              : props.modelUnreachableReason
                ? "Model unreachable"
                : props.modelReady
                  ? (props.agentStatus?.model || props.agentStatus?.capability?.label || "Model connected")
                  : "No model connected"
          }
        />
      </div>

      {props.modelStatusResolved && props.agentStatus?.model_facts && (
        <AssistantModelState model={props.agentStatus.model_facts} />
      )}

      {props.modelUnreachableReason && (
        <Notice severity="warning">
          <span>
            <strong>The selected AI model is not answering.</strong> {props.modelUnreachableReason}
            {" "}Appliance checks still run using built-in read-only diagnostics, but answers will
            not use the model until it is reachable.{" "}
            {/*
              * The Assistant and AI Chat are separate engines with separate
              * availability, which is why one can look broken while the other
              * works. Saying so here is cheaper than the user discovering it.
              */}
            {destinations["ai-chat"].name} uses its own separately configured model and may still
            be working.
          </span>
        </Notice>
      )}

      {!props.agentStatus?.configured && props.intelligenceChoice === "" && props.setupData && (
        <section className="assistant-first-run" aria-labelledby="assistant-first-run-title">
          <div className="assistant-first-run__intro">
            <span className="assistant-first-run__icon"><Icon name="bolt" size={28} /></span>
            <div>
              <span className="page-eyebrow">First-time assistant setup</span>
              <h2 id="assistant-first-run-title">How smart should Vaelor be?</h2>
              <p>Choose once now. You can change this later without losing chats or appliance settings.</p>
            </div>
          </div>
          <div className="assistant-first-run__choices">
            <Button
              className="assistant-first-run__choice assistant-first-run__choice--recommended"
              disabled={props.busy || !props.setupData.recommendation.can_install}
              onClick={() => props.onChooseIntelligence("local")}
              type="button"
              variant="quiet"
            >
              <span>Recommended</span>
              <strong>Install {props.setupData.recommendation.primary.name}</strong>
              {/*
                * The size of the model named beside it. This read "about
                * 1.1 GB" whatever was recommended, which is the sentence that
                * made *"Install Qwen3 32B / about 1.1 GB"* internally
                * contradictory on the Z2 — a 32B at Q4 is about 20 GB, so the
                * size was the only honest figure of the three shown. It is the
                * catalog entry's own note now, derived from the byte count the
                * fit check divides by, so the name and the size cannot part
                * company again.
                */}
              <small>Private local answers · {props.setupData.recommendation.primary.size_note} · reviewed before download</small>
            </Button>
            <Button
              className="assistant-first-run__choice"
              disabled={props.busy || props.session.user.role !== "administrator"}
              onClick={() => props.onChooseIntelligence("provider")}
              type="button"
              variant="quiet"
            >
              <span>Bring your own AI</span>
              <strong>Connect another model</strong>
              <small>OpenAI, LM Studio, Lemonade, llama.cpp, or another compatible endpoint</small>
            </Button>
            <Button
              className="assistant-first-run__choice"
              disabled={props.busy}
              onClick={() => props.onChooseIntelligence("basic")}
              type="button"
              variant="quiet"
            >
              <span>No download</span>
              <strong>Use built-in basic mode</strong>
              <small>Code-based live appliance answers and diagnostics; broader questions stay limited</small>
            </Button>
          </div>
        </section>
      )}

      {props.intelligenceChoice !== "" && (
        <AssistantEngineSummary
          onToggle={() => props.setShowIntelligenceSetup((current) => !current)}
          settingsOpen={props.showIntelligenceSetup}
          status={props.agentStatus}
        />
      )}

      {props.showIntelligenceSetup && props.setupData && (
        <CopilotSetup
          busy={props.busy}
          data={props.setupData}
          onChooseLocal={() => {
            props.onChooseIntelligence("local", false);
            props.setShowIntelligenceSetup(false);
            chat.setChatNotice("Opening Workloads so you can review a hardware-matched local model.");
            window.dispatchEvent(new CustomEvent("pironman:navigate", { detail: "workloads" }));
          }}
          onInstallNpuRelease={() => {
            // The on-device NPU model is installed from Workloads too, where the
            // review-and-install lives; route there rather than installing from
            // the first-run panel.
            props.onChooseIntelligence("local", false);
            props.setShowIntelligenceSetup(false);
            chat.setChatNotice("Opening Workloads so you can review and install the on-device model.");
            window.dispatchEvent(new CustomEvent("pironman:navigate", { detail: "workloads" }));
          }}
          onClose={props.onCloseIntelligenceSetup}
          onIntelligenceConnected={() => {
            props.onChooseIntelligence("provider", false);
            props.onRefresh();
          }}
          session={props.session}
        />
      )}

      <section className="assistant-chat" aria-labelledby="assistant-chat-title">
        <div className="assistant-chat__toolbar">
          <div>
            <strong>{currentConversation?.title || "New chat"}</strong>
            <small><Icon name="database" /> Saved automatically on this Vaelor node</small>
          </div>
          <div className="assistant-chat__toolbar-actions">
            <Button onClick={chat.startNewChat} type="button" variant="quiet">New chat</Button>
            <Button
              aria-expanded={chat.showChatHistory}
              onClick={() => chat.setShowChatHistory(!chat.showChatHistory)}
              type="button"
              variant="quiet"
            >
              {/*
                * Not "History". The tablist above owns that word for the run
                * archive, and two controls with one name on one screen sent a
                * reader looking for a prepared agent run into their saved
                * conversations. The tab name is the structural one, so this is
                * the one that changes.
                */}
              Past chats{chat.conversations.length ? " (" + chat.conversations.length + ")" : ""}
            </Button>
            {chat.conversationId && (
              <details className="assistant-chat__menu">
                <summary className="ui-button ui-button--quiet">More</summary>
                <Button onClick={chat.renameConversation} type="button" variant="quiet">Rename</Button>
                <Button onClick={() => void chat.exportConversation()} type="button" variant="quiet">Export Markdown</Button>
                <Button onClick={() => void chat.archiveConversation(!currentConversation?.archived)} type="button" variant="quiet">{currentConversation?.archived ? "Restore chat" : "Archive chat"}</Button>
                <Button className="danger-text" onClick={() => chat.setConfirmChatDelete(true)} type="button" variant="danger">Delete chat</Button>
              </details>
            )}
          </div>
        </div>
        {chat.showChatHistory && (
          <div className="assistant-chat__history" aria-label="Saved chats">
            <div>
              <strong>{chat.conversationView === "archive" ? "Archived chats" : "Saved chats"}</strong>
              <small>{chat.conversationView === "archive" ? "Open, export, restore, or delete an archived chat." : "Select one to continue where you left off."}</small>
              <Button
                onClick={() => chat.setConversationView(chat.conversationView === "active" ? "archive" : "active")}
                type="button"
                variant="quiet"
              >
                {chat.conversationView === "active" ? "View archive" : "Back to saved chats"}
              </Button>
            </div>
            {visibleConversations.length ? visibleConversations.map((conversation) => (
              <Button
                aria-current={conversation.id === chat.conversationId ? "true" : undefined}
                className="assistant-chat__history-item"
                key={conversation.id}
                onClick={() => void chat.openConversation(conversation)}
                type="button"
                variant="quiet"
              >
                <span><strong>{conversation.title}</strong><small>{conversation.message_count} messages · {timeAgo(conversation.updated_at * 1000)}</small></span>
                <Icon name="chevron" />
              </Button>
            )) : <p>{chat.conversationView === "archive" ? "No archived chats." : "No saved chats yet. Your first message starts one automatically."}</p>}
          </div>
        )}
        {/*
          * No inner scroller and no fixed height. The page scrolls; the
          * transcript is as tall as the answer it is showing.
          */}
        <div className="assistant-chat__stream" role="log" aria-live="polite">
          {chat.chatMessages.length === 0 && (
            <div className="assistant-chat__welcome">
              <span><Icon name="bolt" /></span>
              <div>
                <h2 id="assistant-chat-title">What do you want to know?</h2>
                {/*
                  * These are the only invitation the machine-reading tools
                  * get. The Assistant can read both compute engines and the
                  * suggestions were built around the enclosure, so on a
                  * workstation nothing on screen led anywhere near them.
                  */}
                <p>Try {suggestedPrompts.map((prompt, index) => (
                  <span key={prompt}>{index ? (index === suggestedPrompts.length - 1 ? ", or " : ", ") : ""}“{prompt}”</span>
                ))}</p>
              </div>
            </div>
          )}
          {chat.chatMessages.map((item, index) => (
            <article
              className={`assistant-message assistant-message--${item.role}${item.metadata?.stopped ? " assistant-message--stopped" : ""}`}
              key={item.id ?? `${item.role}-${index}`}
            >
              <div className="assistant-message__meta">
                {/*
                  * A stopped response was not Vaelor speaking, so it is not
                  * signed as if it were. The byline is the terminal state.
                  */}
                <strong>{item.role === "user" ? "You" : item.metadata?.stopped ? "Response stopped" : "Vaelor"}</strong>
                {item.metadata?.source && <span>{assistantSourceLabel(item.metadata.source, props.agentStatus)}</span>}
              </div>
              {item.content.split("\n").filter(Boolean).map((paragraph, paragraphIndex) => (
                <p key={paragraphIndex}>{paragraph}</p>
              ))}
              {item.metadata?.evidence?.length ? (
                <details className="assistant-evidence">
                  <summary>Evidence used · {item.metadata.evidence.length} source{item.metadata.evidence.length === 1 ? "" : "s"}</summary>
                  <ul>
                    {item.metadata.evidence.map((evidence) => (
                      <li key={`${evidence.source}-${evidence.summary}`}>
                        <strong>{evidence.source.replaceAll(".", " ")}</strong>
                        <span>{evidence.summary}</span>
                      </li>
                    ))}
                  </ul>
                </details>
              ) : null}
              {item.metadata?.suggested_actions?.length ? (
                <div className="assistant-next-actions">
                  <strong>Suggested next steps</strong>
                  <ul>{item.metadata.suggested_actions.map((action) => <li key={action}><AssistantNextStep action={action} /></li>)}</ul>
                </div>
              ) : null}
              {/*
                * The routes the answer's destinations resolve to. The
                * suggested steps above link a place they happen to name in a
                * sentence; this is the answer's own machine-readable list, so
                * a redirect that names AI Chat only in its prose — which is
                * every model-authored refusal — still reaches it.
                */}
              <AssistantAnswerDestinations steps={item.metadata?.next_steps} />
              {item.metadata?.proposed_job && !item.metadata?.application_intent ? (
                <AssistantProposalCard
                  busy={props.busy}
                  job={item.metadata.proposed_job}
                  onContinue={() => continueProposedWorkload(item.metadata!.proposed_job!)}
                  onReview={() => props.setProposalReview({
                    job: item.metadata!.proposed_job!,
                    summary: item.content,
                    evidence: item.metadata?.evidence ?? [],
                    suggestedActions: item.metadata?.suggested_actions ?? [],
                  })}
                />
              ) : null}
              {item.metadata?.application_intent ? (
                <AssistantApplicationHandoff intent={item.metadata.application_intent} />
              ) : null}
              {item.metadata?.proposed_agent_task ? (
                <section className="assistant-proposal-card" aria-label="Custom agent run proposal">
                  <div>
                    <small>Custom agent · version {item.metadata.proposed_agent_task.profile_version}</small>
                    <strong>{item.metadata.proposed_agent_task.profile_name}</strong>
                    <p>{item.metadata.proposed_agent_task.task}</p>
                    <small>{item.metadata.proposed_agent_task.capabilities.length ? `Granted capabilities: ${item.metadata.proposed_agent_task.capabilities.join(", ")}` : "No appliance capabilities granted"}</small>
                    <small>{item.metadata.proposed_agent_task.integrations?.length ? `API integrations: ${item.metadata.proposed_agent_task.integrations.join(", ")}` : "No API integrations granted"}</small>
                  </div>
                  <Button disabled={props.busy} onClick={() => props.onPrepareAgentRun(item.metadata!.proposed_agent_task!)} type="button" variant="primary">Review agent run</Button>
                </section>
              ) : null}
            </article>
          ))}
          <AssistantResponseStatus
            active={chat.chatRequestActive}
            onCancel={chat.cancelRequest}
            startedAt={chat.requestStartedAt}
          />
          {/*
            * A question whose answer was still being written when the page
            * reloaded. The appliance finishes it server-side, so the transcript
            * showed the question with no answer, no spinner and no error until
            * the reader navigated away and back — and in the meantime they
            * re-asked, leaving two identical questions and two answers.
            */}
          {chat.awaitingAnswer && !chat.chatRequestActive && (
            <p className="assistant-resumed-wait" role="status">
              Your last question is still being answered on this appliance. The reply appears
              here as soon as it lands — you do not need to ask again.
            </p>
          )}
          {chat.awaitingAnswerLost && (
            <p className="assistant-resumed-wait assistant-resumed-wait--lost" role="status">
              No answer arrived for your last question, and Vaelor has stopped waiting for it.
              Nothing was changed. Ask it again when you are ready.
            </p>
          )}
          {/* The end of the transcript, and the thing "Jump to latest" jumps to
              now that the page rather than the stream is the scroller. */}
          <div className="assistant-chat__latest-anchor" ref={latestRef} />
        </div>
        {!following && chat.chatMessages.length > 0 && (
          <div className="assistant-chat__resume">
            <Button onClick={() => scrollToLatest("smooth")} type="button" variant="quiet">
              <Icon name="chevron" /> Jump to latest
            </Button>
          </div>
        )}
        {/*
          * Validation lives beside the composer, never inside the disclosure.
          *
          * Both sentences were moved into the closed "Save this as a check"
          * details, so the reader was left with a disabled button and no
          * readable reason: the one thing that could explain the dead end was
          * folded behind a control they had no reason to open.
          */}
        {questionTooLong && <p className="field-error" role="alert">Keep the question to 4,000 characters or fewer.</p>}
        {areaUnavailable && <p className="field-error" role="alert">That problem area is unavailable on this appliance. Choose another, or leave it on Automatic.</p>}

        <AssistantChatComposer
          blocked={composerBlocked}
          busy={chat.chatBusy || props.busy}
          input={chat.chatInput}
          onChange={chat.setChatInput}
          onSubmit={props.onSubmit}
          submitLabel={runsCheck ? "Run this check" : "Ask Vaelor"}
        />
        {/*
          * The refinement, not a mode switch, and it sits after the question.
          *
          * Both controls default to the do-nothing value, and both used to sit
          * above the composer — so the first thing the product asked a beginner
          * to do was categorise a problem they had not stated yet. Same
          * controls, same copy, same behaviour; they are simply downstream of
          * the sentence they refine. The durable record is the best artefact
          * this product produces, so it stays one click away rather than on a
          * separate screen.
          */}
        {/*
          * The armed mode, stated where the disclosure cannot hide it.
          *
          * Collapsing the refinement left the checkbox ticked out of sight while
          * the button still read "Run this check": the next ordinary question
          * became an approval-gated run with nothing on screen saying so. The
          * arming is only ever invisible if it is not armed.
          */}
        {runsCheck && !refinementOpen && (
          <p className="assistant-refinement-armed" role="status">
            <span>
              <strong>This will run as an appliance check, not a chat answer.</strong>
              {" "}It needs your approval and its evidence is kept under History
              {selectedArea ? ` · ${selectedArea.name}` : ""}
              {props.durable ? " · saved to re-run" : ""}.
            </span>
            <Button
              onClick={() => { props.setDurable(false); props.setProblemArea(AUTOMATIC_PROBLEM_AREA); }}
              type="button"
              variant="quiet"
            >
              Ask a normal question instead
            </Button>
          </p>
        )}
        <details
          className="assistant-refinement-disclosure"
          onToggle={(event) => setRefinementOpen(event.currentTarget.open)}
        >
          <summary>{runsCheck ? "Saved as a check I can re-run · on" : "Save this as a check I can re-run"}</summary>
          <div className="assistant-refinement">
            <Select
              hint="Optional. Automatic lets Vaelor answer from live readings."
              id="assistant-problem-area"
              label="Problem area"
              onChange={(event) => props.setProblemArea(event.target.value)}
              value={props.problemArea}
            >
              <option value={AUTOMATIC_PROBLEM_AREA}>Automatic</option>
              {applianceProfiles.map((item) => (
                <option disabled={!item.operational} key={item.id} value={item.id}>
                  {item.name}{item.operational ? "" : " unavailable"}
                </option>
              ))}
            </Select>
            <Checkbox
              checked={props.durable}
              hint="Saves the question, its evidence, and its result as a run you can approve and repeat."
              id="assistant-durable"
              label="Keep this as a check I can re-run"
              onChange={(event) => props.setDurable(event.target.checked)}
            />
            <p className="assistant-refinement__outcome">
              {runsCheck
                ? props.durable
                  ? `Runs as a saved appliance check on ${selectedArea?.name ?? "system health"}. You approve it before it runs, and its evidence stays under History.`
                  : `Runs a one-off appliance check on ${selectedArea?.name ?? "system health"} and records its evidence under History.`
                : "Answers in this chat from live readings. Any change it proposes still needs a separate approval."}
            </p>
          </div>
        </details>
        {/*
          * Scoped to the check itself, not to every busy state: an appliance
          * check takes about a minute and the button label alone gave no sign
          * of life, but "Checking this appliance" over a skill review is a lie.
          */}
        <AssistantResponseStatus
          active={props.checkRunning}
          label="Checking this appliance"
          onCancel={props.onCancelCheck}
          startedAt={props.checkStartedAt}
        />
        {/*
          * Below the composer, never above it.
          *
          * A banner inserted here used to open above the question box, pushing
          * it down by its own height and pulling it back up again when it was
          * replaced — so a reader who clicked where the box had just been typed
          * a whole sentence into the hint strip under it and lost every
          * character with no error and no counter movement. Nothing that
          * appears and disappears on its own is allowed above the composer.
          */}
        {chat.chatNotice && <Notice severity="info">{chat.chatNotice}</Notice>}
        {/*
          * The skills disclosure reports the same `notice` with a link to the
          * proposal it just created, so showing it here as well would print the
          * outcome twice on one screen.
          */}
        {props.notice && !props.showSkills && <Notice severity={props.noticeSeverity}>{props.notice}</Notice>}
        <AssistantCapabilityStrip
          memory={props.memoryCount === null ? undefined : { count: props.memoryCount, href: "#/memory" }}
          model={props.agentStatus?.model || props.agentStatus?.provider || "No model selected"}
          scope={{ label: "This machine", detail: "Live readings only" }}
          skills={props.skillCount === null ? undefined : {
            count: props.skillCount,
            expanded: props.showSkills,
            onToggle: props.onToggleSkills,
          }}
        />
      </section>

      {props.showSkills && props.skillsPanel}

      <TextPromptDialog open={chat.renameTitle !== null} title="Rename chat" description="Give this saved conversation a short, recognizable name." label="Chat name" value={chat.renameTitle ?? ""} busy={chat.chatBusy} onChange={chat.setRenameTitle} onCancel={() => chat.setRenameTitle(null)} onSubmit={() => void chat.saveConversationTitle()} />
      {/*
        * "Permanently removes ... every message" alone stopped being true in
        * Alpha 46: a recent conversation may also have a fast-wake snapshot
        * on disk, and deleting the conversation retires it (nothing can
        * restore it again) without erasing its bytes until a later save
        * reuses that slot. The sentence says so rather than overclaiming.
        */}
      <ConfirmDialog open={chat.confirmChatDelete} title="Delete saved chat?" description="This permanently deletes the conversation and every message in it. This cannot be undone. If it had a saved fast-wake snapshot, that snapshot is retired too, though its data may remain on this appliance's disk until a later save reuses that space." confirmLabel="Delete chat" busy={chat.chatBusy} onCancel={() => chat.setConfirmChatDelete(false)} onConfirm={() => void chat.deleteConversation()} />
      <ActionReviewDialog
        busy={props.busy}
        evidence={props.proposalReview?.evidence}
        job={props.proposalReview?.job ?? null}
        onApprove={props.onApproveProposal}
        onCancel={() => props.setProposalReview(null)}
        suggestedActions={props.proposalReview?.suggestedActions}
        summary={props.proposalReview?.summary ?? ""}
      />
    </div>
  );
}
