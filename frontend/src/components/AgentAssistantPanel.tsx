import { type FormEvent, type ReactNode, useEffect, useRef, useState } from "react";
import { useAssistantChat } from "../hooks/useAssistantChat";
import { useLatestInView } from "../hooks/useLatestInView";
import { destinations } from "../lib/destinations";
import type { Session } from "../types";
import { ActionReviewDialog, type ProposedJob } from "./ActionReviewDialog";
import { AssistantAskFirstRun, AssistantAskFirstRunLoading, AssistantAskSuggestions } from "./AssistantAskFirstRun";
import { AssistantBarActions, useAssistantPlace } from "./assistantBar";
import { AssistantCapabilityStrip } from "./AssistantCapabilityStrip";
import { AssistantChatComposer, AssistantResponseStatus } from "./AssistantChatComposer";
import { AssistantConversationBar } from "./AssistantConversationBar";
import { AssistantEngineSummary } from "./AssistantEngineSummary";
import { AssistantIntelligenceDrawer } from "./AssistantIntelligenceDrawer";
import { AssistantMessage } from "./AssistantMessage";
import { AssistantModelSleepNote, AssistantModelState } from "./AssistantModelState";
import { ASSISTANT_TAB_PLACES } from "./AssistantNavigationTabs";
import { ConfirmDialog } from "./ConfirmDialog";
import { CopilotSetup, type CopilotSetupData } from "./CopilotSetup";
import { NpuInstallDialog } from "./NpuInstallDialog";
import { useNpuInstall } from "../hooks/useNpuInstall";
import { useMachineProfile } from "../hooks/useMachineProfile";
import { unknownMachine } from "../lib/machine";
import { suggestedAssistantPrompts } from "../lib/assistantPrompts";
import { Icon } from "./Icon";
import { Button, Checkbox, Notice, Select, type NoticeSeverity } from "./ui";
import { StatusPill } from "./StatusPill";
import { NOT_ANSWERING } from "./ui/status";
import { TextPromptDialog } from "./TextPromptDialog";
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
  /**
   * Reviewed memories in the appliance-wide store: a number when read, `null`
   * when it could not be read (the chip then says "Not read", never 0), and
   * `undefined` for a reader who may not open Memory (the chip is not shown).
   */
  memoryCount: number | null | undefined;
  modelReady: boolean;
  /** False until `/agent/status` has answered at least once. */
  modelStatusResolved: boolean;
  /**
   * Why the Assistant's state could not be read (empty once a read lands). The
   * loading card gives way to this and a Retry, never spinning on (VD-200).
   */
  statusReadError?: string;
  onRetryStatus?: () => void;
  /** Non-empty when a model is configured but its endpoint did not answer. */
  modelUnreachableReason?: string;
  /** Non-empty when the endpoint answers but offers no model to use. */
  modelNotOfferedReason?: string;
  notice: string;
  /** Travels with `notice`: a blocked run must not read like a clean pass. */
  noticeSeverity: NoticeSeverity;
  problemArea: string;
  profiles: AgentProfile[];
  proposalReview: ProposalReview | null;
  /** Why approving the reviewed action was refused (VD-189): shown inside the review. */
  proposalError: string;
  session: Session;
  setupData: CopilotSetupData | null;
  showIntelligenceSetup: boolean;
  showSkills: boolean;
  /** Active skills: a number when read, `null` when the read failed ("Not read"), `undefined` before a read or for a non-administrator (no chip). */
  skillCount: number | null | undefined;
  skillsPanel: ReactNode;
  onApproveProposal: () => void;
  /** Stops waiting for an appliance check; the run itself continues server-side. */
  onCancelCheck: () => void;
  onChooseIntelligence: (choice: "basic" | "local", openSetup?: boolean) => void;
  onCloseIntelligenceSetup: () => void;
  onPrepareAgentRun: (proposal: AgentRunProposal) => void;
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
/** The page line that says why Change intelligence is off until the state is read. */
const ASSISTANT_STATE_REASON_ID = "assistant-state-reason";

/** The skills view's place in the top bar breadcrumb, after Ask's. */
export const SKILLS_PLACE = "Skills";

export function AgentAssistantPanel(props: AgentAssistantPanelProps) {
  const chat = props.chat;
  // Same on-device install + live status the Workloads flow uses, so the
  // first-run panel's "Set up the on-device Assistant" actually sets it up here
  // rather than navigating to Workloads and dropping the intent.
  const npuInstall = useNpuInstall(props.session, chat.setChatNotice);
  // "Assistant / Ask about this machine", and "/ Skills" while the skills view is open (the AssistSkills board).
  useAssistantPlace([ASSISTANT_TAB_PLACES.ask, ...(props.showSkills ? [SKILLS_PLACE] : [])]);
  const latestRef = useRef<HTMLDivElement>(null);
  const machine = useMachineProfile() ?? unknownMachine;
  /*
   * Only suggest questions this machine can answer, and suggest the ones its
   * class makes worth asking. The list is per-class rather than a single
   * capability filter — see `lib/assistantPrompts.ts`.
   */
  const suggestedPrompts = suggestedAssistantPrompts(machine);
  const [moreOpen, setMoreOpen] = useState(false);
  /*
   * The skills chip is drawn on top of the skills view and under the chat, so
   * toggling it remounts the button and focus fell to the page body. A toggle
   * from the chip puts focus on the chip that is drawn next (VD-200 review).
   */
  const panelRef = useRef<HTMLDivElement>(null);
  const skillsToggled = useRef(false);
  useEffect(() => {
    if (!skillsToggled.current) return;
    skillsToggled.current = false;
    panelRef.current?.querySelector<HTMLElement>(".as-strip button[aria-expanded]")?.focus();
  }, [props.showSkills]);
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
  const applianceProfiles = props.profiles.filter((item) => !item.custom);
  const selectedArea = applianceProfiles.find((item) => item.id === props.problemArea);
  const runsCheck = props.durable || props.problemArea !== AUTOMATIC_PROBLEM_AREA;
  const questionTooLong = chat.chatInput.trim().length > 4000;
  const areaUnavailable = Boolean(props.problemArea) && selectedArea?.operational === false;
  const composerBlocked = questionTooLong || areaUnavailable;
  const firstRun = !props.agentStatus?.configured && props.intelligenceChoice === "";

  /*
   * Nothing is claimed about the appliance until the appliance has answered.
   * `/agent/status` resolves about two seconds after paint, so an
   * unconditional pill opened with an amber MODEL REQUIRED and then flipped to
   * green: the first thing the product said to a beginner was a false alarm
   * about their own machine.
   *
   * The pill NAMES the model that backs answers rather than making a generic
   * "Evidence-backed" claim: an owner asked to be told which LLM is answering,
   * and a named model is a truthful, checkable statement where the badge was an
   * assertion about the product. The raw catalog tag leads (e.g. "qwen3.5:4b");
   * the friendlier capability label is the fallback only when the tag is
   * unknown. When ready but neither is known the pill says "Model connected",
   * never "Evidence-backed" - the generic claim the owner rejected must not
   * return through the fallback. With no model connected the pill says exactly
   * that, degraded. "Checking…" and "Model unreachable" stay: they are honest
   * states the model name must not paper over.
   */
  const modelPill = (
    <StatusPill
      reading={!props.modelStatusResolved && props.statusReadError ? "unread" : undefined}
      status={!props.modelStatusResolved ? "neutral" : props.modelUnreachableReason || props.modelNotOfferedReason ? "degraded" : props.modelReady ? "healthy" : "degraded"}
      label={
        !props.modelStatusResolved
          ? (props.statusReadError ? "Not read" : "Checking…")
          : props.modelUnreachableReason
            ? "Model unreachable"
            : props.modelNotOfferedReason
              ? "No model available"
              : props.modelReady
                ? (props.agentStatus?.model || props.agentStatus?.capability?.label || "Model connected")
                : "No model connected"
      }
    />
  );
  // The Active intelligence card's own word for the same state, in the drawer.
  const answeringPill = props.modelUnreachableReason || props.modelNotOfferedReason
    ? <StatusPill label={NOT_ANSWERING.label} tone={NOT_ANSWERING.tone} />
    : props.modelReady
      ? <StatusPill label="Answering" tone="success" />
      : <StatusPill label="Built-in help" tone="neutral" />;

  const capabilityStrip = (
    <AssistantCapabilityStrip
      memory={props.memoryCount === undefined ? undefined : { count: props.memoryCount, href: "#/memory" }}
      model={props.agentStatus?.model || props.agentStatus?.provider || "No model selected"}
      scope={{ label: "This machine", detail: "Live readings only" }}
      skills={props.skillCount === undefined ? undefined : {
        count: props.skillCount,
        expanded: props.showSkills,
        onToggle: () => { skillsToggled.current = true; props.onToggleSkills(); },
      }}
    />
  );

  return (
    <div className="as-panel as-ask" id="ask-panel" ref={panelRef} role="tabpanel">
      {/*
        * One canonical name per destination. The board draws no heading on Ask
        * - the top bar names the page - so the level-one heading is for the
        * reader who navigates by headings.
        */}
      <h1 className="sr-only">{destinations.assistant.name}</h1>
      {/* The limit that separates this destination from AI Chat; the top bar's "Ask about this machine" says its first half. */}
      <p className="sr-only">{destinations.assistant.descriptor}.</p>
      <AssistantBarActions>
        {modelPill}
        {/* Change intelligence opens the drawer; the first run is its own choice, so it is not offered there. */}
        {props.intelligenceChoice !== "" && (
          <Button
            aria-expanded={props.showIntelligenceSetup}
            disabled={props.agentStatus === null}
            // Its reason is the page's own line below (the failure, or the loading
            // card), not a second line in the top-bar slot, which wrapped the bar.
            aria-describedby={props.agentStatus === null ? ASSISTANT_STATE_REASON_ID : undefined}
            onClick={() => props.setShowIntelligenceSetup((current) => !current)}
            type="button"
          >
            Change intelligence
          </Button>
        )}
      </AssistantBarActions>

      {props.modelNotOfferedReason && (
        <Notice severity="warning">
          <span>
            <strong>The model server has no model to answer with.</strong> {props.modelNotOfferedReason}
            {" "}Appliance checks still run using built-in read-only diagnostics. Load a model on
            that server, then ask again.
          </span>
        </Notice>
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

      {!props.modelStatusResolved && (props.statusReadError ? (
        <Notice severity="danger">
          <span className="ar-notice-row">
            <span id={ASSISTANT_STATE_REASON_ID}><strong>The Assistant's state could not be read.</strong> {props.statusReadError}</span>
            {props.onRetryStatus && <Button onClick={props.onRetryStatus} type="button">Retry</Button>}
          </span>
        </Notice>
      ) : <AssistantAskFirstRunLoading labelId={ASSISTANT_STATE_REASON_ID} />)}
      {firstRun && props.setupData && (
        <AssistantAskFirstRun
          busy={props.busy}
          onChoose={(choice) => props.onChooseIntelligence(choice)}
          setupData={props.setupData}
        />
      )}

      {props.showIntelligenceSetup && (
        <AssistantIntelligenceDrawer
          active={(
            <AssistantEngineSummary pill={answeringPill} status={props.agentStatus}>
              {props.agentStatus?.model_facts && <AssistantModelState model={props.agentStatus.model_facts} />}
            </AssistantEngineSummary>
          )}
          onClose={props.onCloseIntelligenceSetup}
          setup={props.setupData ? (
            <CopilotSetup
              busy={props.busy}
              data={props.setupData}
              onChooseLocal={() => {
                props.onChooseIntelligence("local", false);
                props.setShowIntelligenceSetup(false);
                chat.setChatNotice(`Opening ${destinations.workloads.name} so you can review a hardware-matched local model.`);
                window.dispatchEvent(new CustomEvent("pironman:navigate", { detail: "workloads" }));
              }}
              onInstallNpuRelease={(tag) => {
                // Start the on-device install right here and show its live status,
                // rather than navigating to Workloads and making the user find and
                // click the same control again.
                props.onChooseIntelligence("local", false);
                props.setShowIntelligenceSetup(false);
                npuInstall.start(tag);
              }}
              onClose={props.onCloseIntelligenceSetup}
              onChooseBasic={() => props.onChooseIntelligence("basic")}
              session={props.session}
              showHeader={false}
            />
          ) : (
            <Notice severity="info" standing>The setup choices could not be read. Close this and open it again to retry.</Notice>
          )}
        />
      )}

      {npuInstall.job && (
        <NpuInstallDialog job={npuInstall.job} onDismiss={npuInstall.dismiss} onRetryRead={npuInstall.retryRead} readLost={npuInstall.readLost} />
      )}

      {props.showSkills ? (
        /*
         * Skills is a view inside Ask (the AssistSkills board): the chips stay
         * on top, so the "Hide" that opened it is the way back, and the
         * conversation waits underneath unchanged.
         */
        <>
          {capabilityStrip}
          {props.skillsPanel}
        </>
      ) : (
        <section
          // Named by whichever heading is drawn: the welcome while the transcript
          // is empty, else the conversation bar's title. Keyed on firstRun, the
          // first run pointed at a bar that was not drawn (VD-200 review).
          aria-labelledby={chat.chatMessages.length === 0 ? "assistant-chat-title" : "assistant-chat-conversation"}
          className="as-chat"
        >
          {/* The welcome heading names the chat only while it is shown; an open
              conversation is named by its own title (UX-A4: the section once
              pointed at a heading that was gone). */}
          {(chat.chatMessages.length > 0 || chat.conversations.length > 0 || !firstRun) && (
            <AssistantConversationBar chat={chat} moreOpen={moreOpen} setMoreOpen={setMoreOpen} />
          )}
          {/*
            * No inner scroller and no fixed height. The page scrolls; the
            * transcript is as tall as the answer it is showing.
            */}
          <div aria-live="polite" className="as-stream" role="log">
            {chat.chatMessages.length === 0 && (
              <AssistantAskSuggestions onPick={chat.setChatInput} prompts={suggestedPrompts} />
            )}
            {chat.chatMessages.map((item, index) => (
              <AssistantMessage
                busy={props.busy}
                item={item}
                key={item.id ?? `${item.role}-${index}`}
                onPrepareAgentRun={props.onPrepareAgentRun}
                onReview={props.setProposalReview}
              />
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
              <p className="as-wait" role="status">
                Your last question is still being answered on this appliance. The reply appears
                here as soon as it lands — you do not need to ask again.
              </p>
            )}
            {/* The end of the transcript, and the thing "Jump to latest" jumps to
                now that the page rather than the stream is the scroller. */}
            <div className="as-stream__anchor" ref={latestRef} />
          </div>
          {!following && chat.chatMessages.length > 0 && (
            <div className="as-jump">
              <Button onClick={() => scrollToLatest("smooth")} type="button">
                <Icon name="chevron" size={16} /> Jump to latest
              </Button>
            </div>
          )}

          <AssistantChatComposer
            blocked={composerBlocked}
            busy={chat.chatBusy || props.busy}
            input={chat.chatInput}
            onChange={chat.setChatInput}
            onSubmit={props.onSubmit}
            submitLabel={runsCheck ? "Run this check" : "Ask Vaelor"}
          >
            {/*
              * The armed mode, stated where the disclosure cannot hide it.
              *
              * Collapsing the refinement left the checkbox ticked out of sight
              * while the button still read "Run this check": the next ordinary
              * question became an approval-gated run with nothing on screen
              * saying so. The arming is only ever invisible if it is not armed.
              */}
            {runsCheck && !refinementOpen && (
              <div className="as-armed" role="status">
                <span>
                  <strong>This will run as an appliance check, not a chat answer.</strong>
                  {" "}It needs your approval and its evidence is kept under History
                  {selectedArea ? ` · ${selectedArea.name}` : ""}
                  {props.durable ? " · saved to re-run" : ""}.
                </span>
                <Button
                  className="as-btn-ghost"
                  onClick={() => { props.setDurable(false); props.setProblemArea(AUTOMATIC_PROBLEM_AREA); }}
                  type="button"
                >
                  Ask a normal question instead
                </Button>
              </div>
            )}
            {/*
              * The refinement, not a mode switch, and it sits after the
              * question. Both controls default to the do-nothing value; asking a
              * beginner to categorise a problem they had not stated yet was the
              * wrong order. The durable record is the best artefact this product
              * produces, so it stays one click away rather than on a separate
              * screen.
              */}
            <details
              className="as-refinement"
              onToggle={(event) => setRefinementOpen(event.currentTarget.open)}
            >
              <summary>
                <span>{runsCheck ? "Save this as a check I can re-run · on" : "Save this as a check I can re-run"}</span>
                <Icon name="chevron" size={16} />
              </summary>
              <div className="as-refinement__grid">
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
              </div>
              <p className="as-small as-muted">
                {runsCheck
                  ? props.durable
                    ? `Runs as a saved appliance check on ${selectedArea?.name ?? "system health"}. You approve it before it runs, and its evidence stays under History.`
                    : `Runs a one-off appliance check on ${selectedArea?.name ?? "system health"} and records its evidence under History.`
                  : "Answers in this chat from live readings. Any change it proposes still needs a separate approval."}
              </p>
            </details>
          </AssistantChatComposer>
          {/* Why the first question after a quiet spell is slow, stated where it is asked (VD-073; owner, 2026-10-07). */}
          {props.modelStatusResolved && <AssistantModelSleepNote model={props.agentStatus?.model_facts} />}
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
            * The messages about the question sit under the question box, never
            * above it (the AssistAnswer board).
            *
            * A banner inserted above used to push the box down by its own height
            * and pull it back up again when it was replaced — so a reader who
            * clicked where the box had just been typed a whole sentence into the
            * hint strip under it and lost every character. Nothing that appears
            * and disappears on its own is allowed above the composer. Validation
            * is here too, never inside the closed disclosure, where the one
            * thing that could explain a disabled button would be folded away.
            */}
          {chat.awaitingAnswerLost && (
            <Notice severity="warning">
              No answer arrived for your last question, and Vaelor has stopped waiting for it.
              Nothing was changed. Ask it again when you are ready.
            </Notice>
          )}
          {questionTooLong && <Notice severity="danger">Keep the question to 4,000 characters or fewer.</Notice>}
          {areaUnavailable && <Notice severity="danger">That problem area is unavailable on this appliance. Choose another, or leave it on Automatic.</Notice>}
          {chat.chatNotice && <Notice severity={chat.chatNoticeRefused ? "danger" : "info"}>{chat.chatNotice}</Notice>}
          {/*
            * The skills view reports the same `notice` with a link to the
            * proposal it just created, so it is shown here only while that view
            * is closed.
            */}
          {props.notice && <Notice severity={props.noticeSeverity}>{props.notice}</Notice>}
          {capabilityStrip}
        </section>
      )}

      <TextPromptDialog open={chat.renameTitle !== null} title="Rename chat" description="Give this saved conversation a short, recognizable name." label="Chat name" value={chat.renameTitle ?? ""} busy={chat.chatBusy} error={chat.dialogError} onChange={chat.setRenameTitle} onCancel={() => chat.setRenameTitle(null)} onSubmit={() => void chat.saveConversationTitle()} />
      {/*
        * "Permanently removes ... every message" alone stopped being true in
        * Alpha 46: a recent conversation may also have a fast-wake snapshot
        * on disk, and deleting the conversation retires it (nothing can
        * restore it again) without erasing its bytes until a later save
        * reuses that slot. The sentence says so rather than overclaiming.
        */}
      <ConfirmDialog irreversible open={chat.confirmChatDelete} title="Delete saved chat?" description="This permanently deletes the conversation and every message in it. This cannot be undone. If it had a saved fast-wake snapshot, that snapshot is retired too, though its data may remain on this appliance's disk until a later save reuses that space." confirmLabel="Delete chat" busy={chat.chatBusy} error={chat.dialogError} onCancel={() => chat.setConfirmChatDelete(false)} onConfirm={() => void chat.deleteConversation()} />
      <ActionReviewDialog
        busy={props.busy}
        error={props.proposalError}
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
