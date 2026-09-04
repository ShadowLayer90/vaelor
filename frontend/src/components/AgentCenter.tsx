import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { useAssistantAutomations } from "../hooks/useAssistantAutomations";
import { useAssistantChat } from "../hooks/useAssistantChat";
import type { Session } from "../types";
import type { CopilotSetupData } from "./CopilotSetup";
import type { AssistantTab } from "./AssistantNavigationTabs";
import type { SkillDraft } from "./SkillEditorDialog";
import { AgentAssistantPanel, AUTOMATIC_PROBLEM_AREA, type ProposalReview } from "./AgentAssistantPanel";
import { AssistantHistoryPanel } from "./AssistantHistoryPanel";
import { applianceCheckInterrupted, applianceCheckOutcome } from "./applianceCheckOutcome";
import { MODEL_ANSWER_TIMEOUT_MS } from "./modelAnswerTimeout";
import type { RunHistoryFilter } from "./AssistantRunHistory";
import type { NoticeSeverity } from "./ui";
import { CustomAgentsPanel } from "./CustomAgentsPanel";
import type { AgentProfile, AgentStatus, AgentTask, AssistantAnswer, AssistantSkill, HandoffTarget } from "./agentTypes";
import { AgentCenterAutomationsPanel } from "./agent-center-automations-panel";
import { AlertChannelsPanel } from "./AlertChannelsPanel";
import { AgentCenterSkillsPanel } from "./agent-center-skills-panel";
import { AgentCenterTabs } from "./agent-center-tabs";
export { isBuiltinSkill } from "./agent-center-skills-panel";

/** Reviewed-memory totals; administrator-only, like every memory endpoint. */
interface AssistantMemoryStats {
  memories: number;
}

export function AgentCenter({ session }: { session: Session }) {
  const isAdministrator = session.user.role === "administrator";
  const [tab, setTab] = useState<AssistantTab>("ask");
  const [profiles, setProfiles] = useState<AgentProfile[]>([]);
  const [tasks, setTasks] = useState<AgentTask[]>([]);
  const [taskView, setTaskView] = useState<"recent" | "archive">("recent");
  const [historyFilter, setHistoryFilter] = useState<RunHistoryFilter>("all");
  const [handoffTargets, setHandoffTargets] = useState<HandoffTarget[]>([]);
  const [handoffSelections, setHandoffSelections] = useState<Record<string, string>>({});
  const [profile, setProfile] = useState("system");
  const [problemArea, setProblemArea] = useState(AUTOMATIC_PROBLEM_AREA);
  const [durable, setDurable] = useState(false);
  const [busy, setBusy] = useState(false);
  const [checkRunning, setCheckRunning] = useState(false);
  /*
   * When the running check started, in epoch milliseconds. The elapsed counter
   * used to begin at whatever moment its component happened to mount, and Ask
   * is unmounted by every tab change, so a check at "110s elapsed" came back
   * from History reading "3s elapsed".
   */
  const [checkStartedAt, setCheckStartedAt] = useState(0);
  const [notice, setNoticeMessage] = useState("");
  const [noticeSeverity, setNoticeSeverity] = useState<NoticeSeverity>("info");
  /*
   * Severity travels with the message. A blocked run that executed nothing was
   * announced in the same neutral banner as a clean pass, so the two most
   * different outcomes on this screen looked identical. Callers that do not
   * care keep the informational default and cannot leave a stale warning behind.
   */
  const setNotice = useCallback((message: string, severity: NoticeSeverity = "info") => {
    setNoticeMessage(message);
    setNoticeSeverity(severity);
  }, []);
  const [skills, setSkills] = useState<AssistantSkill[]>([]);
  const [showSkills, setShowSkills] = useState(false);
  const [skillName, setSkillName] = useState("");
  const [skillDescription, setSkillDescription] = useState("");
  const [skillContent, setSkillContent] = useState("");
  const [editingSkill, setEditingSkill] = useState<AssistantSkill | null>(null);
  const [deletingSkill, setDeletingSkill] = useState<AssistantSkill | null>(null);
  const [newSkillId, setNewSkillId] = useState("");
  const [memoryCount, setMemoryCount] = useState<number | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  /*
   * The status pill must not claim anything about this appliance before the
   * appliance has answered. `/agent/status` resolves about two seconds after
   * paint, and until it does "not configured" is an assumption, not a reading.
   */
  const [modelStatusResolved, setModelStatusResolved] = useState(false);
  const [setupData, setSetupData] = useState<CopilotSetupData | null>(null);
  const [showIntelligenceSetup, setShowIntelligenceSetup] = useState(false);
  const [intelligenceChoice, setIntelligenceChoice] = useState<"" | "basic" | "local" | "provider" | null>(null);
  const [proposalReview, setProposalReview] = useState<ProposalReview | null>(null);
  const checkAbort = useRef<AbortController | null>(null);
  const chat = useAssistantChat(session);
  // The appliance model, once connected, is ready for every user; a per-user
  // local/provider choice is not required. `intelligence_choice` is stored per
  // user, so gating on it hid a working model from anyone who did not pick it
  // themselves. Only an explicit "basic" opt-out turns the model off.
  const modelReady = Boolean(
    agentStatus?.configured && intelligenceChoice !== "basic",
  );
  /*
   * A configured model is not a working one. The server now probes the
   * endpoint, and an appliance whose model has been switched off must say so
   * here rather than showing verified-health styling and failing a minute
   * later inside an agent run.
   */
  const modelUnreachableReason = modelReady && agentStatus?.reachable === false
    ? (agentStatus.unreachable_reason || "The selected AI model could not be reached.")
    : "";
  /*
   * `refresh` and the automation handlers each need the other, so the container
   * holds the current `refresh` in a ref rather than threading a definition
   * cycle through both.
   */
  const refreshRef = useRef<() => Promise<unknown>>(() => Promise.resolve());
  const automation = useAssistantAutomations({
    csrfToken: session.csrf_token,
    profile,
    refreshRef,
    setBusy,
    setNotice,
  });
  const loadAutomations = automation.load;
  const refresh = useCallback(async () => {
    const [nextProfiles, nextTasks, nextHandoffTargets, nextStatus, nextSetup, nextPreferences] = await Promise.all([
      apiRequest<AgentProfile[]>("/assistant/profiles"),
      apiRequest<AgentTask[]>(`/assistant/tasks${taskView === "archive" ? "?archived=1" : ""}`),
      apiRequest<HandoffTarget[]>("/assistant/handoff-targets"),
      apiRequest<AgentStatus>("/agent/status"),
      apiRequest<CopilotSetupData>("/copilot/setup"),
      apiRequest<{ intelligence_choice: "" | "basic" | "local" | "provider" }>("/assistant/preferences"),
    ]);
    setProfiles(nextProfiles);
    setTasks(nextTasks);
    setHandoffTargets(nextHandoffTargets);
    setAgentStatus(nextStatus);
    setModelStatusResolved(true);
    setSetupData(nextSetup);
    setIntelligenceChoice(nextPreferences.intelligence_choice);
    await loadAutomations();
    if (isAdministrator) {
      setSkills(await apiRequest<AssistantSkill[]>("/assistant/skills"));
      // Every memory endpoint is administrator-only. Reading the count under
      // the same gate keeps the chip from ever offering a link that 403s.
      const stats = await apiRequest<AssistantMemoryStats>("/assistant/status");
      setMemoryCount(stats.memories ?? 0);
    }
  }, [isAdministrator, loadAutomations, taskView]);
  useEffect(() => { refreshRef.current = refresh; }, [refresh]);
  const chooseIntelligence = async (choice: "basic" | "local" | "provider", openSetup = true) => {
    setBusy(true);
    chat.setChatNotice("");
    try {
      await apiRequest(
        "/assistant/preferences",
        {
          method: "PATCH",
          body: JSON.stringify({ intelligence_choice: choice }),
        },
        session.csrf_token,
      );
      setIntelligenceChoice(choice);
      if (choice === "basic") {
        setShowIntelligenceSetup(false);
        chat.setChatNotice("Built-in basic mode is active. Live appliance questions and safe diagnostics remain available.");
      } else {
        setShowIntelligenceSetup(openSetup);
        if (openSetup) {
          chat.setChatNotice(
            choice === "local"
              ? "Qwen3 1.7B is selected as the recommended local default. Review it before installation."
              : "Choose and test the AI provider you want to use.",
          );
        }
      }
    } catch (error) {
      chat.setChatNotice(error instanceof Error ? error.message : "Your assistant choice could not be saved.");
    } finally {
      setBusy(false);
    }
  };
  useEffect(() => {
    void refresh().catch((error) => {
      setNotice(error instanceof Error ? error.message : "Agent tasks could not be loaded.");
    });
  }, [refresh]);

  /*
   * AI Chat can hand a matched agent run over to this page. Without this it
   * only switched destination and left the reader on the chat tab, still
   * looking for a run that lives on another one. The pre-merge tab names are
   * still accepted because AI Chat dispatches them and is owned elsewhere.
   */
  /**
   * Show an agent run where the run actually is.
   *
   * A prepared run is a run, so it belongs under History with the Agent runs
   * filter selected — for everybody. Sending administrators to the agent
   * workshop instead pointed at the thing that authored the run rather than at
   * the run, and sending anybody else to a tab they do not have (Routines is
   * administrator-only) left the ARIA tablist reporting no selection at all.
   */
  const revealAgentRun = useCallback(() => {
    // A tab change that bypasses `onChange` has to clear the page notice for
    // itself, or the last thing that happened somewhere else is still on
    // screen above a run it has nothing to do with.
    setNotice("");
    setTab("history");
    setHistoryFilter("agents");
  }, [setNotice]);
  /**
   * Show an appliance check where the check actually is.
   *
   * "Run this check" created a run and said nothing on the screen the reader
   * was looking at: the composer cleared, no message entered the transcript,
   * and the only trace was a NEEDS APPROVAL card on a different tab. Two
   * submissions were made and believed lost. A created run is a run, so this
   * does what a prepared agent run already does — lands the reader on History
   * with the matching filter, where the banner announcing it is readable.
   */
  const revealChecks = useCallback(() => {
    setNotice("");
    setTab("history");
    setHistoryFilter("checks");
  }, [setNotice]);
  useEffect(() => {
    const openTab = (event: Event) => {
      const requested = (event as CustomEvent<string>).detail;
      // The pre-split names are still accepted because AI Chat dispatches them
      // and is owned elsewhere.
      if (requested === "customAgents" || requested === "agents" || requested === "routines") {
        revealAgentRun();
      } else if (requested === "history") {
        setTab("history");
      } else if (requested === "specialists" || requested === "ask") {
        setTab("ask");
        setHistoryFilter("checks");
      }
    };
    window.addEventListener("vaelor:assistant-tab", openTab);
    return () => window.removeEventListener("vaelor:assistant-tab", openTab);
  }, [revealAgentRun]);
  useEffect(() => {
    if (modelReady) return;
    void refresh().catch((error) => {
      setNotice(error instanceof Error ? error.message : "Assistant readiness could not be refreshed.");
    });
  }, [modelReady, refresh, tab]);
  /*
   * Both tabs now show live runs, so the poll can no longer be scoped to the
   * one that happened to own the card. An approved run used to sit at "ready"
   * indefinitely while the server moved it to running and then failed, and the
   * only way to learn the outcome was a manual page reload.
   */
  useEffect(() => {
    if (
      taskView !== "recent"
      || !tasks.some((task) => ["ready", "running"].includes(task.state))
    ) return;
    const interval = window.setInterval(() => {
      void apiRequest<AgentTask[]>("/assistant/tasks")
        .then(setTasks)
        .catch(() => undefined);
    }, 1500);
    return () => window.clearInterval(interval);
  }, [taskView, tasks]);
  const archiveFinishedTasks = async () => {
    setBusy(true);
    setNotice("");
    try {
      const result = await apiRequest<{ archived: number }>(
        "/assistant/tasks/archive-finished",
        { method: "POST", body: "{}" },
        session.csrf_token,
      );
      setNotice(
        result.archived
          ? `${result.archived} finished task${result.archived === 1 ? "" : "s"} removed from the recent list.`
          : "There are no finished tasks to clear.",
      );
      await refresh();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Finished tasks could not be cleared.");
    } finally {
      setBusy(false);
    }
  };
  const approveProposal = async () => {
    if (!proposalReview) return;
    setBusy(true);
    chat.setChatNotice("");
    try {
      await apiRequest(
        "/jobs",
        { method: "POST", body: JSON.stringify(proposalReview.job) },
        session.csrf_token,
      );
      setProposalReview(null);
      chat.setChatNotice("Action approved and queued. Progress and the audited result are available in Activity.");
    } catch (error) {
      chat.setChatNotice(error instanceof Error ? error.message : "The action could not be queued.");
    } finally {
      setBusy(false);
    }
  };
  const prepareAgentRun = async (proposal: NonNullable<AssistantAnswer["proposed_agent_task"]>) => {
    setBusy(true);
    setNotice("");
    chat.setChatNotice("");
    try {
      await apiRequest(
        "/assistant/tasks",
        { method: "POST", body: JSON.stringify({
          title: `${proposal.profile_name} chat request`,
          description: proposal.task,
          profile: proposal.profile_id,
          profile_version: proposal.profile_version,
          approval_required: true,
          idempotency_key: crypto.randomUUID(),
        }) },
        session.csrf_token,
      );
      await refresh();
      /*
       * The outcome has to be readable on the tab this navigation lands on.
       * It was announced through `chat.setChatNotice`, which renders only
       * inside Ask — the very panel `revealAgentRun` unmounts — so the one
       * sentence saying a run had been prepared was never seen by anybody, and
       * it went on claiming the run was "below" from a tab away. It is the
       * page notice now, which History renders, and which any tab change
       * clears so a handled run cannot greet the reader again later.
       */
      revealAgentRun();
      setNotice(
        `Agent run prepared for ${proposal.profile_name}. History is showing Agent runs — review and approve it below.`,
      );
    } catch (error) {
      chat.setChatNotice(error instanceof Error ? error.message : "The agent run could not be prepared.");
    } finally {
      setBusy(false);
    }
  };
  /*
   * The appliance check and the chat answer are the same question through two
   * endpoints, so one composer submits both. Which one runs is decided only by
   * the reader's own two controls — never by inspecting the words they typed.
   */
  const runApplianceCheck = async (question: string) => {
    const area = problemArea || "system";
    const controller = new AbortController();
    checkAbort.current = controller;
    setBusy(true);
    setCheckRunning(true);
    setCheckStartedAt(Date.now());
    setNotice("");
    chat.setChatInput("");
    try {
      const task = durable
        ? await apiRequest<AgentTask>(
          "/assistant/tasks",
          {
            method: "POST",
            signal: controller.signal,
            timeoutMs: MODEL_ANSWER_TIMEOUT_MS,
            body: JSON.stringify({
              // The typed symptom is the title. A generated
              // "<profile> task" made every finished card read alike, so the
              // one thing that told two runs apart was the run hash.
              title: question.slice(0, 100),
              description: question,
              profile: area,
              approval_required: true,
              idempotency_key: crypto.randomUUID(),
            }),
          },
          session.csrf_token,
        )
        : await apiRequest<AgentTask>(
          "/assistant/delegations",
          {
            method: "POST",
            signal: controller.signal,
            timeoutMs: MODEL_ANSWER_TIMEOUT_MS,
            body: JSON.stringify({
              profile: area,
              task: question,
              idempotency_key: crypto.randomUUID(),
            }),
          },
          session.csrf_token,
        );
      /*
       * The banner reads the run, not the request. Announcing success because
       * the POST resolved is what put "The appliance check finished" above a
       * run that was blocked and had executed nothing.
       */
      /*
       * The reader is taken to the run, then told about it. Setting the banner
       * on Ask and leaving them there was the reported failure: the run was on
       * History as NEEDS APPROVAL and nothing on the screen they were looking
       * at said a check had been created at all.
       */
      const outcome = applianceCheckOutcome(task);
      revealChecks();
      setNotice(outcome.message, outcome.severity);
      await refresh();
      return true;
    } catch (error) {
      /*
       * Stopping means stopping the wait, not undoing a request the appliance
       * has already accepted, and a client-side timeout means the same thing.
       * The run keeps going server-side and shows up in the history like any
       * other; neither is a failure to report as one.
       */
      const timedOut = error instanceof Error && "code" in error && error.code === "request_timeout";
      if (controller.signal.aborted || timedOut) {
        const outcome = applianceCheckInterrupted(controller.signal.aborted);
        setNotice(outcome.message, outcome.severity);
      } else {
        setNotice(
          error instanceof Error ? error.message : "The appliance check could not run.",
          "warning",
        );
      }
      chat.setChatInput(question);
      return false;
    } finally {
      checkAbort.current = null;
      setCheckRunning(false);
      setCheckStartedAt(0);
      setBusy(false);
    }
  };
  const submitQuestion = (event: FormEvent) => {
    event.preventDefault();
    /*
     * A second submit while one is in flight is a double-click, not a second
     * question. Guarding here rather than relying on the button's disabled
     * state means the guard holds even if the click lands in the frame before
     * the re-render.
     */
    if (busy || checkRunning || chat.chatBusy) return;
    const question = chat.chatInput.trim();
    if (!question || question.length > 4000) return;
    if (durable || problemArea !== AUTOMATIC_PROBLEM_AREA) {
      void runApplianceCheck(question);
      return;
    }
    void chat.ask();
  };
  const transitionTask = async (task: AgentTask, state: "ready" | "cancelled") => {
    setNotice("");
    try {
      await apiRequest(
        `/assistant/tasks/${task.id}`,
        { method: "PATCH", body: JSON.stringify({ state }) },
        session.csrf_token,
      );
      setNotice(state === "ready" ? "Task approved. The background worker will run it." : "Task cancelled.");
      await refresh();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The task could not be updated.");
    }
  };
  const retryTask = async (task: AgentTask) => {
    setBusy(true);
    setNotice("");
    try {
      await apiRequest(`/assistant/tasks/${task.id}/retry`, { method: "POST" }, session.csrf_token);
      setNotice("A fresh copy of the task is ready for review.");
      await refresh();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The task could not be retried.");
    } finally {
      setBusy(false);
    }
  };
  const escalateTask = async (task: AgentTask) => {
    setBusy(true);
    setNotice("");
    try {
      // Escalation CREATES a fresh run of the SAME task on the more capable GPU
      // model rather than retrying. The offered run is usually delivered-but-thin
      // and so in state "completed", which retry() rejects; a new approval-gated
      // task with the original description and use_capable_model works for every
      // offered state. Same approval, role, and read-only envelope as any create.
      await apiRequest(
        "/assistant/tasks",
        { method: "POST", body: JSON.stringify({
          title: task.title,
          description: task.description,
          profile: task.profile,
          ...(typeof task.profile_version === "number" && task.profile_version > 0
            ? { profile_version: task.profile_version } : {}),
          approval_required: true,
          use_capable_model: true,
          idempotency_key: crypto.randomUUID(),
        }) },
        session.csrf_token,
      );
      setNotice("Re-running this task on the more capable model. Review and approve it below.");
      await refresh();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The capable re-run could not be started.");
    } finally {
      setBusy(false);
    }
  };
  const handoffTask = async (task: AgentTask) => {
    const assignedTo = handoffSelections[task.id] || task.assigned_to || session.user.username;
    setBusy(true);
    setNotice("");
    try {
      await apiRequest(
        `/assistant/tasks/${task.id}/handoff`,
        { method: "POST", body: JSON.stringify({ assigned_to: assignedTo }) },
        session.csrf_token,
      );
      setNotice(`Task assigned to ${assignedTo}.`);
      await refresh();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The task could not be handed off.");
    } finally {
      setBusy(false);
    }
  };
  const proposeSkill = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setNotice("");
    try {
      const created = await apiRequest<AssistantSkill>(
        "/assistant/skills",
        { method: "POST", body: JSON.stringify({
          name: skillName, description: skillDescription, content: skillContent,
        }) },
        session.csrf_token,
      );
      setSkillName(""); setSkillDescription(""); setSkillContent("");
      setNewSkillId(created.id);
      setNotice("Saved as a proposal — review it now.");
      await refresh();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The skill proposal was rejected.");
    } finally {
      setBusy(false);
    }
  };
  const reviewSkill = async (skill: AssistantSkill, decision: "active" | "rejected") => {
    setBusy(true);
    try {
      await apiRequest(
        `/assistant/skills/${skill.id}/review`,
        { method: "POST", body: JSON.stringify({ decision }) },
        session.csrf_token,
      );
      setNotice(decision === "active" ? "Skill reviewed and activated." : "Skill proposal rejected.");
      await refresh();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The review could not be saved.");
    } finally {
      setBusy(false);
    }
  };
  const updateSkill = async (draft: SkillDraft) => {
    if (!editingSkill) return;
    setBusy(true); setNotice("");
    try {
      await apiRequest(
        `/assistant/skills/${editingSkill.id}`,
        { method: "PATCH", body: JSON.stringify(draft) },
        session.csrf_token,
      );
      setEditingSkill(null);
      setNotice("New skill version saved. Review it before activation.");
      await refresh();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The skill revision could not be saved.");
    } finally { setBusy(false); }
  };
  const deleteSkill = async () => {
    if (!deletingSkill) return;
    setBusy(true); setNotice("");
    try {
      await apiRequest(
        `/assistant/skills/${deletingSkill.id}`,
        { method: "DELETE" },
        session.csrf_token,
      );
      setDeletingSkill(null);
      setNotice("Custom skill deleted.");
      await refresh();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The custom skill could not be deleted.");
    } finally { setBusy(false); }
  };
  return (
    <AgentCenterTabs
      active={tab}
      askPanel={
        <AgentAssistantPanel
          agentStatus={agentStatus}
          busy={busy}
          chat={chat}
          checkRunning={checkRunning}
          checkStartedAt={checkStartedAt}
          durable={durable}
          intelligenceChoice={intelligenceChoice}
          memoryCount={isAdministrator ? memoryCount ?? 0 : null}
          modelReady={modelReady}
          modelStatusResolved={modelStatusResolved}
          modelUnreachableReason={modelUnreachableReason}
          notice={notice}
          noticeSeverity={noticeSeverity}
          onApproveProposal={() => void approveProposal()}
          onCancelCheck={() => checkAbort.current?.abort()}
          onChooseIntelligence={(choice, openSetup) => void chooseIntelligence(choice, openSetup)}
          onCloseIntelligenceSetup={() => {
            setShowIntelligenceSetup(false);
            void refresh();
          }}
          onPrepareAgentRun={(proposal) => void prepareAgentRun(proposal)}
          onRefresh={() => void refresh()}
          onSubmit={submitQuestion}
          onToggleSkills={() => setShowSkills((current) => !current)}
          problemArea={problemArea}
          profiles={profiles}
          proposalReview={proposalReview}
          session={session}
          setDurable={setDurable}
          setProblemArea={setProblemArea}
          setProposalReview={setProposalReview}
          setShowIntelligenceSetup={setShowIntelligenceSetup}
          setupData={setupData}
          showIntelligenceSetup={showIntelligenceSetup}
          showSkills={showSkills && isAdministrator}
          skillCount={isAdministrator ? skills.filter((item) => item.status === "active").length : null}
          skillsPanel={
            <AgentCenterSkillsPanel
              busy={busy}
              deletingSkill={deletingSkill}
              editingSkill={editingSkill}
              modelReady={modelReady}
              newSkillId={newSkillId}
              notice={notice}
              onDeleteSkill={() => void deleteSkill()}
              onEditSkill={setEditingSkill}
              onProposeSkill={(event) => void proposeSkill(event)}
              onReviewSkill={(skill, decision) => void reviewSkill(skill, decision)}
              onSaveSkill={(draft) => void updateSkill(draft)}
              onSetDeletingSkill={setDeletingSkill}
              onSetEditingSkill={setEditingSkill}
              onSetSkillContent={setSkillContent}
              onSetSkillDescription={setSkillDescription}
              onSetSkillName={setSkillName}
              skillContent={skillContent}
              skillDescription={skillDescription}
              skillName={skillName}
              skills={skills}
            />
          }
        />
      }
      historyPanel={
        <AssistantHistoryPanel
          automaticTaskIds={automation.automaticTaskIds}
          automations={automation.automations}
          busy={busy}
          filter={historyFilter}
          handoffSelections={handoffSelections}
          handoffTargets={handoffTargets}
          notice={notice}
          noticeSeverity={noticeSeverity}
          onArchiveFinishedTasks={() => void archiveFinishedTasks()}
          onDiscussTask={(task) => {
            chat.seedChat([
              `Explain this ${task.profile} appliance check and help me decide the safest next step.`,
              `Summary: ${task.result.summary || "No summary supplied."}`,
              `Findings: ${(task.result.findings || []).join("; ") || "None supplied."}`,
              `Recommendations: ${(task.result.recommendations || []).join("; ") || "None supplied."}`,
              `Next actions: ${(task.result.next_actions || []).join("; ") || "None supplied."}`,
            ].join("\n"), "Started a new chat for this appliance check.");
            setProblemArea(AUTOMATIC_PROBLEM_AREA);
            setDurable(false);
            // Discussing a result is a question, so it lands on Ask. This is
            // one of the two tab changes that bypass `onChange`, so it clears
            // the page notice itself rather than carrying History's last
            // outcome onto a screen where it is no longer about anything.
            setNotice("");
            setTab("ask");
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
          onFilterChange={setHistoryFilter}
          onHandoffSelection={(taskId, username) => setHandoffSelections((current) => ({ ...current, [taskId]: username }))}
          onHandoffTask={(task) => void handoffTask(task)}
          onRefresh={() => void refresh()}
          onRetryTask={(task) => void retryTask(task)}
          onEscalateTask={(task) => void escalateTask(task)}
          onTaskViewChange={() => setTaskView((current) => current === "recent" ? "archive" : "recent")}
          onTransitionTask={(task, state) => void transitionTask(task, state)}
          onUpdated={(message) => { setNotice(message); void refresh(); }}
          session={session}
          tasks={tasks}
          taskView={taskView}
          triggers={automation.triggers}
        />
      }
      onChange={(nextTab) => {
        setTab(nextTab);
        setNotice("");
      }}
      role={session.user.role}
      routinesPanel={
        <CustomAgentsPanel
          automations={automation.automations}
          automationsPanel={
            <AgentCenterAutomationsPanel
              alertChannelsPanel={<AlertChannelsPanel canManage={isAdministrator} csrfToken={session.csrf_token} />}
              automationDelete={automation.automationDelete}
              automationName={automation.automationName}
              automationPrompt={automation.automationPrompt}
              automationSchedule={automation.automationSchedule}
              automationScheduleValid={automation.automationScheduleValid}
              automations={automation.automations}
              busy={busy}
              modelReady={modelReady}
              notice={notice}
              onCreateAutomation={(event) => void automation.createAutomation(event)}
              onCreateTrigger={(event) => void automation.createTrigger(event)}
              onDeleteAutomationItem={() => void automation.deleteAutomationItem()}
              onSelectTriggerSource={automation.selectTriggerSource}
              onSetAutomationDelete={automation.setAutomationDelete}
              onSetAutomationName={automation.setAutomationName}
              onSetAutomationPrompt={automation.setAutomationPrompt}
              onSetAutomationSchedule={automation.setAutomationSchedule}
              onSetProfile={setProfile}
              onSetTriggerName={automation.setTriggerName}
              onSetTriggerThreshold={automation.setTriggerThreshold}
              onToggleAutomation={(item) => void automation.toggleAutomation(item)}
              onToggleTrigger={(trigger) => void automation.toggleTrigger(trigger)}
              profile={profile}
              profiles={profiles}
              triggerName={automation.triggerName}
              triggerSource={automation.triggerSource}
              triggerThreshold={automation.triggerThreshold}
              triggerThresholdValid={automation.triggerThresholdValid}
              triggers={automation.triggers}
            />
          }
          modelLabel={agentStatus?.model ?? agentStatus?.provider ?? "Selected model"}
          modelReady={modelReady}
          modelStatusResolved={modelStatusResolved}
          onChanged={() => void refresh()}
          onViewRun={revealAgentRun}
          profiles={profiles}
          session={session}
          tasks={tasks}
          triggers={automation.triggers}
        />
      }
    />
  );
}
