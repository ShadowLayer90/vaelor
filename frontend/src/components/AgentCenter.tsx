import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { useAssistantAutomations } from "../hooks/useAssistantAutomations";
import { useAssistantChat } from "../hooks/useAssistantChat";
import { useGpuServingMode } from "../hooks/useGpuServingMode";
import { assistantModelReadiness } from "../lib/assistantModelReadiness";
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

/**
 * The readiness read failed. Ask and Routines say so beside a Retry, so the
 * page notice must not say it a second time - and a page notice is not cleared
 * by the Retry that recovers, so it stayed red over a working page (VD-200).
 */
class ReadinessReadError extends Error {}

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
  // Why the last readiness read failed; Ask shows it with a Retry instead of loading forever (VD-200).
  const [statusReadError, setStatusReadError] = useState("");
  // Whether the agent list has been read once: before that, Routines says so rather than "0" (VD-200).
  const [profilesRead, setProfilesRead] = useState(false);
  // The skills list read once; before then the chip is not drawn rather than saying "0 skills".
  const [skillsRead, setSkillsRead] = useState(false);
  const [skillsReadFailed, setSkillsReadFailed] = useState(false);
  const [setupData, setSetupData] = useState<CopilotSetupData | null>(null);
  const [showIntelligenceSetup, setShowIntelligenceSetup] = useState(false);
  const [intelligenceChoice, setIntelligenceChoice] = useState<"" | "basic" | "local" | "provider" | null>(null);
  const [proposalReview, setProposalReview] = useState<ProposalReview | null>(null);
  /*
   * The refusal of whichever dialog is open - the action review, the skill
   * editor, or the skill delete - shown inside it, never on the inert page
   * beneath (VD-189). Only one is open at a time; any opening or closing clears it.
   */
  const [dialogError, setDialogError] = useState("");
  useEffect(() => setDialogError(""), [proposalReview, editingSkill, deletingSkill]);
  const checkAbort = useRef<AbortController | null>(null);
  const chat = useAssistantChat(session);
  // A cluster agent is backed by the serving cluster model, not the single-node
  // Assistant model, so authoring one must not be gated on that model. A healthy
  // vLLM cluster deployment - the same backing the deploy modal binds - is that
  // signal; `deployment` is set only when one is actually serving. Viewers get an
  // unknown mode (no /cluster read), so this stays false for them.
  const clusterServing = useGpuServingMode(session.user.role);
  const clusterModelServing = Boolean(clusterServing.known && clusterServing.deployment);
  // The appliance model, once connected, is ready for every user; a per-user
  // local/provider choice is not required. `intelligence_choice` is stored per
  // user, so gating on it hid a working model from anyone who did not pick it
  // themselves. Only an explicit "basic" opt-out turns the model off.
  /*
   * A configured model is not a working one. The server probes the endpoint,
   * and one projection answers "is it answering" for Ask and Routines alike:
   * Routines once read `configured` alone and enabled Run beside Ask saying
   * the same model was not answering (ACC-136). `modelReady` stays "a model
   * is configured" (authoring, the Ask pill's wording); anything that sends
   * work to the model keys on `modelAnswering`.
   */
  const readiness = assistantModelReadiness(agentStatus, intelligenceChoice);
  const modelReady = readiness.configured;
  const modelAnswering = readiness.answering;
  // Ask shows "not answering" and "no model offered" as two notices, so the
  // unreachable one is the not-answering reason minus the not-offered case
  // (ACC-099), which carries its own reason in `readiness.notOffered`.
  const modelUnreachableReason = readiness.notOffered ? "" : readiness.notAnsweringReason;
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
    setNotice: useCallback((message: string, refused?: boolean) => setNotice(message, refused ? "danger" : "info"), [setNotice]),
  });
  const loadAutomations = automation.load;
  const refresh = useCallback(async () => {
    const [nextProfiles, nextTasks, nextHandoffTargets, nextStatus, nextSetup, nextPreferences] = await Promise.all([
      apiRequest<AgentProfile[]>("/assistant/profiles?surface=assistant"),
      apiRequest<AgentTask[]>(`/assistant/tasks${taskView === "archive" ? "?archived=1" : ""}`),
      apiRequest<HandoffTarget[]>("/assistant/handoff-targets"),
      apiRequest<AgentStatus>("/agent/status"),
      apiRequest<CopilotSetupData>("/copilot/setup"),
      apiRequest<{ intelligence_choice: "" | "basic" | "local" | "provider" }>("/assistant/preferences"),
    ]).catch((error: unknown) => {
      const reason = error instanceof Error ? error.message : "Try again in a moment.";
      setStatusReadError(reason);
      throw new ReadinessReadError(reason);
    });
    setStatusReadError("");
    setProfiles(nextProfiles);
    setProfilesRead(true);
    setTasks(nextTasks);
    setHandoffTargets(nextHandoffTargets);
    setAgentStatus(nextStatus);
    setModelStatusResolved(true);
    setSetupData(nextSetup);
    setIntelligenceChoice(nextPreferences.intelligence_choice);
    await loadAutomations();
    if (isAdministrator) {
      // A failed skills read is "Not read" on the chip - not a chip that vanishes,
      // and not a page notice that stops the memory count being read (VD-200).
      const nextSkills = await apiRequest<AssistantSkill[]>("/assistant/skills").catch(() => null);
      setSkillsReadFailed(nextSkills === null);
      if (nextSkills !== null) {
        setSkills(nextSkills);
        setSkillsRead(true);
      }
      // Every memory endpoint is administrator-only. Reading the count under
      // the same gate keeps the chip from ever offering a link that 403s.
      const stats = await apiRequest<AssistantMemoryStats>("/assistant/status");
      // A count the status did not carry is unread, not zero (VD-200 decision 9).
      setMemoryCount(typeof stats.memories === "number" ? stats.memories : null);
    }
  }, [isAdministrator, loadAutomations, taskView]);
  useEffect(() => { refreshRef.current = refresh; }, [refresh]);
  // VD-049 / VD-201 item 2: Vaelor's model or basic mode; "provider" is refused by the server.
  const chooseIntelligence = async (choice: "basic" | "local", openSetup = true) => {
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
          // The recommendation's own name: this once named a fixed model whatever was recommended.
          chat.setChatNotice(`${setupData?.recommendation.primary.name ?? "The recommended local model"} is selected as the recommended local default. Review it before installation.`);
        }
      }
    } catch (error) {
      chat.setChatNotice(error instanceof Error ? error.message : "Your assistant choice could not be saved.", true);
    } finally {
      setBusy(false);
    }
  };
  useEffect(() => {
    void refresh().catch((error) => {
      if (error instanceof ReadinessReadError) return;
      setNotice(error instanceof Error ? error.message : "Agent tasks could not be loaded.", "danger");
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
    if (modelAnswering) return;
    void refresh().catch((error) => {
      if (error instanceof ReadinessReadError) return;
      setNotice(error instanceof Error ? error.message : "Assistant readiness could not be refreshed.", "danger");
    });
  }, [modelAnswering, refresh, tab]);
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
      setNotice(error instanceof Error ? error.message : "Finished tasks could not be cleared.", "danger");
    } finally {
      setBusy(false);
    }
  };
  const approveProposal = async () => {
    if (!proposalReview) return;
    setBusy(true);
    setDialogError("");
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
      setDialogError(error instanceof Error && error.message ? error.message : "The action could not be queued.");
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
      chat.setChatNotice(error instanceof Error ? error.message : "The agent run could not be prepared.", true);
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
      setNotice(error instanceof Error ? error.message : "The task could not be updated.", "danger");
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
      setNotice(error instanceof Error ? error.message : "The task could not be retried.", "danger");
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
      setNotice(error instanceof Error ? error.message : "The capable re-run could not be started.", "danger");
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
      setNotice(error instanceof Error ? error.message : "The task could not be handed off.", "danger");
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
      setNotice(error instanceof Error ? error.message : "The skill proposal was rejected.", "danger");
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
      setNotice(error instanceof Error ? error.message : "The review could not be saved.", "danger");
    } finally {
      setBusy(false);
    }
  };
  const updateSkill = async (draft: SkillDraft) => {
    if (!editingSkill) return;
    setBusy(true); setNotice(""); setDialogError("");
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
      setDialogError(error instanceof Error && error.message ? error.message : "The skill revision could not be saved.");
    } finally { setBusy(false); }
  };
  const deleteSkill = async () => {
    if (!deletingSkill) return;
    setBusy(true); setNotice(""); setDialogError("");
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
      setDialogError(error instanceof Error && error.message ? error.message : "The custom skill could not be deleted.");
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
          memoryCount={isAdministrator ? memoryCount : undefined}
          modelReady={modelReady}
          modelStatusResolved={modelStatusResolved}
          statusReadError={statusReadError}
          onRetryStatus={() => void refresh().catch(() => undefined)}
          modelNotOfferedReason={readiness.notOffered}
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
          onSubmit={submitQuestion}
          onToggleSkills={() => setShowSkills((current) => !current)}
          problemArea={problemArea}
          profiles={profiles}
          proposalReview={proposalReview}
          proposalError={dialogError}
          session={session}
          setDurable={setDurable}
          setProblemArea={setProblemArea}
          setProposalReview={setProposalReview}
          setShowIntelligenceSetup={setShowIntelligenceSetup}
          setupData={setupData}
          showIntelligenceSetup={showIntelligenceSetup}
          showSkills={showSkills && isAdministrator}
          skillCount={!isAdministrator ? undefined : skillsReadFailed ? null : skillsRead ? skills.filter((item) => item.status === "active").length : undefined}
          skillsPanel={
            <AgentCenterSkillsPanel
              busy={busy}
              deletingSkill={deletingSkill}
              dialogError={dialogError}
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
              modelConfigured={modelReady}
              modelReady={modelAnswering}
              notice={notice}
              noticeRefused={noticeSeverity === "danger"}
              // The promise reaches the panel, so its New schedule / New alert rule dialog closes only on success.
              onCreateAutomation={(event) => automation.createAutomation(event)}
              onCreateTrigger={(event) => automation.createTrigger(event)}
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
          clusterModelServing={clusterModelServing}
          modelLabel={agentStatus?.model ?? agentStatus?.provider ?? "Selected model"}
          modelNotAnsweringReason={readiness.notAnsweringReason}
          modelReady={modelReady}
          modelStatusResolved={modelStatusResolved}
          onChanged={() => void refresh()}
          onViewRun={revealAgentRun}
          profiles={profiles}
          profilesRead={profilesRead}
          profilesReadError={statusReadError}
          session={session}
          tasks={tasks}
          triggers={automation.triggers}
        />
      }
    />
  );
}
