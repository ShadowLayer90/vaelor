import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, apiRequest } from "../lib/api";
import type { Session } from "../types";
import type { CopilotSetupData } from "./CopilotSetup";
import { CopilotSetupDialog } from "./CopilotSetupDialog";
import { AddConnectionDialog } from "./AddConnectionDialog";
import { NpuInstallDialog } from "./NpuInstallDialog";
import { ActionReviewDialog } from "./ActionReviewDialog";
import { ApplicationDeploymentContainer } from "./ApplicationDeploymentContainer";
import { type AppTemplate, type PortPreflight } from "./AppCatalog";
import { ComposeImportDialog } from "./ComposeImportDialog";
import { ModelCatalog } from "./ModelCatalog";
import { useNpuInstall } from "../hooks/useNpuInstall";
import { Button, TabSet } from "./ui";
import { APPS_SECTION_LABELS, WorkloadManager, inventoryIsSettling, type ManagedInventory } from "./WorkloadManager";
import { useApplicationResearchLifecycle, confirmApplicationDeploymentClose } from "./workloads-application-research";
import { WorkloadCatalogModal, useWorkloadCatalogState } from "./workloads-catalog";
import { WorkloadJobActivity, WorkloadOperation, type ModelDownloadSelection } from "./workloads-jobs";
import type { AgentPlan, AgentStatus, ApplicationFeatures, InstallFlow, WorkloadCapabilities, WorkloadJob } from "./workloads-types";
import { WorkloadPlannerDialog } from "./WorkloadPlannerDialog";
import { WorkloadsInstallCards, WorkloadsMessages, WorkloadsRunningNow, installDoorReasons } from "./WorkloadsInstallPanel";
import { DockerPill, WorkloadsReadinessCard, dockerReadiness } from "./WorkloadsReadiness";
import { useWorkloadJobs } from "../hooks/useWorkloadJobs";
import { destinations } from "../lib/destinations";
import { TopbarPageActions, usePagePlace } from "../lib/topbarSlot";
import { jobIsTerminal } from "../lib/jobPresentation";
import { clearConnectionFormRequest, connectionFormRequested } from "../lib/connectionFormHandoff";
import "../styles/apps-install.css";

export { applicationResumeRequestSummary, confirmApplicationDeploymentClose } from "./workloads-application-research";
export { catalogFailureNeedsPortChange } from "./workloads-catalog";
export { composeProjectProblem } from "./ComposeImportDialog";

type InventoryState = "loading" | "ok" | "error";
const ENDED_BADLY = ["failed", "cancelled", "blocked"];

function workloadSectionFromLocation(): "install" | "manage" {
  const query = new URLSearchParams(window.location.search);
  return query.get("workloads") === "manage" || query.has("app") ? "manage" : "install";
}

const readTime = () => new Date().toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });

export function Workloads({ session }: { session: Session }) {
  const setupResumeKey = `vaelor.setup-assistant.operation.${session.user.username}`;
  const role = session.user.role;
  const [capabilities, setCapabilities] = useState<WorkloadCapabilities | null>(null);
  // When the setup check last answered: Reload reads it again, so the time is said, not "when the page opened".
  const [capabilitiesReadAt, setCapabilitiesReadAt] = useState<string | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [copilotSetup, setCopilotSetup] = useState<CopilotSetupData | null>(null);
  const [inventory, setInventory] = useState<ManagedInventory | null>(null);
  const [inventoryState, setInventoryState] = useState<InventoryState>("loading");
  const [applicationFeatures, setApplicationFeatures] = useState<ApplicationFeatures | null>(null);
  const [appTemplates, setAppTemplates] = useState<AppTemplate[]>([]);
  const [catalogReadError, setCatalogReadError] = useState(false);
  const [activeSection, setActiveSection] = useState<"install" | "manage">(workloadSectionFromLocation);
  // VD-200/201: AI Chat and Settings link here to add a connection; it opens
  // the Add a connection dialog once (`connectionFormHandoff`).
  const [installFlow, setInstallFlow] = useState<InstallFlow>(() => (connectionFormRequested() ? "connection" : null));
  useEffect(() => clearConnectionFormRequest(), []);
  // VD-189: the open dialog's refusal, shown in that dialog (the page under it is inert); a new flow starts clean.
  const [dialogError, setDialogError] = useState("");
  // Read when an asynchronous refusal lands, which may be after the flow changed.
  const installFlowNow = useRef(installFlow);
  installFlowNow.current = installFlow;
  useEffect(() => setDialogError(""), [installFlow]);
  // Whether the custom-application wizard opened to START A NEW request rather
  // than to RESUME a saved draft. A fresh open must not silently reopen the
  // last saved draft the container would otherwise restore.
  const [applicationStartFresh, setApplicationStartFresh] = useState(false);
  const [workloadPlannerRequest] = useState(() => {
    const request = window.sessionStorage.getItem("vaelor.workload-planner-request") ?? "";
    if (request) window.sessionStorage.removeItem("vaelor.workload-planner-request");
    return request;
  });
  const [agentMessage, setAgentMessage] = useState("");
  const [agentPlan, setAgentPlan] = useState<AgentPlan | null>(null);
  const [agentReview, setAgentReview] = useState(false);
  const [activePlanJobId, setActivePlanJobId] = useState(() => window.localStorage.getItem(setupResumeKey) ?? "");
  const [pendingModelDownload, setPendingModelDownload] = useState<ModelDownloadSelection | null>(null);
  // Artefacts whose download has already been accepted this session. The ref is
  // the guard; the state exists only so the approval control can disable itself.
  const requestedDownloads = useRef<Set<string>>(new Set());
  const [downloadingArtefacts, setDownloadingArtefacts] = useState<string[]>([]);
  const [pendingModelDeploy, setPendingModelDeploy] = useState<WorkloadJob | null>(null);
  const [agentBusy, setAgentBusy] = useState(false);
  const [agentNotice, setAgentNoticeText] = useState("");
  const [agentNoticeRefused, setAgentNoticeRefused] = useState(false); // VD-189 N3: a refusal is an alert
  // W6-D3: the job an in-progress notice ("... is starting it now") speaks
  // for. Once that job ends the sentence is stale, so it is not shown; the
  // operation's own row says how it ended.
  const [noticeJobId, setNoticeJobId] = useState("");
  const setAgentNotice = useCallback((message: string, refused = false) => { setAgentNoticeText(message); setAgentNoticeRefused(refused); setNoticeJobId(""); }, []);
  const setProgressNotice = useCallback((message: string, jobId: string) => { setAgentNoticeText(message); setAgentNoticeRefused(false); setNoticeJobId(jobId); }, []);
  // The on-device model install is a multi-minute, multi-GB job. The shared
  // hook posts it, polls its status and exposes the live job so its progress
  // is shown (identically from the first-run assistant panel).
  const npuInstall = useNpuInstall(session, setAgentNotice);
  const [conversationId, setConversationId] = useState("");
  const [modelRuntimeMode, setModelRuntimeMode] = useState<"efficient" | "balanced" | "quality">("balanced");
  const [discardedApplicationDraftId, setDiscardedApplicationDraftId] = useState("");
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const refreshInventory = useCallback(async () => {
    try {
      setInventory(await apiRequest<ManagedInventory>("/managed"));
      setInventoryState("ok");
    } catch {
      // The job ledger remains useful if inventory needs a manual refresh; a
      // list already read stays on screen rather than turning into nothing.
      setInventoryState("error");
    }
  }, []);
  const { jobs, refreshJobs, setJobs, stalledJobIds } = useWorkloadJobs({ role, onLifecycleChanged: refreshInventory });
  /*
   * Task #76. An app sat at "Needs attention" after its restart completed,
   * because inventory was re-read exactly once, when the job's own state last
   * changed - while the container's health was still `starting`. So while any
   * app is still moving, keep asking. Bounded: a container can sit in
   * `starting` for its whole start period, and an unbounded poll would be a
   * permanent request on a page nobody is looking at.
   */
  const [settleAttempt, setSettleAttempt] = useState(0);
  const settling = inventoryIsSettling(inventory?.apps ?? []);
  useEffect(() => {
    // The counter, not the inventory object, schedules the next read: a failed
    // request leaves the inventory identical and must not stop the loop.
    if (!settling) {
      if (settleAttempt !== 0) setSettleAttempt(0);
      return;
    }
    if (settleAttempt >= 40) return;
    const timer = window.setTimeout(() => {
      setSettleAttempt((attempt) => attempt + 1);
      void refreshInventory();
    }, 3000);
    return () => window.clearTimeout(timer);
  }, [refreshInventory, settleAttempt, settling]);
  const { applicationDeploymentDirty, applicationResearchRequest, applicationResume, researchedRequest, setApplicationDeploymentDirty, setResearchedRequest } = useApplicationResearchLifecycle(jobs);
  // Open the custom-application wizard. `fresh` starts a NEW request (blank
  // Request step, no resume); otherwise the wizard resumes the saved draft.
  const openApplicationResearch = useCallback((fresh: boolean) => {
    setApplicationStartFresh(fresh);
    setResearchedRequest("");
    setInstallFlow("researched");
  }, [setResearchedRequest]);
  const { catalogResume, clearCatalogResume, resumeCatalog } = useWorkloadCatalogState();
  const managedModels = inventory?.models ?? [];
  const activePlanJob = jobs.find((job) => job.id === activePlanJobId) ?? null;
  const activePlanJobEndedBadly = Boolean(activePlanJob && ENDED_BADLY.includes(activePlanJob.operation_state || activePlanJob.state));
  const noticeJob = noticeJobId ? jobs.find((job) => job.id === noticeJobId) : undefined;
  const shownNotice = noticeJob && jobIsTerminal(noticeJob) ? "" : agentNotice;
  const visibleApplicationResume = applicationResume?.draftId === discardedApplicationDraftId ? undefined : applicationResume;

  const selectSection = useCallback((section: "install" | "manage", replace = false) => {
    const url = new URL(window.location.href);
    url.searchParams.set("workloads", section);
    if (section === "install") {
      url.searchParams.delete("app");
      url.searchParams.delete("appTool");
    }
    window.history[replace ? "replaceState" : "pushState"]({}, "", url);
    setActiveSection(section);
  }, []);

  useEffect(() => {
    const restoreSection = () => setActiveSection(workloadSectionFromLocation());
    window.addEventListener("popstate", restoreSection);
    return () => window.removeEventListener("popstate", restoreSection);
  }, []);

  const preflightPort = useCallback(
    (port: number) => apiRequest<PortPreflight>(`/workloads/port-preflight?port=${encodeURIComponent(port)}`),
    [],
  );

  useEffect(() => {
    if (activePlanJobId) window.localStorage.setItem(setupResumeKey, activePlanJobId);
    else window.localStorage.removeItem(setupResumeKey);
  }, [activePlanJobId, setupResumeKey]);

  useEffect(() => {
    if (!activePlanJob) return;
    const state = activePlanJob.operation_state || activePlanJob.state;
    if (state === "completed") {
      setAgentNotice(activePlanJob.message || "Installation completed and the application passed its health check.");
    } else if (ENDED_BADLY.includes(state)) {
      setAgentNotice(activePlanJob.message || `Installation ${state}. Review the operation details before retrying.`);
    }
  }, [activePlanJob?.id, activePlanJob?.message, activePlanJob?.operation_state, activePlanJob?.state]);

  const closeApplicationDeployment = () => {
    if (!confirmApplicationDeploymentClose(applicationDeploymentDirty)) return;
    setApplicationDeploymentDirty(false);
    setInstallFlow(null);
  };

  useEffect(() => {
    // A request handed off from another surface (e.g. the assistant) is a NEW
    // request, not a resume: open fresh so it does not reopen a saved draft.
    if (applicationResearchRequest) { setApplicationStartFresh(true); setInstallFlow("researched"); }
  }, [applicationResearchRequest]);

  useEffect(() => {
    if (!workloadPlannerRequest) return;
    setAgentMessage(workloadPlannerRequest);
    setAgentNotice("");
    setAgentPlan(null);
    setInstallFlow("planner");
  }, [workloadPlannerRequest]);

  const readCatalog = useCallback(async () => {
    try {
      setAppTemplates(await apiRequest<AppTemplate[]>("/apps/catalog"));
      setCatalogReadError(false);
    } catch {
      setCatalogReadError(true);
    }
  }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    const results = await Promise.allSettled([
      apiRequest<WorkloadCapabilities>("/workloads/capabilities"),
      apiRequest<AgentStatus>("/agent/status"),
      apiRequest<CopilotSetupData>("/copilot/setup"),
      apiRequest<ManagedInventory>("/managed"),
      apiRequest<ApplicationFeatures>("/applications/features"),
      apiRequest<AppTemplate[]>("/apps/catalog"),
    ]);
    if (results[0].status === "fulfilled") {
      setCapabilities(results[0].value);
      setCapabilitiesReadAt(readTime());
    }
    if (results[1].status === "fulfilled") setAgentStatus(results[1].value);
    if (results[2].status === "fulfilled") setCopilotSetup(results[2].value);
    if (results[3].status === "fulfilled") setInventory(results[3].value);
    setInventoryState(results[3].status === "fulfilled" ? "ok" : "error");
    if (results[4].status === "fulfilled") setApplicationFeatures(results[4].value);
    if (results[5].status === "fulfilled") setAppTemplates(results[5].value);
    setCatalogReadError(results[5].status === "rejected");
    const failed = results.filter((result) => result.status === "rejected").length;
    setLoadError(failed ? `${failed} setup check${failed === 1 ? "" : "s"} could not be refreshed.` : "");
    if (role !== "viewer") {
      try {
        await refreshJobs();
      } catch {
        setLoadError((current) => current || "Recent setup activity could not be refreshed.");
      }
    }
    setLoading(false);
  }, [refreshJobs, role]);

  useEffect(() => {
    void refresh().catch(() => {
      setLoadError("This device could not be checked. Try again.");
      setLoading(false);
    });
  }, [refresh]);

  const postJob = (type: string, payload: Record<string, unknown>) => apiRequest<WorkloadJob>(
    "/jobs", { method: "POST", body: JSON.stringify({ type, payload }) }, session.csrf_token,
  );
  const failure = (error: unknown, fallback: string) => (error instanceof Error ? error.message : fallback);

  const jobAction = async (job: WorkloadJob, action: "cancel" | "retry") => {
    setAgentNotice("");
    try {
      await apiRequest(`/jobs/${job.id}/${action}`, { method: "POST", body: "{}" }, session.csrf_token);
      setAgentNotice(action === "cancel" ? "Cancellation requested." : "A safe retry was queued.");
      await refresh();
    } catch (error) {
      setAgentNotice(failure(error, "The job action failed."), true);
    }
  };

  const discardApplicationResearch = async () => {
    if (!visibleApplicationResume) return;
    const draftId = visibleApplicationResume.draftId;
    setAgentBusy(true);
    setAgentNotice("");
    try {
      await apiRequest(`/applications/drafts/${encodeURIComponent(draftId)}/discard`, { method: "POST", body: "{}" }, session.csrf_token);
      setDiscardedApplicationDraftId(draftId);
      setResearchedRequest("");
      setInstallFlow(null);
      await refreshJobs();
      setAgentNotice("Saved application research was discarded. No application was installed.");
    } catch (error) {
      // The saved record must never become a dead end. A terminal refusal (the
      // draft backs a running app, so it is not resumable research at all)
      // clears the banner; only a transient error keeps it so the reader can retry.
      if (error instanceof ApiError && error.code === "application_draft_discard_prohibited") {
        setDiscardedApplicationDraftId(draftId);
        setResearchedRequest("");
        setInstallFlow(null);
      }
      setAgentNotice(failure(error, "Saved application research could not be discarded."), true);
    } finally {
      setAgentBusy(false);
    }
  };

  /**
   * Approving a model download must be idempotent: a double-click once sent
   * two POSTs for the same artefact, two multi-gigabyte transfers on a Pi.
   * `requestedDownloads` latches the artefact the moment the first activation
   * is handled - a ref, so the second click is rejected even before React has
   * re-rendered the disabled button. The key ignores the inspection job so a
   * re-run compatibility check cannot re-queue a file already downloading.
   */
  const queueModelDownload = async (inspectionJobId: string, repo: string, file: string, sizeBytes: number) => {
    const artefact = `${repo}/${file}`;
    if (requestedDownloads.current.has(artefact)) return;
    requestedDownloads.current.add(artefact);
    setDownloadingArtefacts((current) => (current.includes(artefact) ? current : [...current, artefact]));
    setAgentBusy(true);
    try {
      const job = await postJob("model.download", { inspection_job_id: inspectionJobId, repo, file, size_bytes: sizeBytes });
      setJobs((current) => (current.some((existing) => existing.id === job.id) ? current : [job, ...current]));
      setActivePlanJobId(job.id);
      setAgentNotice("Exact model download approved. Vaelor will verify the file before deployment is offered.");
    } catch (error) {
      // Only a failed request releases the latch, so retrying stays possible
      // while an accepted download can never be queued twice.
      requestedDownloads.current.delete(artefact);
      setDownloadingArtefacts((current) => current.filter((item) => item !== artefact));
      setAgentNotice(failure(error, "The model download could not be queued."), true);
    } finally {
      setAgentBusy(false);
    }
  };

  // Whether THIS machine has a separate GPU AI-Chat tier. The setup payload's
  // `catalog_recommendation.served_on_gpu` is set by exactly the backend gate
  // the deploy router reads (`discover_gpu_rocm_serving`). Without it a
  // bring-your-own model must NOT be forced onto the ai-chat surface: omitting
  // the surface lets the backend default to `assistant`, which on a
  // single-accelerator box serves BOTH surfaces from one model (VD-071).
  const gpuChatTierAvailable = copilotSetup?.catalog_recommendation?.served_on_gpu === true;
  const modelDeployPayload = (path: string) => ({
    path, port: 0, mode: modelRuntimeMode,
    ...(gpuChatTierAvailable ? { surface: "ai-chat" as const } : {}),
  });

  const queueModelDeploy = async (job: WorkloadJob) => {
    if (!job.result?.path) return;
    setAgentBusy(true);
    try {
      const deployment = await postJob("model.deploy", modelDeployPayload(job.result.path));
      setJobs((current) => [deployment, ...current]);
      setActivePlanJobId(deployment.id);
      setAgentNotice("Local AI deployment approved. Vaelor will wait for its health check.");
    } catch (error) {
      setAgentNotice(failure(error, "The local AI server could not be queued."), true);
    } finally {
      setAgentBusy(false);
    }
  };

  const installTemplate = async (template: AppTemplate, port: number) => {
    setAgentBusy(true);
    setAgentNotice(""); setDialogError("");
    try {
      const job = await postJob("compose.install", { template: template.id, port });
      setJobs((current) => [job, ...current]);
      setActivePlanJobId(job.id);
      setProgressNotice(`${template.name} was approved. Vaelor is validating, downloading, and starting it now.`, job.id);
      clearCatalogResume();
      setInstallFlow(null);
    } catch (error) {
      setDialogError(failure(error, `${template.name} could not be queued.`));
    } finally {
      setAgentBusy(false);
    }
  };

  const readiness = dockerReadiness(capabilities, Boolean(loadError));

  const setupDocker = async (kind: "install" | "repair") => {
    setAgentBusy(true);
    setAgentNotice("");
    try {
      const job = kind === "install"
        ? await postJob("host.docker.install", { confirm: "install-docker" })
        : await postJob("host.docker.repair", { confirm: "repair-docker" });
      setJobs((current) => [job, ...current]);
      setAgentNotice(kind === "install"
        ? `Docker setup was approved for ${capabilities?.os?.name ?? "this host"}. Progress appears below; the dashboard will refresh when activation completes.`
        : "Docker repair was approved. Progress appears below; the dashboard refreshes when it completes.");
    } catch (error) {
      setAgentNotice(failure(error, kind === "install" ? "Docker setup could not be queued." : "Docker repair could not be queued."), true);
    } finally {
      setAgentBusy(false);
    }
  };

  const askAgent = async (message = agentMessage, bootstrap: "" | "model-install" = "") => {
    if (!message.trim()) return;
    setAgentBusy(true);
    setAgentNotice("");
    try {
      const plan = await apiRequest<AgentPlan>("/agent/plan", {
        method: "POST",
        body: JSON.stringify({ message, conversation_id: conversationId || undefined, bootstrap: bootstrap || undefined }),
      }, session.csrf_token);
      setAgentPlan(plan);
      if (plan.conversation_id) setConversationId(plan.conversation_id);
      setAgentMessage(message);
    } catch (error) {
      setAgentNotice(failure(error, "The setup plan could not be prepared."), true);
    } finally {
      setAgentBusy(false);
    }
  };

  const approvePlan = async () => {
    if (!agentPlan?.proposed_job) return;
    setAgentBusy(true);
    setAgentNotice(""); setDialogError("");
    try {
      const job = await apiRequest<WorkloadJob>("/jobs", { method: "POST", body: JSON.stringify(agentPlan.proposed_job) }, session.csrf_token);
      setJobs((current) => [job, ...current]);
      setActivePlanJobId(job.id);
      setAgentNotice(
        agentPlan.proposed_job.type === "model.inspect"
          ? "Compatibility check started. Verified model choices will appear here automatically."
          : "Setup approved. Progress and the verified next step remain in this assistant.",
      );
      setAgentReview(false);
      setAgentPlan(null);
      setAgentMessage("");
    } catch (error) {
      setDialogError(failure(error, "The setup could not be queued."));
    } finally {
      setAgentBusy(false);
    }
  };

  const chooseLocalModel = (query: string) => {
    setInstallFlow("planner");
    void askAgent(query, "model-install");
  };

  // The setup operation's progress. A blueprint install shows it on the page
  // (W4d-D11), and the planner shows it at its foot.
  const setupOperation = (
    <WorkloadOperation
      activePlanJob={activePlanJob}
      csrfToken={session.csrf_token}
      managedModels={managedModels}
      onChangeCatalogPort={(job) => {
        resumeCatalog(job);
        setActivePlanJobId("");
        setAgentNotice("Choose a different available port, then review the installation again.");
        setInstallFlow("catalog");
      }}
      onDone={() => {
        setActivePlanJobId("");
        setInstallFlow(null);
      }}
      onManageModels={() => {
        setActivePlanJobId("");
        setInstallFlow(null);
        setActiveSection("manage");
        void refresh();
      }}
      onResourceRefresh={async () => { await Promise.all([refreshJobs(), refreshInventory()]); }}
      onReviewModelDeploy={(job) => setPendingModelDeploy(job)}
      onReviewModelDownload={(selection) => setPendingModelDownload(selection)}
    />
  );

  // The boards count the installed apps ("Manage 6" beside six apps and three models).
  const manageCount = inventory ? inventory.apps.length : null;
  // "Apps and AI / Install"; on Manage the manager says which app is open.
  usePagePlace(activeSection === "install" ? [APPS_SECTION_LABELS.install] : null);
  const doorReasons = installDoorReasons(role, readiness, applicationFeatures, loading);

  return (
    <div className="workloads-page apps-page">
      {/* The Manage board opens straight on the tabs: the page's heading stays
          for assistive technology and is not drawn there. */}
      <div className={activeSection === "manage" ? "ui-page-header apps-page__header sr-only" : "ui-page-header apps-page__header"}>
        <div>
          <h1 className="ui-page-header__title">{destinations.workloads.name}</h1>
          <p className="apps-page__subtitle">Install apps and local models. Every change is shown before it happens.</p>
        </div>
      </div>

      {/* The boards draw the Docker pill and Reload in the top bar, on both tabs. */}
      <TopbarPageActions>
        <DockerPill readiness={readiness} />
        <Button busy={loading} onClick={() => void refresh()}>Reload</Button>
      </TopbarPageActions>
      <div className="apps-page__tabs">
        <TabSet
          items={[
            { id: "install", label: APPS_SECTION_LABELS.install },
            { id: "manage", label: <>{APPS_SECTION_LABELS.manage}{manageCount !== null && <span className="apps-page__tab-count">{manageCount}</span>}</> },
          ]}
          label="Apps and AI"
          listClassName="apps-page__tablist"
          panelClassName="apps-page__panel"
          onSelect={(id) => selectSection(id === "manage" ? "manage" : "install")}
          selectedId={activeSection}
        >
          {activeSection === "install" ? <>
            <WorkloadsReadinessCard
              administrator={role === "administrator"}
              busy={agentBusy}
              capabilities={capabilities}
              checkedAt={capabilitiesReadAt}
              onInstallDocker={() => void setupDocker("install")}
              onRepairDocker={() => void setupDocker("repair")}
              readiness={readiness}
            />
            <WorkloadsInstallCards
              onOpen={(door) => {
                if (door === "researched") openApplicationResearch(true);
                else {
                  setInstallFlow(door);
                  if (door === "copilot" && !copilotSetup) void refresh();
                }
              }}
              reasons={doorReasons}
            />
            <WorkloadsMessages
              busy={agentBusy}
              loadError={loadError}
              loading={loading}
              notice={installFlow === "planner" ? "" : shownNotice}
              noticeRefused={agentNoticeRefused || activePlanJobEndedBadly}
              onDiscardResearch={() => void discardApplicationResearch()}
              onResumeResearch={() => openApplicationResearch(false)}
              onRetry={() => void refresh()}
              savedResearch={visibleApplicationResume}
            />
            {installFlow !== "planner" && activePlanJob?.type === "compose.install" && setupOperation}
            <WorkloadJobActivity
              applicationResumeJobId={visibleApplicationResume?.jobId}
              jobs={jobs}
              managedModels={managedModels}
              onJobAction={(job, action) => void jobAction(job, action)}
              onQueueModelDeploy={(job) => void queueModelDeploy(job)}
              onQueueModelDownload={(jobId, repo, file, sizeBytes) => void queueModelDownload(jobId, repo, file, sizeBytes)}
              onResumeApplicationResearch={() => openApplicationResearch(false)}
              requestedDownloads={downloadingArtefacts}
              stalledJobIds={stalledJobIds}
            />
            <WorkloadsRunningNow
              inventory={inventory}
              inventoryState={inventoryState}
              manageCount={manageCount}
              onManage={() => selectSection("manage")}
              onRetry={() => void refreshInventory()}
            />
          </> : <WorkloadManager inventory={inventory} inventoryState={inventoryState} onRefresh={() => void refresh()} session={session} />}
        </TabSet>
      </div>

      <WorkloadCatalogModal
        busy={agentBusy} error={dialogError}
        disabled={!readiness.composeReady || role === "viewer"}
        disabledReason={doorReasons.catalog}
        onClose={() => { clearCatalogResume(); setInstallFlow(null); }}
        onDismiss={() => setInstallFlow(null)}
        onInstall={installTemplate}
        onPreflight={preflightPort}
        onRetry={() => void readCatalog()}
        open={installFlow === "catalog"}
        readError={catalogReadError}
        resume={catalogResume}
        templates={appTemplates}
        installedTemplateIds={(inventory?.apps ?? []).flatMap((app) => [app.project, app.id].filter((value): value is string => Boolean(value)))}
        onOpenInstalled={() => { setInstallFlow(null); selectSection("manage"); }}
      />

      {installFlow === "model" && (
        <ModelCatalog busy={agentBusy} disabled={role === "viewer"} onChoose={chooseLocalModel} onClose={() => setInstallFlow(null)} setup={copilotSetup} />
      )}

      {installFlow === "copilot" && copilotSetup && (
        <CopilotSetupDialog
          busy={agentBusy}
          data={copilotSetup}
          error={dialogError}
          session={session}
          onClose={() => setInstallFlow(null)}
          onChooseLocal={(query, mode) => {
            setModelRuntimeMode(mode);
            chooseLocalModel(query);
          }}
          onConnectModel={() => setInstallFlow("connection")}
          onInstallNpuRelease={(tag) => {
            // A release-sourced NPU model does not go through the planner: the
            // shared hook posts the install and polls it, and its dialog shows
            // the ~3.4 GB download instead of it running silently.
            setInstallFlow(null);
            npuInstall.start(tag);
          }}
          onChooseBasic={() => {
            // VD-049 / VD-201: one way back for an Assistant on an added connection.
            void apiRequest(
              "/assistant/preferences",
              { method: "PATCH", body: JSON.stringify({ intelligence_choice: "basic" }) },
              session.csrf_token,
            ).then(
              () => setAgentNotice("Built-in basic mode is on for your account: your Assistant answers no longer use a model."),
              (error) => {
                const refusal = failure(error, "The Assistant choice could not be saved.");
                if (installFlowNow.current === "copilot") setDialogError(refusal);
                else setAgentNotice(refusal, true);
              },
            );
          }}
        />
      )}

      {installFlow === "connection" && (
        <AddConnectionDialog onClose={() => setInstallFlow(null)} onNotice={setAgentNotice} session={session} />
      )}

      {npuInstall.job && (
        <NpuInstallDialog job={npuInstall.job} onDismiss={npuInstall.dismiss} onRetryRead={npuInstall.retryRead} readLost={npuInstall.readLost} />
      )}

      {installFlow === "custom" && (
        <ComposeImportDialog
          csrfToken={session.csrf_token}
          onClose={() => setInstallFlow(null)}
          onQueued={(job) => {
            setJobs((current) => [job, ...current]);
            setInstallFlow(null);
            setProgressNotice("Custom stack approved. Vaelor will normalize it, enforce safety policy, and deploy it.", job.id);
          }}
        />
      )}

      {installFlow === "researched" && (
        <ApplicationDeploymentContainer
          autoClassify={Boolean(researchedRequest)}
          initialRequest={researchedRequest || (applicationStartFresh ? "" : visibleApplicationResume?.requestSummary) || ""}
          resumeResearch={applicationStartFresh ? undefined : visibleApplicationResume}
          startFresh={applicationStartFresh}
          session={session}
          onClose={closeApplicationDeployment}
          onDirtyChange={setApplicationDeploymentDirty}
          onJobQueued={() => void refresh()}
          onManaged={() => {
            setInstallFlow(null);
            setActiveSection("manage");
            void refresh();
          }}
        />
      )}

      {installFlow === "planner" && (
        <WorkloadPlannerDialog
          agentStatus={agentStatus}
          busy={agentBusy}
          canPlan={role !== "viewer"}
          message={agentMessage}
          notice={shownNotice}
          noticeRefused={agentNoticeRefused || activePlanJobEndedBadly}
          onAsk={() => void askAgent()}
          onBack={() => setAgentPlan(null)}
          onClose={() => setInstallFlow(null)}
          onMessageChange={(value) => { setAgentMessage(value); setAgentNotice(""); }}
          onResearch={() => {
            setApplicationStartFresh(true);
            setResearchedRequest(agentMessage.trim() || agentPlan?.application_intent?.application_query || "");
            setAgentNotice("");
            setAgentPlan(null);
            setInstallFlow("researched");
          }}
          onReview={() => setAgentReview(true)}
          plan={agentPlan}
          setupOperation={setupOperation}
        />
      )}

      <ActionReviewDialog
        busy={agentBusy} error={agentReview ? dialogError : undefined}
        evidence={[{ source: "setup-assistant.plan", summary: "The exact server-owned operation is shown before authorization." }]}
        job={agentReview ? agentPlan?.proposed_job ?? null : pendingModelDownload ? { type: "model.download", payload: { inspection_job_id: pendingModelDownload.inspectionJobId, repo: pendingModelDownload.repo, file: pendingModelDownload.file, size_bytes: pendingModelDownload.sizeBytes } } : pendingModelDeploy?.result?.path ? { type: "model.deploy", payload: modelDeployPayload(pendingModelDeploy.result.path) } : null}
        onApprove={() => {
          if (agentReview) void approvePlan();
          else if (pendingModelDownload) {
            const choice = pendingModelDownload;
            setPendingModelDownload(null);
            void queueModelDownload(choice.inspectionJobId, choice.repo, choice.file, choice.sizeBytes);
          } else if (pendingModelDeploy) {
            const job = pendingModelDeploy;
            setPendingModelDeploy(null);
            void queueModelDeploy(job);
          }
        }}
        onCancel={() => { setDialogError(""); setAgentReview(false); setPendingModelDownload(null); setPendingModelDeploy(null); }}
        summary={agentReview ? agentPlan?.summary ?? "Review the exact setup action." : pendingModelDownload ? `Download the verified ${pendingModelDownload.file} model file.` : pendingModelDeploy ? "Start the downloaded model as a managed local AI service." : ""}
        suggestedActions={["Keep this assistant open for progress and the verified next step."]}
      />
    </div>
  );
}
