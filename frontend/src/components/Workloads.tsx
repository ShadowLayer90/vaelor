import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, apiRequest } from "../lib/api";
import type { Session } from "../types";
import { CopilotSetup, type CopilotSetupData } from "./CopilotSetup";
import { ActionReviewDialog } from "./ActionReviewDialog";
import { ApplicationDeploymentContainer } from "./ApplicationDeploymentContainer";
import { type AppTemplate, type PortPreflight } from "./AppCatalog";
import { Icon } from "./Icon";
import { ModelCatalog } from "./ModelCatalog";
import { ModalShell } from "./ModalShell";
import { StatusPill } from "./StatusPill";
import { UpdateJobStatus } from "./UpdateJobStatus";
import { useNpuInstall } from "../hooks/useNpuInstall";
import { Button, Input, Notice, TabSet, Textarea } from "./ui";
import { WorkloadManager, inventoryIsSettling, type ManagedInventory } from "./WorkloadManager";
import { useApplicationResearchLifecycle, confirmApplicationDeploymentClose } from "./workloads-application-research";
import { WorkloadCatalogModal, useWorkloadCatalogState } from "./workloads-catalog";
import { WorkloadJobActivity, WorkloadOperation, type ModelDownloadSelection } from "./workloads-jobs";
import type { AgentPlan, AgentStatus, ApplicationFeatures, InstallFlow, WorkloadCapabilities, WorkloadJob } from "./workloads-types";
import { useWorkloadJobs } from "../hooks/useWorkloadJobs";
import { destinations } from "../lib/destinations";

export { applicationResumeRequestSummary, confirmApplicationDeploymentClose } from "./workloads-application-research";
export { catalogFailureNeedsPortChange } from "./workloads-catalog";

// Shown as the Compose editor's placeholder, not pre-filled content, so the
// caret starts a clean document instead of landing inside example text.
const COMPOSE_EXAMPLE = "services:\n  app:\n    image: nginx:stable-alpine\n    restart: unless-stopped\n    ports:\n      - \"8088:80\"\n    mem_limit: 256m\n    cpus: \"1.0\"\n";

export function composeProjectProblem(value: string) {
  const valid = /^[A-Za-z0-9][A-Za-z0-9_-]{1,47}$/.test(value);
  return value === "" || valid ? "" : (
    "Use 2-48 characters, starting with a letter or number; only letters, numbers, hyphens, and underscores are allowed."
  );
}

function workloadSectionFromLocation(): "install" | "manage" {
  const query = new URLSearchParams(window.location.search);
  return query.get("workloads") === "manage" || query.has("app") ? "manage" : "install";
}

export function Workloads({ session }: { session: Session }) {
  const setupResumeKey = `vaelor.setup-assistant.operation.${session.user.username}`;
  const [capabilities, setCapabilities] = useState<WorkloadCapabilities | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [copilotSetup, setCopilotSetup] = useState<CopilotSetupData | null>(null);
  const [inventory, setInventory] = useState<ManagedInventory | null>(null);
  const [applicationFeatures, setApplicationFeatures] = useState<ApplicationFeatures | null>(null);
  const [appTemplates, setAppTemplates] = useState<AppTemplate[]>([]);
  const [activeSection, setActiveSection] = useState<"install" | "manage">(workloadSectionFromLocation);
  const [installFlow, setInstallFlow] = useState<InstallFlow>(null);
  // Whether the custom-application wizard was opened to START A NEW request
  // ("Describe a custom application", the planner handoff) rather than to RESUME
  // a saved draft (the resume banner / activity log). A fresh open must not
  // silently reopen the last saved draft the container would otherwise restore.
  const [applicationStartFresh, setApplicationStartFresh] = useState(false);
  const [workloadPlannerRequest] = useState(() => {
    const request = window.sessionStorage.getItem("vaelor.workload-planner-request") ?? "";
    if (request) window.sessionStorage.removeItem("vaelor.workload-planner-request");
    return request;
  });
  const [showCopilotSetup, setShowCopilotSetup] = useState(false);
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
  const [agentNotice, setAgentNotice] = useState("");
  // The on-device model install is a multi-minute, multi-GB job. It used to
  // fire-and-forget behind a one-line notice while the modal closed to the
  // dashboard, so a person had no idea it was running - the setup UX defect.
  // The shared hook posts it, polls its status and exposes the live job so its
  // progress is shown here (and identically from the first-run assistant panel).
  const npuInstall = useNpuInstall(session, setAgentNotice);
  const [conversationId, setConversationId] = useState("");
  const [modelRuntimeMode, setModelRuntimeMode] = useState<"efficient" | "balanced" | "quality">("balanced");
  const [customProject, setCustomProject] = useState("");
  const [customReview, setCustomReview] = useState(false);
  const [customComposeError, setCustomComposeError] = useState("");
  const [discardedApplicationDraftId, setDiscardedApplicationDraftId] = useState("");
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  // Starts empty so typing begins a clean document; the example is shown as a
  // placeholder (see COMPOSE_EXAMPLE) rather than pre-filled editable content.
  const [customCompose, setCustomCompose] = useState("");
  const customProjectProblem = composeProjectProblem(customProject);
  const refreshInventory = useCallback(async () => {
    try {
      setInventory(await apiRequest<ManagedInventory>("/managed"));
    } catch {
      // The job ledger remains useful if inventory needs a manual refresh.
    }
  }, []);
  const { jobs, refreshJobs, setJobs } = useWorkloadJobs({ role: session.user.role, onLifecycleChanged: refreshInventory });
  /*
   * Task #76. An app sat at "Needs attention" for 35-60+ seconds after
   * "App restart completed / FINISHED / 100%" with a passing health preflight,
   * and never cleared — while clicking Refresh flipped it to RUNNING at once.
   * The server was right the whole time and the client was stale.
   *
   * The cause is that inventory is re-read exactly once, when the job's own
   * state last changes. A container's health is still `starting` at that
   * instant and settles seconds later, and nothing was watching for the
   * second event. So while any app is still moving, keep asking.
   *
   * Bounded, because a container can sit in `starting` for as long as its
   * start period lasts and an unbounded poll would be a permanent background
   * request on a page nobody is looking at. When the budget runs out the view
   * simply stops updating itself, which is what it did before this existed.
   */
  const [settleAttempt, setSettleAttempt] = useState(0);
  const settling = inventoryIsSettling(inventory?.apps ?? []);
  useEffect(() => {
    // The counter, not the inventory object, is what schedules the next read.
    // A failed request leaves the inventory identical, and depending on it
    // would stop the loop on the first transient error.
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
    } else if (["failed", "cancelled", "blocked"].includes(state)) {
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
    if (results[0].status === "fulfilled") setCapabilities(results[0].value);
    if (results[1].status === "fulfilled") setAgentStatus(results[1].value);
    if (results[2].status === "fulfilled") setCopilotSetup(results[2].value);
    if (results[3].status === "fulfilled") setInventory(results[3].value);
    if (results[4].status === "fulfilled") setApplicationFeatures(results[4].value);
    if (results[5].status === "fulfilled") setAppTemplates(results[5].value);
    const failed = results.slice(0, 6).filter((result) => result.status === "rejected").length;
    setLoadError(failed ? `${failed} setup check${failed === 1 ? "" : "s"} could not be refreshed.` : "");
    if (session.user.role !== "viewer") {
      try {
        await refreshJobs();
      } catch {
        setLoadError((current) => current || "Recent setup activity could not be refreshed.");
      }
    }
    setLoading(false);
  }, [refreshJobs, session.user.role]);

  useEffect(() => {
    void refresh().catch(() => {
      setLoadError("This device could not be checked. Try again.");
      setLoading(false);
    });
  }, [refresh]);

  const jobAction = async (job: WorkloadJob, action: "cancel" | "retry") => {
    setAgentNotice("");
    try {
      await apiRequest(
        `/jobs/${job.id}/${action}`,
        { method: "POST", body: "{}" },
        session.csrf_token,
      );
      setAgentNotice(action === "cancel" ? "Cancellation requested." : "A safe retry was queued.");
      await refresh();
    } catch (error) {
      setAgentNotice(error instanceof Error ? error.message : "The job action failed.");
    }
  };

  const discardApplicationResearch = async () => {
    if (!visibleApplicationResume) return;
    const draftId = visibleApplicationResume.draftId;
    setAgentBusy(true);
    setAgentNotice("");
    try {
      await apiRequest(
        `/applications/drafts/${encodeURIComponent(draftId)}/discard`,
        { method: "POST", body: "{}" },
        session.csrf_token,
      );
      setDiscardedApplicationDraftId(draftId);
      setResearchedRequest("");
      setInstallFlow(null);
      await refreshJobs();
      setAgentNotice("Saved application research was discarded. No application was installed.");
    } catch (error) {
      // The saved record must never become a dead end. A terminal refusal (the
      // draft backs a running app, so it is not resumable research at all)
      // clears the banner and points the reader at the running service; only a
      // transient error keeps the banner so the reader can retry.
      if (error instanceof ApiError && error.code === "application_draft_discard_prohibited") {
        setDiscardedApplicationDraftId(draftId);
        setResearchedRequest("");
        setInstallFlow(null);
      }
      setAgentNotice(error instanceof Error ? error.message : "Saved application research could not be discarded.");
    } finally {
      setAgentBusy(false);
    }
  };

  /**
   * Approving a model download must be idempotent.
   *
   * The approval control had no busy state, so a double-click sent two POSTs
   * for the same artefact. On a Pi that is two multi-gigabyte transfers.
   * `requestedDownloads` latches the artefact the moment the first activation
   * is handled — a ref, not state, so the second click of a double-click is
   * rejected even if it lands before React has re-rendered the disabled
   * button. The artefact key deliberately ignores the inspection job so a
   * re-run compatibility check cannot re-queue a file already downloading.
   */
  const downloadArtefactKey = (repo: string, file: string) => `${repo}/${file}`;
  const queueModelDownload = async (inspectionJobId: string, repo: string, file: string, sizeBytes: number) => {
    const artefact = downloadArtefactKey(repo, file);
    if (requestedDownloads.current.has(artefact)) return;
    requestedDownloads.current.add(artefact);
    setDownloadingArtefacts((current) => (current.includes(artefact) ? current : [...current, artefact]));
    setAgentBusy(true);
    try {
      const job = await apiRequest<WorkloadJob>("/jobs", {
        method: "POST",
        body: JSON.stringify({ type: "model.download", payload: {
          inspection_job_id: inspectionJobId, repo, file, size_bytes: sizeBytes,
        } }),
      }, session.csrf_token);
      setJobs((current) => (current.some((existing) => existing.id === job.id) ? current : [job, ...current]));
      setActivePlanJobId(job.id);
      setAgentNotice("Exact model download approved. Vaelor will verify the file before deployment is offered.");
    } catch (error) {
      // Only a failed request releases the latch, so retrying stays possible
      // while an accepted download can never be queued twice.
      requestedDownloads.current.delete(artefact);
      setDownloadingArtefacts((current) => current.filter((item) => item !== artefact));
      setAgentNotice(error instanceof Error ? error.message : "The model download could not be queued.");
    } finally {
      setAgentBusy(false);
    }
  };

  // Whether THIS machine has a separate GPU AI-Chat tier. The setup payload's
  // `catalog_recommendation.served_on_gpu` is set true by exactly the same
  // backend gate the deploy router reads (`discover_gpu_rocm_serving`): a gfx1151
  // GPU with the ROCmFPX fork and runtime present. On a Pi / x86 box without it
  // the field is absent, and a bring-your-own model must NOT be forced onto the
  // ai-chat surface (see `queueModelDeploy`).
  const gpuChatTierAvailable = copilotSetup?.catalog_recommendation?.served_on_gpu === true;
  const modelDeployPayload = (path: string) => ({
    path, port: 0, mode: modelRuntimeMode,
    ...(gpuChatTierAvailable ? { surface: "ai-chat" as const } : {}),
  });

  const queueModelDeploy = async (job: WorkloadJob) => {
    if (!job.result?.path) return;
    setAgentBusy(true);
    try {
      const deployment = await apiRequest<WorkloadJob>("/jobs", {
        method: "POST",
        // This is the AI-Chat local model catalog (the Assistant is a separate
        // NPU flow with its own install endpoint), so the deploy asks for the
        // AI-Chat surface ONLY on a box that actually has a GPU AI-Chat tier
        // (`gpuChatTierAvailable`). On a single-model box (a CPU-only Pi with no
        // GPU fork), forcing `surface:"ai-chat"` for a bring-your-own GGUF makes
        // the compose path claim only the ai-chat credential and leaves the
        // Assistant's deployment-agent unclaimed; omitting it lets the backend
        // default to `assistant`, which on a single-accelerator box serves BOTH
        // surfaces from one model (VD-071). A stock catalog entry that names its
        // own surface still wins server-side either way.
        body: JSON.stringify({ type: "model.deploy", payload: modelDeployPayload(job.result.path) }),
      }, session.csrf_token);
      setJobs((current) => [deployment, ...current]);
      setActivePlanJobId(deployment.id);
      setAgentNotice("Local AI deployment approved. Vaelor will wait for its health check.");
    } catch (error) {
      setAgentNotice(error instanceof Error ? error.message : "The local AI server could not be queued.");
    } finally {
      setAgentBusy(false);
    }
  };

  const installTemplate = async (template: AppTemplate, port: number) => {
    setAgentBusy(true);
    setAgentNotice("");
    try {
      const job = await apiRequest<WorkloadJob>("/jobs", {
        method: "POST",
        body: JSON.stringify({ type: "compose.install", payload: { template: template.id, port } }),
      }, session.csrf_token);
      setJobs((current) => [job, ...current]);
      setActivePlanJobId(job.id);
      setAgentNotice(`${template.name} was approved. Vaelor is validating, downloading, and starting it now.`);
      clearCatalogResume();
      setInstallFlow("planner");
    } catch (error) {
      setAgentNotice(error instanceof Error ? error.message : `${template.name} could not be queued.`);
    } finally {
      setAgentBusy(false);
    }
  };

  // Docker installed and answering is not Docker able to start containers. A
  // wiped/broken data-root leaves the daemon up but every `docker run` failing;
  // the backend reports it as runtime "storage_missing". A node in that state is
  // NOT ready to run apps, and the fix is Repair, not Install (see
  // docker_health.py). Any other value — "ok", "unknown", or missing (older
  // backend / fallback dict) — is treated as healthy, so this never cries wolf
  // on a fine box, and in particular never on a healthy box whose control-plane
  // user simply cannot reach the Docker socket.
  const dockerRuntimeBroken = capabilities?.docker.runtime === "storage_missing";
  const composeReady = Boolean(capabilities?.docker.installed && capabilities?.docker.compose && !dockerRuntimeBroken);
  const dockerInstallAvailable = Boolean(
    !capabilities?.docker.installed && capabilities?.docker.installation?.available,
  );
  const dockerRepairAvailable = Boolean(capabilities?.docker.installed && dockerRuntimeBroken);

  const installDocker = async () => {
    setAgentBusy(true);
    setAgentNotice("");
    try {
      const job = await apiRequest<WorkloadJob>("/jobs", {
        method: "POST",
        body: JSON.stringify({
          type: "host.docker.install",
          payload: { confirm: "install-docker" },
        }),
      }, session.csrf_token);
      setJobs((current) => [job, ...current]);
      setAgentNotice(`Docker setup was approved for ${capabilities?.os?.name ?? "this host"}. Progress appears below; the dashboard will refresh when activation completes.`);
    } catch (error) {
      setAgentNotice(error instanceof Error ? error.message : "Docker setup could not be queued.");
    } finally {
      setAgentBusy(false);
    }
  };

  const repairDocker = async () => {
    setAgentBusy(true);
    setAgentNotice("");
    try {
      const job = await apiRequest<WorkloadJob>("/jobs", {
        method: "POST",
        body: JSON.stringify({
          type: "host.docker.repair",
          payload: { confirm: "repair-docker" },
        }),
      }, session.csrf_token);
      setJobs((current) => [job, ...current]);
      setAgentNotice("Docker repair was approved. Progress appears below; the dashboard refreshes when it completes.");
    } catch (error) {
      setAgentNotice(error instanceof Error ? error.message : "Docker repair could not be queued.");
    } finally {
      setAgentBusy(false);
    }
  };

  const askAgent = async (
    message = agentMessage,
    bootstrap: "" | "model-install" = "",
  ) => {
    if (!message.trim()) return;
    setAgentBusy(true);
    setAgentNotice("");
    try {
      const plan = await apiRequest<AgentPlan>(
        "/agent/plan",
        {
          method: "POST",
          body: JSON.stringify({
            message,
            conversation_id: conversationId || undefined,
            bootstrap: bootstrap || undefined,
          }),
        },
        session.csrf_token,
      );
      setAgentPlan(plan);
      if (plan.conversation_id) setConversationId(plan.conversation_id);
      setAgentMessage(message);
    } catch (error) {
      setAgentNotice(error instanceof Error ? error.message : "The setup plan could not be prepared.");
    } finally {
      setAgentBusy(false);
    }
  };

  const approvePlan = async () => {
    if (!agentPlan?.proposed_job) return;
    setAgentBusy(true);
    setAgentNotice("");
    try {
      const job = await apiRequest<WorkloadJob>(
        "/jobs",
        { method: "POST", body: JSON.stringify(agentPlan.proposed_job) },
        session.csrf_token,
      );
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
      setAgentNotice(error instanceof Error ? error.message : "The setup could not be queued.");
    } finally {
      setAgentBusy(false);
    }
  };

  const importCompose = async () => {
    setAgentBusy(true); setAgentNotice(""); setCustomComposeError("");
    try {
      const job = await apiRequest<WorkloadJob>("/jobs", {
        method: "POST", body: JSON.stringify({
          type: "compose.import",
          payload: { project: customProject.trim().toLowerCase(), content: customCompose },
        }),
      }, session.csrf_token);
      setJobs((current) => [job, ...current]);
      setInstallFlow(null);
      setAgentNotice("Custom stack approved. Vaelor will normalize it, enforce safety policy, and deploy it.");
    } catch (error) {
      setCustomComposeError(error instanceof Error ? error.message : "The custom stack could not be queued.");
    } finally { setAgentBusy(false); }
  };

  return (
    <div className="workloads-page">
      <div className="page-heading">
        <div>
          <h1>{destinations.workloads.name}</h1>
          <p>{destinations.workloads.descriptor}. Vaelor handles the Linux setup and shows every change before it happens.</p>
        </div>
        <StatusPill label={composeReady ? "Docker ready" : dockerRepairAvailable ? "Docker needs repair" : dockerInstallAvailable ? "Docker setup available" : loadError ? "Check interrupted" : "OS limits apply"} status={composeReady ? "healthy" : dockerRepairAvailable ? "degraded" : loadError ? "degraded" : "neutral"} />
      </div>

      {/* These were tab-shaped buttons with `role="tab"` and nothing else: no
          tablist owner for the panel, no `aria-controls`, no roving tabindex,
          and arrow keys did nothing. The shared TabSet primitive already
          implements the whole ARIA tabs pattern, so the pattern is fixed once
          here instead of being re-hand-rolled. */}
      <TabSet
        items={[
          { id: "install", label: <><Icon name="bolt" />Install</> },
          {
            id: "manage",
            label: <><Icon name="settings" />Manage <span>{(inventory?.apps.length ?? 0) + (inventory?.models.length ?? 0)}</span></>,
          },
        ]}
        label="Apps and AI"
        listClassName="workload-section-tabs"
        panelClassName="workload-tab-panel"
        onSelect={(id) => selectSection(id === "manage" ? "manage" : "install")}
        selectedId={activeSection}
      >
      {activeSection === "install" ? <>
      {loadError && (
        <Notice severity="warning">
          <span>{loadError} Existing tools remain available.</span>
          <Button variant="quiet" disabled={loading} onClick={() => void refresh()} type="button">
              {loading ? "Checking…" : "Try again"}
          </Button>
        </Notice>
      )}
      <section className="workload-runtime" aria-label="App setup readiness">
        <div className="workload-runtime__orb"><Icon name="database" size={34} /></div>
        <div>
          <small>Setup check</small>
          {/*
            * `capabilities` is undefined until the check answers, and a check
            * that has not answered is not a machine without Docker. Saying so
            * is the whole of task #120: for about a minute after every restart
            * this read "Container apps are unavailable on this OS", then
            * "Docker Compose is not ready", on an appliance running five
            * containers — and offered to install Docker over the top of the
            * Docker it already had.
            */}
          <strong>{!capabilities ? "Checking whether this node can run apps" : composeReady ? "This Vaelor node is ready to run apps" : dockerRepairAvailable ? "Docker needs repair on this node" : dockerInstallAvailable ? `Docker is not installed on ${capabilities?.os?.name ?? "this host"}` : capabilities?.docker.installed ? "Docker Compose is not ready" : "Container apps are unavailable on this OS"}</strong>
          <span>{!capabilities ? "Nothing has been established yet, so nothing here is a verdict about this machine." : composeReady ? "No terminal or Linux commands required." : dockerRepairAvailable ? capabilities?.docker.runtime_reason ?? "Docker's storage is not healthy." : capabilities?.docker.installation?.reason ?? "Checking the operating-system integration."}</span>
          {/* Mutually exclusive: a broken-but-installed Docker offers Repair;
              an absent Docker offers Install. dockerRepairAvailable requires
              installed, dockerInstallAvailable requires NOT installed, so both
              can never render at once. */}
          {capabilities && dockerRepairAvailable && session.user.role === "administrator" && (
            <Button variant="primary" disabled={agentBusy} onClick={() => void repairDocker()} type="button">
                {agentBusy ? "Preparing repair…" : "Repair Docker"}
            </Button>
          )}
          {capabilities && dockerInstallAvailable && session.user.role === "administrator" && (
            <Button variant="primary" disabled={agentBusy} onClick={() => void installDocker()} type="button">
                {agentBusy ? "Preparing setup…" : "Review and install Docker"}
            </Button>
          )}
        </div>
        <details className="advanced-details">
          <summary>Advanced system details</summary>
          <div className="workload-runtime__paths">
            <span><small>App engine</small><strong>{capabilities?.docker.compose_version ? `Docker Compose ${capabilities.docker.compose_version}` : "Not ready"}</strong></span>
            <span><small>Operating system</small><strong>{capabilities?.os?.name ?? "Detecting"}</strong></span>
            <span><small>Processor</small><strong>{capabilities?.runtime?.architecture ?? "Detecting"}</strong></span>
            <span><small>App storage</small><strong>{capabilities?.roots?.workloads ?? "/var/lib/vaelor/workloads"}</strong></span>
            <span><small>Model storage</small><strong>{capabilities?.roots?.models ?? "/var/lib/vaelor/models"}</strong></span>
          </div>
        </details>
      </section>

      <div className="workload-product-grid">
        <article className="workload-product workload-product--compose">
          <div className="workload-product__top"><span><Icon name="database" size={22} /></span><small>REVIEWED BLUEPRINTS</small></div>
          <h2>Blueprint apps</h2>
          <p>Choose an application Vaelor already knows how to install, update, back up, restore, and remove.</p>
          <div className="workload-product__features"><span>Known configuration</span><span>Reviewed template</span><span>Managed lifecycle</span></div>
          <Button variant="primary" disabledReason={session.user.role === "viewer" ? "Operator access is required to install an app." : composeReady ? undefined : "Docker is not ready on this node yet. Finish the setup check above first."} onClick={() => setInstallFlow("catalog")}>Browse blueprints</Button>
        </article>

        <article className="workload-product workload-product--ai">
          <div className="workload-product__top"><span><Icon name="cpu" size={22} /></span><small>RUN LOCAL AI</small></div>
          <h2>Install an AI model</h2>
          <p>Find a model that fits this node. Vaelor explains speed, memory use, and storage in plain language.</p>
          <div className="workload-product__features"><span>Recommendations</span><span>Memory check</span><span>Private by default</span></div>
          <Button disabled={!composeReady || session.user.role === "viewer"} onClick={() => { setInstallFlow("model"); setShowCopilotSetup(false); }}>Choose a model</Button>
        </article>

        <article className="workload-product workload-product--agent">
          <div className="workload-product__top"><span><Icon name="shield" size={22} /></span><small>SET UP ASSISTANT</small></div>
          <h2>Improve the assistant</h2>
          <p>Choose private local AI or connect a hosted frontier model for more capable setup help.</p>
          <div className="workload-product__features"><span>Hardware-aware</span><span>Local or hosted</span><span>Approval required</span></div>
          <Button disabled={session.user.role === "viewer"} onClick={() => { setInstallFlow("copilot"); setShowCopilotSetup(true); if (!copilotSetup) void refresh(); }}>Set up assistant</Button>
        </article>

        <article className="workload-product workload-product--custom">
          <div className="workload-product__top"><span><Icon name="terminal" size={22} /></span><small>ADVANCED, GUARDED</small></div>
          <h2>Import a Docker stack</h2>
          <p>Paste Compose YAML. Vaelor blocks unsafe access, checks limits and ports, then shows live progress.</p>
          <div className="workload-product__features"><span>Policy scan</span><span>Automatic backup</span><span>Recoverable removal</span></div>
          <Button disabled={!composeReady || session.user.role !== "administrator"} onClick={() => { setCustomComposeError(""); setInstallFlow("custom"); }}>Open composer</Button>
        </article>
        <article className="workload-product workload-product--planner">
          <div className="workload-product__top"><span><Icon name="memory" size={22} /></span><small>ASSISTED RESEARCH</small></div>
          <h2>Custom application</h2>
          <p>Describe an application outside the blueprint catalog. Vaelor researches it, asks what is missing, and prepares an exact plan.</p>
          <div className="workload-product__features"><span>Guided request</span><span>Cited research</span><span>Approval before install</span></div>
          <Button disabled={!applicationFeatures?.research || session.user.role === "viewer"} onClick={() => openApplicationResearch(true)}>Describe a custom application</Button>
          {applicationFeatures && !applicationFeatures.research
            ? <p className="workload-product__disabled-reason">Assisted research needs a working Assistant model. Install the Assistant model and it turns on automatically — no setting to flip.</p>
            : session.user.role === "viewer"
              ? <p className="workload-product__disabled-reason">Describing a custom application needs an operator or administrator.</p>
              : null}
        </article>
      </div>

      {visibleApplicationResume && <Notice severity="info"><span><strong>Saved application research</strong> {visibleApplicationResume.requestSummary || "A durable application request is waiting to be resumed."}</span><span className="agent-plan__actions"><Button disabled={agentBusy} variant="quiet" type="button" onClick={() => openApplicationResearch(false)}>Resume application research</Button><Button disabled={agentBusy} variant="danger" type="button" onClick={() => void discardApplicationResearch()}>Discard saved research</Button></span></Notice>}
      {agentNotice && installFlow !== "planner" && <Notice severity={activePlanJob && ["failed", "cancelled", "blocked"].includes(activePlanJob.operation_state || activePlanJob.state) ? "danger" : "info"}><Icon name="activity" />{agentNotice}</Notice>}

      <WorkloadCatalogModal
        busy={agentBusy}
        disabled={!composeReady || session.user.role === "viewer"}
        onClose={() => { clearCatalogResume(); setInstallFlow(null); }}
        onDismiss={() => setInstallFlow(null)}
        onInstall={installTemplate}
        onPreflight={preflightPort}
        open={installFlow === "catalog"}
        resume={catalogResume}
        templates={appTemplates}
        installedTemplateIds={(inventory?.apps ?? []).flatMap((app) => [app.project, app.id].filter((value): value is string => Boolean(value)))}
        onOpenInstalled={() => { setInstallFlow(null); selectSection("manage"); }}
      />

      {installFlow === "model" && (
        <ModalShell labelledBy="model-catalog-title" onClose={() => setInstallFlow(null)}>
        <ModelCatalog
          busy={agentBusy}
          disabled={session.user.role === "viewer"}
          setup={copilotSetup}
          onClose={() => setInstallFlow(null)}
          onChoose={(query) => {
            setInstallFlow("planner");
            void askAgent(
              query,
              "model-install",
            );
          }}
        />
        </ModalShell>
      )}

      {installFlow === "copilot" && showCopilotSetup && copilotSetup && (
        <ModalShell labelledBy="copilot-setup-title" onClose={() => { setShowCopilotSetup(false); setInstallFlow(null); }}>
        <CopilotSetup
          busy={agentBusy}
          data={copilotSetup}
          session={session}
          onClose={() => { setShowCopilotSetup(false); setInstallFlow(null); }}
            onChooseLocal={(query, mode) => {
              setModelRuntimeMode(mode);
              setShowCopilotSetup(false);
              setInstallFlow("planner");
              void askAgent(
                query,
                "model-install",
              );
            }}
            onInstallNpuRelease={(tag) => {
              // A release-sourced NPU model does not go through the planner: the
              // shared hook posts the install and polls it; the status modal
              // below shows the ~3.4 GB download instead of it running silently.
              setShowCopilotSetup(false);
              setInstallFlow(null);
              npuInstall.start(tag);
            }}
          onIntelligenceConnected={() => {
            void apiRequest(
              "/assistant/preferences",
              {
                method: "PATCH",
                body: JSON.stringify({ intelligence_choice: "provider" }),
              },
              session.csrf_token,
            ).then(
              () => setAgentNotice("The connected provider is now the active Vaelor Assistant."),
              (error) => setAgentNotice(error instanceof Error ? error.message : "The Assistant choice could not be saved."),
            );
          }}
        />
        </ModalShell>
      )}

      {npuInstall.job && (
        <ModalShell labelledBy="npu-install-title" onClose={npuInstall.dismiss}>
          <section className="copilot-setup" aria-labelledby="npu-install-title">
            <div className="model-catalog__header">
              <div>
                <span className="page-eyebrow">On-device Assistant</span>
                <h2 id="npu-install-title">Setting up the on-device Assistant</h2>
                <p>Vaelor is preparing the on-device model (downloading it first if it is not already on the machine), verifying it, and starting the Assistant on the neural processor. This keeps running if you close this — reopen the Assistant setup to check on it.</p>
              </div>
              <Button variant="quiet" onClick={npuInstall.dismiss}>Close</Button>
            </div>
            <UpdateJobStatus job={npuInstall.job} onDismiss={npuInstall.dismiss} />
          </section>
        </ModalShell>
      )}

      {installFlow === "custom" && (
        <ModalShell labelledBy="custom-compose-title" onClose={() => setInstallFlow(null)}>
        <section className="custom-compose" aria-labelledby="custom-compose-title">
          <div className="model-catalog__header"><div><span className="page-eyebrow">Guarded Docker composer</span><h2 id="custom-compose-title">Import a custom stack</h2><p>Published images and CPU/memory limits are required. Privileged mode, host namespaces, devices, Docker socket access, unsafe bind mounts, and inline secrets are blocked.</p></div><Button variant="quiet" onClick={() => setInstallFlow(null)}>Close</Button></div>
          <Input
            aria-describedby={customProjectProblem ? "custom-project-error" : undefined}
            aria-invalid={Boolean(customProjectProblem)}
            label="Project name"
            maxLength={48}
            onChange={(event) => { setCustomProject(event.target.value); setCustomComposeError(""); }}
            placeholder="example-stack"
            value={customProject}
          />
          {customProjectProblem && <small className="field-error" id="custom-project-error" role="alert">{customProjectProblem}</small>}
          <Textarea
            className="config-editor"
            label="Compose YAML"
            maxLength={65536}
            onChange={(event) => { setCustomCompose(event.target.value); setCustomComposeError(""); }}
            placeholder={COMPOSE_EXAMPLE}
            spellCheck={false}
            value={customCompose}
          />
          {customComposeError && <div className="application-deployment__error" role="alert"><strong>Action needed</strong><span>{customComposeError}</span></div>}
          <div className="tool-explainer"><Icon name="shield" /><span><strong>Validated before anything starts</strong><small>Docker normalizes the file first; Vaelor then evaluates the actual effective configuration.</small></span></div>
              <Button variant="primary" disabled={agentBusy} disabledReason={!customProject ? "Name the project so Vaelor can manage it." : customProjectProblem ? customProjectProblem : !customCompose.trim() ? "Paste the Compose file you want Vaelor to validate." : undefined} onClick={() => setCustomReview(true)}>{agentBusy ? "Preparing…" : "Review deployment"}</Button>
        </section>
        </ModalShell>
      )}
      {installFlow === "researched" && (
        <ModalShell labelledBy="application-deployment-title" onClose={closeApplicationDeployment}>
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
        </ModalShell>
      )}
      <ActionReviewDialog
        busy={agentBusy}
        evidence={[
          { source: "workloads.capabilities", summary: "Docker and Compose readiness are checked again before deployment." },
          { source: "compose.policy", summary: "Host access, inline secrets, missing resource limits, and port conflicts are blocked." },
        ]}
        job={customReview ? {
          type: "compose.import",
          payload: {
            project: customProject.trim().toLowerCase(),
            content: customCompose,
          },
        } : null}
        onApprove={() => {
          setCustomReview(false);
          void importCompose();
        }}
        onCancel={() => setCustomReview(false)}
        summary={`Import ${customProject.trim().toLowerCase()} as a Vaelor-managed Docker project.`}
        suggestedActions={[
          "Watch validation and image download progress in Recent setup activity.",
          "Open Manage to inspect health, logs, configuration, and removal controls.",
        ]}
      />

      {installFlow === "planner" && <ModalShell labelledBy="deployment-copilot-title" onClose={() => setInstallFlow(null)}>
      <section className="deployment-copilot deployment-copilot--modal" aria-labelledby="deployment-copilot-title">
        <div className="deployment-copilot__rail" aria-hidden="true"><Icon name="cpu" size={24} /><span /><small>ASK<br />CHECK<br />APPROVE</small></div>
        <div className="deployment-copilot__body">
          <div className="panel-heading">
            <div>
              <span className="page-eyebrow">Setup assistant</span>
              <h2 id="deployment-copilot-title">Vaelor deployment assistant</h2>
              <p>Describe what you want in everyday language. Vaelor can plan reviewed catalog apps, research other public apps, ask follow-up questions, and prepare an exact deployment plan for approval.</p>
            </div>
            <div className="panel-heading__actions">
              <StatusPill label={agentStatus?.configured ? agentStatus.model ?? "Model enhanced" : "Code intelligence"} status={agentStatus ? "healthy" : "neutral"} />
              <Button variant="quiet" onClick={() => setInstallFlow(null)} type="button">Close</Button>
            </div>
          </div>

          <div className="deployment-copilot__capabilities" aria-label="Setup assistant capabilities">
            <div>
              <small>Built-in planning</small>
              <strong>One request, matched automatically</strong>
              <p>Vaelor checks reviewed blueprints first, then routes an unknown application into guarded public research without asking you to find another screen.</p>
            </div>
            <div>
              <small>{agentStatus?.configured ? "Model enhanced" : "Code intelligence"}</small>
              <strong>{agentStatus?.configured ? "Better interpretation and follow-up questions" : "Reviewed apps and straightforward research are available"}</strong>
              <p>A model improves ambiguous or complex requests, but it is not a blanket requirement for application setup.</p>
            </div>
          </div>

          <div className="agent-quick-prompts" aria-label="Setup examples">
            {["Install Grafana for dashboards", "Find a small private AI model", "Set up the local assistant"].map((prompt) => (
              <Button disabled={agentBusy || session.user.role === "viewer"} key={prompt} onClick={() => setAgentMessage(prompt)}>{prompt}</Button>
            ))}
          </div>

          <div className="agent-composer">
            <Textarea
              disabled={agentBusy || session.user.role === "viewer"}
              id="agent-request"
              label="What do you want to set up?"
              maxLength={4000}
              onChange={(event) => { setAgentMessage(event.target.value); setAgentNotice(""); }}
              placeholder="Example: I want a private dashboard that shows the temperature of my home."
              rows={3}
              value={agentMessage}
            />
            <Button variant="primary" disabled={agentBusy || !agentMessage.trim() || session.user.role === "viewer"} onClick={() => void askAgent()}>
              {agentBusy ? "Checking…" : "Show me the setup plan"}
            </Button>
          </div>

          {agentNotice && <Notice severity={activePlanJob && ["failed", "cancelled", "blocked"].includes(activePlanJob.operation_state || activePlanJob.state) ? "danger" : "info"}><Icon name="activity" />{agentNotice}</Notice>}

          {agentPlan && (
            <div className="agent-plan" aria-live="polite">
              <div className="agent-plan__summary">
                <span><Icon name="shield" /></span>
                <div><small>{agentPlan.source.replaceAll("-", " ")}</small><strong>{agentPlan.summary}</strong><p>{agentPlan.rationale}</p></div>
              </div>
              <div className="agent-plan__details">
                <div>
                  <span className="control-label">What Vaelor will check</span>
                  <ol>{agentPlan.checklist.map((item) => <li key={item}>{item}</li>)}</ol>
                </div>
                <div>
                  <span className="control-label">Before you continue</span>
                  <p>{agentPlan.proposed_job || agentPlan.application_intent ? "Nothing changes until you continue to the reviewed deployment." : "Give the assistant a little more information so it can prepare a safe plan."}</p>
                  {agentPlan.warnings.map((warning) => <p className="agent-plan__warning" key={warning}>{warning}</p>)}
                  {agentPlan.proposed_job && <details className="advanced-details"><summary>Advanced operation</summary><code>{agentPlan.proposed_job.type}</code></details>}
                </div>
              </div>
              <div className="agent-plan__actions">
                <Button disabled={agentBusy} onClick={() => setAgentPlan(null)}>Go back</Button>
                {agentPlan.application_intent && <Button variant="primary" disabled={agentBusy} onClick={() => { setApplicationStartFresh(true); setResearchedRequest(agentMessage.trim() || agentPlan.application_intent?.application_query || ""); setAgentNotice(""); setAgentPlan(null); setInstallFlow("researched"); }}>Research and deploy</Button>}
                {agentPlan.proposed_job && <Button variant="primary" disabled={agentBusy} onClick={() => setAgentReview(true)}>{agentPlan.proposed_job.type === "model.inspect" ? "Review compatibility check" : "Review exact action"}</Button>}
              </div>
            </div>
          )}

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
        </div>
      </section>
      </ModalShell>}

      <ActionReviewDialog
        busy={agentBusy}
        evidence={[{ source: "setup-assistant", summary: "The exact server-owned operation is shown before authorization." }]}
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
        onCancel={() => { setAgentReview(false); setPendingModelDownload(null); setPendingModelDeploy(null); }}
        summary={agentReview ? agentPlan?.summary ?? "Review the exact setup action." : pendingModelDownload ? `Download the verified ${pendingModelDownload.file} model file.` : pendingModelDeploy ? "Start the downloaded model as a managed local AI service." : ""}
        suggestedActions={["Keep this assistant open for progress and the verified next step."]}
      />

      <WorkloadJobActivity
        applicationResumeJobId={visibleApplicationResume?.jobId}
        jobs={jobs}
        managedModels={managedModels}
        onJobAction={(job, action) => void jobAction(job, action)}
        onQueueModelDeploy={(job) => void queueModelDeploy(job)}
        onQueueModelDownload={(jobId, repo, file, sizeBytes) => void queueModelDownload(jobId, repo, file, sizeBytes)}
        requestedDownloads={downloadingArtefacts}
        onResumeApplicationResearch={() => openApplicationResearch(false)}
      />
      </> : <WorkloadManager inventory={inventory} onRefresh={() => void refresh()} session={session} />}
      </TabSet>
    </div>
  );
}
