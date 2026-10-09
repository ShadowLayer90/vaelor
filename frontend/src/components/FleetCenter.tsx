import { recheckFailureFrom, recheckResultFrom, type RecheckResponse, type RecheckResult } from "./RecheckButton";
import { Button, Notice, TabSet } from "./ui";
import { clusterState, readCapacityLedger } from "../lib/clusterState";
import { useCallback, useEffect, useRef, useState } from "react";
import { createReadSequencer } from "../lib/readSequence";
import { readDismissedOperation, writeDismissedOperation } from "../lib/operationDismissal";
import { GPU_DEPLOYMENT_NAME, MODEL_DEPLOYMENT_NAME, openedServeForm, takenDeploymentNames } from "../lib/serveForm";
import {
  FORCE_REMOVE_ACTION,
  offerFromFailedJob,
  offerFromPlanRefusal,
  type ForcedRemovalOffer,
} from "../lib/forcedRemoval";
import { apiRequest } from "../lib/api";
import { jobCanCancel, jobIsReady, jobIsRetryable, jobIsTerminal, jobStateLabel } from "../lib/jobPresentation";
import { buildClusterJob, type ClusterPlanRequest } from "../lib/clusterJobs";
import { useClusterMode } from "../lib/clusterMode";
import { isGpuClusterDeployment } from "../lib/gpuServingMode";
import type { Session } from "../types";
import { FleetLlmModal } from "./FleetLlmModal";
import { ClusterModeToggle } from "./ClusterModeToggle";
import { ClusterSetupTab } from "./ClusterSetupTab";
import { ClusterFleetTab } from "./ClusterFleetTab";
import { ClusterDeploymentsTab } from "./ClusterDeploymentsTab";
import { ClusterPerformance } from "./ClusterPerformance";
import { ClusterActivity } from "./ClusterActivity";
import { AddMachineFlow } from "./AddMachineFlow";
import { InitializeControllerModal } from "./InitializeControllerModal";
import { DeployAppModal } from "./DeployAppModal";
import { AgentDeployModal, type AgentDeployValues } from "./AgentDeployModal";
import { AgentDeployGallery } from "./AgentDeployGallery";
import { AgentsAndToolsTab } from "./AgentsAndToolsTab";
import { RevealKeyModal } from "./RevealKeyModal";
import {
  type ActionPlan,
  type ClusterCapacityLedger,
  type FleetJob,
  type FleetNode,
  type FleetSummary,
  type InferenceRuntimes,
  type LlmForm,
} from "./fleetTypes";
import { listClusterAgents, type AgentDeployment } from "../lib/clusterAgents";
import { CLUSTER_SECTION_EVENT, DEPLOYMENTS_FILTER_PARAM, clusterSectionFromLocation, type ClusterSection } from "../lib/clusterSections";
import { ActionNotice, ClusterFirstLoad, ClusterStatus, ClusterTitleRow } from "./ClusterHeader";
import { ChangePlanBody, OperationDialog, confirmDescription, planAckReason, type ConfirmRequest } from "./ClusterPlanDialogs";
import { ClusterConfirm, serviceConfirmRequest } from "./ClusterConfirm";
import { ClusterDialog } from "./ClusterDialog";

type ProjectedFleetJob = FleetJob & {
  /** The job's own payload (the `/jobs/<id>` record carries it). */
  payload?: Record<string, unknown>;
  operation_state?: string;
  attention?: boolean;
  retryable?: boolean;
  readiness?: string;
  liveness?: string;
};

// The open tab lives in the query string (lib/clusterSections), so a tab is
// linkable and survives Back/Forward.
/**
 * How often the fleet summary and capacity ledger re-read while Fleet is open
 * (ACC-090), so a machine that powers off or drops out of the cluster changes
 * its status on its own rather than only on Reload.
 */
const FLEET_POLL_MS = 15000;

/**
 * The Cluster page (VD-200, the Cluster boards). `onBack` is still accepted
 * from the shell but no longer drawn: the boards have no "Back to overview";
 * the rail's Home and the breadcrumb lead to the same place.
 */
export function FleetCenter({ session }: { session: Session; onBack?: () => void }) {
  const fleetOperationKey = `vaelor.fleet.operation.${session.user.username}`;
  const [mode, setMode] = useClusterMode(session.user.username);
  const [summary, setSummary] = useState<FleetSummary | null>(null);
  const [capacity, setCapacity] = useState<ClusterCapacityLedger | null>(null);
  // SC6: the reload after an action and the background poll both read these
  // two; only the newest-issued read's answer is applied, so an older poll
  // answer that lands late can never overwrite the post-action state.
  const reads = useRef(createReadSequencer<"summary" | "capacity">()).current;
  // The deployed cluster agents (F6c-1), read alongside the fleet summary and
  // capacity ledger from the same central refresh, so the Agents filter reloads
  // on every deployment change the way the models and apps do.
  const [agents, setAgents] = useState<AgentDeployment[]>([]);
  // ACC-080: an unreadable agent list is said as such, not shown as "none".
  const [agentsError, setAgentsError] = useState("");
  // GG14: a deployed agent's first API key, shown ONCE right after the deploy
  // is queued and dropped when the reveal closes. Never stored anywhere else.
  const [agentKeyReveal, setAgentKeyReveal] = useState<{ name: string; key: string } | null>(null);
  // VD-200 review C1/S1 (LESSONS 8): why the summary and the capacity ledger
  // could not be read, so an unread cluster says "Not read" - never "Checking
  // cluster" for ever, never "0 enrolled", never an action with no reason.
  const [summaryError, setSummaryError] = useState("");
  const [capacityError, setCapacityError] = useState("");
  const cluster = clusterState(summary?.runtime, summary?.enrollment, summary ? undefined : summaryError);
  // The first reload is the page's first read: the tabs mount on it and read
  // once, so only later reloads bump the key (review S3: a double first read).
  const reloads = useRef(0);
  const [loading, setLoading] = useState(true);
  const [notice, setNoticeText] = useState("");
  // VD-189 (review round 2, N3): a refusal is an alert, never an info notice.
  const [noticeRefused, setNoticeRefused] = useState(false);
  const setNotice = useCallback((message: string, refused = false) => { setNoticeText(message); setNoticeRefused(refused); }, []);
  const [dialogError, setDialogError] = useState(""); // VD-189: an open dialog's refusal, shown inside it.
  // Why the background status poll last failed, "" once it succeeds again, so
  // stale machine states are never shown as current without saying so.
  const [pollError, setPollError] = useState("");
  const [showInitialize, setShowInitialize] = useState(false);
  const [showAddMachine, setShowAddMachine] = useState(false);
  const [showLlm, setShowLlm] = useState(false);
  const [showApp, setShowApp] = useState(false);
  const [showAgent, setShowAgent] = useState(false);
  // F6c-2b: "Deploy an agent" opens the Easy gallery first; a card or the
  // gallery's "Advanced setup" then opens the one AgentDeployModal, the card
  // seeding `agentInitial` so the modal opens prefilled. One boolean per modal
  // (as the other Fleet modals are managed) plus the prefill it carries.
  const [showAgentGallery, setShowAgentGallery] = useState(false);
  const [agentInitial, setAgentInitial] = useState<Partial<AgentDeployValues> | undefined>(undefined);
  const [inferenceRuntimes, setInferenceRuntimes] = useState<InferenceRuntimes>({});
  const [plan, setPlan] = useState<ActionPlan | null>(null);
  // The operator's typed acknowledgement for a stateful data-loss plan. Empty
  // until they type the exact token the plan carries; the Approve button stays
  // disabled until it matches, and the value rides into the remove/drain payload.
  const [dataLossAck, setDataLossAck] = useState("");
  // B1: a forced removal is offered only after the backend refused a plain
  // one for this worker (see lib/forcedRemoval), never on a card by default.
  const [refusedRemoval, setRefusedRemoval] = useState<ForcedRemovalOffer | null>(null);
  const [planRequest, setPlanRequest] = useState<{
    action: string;
    nodeId?: string;
    payload?: Record<string, unknown>;
  } | null>(null);
  const [llmForm, setLlmForm] = useState<LlmForm>({
    deploymentMode: "single",
    nodeId: "",
    nodeIds: [] as string[],
    pooledModel: "",
    name: MODEL_DEPLOYMENT_NAME,
    // The GPU deployment name lives with the rest of the form (not inside
    // `GpuServeForm`) so switching deployment mode — which unmounts that
    // sub-form — cannot discard a name the owner typed.
    gpuName: GPU_DEPLOYMENT_NAME,
    repository: "",
    file: "",
    sizeGb: "",
    port: "8100",
  });
  const [busy, setBusy] = useState(false);
  const [activeJob, setActiveJobState] = useState<ProjectedFleetJob | null>(() => {
    try {
      const value = JSON.parse(window.localStorage.getItem(fleetOperationKey) || "null");
      return value && typeof value.id === "string" && typeof value.type === "string" ? value : null;
    } catch {
      return null;
    }
  });
  // The operation is stored in the same step that makes it the active one
  // (LESSONS 8). It was written by an effect after the render that showed it,
  // so a reload between that paint and the effect lost the operation, and a
  // test reading storage once the dialog showed failed under load. A browser
  // that blocks storage only loses resume-after-reload; the page still follows it.
  const setActiveJob = useCallback((job: ProjectedFleetJob | null) => {
    try {
      if (job) window.localStorage.setItem(fleetOperationKey, JSON.stringify(job));
      else {
        window.localStorage.removeItem(fleetOperationKey);
        writeDismissedOperation(fleetOperationKey, null);
      }
    } catch {
      // Nothing to do: the operation is still followed on this page.
    }
    setActiveJobState(job);
  }, [fleetOperationKey]);
  // A dismissed operation's dialog stays closed on later loads (ACC-210).
  const [showActiveJob, setShowActiveJob] = useState(
    () => Boolean(activeJob) && readDismissedOperation(fleetOperationKey) !== activeJob?.id,
  );
  const hideActiveJob = () => {
    setShowActiveJob(false);
    if (activeJob) writeDismissedOperation(fleetOperationKey, activeJob.id);
  };
  const [activeSection, setActiveSection] = useState<ClusterSection>(clusterSectionFromLocation);
  // Bumped whenever the fleet is reloaded (including after an operation's Done),
  // so the model library re-reads its inventory once a pull or remove finishes.
  const [modelsReloadKey, setModelsReloadKey] = useState(0);
  // A destructive direct-submit action (GPU remove, model-cache remove) awaiting
  // an explicit confirm — the review step its CPU siblings get from the plan
  // modal, since these have no server-side plan builder (VD-B3b-4).
  // The confirm modal's wording defaults to a removal (its first and most
  // common use); an action that is not a removal supplies its own eyebrow and
  // button labels so unload/load never read as "Confirm removal"/"Remove".
  const [confirmRemoval, setConfirmRemoval] = useState<(ConfirmRequest & { request: ClusterPlanRequest }) | null>(null);

  const eligibleLlmWorkers = summary?.enrolled_nodes.filter(
    (node) =>
      node.labels?.swarm_node_id
      && node.runtime?.status?.toLowerCase() === "ready"
      && node.runtime?.availability?.toLowerCase() === "active",
  ) ?? [];
  const controllerPlacement = summary?.controller.placement?.eligible
    ? summary.controller.placement
    : null;
  const eligibleLlmTargets = controllerPlacement
    ? [controllerPlacement, ...eligibleLlmWorkers]
    : eligibleLlmWorkers;
  // The capacity ledger row for a fleet participant, matched the way the ledger
  // keys nodes: the Swarm id a worker joined under, its own id (the controller's
  // is `CONTROLLER_PLACEMENT_ID`), or its name.
  const hasDiscoveredGpu = (node: FleetNode) => Boolean(
    (capacity?.nodes ?? []).find((entry) =>
      entry.node_id === node.labels?.swarm_node_id
      || entry.node_id === node.id
      || entry.name === node.name)?.capacity.gpu.present,
  );
  // The GPU serve flow's valid targets: every eligible participant whose
  // discovered capacity says a GPU is present. VD-125: the head controller is
  // an ordinary participant here — it has no SSH credential, so its commands
  // travel over the root hardware bridge instead — and its GPU is half of the
  // only two-node cluster the product ships into, so it is offered under its
  // placement id like any other node and is otherwise not special-cased.
  const gpuNodes = [
    ...(controllerPlacement && hasDiscoveredGpu(controllerPlacement) ? [controllerPlacement] : []),
    ...eligibleLlmWorkers.filter(hasDiscoveredGpu),
  ];
  // "Serve a model" opens on GPU serving when a GPU worker is in the cluster,
  // with a fresh name (ACC-203, ACC-211).
  const openedForm = (current: LlmForm) => openedServeForm(current, {
    gpuWorkers: eligibleLlmWorkers.filter(hasDiscoveredGpu).length,
    taken: takenDeploymentNames(summary),
  });
  const selectableLlmTargets = llmForm.deploymentMode === "pooled"
    ? eligibleLlmWorkers
    : eligibleLlmTargets;
  const toggleLlmWorker = (nodeId: string) => {
    setLlmForm((current) => ({
      ...current,
      nodeIds: current.nodeIds.includes(nodeId)
        ? current.nodeIds.filter((id) => id !== nodeId)
        : [...current.nodeIds, nodeId],
    }));
  };

  const refresh = useCallback(async () => {
    setLoading(true);
    const summaryTicket = reads.issue("summary");
    let summaryRead = false;
    let capacityRead = false;
    try {
      const nextSummary = await apiRequest<FleetSummary>("/cluster");
      if (reads.isLatest("summary", summaryTicket)) setSummary(nextSummary);
      summaryRead = true;
      setSummaryError("");
      setNotice("");
    } catch (error) {
      const message = error instanceof Error && error.message ? error.message : "Fleet status is unavailable.";
      setSummaryError(message);
      setNotice(message, true);
    } finally {
      setLoading(false);
    }
    // The capacity ledger is a read-only placement view; a controller too old
    // to serve it, or a transient failure, simply leaves the section off rather
    // than blocking the page it sits on.
    const capacityTicket = reads.issue("capacity");
    try {
      const nextCapacity = await apiRequest<ClusterCapacityLedger>("/cluster/capacity");
      if (reads.isLatest("capacity", capacityTicket)) setCapacity(readCapacityLedger(nextCapacity));
      capacityRead = true;
      setCapacityError("");
    } catch (error) {
      setCapacityError(error instanceof Error && error.message ? error.message : "The capacity ledger did not answer.");
      // Review nit: a transient failure keeps the last ledger on screen (the
      // poll's own warning says when states stop refreshing) rather than
      // blanking every machine card, the same rule the live memory keeps.
    }
    // A reload that read both answers ends any "could not be refreshed" warning.
    if (summaryRead && capacityRead) setPollError("");
    // A failed agents read never blocks the page, and is never passed off as
    // an empty list: the Agents filter says it could not be read (ACC-080).
    try {
      setAgents(await listClusterAgents());
      setAgentsError("");
    } catch (error) {
      setAgents([]);
      setAgentsError(error instanceof Error && error.message ? error.message : "The request did not complete.");
    }
    if (reloads.current++ > 0) setModelsReloadKey((value) => value + 1);
  }, [reads]);

  const selectSection = useCallback((section: ClusterSection) => {
    const url = new URL(window.location.href);
    if (section === "setup") url.searchParams.delete("cluster"); else url.searchParams.set("cluster", section);
    if (section !== "deployments") url.searchParams.delete(DEPLOYMENTS_FILTER_PARAM); // the filter is that tab's
    window.history.pushState({}, "", url);
    setActiveSection(section);
    // The top bar's breadcrumb follows the tab; a pushState fires no popstate.
    window.dispatchEvent(new CustomEvent(CLUSTER_SECTION_EVENT, { detail: section }));
  }, []);

  // A worker recheck, shared by the tabs: it re-reads the node's inventory and
  // reloads the fleet, surfacing an unreachable worker rather than staying stale.
  // The reload comes FIRST on both paths - `refresh` clears the notice - so the
  // sentence that follows survives: on success, what the Recheck did to the
  // worker's telemetry agent (the backend's `telemetry.message`, "" when it
  // changed nothing); on failure, the error, over a fleet that now shows the
  // worker's stored unreachable state rather than the pre-Recheck one.
  // W5-D2: the outcome is returned to the button that asked, which says it
  // beside itself (RecheckButton) - a Recheck that changed nothing used to
  // say nothing at all.
  const recheckNode = useCallback(async (nodeId: string): Promise<RecheckResult> => {
    try {
      const result = await apiRequest<RecheckResponse>(
        `/cluster/nodes/${nodeId}/refresh`,
        { method: "POST", body: "{}" },
        session.csrf_token,
      );
      await refresh();
      return recheckResultFrom(result);
    } catch (error) {
      await refresh();
      return recheckFailureFrom(error);
    }
  }, [refresh, session.csrf_token]);

  // Install the telemetry agent on a machine still awaiting its join (E2b), a
  // direct synchronous admin action like recheck, then a reload. A worker's
  // telemetry is part of its worker software and comes off only when the
  // machine leaves the cluster (VD-194, owner 2026-10-05), so there is no remove.
  const setWorkerTelemetry = useCallback(
    async (action: "install", nodeId: string) => {
      try {
        await apiRequest(
          `/cluster/nodes/${nodeId}/telemetry`,
          { method: "POST", body: "{}" },
          session.csrf_token,
        );
        await refresh();
      } catch (error) {
        setNotice(
          error instanceof Error ? error.message : "Worker telemetry could not be changed.", true,
        );
      }
    },
    [refresh, session.csrf_token],
  );

  useEffect(() => { void refresh(); }, [refresh]);
  // Review S3, kept apart from the shell's useClusterSummary on purpose: this
  // poll reads the summary and the capacity ledger as one pair under the read
  // sequencer an action's reload also uses, every 15 s while Cluster is open;
  // the shell's read is one `/cluster` a minute off Home, with no ledger.
  // ACC-090: re-read the fleet summary and capacity ledger on a light interval
  // while the page is visible - never two reads in flight, withdrawn on
  // unmount - so machine status pills follow the cluster. It touches only the
  // two reads the machine states come from: the operator's notice, the loading
  // state and the model library's reload are left alone.
  useEffect(() => {
    const poll = new AbortController();
    const { signal } = poll;
    let inFlight = false;
    const read = () => {
      if (inFlight || document.hidden) return;
      inFlight = true;
      const summaryTicket = reads.issue("summary");
      const capacityTicket = reads.issue("capacity");
      void Promise.allSettled([
        apiRequest<FleetSummary>("/cluster", { signal }),
        apiRequest<ClusterCapacityLedger>("/cluster/capacity", { signal }),
      ])
        .then(([nextSummary, nextCapacity]) => {
          if (signal.aborted) return;
          const summaryCurrent = reads.isLatest("summary", summaryTicket);
          const capacityCurrent = reads.isLatest("capacity", capacityTicket);
          // Superseded by a newer read (an action's reload): drop it whole.
          if (!summaryCurrent && !capacityCurrent) return;
          if (summaryCurrent && nextSummary.status === "fulfilled") { setSummary(nextSummary.value); setSummaryError(""); }
          if (capacityCurrent && nextCapacity.status === "fulfilled") { setCapacity(readCapacityLedger(nextCapacity.value)); setCapacityError(""); }
          const failed = [nextSummary, nextCapacity].find((result) => result.status === "rejected");
          setPollError(
            failed && failed.status === "rejected"
              ? failed.reason instanceof Error ? failed.reason.message : "The cluster did not answer."
              : "",
          );
        })
        .finally(() => { inFlight = false; });
    };
    const timer = window.setInterval(read, FLEET_POLL_MS);
    const visibilityChanged = () => {
      if (!document.hidden) read();
    };
    document.addEventListener("visibilitychange", visibilityChanged);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", visibilityChanged);
      poll.abort();
    };
  }, [reads]);
  useEffect(() => {
    const restoreSection = () => setActiveSection(clusterSectionFromLocation());
    window.addEventListener("popstate", restoreSection);
    return () => window.removeEventListener("popstate", restoreSection);
  }, []);
  useEffect(() => {
    if (!activeJob || jobIsTerminal(activeJob)) return;
    const timer = window.setTimeout(() => {
      void apiRequest<ProjectedFleetJob>(`/jobs/${activeJob.id}`)
        .then(setActiveJob)
        .catch((error) => setNotice(error instanceof Error ? error.message : "Cluster progress could not be refreshed.", true));
    }, 2000);
    return () => window.clearTimeout(timer);
  }, [activeJob, setActiveJob]);
  useEffect(() => setDialogError(""), [showLlm, showApp, confirmRemoval, plan, showActiveJob]);
  useEffect(() => {
    void apiRequest<InferenceRuntimes>("/cluster/inference/runtimes")
      .then((result) => {
        setInferenceRuntimes(result);
        const firstModel = result.pooled?.models[0]?.id;
        if (firstModel) {
          setLlmForm((current) => (
            current.pooledModel ? current : { ...current, pooledModel: firstModel }
          ));
        }
      })
      .catch(() => setInferenceRuntimes({}));
  }, []);

  const reviewPlan = async (
    action: string,
    nodeId?: string,
    actionPayload: Record<string, unknown> = {},
  ) => {
    try {
      // A fresh plan starts with no acknowledgement typed; the field only shows
      // when the returned plan carries a data-loss token.
      setDataLossAck("");
      setPlan(await apiRequest<ActionPlan>("/cluster/plan", {
        method: "POST",
        body: JSON.stringify({ action, node_id: nodeId, ...actionPayload }),
      }, session.csrf_token));
      setPlanRequest({ action, nodeId, payload: actionPayload });
      if (action === "deploy-llm") setShowLlm(false);
      if (action === "deploy-app") setShowApp(false);
    } catch (error) {
      const offer = offerFromPlanRefusal(action, nodeId, error);
      if (offer) {
        setRefusedRemoval(offer);
        return;
      }
      // Serve and Deploy app stay open until their plan arrives, so their refusal belongs in them.
      const refusal = error instanceof Error ? error.message : "The action plan is unavailable.";
      if (action === "deploy-llm" || action === "deploy-app") setDialogError(refusal);
      else setNotice(refusal, true);
    }
  };

  // The forced-removal offer on screen: a refused plan, else a removal job
  // that failed pointing at a forced removal.
  const failedRemoval = activeJob && jobIsTerminal(activeJob) ? offerFromFailedJob(activeJob) : null;
  const removalOffer = refusedRemoval ?? failedRemoval;
  const reviewForcedRemoval = (offer: ForcedRemovalOffer) => {
    setRefusedRemoval(null);
    if (failedRemoval) {
      setActiveJob(null);
      setShowActiveJob(false);
    }
    void reviewPlan(FORCE_REMOVE_ACTION, offer.nodeId);
  };

  // Build a cluster job from a request and queue it directly at `POST /jobs`,
  // then track it in the same activeJob poll every cluster change uses. This is
  // the shared core of the plan-reviewed flow (`approvePlan`) AND the flows with
  // no server-side plan builder — GPU serve, GPU remove, and the model
  // pull/remove — which are gated client-side (the fit decision, an explicit
  // confirm) and submit straight here so their progress renders identically.
  const queueClusterJob = async (request: ClusterPlanRequest, report: (message: string, refused?: boolean) => void = setNotice): Promise<boolean> => {
    const job = buildClusterJob(request);
    if (!job) return false;
    setBusy(true);
    try {
      const result = await apiRequest<Partial<ProjectedFleetJob> & { key_reveal?: { key?: string } }>("/jobs", {
        method: "POST",
        body: JSON.stringify(job),
      }, session.csrf_token);
      if (request.action === "deploy-agent" && typeof result.key_reveal?.key === "string" && result.key_reveal.key) {
        setAgentKeyReveal({ name: String(request.payload?.name ?? "this agent"), key: result.key_reveal.key });
      }
      setActiveJob({
        id: String(result.id),
        type: result.type ?? job.type,
        payload: result.payload ?? job.payload,
        state: result.state ?? "queued",
        progress: result.progress ?? 0,
        message: result.message ?? "Waiting for the cluster worker",
      });
      setShowActiveJob(true);
      setNotice("Cluster change approved. Progress and verification remain in this workflow.");
      return true;
    } catch (error) {
      report(error instanceof Error ? error.message : "The cluster change could not be queued.", true);
      return false;
    } finally {
      setBusy(false);
    }
  };

  const approvePlan = async () => {
    if (!planRequest || !summary) return;
    if (planRequest.action === "evict-mismatched") {
      // Not a queued job: the controller acts on its own fleet record, and the
      // audit entry is written from the outcome rather than from the request,
      // so the trail says what was destroyed instead of what was asked for.
      setBusy(true);
      try {
        const result = await apiRequest<{
          evicted: Array<{ name: string }>;
          failed: Array<{ name: string; error: string }>;
        }>("/cluster/architecture/evictions", {
          method: "POST",
          body: JSON.stringify({
            confirm: "evict-mismatched-nodes",
            ...(planRequest.nodeId ? { node_id: planRequest.nodeId } : {}),
          }),
        }, session.csrf_token);
        setPlan(null);
        setPlanRequest(null);
        // Reloaded first: `refresh` clears the notice, and the report of what
        // was destroyed is the part of this that must not be swallowed.
        await refresh();
        setNotice(
          result.failed.length
            ? `Removed ${result.evicted.length} mismatched worker(s); ${
              result.failed.map((item) => `${item.name}: ${item.error}`).join("; ")
            }`
            : `Drained and removed ${result.evicted.length} mismatched worker(s): ${
              result.evicted.map((item) => item.name).join(", ")
            }.`,
          // Some machines were not removed: that half is a refusal (N3).
          result.failed.length > 0,
        );
      } catch (error) {
        setDialogError(error instanceof Error ? error.message : "The mismatched worker could not be removed.");
      } finally {
        setBusy(false);
      }
      return;
    }
    // A stateful data-loss plan carries the required token; fold the operator's
    // typed acknowledgement into the payload so the backend gate is satisfied.
    // A stateless plan has no token and rides through unchanged.
    const request = plan?.data_loss_ack
      ? { ...planRequest, payload: { ...planRequest.payload, data_loss_ack: dataLossAck } }
      : planRequest;
    if (await queueClusterJob(request, setDialogError)) {
      setPlan(null);
      setPlanRequest(null);
    }
  };

  const joinedWorkers = summary?.enrolled_nodes.filter(
    (node) => node.labels?.swarm_node_id && node.runtime?.availability?.toLowerCase() !== "drain",
  ) ?? [];

  // The three primary actions offered on Fleet and Deployments share one set of
  // gates, computed here from the same cluster state, so a control cannot outrun
  // the sentence beside it whichever tab it is drawn on (VD-009a).
  const llmDisabled = !eligibleLlmTargets.length;
  // C1: a disabled action always carries a reason - an unread cluster too.
  const unreadReason = cluster.unavailable ?? "The cluster state has not been read yet.";
  const llmDisabledReason = cluster.operatingAsController
    ? "No node currently has enough free memory and storage for a model server."
    : unreadReason;
  const appDisabled = !joinedWorkers.length;
  const appDisabledReason = cluster.operatingAsController
    ? "Enrol at least one worker before deploying a cluster app."
    : unreadReason;
  const addMachineActionable = cluster.enrollment.actionable;
  const addMachineDisabledReason = addMachineActionable ? undefined : cluster.enrollment.reason;
  const deployAgent = () => { setAgentInitial(undefined); setShowAgentGallery(true); };

  const administrator = session.user.role === "administrator";
  // The first read has not answered: the frame and one card that says so,
  // with no action offered against a cluster nobody has read yet.
  const firstLoad = loading && !summary;

  return (
    <section className="fleet-page cl-page">
      <ClusterStatus busy={busy} cluster={cluster} onReload={() => void refresh()} />
      {/* Add machine, Deploy app and Serve a model sit in the title row on every
          tab, each with its gate's reason under it (the ClusterHeader board). */}
      <ClusterTitleRow
        addMachineDisabledReason={addMachineDisabledReason}
        deployAppDisabledReason={appDisabled ? appDisabledReason : undefined}
        onAddMachine={() => setShowAddMachine(true)}
        onDeployApp={() => setShowApp(true)}
        onServeModel={() => { setLlmForm(openedForm); setShowLlm(true); }}
        serveModelDisabledReason={llmDisabled ? llmDisabledReason : undefined}
        showActions={!firstLoad}
        showAddMachine={administrator}
      />

      {/* Newest concern first: a refusal, a forced removal on offer, stale
          machine states, then the operation under way and what was approved. */}
      <div className="cl-notices">
        {notice && noticeRefused && <ActionNotice severity="danger">{notice}</ActionNotice>}
        {removalOffer && (refusedRemoval || !showActiveJob) && (
          <Notice severity="warning">
            <span className="cl-notice-row">
              <span>{removalOffer.reason}</span>
              <span className="cl-notice-row__actions">
                <Button className="cl-danger-outline" onClick={() => reviewForcedRemoval(removalOffer)}>Review forced removal</Button>
                {refusedRemoval && <Button variant="quiet" onClick={() => setRefusedRemoval(null)}>Dismiss</Button>}
              </span>
            </span>
          </Notice>
        )}
        {pollError && (
          <ActionNotice severity="warning">
            Machine status could not be refreshed ({pollError}). {capacity ? "The states below are from the last successful reading." : "No machine status has been read yet."}
          </ActionNotice>
        )}
        {activeJob && !showActiveJob && (
          <Notice severity="info">
            <span className="cl-notice-row">
              <span>Cluster operation {jobStateLabel(activeJob)}: {activeJob.message}</span>
              <span className="cl-notice-row__actions">
                <Button variant="quiet" onClick={() => setShowActiveJob(true)}>View current operation</Button>
              </span>
            </span>
          </Notice>
        )}
        {notice && !noticeRefused && <ActionNotice severity="info">{notice}</ActionNotice>}
      </div>

      <div className="cl-tabs">
        <ClusterModeToggle mode={mode} onChange={setMode} />
        <TabSet
          // Written out, not mapped from CLUSTER_SECTIONS: tools/ui_inventory.py
          // reads the tab labels from this literal (LESSONS 19). A test holds the
          // two lists equal.
          items={[
            { id: "setup", label: "Setup" },
            { id: "fleet", label: "Fleet" },
            { id: "deployments", label: "Deployments" },
            { id: "performance", label: "Performance" },
            { id: "activity", label: "Activity" },
            { id: "agents", label: "Agents & tools" },
          ]}
          label="Cluster views"
          panelClassName="fleet-tab-panel"
          onSelect={(id) => selectSection(id as ClusterSection)}
          selectedId={activeSection}
        >
        {firstLoad && <ClusterFirstLoad />}
        {!firstLoad && activeSection === "setup" && (
          <ClusterSetupTab
            session={session}
            summary={summary}
            cluster={cluster}
            mode={mode}
            onReviewPlan={(action, nodeId, payload) => void reviewPlan(action, nodeId, payload)}
            onInitialize={() => setShowInitialize(true)}
            onRecheck={recheckNode}
            onRefresh={() => void refresh()}
            setNotice={setNotice}
            addMachineActionable={addMachineActionable}
            addMachineDisabledReason={addMachineDisabledReason}
          />
        )}
        {!firstLoad && activeSection === "fleet" && (
          <ClusterFleetTab
            ledger={capacity ?? readCapacityLedger(undefined)}
            ledgerRead={capacity !== null}
            ledgerError={capacityError}
            summary={summary}
            summaryError={summaryError}
            agentsError={agentsError}
            session={session}
            mode={mode}
            reloadKey={modelsReloadKey}
            onNodeAction={(action, nodeId) => void reviewPlan(action, nodeId)}
            onRecheck={recheckNode}
            onWorkerTelemetry={(action, nodeId) => void setWorkerTelemetry(action, nodeId)}
            agents={agents}
          />
        )}
        {!firstLoad && activeSection === "deployments" && (
          <ClusterDeploymentsTab
            services={summary?.runtime.services ?? []}
            pooled={summary?.pooled_deployments ?? []}
            appGroups={summary?.runtime.app_groups ?? []}
            agents={agents}
            agentsError={agentsError}
            ledger={capacity ?? readCapacityLedger(undefined)}
            session={session}
            onServiceReview={(action, payload) => {
              // GPU remove, unload and load and agent remove have no
              // server-side plan builder: each asks for an explicit confirm
              // here, then queues. Every other removal keeps its reviewed
              // `/cluster/plan` step.
              const confirm = serviceConfirmRequest(action, payload, summary?.pooled_deployments ?? []);
              if (confirm) {
                setConfirmRemoval(confirm);
                return Promise.resolve();
              }
              return reviewPlan(action, undefined, payload);
            }}
            onNotice={setNotice}
            onDeployAgent={deployAgent}
            pullTargets={eligibleLlmTargets}
            modelsReloadKey={modelsReloadKey}
            onModelPull={(payload) => void queueClusterJob({ action: "model-pull", payload })}
            onModelRemove={(payload) => setConfirmRemoval({
              title: "Remove cached model",
              body: `This deletes the cached weights for ${String(payload.model_source ?? "this model")} from the selected worker(s). A later deploy would re-pull them.`,
              request: { action: "model-remove", payload },
            })}
          />
        )}
        {!firstLoad && activeSection === "performance" && <ClusterPerformance mode={mode} session={session} clusterServing={(summary?.pooled_deployments ?? []).some(isGpuClusterDeployment)} />}
        {!firstLoad && activeSection === "activity" && <ClusterActivity session={session} reloadKey={modelsReloadKey} />}
        {!firstLoad && activeSection === "agents" && <AgentsAndToolsTab session={session} />}
        </TabSet>
      </div>

      {showInitialize && (
        <InitializeControllerModal
          candidateAddress={summary?.controller.candidate_address}
          busy={busy}
          onClose={() => setShowInitialize(false)}
          onReview={(address) => {
            setShowInitialize(false);
            void reviewPlan("initialize", undefined, { advertise_address: address });
          }}
        />
      )}

      {showAddMachine && (
        <AddMachineFlow
          session={session}
          mode={mode}
          actionable={addMachineActionable}
          disabledReason={addMachineDisabledReason}
          onEnrolled={() => refresh()}
          onReviewJoin={(nodeId) => void reviewPlan("join-node", nodeId)}
          addressHint={summary?.enrollment?.address_hint}
          setNotice={setNotice}
          onClose={() => !busy && setShowAddMachine(false)}
        />
      )}

      {showLlm && (
        <FleetLlmModal
          busy={busy}
          session={session}
          eligibleWorkers={eligibleLlmWorkers}
          gpuNodes={gpuNodes}
          form={llmForm}
          inferenceRuntimes={inferenceRuntimes} error={dialogError}
          onClose={() => !busy && setShowLlm(false)}
          onReview={(nodeId, payload) => void reviewPlan("deploy-llm", nodeId, payload)}
          onServeGpu={(payload) => {
            setShowLlm(false);
            void queueClusterJob({ action: "deploy-gpu", payload });
          }}
          onToggleWorker={toggleLlmWorker}
          // VD-127 D6: a model that fits one machine is served as the AI Chat
          // model, which is installed on Apps and AI — so the refusal has
          // somewhere to send the owner rather than only telling them no.
          onServeAsAiChatModel={() => {
            setShowLlm(false);
            window.dispatchEvent(new CustomEvent("pironman:navigate", { detail: "workloads" }));
          }}
          selectableTargets={selectableLlmTargets}
          setForm={setLlmForm}
        />
      )}

      {showApp && (
        <DeployAppModal
          joinedWorkers={joinedWorkers}
          busy={busy} error={dialogError}
          onClose={() => setShowApp(false)}
          onReview={(nodeId, payload) => void reviewPlan("deploy-app", nodeId, payload)}
        />
      )}

      {showAgentGallery && (
        <AgentDeployGallery
          onClose={() => setShowAgentGallery(false)}
          onAdvanced={() => {
            setShowAgentGallery(false);
            setAgentInitial(undefined);
            setShowAgent(true);
          }}
          onDeploy={(initial) => {
            setShowAgentGallery(false);
            setAgentInitial(initial);
            setShowAgent(true);
          }}
        />
      )}

      {showAgent && (
        <AgentDeployModal
          pooledDeployments={summary?.pooled_deployments ?? []}
          busy={busy}
          csrfToken={session.csrf_token}
          initial={agentInitial}
          onClose={() => !busy && setShowAgent(false)}
          onDeploy={(payload) => {
            setShowAgent(false);
            void queueClusterJob({ action: "deploy-agent", payload });
          }}
        />
      )}

      {confirmRemoval && (
        <ClusterConfirm
          busy={busy}
          busyLabel={confirmRemoval.busyLabel ?? "Removing…"}
          confirmLabel={confirmRemoval.confirmLabel ?? "Remove"}
          confirmVariant={confirmRemoval.constructive ? "primary" : "danger"}
          description={confirmDescription(confirmRemoval.body)}
          error={dialogError}
          eyebrow={confirmRemoval.eyebrow ?? "Confirm removal"}
          onCancel={() => !busy && setConfirmRemoval(null)}
          onConfirm={() => void queueClusterJob(confirmRemoval.request, setDialogError).then((queued) => {
            if (queued) setConfirmRemoval(null);
          })}
          open
          title={confirmRemoval.title}
        />
      )}

      {plan && (
        <ClusterDialog
          error={dialogError}
          eyebrow="Change plan"
          footer={<>
            <Button onClick={() => { setPlan(null); setPlanRequest(null); }} variant="quiet">Close</Button>
            <Button
              disabledReason={planAckReason(plan, dataLossAck)}
              disabled={busy}
              onClick={() => void approvePlan()}
              variant="primary"
            >
              {busy ? "Queuing…" : "Approve and queue"}
            </Button>
          </>}
          onClose={() => { setPlan(null); setPlanRequest(null); }}
          title={plan.title}
          titleId="cluster-plan-title"
        >
          <ChangePlanBody ack={dataLossAck} onAck={setDataLossAck} plan={plan} />
        </ClusterDialog>
      )}

      {activeJob && showActiveJob && (
        <OperationDialog
          busy={busy}
          canCancel={jobCanCancel(activeJob)}
          canFinish={jobIsReady(activeJob) || (jobIsTerminal(activeJob) && !jobIsRetryable(activeJob))}
          canRetry={jobIsRetryable(activeJob)}
          error={dialogError}
          job={activeJob}
          onBackground={hideActiveJob}
          onCancelJob={() => {
            setBusy(true);
            void apiRequest<ProjectedFleetJob>(`/jobs/${activeJob.id}/cancel`, { method: "POST", body: "{}" }, session.csrf_token)
              .then(setActiveJob)
              .catch((error) => setDialogError(error instanceof Error ? error.message : "Cancellation could not be requested."))
              .finally(() => setBusy(false));
          }}
          onDone={() => { setActiveJob(null); setShowActiveJob(false); void refresh(); }}
          onForcedRemoval={failedRemoval ? () => reviewForcedRemoval(failedRemoval) : undefined}
          onRetry={() => {
            setBusy(true);
            void apiRequest<ProjectedFleetJob>(`/jobs/${activeJob.id}/retry`, { method: "POST", body: "{}" }, session.csrf_token)
              .then(setActiveJob)
              .catch((error) => setDialogError(error instanceof Error ? error.message : "Retry could not be queued."))
              .finally(() => setBusy(false));
          }}
        />
      )}

      {agentKeyReveal && (
        <RevealKeyModal
          description={`This is the key for ${agentKeyReveal.name}. It is shown once - copy it now. It works once the deploy finishes; manage its keys under Cluster › Deployments › Models.`}
          keyValue={agentKeyReveal.key}
          onClose={() => setAgentKeyReveal(null)}
          title="Your agent's API key"
        />
      )}
    </section>
  );
}
