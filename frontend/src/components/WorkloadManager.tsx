import { useEffect, useRef, useState } from "react";
import "../styles/apps-manage.css";
import { apiRequest } from "../lib/api";
import { usePagePlace } from "../lib/topbarSlot";
import { useOperationOwner } from "../hooks/useOperationOwner";
import type { Session } from "../types";
import { ActionReviewDialog } from "./ActionReviewDialog";
import { AppManagerPanel } from "./AppManagerPanel";
import { ManageModelsCard, ManageServicesCard, type ListState } from "./ManageLists";
import {
  fileManagerAvailable,
  workloadTools,
  type LifecycleAction,
  type LifecycleJob,
  type ManagedApp,
  type ManagedInventory,
  type Tool,
} from "./manageModel";
import { ModelSwitchPanel, modelTierLabel, type RuntimeMode } from "./ModelSwitchPanel";
import { OperationOwner } from "./OperationOwner";
import { EmptyState, OperationFeedback, type OperationState } from "./ui";
import { RemovalCheckingDialog, WorkloadRemovalDialog, type RemovalResource, type WorkloadRemovalOptions, type WorkloadRemovalPlan } from "./WorkloadRemovalDialog";

/*
 * The Manage tab (VD-200, the Manage* boards): "Your services" and
 * "Downloaded models" on the left, and on the right the app manager for the
 * chosen app - or, while one is open, the Use model panel, which takes the app
 * manager's place beside the model list. This module owns the selection, the
 * deep link (`?app=` / `?appTool=`), the lifecycle job, the model switch and
 * the removal; the panels draw them.
 */

export { inventoryIsSettling, managedAppStatus, settlingHealth } from "./manageModel";
export type { ManagedInventory } from "./manageModel";

/** Apps and AI's two tabs, in the strip and in the top bar breadcrumb ("Apps and AI / Manage"). */
export const APPS_SECTION_LABELS = { install: "Install", manage: "Manage" } as const;

const lifecycleStorageKey = "vaelor.workloads.live_job";

function restoredLifecycleJob(): LifecycleJob | null {
  try {
    const value = JSON.parse(window.localStorage.getItem(lifecycleStorageKey) || "null");
    return value && typeof value.id === "string" && typeof value.type === "string" ? value : null;
  } catch {
    return null;
  }
}

/** R-F6: a deep link or Back may name a tool the app does not offer; it opens on Overview then. */
function toolFor(app: ManagedApp, requested: string | null): Tool {
  if (!requested || !workloadTools.has(requested as Tool)) return "overview";
  const tool = requested as Tool;
  if (tool === "filemanager" && !fileManagerAvailable(app)) return "overview";
  if (tool === "logs" && !app.capabilities.logs) return "overview";
  if (tool === "configuration" && !app.capabilities.configuration) return "overview";
  if (tool === "console" && !app.capabilities.console) return "overview";
  if (tool === "remote" && !app.capabilities.remote_desktop) return "overview";
  if (tool === "restore" && !(app.managed && app.project)) return "overview";
  return tool;
}

const errorText = (error: unknown, fallback: string) => error instanceof Error && error.message ? error.message : fallback;

export function WorkloadManager({
  inventory,
  inventoryState,
  session,
  onRefresh,
}: {
  inventory: ManagedInventory | null;
  /**
   * How the page's last `/managed` read went. "error" with no inventory draws
   * the lists' "could not be read" banners, never "No apps installed yet".
   * Absent, a null inventory is read as not read (Task #122).
   */
  inventoryState?: "loading" | "ok" | "error";
  session: Session;
  onRefresh: () => void;
}) {
  const [selected, setSelected] = useState<ManagedApp | null>(null);
  const [tool, setTool] = useState<Tool>("overview");
  const [notice, setNotice] = useState("");
  const [noticeState, setNoticeState] = useState<OperationState>("idle");
  const setFeedback = (message: string, state: OperationState = "success") => {
    setNotice(message);
    setNoticeState(message ? state : "idle");
  };
  const clearFeedback = () => setFeedback("", "idle");
  const [busy, setBusy] = useState(false);
  // R-F7: the panel reads the LIVE row by id, so a refusal that arrives after
  // it opened (Mode B taking AI Chat) is shown and disables the switch.
  const [selectedModelId, setSelectedModelId] = useState("");
  const selectedModel = inventory?.models.find((model) => model.id === selectedModelId) ?? null;
  // "Apps and AI / Manage / <the open app>", or the Use model panel's own title (the ManageModelSwitch board).
  usePagePlace([APPS_SECTION_LABELS.manage, ...(selectedModel ? [`Use ${selectedModel.name}`] : selected ? [selected.name] : [])]);
  const [removalPlan, setRemovalPlan] = useState<WorkloadRemovalPlan | null>(null);
  const [removalPending, setRemovalPending] = useState<RemovalResource | null>(null);
  const removalRequest = useRef<string | null>(null);
  const [removalError, setRemovalError] = useState("");
  const [recoveryRefresh, setRecoveryRefresh] = useState(0);
  const [pendingLifecycle, setPendingLifecycle] = useState<LifecycleAction | null>(null);
  // W4d-D12: the app a configuration was saved for and the running containers
  // have not taken yet. A save writes the file; start, restart and update apply it.
  const [configNotApplied, setConfigNotApplied] = useState<{ appId: string; when: string } | null>(null);
  const [lifecycleJob, setLifecycleJob] = useState<LifecycleJob | null>(restoredLifecycleJob);
  const [lifecycleRestoreDismissed, setLifecycleRestoreDismissed] = useState(false);
  const lifecycleController = useOperationOwner({
    csrfToken: session.csrf_token,
    enabled: Boolean(lifecycleJob),
    operationKey: lifecycleJob ? `jobs/${lifecycleJob.id}` : null,
    onResourceRefresh: async (operation) => {
      if (operation.type === "compose.backup") setRecoveryRefresh((value) => value + 1);
      onRefresh();
    },
    resumeStorageKey: "vaelor.workloads.lifecycle-operation",
  });

  useEffect(() => {
    if (lifecycleJob) window.localStorage.setItem(lifecycleStorageKey, JSON.stringify(lifecycleJob));
    else window.localStorage.removeItem(lifecycleStorageKey);
  }, [lifecycleJob]);

  useEffect(() => {
    if (!lifecycleController.isTerminal) return;
    setNotice("");
    setNoticeState("idle");
  }, [lifecycleController.isTerminal]);

  // A remembered operation reopens the app it belongs to, until the owner closes it.
  useEffect(() => {
    if (selected || lifecycleRestoreDismissed || !lifecycleJob?.resource_id || !inventory) return;
    const app = inventory.apps.find((candidate) => candidate.id === lifecycleJob.resource_id || candidate.app_instance_id === lifecycleJob.resource_id);
    if (app) {
      setSelected(app);
      setTool("overview");
    }
  }, [inventory, lifecycleJob?.resource_id, lifecycleRestoreDismissed, selected]);

  // The open app follows the live inventory; an app that went away closes.
  useEffect(() => {
    if (!selected) return;
    const current = inventory?.apps.find((app) => app.id === selected.id);
    if (current) setSelected(current);
    else if (inventory) setSelected(null);
  }, [inventory, selected?.id]);

  useEffect(() => {
    const restoreRoute = () => {
      const query = new URLSearchParams(window.location.search);
      const appId = query.get("app");
      if (!appId) {
        if (!lifecycleJob || lifecycleRestoreDismissed) setSelected(null);
        return;
      }
      const app = inventory?.apps.find((candidate) => candidate.id === appId || candidate.app_instance_id === appId);
      if (!app) return;
      setLifecycleRestoreDismissed(false);
      setSelected(app);
      setTool(toolFor(app, query.get("appTool")));
    };
    restoreRoute();
    window.addEventListener("popstate", restoreRoute);
    return () => window.removeEventListener("popstate", restoreRoute);
  }, [inventory, lifecycleJob, lifecycleRestoreDismissed]);

  const updateManagerRoute = (app: ManagedApp | null, nextTool: Tool = "overview") => {
    const url = new URL(window.location.href);
    url.searchParams.set("workloads", "manage");
    if (app) {
      url.searchParams.set("app", app.app_instance_id || app.id);
      url.searchParams.set("appTool", nextTool);
    } else {
      url.searchParams.delete("app");
      url.searchParams.delete("appTool");
    }
    window.history.pushState({}, "", url);
  };

  const closeManager = () => {
    setLifecycleRestoreDismissed(true);
    setSelected(null);
    clearFeedback();
    updateManagerRoute(null);
  };

  const openTool = (app: ManagedApp, nextTool: Tool) => {
    setLifecycleRestoreDismissed(false);
    if (app.id !== selected?.id) clearFeedback();
    setSelected(app);
    setTool(nextTool);
    // The app manager takes the right column back from the Use model panel.
    setSelectedModelId("");
    updateManagerRoute(app, nextTool);
  };

  const lifecycle = async (action: LifecycleAction) => {
    if (!selected?.project) return;
    setBusy(true);
    clearFeedback();
    try {
      const job = await apiRequest<LifecycleJob>(
        "/jobs",
        { method: "POST", body: JSON.stringify({ type: `compose.${action}`, payload: { project: selected.project } }) },
        session.csrf_token,
      );
      setLifecycleJob({ ...job, resource_id: selected.id });
      if (action !== "stop" && action !== "backup") setConfigNotApplied(null);
      setFeedback(`${action[0].toUpperCase()}${action.slice(1)} approved. Progress remains here while Vaelor verifies the application.`, "pending");
    } catch (error) {
      setFeedback(errorText(error, "The app action could not be queued."), "error");
    } finally {
      setBusy(false);
    }
  };

  const activateModel = async (runtimeMode: RuntimeMode | null) => {
    if (!selectedModel) return;
    setBusy(true);
    clearFeedback();
    try {
      await apiRequest("/jobs", {
        method: "POST",
        body: JSON.stringify({
          type: "model.deploy",
          // Send the catalog surface so a GPU AI-Chat model is routed to its own
          // tier even if the executor cannot resolve the path back to the
          // catalog — otherwise the surface silently defaults to "assistant" and
          // a chat model would overwrite the NPU Assistant. Omitted when unknown,
          // preserving the historical default.
          payload: {
            path: selectedModel.path,
            port: 0,
            ...(runtimeMode ? { mode: runtimeMode } : {}),
            ...(selectedModel.surface ? { surface: selectedModel.surface } : {}),
          },
        }),
      }, session.csrf_token);
      setFeedback(`${selectedModel.name} will replace the current local ${modelTierLabel(selectedModel)} model after its health check passes.`, "pending");
      setSelectedModelId("");
      window.setTimeout(onRefresh, 2500);
    } catch (error) {
      setFeedback(errorText(error, "The model switch could not be queued."), "error");
    } finally {
      setBusy(false);
    }
  };

  const reviewRemoval = async (resource: RemovalResource) => {
    // One request per click. Without an in-flight guard a repeated or
    // re-entrant call issued a second identical removal-plan fetch, which on a
    // Pi-class device doubled an already slow interaction.
    const key = `${resource.kind}:${resource.id}`;
    if (removalRequest.current === key) return;
    removalRequest.current = key;
    setBusy(true);
    setRemovalError("");
    clearFeedback();
    // Show the dialog immediately rather than leaving the screen unchanged for
    // seconds while the plan loads; the user otherwise reads it as a dead
    // button and clicks again.
    setRemovalPending(resource);
    try {
      const plan = await apiRequest<WorkloadRemovalPlan>(`/managed/removal-plan?kind=${encodeURIComponent(resource.kind)}&id=${encodeURIComponent(resource.id)}`);
      // A cancelled request must not reopen the dialog or re-disable the page
      // when a slow response finally lands.
      if (removalRequest.current !== key) return;
      setRemovalPending(null);
      // Clear the request state before exposing the interactive plan. Without
      // this ordering, a fast response can render the dialog for one frame while
      // its approval controls still inherit the request's busy state.
      setBusy(false);
      setRemovalPlan(plan);
    } catch (error) {
      if (removalRequest.current !== key) return;
      setRemovalPending(null);
      setFeedback(errorText(error, "The dependency report could not be prepared."), "error");
    } finally {
      // Only the request that still owns the guard may clear it; a late
      // response from a cancelled request must not release a newer one.
      if (removalRequest.current === key) {
        removalRequest.current = null;
        setBusy(false);
      }
    }
  };

  /**
   * Cancelling the "checking dependencies" dialog has to hand the page back
   * immediately (VLR-071). The plan request can take many seconds on a Pi and
   * cannot be aborted, and `busy` gates every control in this manager.
   * Releasing the guard here also makes the late response a no-op.
   */
  const cancelRemovalReview = () => {
    removalRequest.current = null;
    setRemovalPending(null);
    setBusy(false);
  };

  const queueRemoval = async (options: WorkloadRemovalOptions) => {
    if (!removalPlan) return;
    setBusy(true);
    setRemovalError("");
    const identity = removalPlan.resource.display_identity ?? removalPlan.display_identity ?? removalPlan.resource.name;
    try {
      const result = await apiRequest<LifecycleJob>("/jobs", {
        method: "POST",
        body: JSON.stringify({
          type: "managed.remove",
          payload: {
            kind: removalPlan.resource.kind,
            id: removalPlan.resource.id,
            display_identity: identity,
            plan_digest: removalPlan.plan_digest,
            confirmation: options.confirmation,
            dependency_strategy: options.dependency_strategy,
            retain_data: options.retain_data,
            create_backup: options.create_backup,
          },
        }),
      }, session.csrf_token);
      const backup = options.create_backup ? " A verified recovery point will be recorded in the job result." : " No new recovery point was requested.";
      const data = options.retain_data ? " Persistent data will be retained." : " Approved persistent data will be deleted.";
      setLifecycleJob({ ...result, resource_id: removalPlan.resource.id, display_identity: identity });
      setFeedback(`${identity} removal approved.${backup}${data} Progress and verification remain in this manager.`, "pending");
      setRemovalPlan(null);
    } catch (error) {
      setRemovalError(errorText(error, "The reviewed removal could not be queued. Refresh the dependency report."));
    } finally {
      setBusy(false);
    }
  };

  const apps = inventory?.apps ?? [];
  const models = inventory?.models ?? [];
  /*
   * Task #122. `inventory` is null when `/managed` has not answered - on the
   * appliance it timed out while the machine was loading a model. An empty
   * list and an unread list are different facts: the first is about the
   * appliance, the second about this browser's last request.
   */
  const listState: ListState = inventory ? "ok" : inventoryState === "loading" ? "loading" : "error";
  // VD-189: a refusal is always shown; a progress line only while its operation is live.
  const showNotice = Boolean(notice) && (!lifecycleController.isTerminal || noticeState === "error" || noticeState === "warning");
  const feedback = showNotice ? <OperationFeedback className="manage-feedback" message={notice} state={noticeState} /> : null;
  const isAdministrator = session.user.role === "administrator";

  const operation = selected && lifecycleJob?.resource_id === selected.id ? (lifecycleController.operation ? (
    <OperationOwner
      className="manage-operation"
      controller={{
        ...lifecycleController,
        done: async () => {
          const result = await lifecycleController.done();
          setLifecycleJob(null);
          clearFeedback();
          onRefresh();
          return result;
        },
      }}
      description={lifecycleJob.message || "Vaelor is tracking this app action, and will keep tracking it if the appliance restarts."}
      operation={lifecycleController.operation}
      title="Current app operation"
    />
  ) : <div className="manage-operation" role="status"><strong>Current app operation</strong><span>Connecting to the durable operation record…</span></div>) : null;

  return (
    <div className="manage-layout">
      <div className="manage-layout__lists">
        <ManageServicesCard apps={apps} selectedId={selectedModel ? null : selected?.id ?? null} state={listState} onChoose={(app) => openTool(app, "overview")} />
        <ManageModelsCard
          busy={busy}
          models={models}
          selectedId={selectedModelId}
          session={session}
          state={listState}
          onRemove={(resource) => void reviewRemoval(resource)}
          onUse={(model) => { clearFeedback(); setSelectedModelId(model.id); }}
        />
      </div>

      <div className="manage-layout__detail">
        {selectedModel ? (
          <>
            {/* A refused switch is said beside the panel that asked for it. */}
            {feedback}
            <ModelSwitchPanel busy={busy} key={selectedModel.id} model={selectedModel} onClose={() => setSelectedModelId("")} onSwitch={(mode) => void activateModel(mode)} />
          </>
        ) : selected ? (
          <AppManagerPanel
            app={selected}
            busy={busy}
            configNotApplied={configNotApplied?.appId === selected.id && selected.project ? configNotApplied.when : null}
            notice={feedback}
            operation={operation}
            recoveryRefresh={recoveryRefresh}
            session={session}
            tool={tool}
            onClose={closeManager}
            onConfigSaved={(when) => setConfigNotApplied(when ? { appId: selected.id, when } : null)}
            onCopied={(message, failed) => setFeedback(message, failed ? "error" : "success")}
            onLifecycle={setPendingLifecycle}
            onReviewRemoval={() => void reviewRemoval({
              kind: selected.project === "model-assistant" ? "runtime" : "app",
              id: selected.project === "model-assistant" ? "model-assistant" : selected.app_instance_id ?? selected.id,
              name: selected.name,
              display_identity: selected.name,
              project: selected.project ?? undefined,
            })}
            onSelectTool={(next) => openTool(selected, next)}
          />
        ) : (
          <>
            {feedback}
            {apps.length > 0 && (
              <div className="ui-card manage-panel manage-panel--empty">
                <EmptyState icon={null} text={isAdministrator ? "Choose a service to open its app manager, or a model to change how it is used." : "Choose a service to open its app manager."} title="Nothing chosen" />
              </div>
            )}
          </>
        )}
      </div>

      {/* The plan takes seconds to prepare on a Pi. Show the dialog straight
          away so the click has a visible result instead of looking dead. */}
      {removalPending && !removalPlan && <RemovalCheckingDialog name={removalPending.name} onCancel={cancelRemovalReview} />}
      <WorkloadRemovalDialog key={removalPlan?.plan_digest ?? "closed"} busy={busy} error={removalError} plan={removalPlan} onCancel={() => !busy && setRemovalPlan(null)} onConfirm={(options) => void queueRemoval(options)} />
      <ActionReviewDialog
        busy={busy}
        job={pendingLifecycle && selected?.project ? { type: `compose.${pendingLifecycle}`, payload: { project: selected.project } } : null}
        onApprove={() => {
          const action = pendingLifecycle;
          setPendingLifecycle(null);
          if (action) void lifecycle(action);
        }}
        onCancel={() => setPendingLifecycle(null)}
        summary={pendingLifecycle && selected ? pendingLifecycle === "backup" ? `Create and verify a restorable checkpoint for ${selected.name}.` : `${pendingLifecycle[0].toUpperCase()}${pendingLifecycle.slice(1)} ${selected.name} only after the reviewed preflight checks pass.` : ""}
        suggestedActions={["Keep this manager open while Vaelor reports progress and verifies health.", "Use Cancel operation if the change is still safely interruptible."]}
      />
    </div>
  );
}
