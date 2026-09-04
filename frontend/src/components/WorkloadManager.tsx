import { useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { formatQuantity } from "../lib/format";
import { useOperationOwner } from "../hooks/useOperationOwner";
import type { Session } from "../types";
import { ActionReviewDialog } from "./ActionReviewDialog";
import { AppFileManager } from "./AppFileManager";
import { AppLogViewer } from "./AppLogViewer";
import { Icon } from "./Icon";
import { ModalShell } from "./ModalShell";
import { OperationOwner } from "./OperationOwner";
import { Button, Input, Notice, OperationFeedback, TabSet, Textarea, type OperationState } from "./ui";
import type { AppSetup, AppTemplate } from "./AppCatalog";
import { PaginatedItems } from "./PaginatedItems";
import { RecoveryPointList } from "./RecoveryPointList";
import { StatusPill } from "./StatusPill";
import { WorkloadRemovalDialog, type RemovalResource, type WorkloadRemovalOptions, type WorkloadRemovalPlan } from "./WorkloadRemovalDialog";

export interface ManagedInventory {
  apps: ManagedApp[];
  models: ManagedModel[];
}

interface ManagedApp {
  id: string;
  app_instance_id?: string | null;
  name: string;
  image: string;
  status: string;
  health?: string | null;
  running: boolean;
  managed: boolean;
  project?: string | null;
  /**
   * Catalog blueprint this managed app was installed from, read from the
   * `io.pironman.template` container label by workload_inventory.py. Absent for
   * discovered (non-blueprint) containers and on older servers. Used to map the
   * app to its catalog `setup` guidance in the Overview tab.
   */
  template_id?: string | null;
  ports: Array<{ container: string; host: string; address: string }>;
  /**
   * Host port of the app's actual web UI, resolved server-side. Multi-port apps
   * publish more than one host port — Syncthing exposes both its 8384 GUI and
   * its 22000 sync protocol — and the first published port is not reliably the
   * web one. When present it is the port to open; absent on older servers, where
   * appEndpoint() falls back to the first published port.
   */
  web_port?: number | null;
  capabilities: { logs: boolean; configuration: boolean; console: boolean; remote_desktop: boolean };
  remote_desktop?: { kind: string; host_port: string } | null;
}

interface ManagedModel {
  id: string;
  name: string;
  file: string;
  path: string;
  size_bytes: number;
  status: string;
  in_use?: boolean;
  /** Whether the catalog names this file (#147); absent on older servers. */
  catalog?: boolean;
  /**
   * Which tier this model serves: "ai-chat" (GPU AI Chat) or "assistant" (the
   * NPU Assistant). "" / absent when the catalog does not name the file. Drives
   * the switch panel's wording and is sent back on the deploy so a GPU chat
   * model is never labelled — or routed — as the Assistant's.
   */
  surface?: string;
}

type Tool = "overview" | "logs" | "configuration" | "files" | "filemanager" | "console" | "remote";
type LifecycleAction = "start" | "stop" | "restart" | "update" | "backup";
interface LifecycleJob { id: string; type: string; state: string; operation_state?: string; attention?: boolean; retryable?: boolean; readiness?: string; liveness?: string; progress: number; message: string; resource_id?: string; display_identity?: string }
interface ManagedFile { path: string; size: number; exists: boolean }

const lifecycleStorageKey = "vaelor.workloads.live_job";
const workloadTools = new Set<Tool>(["overview", "logs", "configuration", "files", "filemanager", "console", "remote"]);

/**
 * How a model's serving tier reads to a person: the GPU "AI Chat" model or the
 * NPU "Assistant". A model with no catalog surface is treated as the Assistant,
 * the historical default this switch panel always named.
 */
function modelTierLabel(model: ManagedModel): string {
  return model.surface === "ai-chat" ? "AI Chat" : "Assistant";
}

function appEndpoint(app: ManagedApp): string | null {
  // A multi-port app (e.g. Syncthing) publishes both its web GUI and a
  // non-HTTP protocol port, and the first published port is not reliably the
  // web one — opening it lands the user on the sync protocol, not the UI. The
  // server resolves the real web port into `web_port`; prefer it. Fall back to
  // the first published port only for older servers that do not send it.
  if (app.web_port) return `http://${window.location.hostname}:${app.web_port}`;
  const published = app.ports.find((port) => port.host)?.host.split("/")[0];
  return published ? `http://${window.location.hostname}:${published}` : null;
}

/** Human label for a revealed secret env key, e.g. PASSWORD -> "Password". */
function secretLabel(key: string): string {
  if (/password/i.test(key)) return "Password";
  return key.replace(/_/g, " ").toLowerCase().replace(/^./, (c) => c.toUpperCase());
}

/**
 * Task #76. Health values that mean *"not settled yet"*, not *"wrong"*.
 *
 * Docker reports `starting` for the whole of a container's start period, so a
 * restart that has finished successfully still reads `starting` for tens of
 * seconds afterwards. Two things went wrong with that. These name the state;
 * the settle poll in `Workloads` keeps re-reading while it holds.
 */
export const settlingHealth = new Set(["starting", "created", "restarting"]);

/** Whether any managed app is still moving, so the list is not a resting view. */
export function inventoryIsSettling(apps: ReadonlyArray<{ running: boolean; health?: string | null }>): boolean {
  return apps.some((app) => app.running && settlingHealth.has((app.health || "").toLowerCase()));
}

export function managedAppStatus(app: ManagedApp): { label: string; status: "healthy" | "degraded" | "critical" | "neutral" } {
  if (!app.running) return { label: "Stopped", status: "neutral" };
  const health = (app.health || "").toLowerCase();
  if (["unhealthy", "critical", "failed", "error"].includes(health)) return { label: "Blocked", status: "critical" };
  /*
   * Task #76. `starting` used to land in "Needs attention" beside `degraded`
   * and `warning`. It is neither: a container inside its start period has not
   * failed anything and there is nothing for the reader to do. Reported
   * alongside "App restart completed / FINISHED / 100%" and a passing
   * "Application health after restart" preflight, "NEEDS ATTENTION" reads as a
   * contradiction of the operation that just succeeded.
   *
   * Same three-state rule as the connection indicator: in-progress is its own
   * answer and may borrow neither the reassuring display nor the alarming one.
   */
  if (settlingHealth.has(health)) return { label: "Starting", status: "neutral" };
  if (["degraded", "warning"].includes(health)) return { label: "Needs attention", status: "degraded" };
  return { label: "Running", status: "healthy" };
}

function restoredLifecycleJob(): LifecycleJob | null {
  try {
    const value = JSON.parse(window.localStorage.getItem(lifecycleStorageKey) || "null");
    return value && typeof value.id === "string" && typeof value.type === "string" ? value : null;
  } catch {
    return null;
  }
}


export function WorkloadManager({
  inventory,
  session,
  onRefresh,
}: {
  inventory: ManagedInventory | null;
  session: Session;
  onRefresh: () => void;
}) {
  const [selected, setSelected] = useState<ManagedApp | null>(null);
  const [tool, setTool] = useState<Tool>("overview");
  const [output, setOutput] = useState("");
  const [command, setCommand] = useState("");
  const [config, setConfig] = useState("");
  const [notice, setNotice] = useState("");
  const [noticeState, setNoticeState] = useState<OperationState>("idle");
  const setFeedback = (message: string, state: OperationState = "success") => {
    setNotice(message);
    setNoticeState(message ? state : "idle");
  };
  const clearFeedback = () => setFeedback("", "idle");
  const [busy, setBusy] = useState(false);
  const [remoteUrl, setRemoteUrl] = useState("");
  const [selectedModel, setSelectedModel] = useState<ManagedModel | null>(null);
  const [removalPlan, setRemovalPlan] = useState<WorkloadRemovalPlan | null>(null);
  const [removalPending, setRemovalPending] = useState<RemovalResource | null>(null);
  const removalRequest = useRef<string | null>(null);
  const [removalError, setRemovalError] = useState("");
  const [runtimeMode, setRuntimeMode] = useState<"efficient" | "balanced" | "quality">("balanced");
  const [setupGuide, setSetupGuide] = useState<AppSetup | null>(null);
  const [secretEnv, setSecretEnv] = useState<string[]>([]);
  const [configFiles, setConfigFiles] = useState<string[]>([]);
  const [secrets, setSecrets] = useState<Record<string, string> | null>(null);
  const [files, setFiles] = useState<ManagedFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState("");
  const [recoveryRefresh, setRecoveryRefresh] = useState(0);
  const [pendingLifecycle, setPendingLifecycle] = useState<LifecycleAction | null>(null);
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

  useEffect(() => {
    if (selected || lifecycleRestoreDismissed || !lifecycleJob?.resource_id || !inventory) return;
    const app = inventory.apps.find((candidate) => candidate.id === lifecycleJob.resource_id || candidate.app_instance_id === lifecycleJob.resource_id);
    if (app) {
      setSelected(app);
      setTool("overview");
    }
  }, [inventory, lifecycleJob?.resource_id, lifecycleRestoreDismissed, selected]);

  useEffect(() => {
    if (!selected) return;
    const current = inventory?.apps.find((app) => app.id === selected.id);
    if (current) {
      setSelected(current);
    } else if (inventory) {
      setSelected(null);
    }
  }, [inventory, selected?.id]);

  /*
   * Post-deploy setup guidance for the open app. A managed app carries the
   * blueprint it was installed from as `template_id`; the public catalog names
   * the matching `setup` block. Only blueprint apps have a template id, so the
   * fetch is gated on it — a discovered container falls straight through to the
   * generic "open the app to finish its own setup" line without a needless
   * request. Any failure (older server, offline catalog) degrades to that same
   * line rather than surfacing an error in a read-only helper.
   */
  useEffect(() => {
    const templateId = selected?.template_id;
    if (!templateId) {
      setSetupGuide(null);
      setSecretEnv([]);
      setConfigFiles([]);
      return;
    }
    let active = true;
    void apiRequest<AppTemplate[]>("/apps/catalog")
      .then((templates) => {
        if (!active) return;
        const template = templates.find((candidate) => candidate.id === templateId);
        setSetupGuide(template?.setup ?? null);
        setSecretEnv(template?.secret_env ?? []);
        setConfigFiles(template?.config_files ?? []);
      })
      .catch(() => {
        if (!active) return;
        setSetupGuide(null);
        setSecretEnv([]);
        setConfigFiles([]);
      });
    return () => { active = false; };
  }, [selected?.template_id]);

  /*
   * Reveal the Vaelor-generated secret(s) in the setup panel. code-server's
   * setup step tells the user to "copy the generated password shown in this
   * panel", so the value must actually be here. Only fetch when the template
   * declares secret_env — a needless admin-only request otherwise. A non-admin
   * (403) or any error degrades to a muted "sign in with the password set for
   * this app" line; the secret is never logged or cached beyond this state.
   */
  useEffect(() => {
    setSecrets(null);
    if (!selected || secretEnv.length === 0) return;
    let active = true;
    void apiRequest<{ secrets: Record<string, string> }>(`/managed/apps/${selected.id}/credentials`)
      .then((data) => {
        if (active) setSecrets(data.secrets ?? null);
      })
      .catch(() => {
        if (active) setSecrets(null);
      });
    return () => { active = false; };
  }, [selected?.id, secretEnv]);

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
      const requestedTool = query.get("appTool") as Tool | null;
      const nextTool = requestedTool && workloadTools.has(requestedTool) ? requestedTool : "overview";
      setLifecycleRestoreDismissed(false);
      setSelected(app);
      setTool(nextTool);
      if (nextTool === "configuration") {
        setBusy(true);
        void apiRequest<{ content: string }>(`/managed/apps/${app.id}/config`)
          .then((data) => setConfig(data.content))
          .catch((error) => setFeedback(error instanceof Error ? error.message : "Configuration could not be loaded.", "error"))
          .finally(() => setBusy(false));
      }
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
    updateManagerRoute(null);
  };

  const openTool = async (app: ManagedApp, nextTool: Tool) => {
    setLifecycleRestoreDismissed(false);
    setSelected(app);
    setTool(nextTool);
    clearFeedback();
    setOutput("");
    setRemoteUrl("");
    updateManagerRoute(app, nextTool);
    if (nextTool === "configuration") {
      setBusy(true);
      try {
        const data = await apiRequest<{ content: string }>(`/managed/apps/${app.id}/config`);
        setConfig(data.content);
      } catch (error) {
        setFeedback(error instanceof Error ? error.message : "This tool is unavailable.", "error");
      } finally {
        setBusy(false);
      }
    }
    if (nextTool === "files") {
      setFiles([]);
      setSelectedFile(null);
      setFileContent("");
      setBusy(true);
      try {
        const data = await apiRequest<{ files: ManagedFile[] }>(`/managed/apps/${app.id}/files`);
        setFiles(data.files ?? []);
      } catch (error) {
        setFeedback(error instanceof Error ? error.message : "This app's files could not be listed.", "error");
      } finally {
        setBusy(false);
      }
    }
  };

  const openFile = async (path: string) => {
    if (!selected) return;
    setSelectedFile(path);
    setFileContent("");
    setBusy(true);
    clearFeedback();
    try {
      const data = await apiRequest<{ path: string; content: string }>(
        `/managed/apps/${selected.id}/files/content?path=${encodeURIComponent(path)}`,
      );
      setFileContent(data.content);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "That file could not be opened.", "error");
    } finally {
      setBusy(false);
    }
  };

  const saveFile = async () => {
    if (!selected || !selectedFile) return;
    setBusy(true);
    try {
      await apiRequest(
        `/managed/apps/${selected.id}/files/content`,
        { method: "PUT", body: JSON.stringify({ path: selectedFile, content: fileContent }) },
        session.csrf_token,
      );
      setFeedback(`${selectedFile} saved.`);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "The file could not be saved.", "error");
    } finally {
      setBusy(false);
    }
  };

  const copySecret = async (value: string) => {
    try {
      await navigator.clipboard.writeText(value);
      setFeedback("Copied to clipboard.");
    } catch {
      setFeedback("Copy failed. Select the value and copy it manually.", "error");
    }
  };

  const runDiagnostic = async (diagnostic: "stats" | "processes") => {
    if (!selected) return;
    setBusy(true);
    try {
      const data = await apiRequest<{ output: string }>(
        `/managed/apps/${selected.id}/diagnostics`,
        { method: "POST", body: JSON.stringify({ tool: diagnostic }) },
        session.csrf_token,
      );
      setOutput(data.output);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "The diagnostic could not run.", "error");
    } finally {
      setBusy(false);
    }
  };

  const runCommand = async () => {
    if (!selected || !command.trim()) return;
    setBusy(true);
    try {
      const data = await apiRequest<{ output: string; exit_code: number; truncated: boolean }>(
        `/managed/apps/${selected.id}/exec`,
        { method: "POST", body: JSON.stringify({ command }) },
        session.csrf_token,
      );
      const status = `\n[exit ${data.exit_code}${data.truncated ? ", output truncated" : ""}]`;
      setOutput((data.output || "") + status);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "The command could not run.", "error");
    } finally {
      setBusy(false);
    }
  };

  const saveConfig = async () => {
    if (!selected) return;
    setBusy(true);
    try {
      await apiRequest(
        `/managed/apps/${selected.id}/config`,
        { method: "PUT", body: JSON.stringify({ content: config }) },
        session.csrf_token,
      );
      setFeedback("Configuration validated and saved. A backup was created.");
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "Configuration could not be saved.", "error");
    } finally {
      setBusy(false);
    }
  };

  const startRemoteDesktop = async () => {
    if (!selected) return;
    setBusy(true);
    clearFeedback();
    try {
      const data = await apiRequest<{ url: string }>(
        `/managed/apps/${selected.id}/remote-desktop`,
        { method: "POST", body: "{}" },
        session.csrf_token,
      );
      setRemoteUrl(data.url);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "Remote Desktop could not start.", "error");
    } finally {
      setBusy(false);
    }
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
      setFeedback(`${action[0].toUpperCase()}${action.slice(1)} approved. Progress remains here while Vaelor verifies the application.`, "pending");
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "The app action could not be queued.", "error");
    } finally {
      setBusy(false);
    }
  };

  const activateModel = async () => {
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
            mode: runtimeMode,
            ...(selectedModel.surface ? { surface: selectedModel.surface } : {}),
          },
        }),
      }, session.csrf_token);
      setFeedback(`${selectedModel.name} will replace the current local ${modelTierLabel(selectedModel)} model after its health check passes.`, "pending");
      setSelectedModel(null);
      window.setTimeout(onRefresh, 2500);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "The model switch could not be queued.", "error");
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
      // this ordering, a fast response can render the modal for one frame while
      // its approval controls still inherit the request's busy state.
      setBusy(false);
      setRemovalPlan(plan);
    } catch (error) {
      if (removalRequest.current !== key) return;
      setRemovalPending(null);
      setFeedback(error instanceof Error ? error.message : "The dependency report could not be prepared.", "error");
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
   * immediately.
   *
   * The plan request can take many seconds on a Pi and cannot be aborted, and
   * `busy` gates every control in this manager. Waiting for the in-flight
   * request to settle left the whole screen disabled with no way to recover —
   * indefinitely if the request never returned, which is exactly the case that
   * makes someone press Cancel. Releasing the guard here also makes the late
   * response a no-op.
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
    try {
      const result = await apiRequest<LifecycleJob>("/jobs", {
        method: "POST",
        body: JSON.stringify({
          type: "managed.remove",
          payload: {
            kind: removalPlan.resource.kind,
            id: removalPlan.resource.id,
            display_identity: removalPlan.resource.display_identity ?? removalPlan.display_identity ?? removalPlan.resource.name,
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
      setLifecycleJob({ ...result, resource_id: removalPlan.resource.id, display_identity: removalPlan.resource.display_identity ?? removalPlan.display_identity ?? removalPlan.resource.name });
      setFeedback(`${removalPlan.resource.display_identity ?? removalPlan.display_identity ?? removalPlan.resource.name} removal approved.${backup}${data} Progress and verification remain in this manager.`, "pending");
      setRemovalPlan(null);
    } catch (error) {
      setRemovalError(error instanceof Error ? error.message : "The reviewed removal could not be queued. Refresh the dependency report.");
    } finally {
      setBusy(false);
    }
  };

  const apps = inventory?.apps ?? [];
  const models = inventory?.models ?? [];
  /*
   * Task #122. `inventory` is null when `/managed` has not answered — and on
   * the appliance it did not answer, because the read timed out while the
   * machine was loading a model, and `api.ts` reports any failed fetch as a
   * 503. Both lists then render `[]`, and the screen told the owner it had no
   * apps and no models while five containers were running and a 2.33 GiB
   * model sat on disk.
   *
   * An empty list and an unread list are different facts. The first is about
   * the appliance; the second is about this browser's last request, and the
   * only honest thing to say about it is that nothing is known yet.
   */
  const inventoryUnread = inventory === null;
  return (
    <div className="manage-workloads">
      <section className="managed-section">
        <div className="panel-heading">
          <div><span className="page-eyebrow">Installed apps</span><h2>Your services</h2><p>Open an app to see its health and management tools.</p></div>
          <Button variant="quiet" onClick={onRefresh}>Reload</Button>
        </div>
        {apps.length ? (
          <div className="managed-card-grid">
            <PaginatedItems items={apps} label="Installed applications" pageSize={8} render={(app) => (
              <article className="managed-card" key={app.id}>
                <div className="managed-card__head">
                  <span><Icon name="database" /></span>
                  <StatusPill {...managedAppStatus(app)} />
                </div>
                <h3>{app.name}</h3><p>{app.image}</p>
                <div className="managed-card__meta">
                  <span>{app.ports.length ? (() => {
                    const ports = [...new Set(app.ports.map((port) => port.host))];
                    return `${ports.length} published port${ports.length === 1 ? "" : "s"} · ${ports.join(", ")}`;
                  })() : "No published ports"}</span>
                  <span>{app.managed ? "Managed by Vaelor" : "Discovered app"}</span>
                </div>
                <Button onClick={() => void openTool(app, "overview")}>Manage app</Button>
                {appEndpoint(app) && <a className="ui-button ui-button--quiet" href={appEndpoint(app)!} rel="noopener noreferrer" target="_blank">Open {app.name}</a>}
              </article>
            )} />
          </div>
        ) : inventoryUnread ? <div className="managed-empty"><Icon name="database" size={30} /><h3>The installed-app list could not be read</h3><p>This is not a claim that nothing is installed — the appliance did not answer in time. Use Refresh above.</p></div>
          : <div className="managed-empty"><Icon name="database" size={30} /><h3>No apps installed yet</h3><p>Return to Install and choose an app. It will appear here automatically.</p></div>}
      </section>

      <section className="managed-section">
        <div className="panel-heading"><div><span className="page-eyebrow">AI models</span><h2>Downloaded models</h2><p>See models stored locally and ready to use.</p></div></div>
        {models.length ? <div className="managed-model-list"><PaginatedItems items={models} label="Downloaded AI models" pageSize={8} render={(model) => (
          <div key={model.id}><span><Icon name="cpu" /></span><div><strong>{model.name}</strong><small>{model.file}</small></div><b>{formatQuantity(model.size_bytes, "model")}</b><StatusPill label={model.in_use ? "In use" : model.status} status={model.in_use ? "healthy" : "neutral"} />{/* #145: "Use model" sat on the row badged "In use" — an offer to switch
    to the model already serving. The in-use row offers its runtime
    settings instead, under a label that says what it is.
    #147: a file the catalog does not name has no verified identity and no
    measured footprint, so it is not offerable — the row states exactly that
    instead of leading to an unsupported configuration. It must not guess at
    provenance: `catalog: false` says the filename is unknown to the catalog,
    not who put the file there. */}<div className="managed-model-list__actions">{model.catalog === false && !model.in_use ? <small className="managed-model-list__note">Not in Vaelor's verified catalog — no measured settings, so the Assistant does not offer it.</small> : <Button disabled={busy || session.user.role === "viewer"} onClick={() => setSelectedModel(model)}>{model.in_use ? "Adjust runtime…" : "Use model"}</Button>}<Button variant="danger" disabled={busy || session.user.role !== "administrator"} onClick={() => void reviewRemoval({ kind: "model", id: model.id, name: model.name, display_identity: model.name, path: model.path })}>{model.in_use ? "Review and remove…" : "Remove…"}</Button></div></div>
        )} /></div> : inventoryUnread ? <div className="managed-empty managed-empty--compact"><Icon name="cpu" size={28} /><h3>The downloaded-model list could not be read</h3><p>Nothing here says a model is missing — the appliance did not answer in time. Use Refresh above.</p></div>
          : <div className="managed-empty managed-empty--compact"><Icon name="cpu" size={28} /><h3>No local models yet</h3><p>Choose a model from Install to add private AI.</p></div>}
        {notice && !selected && !lifecycleController.isTerminal && <OperationFeedback className="managed-operation-feedback" message={notice} state={noticeState} />}
      </section>

      {selectedModel && (
        <section className="model-switch-panel" aria-labelledby="model-switch-title">
          <div className="panel-heading">
            <div><span className="page-eyebrow">{modelTierLabel(selectedModel)} model</span><h2 id="model-switch-title">Use {selectedModel.name}</h2><p>Vaelor will size the runtime from current RAM and preserve memory for the operating system.</p></div>
            <Button variant="quiet" onClick={() => setSelectedModel(null)}>Close</Button>
          </div>
          <div className="runtime-mode-grid">
            {([
              ["efficient", "Memory saver", "2K context · most RAM left for apps"],
              ["balanced", "Balanced", "4K context · recommended"],
              ["quality", "Long context", "Up to 8K context · automatically reduced if RAM is tight"],
            ] as const).map(([id, name, description]) => (
              <Button aria-pressed={runtimeMode === id} key={id} onClick={() => setRuntimeMode(id)}>
                <strong>{name}</strong><span>{description}</span>
              </Button>
            ))}
          </div>
          <div className="model-switch-panel__actions">
            <span><Icon name="shield" /> Switching briefly restarts local AI. If the replacement fails, Vaelor restores the previous managed service.</span>
            <Button variant="primary" disabled={busy} onClick={() => void activateModel()}>{busy ? "Preparing…" : "Review and switch model"}</Button>
          </div>
        </section>
      )}

      {selected && (
        <ModalShell labelledBy="app-manager-title" onClose={() => { if (!busy) closeManager(); }} size="wide">
        <section className="app-drawer" aria-labelledby="app-manager-title">
          <div className="app-drawer__header">
            <div><span className="page-eyebrow">App manager</span><h2 id="app-manager-title">{selected.name}</h2><p>{selected.image}</p></div>
            <Button variant="quiet" onClick={closeManager}>Close</Button>
          </div>
          {/* Same defect as the section tabs: `role="tab"` with no tablist
              ownership, no `aria-controls`, and no arrow-key movement. The
              shared TabSet primitive owns that pattern. */}
          <TabSet
            items={([
              ["overview", "Overview", true],
              ["logs", "Logs", selected.capabilities.logs],
              ["configuration", "Configuration", selected.capabilities.configuration],
              ["files", "Config files", configFiles.length > 0],
              ["filemanager", "Files", selected.managed],
              ["console", "Console", selected.capabilities.console],
              ["remote", "Remote desktop", selected.capabilities.remote_desktop],
            ] as Array<[Tool, string, boolean]>)
              // The "Config files" tab exists only for apps whose catalog
              // template names editable config files, and the general "Files"
              // browser only for managed apps; unlike the capability-gated tabs
              // both are hidden entirely rather than shown disabled.
              .filter(([id, , available]) => (id !== "files" && id !== "filemanager") || available)
              .map(([id, label, available]) => ({
              id,
              label,
              disabled: !available,
            }))}
            label={`${selected.name} tools`}
            listClassName="app-tool-tabs"
            panelClassName="workload-tab-panel"
            onSelect={(id) => { if (workloadTools.has(id as Tool)) void openTool(selected, id as Tool); }}
            selectedId={tool}
          >
          <div className="app-tool-panel">
            {tool === "overview" && <div className="app-overview-grid">
              {(() => {
                const endpoint = appEndpoint(selected);
                const steps = setupGuide?.steps ?? [];
                const hasSteps = steps.length > 0;
                const hasGuidance = Boolean(setupGuide && (hasSteps || setupGuide.login || setupGuide.password_source || setupGuide.https_required || setupGuide.first_run_wizard || setupGuide.note));
                const httpsRequired = Boolean(setupGuide?.https_required);
                // The template declares a generated secret. When the value is
                // readable (admin), show it as a copyable fact — code-server's
                // step points the user right here. A non-admin/error left it
                // null; degrade to a muted line rather than an empty panel.
                const revealedSecret = secretEnv.length > 0 ? (
                  secrets && Object.keys(secrets).length > 0 ? (
                    <dl className="app-setup__facts app-setup__secrets">
                      {Object.entries(secrets).map(([key, value]) => (
                        <div key={key}>
                          <dt>{secretLabel(key)}</dt>
                          <dd className="app-setup__secret-value">
                            <code>{value}</code>
                            <Button variant="quiet" onClick={() => void copySecret(value)}>Copy</Button>
                          </dd>
                        </div>
                      ))}
                    </dl>
                  ) : (
                    <p className="app-setup__fallback">Sign in with the password set for this app.</p>
                  )
                ) : null;
                return (
                  <div className="app-setup">
                    <div className="app-setup__heading">
                      <div><small>Finish setup</small><strong>{selected.name}</strong></div>
                      {endpoint && !httpsRequired && <a className="ui-button ui-button--primary" href={endpoint} rel="noopener noreferrer" target="_blank">Open &amp; configure {selected.name}</a>}
                    </div>
                    <div className="app-setup__guide">
                      {hasGuidance ? <>
                        {hasSteps && <ol className="app-setup__steps">
                          {steps.map((step, index) => <li key={index}>{step}</li>)}
                        </ol>}
                        {setupGuide!.https_required && <Notice severity="warning" heading="HTTPS required">This app needs a trusted HTTPS reverse proxy (TLS) before you can sign in or create an account. The plain <code>http://</code> address won&apos;t let you finish setup, so set up TLS and reach the app over HTTPS first.</Notice>}
                        {setupGuide!.first_run_wizard && <div className="tool-explainer"><Icon name="settings" /><span><strong>Opens its own setup wizard</strong><small>This app walks you through the rest of setup on your first visit.</small></span></div>}
                        {(setupGuide!.login || setupGuide!.password_source || setupGuide!.note) && <dl className="app-setup__facts">
                          {setupGuide!.login && <div><dt>Default sign-in</dt><dd>{setupGuide!.login}</dd></div>}
                          {setupGuide!.password_source && <div><dt>Password</dt><dd>{setupGuide!.password_source}</dd></div>}
                          {setupGuide!.note && <div><dt>Good to know</dt><dd>{setupGuide!.note}</dd></div>}
                        </dl>}
                      </> : endpoint
                        ? <p className="app-setup__fallback">Open the app to finish its own setup.</p>
                        : <p className="app-setup__fallback">This app has no web interface to open.</p>}
                      {revealedSecret}
                    </div>
                  </div>
                );
              })()}
              <div><small>Status</small><strong>{managedAppStatus(selected).label}</strong></div>
              <div><small>Management</small><strong>{selected.managed ? "Full controls" : "Read-only discovery"}</strong></div>
              <div><small>Network</small><strong>{selected.ports.length ? [...new Set(selected.ports.map((port) => port.host))].join(", ") : "No public ports"}</strong></div>
              {selected.managed && selected.project && <div className="app-lifecycle">
                <small>App controls</small>
                <div>
                  {!selected.running && <Button variant="primary" disabled={busy} onClick={() => setPendingLifecycle("start")}>Start</Button>}
                  {selected.running && <Button disabled={busy} onClick={() => setPendingLifecycle("restart")} variant="quiet">Restart</Button>}
                  {selected.running && <Button disabled={busy} onClick={() => setPendingLifecycle("stop")} variant="danger">Stop</Button>}
                  <Button disabled={busy} onClick={() => setPendingLifecycle("update")} variant="quiet">Review update</Button>
                  <Button disabled={busy} onClick={() => setPendingLifecycle("backup")} variant="primary">Review checkpoint</Button>
                </div>
                {lifecycleJob?.resource_id === selected.id && (lifecycleController.operation ? (
                  <OperationOwner
                    className="app-lifecycle__progress"
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
                ) : <div className="app-lifecycle__progress" role="status"><strong>Current app operation</strong><span>Connecting to the durable operation record…</span></div>)}
              </div>}
              {selected.managed && selected.project && (
                <div className="app-recovery">
                  <div className="app-recovery__heading">
                    <div><small>Recovery</small><strong>Restore points for this app</strong></div>
                    <span>Restore is available here as soon as a checkpoint finishes.</span>
                  </div>
                  <RecoveryPointList
                    project={selected.project}
                    refreshSignal={recoveryRefresh}
                    session={session}
                  />
                </div>
              )}
              {selected.managed && selected.project && session.user.role === "administrator" && <div className="app-danger-zone"><small>Removal</small><strong>Dependency-aware removal</strong><p>Review sibling services, active AI connections, retained data, and the recovery checkpoint before anything is removed.</p><Button variant="danger" disabled={busy} onClick={() => void reviewRemoval({ kind: selected.project === "model-assistant" ? "runtime" : "app", id: selected.project === "model-assistant" ? "model-assistant" : selected.app_instance_id ?? selected.id, name: selected.name, display_identity: selected.name, project: selected.project ?? undefined })}>Review removal impact</Button></div>}
            </div>}
            {tool === "logs" && <AppLogViewer appId={selected.id} appName={selected.name} />}
            {tool === "configuration" && <>
              <div className="tool-explainer"><Icon name="shield" /><span><strong>Safe configuration editor</strong><small>Vaelor validates changes and creates a backup before saving.</small></span></div>
              <Textarea className="config-editor" disabled={busy || session.user.role !== "administrator"} label="Configuration YAML" onChange={(event) => setConfig(event.target.value)} spellCheck={false} value={config} />
              <Button variant="primary" disabled={busy} disabledReason={session.user.role === "administrator" ? undefined : "Administrator access is required to change an app configuration."} onClick={() => void saveConfig()}>Validate and save</Button>
            </>}
            {tool === "files" && <>
              <div className="tool-explainer"><Icon name="shield" /><span><strong>Configuration files</strong><small>Edit the files this app reads. Vaelor writes only the files listed here.</small></span></div>
              <div className="app-files">
                <ul className="app-files__list">
                  {files.length === 0 && <li className="app-files__empty">No editable files were found for this app.</li>}
                  {files.map((file) => (
                    <li key={file.path}>
                      <Button variant={selectedFile === file.path ? "primary" : "quiet"} disabled={busy} onClick={() => void openFile(file.path)}>
                        <strong>{file.path}</strong>
                        <span>{file.exists ? formatQuantity(file.size, "used") : "Not created yet"}</span>
                      </Button>
                    </li>
                  ))}
                </ul>
                {selectedFile && <div className="app-files__editor">
                  <Textarea className="config-editor" disabled={busy || session.user.role !== "administrator"} label={`Editing ${selectedFile}`} onChange={(event) => setFileContent(event.target.value)} spellCheck={false} value={fileContent} />
                  <Button variant="primary" disabled={busy} disabledReason={session.user.role === "administrator" ? undefined : "Administrator access is required to edit configuration files."} onClick={() => void saveFile()}>Save file</Button>
                </div>}
              </div>
            </>}
            {tool === "filemanager" && <AppFileManager appId={selected.id} role={session.user.role} csrfToken={session.csrf_token} />}
            {tool === "console" && <>
              <div className="tool-explainer"><Icon name="terminal" /><span><strong>App console</strong><small>Run a command inside this app&apos;s container - to reset a password or change a setting. Administrator only; each command is audited and confined to the container.</small></span></div>
              {session.user.role === "administrator" ? (
                <div className="app-console-run">
                  <Input
                    label="Command"
                    placeholder="e.g. filebrowser users update admin --password NEW"
                    value={command}
                    disabled={busy}
                    onChange={(event) => setCommand(event.target.value)}
                    onKeyDown={(event) => { if (event.key === "Enter") void runCommand(); }}
                  />
                  <Button variant="primary" disabled={busy || !command.trim()} onClick={() => void runCommand()}>Run</Button>
                </div>
              ) : (
                <div className="tool-explainer"><Icon name="shield" /><span><small>Administrator access is required to run commands in an app.</small></span></div>
              )}
              <div className="diagnostic-actions"><Button disabled={busy} onClick={() => void runDiagnostic("stats")}>Resource use</Button><Button disabled={busy} onClick={() => void runDiagnostic("processes")}>Running processes</Button></div>
              <pre className="app-console">{output || "Run a command, or choose a diagnostic below."}</pre>
            </>}
            {tool === "remote" && (remoteUrl ? (
              <div className="remote-desktop-session">
                <div className="tool-explainer"><Icon name="display" /><span><strong>Remote Desktop connected</strong><small>The access token is short-lived and valid for one connection.</small></span></div>
                <iframe
                  allow="clipboard-read; clipboard-write; fullscreen"
                  sandbox="allow-forms allow-same-origin allow-scripts"
                  src={remoteUrl}
                  title={`${selected.name} Remote Desktop`}
                />
                <Button onClick={() => setRemoteUrl("")}>End session</Button>
              </div>
            ) : (
              <div className="remote-desktop-panel">
                <Icon name="display" size={42} />
                <h3>VNC desktop ready</h3>
                <p>Vaelor will create a one-use 90-second connection ticket, then open the desktop inside this page.</p>
                <Button variant="primary" disabled={busy} onClick={() => void startRemoteDesktop()}>
                  {busy ? "Starting…" : "Open remote desktop"}
                </Button>
              </div>
            ))}
            {notice && !lifecycleController.isTerminal && <OperationFeedback className="managed-operation-feedback" message={notice} state={noticeState} />}
          </div>
          </TabSet>
        </section>
        </ModalShell>
      )}
      {/* The plan takes seconds to prepare on a Pi. Show the dialog straight
          away so the click has a visible result instead of looking dead. */}
      {removalPending && !removalPlan && (
        <ModalShell
          labelledBy="removal-loading-title"
          onClose={cancelRemovalReview}
          size="standard"
        >
          <div className="removal-loading">
            <h2 id="removal-loading-title">Checking what depends on {removalPending.name}</h2>
            <p role="status">Vaelor is reading the live workload graph so it can show exactly what would stop or remain.</p>
            {/* Escape, the backdrop and this button must all get the user out;
                a loading dialog with no exit is a keyboard trap. */}
            <Button onClick={cancelRemovalReview} type="button" variant="quiet">
              Cancel
            </Button>
          </div>
        </ModalShell>
      )}
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
