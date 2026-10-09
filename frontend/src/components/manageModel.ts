import type { StatusTone } from "./ui";
import { statusTone } from "./ui";

/*
 * The Manage tab's data and the pure rules that read it (VD-200, the Manage
 * boards). Split out of WorkloadManager (1,000-line ceiling); WorkloadManager
 * re-exports what other files import from it.
 */

export interface ManagedInventory {
  apps: ManagedApp[];
  models: ManagedModel[];
}

export interface ManagedApp {
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
  /** Ports bound only to loopback (W4d-D5): never an Open link. */
  local_only_ports?: number[];
  host_network?: boolean;
  /** file_manager (W4d-D4): the backend's catalog_template() rule; absent from an older appliance. */
  capabilities: { logs: boolean; configuration: boolean; console: boolean; remote_desktop: boolean; file_manager?: boolean };
  remote_desktop?: { kind: string; host_port: string } | null;
}

export interface ManagedModel {
  id: string;
  name: string;
  file: string;
  path: string;
  /** null when the observer could not read the size (W4d-D3); `size_reason` says why. */
  size_bytes: number | null;
  size_reason?: string;
  /** The window the GPU route runs whatever profile is chosen (W4d-D28). */
  serving_context?: number;
  /** Why it cannot be switched to now, in the backend's words (Mode B). */
  switch_refusal?: string;
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
  /** ACC-106: false when Docker could not be read - use is then unknown, never "not in use". */
  in_use_known?: boolean;
  /** The Assistant's compose file names this model, whether or not it is running. */
  selected?: boolean;
  /** Which running servers load it: "assistant", "ai-chat", "gpu-ai-chat", "npu-assistant". */
  served_by?: string[];
  /** "npu" for an on-device model folder, which is listed but not removed here. */
  kind?: string;
  removable?: boolean;
  status_reason?: string;
}

/** The app manager's tools; the ids are the `?appTool=` deep-link values. */
export type Tool = "overview" | "logs" | "configuration" | "files" | "filemanager" | "console" | "remote" | "restore";
export const workloadTools = new Set<Tool>(["overview", "logs", "configuration", "files", "filemanager", "console", "remote", "restore"]);

export type LifecycleAction = "start" | "stop" | "restart" | "update" | "backup";
export interface LifecycleJob { id: string; type: string; state: string; operation_state?: string; attention?: boolean; retryable?: boolean; readiness?: string; liveness?: string; progress: number; message: string; resource_id?: string; display_identity?: string }

/** W4d-D4: the backend's catalog_template() rule; `managed` only for an appliance too old to say. */
export const fileManagerAvailable = (app: ManagedApp) => app.capabilities.file_manager ?? app.managed;

/** Lifecycle controls, restore points and removal exist only for an app Vaelor runs as a project. */
export const lifecycleAvailable = (app: ManagedApp) => Boolean(app.managed && app.project);

/** What each running server is called on a model row. */
export const SERVED_BY_LABELS: Record<string, string> = {
  assistant: "Serving the Assistant",
  "ai-chat": "Serving AI Chat",
  "gpu-ai-chat": "Serving AI Chat on the GPU",
  "npu-assistant": "Serving the Assistant on the neural processor",
};

/** The operator's words for the backend's model states ("ready", "degraded"), never the slug itself (LESSONS 5). */
const modelStatusWords: Record<string, { label: string; tone: StatusTone }> = {
  // A file on disk that nothing serves: installed, grey - not a green reading of anything running.
  ready: { label: "Installed", tone: "neutral" },
  degraded: { label: "Needs attention", tone: "warning" },
};

/**
 * One row's badge, from what the backend MEASURED (ACC-106): a running server
 * is "In use"; a Docker that could not be read is "Use not checked", never a
 * quiet "ready"; a model the Assistant is set to but is not running is
 * "Selected, not running".
 */
export function modelUsePill(model: ManagedModel): { label: string; tone: StatusTone } {
  if (model.in_use) return { label: "In use", tone: "success" };
  if (model.in_use_known === false) return { label: "Use not checked", tone: "warning" };
  if (model.selected) return { label: "Selected, not running", tone: "neutral" };
  if (model.kind === "npu") return { label: "Installed", tone: "neutral" };
  return modelStatusWords[model.status] ?? { label: "Status not recognised", tone: statusTone(model.status) };
}

/** The served-by line under a model's name, or "" when nothing serves it. */
export function servedByLine(model: ManagedModel): string {
  return (model.served_by ?? []).map((server) => SERVED_BY_LABELS[server] ?? "Serving a model").join(" · ");
}

/** Human label for a revealed secret env key, e.g. PASSWORD -> "Password". */
export function secretLabel(key: string): string {
  if (/password/i.test(key)) return "Password";
  return key.replace(/_/g, " ").toLowerCase().replace(/^./, (c) => c.toUpperCase());
}

/**
 * Task #76. Health values that mean *"not settled yet"*, not *"wrong"*.
 *
 * Docker reports `starting` for the whole of a container's start period, so a
 * restart that has finished successfully still reads `starting` for tens of
 * seconds afterwards. These name the state; the settle poll in `Workloads`
 * keeps re-reading while it holds.
 */
export const settlingHealth = new Set(["starting", "created", "restarting"]);

/** Whether any managed app is still moving, so the list is not a resting view. */
export function inventoryIsSettling(apps: ReadonlyArray<{ running: boolean; health?: string | null }>): boolean {
  return apps.some((app) => app.running && settlingHealth.has((app.health || "").toLowerCase()));
}

/**
 * The Manage board's four app states: Running, Needs attention, Stopped,
 * Starting. A failing health check is "Needs attention" on the board, as a
 * degraded one is; the pill's description says which.
 */
export function managedAppStatus(app: ManagedApp): { label: string; status: "healthy" | "degraded" | "critical" | "neutral" } {
  if (!app.running) return { label: "Stopped", status: "neutral" };
  const health = (app.health || "").toLowerCase();
  if (["unhealthy", "critical", "failed", "error"].includes(health)) return { label: "Needs attention", status: "critical" };
  /*
   * Task #76. `starting` used to land in "Needs attention" beside `degraded`
   * and `warning`. It is neither: a container inside its start period has not
   * failed anything and there is nothing for the reader to do. Same
   * three-state rule as the connection indicator: in-progress is its own
   * answer and may borrow neither the reassuring display nor the alarming one.
   */
  if (settlingHealth.has(health)) return { label: "Starting", status: "neutral" };
  if (["degraded", "warning"].includes(health)) return { label: "Needs attention", status: "degraded" };
  return { label: "Running", status: "healthy" };
}

/** The list pill: the board draws Starting blue (in progress) and a failing app amber. */
export function appStatePill(app: ManagedApp): { label: string; tone: StatusTone; description?: string } {
  const state = managedAppStatus(app);
  if (state.label === "Starting") return { label: state.label, tone: "info", description: "The container is inside its start period." };
  if (state.status === "critical") return { label: state.label, tone: "warning", description: "Its health check is failing." };
  if (state.status === "degraded") return { label: state.label, tone: "warning", description: "Its health check reports it degraded." };
  if (state.status === "healthy") return { label: state.label, tone: "success" };
  return { label: state.label, tone: "neutral" };
}

/** The manager header's pill: "Running · healthy" when Docker reported a health check. */
export function appHeaderPill(app: ManagedApp): { label: string; tone: StatusTone; description?: string } {
  const pill = appStatePill(app);
  const health = (app.health || "").toLowerCase();
  if (pill.label === "Running" && health === "healthy") return { ...pill, label: "Running · healthy" };
  return pill;
}

/** The host port a reader opens or checks: the web port, else the first published one. */
export function primaryPort(app: ManagedApp): string | null {
  if (app.web_port) return String(app.web_port);
  // The guard is for a reader, not the wire: /managed always sends `ports`
  // (workload_inventory.py), but an app without it must not take the page down.
  const host = (app.ports ?? []).map((port) => String(port.host ?? "").split("/")[0]).find(Boolean);
  return host || null;
}

/** "Port 8096 · managed by Vaelor" under an app's name. */
export function appRowDetail(app: ManagedApp): string {
  const port = primaryPort(app);
  const where = port ? `Port ${port}` : app.host_network ? "Host network" : "No published port";
  return `${where} · ${app.managed ? "managed by Vaelor" : "discovered app"}`;
}

/** CPU and memory from the Resource use diagnostic, or null for a reading not taken. */
export interface AppReadings {
  cpu: string | null;
  memory: string | null;
  /** When the diagnostic answered (ms): a one-off reading says its age, not "live" (LESSONS 1). */
  readAt?: number | null;
}

/** "4% · read at 14:05": a reading taken once is shown with the time it was taken. */
export function readingWithAge(value: string | null, readAt: number | null | undefined): string | null {
  if (value === null) return null;
  if (!readAt) return value;
  return `${value} · read at ${new Date(readAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

/**
 * Parse `docker stats` as the diagnostic formats it:
 * "CPU 4.12% | Memory 1.2GiB / 2GiB | Network 1kB / 2kB". Anything else is not
 * a reading, and says "Not read" (LESSONS 8) - never a zero.
 */
export function parseResourceUse(output: string): AppReadings {
  const cpu = output.match(/CPU\s+([\d.]+)%/);
  const memory = output.match(/Memory\s+([\d.]+\s*[KMGT]?i?B)\b/i);
  return {
    cpu: cpu ? `${Math.round(Number(cpu[1]) * 10) / 10}%` : null,
    memory: memory ? memory[1].replace(/\s+/g, " ") : null,
  };
}
