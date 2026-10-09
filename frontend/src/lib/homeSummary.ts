/**
 * What Home's redesigned panels say, derived from reads that already exist
 * (VD-200: "match the mockups, wired to the data that already exists").
 *
 * Every row here comes from a real read, or says plainly that it was not read
 * (LESSONS 1, 8). Nothing is invented: where the mockup showed an
 * illustrative number that no route serves, the panel shows a smaller true
 * thing instead, and the report lists the gap.
 */
import type { AgentStatus } from "../components/agentTypes";
import type { AiChatSetup } from "../components/aiChatTypes";
import type { FleetSummary, NodeCapacity } from "../components/fleetTypes";
import { NOT_ANSWERING, type StatusTone } from "../components/ui/status";
import type { Health } from "../types";
import type { HomeSummary, Read } from "../hooks/useHomeSummary";
import type { HistoryPoint, TelemetryHistory } from "../hooks/useMachineMetrics";
import { assistantModelReadiness } from "./assistantModelReadiness";
import { toEndpoint } from "./endpoints";
import { pooledStatus, ROW_PILL_TONE } from "./deploymentRows";
import {
  controllerLeadsDeployment, isModeBDeployment, LOCAL_BY_MANAGED_PREFIX, type PooledDeployment, SERVING_KIND_CLUSTER, servingModeFromDeployments,
  UNLOAD_CAUSE_IDLE,
} from "./gpuServingMode";
import { humanizeJobType, jobLabels } from "./jobPresentation";
import { llmServerHomeBadge, llmServerRuntimeNote } from "./llmServerStatus";
import { modelDisplayName } from "./modelIdentity";
import { routeHref } from "./navigation";
import { DEPLOYMENTS_MODELS_HREF } from "./clusterSections";

/** A trend as fractions 0..1 of `scale`, grouped into `count` bars; a bar with no reading is null. */
export function trendBars(points: HistoryPoint[] | undefined, scale: number, count = 12): Array<number | null> {
  if (!points?.length) return [];
  const size = Math.max(1, Math.ceil(points.length / count));
  const bars: Array<number | null> = [];
  for (let start = Math.max(0, points.length - size * count); start < points.length; start += size) {
    const values = points.slice(start, start + size).map((point) => point.v).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    bars.push(values.length ? values.reduce((sum, value) => sum + value, 0) / values.length / scale : null);
  }
  return bars.slice(-count);
}

/** The highest measured value in a series, or null when nothing was measured. */
export function seriesPeak(points: HistoryPoint[] | undefined): number | null {
  const values = (points ?? []).map((point) => point.v).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return values.length ? Math.max(...values) : null;
}

/**
 * Two series of one history, bucket by bucket, as the hotter of the two: the
 * Temperature tile's value is the hottest sensor, so its trend is too. A bucket
 * neither series measured stays empty.
 */
export function hottestSeries(first: HistoryPoint[] | undefined, second: HistoryPoint[] | undefined): HistoryPoint[] | undefined {
  if (!first?.length) return second;
  if (!second?.length) return first;
  const other = new Map(second.map((point) => [point.t, point.v]));
  return first.map((point) => {
    const values = [point.v, other.get(point.t)].filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    return { ...point, v: values.length ? Math.max(...values) : null };
  });
}

export function historySeries(history: Read<TelemetryHistory>, key: string): HistoryPoint[] | undefined {
  return history.state === "ok" && history.data.available ? history.data.series?.[key]?.points : undefined;
}

/**
 * A model's name for a one-line summary: the file-name reading of
 * modelDisplayName, then a Hugging Face style id's repository name without
 * its owner ("Qwen/Qwen3-30B-A3B" reads "Qwen3-30B-A3B").
 */
export function shortModelName(identifier: string): string {
  return modelDisplayName(identifier).split("/").filter(Boolean).at(-1) ?? identifier;
}

export interface Pill {
  label: string;
  tone: StatusTone;
  reading?: "unread" | "stale";
}

export interface ServingRow {
  name: string;
  detail: string;
  pill: Pill;
  icon: "memory" | "cpu" | "network";
  accent: boolean;
  href: string;
}

const NOT_READ: Pill = { label: "Not read", tone: "neutral", reading: "unread" };

/** AI Chat's serving, when the cluster that would say how it is served was not read. */
export const SERVING_UNREAD = "Serving not read";

function unread<T>(read: Read<T>, who: string): { detail: string; pill: Pill } {
  if (read.state === "forbidden") return { detail: `Only ${who} can read this.`, pill: NOT_READ };
  if (read.state === "loading") return { detail: "Reading…", pill: { label: "Checking", tone: "neutral", reading: "unread" } };
  return { detail: "Could not be read just now.", pill: NOT_READ };
}

const CONNECTED: Pill = { label: "Connected", tone: "neutral" };

/**
 * What AI Chat sends when no model is pinned and Home cannot name it. The
 * connection decides (`chat_inference._connection` / `answer`): its profile's
 * model, a hosted provider's default, or the first model it lists - so the
 * row names the connection's default, not one of those mechanisms.
 */
export const CONNECTION_DEFAULT_MODEL = "Connection's default model";
export const CLUSTER_DEFAULT_MODEL = "Cluster's default model";
/** Said after a model AI Chat uses because nothing is pinned, not because the owner chose it. */
export const USED_AUTOMATICALLY = "used automatically";

/**
 * The controller-led GPU cluster deployments AI Chat is answered from: the
 * serving ones (Mode B), or, when none serves, the ones an idle unload paused.
 * Only an idle unload keeps the `ai-chat` lease on the cluster's credential; a
 * manual one parks AI Chat on another connection
 * (`gpu_serving_target.UNLOAD_CAUSE_*`), so its deployment is not behind AI
 * Chat and the row describes the connection AI Chat really has (LESSONS 8).
 */
function aiChatClusterRows(pooled: PooledDeployment[], paused: boolean): PooledDeployment[] {
  return paused
    ? pooled.filter((row) => row.engine === "vllm" && row.state === "unloaded"
      && row.unload_cause === UNLOAD_CAUSE_IDLE && controllerLeadsDeployment(row))
    : pooled.filter(isModeBDeployment);
}

/** The cluster deployment's own state as Home's pill: green only where the Cluster page's rule says it serves. */
function clusterPill(deployment: PooledDeployment): Pill {
  const status = pooledStatus(deployment);
  return status.tone === "ok" ? { label: "Serving", tone: "success" } : { label: status.label, tone: ROW_PILL_TONE[status.tone] };
}

/**
 * AI Chat. No route probes whether AI Chat's model is answering, so off the
 * cluster the row never says "Serving": it says which model is chosen and how
 * it is served, and its pill is "Connected" (grey).
 *
 * On the GPU cluster the pill is that deployment's own state, by the Cluster
 * page's rule (`pooledStatus`), read from this poll's `/cluster` answer: green
 * "Serving", a partial or failing reading in its own words, "Paused" when it
 * is unloaded by an idle unload (a manual one moved AI Chat elsewhere). With no
 * model pinned, AI Chat sends the cluster credential's profile model, which the
 * deploy registers as the deployment's `model_id` (`gpu_serving_target`); vLLM
 * serves it under that id (no `--served-model-name`). So when one deployment is
 * behind AI Chat the row names that model the way the LLM Server row does,
 * never "No model chosen": nothing is wrong (LESSONS 8). With several it says
 * only that the cluster's default is used.
 */
function aiChatRow(summary: HomeSummary): ServingRow {
  const base = { name: "AI Chat", icon: "memory" as const, accent: true, href: routeHref("ai-chat") };
  if (summary.aiChat.state !== "ok") return { ...base, ...unread(summary.aiChat, "operators and administrators") };
  const setup: AiChatSetup = summary.aiChat.data;
  const active = setup.active_connection ?? null;
  if (!active) return { ...base, detail: "No model is connected for AI Chat yet.", pill: { label: "Not set up", tone: "neutral" } };
  const pinned = setup.preference?.model || active.selected_model || "";
  // How the model is served is the cluster's to say; an unread cluster says
  // so, never the single-machine fallback (LESSONS 8).
  if (summary.cluster.state !== "ok") {
    const pill = summary.cluster.state === "loading" ? unread(summary.cluster, "").pill : NOT_READ;
    return { ...base, detail: [pinned ? shortModelName(pinned) : CONNECTION_DEFAULT_MODEL, SERVING_UNREAD].join(" · "), pill };
  }
  const pooled = summary.cluster.data.pooled_deployments ?? [];
  const mode = servingModeFromDeployments(pooled);
  // Which paused deployments still hold AI Chat is the idle-cause filter's to
  // say (aiChatClusterRows); none leaves the row on AI Chat's own connection.
  const paused = Boolean(mode.pausedCluster);
  const behind = mode.kind === SERVING_KIND_CLUSTER || paused ? aiChatClusterRows(pooled, paused) : [];
  const cluster = behind[0];
  if (!cluster) {
    const how = active.local_source === LOCAL_BY_MANAGED_PREFIX ? "Managed local model" : active.label;
    // No pin is not no model: AI Chat sends the connection's default, and Home
    // reads nothing that could say it has none (LESSONS 8).
    const detail = [pinned ? shortModelName(pinned) : CONNECTION_DEFAULT_MODEL, how].filter(Boolean).join(" · ");
    return { ...base, detail, pill: CONNECTED };
  }
  const count = cluster.node_ids.length;
  const how = `GPU cluster, ${count} machine${count === 1 ? "" : "s"}`;
  // Named only when one deployment can answer: then the credential's model is its model_id.
  const listed = behind.length === 1 ? cluster.model_id : "";
  const model = pinned || listed;
  const detail = pinned
    ? [shortModelName(pinned), how]
    : listed ? [shortModelName(listed), USED_AUTOMATICALLY, how] : [CLUSTER_DEFAULT_MODEL, how];
  // A pin the cluster does not serve is not answered by it, so its state is not this row's.
  const pill = paused ? { label: "Paused", tone: "neutral" as const } : model && model === cluster.model_id ? clusterPill(cluster) : CONNECTED;
  return { ...base, detail: detail.join(" · "), pill };
}

/** The Assistant: its model and the server's own readiness probe (`/agent/status`). */
function assistantRow(summary: HomeSummary): ServingRow {
  const base = { name: "Assistant", icon: "cpu" as const, accent: false, href: routeHref("assistant") };
  if (summary.agent.state !== "ok") return { ...base, ...unread(summary.agent, "a signed-in user") };
  const status: AgentStatus = summary.agent.data;
  const choice = summary.intelligenceChoice.state === "ok" ? summary.intelligenceChoice.data.intelligence_choice : undefined;
  const readiness = assistantModelReadiness(status, choice);
  const model = status.model || status.capability?.label || "";
  if (!readiness.configured) return { ...base, detail: "Built-in appliance help, no model", pill: { label: "Not set up", tone: "neutral" } };
  if (!readiness.answering) return { ...base, detail: readiness.notAnsweringReason, pill: NOT_ANSWERING };
  return { ...base, detail: [model && shortModelName(model), status.provider].filter(Boolean).join(" · "), pill: { label: "Answering", tone: "success" } };
}

/** The LLM Server: its own badge and the backend's sentence for its door. */
function llmServerRow(summary: HomeSummary): ServingRow {
  const base = { name: "LLM Server", icon: "network" as const, accent: false, href: DEPLOYMENTS_MODELS_HREF };
  if (summary.llmServer.state !== "ok") return { ...base, ...unread(summary.llmServer, "administrators") };
  const endpoint = toEndpoint(summary.llmServer.data);
  const badge = llmServerHomeBadge(endpoint);
  // The pill says the state in a few words; the line keeps the whole of it
  // ("Enabled, no API keys"), then what is behind the door.
  const keys = endpoint.keys.length;
  const detail = endpoint.enabled
    ? [badge.line, endpoint.modelKnown && endpoint.model ? shortModelName(endpoint.model) : "", badge.line ? "" : `${keys} API key${keys === 1 ? "" : "s"}`].filter(Boolean).join(" · ")
    : endpoint.unavailable_reason || "Turned off";
  return { ...base, detail, pill: { label: badge.label, tone: badge.tone } };
}

export function servingRows(summary: HomeSummary): ServingRow[] {
  return [aiChatRow(summary), assistantRow(summary), llmServerRow(summary)];
}

export interface AttentionItem {
  key: string;
  title: string;
  text: string;
  /** Milliseconds since the epoch, or null when the source carries no time. */
  at: number | null;
  /** An approved glyph (VD-200): the LLM Server is a server, a worker's software a package. */
  icon: "alert" | "server" | "package" | "activity" | "shield";
  accent: boolean;
  href: string;
}

/** Jobs carry milliseconds, agent tasks float seconds (the operations ledger's own rule). */
function epochMs(value: unknown): number | null {
  const number = typeof value === "number" ? value : typeof value === "string" ? Date.parse(value) : NaN;
  if (!Number.isFinite(number)) return null;
  return number < 1e11 ? number * 1000 : number;
}

/**
 * Everything the console already treats as needing the owner: operations the
 * server marks `needs_attention` (VD-139), the Assistant when its model is not
 * answering, the LLM Server's door when its badge warns, a worker whose
 * software warns, and the health check's own reasons.
 *
 * `unread` names each source that could not be read (failed, refused for this
 * role, or not answered yet): the count leaves them out, so the panel must not
 * call it complete or say nothing needs the owner (LESSONS 8).
 */
export function attentionItems(summary: HomeSummary, health: Health): { items: AttentionItem[]; total: number | null; unread: string[] } {
  const items: AttentionItem[] = [];
  const unread: string[] = [];
  let total: number | null = 0;
  if (summary.attention.state === "ok") {
    for (const operation of summary.attention.data.items ?? []) {
      if (operation.needs_attention === false) continue;
      items.push({
        key: `op-${operation.operation_id}`,
        title: operation.title || jobLabels[operation.type] || humanizeJobType(operation.type),
        text: operation.recoverable_error?.message || operation.message || "",
        at: epochMs(operation.timestamps?.updated_at ?? operation.timestamps?.created_at),
        icon: "activity",
        accent: false,
        href: operation.owner_route ? routeHref(operation.owner_route) : routeHref("activity"),
      });
    }
    total += summary.attention.data.summary?.attention ?? items.length;
  } else {
    total = null;
  }
  if (summary.agent.state === "ok") {
    const choice = summary.intelligenceChoice.state === "ok" ? summary.intelligenceChoice.data.intelligence_choice : undefined;
    const readiness = assistantModelReadiness(summary.agent.data, choice);
    if (readiness.configured && !readiness.answering) {
      items.push({ key: "assistant", title: `Assistant: ${NOT_ANSWERING.label}`, text: readiness.notAnsweringReason, at: null, icon: "alert", accent: false, href: routeHref("assistant") });
      if (total !== null) total += 1;
    }
  } else {
    unread.push("the Assistant");
  }
  if (summary.llmServer.state !== "ok") unread.push("the LLM Server");
  if (summary.cluster.state !== "ok") unread.push("the cluster's machines");
  if (summary.llmServer.state === "ok") {
    const endpoint = toEndpoint(summary.llmServer.data);
    const badge = llmServerHomeBadge(endpoint);
    if (badge.tone === "warning" || badge.tone === "danger") {
      items.push({ key: "llm-server", title: `LLM Server: ${badge.label}`, text: llmServerRuntimeNote(endpoint), at: null, icon: "server", accent: false, href: DEPLOYMENTS_MODELS_HREF });
      if (total !== null) total += 1;
    }
  }
  if (summary.cluster.state === "ok") {
    for (const node of summary.cluster.data.enrolled_nodes ?? []) {
      const software = node.worker_software;
      if (!software || (software.tone !== "warning" && software.tone !== "danger")) continue;
      items.push({ key: `worker-${node.id}`, title: `${node.name}: ${software.label}`, text: software.sentence, at: epochMs(software.checked_at), icon: "package", accent: true, href: "?cluster=fleet#/fleet" });
      if (total !== null) total += 1;
    }
  }
  if (health.status !== "healthy" && health.status !== "offline") {
    for (const [index, reason] of health.reasons.entries()) {
      items.push({ key: `health-${index}`, title: "Health check", text: reason, at: epochMs(health.sampled_at), icon: "alert", accent: false, href: routeHref("system") });
      if (total !== null) total += 1;
    }
  }
  return { items, total, unread };
}

export interface MachineRow {
  key: string;
  name: string;
  hardware: string;
  role: string;
  cpu: string;
  gpuFraction: number | null;
  gpu: string;
  temperature: string;
  software: Pill;
}

/** A worker's capacity row, by the ledger's node id or its fleet record's ids. */
export function capacityFor(nodes: NodeCapacity[], node: FleetSummary["enrolled_nodes"][number]): NodeCapacity | undefined {
  return nodes.find((row) => row.node_id === node.id || row.node_id === node.labels?.swarm_node_id || row.name === node.name);
}
