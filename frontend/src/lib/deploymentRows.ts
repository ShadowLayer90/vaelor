import type { AppGroup, FleetSummary } from "../components/fleetTypes";
import type { AgentDeployment } from "./clusterAgents";
import type { DeploymentFilter } from "../components/ClusterDeployments";
import type { RowStatus } from "../components/clusterAppStatus";
import { NOT_ANSWERING, type StatusTone } from "../components/ui/status";

/**
 * Which deployments the cluster lists, and how many - the ONE derivation the
 * Deployments tab draws its rows from and the Fleet hero counts (ACC-092).
 *
 * The two used to count differently: Fleet added the raw service list to the
 * pooled models, so a researched app's member services each counted once and
 * agents not at all, while Deployments folded each app group into one row and
 * listed agents. Both now call this.
 *
 * - A researched app's member services are folded into its one grouped row, so
 *   they are dropped from the loose per-service list (matched by the exact Swarm
 *   name the group's breakdown carries - never a name parse). A catalog service
 *   is in no group and stays its own row.
 * - Every pooled model, app group and agent is one row, whatever its state: a
 *   failed or unloaded deployment is still listed (with its state) until it is
 *   removed.
 */

type Service = { name?: unknown };
type Pooled = NonNullable<FleetSummary["pooled_deployments"]>[number];

export interface DeploymentSources<S extends Service> {
  services: S[];
  pooled: Pooled[];
  appGroups: AppGroup[];
  agents: AgentDeployment[];
}

export interface ListedDeployments<S extends Service> extends DeploymentSources<S> {
  /** How many rows are listed under the filter. */
  total: number;
}

/**
 * A split model's two fence notes, as the backend wrote them (ACC-187,
 * `cluster_manager.split_fence_notes`): `cleanup` when an unload could not
 * clear a machine's firewall and slice, `fence` when the split predates the
 * firewall check at every start. Each is the backend's sentence verbatim, or
 * "" when the row carries none - the browser decides only whether to show it,
 * never what it says.
 */
export interface SplitFenceNotes {
  cleanup: string;
  fence: string;
}

/**
 * The word and tone for each `state` the backend writes on a pooled record:
 * `deploying`, `healthy` and `failed` (`gpu_pool_operations.deploy`,
 * `pooled_operations.deploy`, and `gpu_cluster_mode_watch._mark_left`, which
 * turns a record the mode switch abandoned into `failed`).
 *
 * This used to be a boolean: healthy read "Serving" and everything else read
 * "Deploying" — so a deploy that had already died and been rolled back sat on
 * the row as one still in progress, and the sentence the backend wrote to say
 * what to do next was never shown. An operator would wait forever. A state this
 * map has no word for is reported as unknown, never as any specific state.
 */
const POOLED_STATUS: Partial<Record<string, RowStatus>> = {
  healthy: { label: "Serving", tone: "ok" },
  deploying: { label: "Deploying", tone: "warn" },
  failed: { label: "Failed", tone: "danger" },
  // G3a: a deployment whose serving units were stopped to reclaim the GPU,
  // kept whole so a Load re-serves it warm. A resting state, not a fault.
  unloaded: { label: "Unloaded", tone: "neutral" },
};
const UNKNOWN_POOLED_STATUS: RowStatus = { label: "Unknown state", tone: "neutral" };

/** A row status's tone as the boards' outline pill draws it. */
export const ROW_PILL_TONE: Record<RowStatus["tone"], StatusTone> = {
  ok: "success",
  warn: "warning",
  danger: "danger",
  neutral: "neutral",
};

export type Replica = NonNullable<NonNullable<Pooled["units"]>["replicas"]>[number];

/**
 * The replica list of a healthy `replicated` record (VD-129), or null for any
 * other row. The backend writes the list on every replicated row after the
 * DEPLOYING one, so a healthy replicated row without it is a shape the
 * contract says cannot exist; it falls through to the plain state word rather
 * than to a count invented from the node list.
 */
export function servingReplicas(deployment: Pooled): Replica[] | null {
  const units = deployment.units;
  if (deployment.state !== "healthy" || units?.mode !== "replicated") return null;
  return Array.isArray(units.replicas) && units.replicas.length ? units.replicas : null;
}

/**
 * A pooled row's status: the Cluster page's Deployments table, its Fleet
 * chips, and Home's AI Chat row when AI Chat runs on the cluster all read
 * this, so one deployment has one state word.
 *
 * A replicated row never reads "Serving" alone: it reads "k of N serving"
 * from `units.replicas[].alive`, which the mode reconcile's healthy pass
 * writes from each replica's `/health` probe (VD-129). All alive is ok; some
 * is a warning — the endpoint answers, at fewer machines' worth of concurrency
 * than was deployed; none is danger — the record stays healthy for the four
 * passes the reconcile allows before it tears the cluster down, and during
 * them nothing behind the balancer answers.
 */
export function pooledStatus(deployment: Pooled): RowStatus {
  // B1: a healthy record serving on fewer machines than it was deployed to
  // (a forced removal took one) says so, on its row and on every Fleet chip
  // that reads this, with the reason beneath the row (`DegradedNote`). The
  // backend's `degraded_reason` is the one source; `state` is not re-read.
  if (deployment.state === "healthy" && (deployment.degraded_reason ?? "").trim()) {
    return { label: "Degraded", tone: "warn" };
  }
  // W4-D6: a split that failed its last health check is not shown as plainly
  // serving for the two minutes the watch waits; the sentence is beneath.
  if (deployment.state === "healthy" && (deployment.checking_note ?? "").trim()) {
    // Read, and broken: red, as everywhere (ui/status NOT_ANSWERING; this table's own tone words).
    return { label: NOT_ANSWERING.label, tone: "danger" };
  }
  const replicas = servingReplicas(deployment);
  if (replicas) {
    const alive = replicas.filter((replica) => replica.alive === true).length;
    const total = replicas.length;
    return {
      label: `${alive} of ${total} serving`,
      tone: alive === total ? "ok" : alive > 0 ? "warn" : "danger",
    };
  }
  return POOLED_STATUS[deployment.state] ?? UNKNOWN_POOLED_STATUS;
}

export function splitFenceNotes(deployment: Pooled): SplitFenceNotes {
  return {
    cleanup: (deployment.cleanup_note ?? "").trim(),
    fence: (deployment.fence_note ?? "").trim(),
  };
}

/**
 * What the post-upgrade refresh could not do on this row (W4-D1,
 * `gpu_render_ledger.refresh_note`): the backend's sentence verbatim, or ""
 * when the row carries none.
 */
export function refreshNote(deployment: Pooled): string {
  return (deployment.refresh_note ?? "").trim();
}

export function listedDeployments<S extends Service>(
  sources: DeploymentSources<S>,
  filter: DeploymentFilter = "all",
): ListedDeployments<S> {
  const showModels = filter === "all" || filter === "models";
  const showApps = filter === "all" || filter === "apps";
  const showAgents = filter === "all" || filter === "agents";
  const groupedMemberNames = new Set(
    sources.appGroups.flatMap((group) => group.services.map((member) => member.name)),
  );
  const services = (showApps ? sources.services : []).filter(
    (service) => !groupedMemberNames.has(String(service.name ?? "")),
  );
  const pooled = showModels ? sources.pooled : [];
  const appGroups = showApps ? sources.appGroups : [];
  const agents = showAgents ? sources.agents : [];
  return {
    services,
    pooled,
    appGroups,
    agents,
    total: services.length + pooled.length + appGroups.length + agents.length,
  };
}
