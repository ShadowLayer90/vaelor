import { CONTROLLER_PLACEMENT_ID, type FleetSummary } from "../components/fleetTypes";

/**
 * Which engine is serving this appliance's GPU model right now — Mode A or
 * Mode B — and every sentence the console says about the switch between them.
 *
 * VD-127 gave the backend one gate for this question
 * (`vaelor.gpu_serving_target.resolve_gpu_serving_target`, whose `kind` is
 * `none` / `managed-local` / `cluster`) precisely because three call sites had
 * been answering it three ways. The console must not become a fourth: the mode
 * is derived HERE, once, and every surface that shows it — the AI Chat
 * connection card, the LLM Server panel, the Cluster → Deployments header, the
 * removal confirmation — reads this module rather than re-deriving from
 * whatever data happens to be in its own props.
 *
 * **Where the mode comes from, and why.** The preferred source is the backend's
 * own kind, published on the LLM Server surface as `target_kind`; when it is
 * present it wins outright, because it is the value the executor's failure-watch
 * and the auth proxy act on. Today's control plane does not yet publish it
 * (`api_llm_server_routes._surface()` reports `available` and
 * `unavailable_reason` but not the kind), so the fallback is the one derivation
 * VD-127 makes safe to draw: **a healthy vLLM deployment whose participants
 * include the controller is Mode B**, because that is exactly the condition
 * under which `gpu_pool_operations.deploy` enters the mode switch — the
 * controller leads, the API binds loopback, and AI Chat plus the LLM Server were
 * moved onto it.
 *
 * **A worker-led cluster is read too, but is not Mode B.** Its deploy registers
 * a credential for `cluster-inference` only and deliberately never claims
 * `ai-chat` (`gpu_pool_operations._register_credential`), so AI Chat and the
 * LLM Server stay where they were; and its API binds the real NIC, so the
 * backend records `units.lan_exposed` on it (VD-127 D2/D7). The derivation
 * carries that record — the deployment, the model, who leads, and the exposure
 * — so the console can LABEL the open endpoint, while `kind` says `cluster`
 * only when the switch was actually entered. Should the backend ever publish
 * `target_kind: "cluster"` over a worker-led record, `resolveServingMode`
 * keeps the exposure, and the AI Chat and LLM Server surfaces say the cluster
 * they are on is open.
 */

/** The serving kinds, spelled as `vaelor.gpu_serving_target` spells them. */
/**
 * How the AI-Chat connection says it is served on this appliance, as
 * `vaelor.chat_connections.connection_locality` spells it. `vaelor-managed` is
 * the one value that proves Mode A: a credential Vaelor minted for its own
 * deploy. A loopback address alone does not — somebody's own server on
 * 127.0.0.1 is loopback and is not this appliance's managed model.
 */
export const LOCAL_BY_MANAGED_PREFIX = "vaelor-managed";

export const SERVING_KIND_NONE = "none";
export const SERVING_KIND_MANAGED_LOCAL = "managed-local";
export const SERVING_KIND_CLUSTER = "cluster";

export type GpuServingKind =
  | typeof SERVING_KIND_NONE
  | typeof SERVING_KIND_MANAGED_LOCAL
  | typeof SERVING_KIND_CLUSTER;

/** What clustering is being asked FOR (`vaelor.cluster_gpu_sizing`'s intents). */
export const INTENT_CAPACITY = "capacity";
export const INTENT_THROUGHPUT = "throughput";
export type GpuClusterIntent = typeof INTENT_CAPACITY | typeof INTENT_THROUGHPUT;

/**
 * The GPU memory fraction vLLM launches with when the operator has not typed
 * one, per intent — the same defaults the backend applies to a request that
 * omits the field (VD-129; VD-127 cleanup item 9c: 0.90 left no CUDA-graph
 * capture headroom on a 30 GiB unified aperture, and the replicated launch
 * defaults lower). Spelled as the form's text so the field shows exactly what
 * the fit and the deploy will send; `Number()` of either is the wire value.
 */
export const GPU_MEMORY_UTILIZATION_DEFAULTS: Record<GpuClusterIntent, string> = {
  [INTENT_CAPACITY]: "0.90",
  [INTENT_THROUGHPUT]: "0.80",
};

export interface GpuServingMode {
  kind: GpuServingKind;
  /** The healthy GPU cluster deployment's name, empty when none is running. */
  deployment: string;
  /** That cluster's model repository, empty when none is running. */
  model: string;
  /**
   * Whether this controller is one of that cluster's participants. When it is,
   * it leads (VD-127 D2), the API binds loopback, and the deploy entered the
   * mode switch — which is what makes `kind` read `cluster` from the fleet.
   */
  controllerLeads: boolean;
  /**
   * The backend's own record that the cluster's API binds the real NIC, so it
   * answers on the LAN with no key until a worker-side proxy exists (VD-127
   * D2/D7). Written as `units.lan_exposed` on a worker-led deployment; the
   * console reads it rather than re-deriving "did the controller lead".
   */
  lanExposed: boolean;
  /**
   * How many replicas a `replicated` cluster runs (VD-129), read from the
   * record's `units.replicas` list — the served-by line says "(N replicas)"
   * from it. Zero for a sharded (`distributed`) cluster, for a record that
   * carries no mode, and when nothing is clustered.
   */
  replicas: number;
  /**
   * Whether the mode could be read at all. A surface must render nothing rather
   * than claim "managed local model" on a question it could not ask — an
   * unreadable answer is not evidence of Mode A.
   */
  known: boolean;
  /**
   * A controller-led cluster model that is paused (`unloaded`) while nothing
   * else serves. `wakesOnRequest` is the server's reading of who unloaded it:
   * true for a scale-to-zero (idle) unload, false for a manual one, null when
   * the server could not tell. Absent when a cluster is serving or none exists.
   */
  pausedCluster?: PausedCluster;
}

export interface PausedCluster {
  deployment: string;
  model: string;
  wakesOnRequest: boolean | null;
}

/** The server's unload causes (`gpu_serving_target.UNLOAD_CAUSE_*`). */
export const UNLOAD_CAUSE_IDLE = "unloaded-idle";
export const UNLOAD_CAUSE_MANUAL = "unloaded-manual";

export const UNKNOWN_SERVING_MODE: GpuServingMode = {
  kind: SERVING_KIND_NONE,
  deployment: "",
  model: "",
  controllerLeads: false,
  lanExposed: false,
  replicas: 0,
  known: false,
};

export type PooledDeployment = NonNullable<FleetSummary["pooled_deployments"]>[number];

/**
 * A healthy vLLM deployment, whoever leads it: the fleet's GPU cluster. The
 * `engine` read here is the summary's projection of the record's
 * `units.engine` (`cluster_manager.projected_pooled_deployments`), the same
 * discriminator the row's Remove already keys on. Keyed on engine and state
 * only, never on `units.mode`: a replicated cluster (VD-129) enters the mode
 * switch exactly as a sharded one does, so it is a GPU cluster on the same
 * two facts.
 */
export function isGpuClusterDeployment(deployment: PooledDeployment): boolean {
  return deployment.engine === "vllm" && deployment.state === "healthy";
}

/**
 * How many replicas a record runs: the length of its `units.replicas` list on
 * a `replicated` record (VD-129), else zero. The list is read only under that
 * mode, so a sharded record that somehow carried one would still count none.
 */
export function replicaCount(deployment: PooledDeployment): number {
  const units = deployment.units;
  if (units?.mode !== "replicated") return 0;
  return Array.isArray(units.replicas) ? units.replicas.length : 0;
}

/**
 * The one spelling of what a cluster-mode surface is served by: "GPU cluster",
 * the deployment's name when known, and "(N replicas)" for a replicated
 * cluster (VD-129). The LLM Server panel's Served-by line and the AI Chat
 * card's mode line both print this, so the two cannot name the same cluster
 * two ways.
 */
export function clusterServedBy(mode: GpuServingMode): string {
  if (!mode.deployment) return "GPU cluster";
  const replicas = mode.replicas > 0 ? ` (${mode.replicas} replicas)` : "";
  return `GPU cluster · ${mode.deployment}${replicas}`;
}

/** Whether this controller takes part in the deployment (and so leads it). */
export function controllerLeadsDeployment(deployment: PooledDeployment): boolean {
  return (deployment.node_ids ?? []).includes(CONTROLLER_PLACEMENT_ID);
}

/**
 * A running GPU cluster whose API is open on the LAN without a key, on the
 * backend's own record of it (VD-127 D2). Per deployment, because it is a
 * property of that endpoint and is labelled on that row.
 */
export function isLanExposedDeployment(deployment: PooledDeployment): boolean {
  return isGpuClusterDeployment(deployment) && deployment.units?.lan_exposed === true;
}

/** Whether this row is the healthy, controller-led vLLM cluster (see above). */
export function isModeBDeployment(deployment: PooledDeployment): boolean {
  return isGpuClusterDeployment(deployment) && controllerLeadsDeployment(deployment);
}

/**
 * The serving mode implied by the fleet summary's deployments.
 *
 * `pooled` being an empty list is a real answer (nothing is clustered), so the
 * result is `known`; `undefined` means the summary was never read, and the
 * result is not.
 *
 * **Not clustering is reported as `none`, never as `managed-local`.** The
 * deployments say whether Mode B is on; they say nothing about whether AI Chat
 * points at a model this appliance serves or at a hosted provider, and
 * answering the second question from the first is how a console comes to print
 * "prompts never leave the machine" over an OpenAI endpoint. Only the backend's
 * own `target_kind` — or a surface's own evidence, such as the AI-Chat
 * connection's `local_source` — may claim `managed-local`.
 *
 * **A worker-led cluster is carried but reported as `none`**, for the reason in
 * the module comment: it is running and it is LAN-open, but it did not take AI
 * Chat or the LLM Server over. The Mode B record wins when both shapes exist.
 */
export function servingModeFromDeployments(
  pooled: PooledDeployment[] | undefined,
): GpuServingMode {
  if (!pooled) return UNKNOWN_SERVING_MODE;
  const cluster = pooled.find(isModeBDeployment) ?? pooled.find(isGpuClusterDeployment);
  if (!cluster) {
    const paused = pooled.find(
      (row) => row.engine === "vllm" && row.state === "unloaded" && controllerLeadsDeployment(row),
    );
    if (!paused) return { ...UNKNOWN_SERVING_MODE, known: true };
    const cause = paused.unload_cause ?? "";
    return {
      ...UNKNOWN_SERVING_MODE,
      known: true,
      pausedCluster: {
        deployment: paused.name,
        model: paused.model_id,
        wakesOnRequest: cause === UNLOAD_CAUSE_IDLE ? true : cause === UNLOAD_CAUSE_MANUAL ? false : null,
      },
    };
  }
  const controllerLeads = controllerLeadsDeployment(cluster);
  return {
    kind: controllerLeads ? SERVING_KIND_CLUSTER : SERVING_KIND_NONE,
    deployment: cluster.name,
    model: cluster.model_id,
    controllerLeads,
    lanExposed: cluster.units?.lan_exposed === true,
    replicas: replicaCount(cluster),
    known: true,
  };
}

/**
 * The backend's own kind when it publishes one, else the derivation above.
 * `targetKind` is whatever `GET /llm-server` reported; an unknown token is
 * ignored rather than rendered, so a newer control plane cannot make this
 * console print a word it has no copy for.
 */
export function resolveServingMode(
  targetKind: string | undefined,
  derived: GpuServingMode = UNKNOWN_SERVING_MODE,
): GpuServingMode {
  if (targetKind === SERVING_KIND_CLUSTER) {
    return { ...derived, kind: SERVING_KIND_CLUSTER, known: true };
  }
  if (targetKind === SERVING_KIND_MANAGED_LOCAL) {
    return {
      ...UNKNOWN_SERVING_MODE, kind: SERVING_KIND_MANAGED_LOCAL, known: true,
    };
  }
  if (targetKind === SERVING_KIND_NONE) {
    return { ...UNKNOWN_SERVING_MODE, known: true };
  }
  return derived;
}

/**
 * The Deployments header badge, shown only in Mode B, in two parts: the pill's
 * short state and the sentence that says what it means. Split because the
 * status pill is an uppercase, letter-spaced primitive — a whole sentence set
 * that way is a wall of capitals on a phone — and because the severity belongs
 * to the pill while the explanation belongs to body text beside it.
 */
export const MODE_B_BADGE_LABEL = "GPU clustering active";
export const MODE_B_BADGE_DETAIL =
  "AI Chat and the LLM Server run on the cluster.";

// The mode-switch notice the serve form shows before Serve is the backend's
// (`cluster_fit_guards.MODE_SWITCH_NOTICE`, W4-D4), sent on the fit only when
// this deploy would take the switch.

/**
 * The label on a LAN-open deployment's row, and the clause the AI Chat and LLM
 * Server surfaces attach to "the cluster" when the one they are on is open
 * (VD-127 D2). One vocabulary, so the row and the panels name the same fact
 * with the same words.
 */
export const LAN_EXPOSED_PILL_LABEL = "Open on the LAN without a key";
/**
 * Why a row carries that pill, and until when (ACC-162). Only a worker-led
 * split deployed before its lead was put behind a keyed gate is still open;
 * a Load re-serves it on loopback behind the gate.
 */
export const LAN_EXPOSED_PILL_DESCRIPTION =
  "Deployed by an earlier version of Vaelor, so its model API answers on your "
  + "LAN without a key until it is Loaded again.";
export const CLUSTER_LAN_OPEN_CLAUSE = "which is open on your LAN without an API key";

/**
 * A cluster led by a worker (the controller is not one of the selected
 * machines). Its model API binds loopback on that machine and is reached only
 * through a keyed gate there (ACC-162), and its Ray cluster admits only the
 * deployment's own token (ACC-163) — so this is information, not a warning.
 * Shown for the capacity intent only: under the throughput intent the engine
 * refuses a selection without this controller before anything is deployed
 * (VD-129), and the console renders that refusal rather than this notice.
 */
export const WORKER_LED_LAN_NOTICE =
  "No controller is selected, so a worker leads this cluster. Its model API is "
  + "reached only through a keyed gate on that machine, and AI Chat and the LLM "
  + "Server stay where they are.";

/**
 * What removing a Mode B deployment gives back (VD-127's restore contract).
 * AI Chat is given back only while it is still on the cluster: a connection
 * the owner chose for it while clustered stays (VD-210 item 4).
 */
export const MODE_B_REMOVAL_NOTE =
  "GPU clustering is active, so removing this also ends it: this controller's "
  + "LLM Server, and AI Chat if it is still using the cluster, return to "
  + "single-node serving if a local model is available, otherwise they stop "
  + "until you serve a model again. A connection you chose for AI Chat stays.";
