import type { WorkerSoftware } from "./WorkerSoftwareRow";
import { formatQuantity } from "../lib/format";
import type { GpuClusterIntent } from "../lib/gpuServingMode";

/**
 * How the head controller names itself as a placement target, matching
 * `vaelor.cluster_placement.CONTROLLER_PLACEMENT_ID`. The controller has no
 * enrolled-node record — it is the Swarm manager, not one of its members — so
 * every placement it takes part in (the single-node compute target, a GPU
 * serve) refers to it by this id, and the capacity ledger keys its row the
 * same way.
 */
export const CONTROLLER_PLACEMENT_ID = "controller";

export interface FleetNode {
  id: string;
  name: string;
  host: string;
  port: number;
  role: string;
  state: string;
  /**
   * The worker's cluster runtime as last read: "ready", "down", "missing",
   * "enrolled" (not joined), or "unreadable" (a cluster id is recorded but the
   * cluster state could not be read).
   */
  runtime_state?: string;
  runtime?: {
    status?: string;
    availability?: string;
    manager_status?: string;
    engine?: string;
  } | null;
  host_key_fingerprint: string;
  labels?: Record<string, string>;
  /**
   * E2b. Whether this worker has a telemetry ingest key provisioned — true
   * between install and remove of the telemetry agent. It is the install-state
   * the machine card reads to offer Install vs Remove telemetry, distinct from
   * whether the agent is currently reporting (the metrics panel shows that from
   * sample freshness). Only the boolean crosses the wire; the key hash does not.
   */
  telemetry_provisioned?: boolean;
  /**
   * The backend's sentence when the telemetry agent could not be brought up to
   * date on this machine, "" normally. Absent from an older server.
   */
  telemetry_repair_note?: string;
  /** VD-194 P1: this worker's software against its profile. Absent from an older server. */
  worker_software?: WorkerSoftware;
  /**
   * Published for every enrolled node, matching or not. A node admitted
   * before VD-031 existed is still in the store, and drawing it as ordinary
   * would hide the one fact that stops its workloads from ever starting.
   */
  architecture?: {
    class: string;
    label: string;
    matches_controller: boolean;
    reason: string;
    /**
     * VD-033. `matches_controller` cannot separate "this is the other
     * architecture" from "this appliance could not read one of the two
     * values", and only the first authorises removal. `removable` is the one
     * the destructive action reads; anything else is reported and left alone.
     */
    disposition: string;
    removable: boolean;
    removal_reason: string;
  };
  inventory: {
    architecture?: string;
    cpu_count?: number;
    memory_bytes?: number;
    os?: string;
    docker?: boolean;
    /** False after a failed Recheck; the rest is then the last good snapshot. */
    reachable?: boolean;
    unreachable_reason?: string;
    /** Epoch seconds of the last Recheck attempt, success or failure. */
    checked_at?: number;
    /**
     * VD-125. Which of this machine's network connections carries the address
     * the cluster reaches it on, recorded when the machine joined. GPU
     * clustering binds that connection by name, so a machine that could not
     * name one states why here rather than hanging a later deployment.
     */
    cluster_interface?: ClusterInterface;
  };
}

/**
 * One researched multi-service application, folded from its member services into
 * a single Deployments row (D4d). The backend keys the group on the
 * ``vaelor.app-group`` label (`cluster_service_state.aggregate_app_groups`) and
 * derives the app-level `state` as the WORST member state
 * (unknown > unavailable > failed > rolled-back > updating > deploying >
 * healthy), with a one-line `reason` naming the worst service. A member whose
 * inspect is unreadable makes the whole app `unknown` — an app NEVER reads
 * healthy while a member is down or unreadable. `services` is the per-service
 * breakdown the Manage panel renders.
 */
export interface AppGroup {
  app_group: string;
  /** The honest worst-of state, one of the same words a per-service app row uses. */
  state: string;
  /** One line naming the worst service; empty for a healthy app. */
  reason: string;
  services: AppGroupService[];
}

/** One member service of a grouped researched app, as the backend breaks it down. */
export interface AppGroupService {
  /** The Swarm service name (`vaelor-app-<app>` or `vaelor-app-<app>-<svc>`). */
  name: string;
  /** The manifest service key (the compose service name), e.g. `web` or `db`. */
  service: string;
  /** The member's honest per-service state, the same contract as the app row. */
  state: string;
  /** The member's one-line reason; empty when the member is healthy. */
  reason: string;
  /** The live "running/desired" replica string, e.g. "1/1"; absent when unread. */
  replicas?: string | null;
}

export interface FleetSummary {
  controller: {
    initialized: boolean;
    driver: string;
    role: string;
    advertise_address?: string;
    candidate_address?: string;
    /**
     * VD-125. The controller's OWN cluster link, read live off this machine by
     * `cluster_manager.summary` in the same `{name, address, speed_mbps,
     * medium}` shape a worker's is captured in. Absent when the controller is
     * not yet initialised (no address to read against), so the fleet card falls
     * back to an honest "unknown" rather than a guess.
     */
    inventory?: { cluster_interface?: ClusterInterface };
    placement?: FleetNode & { eligible: boolean; reason?: string };
  };
  runtime: {
    available: boolean;
    /** Vaelor set this node up as controller - see `ClusterRuntimeFlags`. */
    initialized: boolean;
    control_available: boolean;
    /** Docker's own Swarm fact, which `initialized` no longer is (LESSONS 6). */
    swarm_active?: boolean;
    /** Task #75: the engine fact, separate from the Swarm fact. */
    engine?: "ready" | "absent" | "unreadable";
    engine_reason?: string;
    nodes?: Array<Record<string, string>>;
    services?: Array<Record<string, string>>;
    /**
     * D4d. The researched multi-service apps, folded by their
     * ``vaelor.app-group`` label into ONE row each by
     * `cluster_service_state.aggregate_app_groups` (never a name parse). This is
     * an ADDITIVE view alongside — not in place of — the per-service `services`
     * rows above: a catalog service carries no app-group label and stays its own
     * row, and a researched single-service app is simply a group of one. Absent
     * when no researched app is running. See `AppGroup`.
     */
    app_groups?: AppGroup[];
  };
  /**
   * `cluster_manager.enrollment_readiness`, served since the sudo-password
   * form was first gated and never read here — which is how "Add worker"
   * stayed an enabled primary button on a screen stating that workers could
   * not be enrolled.
   */
  enrollment?: {
    available: boolean;
    reason: string;
    requires_sudo_password?: boolean;
    address_hint?: string;
  };
  enrolled_nodes: FleetNode[];
  pooled_deployments?: Array<{
    name: string;
    state: string;
    model_id: string;
    node_ids: string[];
    endpoint: string;
    /**
     * B3a projects the serving engine onto every pooled-table row so the console
     * can pick the right removal path: a `vllm` row is a GPU (vLLM) deployment
     * removed through `cluster.gpu.remove`; every other row is a CPU
     * `distributed-llama` deployment removed through `cluster.pooled.remove`.
     */
    engine?: "vllm" | "distributed-llama";
    /**
     * G3b scale-to-zero: the idle window in seconds after which the mode
     * reconcile enqueues an unload to reclaim the GPU (0 or absent = off),
     * projected from the units blob by `projected_pooled_deployments`.
     */
    idle_timeout?: number;
    /**
     * Whether the model thinks before answering: the setting the record carries,
     * or null for a deployment from before the setting existed - it still
     * thinks until its next Load serves it with the new default (off).
     */
    thinking?: boolean | null;
    /** Whether the setting reaches this model at all (`vaelor/model_thinking`). */
    thinking_switch?: boolean;
    /** Which vLLM a cluster deployment runs, in a few words ("" for the CPU engine). */
    vllm_version?: string;
    /**
     * On an `unloaded` vLLM row only: who unloaded it, which decides whether a
     * request wakes it - `unloaded-idle` (scale-to-zero; the next request
     * loads it), `unloaded-manual` (only Load in Deployments does), or `""`
     * when the server could not tell (`gpu_serving_target.deployment_unload_cause`).
     */
    unload_cause?: string;
    /**
     * The stored serving units. Only the facts the console reads are named:
     * `engine` (the same discriminator the row's projected `engine` carries) and
     * `lan_exposed`, which `gpu_pool_operations.deploy` sets on a worker-led
     * cluster whose model API binds the real NIC rather than loopback (VD-127).
     */
    units?: {
      engine?: string;
      lan_exposed?: boolean;
      /**
       * VD-129. How the units serve the model: `distributed` shards one copy
       * across the nodes; `replicated` runs one whole copy per node behind a
       * balancer on this controller. Written on the DEPLOYING row and every row
       * after it; a record from before the word existed carries none, and the
       * console keeps its older wording for that row rather than guessing.
       */
      mode?: "distributed" | "replicated";
      /**
       * On a `replicated` record: one entry per replica, with `alive` and
       * `reachable` written by the mode reconcile's healthy pass from each
       * replica's `/health` probe. The row's "k of N serving" is k = the
       * alive count, N = the list's length — both read from here and nowhere
       * else. `reachable` is false when nothing at the address answered at
       * all (a node that left the cluster, was unplugged, or is rebooting),
       * as against alive-false-reachable-true for a replica that answered
       * its check unhealthy; a row written before the field existed carries
       * none and reads as reachable.
       */
      replicas?: Array<{
        node_id: string;
        address: string;
        port: number;
        unit: string;
        /**
         * On a WORKER's entry: the gate unit beside its replica (VD-129, the
         * amendment) — the nginx door that is the only thing on that machine
         * listening on the LAN, and what the probe goes through. Absent on
         * the controller's entry, whose replica sits behind the balancer.
         */
        gate?: string;
        alive: boolean;
        reachable?: boolean;
      }>;
      /** The balancer's port on a `replicated` record. */
      port?: number;
      /** The port every replica listens on (`port + 1`) on a `replicated` record. */
      replica_port?: number;
      /**
       * On a `failed` record: the backend's own sentence for why the deploy
       * failed and what to do (`gpu_pool_operations.deploy`'s rollback,
       * `gpu_cluster_mode_watch._mark_left`), and every unit that rollback
       * could NOT stop — node and unit, with the error
       * (`gpu_pool_units.unstopped`) — so the row states both rather than a
       * log nobody reads until a GPU is found full.
       */
      failure?: string;
      unstopped?: Array<{ node_id?: string; node?: string; unit?: string; error?: string }>;
      /**
       * B1: machines this deployment lost to a forced removal (the worker was
       * unreachable and the owner removed it anyway), each with when and why.
       */
      lost_nodes?: Array<{ node_id: string; name?: string; at?: number; reason?: string }>;
    };
    /**
     * B1: the backend's plain sentence when a `healthy` record is serving on
     * fewer machines than it was deployed to (a forced removal took one away),
     * projected at the top level by the `/cluster` summary. Empty or absent
     * when it is whole. `state` stays `healthy` for the mode switch.
     */
    degraded_reason?: string;
    /**
     * ACC-187 (`cluster_manager.split_fence_notes`), "" or absent when none:
     * `cleanup_note` - on an `unloaded` split, an unload could not clear a
     * machine's firewall and slice; `fence_note` - on a serving or unloaded
     * split deployed before the firewall check at every start, the step that
     * applies it. Both are the backend's sentences, shown verbatim.
     */
    cleanup_note?: string;
    fence_note?: string;
    /**
     * W4-D1 (`gpu_render_ledger.refresh_note`), "" or absent when none: what
     * the post-upgrade refresh could not do on this row (a machine it could
     * not reach, a Load it could not finish), worded by the backend for the
     * row's state with the step that applies the release. Shown verbatim.
     */
    refresh_note?: string;
    /**
     * W4-D7 (`gpu_pool_recover.load_offered`): whether Load serves this row -
     * an unloaded one, or a failed one that served before. Absent from an
     * older backend, where only an unloaded row loads.
     */
    load_offered?: boolean;
    /**
     * W4-D6 (`gpu_pool_serving_health.split_checking_note`): on a serving
     * split that failed its last health check, what it found and how long the
     * watch waits before failing it. "" or absent otherwise.
     */
    checking_note?: string;
  }>;
  /**
   * A cluster runs one processor architecture (VD-031). `worker_os` used to
   * stand here — an OS whitelist doing the job of an architecture constraint,
   * and one this dashboard never rendered.
   */
  architecture?: {
    controller: string;
    label: string;
    homogeneous: boolean;
    note: string;
    conflicts: Array<{
      node_id: string;
      name: string;
      host: string;
      architecture: string;
      label: string;
      reason: string;
      disposition: string;
      removable: boolean;
      removal_reason: string;
    }>;
    /**
     * Nodes this controller drained and removed. A removed node leaves no
     * entry in `enrolled_nodes`, and a view that simply stopped showing it
     * would be silent about the most destructive thing Vaelor does to its own
     * state (VD-033).
     */
    removals?: Array<{
      node_id: string;
      name: string;
      host: string;
      architecture: string;
      label: string;
      controller_architecture: string;
      reason: string;
      drained: boolean;
      removed: boolean;
      forced: boolean;
      at: number;
    }>;
  };
  requirements: {
    ports: string[];
    network: string;
    container_runtime?: string;
  };
}

export interface ActionPlan {
  title: string;
  steps: string[];
  impact: string;
  terminology?: string;
  // Present only on a stateful remove/drain the backend will refuse without a
  // typed acknowledgement (no recovery backup exists). Its value is the exact
  // token the operator must type; the plan modal renders a typed-confirm and
  // threads it into the payload so the backend data-loss gate is satisfied.
  data_loss_ack?: string;
  /**
   * B1: the plan's own plain sentence to show as the typed-confirm label (a
   * forced removal says what it loses), instead of the generic data-loss one.
   */
  ack_prompt?: string;
}

export interface FleetJob {
  id: string;
  type: string;
  state: string;
  progress: number;
  message: string;
}

export interface AppTemplate {
  id: string;
  name: string;
  description: string;
  default_port: number;
  /** The port an install would get (W7-D2): the default unless a stored model holds it. */
  offered_port?: number | null;
  memory: string;
  /** Keeps data on the machine it runs on, so it cannot be spread. */
  stateful?: boolean;
}

export interface InferenceRuntimes {
  pooled?: {
    engine: string;
    commit: string;
    node_counts: number[];
    network: string;
    models: Array<{
      id: string;
      name: string;
      download_bytes: number;
      kv_heads: number;
      max_sequence_length: number;
    }>;
  };
}

export interface LlmForm {
  deploymentMode: "single" | "replicated" | "pooled" | "gpu";
  nodeId: string;
  nodeIds: string[];
  pooledModel: string;
  name: string;
  /**
   * The GPU (vLLM) deployment name. It lives here rather than inside
   * `GpuServeForm` because that sub-form is unmounted whenever the
   * deployment-mode radios leave "GPU serving", and a name the owner typed must
   * survive that — a lost typed value is a defect, not a reset.
   */
  gpuName: string;
  repository: string;
  file: string;
  sizeGb: string;
  port: string;
}

/**
 * Accelerator facts for one node, as `GET /api/v2/cluster/capacity` reports
 * them. Discovered from sysfs (over SSH for a worker, from the controller's own
 * probe for the head) without rocm-smi, so an absent GPU is reported absent
 * with a `reason` and never guessed present.
 */
export interface GpuCapacity {
  present: boolean;
  /** Why a GPU is absent, when it is; empty when one is present. */
  reason: string;
  gfx_target_version: string;
  device_count: number;
  vram_total_bytes: number;
  vram_used_bytes: number;
  gtt_total_bytes: number;
  gtt_used_bytes: number;
  /** MemTotal, for sizing a raisable GTT ceiling in a later phase. */
  system_ram_bytes: number | null;
  /** Addressable memory a model could be placed in (VRAM, else GTT aperture). */
  addressable_bytes: number;
}

export interface NodeCapacity {
  node_id: string;
  name: string;
  role: string;
  capacity: {
    /** Physical cores, or null when the node could not read its topology. */
    cpu: number | null;
    cpu_threads: number;
    memory_bytes: number;
    gpu: GpuCapacity;
  };
  reserved: {
    memory_bytes: number;
    /** Phase 1: always 0. Swarm does not reserve GPU memory (see `notes`). */
    gpu_memory_bytes: number;
    /** Names of the managed services committed to this node. */
    from: string[];
  };
  free: { memory_bytes: number; gpu_memory_bytes: number };
  /**
   * The node's usable state as the capacity ledger derives it (ACC-091):
   * "controller" | "ready" | "not-joined" | "drained" | "paused" | "offline" |
   * "unknown". Kept as a string so a word this console does not know reads as
   * unknown rather than being narrowed away. Absent from a ledger too old to
   * carry it, which the machine card also reads as unknown.
   */
  state?: NodeUsableState | string;
  /** True only for the active controller and a joined, Ready, active worker. */
  schedulable?: boolean;
  /** Why the node is not usable, in plain English; "" for the controller/ready. */
  state_reason?: string;
}

/** The node states the capacity ledger writes (contract 3). */
export type NodeUsableState =
  | "controller"
  | "ready"
  | "not-joined"
  | "drained"
  | "paused"
  | "offline"
  | "unknown";

/**
 * The whole-cluster capacity ledger. Phase 1 of the cluster capacity
 * foundation: the shared, read-only data layer that cluster app placement and
 * distributed-GPU LLM sizing both read.
 */
export interface ClusterCapacityLedger {
  nodes: NodeCapacity[];
  cluster: {
    node_count: number;
    capacity: {
      cpu: number;
      cpu_threads: number;
      memory_bytes: number;
      gpu: {
        present_nodes: number;
        vram_total_bytes: number;
        gtt_total_bytes: number;
        addressable_bytes: number;
      };
    };
    reserved: { memory_bytes: number; gpu_memory_bytes: number; from: string[] };
    /** Sums SCHEDULABLE nodes only (contract 3); `capacity` sums every node. */
    free: { memory_bytes: number; gpu_memory_bytes: number };
    /** How many nodes can take work now, and the memory they hold (contract 3). */
    schedulable_node_count?: number;
    schedulable_memory_bytes?: number;
  };
  /**
   * Reservations with no single node to attribute to yet — a pooled/replicated
   * LLM's per-shard placement is deferred to the sizing phase — recorded rather
   * than guessed onto a node.
   */
  unattributed_reservations: Array<{
    name: string;
    node_id: string;
    pool_label: string;
    workload: string;
    reservation_bytes: number;
    replicas: number;
    reason: string;
  }>;
  /** Phase-1 caveats, keyed by topic (e.g. `gpu_reservation`). */
  notes: Record<string, string>;
}

/**
 * The GPU fit/sizing decision `POST /api/v2/cluster/fit` returns (Phase 2a).
 * A pure, honest verdict the deploy UI shows *before* the owner commits: does
 * this model fit on one GPU node, sharded across several, or not at all — with
 * every number (weights, KV cache, overhead, deficit) derivable from the
 * `budget` block and the formulas beside it.
 */
export type GpuFitVerdict =
  | "single"
  | "distributed"
  /**
   * VD-129. Under `intent: throughput`, every selected node holds the whole
   * model at the launch fraction, so one copy runs on each behind a balancer
   * on this controller. A placement, so it is servable; the engine's summary
   * says what the cluster gains (concurrency, never a faster answer).
   */
  | "replicated"
  | "wont_fit"
  /**
   * VD-127 D6. The model fits one of the machines that were ticked, so
   * clustering is the wrong tool: a one-box model is served as the AI Chat
   * model on llama.cpp (Mode A), and vLLM is never offered for a single node.
   * The refusal is produced by the pure fit engine, so the preview and the
   * deploy say the same sentence.
   */
  | "single_refused"
  /**
   * VD-127 D10, narrowed by VD-129. What the throughput intent cannot do yet
   * — replicate without this controller holding a replica — is refused by the
   * engine with its own sentence, and no placement is invented for it.
   */
  | "unsupported_intent";

/** One GPU node as the decision echoes it, free memory first. */
export interface GpuFitNode {
  node_id: string;
  name: string;
  free_gpu_bytes: number;
  free_gpu_gb: number;
  /**
   * What stopping this node's AI Chat model would give back (VD-127 D1).
   * Reported beside the free figure rather than folded into it, so nobody has
   * to reconcile a "free" number against what the GPU is visibly holding. Only
   * the controller ever carries one, and only while Mode A is actually running.
   */
  reclaimable_gpu_bytes?: number;
  reclaimable_gpu_gb?: number;
  usable_gpu_bytes?: number;
  usable_gpu_gb?: number;
  addressable_bytes: number;
  gtt_total_bytes: number;
  system_ram_bytes: number | null;
  gfx_target_version: string;
}

export interface GpuFitModel {
  name: string;
  weight_bytes: number;
  weight_gb: number;
  weight_estimated: boolean;
  /** `measured` when a weight byte size was supplied, else estimated. */
  weight_source: "measured" | "estimated-from-parameters";
  parameter_billions: number;
  quantization: string;
  /** Nominal bytes/param used for an estimate; null when weights were measured. */
  bytes_per_parameter: number | null;
  hidden_layers: number;
  attention_heads: number;
  kv_heads: number;
  head_dim: number;
  context_length: number;
  slots: number;
}

/** Weights + minimum KV budget + overhead margin, each part and the total. */
export interface GpuFitBudget {
  weight_bytes: number;
  kv_cache_bytes: number;
  overhead_bytes: number;
  required_bytes: number;
  required_gb: number;
  overhead_fraction: number;
  /** Plain-language KV-cache formula behind `kv_cache_bytes`. */
  kv_cache_formula: string;
  /** Plain-language overhead formula behind `overhead_bytes`. */
  overhead_formula: string;
}

export interface GpuFitCluster {
  gpu_node_count: number;
  total_free_gpu_bytes: number;
  total_free_gpu_gb: number;
  /** The sum of every node's `reclaimable_gpu_bytes` (VD-127 D1). */
  total_reclaimable_gpu_bytes?: number;
  total_reclaimable_gpu_gb?: number;
  nodes: GpuFitNode[];
}

/** `verdict === "single"`: the named node that holds the whole model. */
export interface GpuFitSinglePlacement extends GpuFitNode {
  headroom_bytes: number;
  headroom_gb: number;
}

/**
 * `verdict === "distributed"`: the N nodes the model is split across. A
 * pipeline unless the owner chose tensor-parallel (VD-167); either way
 * `parallelism_reason` is the backend's own sentence for the chosen split.
 */
export interface GpuFitDistributedPlacement {
  node_count: number;
  parallelism: "tensor" | "pipeline";
  parallelism_reason: string;
  link: string;
  shard_bytes: number;
  shard_gb: number;
  weight_shard_bytes: number;
  total_free_gpu_bytes: number;
  nodes: GpuFitNode[];
}

/** `verdict === "wont_fit"`: one concrete way out, with the numbers to act on. */
export interface GpuFitOption {
  kind: "shorter_context" | "raise_gtt_ceiling" | "smaller_quantization" | "add_node";
  detail: string;
  /** For `raise_gtt_ceiling`: the achievable ceiling per node, from its RAM. */
  per_node?: Array<{
    node_id: string;
    name: string;
    current_addressable_bytes: number;
    current_free_gpu_bytes: number;
    system_ram_bytes: number | null;
    achievable_gtt_ceiling_bytes: number | null;
    achievable_gtt_ceiling_gb: number | null;
    reason: string;
  }>;
}

export interface GpuFitDecision {
  verdict: GpuFitVerdict;
  model: GpuFitModel;
  budget: GpuFitBudget;
  cluster: GpuFitCluster;
  /** What clustering was asked for; echoed back by the engine (VD-127 D10). */
  intent?: GpuClusterIntent | string;
  /**
   * Present for `single` and `distributed`, and also for `single_refused` —
   * which names the one machine the model fits on, so the refusal can say where
   * to serve it instead. Absent for `wont_fit` and `unsupported_intent`. Not
   * read for `replicated`: that verdict's whole story is its summary, and the
   * console draws nothing from a placement shape it has no contract for.
   */
  placement?: GpuFitSinglePlacement | GpuFitDistributedPlacement;
  /** Present when `verdict === "wont_fit"`. */
  deficit_bytes?: number;
  deficit_gb?: number;
  options?: GpuFitOption[];
  /** One-sentence plain-language verdict for the deploy UI. */
  summary: string;
  /**
   * Present only for a `wont_fit` that the largest allowed GPU memory pools
   * would turn into a fit (`vaelor/gpu_memory_pool` `pool_fit_hint`): the
   * same engine asked again, so the sentence never promises a fit it would
   * then refuse.
   */
  gpu_memory_pool?: {
    would_fit: boolean;
    machines: Array<{ node_id: string; name: string; size_gib: number }>;
    sentence: string;
  };
  /**
   * Whether the model's chat template has a thinking switch (`vaelor/model_thinking`,
   * the one owner), asked about the repo the deploy would serve. Absent when
   * the fit request named no model source.
   */
  thinking?: GpuThinkingSwitch;
  /**
   * What the deploy form may offer this model (`vaelor/vllm_serve_options`
   * `serving_choices`): the pinned vLLM images, multi-token prediction and the
   * text-only switch, from the profile the deploy decides with.
   */
  serving?: GpuServingChoices;
  /** The split the fit was asked for (VD-167); nothing chosen is `pipeline`. */
  split_mode?: GpuSplitMode;
  /**
   * What the deploy would refuse before starting (VD-168), in its own words:
   * the held mode switch, a cluster link that cannot be used, a tensor split
   * the model's heads cannot take, machines with unequal GPU counts. The
   * verdict is kept; a surface offers no Serve while this is present.
   */
  refusal?: { code: string; message: string };
  /**
   * W4-D4 (`cluster_fit_guards.mode_switch_notice`): present only when this
   * deploy would turn GPU clustering on (this controller selected, the switch
   * free), with the heading and sentence the form shows.
   */
  mode_switch?: { heading: string; message: string };
  /**
   * A tensor split's link recommendation (`cluster_link_recommendation`): the
   * fastest link every machine shares, or `null` with the reason none is.
   * A recommendation only; the form never switches the link for the owner.
   */
  link_recommendation?: GpuLinkRecommendation;
}

/** How a model split across machines is split (VD-167). */
export type GpuSplitMode = "pipeline" | "tensor";

/** The link a split rides, by the key the fit and the deploy take as `split_link`. */
export type GpuSplitLink = "ethernet" | "cluster-link";

export interface GpuLinkRecommendation {
  recommended: GpuSplitLink | null;
  links: Array<{ key: GpuSplitLink; label: string; mbps: number | null }>;
  reason: string;
}

/** One pinned vLLM image, and how fast this model's 4-bit weights run in it. */
export interface GpuServingImage {
  key: string;
  label: string;
  /** A plain sentence when the weights run slowly in this image; "" otherwise. */
  kernel_note: string;
}

/** The serving choices one model allows (`vaelor/vllm_serve_options`). */
export interface GpuServingChoices {
  images: GpuServingImage[];
  default_image: string;
  /**
   * A plain sentence when a hybrid model will be started without its cache
   * settings because a fact they are decided from could not be read - its
   * config.json, or a machine's GPU family; "" or absent otherwise.
   */
  cache_note?: string;
  /** `suggested_tokens` is the draft count that was measured; absent on an older backend. */
  mtp: { available: boolean; max_tokens: number; suggested_tokens?: number; detail: string };
  text_only: { applies: boolean; detail: string };
}

/** What the deploy form may offer about thinking for one model (owner decision 2026-09-29). */
export interface GpuThinkingSwitch {
  switch: boolean;
  /** Always false: a cluster model answers without thinking unless asked. */
  default: boolean;
  detail: string;
}

/**
 * One cached-weights row as `GET /api/v2/cluster/models` reports it, from the
 * ``model_cache`` store. A row is `ready` when its weights are fully on disk,
 * `pulling` while a supervised fetch runs, and `error` when a pull failed — each
 * state rendered as itself, never smoothed into a lie.
 */
/** A cluster deployment that still needs a node's cached weights (ACC-110). */
export interface ClusterModelUser {
  name: string;
  state: string;
  /** Plain words for the state: "serving", "loading", "paused, and it loads again…". */
  state_label: string;
}

export interface ClusterModel {
  node_id: string;
  repo: string;
  revision?: string;
  state: "ready" | "pulling" | "error" | string;
  bytes_on_disk?: number;
  message?: string;
  unit?: string;
  /** Deployments that will read these weights again; Remove is refused while any do. */
  in_use_by?: ClusterModelUser[];
  /** False when the deployments could not be read: use is then unknown, not "none". */
  in_use_known?: boolean;
}

/** One node's slice of the library: what it holds and how much root space is
 *  left. `cached_bytes` sums only the node's `ready` rows. */
export interface ClusterModelNode {
  node_id: string;
  name: string;
  root_free_bytes: number;
  cached_bytes: number;
  models: ClusterModel[];
}

/** The whole cached-weights inventory joined with per-node disk accounting. */
export interface ClusterModelInventory {
  models: ClusterModel[];
  nodes: ClusterModelNode[];
}

/**
 * VD-125. The cluster NIC a machine recorded when it joined: its name, the
 * address the fact was captured against, and — since the link-speed fix — the
 * link's real negotiated speed and medium, read off the machine's own sysfs.
 * Every measured field is honest: `speed_mbps` is `null` (with a `speed_reason`)
 * when the rate could not be read, and `medium` is `"unknown"` (with a
 * `medium_reason`) rather than a guessed "wired". `reason` is set only when the
 * NIC itself could not be named, in which case the link fields are absent.
 */
export interface ClusterInterface {
  name: string;
  address?: string;
  reason?: string;
  speed_mbps?: number | null;
  speed_reason?: string;
  medium?: "wired" | "wireless" | "unknown" | string;
  medium_reason?: string;
}

/**
 * A link rate as an operator reads it: "2.5 GbE", "1 GbE", "100 MbE". Whole
 * gigabit values drop the decimal; sub-gigabit stays in MbE. Empty string when
 * the rate is unknown (`null`/absent), so callers render their own honest
 * fallback rather than a fabricated "0".
 */
export const formatLinkSpeed = (mbps?: number | null): string => {
  if (typeof mbps !== "number" || !Number.isFinite(mbps) || mbps <= 0) return "";
  if (mbps >= 1000) {
    const gbe = mbps / 1000;
    return `${Number.isInteger(gbe) ? gbe : gbe.toFixed(1)} GbE`;
  }
  return `${mbps} MbE`;
};

//: The medium word shown to an operator, keyed on the backend's own value.
const LINK_MEDIUM_WORDS: Record<string, string> = {
  wired: "wired",
  wireless: "Wi-Fi",
};

/**
 * One machine's real cluster link as a short phrase — "2.5 GbE, wired",
 * "Wi-Fi", "link speed unknown" — derived from the captured fact and never a
 * guess. This is the ONE home of that derivation, so the fleet card and the
 * serve form read a link identically. The medium alone (a Wi-Fi NIC has no
 * sysfs speed) or the speed alone (a wired link whose medium could not be
 * confirmed) each render on their own; only a fact with neither reads unknown.
 */
export const formatClusterLink = (link?: ClusterInterface): string => {
  const speed = formatLinkSpeed(link?.speed_mbps);
  const medium = link?.medium ? LINK_MEDIUM_WORDS[link.medium] ?? "" : "";
  if (speed && medium) return `${speed}, ${medium}`;
  if (speed) return speed;
  if (medium) return medium;
  return "link speed unknown";
};

export const formatMemory = (bytes = 0) =>
  bytes ? formatQuantity(bytes, "capacity") : "Unknown";

export const isPrivateIpv4 = (value: string) => {
  const octets = value.split(".");
  if (octets.length !== 4 || octets.some((part) => !/^\d{1,3}$/.test(part))) {
    return false;
  }
  const numbers = octets.map(Number);
  if (numbers.some((part) => part < 0 || part > 255)) return false;
  return numbers[0] === 10
    || (numbers[0] === 172 && numbers[1] >= 16 && numbers[1] <= 31)
    || (numbers[0] === 192 && numbers[1] === 168);
};
