import type {
  ClusterCapacityLedger,
  GpuCapacity,
  NodeCapacity,
} from "../components/fleetTypes";

/**
 * One authoritative cluster state, derived from the three separate flags the
 * API reports.
 *
 * Alpha 11 read only `runtime.initialized` for a badge, then described the node
 * as a working "Docker Swarm manager · scheduling authority" regardless — so
 * the screen could say "Head controller", "Engine: Docker Swarm" and
 * "NOT INITIALIZED" simultaneously while the activity log said cluster
 * initialize had completed. `runtime.available` — the actual reason — was never
 * shown at all.
 */

export interface ClusterRuntimeFlags {
  available: boolean;
  /**
   * Whether Vaelor set this node up as the cluster controller: Docker's Swarm
   * is active AND the controller record the join reads exists
   * (`cluster_store.controller_recorded`, LESSONS 6). It used to be Docker's
   * state alone, so a Swarm an uninstall left behind read "Controller active"
   * while the join refused, and "Review controller setup" was never offered
   * (v1.5 cold install, 2026-10-09).
   */
  initialized: boolean;
  control_available: boolean;
  /** Docker's own Swarm state, published beside `initialized`. */
  swarm_active?: boolean;
  /**
   * Task #75. What the controller observed about the container engine, kept
   * apart from what it observed about Swarm. `available: false` used to be the
   * single answer to every way one `docker info` call could fail, and this
   * screen turned it into "Check that Docker is installed and running" — on
   * two machines that were at that moment running containers Vaelor itself had
   * deployed. Absent from an older controller, which is why the copy below
   * falls back to naming the query rather than the engine.
   */
  engine?: "ready" | "absent" | "unreadable";
  engine_reason?: string;
}

/**
 * The served enrollment gate (`cluster_manager.enrollment_readiness`). The
 * backend has published this since the sudo-password form was first disabled;
 * this screen simply never read it, so "Add worker" stayed enabled — and
 * opened a form asking for an SSH/sudo password — beside its own sentence
 * saying workers could not be enrolled.
 */
export interface ClusterEnrollmentFact {
  available: boolean;
  reason: string;
  requires_sudo_password?: boolean;
}

export type ClusterStateId =
  | "not-read"
  | "runtime-unavailable"
  | "not-initialized"
  | "control-unavailable"
  | "ready";

export interface ClusterState {
  id: ClusterStateId;
  /** Short badge label. */
  label: string;
  tone: "healthy" | "degraded" | "neutral";
  /** What is true right now, in one sentence. */
  summary: string;
  /** What the user cannot do while this holds. */
  unavailable?: string;
  /** The next step, when there is one. */
  nextStep?: string;
  /** Whether the node may be described as an operating Swarm manager. */
  operatingAsController: boolean;
  /**
   * Whether the Add-worker control may be offered, and why not.
   *
   * VD-009a / task #54: `actionable` is **derived, never passed**. It is
   * computed here from the state this same call just reported, so a caller
   * cannot enable a control that outruns the sentence beside it — which is
   * exactly what happened: "Workers cannot be enrolled." above an enabled
   * primary button that opened a form with an SSH/sudo password field.
   */
  enrollment: { actionable: boolean; reason: string };
}

/**
 * The remedy sentence for an engine Vaelor could not read.
 *
 * Task #75. This used to be "Check that Docker is installed and running,
 * then reload" for every failure of one Swarm query. Docker *was* running: a
 * container had been deployed through Vaelor minutes earlier and Apps and AI
 * said `DOCKER READY` on the same machine at the same moment. Naming the
 * wrong condition sends the owner to reinstall something that is working.
 */
function engineCopy(runtime: ClusterRuntimeFlags): { summary: string; nextStep: string } {
  if (runtime.engine === "absent") {
    return {
      summary: "Docker is not installed on this node, so there is no cluster engine to use.",
      nextStep: "Install Docker on this node, then reload.",
    };
  }
  // `unreadable`, and the older controllers that report neither. Both cases
  // know one thing — the query did not answer — and stating more than that is
  // the defect. The next step is a control this screen's reader actually has
  // (#141): "check that the Vaelor service can reach the Docker socket" was a
  // shell instruction offered to a non-developer, with no control attached.
  return {
    summary: runtime.engine_reason
      || "Vaelor could not read this node's cluster state from Docker.",
    nextStep: "Reload this page. If the cluster stays unreadable, reboot the device from Home.",
  };
}

const number = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value) ? value : 0;

function readGpu(raw: unknown): GpuCapacity {
  const gpu = (raw ?? {}) as Partial<GpuCapacity>;
  return {
    present: gpu.present === true,
    reason: typeof gpu.reason === "string" ? gpu.reason : "",
    gfx_target_version:
      typeof gpu.gfx_target_version === "string" ? gpu.gfx_target_version : "",
    device_count: number(gpu.device_count),
    vram_total_bytes: number(gpu.vram_total_bytes),
    vram_used_bytes: number(gpu.vram_used_bytes),
    gtt_total_bytes: number(gpu.gtt_total_bytes),
    gtt_used_bytes: number(gpu.gtt_used_bytes),
    system_ram_bytes:
      typeof gpu.system_ram_bytes === "number" ? gpu.system_ram_bytes : null,
    addressable_bytes: number(gpu.addressable_bytes),
  };
}

/**
 * A defensive reader for `GET /api/v2/cluster/capacity`, in the same role
 * `clusterState` plays for the fleet summary: it turns the raw API payload into
 * a fully-shaped ledger with every numeric field defaulted, so a Phase-2
 * consumer (app placement, distributed-GPU LLM sizing) never has to guard an
 * absent field itself. It derives nothing new — the arithmetic is the
 * backend's; this only makes the shape safe to read.
 */
export function readCapacityLedger(
  raw: Partial<ClusterCapacityLedger> | undefined,
): ClusterCapacityLedger {
  const cluster = raw?.cluster;
  const clusterGpu = cluster?.capacity?.gpu;
  const nodes: NodeCapacity[] = (raw?.nodes ?? []).map((node) => ({
    node_id: node.node_id,
    name: node.name,
    role: node.role,
    capacity: {
      cpu: typeof node.capacity?.cpu === "number" ? node.capacity.cpu : null,
      cpu_threads: number(node.capacity?.cpu_threads),
      memory_bytes: number(node.capacity?.memory_bytes),
      gpu: readGpu(node.capacity?.gpu),
    },
    reserved: {
      memory_bytes: number(node.reserved?.memory_bytes),
      gpu_memory_bytes: number(node.reserved?.gpu_memory_bytes),
      from: node.reserved?.from ?? [],
    },
    free: {
      memory_bytes: number(node.free?.memory_bytes),
      gpu_memory_bytes: number(node.free?.gpu_memory_bytes),
    },
    // Contract 3 (ACC-091): the node's usable state, carried as the ledger
    // wrote it. A row without one is "unknown" and not schedulable - never
    // assumed ready.
    state: typeof node.state === "string" && node.state ? node.state : "unknown",
    schedulable: node.schedulable === true,
    state_reason: typeof node.state_reason === "string" ? node.state_reason : "",
  }));
  return {
    nodes,
    cluster: {
      node_count: number(cluster?.node_count),
      capacity: {
        cpu: number(cluster?.capacity?.cpu),
        cpu_threads: number(cluster?.capacity?.cpu_threads),
        memory_bytes: number(cluster?.capacity?.memory_bytes),
        gpu: {
          present_nodes: number(clusterGpu?.present_nodes),
          vram_total_bytes: number(clusterGpu?.vram_total_bytes),
          gtt_total_bytes: number(clusterGpu?.gtt_total_bytes),
          addressable_bytes: number(clusterGpu?.addressable_bytes),
        },
      },
      reserved: {
        memory_bytes: number(cluster?.reserved?.memory_bytes),
        gpu_memory_bytes: number(cluster?.reserved?.gpu_memory_bytes),
        from: cluster?.reserved?.from ?? [],
      },
      free: {
        memory_bytes: number(cluster?.free?.memory_bytes),
        gpu_memory_bytes: number(cluster?.free?.gpu_memory_bytes),
      },
      schedulable_node_count: number(cluster?.schedulable_node_count),
      schedulable_memory_bytes: number(cluster?.schedulable_memory_bytes),
    },
    unattributed_reservations: raw?.unattributed_reservations ?? [],
    notes: raw?.notes ?? {},
  };
}

export function clusterState(
  runtime: ClusterRuntimeFlags | undefined,
  enrollment?: ClusterEnrollmentFact,
  /**
   * Why the cluster summary could not be read, when the newest read failed or
   * was refused and none is in hand. Without it a refused read stayed
   * "Checking cluster" for ever, and the title row's actions were enabled with
   * no reason beside them (VD-200 review C1).
   */
  readError?: string,
): ClusterState {
  /*
   * Written once, at the end of every branch, from the state that branch just
   * decided. `capability()` in `vaelor/console_capabilities.py` does the same
   * thing for the console ladder and for the same reason: the only way to stop
   * a control outrunning its state is to give nobody the chance to set them
   * separately.
   */
  const gate = (id: ClusterStateId): { actionable: boolean; reason: string } => {
    if (id !== "ready") {
      return {
        actionable: false,
        reason: enrollment?.reason
          || "Workers cannot be enrolled until this node is an active cluster controller.",
      };
    }
    // Ready here, and the controller still gets the last word: it also checks
    // its own architecture, which this screen cannot observe (VD-031).
    return enrollment && !enrollment.available
      ? { actionable: false, reason: enrollment.reason || "This controller cannot enrol a worker right now." }
      : { actionable: true, reason: "" };
  };

  if (!runtime && readError) {
    const reason = readError.replace(/\.$/, "");
    return {
      id: "not-read",
      label: "Not read",
      tone: "neutral",
      summary: `The cluster state could not be read: ${reason}.`,
      // The reason itself is in the page's notice and the pill's description;
      // the sentence beside a disabled action stays short.
      unavailable: "The cluster state could not be read, so nothing can be placed on it.",
      nextStep: "Reload this page to read it again.",
      operatingAsController: false,
      enrollment: { actionable: false, reason: `The cluster state could not be read (${reason}).` },
    };
  }

  if (!runtime) {
    return {
      id: "runtime-unavailable",
      label: "Checking cluster",
      tone: "neutral",
      summary: "Vaelor is still reading the cluster runtime.",
      operatingAsController: false,
      enrollment: {
        actionable: false,
        reason: "Vaelor is still reading the cluster state on this node.",
      },
    };
  }

  // Ordered by what blocks what: without the engine, nothing else is meaningful.
  if (!runtime.available) {
    return {
      id: "runtime-unavailable",
      label: "Cluster engine unavailable",
      tone: "degraded",
      ...engineCopy(runtime),
      unavailable: "Workers cannot be enrolled and cluster apps cannot be placed.",
      operatingAsController: false,
      enrollment: gate("runtime-unavailable"),
    };
  }

  if (!runtime.initialized) {
    return {
      id: "not-initialized",
      label: "Not initialized",
      tone: "degraded",
      summary: runtime.swarm_active
        ? "Docker on this node still runs a cluster, likely from an earlier installation, but Vaelor has not set this node up as its controller."
        : "Docker is running, but this node is not yet a cluster controller.",
      unavailable: "Workers cannot be enrolled until the controller is set up.",
      nextStep: "Review controller setup to make this node the head controller.",
      operatingAsController: false,
      enrollment: gate("not-initialized"),
    };
  }

  if (!runtime.control_available) {
    return {
      id: "control-unavailable",
      label: "Controller degraded",
      tone: "degraded",
      summary: "This node is the cluster controller, but its control plane is not responding.",
      unavailable: "Existing workloads keep running; new placements will not be accepted.",
      nextStep: "Reload, then check the Docker service if the state persists.",
      operatingAsController: true,
      enrollment: gate("control-unavailable"),
    };
  }

  return {
    id: "ready",
    label: "Controller active",
    tone: "healthy",
    summary: "This node is the cluster controller and is accepting placements.",
    operatingAsController: true,
    enrollment: gate("ready"),
  };
}
