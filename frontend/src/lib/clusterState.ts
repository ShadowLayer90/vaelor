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
  initialized: boolean;
  control_available: boolean;
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

export function clusterState(
  runtime: ClusterRuntimeFlags | undefined,
  enrollment?: ClusterEnrollmentFact,
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
      summary: "Docker is running, but this node is not yet a cluster controller.",
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
