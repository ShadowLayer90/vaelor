import { formatQuantity } from "../lib/format";

export interface FleetNode {
  id: string;
  name: string;
  host: string;
  port: number;
  role: string;
  state: string;
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
    reachable?: boolean;
  };
}

export interface FleetSummary {
  controller: {
    initialized: boolean;
    driver: string;
    role: string;
    advertise_address?: string;
    candidate_address?: string;
    placement?: FleetNode & { eligible: boolean; reason?: string };
  };
  runtime: {
    available: boolean;
    initialized: boolean;
    control_available: boolean;
    /** Task #75: the engine fact, separate from the Swarm fact. */
    engine?: "ready" | "absent" | "unreadable";
    engine_reason?: string;
    nodes?: Array<Record<string, string>>;
    services?: Array<Record<string, string>>;
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
  memory: string;
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
  deploymentMode: "single" | "replicated" | "pooled";
  nodeId: string;
  nodeIds: string[];
  pooledModel: string;
  name: string;
  repository: string;
  file: string;
  sizeGb: string;
  port: string;
  useForAssistant: boolean;
}

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
