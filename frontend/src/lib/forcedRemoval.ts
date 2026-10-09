import { ApiError } from "./api";

/**
 * When the console offers a FORCED removal of a worker (owner decision B1).
 *
 * A plain removal drains the worker and waits for it to leave; for a machine
 * the controller cannot reach, or one a deployment still needs, the backend
 * refuses that. Only then - never on a card by default - the console offers
 * "Review forced removal", which opens the backend's own plan (typed
 * acknowledgement included) for `force-remove-node`. The backend itself
 * refuses to force a machine it CAN reach, so the console carries no
 * reachability rule of its own.
 *
 * The two refusals that open the offer:
 * - `POST /cluster/plan {action: "remove-node"}` answered 409 with code
 *   `cluster_node_in_use` or `cluster_node_unreachable`;
 * - a `cluster.node.remove` job (not itself forced) that ended failed with a
 *   message that points at a forced removal ("... or review a forced
 *   removal ...").
 * Both removal surfaces (the Fleet card and Setup's awaiting-join card) go
 * through the same plan request, so both behave the same.
 */

export const FORCE_REMOVE_ACTION = "force-remove-node";

const REFUSAL_CODES = new Set(["cluster_node_in_use", "cluster_node_unreachable"]);
const FORCED_REMOVAL_PHRASE = "forced removal";

export interface ForcedRemovalOffer {
  nodeId: string;
  /** The backend's refusal, shown beside the offer. */
  reason: string;
}

/** The offer a refused `remove-node` plan opens, or null. */
export function offerFromPlanRefusal(
  action: string,
  nodeId: string | undefined,
  error: unknown,
): ForcedRemovalOffer | null {
  if (action !== "remove-node" || !nodeId) return null;
  if (!(error instanceof ApiError) || !REFUSAL_CODES.has(error.code)) return null;
  return { nodeId, reason: error.message };
}

/** The shape of a cluster job the offer reads (the `/jobs/<id>` record). */
export interface RemovalJobLike {
  type: string;
  state: string;
  message?: string;
  payload?: Record<string, unknown>;
}

/** The offer a failed, non-forced `cluster.node.remove` job opens, or null. */
export function offerFromFailedJob(job: RemovalJobLike | null | undefined): ForcedRemovalOffer | null {
  if (!job || job.type !== "cluster.node.remove" || job.state !== "failed") return null;
  if (job.payload?.force === true) return null;
  const nodeId = typeof job.payload?.node_id === "string" ? job.payload.node_id : "";
  const message = job.message ?? "";
  if (!nodeId || !message.toLowerCase().includes(FORCED_REMOVAL_PHRASE)) return null;
  return { nodeId, reason: message };
}
