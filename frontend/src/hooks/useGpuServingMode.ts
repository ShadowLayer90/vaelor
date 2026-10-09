import { useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import {
  UNKNOWN_SERVING_MODE,
  resolveServingMode,
  servingModeFromDeployments,
  type GpuServingMode,
} from "../lib/gpuServingMode";
import type { FleetSummary } from "../components/fleetTypes";
import type { Role } from "../types";

/**
 * Which GPU engine is serving this appliance right now — Mode A or Mode B.
 *
 * Read from the fleet summary's deployments through the one derivation in
 * `lib/gpuServingMode`, so the AI Chat panel and the Cluster tab cannot reach
 * different answers about the same appliance (VD-127 D3's rule, applied to the
 * console).
 *
 * **One resolution, here.** `targetKind` is whatever the LLM Server surface
 * published (`none` / `managed-local` / `cluster`, or nothing on a control
 * plane that does not publish it yet). When it is present it wins outright —
 * it is the value the executor's failure-watch and the auth proxy act on — and
 * the deployments derivation is only the fallback. The host of every surface
 * that shows the mode calls this once and hands the answer down; a leaf that
 * re-resolved from its own copy of the inputs could disagree with the card
 * beside it the moment the two inputs differed.
 *
 * Three deliberate refusals:
 *
 * * A **viewer never asks.** `GET /cluster` is operator-gated, so the request
 *   would be a guaranteed 403 on every AI Chat panel a viewer opens.
 * * A failure is **`known: false`**, never Mode A. An unreadable answer is not
 *   evidence that nothing is clustered, and a surface must render nothing
 *   rather than assert "managed local model" on a question it could not ask.
 * * The read is **withdrawn with the surface** (VD-126): the panel is opened
 *   and closed freely, and a closed panel's discovery — including the
 *   transport's dead-socket retry — must not outlive it.
 */
export function useGpuServingMode(
  role: Role | undefined,
  targetKind?: string,
): GpuServingMode {
  const allowed = role === "operator" || role === "administrator";
  const [derived, setDerived] = useState<GpuServingMode>(UNKNOWN_SERVING_MODE);

  useEffect(() => {
    if (!allowed) {
      setDerived(UNKNOWN_SERVING_MODE);
      return;
    }
    const discovery = new AbortController();
    const { signal } = discovery;
    void apiRequest<FleetSummary>("/cluster", { signal })
      .then((summary) => {
        if (signal.aborted) return;
        setDerived(servingModeFromDeployments(summary.pooled_deployments ?? []));
      })
      .catch(() => {
        if (!signal.aborted) setDerived(UNKNOWN_SERVING_MODE);
      });
    return () => discovery.abort();
  }, [allowed]);

  return resolveServingMode(targetKind, derived);
}
