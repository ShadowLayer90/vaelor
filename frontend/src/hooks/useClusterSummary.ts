import { useEffect, useState } from "react";
import type { FleetSummary } from "../components/fleetTypes";
import type { Role } from "../types";
import { read, type Read } from "./useHomeSummary";

/** How often the shell re-reads the cluster summary off Home: worker software changes on a recheck, not per second. */
export const CLUSTER_SUMMARY_REFRESH_MS = 60_000;

/**
 * The one `/cluster` read the console makes (VD-200): whether this machine
 * leads a cluster (the breadcrumb's "This controller", This machine's
 * "controller", Home's Machines role), each worker's software state (the
 * rail's warning card, Home's attention list) and what the GPU serves.
 *
 * It is a `Read`, so a failed or refused read is never mistaken for an empty
 * cluster. A viewer never asks (the route is for operators and
 * administrators), a refused read is not asked again, and nothing is asked
 * while the tab is hidden.
 */
export function useClusterSummary(role: Role, intervalMs: number = CLUSTER_SUMMARY_REFRESH_MS): Read<FleetSummary> {
  const allowed = role === "operator" || role === "administrator";
  const [summary, setSummary] = useState<Read<FleetSummary>>(allowed ? { state: "loading" } : { state: "forbidden" });
  useEffect(() => {
    if (!allowed) {
      setSummary({ state: "forbidden" });
      return undefined;
    }
    const controller = new AbortController();
    let timer: number | undefined;
    const refresh = () => {
      void read<FleetSummary>("/cluster", controller.signal).then((next) => {
        if (controller.signal.aborted) return;
        setSummary(next);
        // A route that refused this account will refuse it again: stop asking.
        if (next.state === "forbidden") window.clearInterval(timer);
      });
    };
    refresh();
    timer = window.setInterval(() => { if (!document.hidden) refresh(); }, intervalMs);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [allowed, intervalMs]);
  return summary;
}

/** The read's data, or null when it was not read. */
export function clusterData(summary: Read<FleetSummary>): FleetSummary | null {
  return summary.state === "ok" ? summary.data : null;
}

/** "controller" when this machine leads an initialized cluster; otherwise nothing. */
export function clusterRoleOf(summary: FleetSummary | null): "controller" | undefined {
  return summary?.controller?.initialized === true ? "controller" : undefined;
}
