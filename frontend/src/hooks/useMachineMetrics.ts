import { useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import type { Metrics } from "../types";
import type { RetentionStatus } from "../lib/retention";
import type { LatestSample, WorkerIngest } from "../lib/workerFreshness";

/**
 * A node's live + short-history metrics for the Fleet machine-detail
 * (Phase E1 controller, Phase E2c worker).
 *
 * Every read is lazy — the hook does nothing until `enabled` (the card is
 * expanded on an online node) — and every read is withdrawn when the card
 * collapses or unmounts, exactly as `useMachineProfile` withdraws its
 * discovery, so a closed card leaves no poller running and a straggler cannot
 * land in another card's render.
 *
 * Two shapes, decided by `nodeId`:
 *
 * - CONTROLLER (`nodeId` omitted): the no-node `/telemetry/history` route for
 *   the trend - which is the controller's OWN series only, never an average
 *   over every node (ACC-082) - fetched once per window, plus
 *   `/telemetry/current` polled on a light interval for the live headline.
 * - WORKER (`nodeId` given, E2c): `/telemetry/history?node=<id>` ONLY — that
 *   route carries the worker's own series (workers push to the controller since
 *   E2b), its newest raw sample (`latest`) and its ingest facts. It is
 *   re-fetched on a light interval. A worker has NO live-current endpoint —
 *   `/telemetry/current` is the controller's own snapshot, so calling it for a
 *   worker would misattribute the controller's metrics to the worker — the
 *   worker's headline is its fresh `latest` sample (`lib/workerFreshness`).
 */

export interface HistoryPoint {
  t: string | null;
  v: number | null;
}

export interface HistorySeries {
  points: HistoryPoint[];
}

export interface TelemetryHistory {
  scope: string;
  node: string;
  requested_window: string;
  window_seconds: number;
  bucket_seconds: number;
  sample_interval_seconds: number;
  available: boolean;
  reason: string;
  retention: RetentionStatus | null;
  series: Record<string, HistorySeries>;
  /**
   * Epoch seconds of the resolved node's most recent raw sample, or `null`
   * when it has never reported. Filled for a worker (E2a) and, since the
   * controller-only history fix, for the controller's own series too.
   */
  last_sample_at?: number | null;
  /** Age of `last_sample_at` in seconds, or `null` when never reported. */
  last_sample_age_seconds?: number | null;
  /**
   * The server's verdict on whether this node is reporting (ACC-128): the one
   * freshness rule the alert engine also uses. Absent from an older server.
   */
  reporting?: boolean;
  reporting_window_seconds?: number;
  /**
   * The newest RAW sample for the resolved node, keyed like `series`. The
   * current figure a card shows is read from here (through
   * `currentWorkerValue`), never from the newest bucket of the trend.
   */
  latest?: LatestSample | null;
  /** Worker scope only: when the controller last accepted or refused a post. */
  ingest?: WorkerIngest | null;
}

export interface MachineMetricsState {
  history: TelemetryHistory | null;
  historyError: string;
  loading: boolean;
  current: Metrics | null;
}

/** How often the controller's live headline refreshes while its card is open. */
const CURRENT_POLL_MS = 5000;
/** How often a worker's node-scoped history re-reads while its card is open. */
const WORKER_POLL_MS = 10000;

export function useMachineMetrics(
  enabled: boolean,
  nodeId?: string,
  windowSpec = "24h",
): MachineMetricsState {
  const [history, setHistory] = useState<TelemetryHistory | null>(null);
  const [historyError, setHistoryError] = useState("");
  const [loading, setLoading] = useState(enabled);
  const [current, setCurrent] = useState<Metrics | null>(null);

  useEffect(() => {
    if (!enabled) return;
    const discovery = new AbortController();
    const { signal } = discovery;
    const path = nodeId
      ? `/telemetry/history?node=${encodeURIComponent(nodeId)}&window=${encodeURIComponent(windowSpec)}`
      : `/telemetry/history?window=${encodeURIComponent(windowSpec)}`;
    // The first read owns the loading line and the error surface; a later
    // worker refresh updates the trend silently and, if it fails, keeps the
    // last-known trend rather than blanking a card that was working.
    let first = true;
    setLoading(true);
    setHistoryError("");
    const read = () => {
      void apiRequest<TelemetryHistory>(path, { signal })
        .then((data) => {
          if (signal.aborted) return;
          setHistory(data);
          setLoading(false);
          first = false;
        })
        .catch((error: unknown) => {
          if (signal.aborted || !first) return;
          setHistoryError(
            error instanceof Error ? error.message : "Recent metrics could not be read.",
          );
          setLoading(false);
          first = false;
        });
    };
    read();
    const timer = nodeId ? window.setInterval(read, WORKER_POLL_MS) : null;
    return () => {
      if (timer !== null) window.clearInterval(timer);
      discovery.abort();
    };
  }, [enabled, nodeId, windowSpec]);

  useEffect(() => {
    // A worker has no live-current endpoint; only the controller polls it.
    if (!enabled || nodeId) return;
    const poll = new AbortController();
    const { signal } = poll;
    const read = () => {
      void apiRequest<{ metrics: Metrics }>("/telemetry/current", { signal })
        .then((data) => {
          if (!signal.aborted) setCurrent(data.metrics ?? null);
        })
        .catch(() => {
          // The live headline is best-effort; the trend carries the honest
          // story, so a dropped current read does not blank the card.
        });
    };
    read();
    const timer = window.setInterval(read, CURRENT_POLL_MS);
    return () => {
      window.clearInterval(timer);
      poll.abort();
    };
  }, [enabled, nodeId]);

  return { history, historyError, loading, current };
}
