import { useEffect, useMemo, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { metricNumber } from "../lib/metrics";
import {
  agedFreshnessSource,
  currentWorkerValue,
  workerFreshness,
  workerReadingAge,
  type FreshnessSource,
} from "../lib/workerFreshness";
import type { Metrics } from "../types";
import type { TelemetryHistory } from "./useMachineMetrics";

/**
 * The live MEASURED memory of every Fleet machine, for the collapsed machine
 * cards and the fleet headline (VD-129, ACC-083, ACC-119, ACC-120).
 *
 * Per machine, from the same sources the Home screen and the expanded machine
 * metrics read:
 * - the CONTROLLER reads `/telemetry/current`: `memory_used` (bytes, the
 *   operating system's own figure, MemTotal - MemAvailable, exactly what Home
 *   shows) and `gpu_gtt_used_bytes`;
 * - a WORKER reads `/telemetry/history?node=<id>` and takes its newest RAW
 *   sample (`latest`) for `memory_percent` and `gpu_gtt_used_bytes` - only
 *   while fresh, through the one rule in `lib/workerFreshness`.
 *
 * One failed read does not blank a machine (SC5). The last SUCCESSFUL answer
 * per machine is kept with the moment it arrived, and freshness is decided
 * again at every render from that answer's own timestamps plus the time since
 * it arrived: a worker's receive/sample age grows by the elapsed seconds and a
 * controller snapshot ages from its arrival. So a transient failure keeps the
 * figures (marked `readFailed`), and reads that keep failing let them age out
 * to `stale` on the same 45-second rule the expanded card uses - never a
 * reading shown as current past that window, never a figure silently dropped.
 *
 * A machine with no current figure maps to null fields and the card and the
 * headline say why - not reported, out of date, or could not be read - never
 * the placement ledger's reservation figure presented as free. The set is
 * re-read on a light interval, never with two reads in flight, and every read
 * is withdrawn on unmount.
 *
 * The FIRST read is made whether or not the browser calls the page visible.
 * It used to be skipped while hidden too, so a page opened in a tab the
 * browser reported hidden (a collapsed tab group) said "Memory use not
 * reported" on every card while the tiles beneath showed the figure: an
 * absence made by the observer, not by the machine (ACC-212). The poll after
 * it still pauses while the page is hidden, and showing the page reads at once.
 */

/** One machine's telemetry identity for the live read. */
export interface FleetMemoryNode {
  /** The card key this reading is looked up by (node_id or name). */
  key: string;
  /** The id `/telemetry/history?node=` validates against; omitted = controller. */
  telemetryId?: string;
  isController: boolean;
}

/** One machine's current measured figures; every figure null when not current. */
export interface FleetLiveReading {
  /** The controller's operating-system memory in use, in bytes. */
  memoryUsedBytes: number | null;
  /** A worker's operating-system memory in use, as a percentage of its total. */
  memoryPercent: number | null;
  /** Resident unified (GPU/GTT) memory, in bytes. */
  gttUsedBytes: number | null;
  /**
   * The newest read of this machine failed. Any figures above are from the
   * last successful read, still inside the freshness window.
   */
  readFailed?: boolean;
  /**
   * A reading was in hand but is now older than the freshness window, so no
   * figure is current (the figures above are null).
   */
  stale?: boolean;
  /**
   * Processor load (%) and CPU temperature (degrees C) from the SAME answer,
   * for the closed machine card's other two readings (VD-200 review): the card
   * used to poll the routes this hook already reads, a second reader of one
   * value (LESSONS 6), on every closed card.
   */
  cpuPercent?: number | null;
  cpuTemperatureC?: number | null;
  /** How old the figures are, in seconds, when they are current. */
  ageSeconds?: number | null;
  /** When a reading that went stale was taken (epoch seconds), when known. */
  since?: number | null;
  /** The controller refused this worker's newest post: its clock is off. */
  clockRefused?: boolean;
}

export interface FleetLiveMemory {
  /** The current reading per machine key; null when none is in hand. */
  readings: Record<string, FleetLiveReading | null>;
}

/** How often the fleet-wide readings refresh while the tab is open. */
const POLL_MS = 10000;

/**
 * How often the kept readings are re-aged even when no read has come back
 * (review nit). `apiRequest`'s timeout bounds the request, not the body read,
 * so a response whose body never finishes would otherwise hold `inFlight` and
 * freeze every figure at its last age for ever.
 */
const AGE_TICK_MS = 5000;

/** The last successful answer for one machine, stamped with when it arrived. */
type Snapshot =
  | {
    kind: "controller"; arrivedAtMs: number; memoryUsedBytes: number | null; gttUsedBytes: number | null;
    cpuPercent: number | null; cpuTemperatureC: number | null;
  }
  | { kind: "worker"; arrivedAtMs: number; source: FreshnessSource };

interface Entry {
  last: Snapshot | null;
  failed: boolean;
}

/**
 * How long a kept `/telemetry/current` answer for the CONTROLLER stands in for
 * a fresh one. The controller's live snapshot carries no reporting verdict (it
 * is read in-process, not posted), so this bounds only how stale a cached read
 * of it may be between polls; it is not the worker freshness rule, which is
 * the server's (`workerFreshness`).
 */
const CONTROLLER_READING_KEPT_SECONDS = 45;

/** The reading a kept entry supports at `nowMs`. */
export function readingAt(entry: Entry | undefined, nowMs: number): FleetLiveReading | null {
  if (!entry) return null;
  const empty = { memoryUsedBytes: null, memoryPercent: null, gttUsedBytes: null };
  const { last, failed } = entry;
  if (!last) return failed ? { ...empty, readFailed: true } : null;
  const elapsedSeconds = Math.max(0, (nowMs - last.arrivedAtMs) / 1000);
  if (last.kind === "controller") {
    if (elapsedSeconds > CONTROLLER_READING_KEPT_SECONDS) {
      return { ...empty, readFailed: failed, stale: true, since: Math.floor(last.arrivedAtMs / 1000) };
    }
    return {
      memoryUsedBytes: last.memoryUsedBytes,
      memoryPercent: null,
      gttUsedBytes: last.gttUsedBytes,
      readFailed: failed,
      cpuPercent: last.cpuPercent,
      cpuTemperatureC: last.cpuTemperatureC,
      // The snapshot's age is the time since it arrived, so a kept answer
      // after a failed read is never "updated just now" (VD-200 review).
      ageSeconds: elapsedSeconds,
    };
  }
  const aged = agedFreshnessSource(last.source, elapsedSeconds);
  const freshness = workerFreshness(aged);
  if (freshness.state === "stale") return { ...empty, readFailed: failed, stale: true, since: freshness.since };
  return {
    memoryUsedBytes: null,
    memoryPercent: currentWorkerValue(aged, "memory_percent"),
    gttUsedBytes: currentWorkerValue(aged, "gpu_gtt_used_bytes"),
    readFailed: failed,
    cpuPercent: currentWorkerValue(aged, "processor_load"),
    cpuTemperatureC: currentWorkerValue(aged, "cpu_temperature_c"),
    ageSeconds: freshness.state === "fresh" ? workerReadingAge(aged) ?? freshness.ageSeconds : null,
    clockRefused: freshness.state === "clock-refused",
  };
}

export function useFleetLiveMemory(nodes: FleetMemoryNode[]): FleetLiveMemory {
  const [entries, setEntries] = useState<Record<string, Entry>>({});
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    const tick = window.setInterval(() => setNowMs(Date.now()), AGE_TICK_MS);
    return () => window.clearInterval(tick);
  }, []);
  // The effect re-runs only when the machine set changes, not on every parent
  // render (the parent rebuilds the array each time); the current nodes are
  // read from a ref so the fetch always sees the latest identities.
  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;
  const signature = useMemo(
    () => nodes.map((node) => `${node.key}:${node.telemetryId ?? "-"}:${node.isController}`).join("|"),
    [nodes],
  );

  useEffect(() => {
    const targets = nodesRef.current;
    if (!targets.length) {
      setEntries({});
      return;
    }
    const controller = new AbortController();
    const { signal } = controller;
    let inFlight = false;

    // A snapshot on success, "failed" when the read failed, null when this
    // machine has no telemetry identity to read (nothing was attempted).
    const readOne = async (node: FleetMemoryNode): Promise<Snapshot | "failed" | null> => {
      try {
        if (node.isController) {
          const data = await apiRequest<{ metrics: Metrics }>("/telemetry/current", { signal });
          const metrics = data.metrics ?? {};
          return {
            kind: "controller",
            arrivedAtMs: Date.now(),
            memoryUsedBytes: metricNumber(metrics, "memory_used"),
            gttUsedBytes: metricNumber(metrics, "gpu_gtt_used_bytes"),
            cpuPercent: metricNumber(metrics, "cpu_percent"),
            cpuTemperatureC: metricNumber(metrics, "cpu_temperature"),
          };
        }
        if (!node.telemetryId) return null;
        const path = `/telemetry/history?node=${encodeURIComponent(node.telemetryId)}&window=15m`;
        const history = await apiRequest<TelemetryHistory>(path, { signal });
        return { kind: "worker", arrivedAtMs: Date.now(), source: history };
      } catch {
        return "failed";
      }
    };

    const read = () => {
      if (inFlight) return;
      inFlight = true;
      void Promise.all(targets.map(readOne))
        .then((results) => {
          if (signal.aborted) return;
          setEntries((previous) => {
            const next: Record<string, Entry> = {};
            targets.forEach((node, index) => {
              const result = results[index];
              if (result === "failed") {
                next[node.key] = { last: previous[node.key]?.last ?? null, failed: true };
              } else {
                next[node.key] = { last: result, failed: false };
              }
            });
            return next;
          });
          setNowMs(Date.now());
        })
        .finally(() => {
          inFlight = false;
        });
    };
    // Only the poll pauses while the page is hidden; the first read does not,
    // and a return to view reads at once.
    const visibilityChanged = () => {
      if (!document.hidden) read();
    };
    read();
    const timer = window.setInterval(visibilityChanged, POLL_MS);
    document.addEventListener("visibilitychange", visibilityChanged);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", visibilityChanged);
      controller.abort();
    };
  }, [signature]);

  const readings = useMemo(() => {
    const result: Record<string, FleetLiveReading | null> = {};
    for (const [key, entry] of Object.entries(entries)) result[key] = readingAt(entry, nowMs);
    return result;
  }, [entries, nowMs]);
  return { readings };
}
