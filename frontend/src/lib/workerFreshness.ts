/**
 * The one freshness rule for a worker's telemetry reading, and the one place a
 * worker's CURRENT figure is read from (ACC-083, ACC-119, ACC-126).
 *
 * Two Fleet surfaces show a worker's live figures: the collapsed machine card
 * and the fleet headline (through `useFleetLiveMemory`), and the expanded
 * machine metrics (through `useMachineMetrics`). Each used to decide on its own
 * whether a reading was current, and they disagreed: the collapsed card kept a
 * GPU figure up to 15 minutes old while the expanded card already said "Not
 * reporting", and the expanded headline showed a partial bucket MEAN labelled
 * "updated just now". Both now ask this module.
 *
 * - WHETHER the worker is reporting is the server's one verdict (`reporting`,
 *   `telemetry_ingest_status.node_is_reporting` over
 *   `telemetry_store.REPORTING_WINDOW_SECONDS`), the same answer the alert
 *   engine, the Performance tab and the self-heal read (ACC-128). The server
 *   judges on the controller's receive age first, so a worker whose clock is
 *   skewed but which IS posting reads live (ACC-126). This module holds no
 *   window of its own: a reply without the verdict is not fresh.
 * - AGE, shown beside a fresh reading, is how long ago the controller last
 *   ACCEPTED a post (`ingest.received_age_seconds`), else the newest stored
 *   sample's age (`last_sample_age_seconds`).
 * - A stale reading is never a current value anywhere: `currentWorkerValue`
 *   returns null for it.
 * - A post the controller REFUSED because its timestamps were outside the
 *   accepted window (`ingest.clock_refused`) is its own state, carrying the
 *   backend's plain-English reason, rather than a bare "Not reporting".
 * - The current figure is the newest RAW sample (`latest.values`), never the
 *   newest bucket of the trend, which is a mean over a partial window.
 */

/** The ingest facts the history route carries for a worker (null for the controller). */
export interface WorkerIngest {
  received_at: number | null;
  received_age_seconds: number | null;
  clock_offset_seconds: number | null;
  clock_refused: boolean;
  clock_refused_at: number | null;
  reason: string;
}

/** The newest RAW sample for the resolved node, keyed by the trend's series keys. */
export interface LatestSample {
  t: number | null;
  values: Record<string, number | null>;
  /**
   * Which of the node's two GPU temperatures this row shows: "graphics engine"
   * or "edge" (`vaelor.platforms.gpu_temperature`), null when it has neither.
   * Absent from an older server.
   */
  gpu_temperature_sensor?: string | null;
  /** What measured this row's GPU power, by the telemetry owner's word; null with no power. */
  gpu_power_sensor?: string | null;
  /**
   * The backend's sentence on why this node's GPU readings are missing (a
   * stopped or absent sampler), "" when there is nothing to say. Absent from
   * an older server.
   */
  gpu_readings_note?: string;
  /** The backend's sentence on readings dropped as outside physical limits, or "". */
  implausible_note?: string;
}

/** The fields of a `/telemetry/history` answer this rule reads. */
export interface FreshnessSource {
  last_sample_at?: number | null;
  last_sample_age_seconds?: number | null;
  latest?: LatestSample | null;
  ingest?: WorkerIngest | null;
  /** The server's verdict: is this node reporting right now? */
  reporting?: boolean;
  /** The window the server judged `reporting` on, for ageing a kept answer. */
  reporting_window_seconds?: number;
}

export type WorkerFreshness =
  /** The worker is posting and its newest reading is current. */
  | { state: "fresh"; ageSeconds: number }
  /** The worker reported before but not within the window; `since` is epoch seconds. */
  | { state: "stale"; since: number | null }
  /** The controller refused the newest post because the worker's clock is off. */
  | { state: "clock-refused"; reason: string }
  /** Nothing has ever been stored or received for this worker. */
  | { state: "never" };

const CLOCK_REFUSED_FALLBACK =
  "This worker's readings are being refused because its clock does not match this controller's.";

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Seconds since the worker's newest reading, by the receive clock when known. */
export function workerReadingAge(source: FreshnessSource | null | undefined): number | null {
  return finite(source?.ingest?.received_age_seconds) ?? finite(source?.last_sample_age_seconds);
}

/** The single freshness verdict for a worker's telemetry. */
export function workerFreshness(source: FreshnessSource | null | undefined): WorkerFreshness {
  const ingest = source?.ingest;
  if (ingest?.clock_refused) {
    return { state: "clock-refused", reason: ingest.reason || CLOCK_REFUSED_FALLBACK };
  }
  const age = workerReadingAge(source);
  const since = finite(ingest?.received_at) ?? finite(source?.last_sample_at);
  if (source?.reporting === true) return { state: "fresh", ageSeconds: age ?? 0 };
  if (age === null && since === null) return { state: "never" };
  return { state: "stale", since };
}

/**
 * A kept `/telemetry/history` answer as it reads `elapsedSeconds` after it
 * arrived. Both ages grow by the elapsed time, and the server's `reporting`
 * verdict is re-asked against the server's own `reporting_window_seconds` with
 * the age the server judged by (receive age first), so a kept answer goes
 * stale exactly when a fresh read would say so. Without the server's window
 * the verdict is not carried forward at all.
 */
export function agedFreshnessSource(source: FreshnessSource, elapsedSeconds: number): FreshnessSource {
  const received = finite(source.ingest?.received_age_seconds);
  const sampled = finite(source.last_sample_age_seconds);
  const aged: FreshnessSource = {
    ...source,
    last_sample_age_seconds: sampled === null ? sampled : sampled + elapsedSeconds,
    ingest: source.ingest
      ? { ...source.ingest, received_age_seconds: received === null ? received : received + elapsedSeconds }
      : source.ingest,
  };
  if (elapsedSeconds <= 0) return aged;
  const window = finite(source.reporting_window_seconds);
  const age = workerReadingAge(aged);
  return {
    ...aged,
    reporting: source.reporting === true && window !== null && age !== null && age <= window,
  };
}

/**
 * A worker's current value for one series: the newest raw sample, and only
 * while the reading is fresh. Null when stale, refused, never reported, or not
 * measured - never a bucket mean and never an old figure presented as current.
 */
export function currentWorkerValue(
  source: FreshnessSource | null | undefined,
  seriesKey: string,
): number | null {
  if (workerFreshness(source).state !== "fresh") return null;
  return finite(source?.latest?.values?.[seriesKey]);
}
