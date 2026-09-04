import type { Metrics, TelemetrySample } from "../types";

/**
 * A reading, or nothing.
 *
 * The previous helper returned `0` for an absent sensor, so every consumer of
 * a metric this host does not emit received a real-looking measurement: a
 * storage progressbar announcing "Storage 0%", a `Fan — 0 RPM` channel on a
 * machine with no readable fan, and flat-at-zero sparklines indistinguishable
 * from a genuine idle trace. "Measured zero" and "not measured" are different
 * facts and the type now says so, which forces every caller to decide.
 */
export function metricNumber(metrics: Metrics, key: string): number | null {
  const value = metrics[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** A rolling window in which missing samples stay missing. */
export function metricSeries(history: TelemetrySample[], key: string): Array<number | null> {
  return history.map((sample) => metricNumber(sample.metrics, key));
}

/**
 * Sum of the readings that exist, or `null` when none of them do. Adding a
 * missing upload rate to a missing download rate must not produce `0 B/s`.
 */
export function metricSum(metrics: Metrics, keys: string[]): number | null {
  const values = keys.map((key) => metricNumber(metrics, key)).filter((value): value is number => value !== null);
  return values.length ? values.reduce((total, value) => total + value, 0) : null;
}

export function metricSumSeries(history: TelemetrySample[], keys: string[]): Array<number | null> {
  return history.map((sample) => metricSum(sample.metrics, keys));
}
