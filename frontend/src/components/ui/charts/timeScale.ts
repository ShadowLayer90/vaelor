/**
 * Pure geometry for the time-series charts: scales, ticks, gap segmentation,
 * stacking and time labels. No React and no DOM, so every rule here is
 * testable as arithmetic.
 *
 * The rule the rest of the chart leans on: a `null` is "no sample". It ends a
 * segment and starts a gap. Nothing in this module turns it into a number.
 */
import type { ChartSeries } from "./types";

/** The inner plot rectangle, in the SVG's own pixels. */
export interface PlotBox { left: number; right: number; top: number; bottom: number }

/** A drawn stretch of one series: bucket indexes `from..to`, both included. */
export interface Segment { from: number; to: number; partial: boolean }

export interface TimeTick { t: number; label: string }

const isSample = (value: number | null | undefined): value is number =>
  typeof value === "number" && Number.isFinite(value);

/** The backend decides what is partial; the chart only reads what it carries. */
export function bucketIsPartial(series: Pick<ChartSeries, "coverage" | "partial">, index: number): boolean {
  const coverage = series.coverage?.[index];
  return Boolean(series.partial?.[index]) || (isSample(coverage) && coverage < 1);
}

/**
 * Split a series into drawn segments. A run of samples between two gaps is
 * one or more segments; inside a run the line changes style wherever the
 * "partial" reading changes. The stretch between two buckets is partial when
 * either end is. A sample with a gap on both sides is a one-bucket segment,
 * which the chart draws as a dot so a lone reading is not invisible.
 */
export function segmentSeries(values: readonly (number | null)[], partialAt: (index: number) => boolean): Segment[] {
  const segments: Segment[] = [];
  let index = 0;
  while (index < values.length) {
    if (!isSample(values[index])) { index += 1; continue; }
    let end = index;
    while (end + 1 < values.length && isSample(values[end + 1])) end += 1;
    let from = index;
    let partial = partialAt(index) || (end > index && partialAt(index + 1));
    for (let at = index + 1; at < end; at += 1) {
      const next = partialAt(at) || partialAt(at + 1);
      if (next === partial) continue;
      segments.push({ from, to: at, partial });
      from = at;
      partial = next;
    }
    segments.push({ from, to: end, partial });
    index = end + 1;
  }
  return segments;
}

/** A 1, 2 or 5 times a power of ten, at or above `rough`. */
function niceStep(rough: number): number {
  const power = 10 ** Math.floor(Math.log10(rough));
  const unit = rough / power;
  return (unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10) * power;
}

const tidy = (value: number) => Number(value.toFixed(10));

/** Round tick values inside `[min, max]`, about `target` of them. */
export function niceTicks(min: number, max: number, target = 4): number[] {
  if (!(max > min)) return [min];
  const step = niceStep((max - min) / Math.max(1, target));
  const ticks: number[] = [];
  for (let value = Math.ceil(tidy(min / step)) * step; value <= max + step * 1e-9; value += step) {
    ticks.push(tidy(value));
  }
  return ticks;
}

/**
 * One y axis. It starts at `yMin` (0 for rates, power and percentages; the
 * caller passes 20 for temperature) and ends on a round value at or above
 * the largest drawn sample and the largest threshold mark; an explicit
 * `yMax` is kept as the top while every sample fits under it.
 *
 * The axis always holds every drawn sample. A reading below `yMin` or above
 * `yMax` moves that end out to a round value past it, because a point pinned
 * to the edge would show a number the table does not say (S-30).
 */
export function valueDomain(
  columns: readonly (readonly (number | null)[])[],
  yMin = 0,
  yMax?: number,
  marks: readonly number[] = [],
): [number, number] {
  let low = yMin;
  let top = yMin;
  for (const column of columns) {
    for (const value of column) {
      if (!isSample(value)) continue;
      if (value < low) low = value;
      if (value > top) top = value;
    }
  }
  const fixedTop = isSample(yMax) && yMax > yMin && top <= yMax;
  if (fixedTop && low >= yMin) return [yMin, yMax as number];
  for (const mark of marks) if (mark > top) top = mark;
  if (fixedTop) top = yMax as number;
  if (top <= low) return [low, low + 1];
  const step = niceStep((top - low) / 4);
  const bottom = low < yMin ? tidy(Math.floor(tidy(low / step)) * step) : yMin;
  const end = fixedTop ? top : tidy(bottom + Math.ceil(tidy((top - bottom) / step)) * step);
  return [bottom, end];
}

/** The y of a value. The domain holds every drawn sample, so nothing is pinned to an edge here. */
export function valueY(value: number, domain: readonly [number, number], box: PlotBox): number {
  const fraction = (value - domain[0]) / (domain[1] - domain[0]);
  return box.bottom - fraction * (box.bottom - box.top);
}

/** The x of bucket `index`. Buckets sit at `start + index * step`. */
export function bucketX(index: number, count: number, box: PlotBox): number {
  if (count <= 1) return (box.left + box.right) / 2;
  return box.left + (index / (count - 1)) * (box.right - box.left);
}

/** The bucket a pointer x snaps to. */
export function nearestBucket(x: number, count: number, box: PlotBox): number {
  if (count <= 1 || box.right <= box.left) return 0;
  const raw = Math.round(((x - box.left) / (box.right - box.left)) * (count - 1));
  return Math.min(count - 1, Math.max(0, raw));
}

/** The bucket an event falls in, or null when it is outside the range. */
export function eventBucket(t: number, start: number, step: number, count: number): number | null {
  if (!(step > 0)) return null;
  const index = Math.floor((t - start) / step);
  return index >= 0 && index < count ? index : null;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAY_SECONDS = 86_400;
const two = (value: number) => String(value).padStart(2, "0");

/** "14:20" in a day, "Sep 30 14:20" past one; seconds when buckets are under a minute apart. */
export function formatTimeLabel(epochSeconds: number, spanSeconds: number, stepSeconds = 60): string {
  const date = new Date(epochSeconds * 1000);
  const clock = two(date.getHours()) + ":" + two(date.getMinutes()) + (stepSeconds < 60 ? ":" + two(date.getSeconds()) : "");
  return spanSeconds > DAY_SECONDS ? MONTHS[date.getMonth()] + " " + date.getDate() + " " + clock : clock;
}

const TICK_SECONDS = [60, 120, 300, 600, 900, 1800, 3600, 7200, 10_800, 21_600, 43_200, DAY_SECONDS, 2 * DAY_SECONDS];

/** Time-axis ticks on round local clock times. */
export function timeTicks(start: number, step: number, count: number, target = 5): TimeTick[] {
  const span = step * Math.max(0, count - 1);
  if (!(span > 0)) return count > 0 ? [{ t: start, label: formatTimeLabel(start, 0) }] : [];
  const every = TICK_SECONDS.find((seconds) => seconds >= span / target) ?? TICK_SECONDS[TICK_SECONDS.length - 1];
  const offset = new Date(start * 1000).getTimezoneOffset() * 60;
  const ticks: TimeTick[] = [];
  for (let t = Math.ceil((start - offset) / every) * every + offset; t <= start + span; t += every) {
    ticks.push({ t, label: formatTimeLabel(t, span) });
  }
  return ticks;
}

export function timeX(t: number, start: number, step: number, count: number, box: PlotBox): number {
  const span = step * (count - 1);
  if (!(span > 0)) return (box.left + box.right) / 2;
  return box.left + ((t - start) / span) * (box.right - box.left);
}

/** The smallest and largest drawn sample, or null when there is none. */
export function sampleExtent(values: readonly (number | null)[]): [number, number] | null {
  const samples = values.filter(isSample);
  return samples.length ? [Math.min(...samples), Math.max(...samples)] : null;
}
