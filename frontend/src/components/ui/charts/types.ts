/**
 * The chart-facing contract of the Performance dashboard (VD-147).
 *
 * The backend sends base units, state words and whole sentences; these
 * primitives draw what they are given. They hold no threshold, no freshness
 * word and no state label of their own: a label for a state arrives as
 * `stateLabel`, a reason arrives as `reason`, and a number is formatted by
 * the caller's `formatValue`.
 */

/**
 * What a value is, in the backend's word (`measured`, `partial`, ...). The
 * primitives never switch on it: it is carried as `data-state` only, and the
 * words shown for it arrive as `stateLabel`.
 */
export type ValueState = string;

export type SeriesKind = "node" | "aggregate" | "layer";

/** A per-request band for one bucket. The upper edge is null when open-ended. */
export interface SeriesBand {
  p50: [number, number | null];
  p90: [number, number | null];
}

export interface ChartSeries {
  /** Internal identity. NEVER rendered, not as text and not as an attribute. */
  key: string;
  /** Always rendered, inserted as text. */
  name: string;
  kind: SeriesKind;
  /** The node's persisted colour slot. A node without one goes through the chooser. */
  slot?: 1 | 2 | 3;
  /** `null` is "no sample" and draws a GAP. It is never drawn as zero. */
  values: (number | null)[];
  /** 0..1 per bucket. Anything below 1 draws that stretch dashed and lighter. */
  coverage?: (number | null)[];
  /** The aggregate flagged partial per bucket by the backend. */
  partial?: boolean[];
  state: ValueState;
  /** The backend's words for `state`. Shown beside the name of an absent series. */
  stateLabel?: string;
  /** The backend's sentence. Shown when the series is absent (`values: []`). */
  reason: string;
  /** Tooltip only. Never drawn as a mark. */
  bands?: (SeriesBand | null)[];
  /** The backend's sentence for a bucket with no sample, per bucket. Tooltip and table only. */
  reasons?: (string | null)[];
  /**
   * A second line for the same machine (host memory beside host CPU): it keeps
   * the machine's colour and line style and is told apart by hollow rings.
   */
  marker?: "ring";
  /** The backend's words for a series drawn off until chosen ("Cluster total (off - select to show)"). */
  offLabel?: string;
}

export interface ChartEvent {
  /** Epoch seconds. */
  t: number;
  kind: string;
  label: string;
  /** A node key. Internal, like `ChartSeries.key`: never rendered. */
  node?: string;
}

/** A horizontal mark, such as a warning or critical limit the backend sent. */
export interface ChartThreshold {
  value: number;
  label: string;
}

/** One machine's value on the "beneath" line of a stat tile. */
export interface TileNodeValue {
  /** Internal. Never rendered. */
  key: string;
  name: string;
  /** Already formatted by the caller, or null when this machine has no value. */
  value: string | null;
  /** The backend's words for why there is no value. */
  stateLabel?: string;
}

/**
 * The few words these primitives own: interface chrome, not data. A state
 * label or a reason never appears here.
 */
export interface ChartLabels {
  noSample: string;
  partial: string;
  time: string;
  events: string;
  coverageColumn: string;
  typicalBand: string;
  slowestBand: string;
  orMore: string;
  noReading: string;
  marks: string;
  keyboardHint: string;
  /** `percent` arrives formatted, for example "40%". */
  coverage: (percent: string) => string;
  andMore: (count: number) => string;
  range: (low: string, high: string) => string;
  eventAt: (label: string, time: string) => string;
}

export const DEFAULT_CHART_LABELS: ChartLabels = {
  noSample: "no sample",
  partial: "partial",
  time: "Time",
  events: "Events",
  coverageColumn: "read",
  typicalBand: "Typical request",
  slowestBand: "Slowest 10%",
  orMore: "or more",
  noReading: "no reading",
  marks: "Marks",
  keyboardHint: "Left and Right move one interval, Home and End jump, Escape closes.",
  coverage: (percent) => "read for " + percent + " of this interval",
  andMore: (count) => "and " + count + " more",
  range: (low, high) => {
    // "11.1 tok/s to 14.2 tok/s" says the unit twice; say it once.
    const unit = (text: string) => text.replace(/^[-+]?[\d.,]+/, "");
    const shared = unit(low) !== "" && unit(low) === unit(high);
    return (shared ? low.slice(0, low.length - unit(low).length) : low) + " to " + high;
  },
  eventAt: (label, time) => label + " at " + time,
};

/** An absent series is present with no values, a state and a reason. */
export function isAbsent(series: ChartSeries): boolean {
  return series.values.length === 0;
}
