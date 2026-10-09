import type { ChartSeries, ChartThreshold, SeriesBand, StatTileProps, TileNodeValue } from "../components/ui/charts";
import { formatBytes, typedFromBytes } from "./format";
import {
  fillTime,
  type DashboardPanel,
  type DashboardTile,
  type TemperatureBands,
  type TemperatureMark,
  type WireBand,
} from "./performanceDashboard";

/**
 * From the dashboard's wire shapes to the chart primitives' props (VD-147 S6).
 *
 * Pure, so every honesty rule is tested without a screen: a `null` stays a
 * gap, a series with `values: []` stays present with its state label and
 * reason, colour follows the machine's persisted slot (never its rank), a node
 * key is used to find a slot and is never rendered. Every word is the
 * backend's; the only thing made here is a formatted number, switched on the
 * API's base-unit identifier, with the suffix the backend sends in
 * `labels.units`.
 */

const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

/** Unit identifier to suffix, from the range layer's `labels.units`. */
export type Units = Readonly<Record<string, string>>;

/** The number alone, for a tile's value row (the suffix sits beside it). */
export function formatNumber(unit: string, value: number): string {
  if (unit === "ratio") return String(Math.round(value * 100));
  if (unit === "celsius" || unit === "percent") return String(Math.round(value));
  if (unit === "bytes") return formatBytes(value);
  return value.toFixed(1);
}

/** A value with the backend's suffix, for tooltips, tables and the machines beneath a tile. */
export function formatValue(unit: string, units: Units): (value: number) => string {
  return (value) => {
    if (unit === "bytes") return formatBytes(value);
    const suffix = units[unit] ?? "";
    const number = formatNumber(unit, value);
    // A percentage sign sits on the number; any other suffix after a space.
    return suffix === "%" ? number + suffix : suffix ? number + " " + suffix : number;
  };
}

/** Axis ticks: the bare number (a ratio as a percentage; bytes in GB). */
export function formatTick(unit: string): (value: number) => string {
  if (unit === "ratio") return (value) => String(Math.round(value * 100));
  if (unit === "bytes") return (value) => String(typedFromBytes(value, "GB"));
  return (value) => String(Number(value.toFixed(2)));
}

/** A slot the chart can colour (1-3); a fourth machine goes through the chooser. */
function chartSlot(slot: unknown): 1 | 2 | 3 | undefined {
  return slot === 1 || slot === 2 || slot === 3 ? slot : undefined;
}

/** A (slowest, fastest) wire band as the chart's [low, high or open]; null when the low end is open. */
function band(edges: [number | null, number | null] | null | undefined): [number, number | null] | null {
  if (!edges || !finite(edges[0])) return null;
  return [edges[0], finite(edges[1]) ? edges[1] : null];
}

function seriesBands(wire: WireBand[] | null | undefined): (SeriesBand | null)[] | undefined {
  if (!Array.isArray(wire)) return undefined;
  return wire.map((one) => {
    if (!one || !("p50" in one)) return null;
    const [p50, p90] = [band(one.p50), band(one.p90)];
    return p50 && p90 ? { p50, p90 } : null;
  });
}

export interface SeriesContext {
  /** Each machine's persisted colour slot, by its (internal) key. */
  slots: ReadonlyMap<string, number | null>;
}

/** One panel's series as the chart draws them: names, state labels and reasons as sent. */
export function panelSeries(panel: DashboardPanel, context: SeriesContext): ChartSeries[] {
  const wire = Array.isArray(panel.series) ? panel.series : [];
  const coloured = new Set<number>();
  const shaped = wire.map((one): ChartSeries => {
    // A host line carries its slot; a serving line is matched to its machine by key.
    const slot = chartSlot(finite(one.slot) ? one.slot : context.slots.get(one.key));
    // A machine's second line in one panel (its memory beside its CPU) keeps its colour and gets rings.
    const second = one.kind === "node" && slot !== undefined && coloured.has(slot);
    if (slot !== undefined) coloured.add(slot);
    return {
      key: one.key,
      name: one.name ?? "",
      kind: one.kind === "layer" ? "layer" : one.kind === "aggregate" ? "aggregate" : "node",
      slot,
      values: Array.isArray(one.values) ? one.values.map((value) => (finite(value) ? value : null)) : [],
      coverage: one.coverage,
      partial: one.partial,
      state: one.state ?? "",
      stateLabel: one.state_label || undefined,
      reason: one.reason ?? "",
      bands: seriesBands(one.bands),
      reasons: one.bucket_reasons,
      marker: second ? "ring" : undefined,
      offLabel: one.off_label || undefined,
    };
  });
  // Without a cluster line the panel carries the bands itself (one machine serving).
  const bands = seriesBands(panel.bands);
  if (bands && shaped.length && !shaped.some((one) => one.bands)) shaped[0] = { ...shaped[0], bands };
  return shaped;
}

/** The marks to draw, with the backend's words; a mark the backend gave no words is not drawn. */
export function thresholdMarks(thresholds: DashboardPanel["thresholds"] | TemperatureBands | undefined): ChartThreshold[] {
  if (!thresholds) return [];
  const marks: TemperatureMark[] = Array.isArray(thresholds)
    ? thresholds
    : thresholds.marks ?? [
      { value: thresholds.warning_c, label: thresholds.labels?.warning },
      { value: thresholds.critical_c, label: thresholds.labels?.critical },
    ];
  return marks.filter((mark): mark is { value: number; label: string } => finite(mark.value) && Boolean(mark.label))
    .map((mark) => ({ value: mark.value, label: mark.label }));
}

/** When machines have different bands, each machine's, in the backend's words, for the caption. */
export function bandsByNode(panel: DashboardPanel): string {
  const sets = panel.bands_by_node;
  if (!Array.isArray(sets)) return "";
  const drawn = (panel.series ?? []).filter((one) => Array.isArray(one.values) && one.values.length > 0);
  return sets.map((bands, index) => {
    const marks = thresholdMarks(bands).map((mark) => mark.label).join(", ");
    return marks && drawn[index]?.name ? drawn[index].name + ": " + marks : "";
  }).filter(Boolean).join("; ");
}

/** The machines a by-node panel can be narrowed to, in the backend's order and words. */
export function panelMachines(panel: DashboardPanel): { key: string; name: string }[] {
  const seen = new Map<string, string>();
  for (const layer of Object.values(panel.by_node ?? {})) {
    for (const line of Array.isArray(layer) ? layer : []) if (!seen.has(line.key)) seen.set(line.key, line.name ?? "");
  }
  return [...seen].map(([key, name]) => ({ key, name }));
}

/**
 * One machine's layers (spec 1.2 P3): each summed layer, redrawn from that
 * machine's own line in `by_node`. A layer the machine has no line for is
 * drawn as absent, keeping the layer's own state label and reason.
 */
export function machineLayers(panel: DashboardPanel, machine: string): DashboardPanel {
  return {
    ...panel,
    series: panel.series.map((layer) => {
      const own = panel.by_node?.[layer.key]?.find((line) => line.key === machine);
      return own
        ? { ...layer, values: own.values, coverage: own.coverage, partial: own.partial, bucket_reasons: own.bucket_reasons,
            state: own.state ?? layer.state, state_label: own.state_label ?? layer.state_label, reason: own.reason ?? layer.reason }
        : { ...layer, values: [] };
    }),
  };
}

// --------------------------------------------------------------------------
// Tiles
// --------------------------------------------------------------------------

/** One tile's props: title, caption, state label and reason as sent; the number formatted here. */
export function tileProps(tile: DashboardTile | undefined, units: Units, meter: "none" | "share" | "limits"): StatTileProps {
  if (!tile || typeof tile !== "object") return { title: "", value: null, state: "" };
  const unit = tile.unit ?? "";
  const format = formatValue(unit, units);
  const value = tile.cluster && finite(tile.cluster.value) ? tile.cluster.value : null;
  const nodes: TileNodeValue[] = (tile.nodes ?? []).map((node, index) => ({
    key: String(index),
    name: node.name ?? "",
    value: finite(node.value) ? format(node.value) : null,
    stateLabel: node.state_label || undefined,
  }));
  const excluded = (tile.excluded ?? []).map((item) => (typeof item === "string" ? item : item.reason)).filter(Boolean);
  const bar = (() => {
    if (value === null || meter === "none") return undefined;
    if (meter === "share") return { value: unit === "ratio" ? value * 100 : value, max: 100, valueText: format(value) };
    const marks = thresholdMarks(tile.cluster?.bands);
    const top = Math.max(110, value + 10, ...marks.map((mark) => mark.value + 10));
    return { value, max: top, valueText: format(value), marks, marksSource: tile.cluster?.bands?.source };
  })();
  return {
    title: tile.title ?? "",
    value: value === null ? null : formatNumber(unit, value),
    unit: units[unit] ?? "",
    state: tile.state ?? "",
    stateLabel: tile.state_label || undefined,
    reason: [tile.reason, ...excluded].filter(Boolean).join(" "),
    caption: tile.caption ? fillTime(tile.caption, tile.scope_since, "{since}") : "",
    meter: bar,
    nodes,
  };
}

// --------------------------------------------------------------------------
// The numbers a chart card carries (VD-200, the ClusterPerformance boards)
// --------------------------------------------------------------------------

/** A series' newest sample and the bucket it is in, or null when it has none. */
export function newestSample(values: readonly (number | null)[]): { value: number; index: number } | null {
  for (let index = values.length - 1; index >= 0; index -= 1) {
    const value = values[index];
    if (finite(value)) return { value, index };
  }
  return null;
}

/** The smallest, the mean and the largest sample of every column together; null when none has a sample. */
export function sampleStats(columns: readonly (readonly (number | null)[])[]): { min: number; avg: number; peak: number } | null {
  const samples = columns.flatMap((column) => column.filter(finite));
  if (!samples.length) return null;
  return {
    min: Math.min(...samples),
    avg: samples.reduce((sum, value) => sum + value, 0) / samples.length,
    peak: Math.max(...samples),
  };
}

/**
 * The machine a line belongs to: a host panel keys a machine's second line
 * `<machine>:<metric>` ("controller:memory"), every other line is the
 * machine's own key. Internal, like the key: used to group, never rendered.
 */
export function machineOf(seriesKey: string): string {
  return seriesKey.split(":")[0];
}
