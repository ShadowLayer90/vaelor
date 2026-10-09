import { useLayoutEffect, useRef, useState } from "react";
import { formatPercent } from "../../../lib/format";
import { LineKey, type SeriesMark } from "./ChartLegend";
import type { ChartLabels, ChartSeries } from "./types";

/** What one series says at one bucket: the same words for eye and ear. */
export interface ReadoutRow {
  name: string;
  mark: SeriesMark;
  /** The formatted sample, or the "no sample" words. Never a zero for a gap. */
  value: string;
  /** Coverage and the partial flag, when the series carries them. */
  notes: string[];
  /** The per-request bands, when the series carries them. */
  bands: string[];
  /** A ring-marked companion line. */
  ring?: boolean;
}

type Words = Pick<ChartLabels, "noSample" | "partial" | "coverage" | "typicalBand" | "slowestBand" | "orMore" | "range">;

function bandText(edges: [number, number | null], formatValue: (value: number) => string, words: Words): string {
  return edges[1] === null
    ? formatValue(edges[0]) + " " + words.orMore
    : words.range(formatValue(edges[0]), formatValue(edges[1]));
}

/**
 * One sample as text. The tooltip, the live region and the table all go
 * through this one function, so a gap is the same words everywhere and none
 * of them can turn it into a zero.
 */
export function sampleText(
  sample: number | null | undefined,
  formatValue: (value: number) => string,
  noSample: string,
): string {
  return typeof sample === "number" && Number.isFinite(sample) ? formatValue(sample) : noSample;
}

/** The readout of one bucket: what the tooltip shows and the live region says. */
export function readBucket(
  drawn: readonly { series: ChartSeries; mark: SeriesMark }[],
  index: number,
  formatValue: (value: number) => string,
  words: Words,
): ReadoutRow[] {
  return drawn.map(({ series, mark }) => {
    const notes: string[] = [];
    const coverage = series.coverage?.[index];
    if (typeof coverage === "number" && coverage < 1) notes.push(words.coverage(formatPercent(coverage * 100)));
    if (series.partial?.[index]) notes.push(words.partial);
    const reason = series.reasons?.[index];
    if (reason && !(typeof series.values[index] === "number" && Number.isFinite(series.values[index]))) notes.push(reason);
    // A band describes requests in a bucket that has a sample; under "no sample" it would read as one.
    const sampled = typeof series.values[index] === "number" && Number.isFinite(series.values[index]);
    const band = sampled ? series.bands?.[index] : null;
    const bands = band
      ? [words.typicalBand + " " + bandText(band.p50, formatValue, words), words.slowestBand + " " + bandText(band.p90, formatValue, words)]
      : [];
    return { name: series.name, mark, value: sampleText(series.values[index], formatValue, words.noSample), notes, bands, ring: series.marker === "ring" };
  });
}

/** The same readout as one sentence, for the live region. */
export function readoutText(time: string, rows: readonly ReadoutRow[], events: readonly string[]): string {
  const parts = rows.map((row) => [row.name + " " + row.value, ...row.notes, ...row.bands].join(", "));
  return [time, ...parts, ...events].join(". ") + ".";
}

export interface ChartTooltipProps {
  time: string;
  rows: ReadoutRow[];
  events: string[];
  /** How far across the plot the crosshair is, 0..1. */
  fraction: number;
}

function Rows({ rows, bands, notes = true }: { rows: ReadoutRow[]; bands: boolean; notes?: boolean }) {
  return (
    <ul className="chart-tooltip__rows">
      {rows.map((row, position) => (
        <li className="chart-tooltip__row" key={position}>
          <strong className="chart-tooltip__value">{row.value}</strong>
          <span className="chart-tooltip__series">
            <LineKey mark={row.mark} ring={row.ring} />
            <span className="chart-tooltip__name">{row.name}</span>
          </span>
          {notes && row.notes.map((note, at) => <span className="chart-tooltip__note" key={at}>{note}</span>)}
          {bands && row.bands.map((note, at) => (
            <span className="chart-tooltip__note chart-tooltip__band" key={"band-" + at}>{note}</span>
          ))}
        </li>
      ))}
    </ul>
  );
}

/**
 * One tooltip for every series at the crosshair, floating inside the plot.
 * The value leads and the name follows, because here the reader already has
 * the series and wants the number. Names are inserted as text.
 *
 * It never leaves the plot: the stylesheet caps it to the plot's height and to
 * the half of the plot it opens into. Events come straight after the time, so
 * a cap never cuts one off; one that still does not fit drops its band notes,
 * then its coverage notes, which the live region and the table keep. The
 * chart mounts it once per bucket, so each bucket starts with everything.
 */
export function ChartTooltip({ time, rows, events, fraction }: ChartTooltipProps) {
  const offset = (Math.min(1, Math.max(0, fraction)) * 100).toFixed(2) + "%";
  const side = fraction > 0.5 ? { right: "calc(100% - " + offset + ")" } : { left: offset };
  const ref = useRef<HTMLDivElement>(null);
  const [fit, setFit] = useState<"full" | "no-bands" | "no-notes" | "columns" | "keys">("full");
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element || element.scrollHeight <= element.clientHeight + 1) return;
    if (fit === "full") setFit("no-bands");
    else if (fit === "no-bands") setFit("no-notes");
    // Still too tall (four machines on a short plot): two rows to a line, so no row is ever cut.
    else if (fit === "no-notes") setFit("columns");
    // A short plot and many machines: each value beside its line key, the names left to the legend,
    // the live region and the table.
    else if (fit === "columns") setFit("keys");
  }, [fit]);
  return (
    <div className="chart-tooltip" data-fit={fit} data-side={fraction > 0.5 ? "before" : "after"} ref={ref} style={side}>
      <p className="chart-tooltip__time">{time}</p>
      {events.map((label, position) => <p className="chart-tooltip__event" key={position}>{label}</p>)}
      <Rows bands={fit === "full"} notes={fit === "full" || fit === "no-bands"} rows={rows} />
    </div>
  );
}

export interface ChartReadoutProps {
  time: string;
  rows: ReadoutRow[];
  events: string[];
  /** Keep one events line even when this bucket has none, so the strip never changes height. */
  reserveEvents: boolean;
}

/**
 * The narrow chart's readout: a strip under the plot instead of a tooltip on
 * it, so the drawing is never covered. It always shows a bucket (the one under
 * the crosshair, else the newest), one line per series, so it has one height
 * and nothing below it moves. The bands are left to the live region and the
 * table.
 */
export function ChartReadout({ time, rows, events, reserveEvents }: ChartReadoutProps) {
  return (
    <div className="chart-readout">
      <p className="chart-tooltip__time">{time}</p>
      <Rows bands={false} rows={rows} />
      {reserveEvents && <p className="chart-tooltip__event">{events.join(", ") || "\u00a0"}</p>}
    </div>
  );
}
