import { useEffect, useId, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import { Button } from "./ui";
import { DEFAULT_CHART_LABELS, SeriesTable, type ChartSeries, type ChartThreshold, type SeriesMark } from "./ui/charts";
import { ChartTooltip, readBucket, readoutText } from "./ui/charts/ChartTooltip";
import { areaPath, linePath, stackLayers } from "./ui/charts/chartPaths";
import {
  bucketIsPartial, bucketX, eventBucket, formatTimeLabel, nearestBucket, niceTicks, segmentSeries, timeTicks, timeX, valueDomain, valueY,
  type PlotBox,
} from "./ui/charts/timeScale";
import { FreshnessBadge } from "./DashboardStatus";
import {
  bandsByNode, formatTick, formatValue, machineLayers, machineOf, newestSample, panelMachines, panelSeries, sampleStats, thresholdMarks,
  type SeriesContext, type Units,
} from "../lib/dashboardSeries";
import type { DashboardPanel, Freshness, NowLayer, RangeLayer } from "../lib/performanceDashboard";
import type { ClusterMode } from "../lib/clusterMode";

/**
 * The chart grid (VD-200, the ClusterPerformance and ClusterPerformanceCharts
 * boards): eight cards in a four-column grid, and Advanced's three more. Every
 * card is one size, built from fixed-height regions so the plots line up: the
 * title and freshness, the scope, the current value large, min · avg · peak,
 * the machine chips, one legend row per line with that line's newest value,
 * the plot with labelled gridlines, times and threshold marks, and one line
 * of notes. A line with no sample says "Not read" and draws nothing.
 *
 * Titles, names, state labels, reasons and marks are the backend's; the keys
 * here only place the panels. The only words this module owns are the chart's
 * own chrome ("min", "avg", "peak", "now", "total", "Not read", "All").
 */
const PANELS: { key: string; advanced?: boolean; stacked?: boolean; floor?: boolean }[] = [
  { key: "output_throughput" },
  { key: "gpu_temperature" },
  { key: "prompt_tokens", stacked: true },
  { key: "gpu_power" },
  { key: "decode_speed", floor: true },
  { key: "kv_cache" },
  { key: "host_cpu_memory" },
  { key: "gpu_busy" },
  { key: "package_power", advanced: true },
  { key: "gtt_used", advanced: true },
  { key: "gpu_edge_temperature", advanced: true },
];

/** A share of 100 % has a fixed top, unless a value goes past it. */
const TOP: Record<string, number> = { percent: 100, ratio: 1 };
/** The plot's own height: the owner's readable-charts floor (124 px), the same on every card. */
const PLOT_HEIGHT = 124;
/** Used until the plot has been measured, and in a test environment that cannot measure. */
const FALLBACK_WIDTH = 300;
const BOX = { top: 8, bottomGap: 2 };
/** Two threshold labels closer than this print on either side of their lines. */
const LABEL_ROOM = 16;
const LIVE_THROTTLE_MS = 250;
const NOT_READ = "Not read";
const NODE_MARKS: SeriesMark[] = ["slot-1", "slot-2", "slot-3"];
const LAYER_MARKS: SeriesMark[] = ["layer-1", "layer-2", "layer-3"];

/**
 * Each series' mark. Colour follows the machine's slot, never its rank: the
 * controller (slot 1) is the accent, the first worker blue. A machine with no
 * free slot is "extra": listed, and drawn only when its chip is chosen, never
 * in a generated hue. A machine's second line (memory beside CPU) keeps its
 * machine's mark and is drawn dashed.
 */
function assignMarks(series: readonly ChartSeries[]): SeriesMark[] {
  const taken = new Set<number>();
  let layer = 0;
  return series.map((one) => {
    if (one.kind === "aggregate") return "aggregate";
    if (one.kind === "layer") return LAYER_MARKS[Math.min(layer++, LAYER_MARKS.length - 1)];
    if (one.marker === "ring" && one.slot) return NODE_MARKS[one.slot - 1];
    if (one.slot && !taken.has(one.slot)) {
      taken.add(one.slot);
      return NODE_MARKS[one.slot - 1];
    }
    return "extra";
  });
}

const hasSample = (one: ChartSeries) => newestSample(one.values) !== null;

/** The legend's key: a short stroke in the line's mark, or a hollow ring for a line with nothing to draw. */
function Swatch({ mark, second = false, empty = false }: { mark: SeriesMark; second?: boolean; empty?: boolean }) {
  if (empty) return <span aria-hidden="true" className="perf-swatch perf-swatch--empty" />;
  return <span aria-hidden="true" className={"perf-swatch perf-swatch--" + mark + (second ? " perf-swatch--second" : "")} />;
}

interface FrameProps {
  title: string;
  badge?: Freshness;
  subtitle: string;
  value: string | null;
  valueNote: string;
  stats: { min: string; avg: string; peak: string } | null;
  chips?: ReactNode;
  legend?: ReactNode;
  plot: ReactNode;
  foot?: ReactNode;
  state: string;
  busy?: boolean;
  describedBy?: string;
}

/** The fixed regions every card has, filled or not, so every card is one size and every plot lines up. */
function CardFrame({ title, badge, subtitle, value, valueNote, stats, chips, legend, plot, foot, state, busy, describedBy }: FrameProps) {
  return (
    <figure aria-busy={busy ? "true" : undefined} aria-describedby={describedBy} className="perf-chart" data-state={state}>
      <header className="perf-chart__head">
        <h3 className="perf-chart__title" title={title}>{title}</h3>
        <span className="chart__badge"><FreshnessBadge freshness={badge} /></span>
      </header>
      <p className="perf-chart__sub" title={subtitle}>{subtitle}</p>
      <p className="perf-chart__value">
        {value === null
          ? <span className="perf-chart__number perf-chart__number--unread">{NOT_READ}</span>
          : <span className="perf-chart__number">{value}</span>}
        {valueNote && <span className="perf-chart__value-note" title={valueNote}>{valueNote}</span>}
      </p>
      <dl className="perf-chart__stats">
        {(["min", "avg", "peak"] as const).map((word) => (
          <div key={word}><dt>{word}</dt><dd>{stats ? stats[word] : "—"}</dd></div>
        ))}
      </dl>
      <div className="perf-chart__chips">{chips}</div>
      <div className="perf-chart__legend">{legend}</div>
      {plot}
      <div className="perf-chart__foot">{foot}</div>
    </figure>
  );
}

/** A panel with nothing to draw (or not sent yet): the same card, its words in the plot's place. */
function PanelMessage({ panel, badge, units, lead, unread = "" }: {
  panel: DashboardPanel | undefined; badge?: Freshness; units: Units; lead: string;
  /** Why the range layer could not be read, when it was not: the card then says so, never "Reading…" for ever. */
  unread?: string;
}) {
  const unit = panel?.unit ? units[panel.unit] ?? "" : "";
  return (
    <CardFrame
      badge={badge}
      busy={!panel && !unread}
      plot={(
        <div className="perf-chart__plotarea perf-chart__plotarea--message">
          {panel ? (
            <p className="chart__message">
              {panel.state_label && <strong>{panel.state_label}</strong>}
              {panel.state_label && panel.reason ? " " : ""}
              {panel.reason}
            </p>
          ) : unread
            ? <p className="chart__message"><strong>{NOT_READ}</strong> {unread}</p>
            : <p className="chart__message">Reading…</p>}
        </div>
      )}
      state={panel?.state ?? ""}
      stats={null}
      subtitle={[panel?.scope || lead, unit].filter(Boolean).join(" · ")}
      title={panel?.title ?? ""}
      value={null}
      valueNote=""
    />
  );
}

/** The machines a panel can be narrowed to: its by-node lines, else its machines' own lines, named as the strip names them. */
function machinesOf(sent: DashboardPanel, series: ChartSeries[], names: ReadonlyMap<string, string>): { key: string; name: string }[] {
  if (sent.by_node) return panelMachines(sent);
  const seen = new Map<string, string>();
  for (const one of series) {
    if (one.kind !== "node") continue;
    const machine = machineOf(one.key);
    if (!seen.has(machine)) seen.set(machine, names.get(machine) ?? one.name);
  }
  return [...seen].map(([key, name]) => ({ key, name }));
}

/** What the live region says for the bucket under the crosshair: the tooltip's words, as one sentence. */
function liveText(
  active: number | null, drawn: { series: ChartSeries; mark: SeriesMark }[], format: (value: number) => string,
  time: string, events: string[],
): string {
  if (active === null) return "";
  return readoutText(time, readBucket(drawn, active, format, DEFAULT_CHART_LABELS), events);
}

function Panel({ spec, panel: sent, range, badge, context, units, names, floor, unread }: {
  spec: (typeof PANELS)[number]; panel: DashboardPanel | undefined; range: RangeLayer | null; badge?: Freshness;
  context: SeriesContext; units: Units; names: ReadonlyMap<string, string>; floor: ChartThreshold | null; unread: string;
}) {
  const [picked, setPicked] = useState<string | null>(null);
  // The lines the reader turned over from how the backend sends them (a total is sent off until chosen).
  const [toggled, setToggled] = useState<ReadonlySet<string>>(() => new Set());
  const [cursor, setCursor] = useState<number | null>(null);
  const [asTable, setAsTable] = useState(false);
  const [width, setWidth] = useState(FALLBACK_WIDTH);
  const [live, setLive] = useState("");
  const lastSaid = useRef(0);
  const plotRef = useRef<HTMLDivElement>(null);
  const footId = "perf-chart-" + useId().replaceAll(":", "");

  const empty = !range || !sent || !Array.isArray(sent.series) || sent.series.length === 0;
  useEffect(() => {
    const element = plotRef.current;
    if (!element || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width > 0) setWidth(Math.round(entry.contentRect.width));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [asTable, empty]);

  const offByDefault = new Set(sent?.hidden_by_default ?? []);
  const hidden = { has: (key: string) => offByDefault.has(key) !== toggled.has(key) };
  const allSeries = empty ? [] : panelSeries(sent, context);
  const machines = empty ? [] : machinesOf(sent, allSeries, names);
  // Chips only when there is more than one machine to tell apart; a machine that left falls back to All.
  const chosen = machines.length > 1 && machines.some((machine) => machine.key === picked) ? picked : null;
  const shown = empty ? null : chosen !== null && sent.by_node ? machineLayers(sent, chosen) : sent;
  const series = shown ? panelSeries(shown, context) : [];
  const marks = assignMarks(series);
  const count = Math.max(0, ...series.map((one) => one.values.length));
  const step = range?.step ?? 0;
  const start = range?.start ?? 0;
  const span = step * Math.max(0, count - 1);
  const timeOf = (index: number) => formatTimeLabel(start + index * step, span, step);
  const active = cursor !== null && cursor < count ? cursor : null;
  const unit = shown?.unit ?? "";
  const format = formatValue(unit, units);

  const inScope = (one: ChartSeries) => chosen === null || Boolean(sent?.by_node) || (one.kind === "node" && machineOf(one.key) === chosen);
  const scopedAt = series.map((_, at) => at).filter((at) => inScope(series[at]));
  // Under All a machine with no colour slot is listed, and drawn only once its chip is chosen.
  const drawnAt = scopedAt.filter((at) => hasSample(series[at]) && !hidden.has(series[at].key) && (marks[at] !== "extra" || chosen !== null));
  const stackedAt = spec.stacked ? drawnAt.filter((at) => series[at].kind === "layer") : [];
  const stack = stackLayers(stackedAt.map((at) => series[at].values));
  const drawn = drawnAt.map((at) => {
    const layer = stackedAt.indexOf(at);
    return { series: series[at], mark: marks[at], upper: layer < 0 ? series[at].values : stack[layer].upper, lower: layer < 0 ? undefined : stack[layer].lower };
  });
  const eventsAt = new Map<number, string[]>();
  for (const event of range?.events ?? []) {
    const bucket = eventBucket(event.t, start, step, count);
    if (bucket !== null) eventsAt.set(bucket, [...(eventsAt.get(bucket) ?? []), event.label]);
  }

  const said = liveText(active, drawn, format, active === null ? "" : timeOf(active), active === null ? [] : eventsAt.get(active) ?? []);
  useEffect(() => {
    // Closing the tooltip clears the region at once and does not use up the throttle.
    if (said === "") { setLive(""); return undefined; }
    const wait = lastSaid.current + LIVE_THROTTLE_MS - Date.now();
    const say = () => { lastSaid.current = Date.now(); setLive(said); };
    if (wait <= 0) { say(); return undefined; }
    const timer = setTimeout(say, wait);
    return () => clearTimeout(timer);
  }, [said]);

  if (empty || !shown || !range) {
    return <PanelMessage badge={badge} lead={range?.caption_lead ?? ""} panel={range ? sent : undefined} units={units} unread={range ? "" : unread} />;
  }

  const suffix = unit === "bytes" ? "GB" : units[unit] ?? "";
  // The headline: the cluster line when the backend sent one, the stacked total, else the first machine's line.
  const aggregate = scopedAt.map((at) => series[at]).find((one) => one.kind === "aggregate" && hasSample(one));
  const total = stack.length ? stack[stack.length - 1].upper : null;
  const primaries = scopedAt.filter((at) => series[at].kind === "node" && series[at].marker !== "ring" && hasSample(series[at])
    && (chosen !== null || marks[at] !== "extra")).map((at) => series[at]);
  const basis: { name: string; values: (number | null)[] }[] = aggregate
    ? [aggregate]
    : total && newestSample(total) ? [{ name: "total", values: total }] : primaries;
  const headline = basis.length ? newestSample(basis[0].values) : null;
  const stats = sampleStats(basis.map((one) => one.values));
  const valueNote = headline ? basis[0].name + ", " + (headline.index === count - 1 ? "now" : timeOf(headline.index)) : "";

  const marksSent = [...thresholdMarks(shown.thresholds), ...(spec.floor && floor ? [floor] : [])];
  const anySample = drawn.some((one) => newestSample(one.upper) !== null);
  const marksDrawn = anySample ? marksSent : [];
  const domain = valueDomain(drawn.map((one) => one.upper), 0, TOP[unit], marksDrawn.map((mark) => mark.value));
  const box: PlotBox = { left: 0, right: Math.max(1, width), top: BOX.top, bottom: PLOT_HEIGHT - BOX.bottomGap };
  const yTicks = anySample ? niceTicks(domain[0], domain[1], 2) : [];
  const tick = formatTick(unit);
  const tickText = (value: number) => (suffix === "%" ? tick(value) + suffix : suffix ? tick(value) + " " + suffix : tick(value));
  const xTicks = count > 0 && anySample ? timeTicks(start, step, count, 3) : [];
  // Nothing to draw: the backend's bucket reasons say why (each distinct one once), else "Not read".
  const blankReasons = anySample ? [] : [...new Set(scopedAt.flatMap((at) => series[at].reasons ?? [])
    .filter((reason): reason is string => Boolean(reason && reason.trim())))];
  const absent = scopedAt.map((at) => series[at]).filter((one) => !hasSample(one) && (one.stateLabel || one.reason));
  const notes = [shown.caption, shown.bands_note, bandsByNode(shown), shown.reason,
    ...absent.map((one) => one.name + ": " + [one.stateLabel, one.reason].filter(Boolean).join(" "))]
    .filter((part): part is string => Boolean(part && part.trim())).map((part) => part.trim());

  const xAt = (index: number) => bucketX(index, count, box);
  const yOf = (value: number) => valueY(value, domain, box);
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const last = count - 1;
    if (last < 0) return;
    const moves: Record<string, number | null> = {
      ArrowLeft: active === null ? last : Math.max(0, active - 1),
      ArrowRight: active === null ? last : Math.min(last, active + 1),
      Home: 0,
      End: last,
      Escape: null,
    };
    if (!Object.hasOwn(moves, event.key) || (event.key === "Escape" && active === null)) return;
    event.preventDefault();
    setCursor(moves[event.key]);
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    if (bounds.width > 0 && count > 0) setCursor(nearestBucket(event.clientX - bounds.left, count, box));
  };
  const toggle = (key: string) => setToggled((current) => {
    const next = new Set(current);
    if (!next.delete(key)) next.add(key);
    return next;
  });
  const sortedMarks = marksDrawn.filter((mark) => mark.value >= domain[0] && mark.value <= domain[1]).sort((one, other) => one.value - other.value);

  const legend = (
    <ul aria-label="Lines" className="perf-legend">
      {scopedAt.map((at) => {
        const one = series[at];
        const newest = newestSample(one.values);
        const off = hidden.has(one.key);
        const name = off && one.offLabel ? one.offLabel : one.name;
        // A value older than the newest bucket says when it was read.
        const value = newest ? format(newest.value) + (newest.index === count - 1 ? "" : " · " + timeOf(newest.index)) : NOT_READ;
        return (
          <li className="perf-legend__row" data-read={newest ? "true" : "false"} key={one.key}>
            {newest ? (
              <Button aria-pressed={!off} className="perf-legend__toggle" onClick={() => toggle(one.key)} title={name} variant="quiet">
                <Swatch mark={marks[at]} second={one.marker === "ring"} />
                <span className="perf-legend__name">{name}</span>
              </Button>
            ) : (
              <span className="perf-legend__toggle perf-legend__toggle--static" title={one.name}>
                <Swatch empty mark={marks[at]} />
                <span className="perf-legend__name">{one.name}</span>
              </span>
            )}
            {/* Why a line was not read is in the card's notes; here it is the value's title too. */}
            <b className="perf-legend__value" title={newest ? undefined : [one.stateLabel, one.reason].filter(Boolean).join(" ") || undefined}>{value}</b>
          </li>
        );
      })}
    </ul>
  );

  const chips = machines.length > 1 ? (
    <div aria-label="Machine" className="perf-chips" role="group">
      <Button aria-pressed={chosen === null} className="perf-chip" onClick={() => setPicked(null)} variant="quiet">All</Button>
      {machines.map((machine) => (
        <Button aria-pressed={chosen === machine.key} className="perf-chip" key={machine.key} onClick={() => setPicked(machine.key)} title={machine.name} variant="quiet">
          {machine.name}
        </Button>
      ))}
    </div>
  ) : undefined;

  const plot = asTable ? (
    <div className="perf-chart__plotarea perf-chart__plotarea--table">
      <SeriesTable
        eventsAt={eventsAt}
        formatValue={format}
        labels={DEFAULT_CHART_LABELS}
        series={scopedAt.map((at) => series[at]).filter(hasSample)}
        times={Array.from({ length: count }, (_, index) => timeOf(index))}
        title={shown.title ?? ""}
      />
    </div>
  ) : (
    <div className="perf-chart__plotarea">
      <div aria-hidden="true" className="perf-chart__yaxis">
        {yTicks.map((value) => <span key={value} style={{ top: yOf(value) + "px" }}>{tickText(value)}</span>)}
      </div>
      <div
        aria-describedby={footId}
        aria-label={(shown.title ?? "") + ". " + DEFAULT_CHART_LABELS.keyboardHint}
        className="chart-plot perf-chart__plot"
        onBlur={() => setCursor(null)}
        onKeyDown={onKeyDown}
        onPointerLeave={() => setCursor(null)}
        onPointerMove={onPointerMove}
        ref={plotRef}
        role="group"
        tabIndex={0}
      >
        {anySample && (
          <svg aria-hidden="true" className="perf-chart__svg" focusable="false" height={PLOT_HEIGHT} width={box.right}>
            {yTicks.map((value) => <line className="perf-grid" key={value} x1={box.left} x2={box.right} y1={yOf(value)} y2={yOf(value)} />)}
            {sortedMarks.map((mark, position) => (
              <line className="perf-threshold" key={"mark-" + position} x1={box.left} x2={box.right} y1={yOf(mark.value)} y2={yOf(mark.value)} />
            ))}
            {[...eventsAt.keys()].map((bucket) => (
              <line className="perf-event" data-bucket={bucket} key={"event-" + bucket} x1={xAt(bucket)} x2={xAt(bucket)} y1={box.top} y2={box.bottom} />
            ))}
            {drawn.map((one, position) => (
              <g className="chart-series" data-mark={one.mark} data-second={one.series.marker === "ring" ? "true" : undefined} key={position}>
                {segmentSeries(one.upper, (index) => bucketIsPartial(one.series, index)).map((segment) => {
                  const upperAt = (index: number) => yOf(one.upper[index] as number);
                  const lower = one.lower;
                  const lowerAt = lower ? (index: number) => yOf(lower[index] as number) : () => box.bottom;
                  const id = segment.from + "-" + segment.to;
                  if (segment.from === segment.to) {
                    return <circle className={"chart-dot chart-dot--" + one.mark} cx={xAt(segment.from)} cy={upperAt(segment.from)} key={id} r="3" />;
                  }
                  return (
                    <g key={id}>
                      {one.series.marker !== "ring" && (
                        <path className={"perf-area perf-area--" + one.mark + (lower ? " perf-area--stacked" : "")} d={areaPath(segment, xAt, upperAt, lowerAt)} />
                      )}
                      <path
                        className={"chart-line chart-line--" + one.mark + (segment.partial ? " chart-line--partial" : "")}
                        d={linePath(segment, xAt, upperAt)}
                        data-partial={segment.partial ? "true" : "false"}
                      />
                    </g>
                  );
                })}
              </g>
            ))}
            {active !== null && (
              <g className="chart-crosshair">
                <line className="chart-crosshair__line" x1={xAt(active)} x2={xAt(active)} y1={box.top} y2={box.bottom} />
                {drawn.map((one, position) => (typeof one.upper[active] === "number"
                  ? <circle className={"chart-focus-dot chart-dot--" + one.mark} cx={xAt(active)} cy={yOf(one.upper[active] as number)} key={position} r="4" />
                  : null))}
              </g>
            )}
          </svg>
        )}
        {sortedMarks.map((mark, position) => {
          const y = yOf(mark.value);
          // A warning mark just under a critical one prints its words below its line.
          const crowded = position + 1 < sortedMarks.length && y - yOf(sortedMarks[position + 1].value) < LABEL_ROOM;
          return (
            <span className="chart-threshold perf-threshold__label" data-below={crowded ? "true" : undefined} key={"label-" + position} style={{ top: y + "px" }}>
              {mark.label}
            </span>
          );
        })}
        {!anySample && (
          <div className="chart-plot__empty">
            {blankReasons.length ? <ul>{blankReasons.map((reason) => <li key={reason}>{reason}</li>)}</ul> : <p>{NOT_READ}</p>}
          </div>
        )}
        {active !== null && (
          <ChartTooltip
            events={eventsAt.get(active) ?? []}
            fraction={xAt(active) / Math.max(1, box.right)}
            key={active}
            rows={readBucket(drawn, active, format, DEFAULT_CHART_LABELS)}
            time={timeOf(active)}
          />
        )}
      </div>
      <div aria-hidden="true" className="perf-chart__xaxis">
        {xTicks.map((one) => {
          const at = timeX(one.t, start, step, count, box) / Math.max(1, box.right);
          return <span data-edge={at < 0.08 ? "start" : at > 0.92 ? "end" : undefined} key={one.t} style={{ left: (at * 100).toFixed(2) + "%" }}>{one.label}</span>;
        })}
      </div>
    </div>
  );

  return (
    <CardFrame
      badge={badge}
      chips={chips}
      describedBy={footId}
      foot={(
        <>
          <p className="perf-chart__notes" id={footId} title={notes.join(" · ") || undefined}>
            <span className="sr-only">{[range.caption_lead, ...notes].filter(Boolean).join(". ")}</span>
            <span aria-hidden="true">{notes.join(" · ")}</span>
          </p>
          <Button aria-pressed={asTable} className="perf-chart__table-toggle" onClick={() => setAsTable((value) => !value)} variant="quiet">
            Show as table
          </Button>
          <p aria-live="polite" className="sr-only" data-chart-live="true">{live}</p>
        </>
      )}
      legend={legend}
      plot={plot}
      state={shown.state ?? ""}
      stats={stats ? { min: format(stats.min), avg: format(stats.avg), peak: format(stats.peak) } : null}
      subtitle={[sent?.scope || range.caption_lead, suffix].filter(Boolean).join(" · ")}
      title={shown.title ?? ""}
      value={headline ? format(headline.value) : null}
      valueNote={valueNote}
    />
  );
}

export function DashboardPanels({ now, range, mode, units, floor = null, unread = "" }: {
  now: NowLayer | null; range: RangeLayer | null; mode: ClusterMode; units: Units;
  /** Why the range layer could not be read (VD-200 review S4), or "" while it is read or being read. */
  unread?: string;
  /** The decode floor the speed verdict is judged against, in tokens a second, drawn on the decode-speed card. */
  floor?: number | null;
}) {
  const context: SeriesContext = { slots: new Map((now?.nodes ?? []).map((node) => [node.key, node.slot])) };
  const names = new Map((now?.nodes ?? []).map((node) => [node.key, node.name]));
  const floorMark: ChartThreshold | null = typeof floor === "number" && Number.isFinite(floor) && floor > 0
    ? { value: floor, label: ["floor", String(Number(floor.toFixed(1))), units.tokens_per_second ?? ""].filter(Boolean).join(" ") }
    : null;
  return (
    <section className="perf-panels" aria-label="Charts">
      {PANELS.filter((spec) => !spec.advanced || mode === "advanced").map((spec) => (
        <Panel
          badge={now?.panels_freshness?.[spec.key]}
          context={context}
          floor={floorMark}
          key={spec.key}
          names={names}
          panel={range?.panels?.[spec.key]}
          range={range}
          spec={spec}
          units={units}
          unread={unread}
        />
      ))}
    </section>
  );
}
