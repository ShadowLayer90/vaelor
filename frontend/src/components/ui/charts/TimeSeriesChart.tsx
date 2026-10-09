import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import { Button } from "../Button";
import { ChartLegend, LineKey, type LegendItem, type SeriesMark } from "./ChartLegend";
import { ChartPlot, type DrawnSeries } from "./ChartPlot";
import { ChartReadout, ChartTooltip, readBucket, readoutText } from "./ChartTooltip";
import { SeriesTable } from "./SeriesTable";
import { stackLayers } from "./chartPaths";
import {
  bucketX,
  eventBucket,
  formatTimeLabel,
  nearestBucket,
  niceTicks,
  sampleExtent,
  timeTicks,
  valueDomain,
  type PlotBox,
} from "./timeScale";
import { DEFAULT_CHART_LABELS, isAbsent, type ChartEvent, type ChartLabels, type ChartSeries, type ChartThreshold } from "./types";

export interface TimeSeriesChartProps {
  title: string;
  /** The unit, as short text beside the title ("tok/s"). */
  unit?: string;
  /** Epoch seconds of bucket 0, and the seconds between buckets. */
  start: number;
  step: number;
  series: ChartSeries[];
  events?: ChartEvent[];
  thresholds?: ChartThreshold[];
  /** Rates, power and percentages start at 0. Temperature passes 20. */
  yMin?: number;
  yMax?: number;
  /** Stack the `layer` series as areas (the prompt-token panel). */
  stacked?: boolean;
  formatValue: (value: number) => string;
  /** Axis tick text. Defaults to the bare number; the unit is in the header. */
  formatTick?: (value: number) => string;
  /** The lead of the summary sentence, such as "Last 1 h". */
  caption?: string;
  /** The whole summary sentence, when the caller has an exact one. */
  summary?: string;
  labels?: Partial<ChartLabels>;
  /** Beside the title: the panel's freshness, in the caller's words. */
  badge?: ReactNode;
  /** Controls of the caller's own that choose what is drawn (the prompt-tokens machine chips), in the title row. */
  toolbar?: ReactNode;
  /** The backend's sentences about the whole panel, shown whole under the title. */
  notes?: string[];
  /** Series keys drawn off until the reader turns them on (the GPU power total). */
  initiallyHidden?: string[];
}

/** The live region says at most one thing per this many milliseconds. */
const LIVE_THROTTLE_MS = 250;
/** Used until the plot has been measured, and in a test environment that cannot measure. */
const FALLBACK_SIZE = { width: 640, height: 124 };
/** A plot narrower than this reads its values in a strip under the plot (a phone). */
const COMPACT_WIDTH = 480;
const MARGIN = { left: 44, right: 16, top: 14, bottom: 24 };
/** A y-axis label needs this many pixels of plot height, so a short plot gets fewer, legible ticks. */
const TICK_SPACING = 20;
const tickTarget = (box: PlotBox) => Math.max(2, Math.min(4, Math.floor((box.bottom - box.top) / TICK_SPACING)));
const NODE_MARKS: SeriesMark[] = ["slot-1", "slot-2", "slot-3"];
const LAYER_MARKS: SeriesMark[] = ["layer-1", "layer-2", "layer-3"];

/**
 * Give each series its mark. Colour follows the node's slot, never its rank:
 * a node with no slot, or whose slot is already taken, gets the neutral
 * "extra" mark and is drawn only when chosen in the "Show node" chooser.
 */
function assignMarks(series: readonly ChartSeries[]): SeriesMark[] {
  const taken = new Set<number>();
  let layer = 0;
  return series.map((one) => {
    if (one.kind === "aggregate") return "aggregate";
    if (one.kind === "layer") return LAYER_MARKS[Math.min(layer++, LAYER_MARKS.length - 1)];
    // A ring-marked companion is the same machine's second line: same slot, same mark.
    if (one.marker === "ring" && one.slot) return NODE_MARKS[one.slot - 1];
    if (one.slot && !taken.has(one.slot)) {
      taken.add(one.slot);
      return NODE_MARKS[one.slot - 1];
    }
    return "extra";
  });
}

export function TimeSeriesChart({
  title, unit, start, step, series, events = [], thresholds = [], yMin = 0, yMax, stacked = false,
  formatValue, formatTick = String, caption, summary, labels, badge, toolbar, notes = [], initiallyHidden,
}: TimeSeriesChartProps) {
  const words = useMemo(() => ({ ...DEFAULT_CHART_LABELS, ...labels }), [labels]);
  const captionId = "chart-" + useId().replaceAll(":", "");
  const plotRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState(FALLBACK_SIZE);
  // Toggles and the chooser remember a series by its key, so a refresh that
  // reorders the series does not hide or draw a different machine.
  const [hidden, setHidden] = useState<ReadonlySet<string>>(() => new Set(initiallyHidden ?? []));
  const [chosenKey, setChosenKey] = useState<string | null>(null);
  const [cursor, setActive] = useState<number | null>(null);
  const [asTable, setAsTable] = useState(false);
  const [live, setLive] = useState("");
  const lastSaid = useRef(0);

  useEffect(() => {
    const element = plotRef.current;
    if (!element || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      if (width > 0 && height > 0) setSize({ width: Math.round(width), height: Math.round(height) });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [asTable]);

  const marks = assignMarks(series);
  const present = series.filter((one) => !isAbsent(one));
  const count = Math.max(0, ...present.map((one) => one.values.length));
  const span = step * Math.max(0, count - 1);
  // A shorter range can arrive while the crosshair sits past its new end.
  const active = cursor !== null && cursor < count ? cursor : null;
  const extras = series.map((_, at) => at).filter((at) => marks[at] === "extra" && !isAbsent(series[at]));
  const chosen = extras.findIndex((at) => series[at].key === chosenKey);
  // The legend toggles every drawn series except an unchosen slotless node, which lives in the chooser.
  // A series the backend sent with no values is named, with its state and reason, in the notes instead,
  // so its sentence is shown whole and the control row stays one row.
  const legendAt = series.map((_, at) => at).filter((at) => !isAbsent(series[at]) && (marks[at] !== "extra" || at === extras[chosen]));
  const absentAt = series.map((_, at) => at).filter((at) => isAbsent(series[at]));
  const items: LegendItem[] = legendAt.map((at) => ({ series: series[at], mark: marks[at], shown: !hidden.has(series[at].key) }));
  const visibleAt = legendAt.filter((at) => !isAbsent(series[at]) && !hidden.has(series[at].key));

  const stackedAt = stacked ? visibleAt.filter((at) => series[at].kind === "layer") : [];
  const stack = stackLayers(stackedAt.map((at) => series[at].values));
  const drawn: DrawnSeries[] = visibleAt.map((at) => {
    const layer = stackedAt.indexOf(at);
    return layer < 0
      ? { series: series[at], mark: marks[at], upper: series[at].values }
      : { series: series[at], mark: marks[at], upper: stack[layer].upper, lower: stack[layer].lower };
  });

  const box: PlotBox = {
    left: MARGIN.left, right: Math.max(MARGIN.left + 1, size.width - MARGIN.right),
    top: MARGIN.top, bottom: Math.max(MARGIN.top + 1, size.height - MARGIN.bottom),
  };
  // With no sample drawn there is nothing to scale: no axis values and no
  // marks are invented, and the plot says "no sample".
  const anySample = drawn.some((one) => sampleExtent(one.upper) !== null);
  // Nothing to draw: the backend's bucket reasons say why (each distinct one once), else "no sample".
  const blankReasons = anySample ? [] : [...new Set(drawn.flatMap(({ series: one }) => one.reasons ?? [])
    .filter((reason): reason is string => Boolean(reason && reason.trim())))];
  const marksDrawn = anySample ? thresholds : [];
  const domain = valueDomain(drawn.map((one) => one.upper), yMin, yMax, marksDrawn.map((mark) => mark.value));
  const timeOf = (index: number) => formatTimeLabel(start + index * step, span, step);

  const eventsAt = new Map<number, string[]>();
  for (const event of events) {
    const bucket = eventBucket(event.t, start, step, count);
    if (bucket !== null) eventsAt.set(bucket, [...(eventsAt.get(bucket) ?? []), event.label]);
  }

  const rows = active === null ? [] : readBucket(drawn, active, formatValue, words);
  const docked = size.width < COMPACT_WIDTH;
  // The docked strip shows the bucket under the crosshair, else the newest.
  const readoutAt = active ?? count - 1;
  const said = active === null ? "" : readoutText(timeOf(active), rows, eventsAt.get(active) ?? []);
  useEffect(() => {
    // Closing the tooltip clears the region at once and does not use up the throttle.
    if (said === "") { setLive(""); return undefined; }
    const wait = lastSaid.current + LIVE_THROTTLE_MS - Date.now();
    const say = () => { lastSaid.current = Date.now(); setLive(said); };
    if (wait <= 0) { say(); return undefined; }
    const timer = setTimeout(say, wait);
    return () => clearTimeout(timer);
  }, [said]);

  const parts = drawn.map(({ series: one }) => {
    const extent = sampleExtent(one.values);
    if (!extent) return one.name + " " + words.noSample;
    const [low, high] = [formatValue(extent[0]), formatValue(extent[1])];
    return one.name + " " + (low === high ? low : words.range(low, high));
  });
  const happened = events.filter((event) => eventBucket(event.t, start, step, count) !== null)
    .map((event) => words.eventAt(event.label, formatTimeLabel(event.t, span)));
  // A caption with no drawn series after it takes no colon ("Last 1 h", never "Last 1 h: ").
  const lead = caption && parts.length ? caption + ": " : caption ?? "";
  const computed = [lead + parts.join(", "), ...(happened.length ? [happened.join(", ")] : [])].filter(Boolean).join("; ");

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
    setActive(moves[event.key]);
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    if (bounds.width > 0 && count > 0) setActive(nearestBucket(event.clientX - bounds.left, count, box));
  };
  const toggle = (position: number) => {
    const key = series[legendAt[position]].key;
    setHidden((current) => {
      const next = new Set(current);
      if (!next.delete(key)) next.add(key);
      return next;
    });
  };
  const choose = (position: number) => setChosenKey(position < 0 ? null : series[extras[position]].key);


  return (
    <figure className="chart" data-stacked={stacked ? "true" : "false"}>
      <header className="chart__header">
        <h3 className="chart__title">{title}</h3>
        {unit && <span className="chart__unit">{unit}</span>}
        {toolbar && <div className="chart__toolbar">{toolbar}</div>}
        {badge && <span className="chart__badge">{badge}</span>}
      </header>
      {/* The summary is on screen for everyone (owner decision "readable charts"); it wraps, never clipped. */}
      <figcaption className="chart__caption" id={captionId}>{summary ?? computed}</figcaption>
      {(notes.length > 0 || absentAt.length > 0) && (
        <ul className="chart__notes">
          {notes.map((note, at) => <li key={"note-" + at}>{note}</li>)}
          {absentAt.map((at) => (
            <li className="chart__note-absent" key={"absent-" + at}>
              <LineKey mark={marks[at]} />
              <strong className="chart-legend__name">{series[at].name}</strong>
              {series[at].stateLabel && <span className="chart-legend__state">{series[at].stateLabel}</span>}
              <span className="chart-legend__reason">{series[at].reason}</span>
            </li>
          ))}
        </ul>
      )}
      {asTable ? (
        <SeriesTable
          eventsAt={eventsAt}
          formatValue={formatValue}
          labels={words}
          series={present}
          times={Array.from({ length: count }, (_, index) => timeOf(index))}
          title={title}
        />
      ) : (
        <div
          aria-describedby={captionId}
          aria-label={title + ". " + words.keyboardHint}
          className="chart-plot"
          onBlur={() => setActive(null)}
          onKeyDown={onKeyDown}
          onPointerLeave={() => setActive(null)}
          onPointerMove={onPointerMove}
          ref={plotRef}
          role="group"
          tabIndex={0}
        >
          <ChartPlot
            active={active}
            box={box}
            count={count}
            domain={domain}
            drawn={drawn}
            eventBuckets={[...eventsAt.keys()]}
            formatTick={formatTick}
            height={size.height}
            start={start}
            step={step}
            thresholds={marksDrawn}
            width={size.width}
            xTicks={timeTicks(start, step, count)}
            yTicks={anySample ? niceTicks(domain[0], domain[1], tickTarget(box)) : []}
          />
          {!anySample && (
            <div className="chart-plot__empty">
              {blankReasons.length
                ? <ul>{blankReasons.map((reason) => <li key={reason}>{reason}</li>)}</ul>
                : <p>{words.noSample}</p>}
            </div>
          )}
          {active !== null && !docked && (
            <ChartTooltip
              events={eventsAt.get(active) ?? []}
              fraction={bucketX(active, count, box) / size.width}
              key={active}
              rows={rows}
              time={timeOf(active)}
            />
          )}
        </div>
      )}
      {!asTable && docked && readoutAt >= 0 && (
        <ChartReadout
          events={eventsAt.get(readoutAt) ?? []}
          reserveEvents={eventsAt.size > 0}
          rows={readBucket(drawn, readoutAt, formatValue, words)}
          time={timeOf(readoutAt)}
        />
      )}
      <p aria-live="polite" className="sr-only" data-chart-live="true">{live}</p>
      <div className="chart__controls">
        <ChartLegend
          chooser={extras.length ? { names: extras.map((at) => series[at].name), chosen, onChoose: choose } : undefined}
          items={items}
          onToggle={toggle}
        />
        <Button aria-pressed={asTable} className="chart__table-toggle" onClick={() => setAsTable((value) => !value)} variant="quiet">
          Show as table
        </Button>
      </div>
    </figure>
  );
}
