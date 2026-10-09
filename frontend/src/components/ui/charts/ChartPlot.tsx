import type { SeriesMark } from "./ChartLegend";
import { areaPath, linePath } from "./chartPaths";
import {
  bucketIsPartial,
  bucketX,
  segmentSeries,
  timeX,
  valueY,
  type PlotBox,
  type TimeTick,
} from "./timeScale";
import type { ChartSeries, ChartThreshold } from "./types";

export interface DrawnSeries {
  series: ChartSeries;
  mark: SeriesMark;
  /** What the line follows: the series' own values, or a stacked layer's upper edge. */
  upper: (number | null)[];
  /** A stacked layer's lower edge. Absent for a plain line. */
  lower?: (number | null)[];
}

export interface ChartPlotProps {
  width: number;
  height: number;
  box: PlotBox;
  count: number;
  start: number;
  step: number;
  domain: [number, number];
  yTicks: number[];
  xTicks: TimeTick[];
  drawn: DrawnSeries[];
  thresholds: ChartThreshold[];
  /** The buckets that hold at least one event. */
  eventBuckets: number[];
  /** The bucket under the crosshair, or null. */
  active: number | null;
  formatTick: (value: number) => string;
}

/** The pixels one line of axis text needs above a threshold line. */
const LABEL_ROOM = 16;
/** A ring-marked line carries a ring about every this many pixels. */
const RING_SPACING = 36;

/**
 * The drawing itself. It is `aria-hidden`: everything it shows is also in
 * the tooltip, the live region and the table, as text.
 *
 * A series is drawn segment by segment. A gap between two segments is left
 * empty, a partial segment carries `chart-line--partial` (dashed, half
 * opacity), and a sample with a gap on both sides is a dot.
 */
export function ChartPlot({
  width, height, box, count, start, step, domain, yTicks, xTicks, drawn, thresholds, eventBuckets, active, formatTick,
}: ChartPlotProps) {
  const xAt = (index: number) => bucketX(index, count, box);
  const yOf = (value: number) => valueY(value, domain, box);
  const marks = thresholds.filter((mark) => mark.value >= domain[0] && mark.value <= domain[1])
    .sort((one, other) => one.value - other.value);
  return (
    <svg aria-hidden="true" className="chart-plot__svg" focusable="false" height={height} width={width}>
      {yTicks.map((tick) => (
        <g className="chart-tick" key={tick}>
          <line className="chart-grid" x1={box.left} x2={box.right} y1={yOf(tick)} y2={yOf(tick)} />
          <text className="chart-axis-text" dy="0.32em" textAnchor="end" x={box.left - 8} y={yOf(tick)}>
            {formatTick(tick)}
          </text>
        </g>
      ))}
      {xTicks.map((tick) => (
        <text
          className="chart-axis-text"
          key={tick.t}
          textAnchor="middle"
          x={timeX(tick.t, start, step, count, box)}
          y={box.bottom + 16}
        >
          {tick.label}
        </text>
      ))}
      {marks.map((mark, position) => {
        const y = yOf(mark.value);
        // A warning mark just under a critical one: its label goes below its
        // line, so the two labels do not print on top of each other.
        const crowded = position + 1 < marks.length && y - yOf(marks[position + 1].value) < LABEL_ROOM;
        return (
          <g className="chart-threshold" key={position}>
            <line className="chart-threshold__line" x1={box.left} x2={box.right} y1={y} y2={y} />
            <text className="chart-axis-text" textAnchor="end" x={box.right} y={crowded ? y + 13 : y - 4}>{mark.label}</text>
          </g>
        );
      })}
      {eventBuckets.map((bucket) => (
        <g className="chart-event" data-bucket={bucket} key={bucket}>
          <line className="chart-event__line" x1={xAt(bucket)} x2={xAt(bucket)} y1={box.top} y2={box.bottom} />
          <path
            className="chart-event__marker"
            d={"M" + (xAt(bucket) - 5) + " " + (box.top - 8) + " h10 l-5 8 Z"}
          />
        </g>
      ))}
      {drawn.map((one, position) => {
        const partialAt = (index: number) => bucketIsPartial(one.series, index);
        const upperAt = (index: number) => yOf(one.upper[index] as number);
        const lower = one.lower;
        return (
          <g className="chart-series" data-mark={one.mark} key={position}>
            {segmentSeries(one.upper, partialAt).map((segment) => {
              const partial = segment.partial ? " chart-line--partial" : "";
              const id = segment.from + "-" + segment.to;
              if (segment.from === segment.to) {
                return (
                  <circle
                    className={"chart-dot chart-dot--" + one.mark + (segment.partial ? " chart-dot--partial" : "")}
                    cx={xAt(segment.from)}
                    cy={upperAt(segment.from)}
                    data-from={segment.from}
                    key={id}
                    r="3"
                  />
                );
              }
              return (
                <g key={id}>
                  {lower && (
                    <path
                      className={"chart-area chart-area--" + one.mark + (segment.partial ? " chart-area--partial" : "")}
                      d={areaPath(segment, xAt, upperAt, (index) => yOf(lower[index] as number))}
                    />
                  )}
                  <path
                    className={"chart-line chart-line--" + one.mark + partial}
                    d={linePath(segment, xAt, upperAt)}
                    data-from={segment.from}
                    data-partial={segment.partial ? "true" : "false"}
                    data-to={segment.to}
                  />
                </g>
              );
            })}
            {one.series.marker === "ring" && one.upper.map((value, index) => (
              typeof value === "number" && index % Math.max(1, Math.round(RING_SPACING / Math.max(1, (box.right - box.left) / Math.max(1, count - 1)))) === 0 ? (
                <circle className={"chart-ring chart-ring--" + one.mark} cx={xAt(index)} cy={upperAt(index)} key={"ring-" + index} r="3" />
              ) : null
            ))}
          </g>
        );
      })}
      {active !== null && (
        <g className="chart-crosshair">
          <line className="chart-crosshair__line" x1={xAt(active)} x2={xAt(active)} y1={box.top} y2={box.bottom} />
          {drawn.map((one, position) => (typeof one.upper[active] === "number" ? (
            <circle
              className={"chart-focus-dot chart-dot--" + one.mark}
              cx={xAt(active)}
              cy={yOf(one.upper[active] as number)}
              key={position}
              r="4"
            />
          ) : null))}
        </g>
      )}
    </svg>
  );
}
