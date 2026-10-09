import { formatPercent } from "../../../lib/format";
import { sampleText } from "./ChartTooltip";
import type { ChartLabels, ChartSeries } from "./types";

export interface SeriesTableProps {
  /** The chart's title, used as the table's caption. */
  title: string;
  /** Every series that has values. Legend toggles do not remove a column. */
  series: ChartSeries[];
  /** One label per bucket, already formatted. Its length is the row count. */
  times: string[];
  /** The event labels that fall in each bucket. */
  eventsAt: ReadonlyMap<number, string[]>;
  formatValue: (value: number) => string;
  labels: Pick<ChartLabels, "noSample" | "partial" | "time" | "events" | "coverageColumn">;
}

/**
 * "Show as table": every value, every coverage reading and every event the
 * chart plots, as text. A gap is the words for "no sample", never a zero and
 * never an empty cell, so the table says exactly what the line does.
 */
export function SeriesTable({ title, series, times, eventsAt, formatValue, labels }: SeriesTableProps) {
  const hasEvents = eventsAt.size > 0;
  return (
    <div aria-label={title} className="chart-table-wrap" role="region" tabIndex={0}>
      <table className="chart-table">
        <caption className="sr-only">{title}</caption>
        <thead>
          <tr>
            <th scope="col">{labels.time}</th>
            {series.flatMap((one, position) => [
              <th key={"v" + position} scope="col">{one.name}</th>,
              ...(one.coverage ? [<th key={"c" + position} scope="col">{one.name + " " + labels.coverageColumn}</th>] : []),
            ])}
            {hasEvents && <th scope="col">{labels.events}</th>}
          </tr>
        </thead>
        <tbody>
          {times.map((time, index) => (
            <tr key={index}>
              <th scope="row">{time}</th>
              {series.flatMap((one, position) => {
                const value = sampleText(one.values[index], formatValue, labels.noSample);
                const coverage = one.coverage?.[index];
                return [
                  <td data-cell="value" key={"v" + position}>
                    {one.partial?.[index] ? value + " (" + labels.partial + ")" : value}
                  </td>,
                  ...(one.coverage ? [
                    <td data-cell="coverage" key={"c" + position}>
                      {typeof coverage === "number" ? formatPercent(coverage * 100) : labels.noSample}
                    </td>,
                  ] : []),
                ];
              })}
              {hasEvents && <td data-cell="events">{(eventsAt.get(index) ?? []).join(", ")}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
