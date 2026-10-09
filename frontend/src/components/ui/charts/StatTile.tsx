import { useId } from "react";
import { UnavailableValue } from "../UnavailableValue";
import { Meter, type MeterProps } from "./Meter";
import { DEFAULT_CHART_LABELS, type ChartLabels, type TileNodeValue, type ValueState } from "./types";

/** How many machines the "beneath" line names before it says "and N more". */
const BENEATH_LIMIT = 3;

export interface StatTileProps {
  title: string;
  /** Already formatted by the caller. `null` means there is no value to show. */
  value: string | null;
  unit?: string;
  state: ValueState;
  /** The backend's words for `state`. */
  stateLabel?: string;
  /** The backend's sentence. Printed in full when there is no value. */
  reason?: string;
  /** The scope the value covers, in the backend's words. */
  caption?: string;
  meter?: Omit<MeterProps, "label" | "caption" | "className">;
  /** Per-machine values, in slot order. */
  nodes?: TileNodeValue[];
  labels?: Partial<Pick<ChartLabels, "andMore" | "noReading">>;
}

/**
 * One stat tile. Its rows are fixed: title, value, then a detail block of a
 * caption, a meter row and a "beneath" line. Every row is reserved whether or
 * not it is filled, so a tile with no meter, no machines or no value at all
 * is the same size as its neighbours and its title sits in the same place.
 *
 * A missing value is the shared `UnavailableValue` mark, and the reason is
 * printed beside it in the detail block. It is never a zero.
 */
export function StatTile({
  title,
  value,
  unit,
  state,
  stateLabel,
  reason = "",
  caption = "",
  meter,
  nodes = [],
  labels,
}: StatTileProps) {
  const words = { ...DEFAULT_CHART_LABELS, ...labels };
  const titleId = "chart-tile-" + useId().replaceAll(":", "");
  const shown = nodes.slice(0, BENEATH_LIMIT);
  const more = nodes.length - shown.length;
  // A value can still carry a reason (a partial total names the machine it
  // left out). It is printed after the scope, never dropped. The scope is a
  // caption that may not end its sentence, so it is ended before the reason
  // begins; joined bare they read as one run-on sentence (ACC-207).
  const scope = [caption, reason].filter(Boolean)
    .map((part, index, parts) => (index < parts.length - 1 && !/[.!?]$/.test(part.trim()) ? part.trim() + "." : part))
    .join(" ");
  return (
    // A tile listing as many machines as it shows is crowded: on a wide screen each name keeps to one
    // line (cut there, whole in its title and to a screen reader) so the dashboard's first row fits.
    <article aria-labelledby={titleId} className="chart-tile" data-crowded={nodes.length >= BENEATH_LIMIT ? "true" : undefined} data-state={state}>
      <h3 className="chart-tile__title" id={titleId} title={title}>{title}</h3>
      <p className="chart-tile__value">
        {value === null ? (
          <UnavailableValue label={title + " unavailable"} reason={reason || undefined} />
        ) : (
          <>
            <span className="chart-tile__number">{value}</span>
            {unit && <span className="chart-tile__unit">{unit}</span>}
          </>
        )}
      </p>
      {value === null ? (
        <p className="chart-tile__reason" title={reason || undefined}>
          {stateLabel && <strong className="chart-tile__state">{stateLabel}</strong>}
          {stateLabel && reason ? " " : ""}
          {reason}
        </p>
      ) : (
        <div className="chart-tile__detail">
          <p className="chart-tile__caption" title={scope || undefined}>{scope}</p>
          <div className="chart-tile__meter">
            {meter && <Meter {...meter} caption="hidden" label={title} />}
          </div>
          <p className="chart-tile__beneath">
            {shown.map((node, index) => (
              <span className="chart-tile__node" key={index}>
                <span className="chart-tile__node-name" title={node.name}>{node.name}</span>
                <span className="chart-tile__node-value">
                  {node.value ?? node.stateLabel ?? words.noReading}
                </span>
              </span>
            ))}
            {more > 0 && <span className="chart-tile__more">{words.andMore(more)}</span>}
          </p>
        </div>
      )}
    </article>
  );
}
