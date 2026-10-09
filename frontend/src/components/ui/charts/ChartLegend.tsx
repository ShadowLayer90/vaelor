import { Button } from "../Button";
import { Select } from "../Select";
import { isAbsent, type ChartSeries } from "./types";

/**
 * The mark a series is drawn with. It is the secondary encoding as well as
 * the colour, so colour is never the only key:
 *
 *   slot-1 solid, slot-2 dashed, slot-3 dotted, aggregate a thicker neutral
 *   line, layer-N a neutral fill, and "extra" a neutral dash-dot line for a
 *   node that has no colour slot. No hue is ever generated for a fourth node.
 */
export type SeriesMark = "slot-1" | "slot-2" | "slot-3" | "aggregate" | "extra" | "layer-1" | "layer-2" | "layer-3";

/** The line-key swatch: a short stroke painted by the same class as the line. */
export function LineKey({ mark, ring = false }: { mark: SeriesMark; ring?: boolean }) {
  return (
    <svg aria-hidden="true" className="chart-key" focusable="false" height="10" width="24">
      {mark.startsWith("layer") && (
        <rect className={"chart-area chart-area--" + mark} height="7" width="24" x="0" y="3" />
      )}
      <line className={"chart-line chart-line--" + mark} x1="0" x2="24" y1={mark.startsWith("layer") ? 3 : 5} y2={mark.startsWith("layer") ? 3 : 5} />
      {ring && <circle className={"chart-ring chart-ring--" + mark} cx="12" cy="5" r="3" />}
    </svg>
  );
}

export interface LegendItem {
  series: ChartSeries;
  mark: SeriesMark;
  /** False when the reader has toggled the series off. */
  shown: boolean;
}

export interface LegendChooser {
  /** The names of the nodes that have no colour slot. */
  names: string[];
  /** The position in `names` that is drawn, or -1 for none. */
  chosen: number;
  onChoose: (position: number) => void;
}

export interface ChartLegendProps {
  items: LegendItem[];
  /** Called with the item's position in `items`. */
  onToggle: (position: number) => void;
  chooser?: LegendChooser;
}

/**
 * One entry per series, always. A series with data is a toggle; a series the
 * backend sent with no values is named with its state and its reason, because
 * leaving it out would read as "this machine does not exist".
 *
 * Toggling never recolours: the swatch keeps its mark class whether the
 * series is shown or not, and only the name is struck through.
 */
export function ChartLegend({ items, onToggle, chooser }: ChartLegendProps) {
  return (
    <div className="chart-legend">
      <ul className="chart-legend__list">
        {items.map((item, position) => (isAbsent(item.series) ? (
          <li className="chart-legend__absent" key={position}>
            <LineKey mark={item.mark} />
            <span className="chart-legend__name">{item.series.name}</span>
            {item.series.stateLabel && <span className="chart-legend__state">{item.series.stateLabel}</span>}
            <span className="chart-legend__reason">{item.series.reason}</span>
          </li>
        ) : (
          <li className="chart-legend__item" key={position}>
            <Button
              aria-pressed={item.shown}
              className={"chart-legend__toggle" + (!item.shown && item.series.offLabel ? " chart-legend__toggle--off-label" : "")}
              onClick={() => onToggle(position)}
              title={!item.shown && item.series.offLabel ? item.series.offLabel : item.series.name}
              variant="quiet"
            >
              <LineKey mark={item.mark} ring={item.series.marker === "ring"} />
              {/* Off by default: the backend's words say so, instead of a struck-through name. */}
              <span className="chart-legend__name">{!item.shown && item.series.offLabel ? item.series.offLabel : item.series.name}</span>
            </Button>
          </li>
        )))}
      </ul>
      {chooser && chooser.names.length > 0 && (
        <div className="chart-legend__chooser">
          <Select
            label="Show node"
            onChange={(event) => chooser.onChoose(Number(event.target.value))}
            value={String(chooser.chosen)}
          >
            <option value="-1">None</option>
            {chooser.names.map((name, position) => (
              <option key={position} value={String(position)}>{name}</option>
            ))}
          </Select>
        </div>
      )}
    </div>
  );
}
