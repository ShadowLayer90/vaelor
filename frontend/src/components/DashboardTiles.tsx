import { useId } from "react";
import { tileProps, type Units } from "../lib/dashboardSeries";
import type { DashboardTile, NowLayer, RangeLayer } from "../lib/performanceDashboard";

/**
 * The six stat tiles (VD-200, the ClusterPerformance board), in a fixed order
 * that never changes with the engine or the mode: four from the range layer
 * and two (the hottest GPU and the busiest host CPU, "now") from the now
 * layer. Every tile is one height: the title is one line, the value one row,
 * the meter row is reserved whether or not the tile has a meter, and the foot
 * is one line, ellipsised (its whole sentence is in its title and read out):
 * at 120 px a second line was cut through its letters, and a taller tile
 * pushes the first chart row under the fold (VD-200 review S5). At 1440 px and
 * below, where the fold is already past row 1, every tile grows by the one line the
 * foot needs to show its second line (performance-dashboard.css).
 * Titles, captions, state labels and reasons are the backend's; the keys here
 * only place them. Each machine's own value is on the charts' legends.
 */
const TILES: { key: string; layer: "range" | "now"; meter: "none" | "share" | "limits"; nameFirst?: boolean }[] = [
  { key: "decode_speed_mean", layer: "range", meter: "none" },
  { key: "slowest_machine", layer: "range", meter: "none", nameFirst: true },
  { key: "prefix_cache_hits", layer: "range", meter: "share" },
  { key: "hottest_gpu", layer: "now", meter: "limits" },
  { key: "avg_gpu_power", layer: "range", meter: "none" },
  { key: "busiest_cpu", layer: "now", meter: "share" },
];

function Tile({ tile, units, meter, nameFirst }: { tile: DashboardTile; units: Units; meter: "none" | "share" | "limits"; nameFirst?: boolean }) {
  const titleId = "perf-tile-" + useId().replaceAll(":", "");
  const props = tileProps(tile, units, meter);
  // The slowest machine is a machine: its name is the reading and its speed the foot.
  const name = nameFirst && props.value !== null ? tile.cluster?.name ?? "" : "";
  const value = name || props.value;
  const nodes = (props.nodes ?? []).map((node) => node.name + " " + (node.value ?? node.stateLabel ?? "")).join(", ");
  const foot = props.value === null
    ? [props.stateLabel, props.reason].filter(Boolean).join(" ")
    : [name ? props.value + (props.unit ? " " + props.unit : "") : "", props.caption, props.reason].filter(Boolean).join(" · ");
  const bar = props.meter;
  const along = bar && bar.max > 0 ? Math.min(100, Math.max(0, (bar.value / bar.max) * 100)) : 0;
  const marks = bar?.marks?.map((mark) => mark.label).join(", ") ?? "";
  return (
    <article aria-labelledby={titleId} className="ui-card perf-tile" data-state={props.state}>
      <h3 className="perf-tile__title" id={titleId} title={props.title}>{props.title}</h3>
      <p className="perf-tile__value">
        {value === null ? (
          <span className="perf-tile__unread">Not read</span>
        ) : (
          <>
            {/* A machine's name is words, not a figure: it takes two lines of the value's own row
                rather than one cut line (polish audit, cause 14), and keeps its whole name as its title. */}
            <span className={name ? "chart-tile__number perf-tile__number perf-tile__number--name" : "chart-tile__number perf-tile__number"} title={name || undefined}>{value}</span>
            {!name && props.unit && <span className="perf-tile__unit">{props.unit}</span>}
          </>
        )}
      </p>
      {bar ? (
        <div
          aria-label={props.title}
          aria-valuemax={bar.max}
          aria-valuemin={0}
          aria-valuenow={Math.min(bar.max, Math.max(0, bar.value))}
          aria-valuetext={marks ? bar.valueText + "; " + marks : bar.valueText}
          className="perf-tile__meter"
          role="meter"
          title={marks || undefined}
        >
          <i style={{ width: along + "%" }} />
        </div>
      ) : <div aria-hidden="true" className="perf-tile__meter perf-tile__meter--none" />}
      <p className="perf-tile__foot" title={[foot, nodes].filter(Boolean).join(" · ") || undefined}>
        {props.value === null && props.stateLabel && <strong className="perf-tile__state">{props.stateLabel}</strong>}
        {props.value === null ? (props.stateLabel && props.reason ? " " : "") + props.reason : foot}
      </p>
      {nodes && <p className="sr-only">{nodes}</p>}
    </article>
  );
}

export function DashboardTiles({ now, range, units, nowError = "", rangeError = "" }: {
  now: NowLayer | null; range: RangeLayer | null; units: Units;
  /** Why each layer could not be read, or "" (VD-200 review S4: never "Reading…" for ever). */
  nowError?: string; rangeError?: string;
}) {
  return (
    <section className="perf-tiles" aria-label="Summary">
      {TILES.map((spec) => {
        const layer = spec.layer === "now" ? now : range;
        const tile = layer?.tiles?.[spec.key];
        const error = spec.layer === "now" ? nowError : rangeError;
        // A layer that could not be read says so, with why; it is not still reading.
        if (!tile && !layer && error) {
          return (
            <div className="ui-card perf-tile perf-tile--reading" data-state="unread" key={spec.key}>
              <span className="perf-tile__unread">Not read</span>
              <span className="perf-tile__foot" title={error}>{error}</span>
            </div>
          );
        }
        // Before the first answer a tile says it is reading, never a zero.
        if (!tile) return <div aria-busy="true" className="ui-card perf-tile perf-tile--reading" key={spec.key}>Reading…</div>;
        return <Tile key={spec.key} meter={spec.meter} nameFirst={spec.nameFirst} tile={tile} units={units} />;
      })}
    </section>
  );
}
