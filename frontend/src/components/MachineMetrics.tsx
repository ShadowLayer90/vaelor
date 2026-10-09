import type { ReactNode } from "react";
import { StatTile, type TrendBar } from "./ui";
import { StatusPill } from "./StatusPill";
import { gpuTemperature } from "../lib/gpuTemperature";
import { metricNumber } from "../lib/metrics";
import { exactTime, formatQuantity } from "../lib/format";
import { trendBars } from "../lib/homeSummary";
import { retentionNote } from "../lib/retention";
import { useMachineMetrics, type HistoryPoint, type TelemetryHistory } from "../hooks/useMachineMetrics";
import { currentWorkerValue, workerFreshness } from "../lib/workerFreshness";
import type { ClusterMode } from "../lib/clusterMode";
import type { Metrics } from "../types";

/**
 * The per-node telemetry of a Fleet machine card (VD-200, the ClusterFleet
 * and ClusterFleetDetail boards; Phase E1 for the controller, E2c for a worker).
 *
 * The trend and the store's retention state come from the `/telemetry/history`
 * route, so this is honest about scope by construction:
 *
 * - The CONTROLLER shows a live headline (from `/telemetry/current`) beside a
 *   short trend per metric (from the no-node `/telemetry/history`), the window
 *   it covers, and the retention state.
 * - A WORKER reads its OWN series from `/telemetry/history?node=<id>` (workers
 *   push to the controller since E2b) and NEVER `/telemetry/current` — that is
 *   the controller's live snapshot, so showing it on a worker would misattribute
 *   the controller's metrics to the worker. Its headline is its newest RAW
 *   sample (`latest`), read through the ONE freshness rule in
 *   `lib/workerFreshness` that the collapsed card and the fleet headline also
 *   use, so the card never carries two different current figures (ACC-119). A
 *   fresh reading renders "updated <n>s ago"; a stale one shows "Not reporting
 *   since <time>" over a faded last-known trend with a "—" headline; a post the
 *   controller refused for clock skew says so with the backend's reason; a
 *   worker with no telemetry agent installed says that, and one with an agent
 *   that has never reported says it is not reporting yet.
 * - An OFFLINE/unreachable node says metrics are unavailable while it is in that
 *   state and reads nothing.
 *
 * Each reading is a stat tile with its short trend. A metric with no samples
 * reads "Not read" with the reason, never a fabricated flat line or a
 * substituted zero. Easy shows a headline set; Advanced adds memory, GPU
 * utilisation, GPU temperature and GPU power.
 */

interface Props {
  isController: boolean;
  /**
   * This machine's cluster node id, used to read its own node-scoped history.
   * Only threaded for a worker; the controller reads the no-node route.
   */
  nodeId?: string;
  /** The node is unreachable/down/missing — the card computed this. */
  offline: boolean;
  /** The status the card's pill shows, for the offline sentence. */
  stateLabel: string;
  gpuPresent: boolean;
  mode: ClusterMode;
  /**
   * A worker's install-state (`FleetNode.telemetry_provisioned`). `false`
   * means no telemetry agent is installed, which is said as such rather than
   * as "not reporting yet" (ACC-130). Undefined when the record carries none.
   */
  telemetryProvisioned?: boolean;
  /** Whether the viewer may install telemetry, to point at the right action. */
  administrator?: boolean;
  /**
   * Whether the worker has joined the cluster (VD-194): a joined worker's agent
   * comes with its worker software at Recheck; one awaiting its join is
   * installed from the card.
   */
  joined?: boolean;
}

/** A tile's reading: the figure, and its unit drawn small beside it. */
interface TileReading {
  value: string;
  unit?: string;
}

interface TileSpec {
  id: string;
  /** The tile's name; a function when it depends on what was read (the GPU's sensor). */
  label: string | ((metrics: Metrics) => string);
  /** The trend series key, and the reason shown when it is unmeasured. */
  series: string | ((metrics: Metrics) => string);
  unmeasured: string;
  /** The live headline from the current snapshot; null when it was not read. */
  reading: (metrics: Metrics) => TileReading | null;
  /** The live figure used to decide "no reading at all". */
  live: (metrics: Metrics) => number | null;
  detail: (metrics: Metrics) => ReactNode;
  /** The top of the trend's range, for bars drawn as a share of it. */
  scale: (metrics: Metrics, points: HistoryPoint[]) => number;
  show: (mode: ClusterMode, gpu: boolean) => boolean;
}

const ALWAYS = () => true;
const PERCENT_SCALE = () => 100;

/**
 * A worker has no `/telemetry/current`, so its live headline is read from its
 * newest raw sample, which is keyed like the trend's series. The tiles read a
 * live snapshot by the metric's own key, so this maps one to the other.
 */
const SERIES_TO_METRIC: Record<string, string> = {
  processor_load: "cpu_percent",
  memory_percent: "memory_percent",
  gpu_busy_percent: "gpu_busy_percent",
  gpu_gtt_used_bytes: "gpu_gtt_used_bytes",
  gpu_gtt_total_bytes: "gpu_gtt_total_bytes",
  cpu_temperature_c: "cpu_temperature",
  gpu_temperature_c: "gpu_temperature_c",
  gpu_gfx_temperature_c: "gpu_gfx_temperature_c",
  gpu_power_watts: "gpu_power_watts",
};

/**
 * What measured a machine's GPU power, for the tile's name (ACC-204): the
 * telemetry owner's own word (`gpu_power_sensor`, "graphics engine" or
 * "board"), sent with the controller's sample and a worker's newest row.
 * With no word the tile claims no sensor.
 */
function gpuPowerSensor(metrics: Metrics): string {
  return typeof metrics.gpu_power_sensor === "string" ? metrics.gpu_power_sensor : "";
}

/** A figure kept with its unit ("39.1 GB"), so a narrow tile never breaks between them. */
function withUnit(quantity: string): string {
  return quantity.replace(/ (?=\S+$)/, " ");
}

/** A whole-number reading with its unit ("18" and "%"). */
function rounded(value: number | null, unit: string): TileReading | null {
  return value === null ? null : { value: String(Math.round(value)), unit };
}

/** A GPU power figure: an idle graphics engine draws a few hundredths of a watt. */
function gpuPower(value: number | null): TileReading | null {
  if (value === null) return null;
  return value < 0.05 ? { value: "under 0.1", unit: "W" } : { value: value.toFixed(1), unit: "W" };
}

/** The highest measured point, so a trend of watts or bytes fills its own range. */
function peakOf(points: HistoryPoint[]): number {
  const values = points.map((point) => point.v).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  return values.length ? Math.max(...values, 1) : 1;
}

function windowLabel(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "recent";
  // Days only from two days up, so a 24h window reads "last 24h" rather than
  // the stiffer "last 1d"; 7d still reads "last 7d".
  if (seconds % 86400 === 0 && seconds >= 2 * 86400) return `last ${seconds / 86400}d`;
  if (seconds % 3600 === 0) return `last ${seconds / 3600}h`;
  return `last ${Math.round(seconds / 60)}m`;
}

/** How old a fresh reading is, as the card's foot and the telemetry header say it. */
export function freshnessLabel(ageSeconds: number): string {
  if (ageSeconds < 10) return "updated just now";
  if (ageSeconds < 60) return `updated ${Math.round(ageSeconds)}s ago`;
  return `updated ${Math.round(ageSeconds / 60)} min ago`;
}

/**
 * A worker's live headline: its newest RAW sample, only while fresh, keyed the
 * way the tiles read a live snapshot. Never the newest bucket of the trend,
 * which is a mean over a partial window. An unmeasured value leaves its key
 * absent, so its tile draws the unmeasured state rather than a zero.
 */
export function latestMetrics(history: TelemetryHistory | null): Metrics {
  const metrics: Record<string, number | string> = {};
  for (const [seriesKey, metricKey] of Object.entries(SERIES_TO_METRIC)) {
    const value = currentWorkerValue(history, seriesKey);
    if (value !== null) metrics[metricKey] = value;
  }
  // Which of the worker's two GPU temperatures its newest row shows is the
  // backend's choice, sent beside the values (VD-147).
  const sensor = history?.latest?.gpu_temperature_sensor;
  if (sensor) metrics.gpu_temperature_sensor = sensor;
  const powerSensor = history?.latest?.gpu_power_sensor;
  if (powerSensor) metrics.gpu_power_sensor = powerSensor;
  return metrics;
}

/** What a worker with no agent says, pointing at the action the viewer can take. */
export function notInstalledSentence(administrator: boolean, joined: boolean): string {
  if (joined) {
    return administrator
      ? "Telemetry is not installed on this worker yet. Recheck lays it down with the worker software."
      : "Telemetry is not installed on this worker yet. An administrator's Recheck lays it down with the worker software.";
  }
  return administrator
    ? "Telemetry is not installed on this worker. Use Install telemetry below to start its readings."
    : "Telemetry is not installed on this worker. An administrator can install it from this card.";
}

const TILES: TileSpec[] = [
  {
    id: "processor_load",
    label: "Processor load",
    series: "processor_load",
    unmeasured: "Processor load has not been recorded in this window",
    reading: (metrics) => rounded(metricNumber(metrics, "cpu_percent"), "%"),
    live: (metrics) => metricNumber(metrics, "cpu_percent"),
    detail: () => "Share of the processor in use",
    scale: PERCENT_SCALE,
    show: ALWAYS,
  },
  {
    id: "memory_percent",
    label: "Memory in use",
    series: "memory_percent",
    unmeasured: "Memory use has not been recorded in this window",
    reading: (metrics) => rounded(metricNumber(metrics, "memory_percent"), "%"),
    live: (metrics) => metricNumber(metrics, "memory_percent"),
    detail: () => "Share of system memory in use",
    scale: PERCENT_SCALE,
    // Easy already shows unified memory when there is a GPU; system memory is an
    // Advanced addition there, but the headline on a GPU-less machine.
    show: (mode, gpu) => mode === "advanced" || !gpu,
  },
  {
    id: "gpu_busy_percent",
    label: "GPU utilisation",
    series: "gpu_busy_percent",
    unmeasured: "This graphics processor does not report a utilisation figure",
    reading: (metrics) => rounded(metricNumber(metrics, "gpu_busy_percent"), "%"),
    live: (metrics) => metricNumber(metrics, "gpu_busy_percent"),
    detail: () => "Share of the graphics processor in use",
    scale: PERCENT_SCALE,
    show: (mode, gpu) => gpu && mode === "advanced",
  },
  {
    id: "unified",
    label: "Unified memory",
    series: "gpu_gtt_used_bytes",
    unmeasured: "This adapter does not report how much unified memory is in use",
    reading: (metrics) => {
      const used = metricNumber(metrics, "gpu_gtt_used_bytes");
      const total = metricNumber(metrics, "gpu_gtt_total_bytes");
      if (used === null) return null;
      return {
        value: withUnit(formatQuantity(used, "used")),
        unit: total === null ? undefined : `of ${withUnit(formatQuantity(total, "capacity"))}`,
      };
    },
    live: (metrics) => metricNumber(metrics, "gpu_gtt_used_bytes"),
    detail: (metrics) => {
      const total = metricNumber(metrics, "gpu_gtt_total_bytes");
      return total === null
        ? "Shared aperture in use"
        : `${formatQuantity(total, "capacity")} shared aperture`;
    },
    scale: (metrics, points) => metricNumber(metrics, "gpu_gtt_total_bytes") ?? peakOf(points),
    show: (_mode, gpu) => gpu,
  },
  {
    id: "cpu_temperature_c",
    label: "CPU temperature",
    series: "cpu_temperature_c",
    unmeasured: "No processor temperature sensor was recorded in this window",
    reading: (metrics) => rounded(metricNumber(metrics, "cpu_temperature"), "°C"),
    live: (metrics) => metricNumber(metrics, "cpu_temperature"),
    detail: () => "Processor package temperature",
    scale: PERCENT_SCALE,
    show: ALWAYS,
  },
  {
    id: "gpu_temperature_c",
    // The reading and its name follow the sensor the backend chose: the
    // graphics engine where it was read, else the edge sensor (VD-147).
    label: (metrics) => gpuTemperature(metrics).label,
    series: (metrics) => gpuTemperature(metrics).field,
    unmeasured: "This graphics processor does not report a temperature",
    reading: (metrics) => rounded(gpuTemperature(metrics).value, "°C"),
    live: (metrics) => gpuTemperature(metrics).value,
    detail: (metrics) => (gpuTemperature(metrics).sensor === "graphics engine"
      ? "The graphics engine's own sensor"
      : "The edge sensor on the graphics processor"),
    scale: PERCENT_SCALE,
    show: (mode, gpu) => gpu && mode === "advanced",
  },
  {
    id: "gpu_power_watts",
    label: (metrics) => (gpuPowerSensor(metrics) ? `GPU power (${gpuPowerSensor(metrics)})` : "GPU power"),
    series: "gpu_power_watts",
    unmeasured: "This graphics processor's power was not read",
    reading: (metrics) => gpuPower(metricNumber(metrics, "gpu_power_watts")),
    live: (metrics) => metricNumber(metrics, "gpu_power_watts"),
    // The backend's own word for what measured it; the screen compares no word.
    detail: (metrics) => (gpuPowerSensor(metrics)
      ? `Measured at the ${gpuPowerSensor(metrics)}`
      : "The graphics processor's own draw"),
    scale: (_metrics, points) => peakOf(points),
    show: (mode, gpu) => gpu && mode === "advanced",
  },
];

/**
 * The backend's sentences about this node's readings: why its GPU figures are
 * missing, and how many readings were dropped as impossible. Shown as they
 * arrive, beneath the tiles; an empty or absent one says nothing.
 */
function ReadingNotes({ history }: { history: TelemetryHistory | null }) {
  const notes = [history?.latest?.gpu_readings_note, history?.latest?.implausible_note]
    .filter((note): note is string => typeof note === "string" && note.trim() !== "");
  return (
    <>
      {notes.map((note) => (
        <p className="cf-note" data-telemetry-note="" key={note}>{note}</p>
      ))}
    </>
  );
}

/** One tile, shared by the controller and worker grids. */
function renderTile(
  tile: TileSpec,
  history: TelemetryHistory | null,
  metrics: Metrics,
  stale: boolean,
) {
  const points = history?.series?.[typeof tile.series === "function" ? tile.series(metrics) : tile.series]?.points ?? [];
  const measured = points.some((point) => point.v !== null);
  const reading = tile.reading(metrics);
  const unmeasured = tile.live(metrics) === null && !measured;
  // A stale worker keeps its last-known trend (faded by the grid) but never
  // presents the old number as current: the headline reads "—", and the
  // detail line says the reading is stale while keeping what it names.
  const described = tile.detail(metrics);
  const detail = unmeasured
    ? tile.unmeasured
    : stale && typeof described === "string" && described
      ? "Stale: " + described.charAt(0).toLowerCase() + described.slice(1)
      : described;
  const trend: TrendBar[] = unmeasured ? [] : trendBars(points, tile.scale(metrics, points));
  const value = unmeasured ? null : stale ? "—" : reading?.value ?? "—";
  return (
    <StatTile
      foot={detail}
      key={tile.id}
      label={typeof tile.label === "function" ? tile.label(metrics) : tile.label}
      trend={trend}
      unit={stale || unmeasured ? undefined : reading?.unit}
      value={value}
    />
  );
}

/** The minutes since a stale worker last reported, for the grey "22 min old" pill. */
function staleAge(since: number | null): string | null {
  if (since === null) return null;
  const minutes = Math.max(1, Math.round((Date.now() / 1000 - since) / 60));
  return `${minutes} min old`;
}

export function MachineMetrics({
  isController,
  nodeId,
  offline,
  stateLabel,
  gpuPresent,
  mode,
  telemetryProvisioned,
  administrator = false,
  joined = false,
}: Props) {
  // Both an online controller and an online worker fetch; the controller reads
  // the no-node route (+ live current), a worker reads its own node history. A
  // worker with no resolvable id must NOT fetch — the no-node route would return
  // the controller's series, misattributing it to this worker — so it stays
  // disabled and falls through to the honest "not reporting yet" copy. A worker
  // with no agent installed has nothing to read either.
  const notInstalled = !isController && telemetryProvisioned === false;
  const enabled = !offline && !notInstalled && (isController || Boolean(nodeId));
  const { history, historyError, loading, current } = useMachineMetrics(
    enabled,
    isController ? undefined : nodeId,
  );

  if (offline) {
    return (
      <section className="cf-telemetry" aria-label="Node metrics">
        <p className="cf-note">
          Live metrics are unavailable while this machine is {stateLabel.toLowerCase()}.
        </p>
      </section>
    );
  }

  const note = retentionNote(history?.retention);
  const tiles = TILES.filter((tile) => tile.show(mode, gpuPresent));

  if (isController) {
    const metrics = current ?? {};
    return (
      <section className="cf-telemetry" aria-label="Controller metrics">
        <div className="cf-telemetry__head">
          <span className="cl-eyebrow">Controller telemetry</span>
          <span className="cf-telemetry__window">
            {history ? windowLabel(history.window_seconds) : "recent"} on this controller
          </span>
        </div>

        {note && <p className="cf-note">{note}</p>}
        {/* ACC-082: the figures are this controller's live snapshot and the
            trends are its own history - never an average over the fleet. */}
        {history?.available && (
          <p className="cf-note">Figures are live; trends are this controller&apos;s own history.</p>
        )}

        {loading && !history ? (
          <p className="cf-note">Reading recent metrics…</p>
        ) : historyError ? (
          <p className="cf-note">Recent metrics could not be read: {historyError}</p>
        ) : history && !history.available ? (
          <p className="cf-note">{history.reason}</p>
        ) : (
          <>
            <div className="cf-telemetry__tiles">
              {tiles.map((tile) => renderTile(tile, history, metrics, false))}
            </div>
            <ReadingNotes history={history} />
          </>
        )}
      </section>
    );
  }

  // Worker: everything below derives from its own node-tagged history — never
  // from the controller's `/telemetry/current` — through the one freshness rule
  // the collapsed card and the fleet headline share.
  const freshness = workerFreshness(history);
  const anyMeasured = history
    ? Object.values(history.series).some((s) => s.points.some((point) => point.v !== null))
    : false;
  const stale = freshness.state !== "fresh";
  const liveMetrics = latestMetrics(history);
  const statusLine =
    freshness.state === "clock-refused"
      ? freshness.reason
      : freshness.state === "stale" && freshness.since !== null
        ? `Not reporting since ${exactTime(freshness.since * 1000)}.`
        : "Not reporting recently.";
  const oldBy = freshness.state === "stale" ? staleAge(freshness.since) : null;
  const showTiles = !notInstalled && !(loading && !history) && !historyError
    && !(history && !history.available) && !(freshness.state === "never" && !anyMeasured);

  return (
    <section className="cf-telemetry" aria-label="Worker metrics">
      <div className="cf-telemetry__head">
        <span className="cl-eyebrow">Worker telemetry</span>
        <span className="cf-telemetry__window">
          <span>{history ? windowLabel(history.window_seconds) : "recent"}</span>
          {showTiles && freshness.state === "fresh" && (
            <>
              <span aria-hidden="true">·</span>
              <span>{freshnessLabel(freshness.ageSeconds)}</span>
            </>
          )}
          {showTiles && oldBy && <StatusPill label={oldBy} reading="stale" />}
        </span>
      </div>

      {note && <p className="cf-note">{note}</p>}

      {notInstalled ? (
        <p className="cf-note">{notInstalledSentence(administrator, joined)}</p>
      ) : loading && !history ? (
        <p className="cf-note">Reading recent metrics…</p>
      ) : historyError ? (
        <p className="cf-note">Recent metrics could not be read: {historyError}</p>
      ) : history && !history.available ? (
        <p className="cf-note">{history.reason}</p>
      ) : freshness.state === "never" && !anyMeasured ? (
        <p className="cf-note">This worker is not reporting telemetry yet.</p>
      ) : (
        <>
          {/* A clock drifting toward refusal is said before posts are refused. */}
          {freshness.state === "fresh"
            ? history?.ingest?.reason && <p className="cf-note">{history.ingest.reason}</p>
            : <p className="cf-telemetry__stale" role="status">{statusLine}</p>}
          <div className={stale ? "cf-telemetry__tiles cf-telemetry__tiles--stale" : "cf-telemetry__tiles"}>
            {tiles.map((tile) => renderTile(tile, history, liveMetrics, stale))}
          </div>
          <ReadingNotes history={history} />
        </>
      )}
    </section>
  );
}
