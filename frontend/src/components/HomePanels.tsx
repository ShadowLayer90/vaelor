import type { Device, Health, Metrics } from "../types";
import { formatPercent, formatQuantity, formatTemperature, timeAgo } from "../lib/format";
import { gpuTemperature } from "../lib/gpuTemperature";
import {
  attentionItems,
  capacityFor,
  historySeries,
  hottestSeries,
  seriesPeak,
  servingRows,
  trendBars,
  type MachineRow,
  type Pill,
} from "../lib/homeSummary";
import { thermalLimits, type MachineProfile } from "../lib/machine";
import { metricNumber } from "../lib/metrics";
import { routeHref } from "../lib/navigation";
import { DEPLOYMENTS_HREF } from "../lib/clusterSections";
import { headlineFan, parseBoardSensors } from "../lib/wmiSensors";
import { currentWorkerValue } from "../lib/workerFreshness";
import type { HomeSummary } from "../hooks/useHomeSummary";
import { Icon, ICON_SIZE } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Card, DataTable, EmptyState, ListRow, LoadingLines, MeterBar, StatTile } from "./ui";

/*
 * Home's redesigned panels (VD-200, the Main board): four stat tiles, What's
 * serving, Needs your attention and Machines. Each reads only what a route
 * already serves (lib/homeSummary.ts); a reading that was not taken says so.
 */

const NOT_READ = "Not read";

/** Each tile opens its readings in full: System › Compute (the design guide's map). */
const COMPUTE_HREF = "#/system/compute";

function pill(value: Pill) {
  return <StatusPill label={value.label} reading={value.reading} tone={value.tone} />;
}

/** "33.9 GB" -> ["33.9", "GB"], so a tile can draw the unit small. */
function splitQuantity(text: string): [string, string] {
  const match = text.match(/^([\d.,]+)\s*(.*)$/);
  return match ? [match[1], match[2]] : [text, ""];
}

export function HomeStatTiles({
  device,
  live,
  machine,
  metrics,
  summary,
}: {
  device: Device | null;
  /** False when there is no fresh sample: the tiles then say "Not read". */
  live: boolean;
  machine: MachineProfile;
  metrics: Metrics;
  summary: HomeSummary;
}) {
  const cpu = live ? metricNumber(metrics, "cpu_percent") : null;
  const cores = device?.platform?.board.cpu_cores ?? metricNumber(metrics, "cpu_cores");
  const cpuPeak = seriesPeak(historySeries(summary.history, "processor_load"));
  const used = live ? metricNumber(metrics, "memory_used") : null;
  const total = metricNumber(metrics, "memory_total") ?? device?.platform?.board.memory_total_bytes ?? null;
  const [usedValue, usedUnit] = used === null ? ["", ""] : splitQuantity(formatQuantity(used, "used"));
  const gpuPresent = machine.capabilities.gpu.available;
  const gpuBusy = live ? metricNumber(metrics, "gpu_busy_percent") : null;
  // A foot is a reading too: an old sample prints none of it (LESSONS 5).
  const gttUsed = live ? metricNumber(metrics, "gpu_gtt_used_bytes") : null;
  const gttTotal = metricNumber(metrics, "gpu_gtt_total_bytes");
  const watts = live ? metricNumber(metrics, "gpu_power_watts") : null;
  const temperatures = [metricNumber(metrics, "cpu_temperature"), gpuTemperature(metrics).value]
    .filter((value): value is number => value !== null);
  const hottest = live && temperatures.length ? Math.max(...temperatures) : null;
  const fan = headlineFan(parseBoardSensors(metrics));
  const fanRpm = live ? metricNumber(metrics, "pwm_fan_speed") ?? fan?.rpm ?? null : null;
  const warn = thermalLimits(machine).cpuWarn;
  const history = summary.history.state === "ok" && summary.history.data.available;
  return (
    <div className="home-tiles">
      <StatTile
        href={COMPUTE_HREF}
        foot={cpuPeak !== null ? `Peak ${Math.round(cpuPeak)}% in the last hour` : history ? "No reading in the last hour" : "Last hour not read"}
        label="CPU"
        side={cores ? `${cores} cores` : undefined}
        trend={trendBars(historySeries(summary.history, "processor_load"), 100)}
        unit="%"
        value={cpu === null ? null : String(Math.round(cpu))}
      />
      <StatTile
        href={COMPUTE_HREF}
        foot={used !== null && total !== null ? `${formatQuantity(Math.max(0, total - used), "free")} free` : NOT_READ}
        label="Memory"
        side={total !== null ? formatQuantity(total, "capacity") : undefined}
        trend={trendBars(historySeries(summary.history, "memory_percent"), 100)}
        unit={usedUnit}
        value={used === null ? null : usedValue}
      />
      {gpuPresent ? (
        <StatTile
          href={COMPUTE_HREF}
          foot={[
            gttUsed !== null ? `${formatQuantity(gttUsed, "used")} used` : "",
            watts !== null ? `${Math.round(watts)} W` : "",
          ].filter(Boolean).join(" · ") || NOT_READ}
          label="GPU"
          // The engine's name is on the hardware chips, once (overviewTruthfulness);
          // the tile's side note is the shared pool it draws from.
          side={gttTotal !== null ? `${formatQuantity(gttTotal, "capacity")} pool` : undefined}
          trend={trendBars(historySeries(summary.history, "gpu_busy_percent"), 100)}
          unit="%"
          value={gpuBusy === null ? null : String(Math.round(gpuBusy))}
        />
      ) : (
        <StatTile foot="No graphics processor was found on this machine" label="GPU" value="None" />
      )}
      <StatTile
        href={COMPUTE_HREF}
        foot={[
          fanRpm !== null ? `${fan?.label ?? "Fan"} ${Math.round(fanRpm)} RPM` : "",
          hottest === null ? "" : hottest >= warn ? `above ${warn} °C` : "normal range",
        ].filter(Boolean).join(" · ") || NOT_READ}
        label="Temperature"
        side="hottest sensor"
        // The trend follows the value's rule: each bucket's hotter of CPU and GPU.
        trend={trendBars(hottestSeries(historySeries(summary.history, "cpu_temperature_c"), historySeries(summary.history, "gpu_temperature_c")), 100)}
        unit="°C"
        value={hottest === null ? null : String(Math.round(hottest))}
      />
    </div>
  );
}

/** "a", "a and b", "a, b and c". */
function listWords(words: string[]): string {
  return words.length < 2 ? words.join("") : `${words.slice(0, -1).join(", ")} and ${words[words.length - 1]}`;
}

// The approved glyphs (VD-200): AI Chat and the Assistant as in the rail, the LLM Server as a server.
const SERVING_ICONS = { memory: "chat", cpu: "assistant", network: "server" } as const;

export function HomeServing({ summary }: { summary: HomeSummary }) {
  return (
    <Card actions={<a href={DEPLOYMENTS_HREF}>Open Cluster</a>} as="section" className="home-panel" flush heading="What's serving">
      {servingRows(summary).map((row) => (
        <ListRow
          detail={row.detail}
          icon={<Icon name={SERVING_ICONS[row.icon]} size={ICON_SIZE.nav} />}
          iconAccent={row.accent}
          key={row.name}
          title={<a className="home-row-link" href={row.href}>{row.name}</a>}
          trailing={pill(row.pill)}
        />
      ))}
    </Card>
  );
}

export function HomeAttention({ health, summary }: { health: Health; summary: HomeSummary }) {
  const { items, total, unread } = attentionItems(summary, health);
  const loading = summary.attention.state === "loading";
  // A count that leaves a source out is a floor, not a total (LESSONS 8).
  const unreadNote = unread.length ? `Not read: ${listWords(unread)}.` : "";
  return (
    <Card
      actions={total === null ? <StatusPill label={NOT_READ} reading="unread" />
        : unread.length ? <StatusPill description={unreadNote} label={total > 0 ? `At least ${total}` : "Partly read"} reading="unread" />
          : <span className="home-count">{total}</span>}
      as="section"
      className="home-panel"
      flush
      heading="Needs your attention"
    >
      {loading ? <LoadingLines label="Reading what needs attention" lines={3} />
        : items.length ? items.slice(0, 5).map((item) => (
          <ListRow
            align="start"
            detail={item.text}
            icon={<Icon name={item.icon} size={ICON_SIZE.nav} />}
            iconAccent={item.accent}
            key={item.key}
            title={<a className="home-row-link" href={item.href}>{item.title}</a>}
            trailing={item.at !== null ? timeAgo(item.at) : undefined}
          />
        )) : total === null
          ? <EmptyState icon={<Icon name="alert" size={ICON_SIZE.nav} />} text="The operations list could not be read, so this is not a clean bill." title="Not read" />
          : unread.length
            ? <EmptyState icon={<Icon name="alert" size={ICON_SIZE.nav} />} text={`${unreadNote} Nothing that was read needs you, but this is not a clean bill.`} title="Nothing in what was read" />
            : <EmptyState icon={<Icon name="shield" size={ICON_SIZE.nav} />} text="No failed or waiting operation, and no warning from the machines Vaelor reads." title="Nothing needs you" />}
      {!loading && items.length > 0 && unread.length > 0 && <p className="home-panel__note">{unreadNote}</p>}
      {total !== null && total > Math.min(items.length, 5) && (
        <ListRow detail={`${total - Math.min(items.length, 5)} more in Activity`} title={<a className="home-row-link" href={routeHref("activity")}>See all</a>} />
      )}
    </Card>
  );
}

function workerRows(summary: HomeSummary): MachineRow[] {
  if (summary.cluster.state !== "ok") return [];
  const ledger = summary.capacity.state === "ok" ? summary.capacity.data.nodes ?? [] : [];
  return (summary.cluster.data.enrolled_nodes ?? []).map((node) => {
    const read = summary.workers[node.id];
    const history = read?.state === "ok" ? read.data : null;
    const value = (key: string) => currentWorkerValue(history, key);
    const capacity = capacityFor(ledger, node);
    const gtt = value("gpu_gtt_used_bytes");
    const gttTotal = capacity?.capacity.gpu.gtt_total_bytes || null;
    const cpu = value("processor_load");
    const temperature = value("cpu_temperature_c");
    const software = node.worker_software;
    return {
      key: node.id,
      name: node.name,
      hardware: [node.inventory?.cpu_count ? `${node.inventory.cpu_count} cores` : "", node.inventory?.memory_bytes ? formatQuantity(node.inventory.memory_bytes, "capacity") : ""].filter(Boolean).join(" · "),
      role: "Worker",
      cpu: cpu === null ? NOT_READ : formatPercent(cpu),
      gpuFraction: gtt !== null && gttTotal ? gtt / gttTotal : null,
      gpu: gtt !== null && gttTotal ? `${formatQuantity(gtt, "used")} / ${formatQuantity(gttTotal, "capacity")}` : NOT_READ,
      temperature: temperature === null ? NOT_READ : formatTemperature(temperature),
      software: software ? { label: software.label, tone: software.tone, reading: software.stale ? "stale" : undefined } : { label: "Not checked", tone: "neutral", reading: "unread" },
    };
  });
}

export function HomeMachines({
  device,
  live,
  machine,
  metrics,
  summary,
}: {
  device: Device | null;
  live: boolean;
  machine: MachineProfile;
  metrics: Metrics;
  summary: HomeSummary;
}) {
  const clustered = summary.cluster.state === "ok" && summary.cluster.data.controller?.initialized === true;
  // The role is the cluster's to say: unread, it is "Not read", never "Standalone" (LESSONS 8).
  const role = summary.cluster.state === "ok" ? (clustered ? "Controller" : "Standalone")
    : summary.cluster.state === "loading" ? "Checking" : NOT_READ;
  const gtt = live ? metricNumber(metrics, "gpu_gtt_used_bytes") : null;
  const gttTotal = metricNumber(metrics, "gpu_gtt_total_bytes");
  const memory = device?.platform?.board.memory_total_bytes ?? metricNumber(metrics, "memory_total");
  const self: MachineRow = {
    key: "this-machine",
    name: device?.name ?? "Detecting hardware",
    // The processor, not the graphics engine: the engine is named once, on
    // the hardware chips (overviewTruthfulness).
    hardware: [device?.platform?.board.cpu_model ?? "", memory ? formatQuantity(memory, "capacity") : ""].filter(Boolean).join(" · "),
    role,
    cpu: live && metricNumber(metrics, "cpu_percent") !== null ? formatPercent(metrics.cpu_percent) : NOT_READ,
    gpuFraction: gtt !== null && gttTotal ? gtt / gttTotal : null,
    gpu: gtt !== null && gttTotal ? `${formatQuantity(gtt, "used")} / ${formatQuantity(gttTotal, "capacity")}` : machine.capabilities.gpu.available ? NOT_READ : "No GPU",
    temperature: live && metricNumber(metrics, "cpu_temperature") !== null ? formatTemperature(metrics.cpu_temperature) : NOT_READ,
    software: device?.version ? { label: `Vaelor ${device.version}`, tone: "neutral" } : { label: NOT_READ, tone: "neutral", reading: "unread" },
  };
  const rows = [self, ...workerRows(summary)];
  return (
    <Card
      actions={summary.cluster.state === "ok" ? <a className="ui-button ui-button--secondary" href="#/fleet">Add a machine</a> : undefined}
      as="section"
      className="home-panel"
      flush
      heading="Machines"
    >
      <DataTable
        caption="Machines"
        columns={[
          { key: "machine", label: "Machine" }, { key: "role", label: "Role" }, { key: "cpu", label: "CPU" },
          { key: "gpu", label: "GPU memory" }, { key: "temperature", label: "Temperature" }, { key: "software", label: "Software" },
        ]}
        rows={rows.map((row) => ({
          key: row.key,
          cells: {
            machine: <><div className="home-machine-name">{row.name}</div>{row.hardware && <div className="ui-small ui-muted">{row.hardware}</div>}</>,
            role: <span className="ui-muted">{row.role}</span>,
            cpu: <span className="ui-mono">{row.cpu}</span>,
            gpu: <MeterBar fraction={row.gpuFraction} label={row.gpu} />,
            temperature: <span className="ui-mono">{row.temperature}</span>,
            software: pill(row.software),
          },
        }))}
      />
      {summary.cluster.state === "forbidden" && (
        <p className="home-panel__note">Other machines are listed for operators and administrators.</p>
      )}
      {summary.cluster.state === "failed" && (
        <p className="home-panel__note">The cluster could not be read just now, so any other machines are not listed.</p>
      )}
    </Card>
  );
}
