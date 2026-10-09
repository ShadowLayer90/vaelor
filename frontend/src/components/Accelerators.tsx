import { useEffect, useId, useState, type ReactNode } from "react";
import { apiRequest } from "../lib/api";
import type { Metrics, Role, TelemetrySample } from "../types";
import type { MachineProfile } from "../lib/machine";
import type { AccelerationReading } from "../lib/acceleration";
import { parseMachineHardware, type NeuralAcceleratorSummary } from "../lib/accelerators";
import { sampleTimeMs, STALE_AFTER_MS } from "../lib/connectionState";
import { formatQuantity, timeAgo } from "../lib/format";
import { gpuTemperature } from "../lib/gpuTemperature";
import { metricNumber, metricSeries } from "../lib/metrics";
import { shortModelName } from "../lib/homeSummary";
import { replicaCount } from "../lib/gpuServingMode";
import type { LlmServerSurface } from "../lib/endpoints";
import { NOT_ANSWERING, type StatusTone } from "./ui/status";
import { CONTROLLER_PLACEMENT_ID, type FleetSummary } from "./fleetTypes";
import { AccelerationVerdict } from "./AccelerationVerdict";
import { FactGrid, type EngineFact } from "./AcceleratorCard";
import { graphicsFacts, neuralFacts } from "./ComputePanel";
import { Icon, ICON_SIZE, type IconName } from "./Icon";
import { StatusPill } from "./StatusPill";
import { readingBars } from "./SystemLiveReadings";
import { Button, MeterBar, UnavailableValue } from "./ui";

/** How often the engine reading is refreshed while the page is open. */
export const ENGINE_REFRESH_MS = 30_000;

/** How many bars a stat cell's trend draws (the SystemCompute board). */
const STAT_BARS = 12;

/** One engine as `/inference/status` reports it (`vaelor.inference_status`). */
export interface InferenceEngine {
  kind: string;
  /** `flm-real`, `llama.cpp`, or `vllm` when this adapter serves a cluster deployment. */
  backend?: string | null;
  model?: string | null;
  cluster_deployment?: string | null;
  context_tokens?: number;
  /** `ready` answered, `unreachable` did not, `unknown` was never asked. */
  health?: { state?: string; detail?: string };
  acceleration?: AccelerationReading | null;
}

/**
 * The GPU engine's acceleration reading, if this machine has one to report.
 * An engine list with no GPU tier, and a GPU tier that reported nothing, are
 * both "no reading" and neither is a fault.
 */
export function gpuAcceleration(engines: InferenceEngine[] | null): AccelerationReading | null {
  return engines?.find((engine) => engine.kind === "gpu")?.acceleration ?? null;
}

/** The neural processor's serving state as `/system/machine` last answered it, and when. */
interface NeuralReading {
  npu: NeuralAcceleratorSummary | null;
  /** Milliseconds since the epoch. */
  at: number;
}

/** `pending` before the first answer; `failed` keeps the last good reading so its age can be said. */
type NeuralRead =
  | { state: "pending" }
  | { state: "ok"; reading: NeuralReading }
  | { state: "failed"; last: NeuralReading | null };

/**
 * `/inference/status` and the neural processor's serving state from
 * `/system/machine`, re-read together while the page is open. Engines are
 * `null` when they could not be read. The serving state is re-read here, not
 * taken from the page's one discovery read, because its pills are green and
 * green means read just now (the States board).
 */
function useEngineReads(): { engines: InferenceEngine[] | null; neural: NeuralRead } {
  const [engines, setEngines] = useState<InferenceEngine[] | null>(null);
  const [neural, setNeural] = useState<NeuralRead>({ state: "pending" });
  useEffect(() => {
    let cancelled = false;
    const read = () => {
      void apiRequest<{ engines: InferenceEngine[] }>("/inference/status")
        .then((status) => { if (!cancelled) setEngines(status.engines ?? []); })
        // A reading that could not be taken says nothing, and nothing is put in its place.
        .catch(() => { if (!cancelled) setEngines(null); });
      void apiRequest<unknown>("/system/machine")
        .then((payload) => {
          if (cancelled) return;
          const npu = parseMachineHardware(payload).neuralAccelerators[0] ?? null;
          setNeural({ state: "ok", reading: { npu, at: Date.now() } });
        })
        .catch(() => {
          if (cancelled) return;
          setNeural((current) => ({
            state: "failed",
            last: current.state === "ok" ? current.reading : current.state === "failed" ? current.last : null,
          }));
        });
    };
    read();
    // Re-read so a GPU clustered or released while this page is open is
    // described by the engine now using it (ACC-114). Not while hidden.
    const timer = window.setInterval(() => { if (!document.hidden) read(); }, ENGINE_REFRESH_MS);
    const onShown = () => { if (!document.hidden) read(); };
    document.addEventListener("visibilitychange", onShown);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onShown);
    };
  }, []);
  return { engines, neural };
}

/**
 * The live sample, when no shell hands one over (System opened on its own):
 * `/telemetry/current` every five seconds, with no history - so no trend.
 * `sampledAt` is the sample's own time, so its age is judged by the shell's
 * one rule (`STALE_AFTER_MS`).
 */
function useOwnSample(enabled: boolean): { metrics: Metrics; sampledAt: number } {
  const [sample, setSample] = useState<{ metrics: Metrics; sampledAt: number }>({ metrics: {}, sampledAt: 0 });
  useEffect(() => {
    if (!enabled) return undefined;
    let cancelled = false;
    const read = () => {
      void apiRequest<TelemetrySample>("/telemetry/current")
        .then((next) => { if (!cancelled) setSample({ metrics: next.metrics ?? {}, sampledAt: next.sampled_at ?? 0 }); })
        .catch(() => undefined);
    };
    read();
    const timer = window.setInterval(read, 5_000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [enabled]);
  return sample;
}

/**
 * `/llm-server`, which only administrators may read: no one else is sent to
 * ask. `null` for them, and on a failed read.
 */
function useLlmServer(role: Role): LlmServerSurface | null {
  const [surface, setSurface] = useState<LlmServerSurface | null>(null);
  useEffect(() => {
    if (role !== "administrator") return undefined;
    let cancelled = false;
    void apiRequest<LlmServerSurface>("/llm-server")
      .then((next) => { if (!cancelled) setSurface(next); })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, [role]);
  return surface;
}

const ENGINE_NAMES: Record<string, string> = { vllm: "vLLM", "llama.cpp": "llama.cpp", "flm-real": "FastFlowLM" };

interface Stat {
  label: string;
  value: string | null;
  unit: string;
  /** Why there is no value, printed under "Not read". */
  reason: string;
  /** Which sensor or field the value came from. */
  note: string | null;
  /** The recorded samples behind the value; no trend is drawn without them. */
  series: Array<number | null>;
  /** A natural ceiling for the trend (100 for a percentage), or null to scale to the samples. */
  scale: number | null;
}

type Pill = { label: string; tone: StatusTone; reading?: "unread" | "stale" };

interface Running {
  name: string;
  detail: string;
  pill: Pill;
}

interface Accelerator {
  key: "gpu" | "npu";
  icon: IconName;
  title: string;
  sub: string;
  pill: Pill;
  stats: Stat[];
  memory?: { label: string; fraction: number | null; value: ReactNode; note: string };
  running: Running[];
  /** Why "Running on it" could not be read, when it could not; null when it was read. */
  runningUnread: string | null;
  detailsLabel: string;
  facts: EngineFact[];
  extra?: ReactNode;
}

/** How old the live sample is, when it is too old to show as live; null while it is live. */
type SampleAge = string | null;

const pick = (facts: EngineFact[], labels: string[]) => facts.filter((fact) => labels.includes(fact.label));
const notServing: Pill = { label: "Not serving", tone: "neutral" };
const notRead: Pill = { label: "Not read", tone: "neutral", reading: "unread" };
const notChecked: Pill = { label: "Not checked", tone: "neutral", reading: "unread" };
const ENGINES_UNREAD = "The list of running models could not be read just now.";

/** How long a serving reading counts as "just now": one missed re-read, no more. */
const SERVING_FRESH_MS = ENGINE_REFRESH_MS * 2;

/** The roles the GPU's model has, from what each surface says it reaches. */
function gpuRoles(engine: InferenceEngine, llmServer: LlmServerSurface | null): string {
  const roles: string[] = [];
  // The GPU tier is probed through AI Chat's own lease; "unknown" means no lease reached it.
  if (engine.health?.state === "ready" || engine.health?.state === "unreachable") roles.push("AI Chat");
  const wanted = engine.cluster_deployment ? "cluster" : "managed-local";
  if (llmServer?.enabled && llmServer.target_kind === wanted && llmServer.runtime?.state === "serving") {
    roles.push("the LLM Server");
  }
  return roles.join(" and ");
}

/** "copy 1 of 2 in the cluster" for a replicated deployment, "split across 2 machines" for a sharded one. */
function clusterPlace(engine: InferenceEngine, cluster: FleetSummary | null): string | null {
  if (!engine.cluster_deployment) return null;
  const record = cluster?.pooled_deployments?.find((row) => row.name === engine.cluster_deployment);
  if (!record) return "in the GPU cluster";
  const nodes = record.node_ids ?? [];
  const replicas = replicaCount(record);
  if (replicas > 0) {
    const index = nodes.indexOf(CONTROLLER_PLACEMENT_ID);
    return index >= 0 ? `copy ${index + 1} of ${replicas} in the cluster` : `one of ${replicas} copies in the cluster`;
  }
  return nodes.length > 1 ? `split across ${nodes.length} machines` : "in the GPU cluster";
}

function gpuAccelerator(
  machine: MachineProfile, metrics: Metrics, history: TelemetrySample[],
  engines: InferenceEngine[] | null, cluster: FleetSummary | null, llmServer: LlmServerSurface | null,
  age: SampleAge,
): Accelerator {
  const adapter = machine.hardware.accelerators[0];
  const rocm = machine.hardware.graphics?.rocmVersion;
  const engine = engines?.find((entry) => entry.kind === "gpu");
  const temperature = gpuTemperature(metrics);
  const clock = metricNumber(metrics, "gpu_clock_mhz");
  const watts = metricNumber(metrics, "gpu_power_watts");
  const busy = metricNumber(metrics, "gpu_busy_percent");
  const powerSensor = typeof metrics.gpu_power_sensor === "string" ? metrics.gpu_power_sensor : null;
  const gttUsed = metricNumber(metrics, "gpu_gtt_used_bytes");
  const gttTotal = metricNumber(metrics, "gpu_gtt_total_bytes");
  const vramTotal = metricNumber(metrics, "gpu_vram_total_bytes");
  const vramUsed = metricNumber(metrics, "gpu_vram_used_bytes");
  // The pool that fills: the shared aperture where the driver reports one,
  // else the carve-out, named as a carve-out. Never `?? 0` (LESSONS 5), and a
  // size of zero is no size: "29.3 GB of 0 B" is not a reading.
  const pool = gttTotal !== null && gttTotal > 0
    ? { noun: "Shared memory", used: gttUsed, total: gttTotal,
      missing: "This adapter publishes the size of its shared aperture but not how much of it is in use",
      note: ["Unified memory the GPU borrows from the system",
        adapter?.unifiedMemory && vramTotal !== null ? `reserved video memory ${formatQuantity(vramTotal, "capacity")}, a firmware setting` : null]
        .filter(Boolean).join(" · ") }
    : vramTotal !== null && vramTotal > 0
      ? { noun: "Reserved video memory", used: vramUsed, total: vramTotal,
        missing: "This adapter publishes the size of its reserved video memory but not how much of it is in use",
        note: "The adapter's own video memory" }
      : null;
  const health = engine?.health?.state;
  const serving = Boolean(engine?.model) && health === "ready";
  const running: Running[] = engine?.model ? [{
    name: shortModelName(engine.model),
    detail: [ENGINE_NAMES[engine.backend ?? ""] ?? engine.backend, gpuRoles(engine, llmServer), clusterPlace(engine, cluster)]
      .filter(Boolean).join(" · "),
    pill: health === "ready" ? { label: "Serving", tone: "success" }
      : health === "unreachable" ? NOT_ANSWERING
        : notChecked,
  }] : [];
  const poolMissing = "This adapter did not report the size of its memory pool";
  return {
    key: "gpu",
    icon: "gpu",
    title: "Graphics",
    sub: [adapter?.name ?? "Graphics processor", adapter?.driver, rocm ? `ROCm ${rocm}` : null].filter(Boolean).join(" · "),
    // Green only on the probe's own "answered" (LESSONS 8): never on a lease or
    // a plan alone. A model whose probe was never asked is "Not checked".
    pill: engines === null ? notRead
      : serving ? { label: "Serving AI Chat", tone: "success" }
        : health === "unreachable" ? NOT_ANSWERING
          : engine?.model ? notChecked : notServing,
    stats: [
      { label: "Busy", value: busy === null ? null : `${Math.round(busy)}`, unit: "%", reason: "This adapter does not report utilisation",
        note: "graphics engine", series: metricSeries(history, "gpu_busy_percent"), scale: 100 },
      { label: "Temperature", value: temperature.value === null ? null : `${Math.round(temperature.value)}`, unit: "°C",
        reason: "No temperature sensor was read for this adapter", note: temperature.sensor ? `${temperature.sensor} sensor` : null,
        series: metricSeries(history, temperature.field), scale: 100 },
      { label: "Power", value: watts === null ? null : `${Math.round(watts)}`, unit: "W", reason: "This adapter does not report its power draw",
        note: powerSensor, series: metricSeries(history, "gpu_power_watts"), scale: null },
      { label: "Clock", value: clock === null ? null : (clock / 1000).toFixed(2), unit: "GHz", reason: "This adapter does not report its clock",
        note: "graphics clock", series: metricSeries(history, "gpu_clock_mhz"), scale: null },
    ],
    memory: pool === null ? {
      label: `${adapter?.unifiedMemory === false ? "Video" : "Shared"} memory in use`,
      fraction: null,
      value: <UnavailableValue label="Graphics memory use unavailable" mark="Not read" reason={poolMissing} />,
      note: poolMissing,
    } : {
      label: `${pool.noun} in use`,
      fraction: pool.used === null || pool.total <= 0 ? null : pool.used / pool.total,
      value: pool.used === null
        ? <><UnavailableValue label={`${pool.noun} use unavailable`} mark="Not read" reason={pool.missing} />{` of ${formatQuantity(pool.total, "capacity")}`}</>
        : `${formatQuantity(pool.used, "used")} of ${formatQuantity(pool.total, "capacity")}`,
      note: age === null ? pool.note : `Last read ${age}`,
    },
    running,
    runningUnread: engines === null ? ENGINES_UNREAD : null,
    detailsLabel: "Driver, PCI id, ROCm and graphics userspace",
    facts: pick(graphicsFacts(machine, metrics), ["Driver", "PCI id", "ROCm", "Graphics userspace (Mesa)", "Reserved video memory"]),
    extra: <AccelerationVerdict acceleration={gpuAcceleration(engines)} headingLevel="h4" title="Local AI on this adapter" />,
  };
}

/**
 * The NPU's serving pills from its re-read serving state: green only on a
 * reading taken within one re-read; an older one keeps its state's words, grey,
 * with its age ("Serving the Assistant · read 1 min ago"); none at all is
 * "Not read" (the States board).
 */
function neuralServing(neural: NeuralRead, now: number): {
  reading: NeuralReading | null;
  pillFor: (current: Pill) => Pill;
} {
  const reading = neural.state === "ok" ? neural.reading : neural.state === "failed" ? neural.last : null;
  const fresh = neural.state === "ok" && now - neural.reading.at <= SERVING_FRESH_MS;
  if (fresh) return { reading, pillFor: (current) => current };
  if (reading) {
    const age = timeAgo(reading.at, now);
    return { reading, pillFor: (current) => ({ label: `${current.label} · read ${age}`, tone: "neutral", reading: "stale" }) };
  }
  const unread: Pill = neural.state === "pending" ? { label: "Checking", tone: "neutral", reading: "unread" } : notRead;
  return { reading, pillFor: () => unread };
}

function npuAccelerator(
  machine: MachineProfile, metrics: Metrics, history: TelemetrySample[], engines: InferenceEngine[] | null,
  neural: NeuralRead, now: number,
): Accelerator {
  const { reading, pillFor } = neuralServing(neural, now);
  // The device's identity is the page's discovery read; whether it serves is the re-read.
  const npu = reading?.npu ?? machine.hardware.neuralAccelerators[0];
  const engine = engines?.find((entry) => entry.kind === "npu");
  // `npu_serving_state` decides serving: the Assistant's NPU lease, then the
  // same reachability probe the Assistant's own status takes (ACC-100).
  const serving = reading?.npu?.servingAssistant === true;
  const down = reading?.npu?.assistantDown ?? null;
  const activity = metricNumber(metrics, "npu_activity_percent");
  const power = metricNumber(metrics, "npu_power_watts");
  const clock = metricNumber(metrics, "npu_clock_mhz");
  // One reason for every amd-smi reading that is missing, in the backend's words (VD-040).
  const gap = typeof metrics.npu_reading_reason === "string" && metrics.npu_reading_reason
    ? metrics.npu_reading_reason : "Not reported yet";
  const model = reading?.npu?.servingModel ?? down?.model ?? null;
  // The window as the plan states it, exactly: "16K" would round a 32,768 window to 33K.
  const context = engine?.context_tokens ? `${engine.context_tokens.toLocaleString("en-GB")}-token context` : null;
  const servingUnread = reading === null
    ? "Whether the Assistant runs on it could not be read just now."
    : engines === null && !serving && !down ? ENGINES_UNREAD : null;
  return {
    key: "npu",
    icon: "npu",
    title: "Neural processor",
    sub: [npu?.name ?? "Neural accelerator", npu?.driver,
      (serving || down) ? "FastFlowLM" : null].filter(Boolean).join(" · "),
    pill: pillFor(serving ? { label: "Serving the Assistant", tone: "success" } : down ? NOT_ANSWERING : notServing),
    stats: [
      { label: "Busy", value: activity === null ? null : `${Math.round(activity)}`, unit: "%", reason: gap,
        note: "from amd-smi", series: metricSeries(history, "npu_activity_percent"), scale: 100 },
      // Nothing publishes one: amdxdna has no hwmon sensor and Vaelor reads no NPU field for it.
      { label: "Temperature", value: null, unit: "", reason: "Vaelor reads no temperature for this device", note: null, series: [], scale: null },
      { label: "Power", value: power === null ? null : power.toFixed(2), unit: "W", reason: gap,
        note: "from amd-smi", series: metricSeries(history, "npu_power_watts"), scale: null },
      { label: "Clock", value: clock === null ? null : `${Math.round(clock)}`, unit: "MHz", reason: gap,
        note: "from amd-smi", series: metricSeries(history, "npu_clock_mhz"), scale: null },
    ],
    running: model && (serving || down) ? [{
      name: model,
      detail: ["FastFlowLM", "the Assistant", context].filter(Boolean).join(" · "),
      pill: pillFor(serving ? { label: "Answering", tone: "success" } : NOT_ANSWERING),
    }] : [],
    runningUnread: servingUnread,
    detailsLabel: "Driver, device, firmware version",
    facts: pick(neuralFacts(machine, metrics), ["Driver", "Device", "Firmware version"]),
  };
}

function StatCell({ age, stat }: { age: SampleAge; stat: Stat }) {
  const bars = stat.value === null ? [] : readingBars(stat.series, stat.scale).slice(-STAT_BARS);
  // An old sample is grey and gives its age in words, never shown as live (the States board).
  const old = age !== null && stat.value !== null;
  return (
    <div className={old ? "accel-stat accel-stat--stale" : "accel-stat"}>
      <span className="accel-stat__label">{stat.label}</span>
      {stat.value === null
        ? <UnavailableValue className="accel-stat__unread" label={`${stat.label} unavailable`} mark="Not read" reason={stat.reason} />
        : <div className="accel-stat__value"><strong>{stat.value}</strong>{stat.unit && <span>{stat.unit}</span>}</div>}
      {/* A trend only from recorded samples; a value with none is drawn alone (LESSONS 5). */}
      {bars.some((bar) => bar !== null) && (
        <div aria-hidden="true" className="ui-trend accel-stat__trend">
          {bars.map((bar, index) => (
            <i
              className={index === bars.length - 1 && bar !== null ? "is-latest" : bar === null ? "is-empty" : undefined}
              key={index}
              style={{ height: bar === null ? undefined : `${Math.max(6, Math.min(100, bar * 100))}%` }}
            />
          ))}
        </div>
      )}
      <small>{stat.value === null ? stat.reason : old ? `Last read ${age}` : stat.note ?? "source not named"}</small>
    </div>
  );
}

function AcceleratorCardView({ accel, age }: { accel: Accelerator; age: SampleAge }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <article aria-labelledby={`${id}-title`} className="ui-card accel-card" data-accelerator={accel.key}>
      <div className="accel-card__head">
        <span aria-hidden="true" className="ui-row__icon"><Icon name={accel.icon} size={ICON_SIZE.nav} /></span>
        <div className="accel-card__titles">
          <h3 id={`${id}-title`}>{accel.title}</h3>
          <p>{accel.sub}</p>
        </div>
        <StatusPill label={accel.pill.label} reading={accel.pill.reading} tone={accel.pill.tone} />
      </div>
      <div className="accel-card__stats">
        {accel.stats.map((stat) => <StatCell age={age} key={stat.label} stat={stat} />)}
      </div>
      {accel.memory && (
        <div className={age === null ? "accel-card__memory" : "accel-card__memory accel-card__memory--stale"}>
          <MeterBar fraction={accel.memory.fraction} label={<><strong>{accel.memory.label}</strong><span>{accel.memory.value}</span></>} />
          <small>{accel.memory.note}</small>
        </div>
      )}
      <div className="accel-card__running">
        <h4>Running on it</h4>
        {accel.running.length
          ? accel.running.map((row) => (
            <div className="accel-card__model" key={row.name}>
              <div><strong>{row.name}</strong><small>{row.detail}</small></div>
              <StatusPill label={row.pill.label} reading={row.pill.reading} tone={row.pill.tone} />
            </div>
          ))
          : accel.runningUnread
            // A list that could not be read is not an empty one (LESSONS 8).
            ? <p className="accel-card__empty"><UnavailableValue label="Running models unavailable" mark="Not read" reason={accel.runningUnread} /> {accel.runningUnread}</p>
            : <p className="accel-card__empty">Nothing is running on it.</p>}
      </div>
      <h4 className="accel-card__toggle-heading">
        <Button aria-controls={`${id}-details`} aria-expanded={open} className="accel-card__toggle" onClick={() => setOpen((value) => !value)} variant="quiet">
          <span>{accel.detailsLabel}</span>
          <Icon name="chevron" size={ICON_SIZE.inline} />
        </Button>
      </h4>
      <div className="accel-card__details" hidden={!open} id={`${id}-details`}>
        <FactGrid facts={accel.facts} />
        {accel.extra}
      </div>
    </article>
  );
}

/**
 * System › Compute › Accelerators (VD-200, the SystemCompute board): a card
 * for each device that runs AI models, its four live readings, its shared
 * memory, and what is running on it. Every value is a reading or "Not read"
 * with its reason; an old sample is grey with its age; a trend is drawn only
 * from samples this page recorded; a pill is green only when the model server
 * answered within one re-read (LESSONS 5, 8, 19).
 */
export function Accelerators({
  cluster = null,
  history = [],
  machine,
  metrics: shellMetrics,
  role,
  sampledAt: shellSampledAt,
  stale: shellStale,
}: {
  cluster?: FleetSummary | null;
  history?: TelemetrySample[];
  machine: MachineProfile;
  /** The shell's live sample; without one the section reads its own. */
  metrics?: Metrics;
  /** Who is signed in: only an administrator's page asks for the LLM Server. */
  role: Role;
  /** The shell sample's own time (seconds or milliseconds), for its age. */
  sampledAt?: number;
  /** The shell's verdict that its sample is too old to show as live. */
  stale?: boolean;
}) {
  const own = useOwnSample(shellMetrics === undefined);
  const metrics = shellMetrics ?? own.metrics;
  const sampledAt = shellMetrics === undefined ? own.sampledAt : shellSampledAt ?? 0;
  const now = Date.now();
  // The shell's one freshness rule, applied the same way to a sample read here.
  const stale = shellMetrics === undefined
    ? sampledAt > 0 && now - sampleTimeMs(sampledAt) > STALE_AFTER_MS
    : shellStale === true;
  const age: SampleAge = stale && sampledAt > 0 ? timeAgo(sampleTimeMs(sampledAt), now) : null;
  const { engines, neural } = useEngineReads();
  const llmServer = useLlmServer(role);
  const hasGpu = machine.hardware.accelerators.length > 0 || machine.capabilities.gpu.available;
  const hasNpu = machine.hardware.neuralAccelerators.length > 0 || machine.capabilities.npu.available;
  const cards = [
    ...(hasGpu ? [gpuAccelerator(machine, metrics, history, engines, cluster, llmServer, age)] : []),
    ...(hasNpu ? [npuAccelerator(machine, metrics, history, engines, neural, now)] : []),
  ];
  return (
    <section aria-labelledby="accelerators-heading" className="accelerators">
      <h2 id="accelerators-heading">Accelerators <span>· what runs your AI models</span></h2>
      {cards.length > 0 && (
        <div className="accelerators__grid">
          {cards.map((accel) => <AcceleratorCardView accel={accel} age={age} key={accel.key} />)}
        </div>
      )}
      {/* Never an empty grid: a device this machine lacks is said once, plainly. */}
      {!hasGpu && !hasNpu && <p className="accelerators__absent">This machine has no graphics or neural processor Vaelor can use.</p>}
      {hasGpu && !hasNpu && <p className="accelerators__absent">No neural processor on this machine.</p>}
      {!hasGpu && hasNpu && <p className="accelerators__absent">No graphics processor on this machine.</p>}
    </section>
  );
}
