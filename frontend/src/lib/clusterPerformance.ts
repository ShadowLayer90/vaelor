import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "./api";
import type { GpuTemperatureSensor } from "./gpuTemperature";
import { exactTime } from "./format";

/**
 * The §6b Performance snapshot, the typed shape `GET /api/v2/cluster/performance`
 * returns. It mirrors `vaelor/performance_snapshot.build_snapshot`, and the
 * honesty rules are encoded in the types: a latency block is `null` when there
 * was no traffic (never a fabricated `0 ms`), a node's USE value is `number |
 * null` (never a substituted zero for a sensor that did not report), and the
 * deep serving signals live only in `uncollected`, never as a value.
 */
export interface RequestLatency {
  p50: number;
  p95: number;
  p99: number;
  max: number;
}

/**
 * Whether a door's requests are in the figures (`vaelor/request_health.DOOR_STATES`,
 * the one owner of these words): read and counting, not known (its record
 * could not be read or is not being written), switched off, or not timed one
 * by one right now.
 */
export type DoorState = "measured" | "not-known" | "off" | "not-timed";

/**
 * Time to first word as `vaelor/generation_health` reports it: the share of
 * answers in the window that began within the budget (exact - the budget is a
 * bucket edge of the model's own timer) and the p95 as a bucket band
 * `[lower, upper]` in ms, never interpolated. `upper` is null when the p95 is
 * past the timer's largest bucket.
 */
export interface TtftHealth {
  state: DoorState;
  detail: string;
  answers: number | null;
  budget_ms: number;
  budget_share: number;
  p95_band_ms: [number, number | null] | null;
  share_within_budget: number | null;
  within_budget: boolean | null;
}

/** Speed while answering: tokens a second on average while writing, every answer pooled, against the floor. */
export interface DecodeHealth {
  state: DoorState;
  detail: string;
  tokens_per_second: number | null;
  floor_tokens_per_second: number;
  within_budget: boolean | null;
}

/**
 * The speed verdict's basis (owner decision 2026-09-29): the model's own time
 * to first word and writing speed over the window. `within_budget` is null
 * when nothing could be judged; `judged` names what was.
 */
export interface GenerationHealth {
  basis: "model-timings";
  ttft: TtftHealth;
  decode: DecodeHealth;
  within_budget: boolean | null;
  judged: Array<"ttft" | "decode">;
}

/** A duration in plain units: `250 ms` under a second, `2.5 s` above. */
export function formatDuration(milliseconds: number): string {
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  return `${(milliseconds / 1000).toFixed(1).replace(/\.0$/, "")} s`;
}

/**
 * One door's requests - the inference gateway, the LLM Server, AI Chat - as
 * `vaelor/request_health.request_health` reports it. Counts are `null` when the
 * door could not be read and recorded nothing in the span: not known, never 0.
 * Its percentiles are its own requests' only.
 */
export interface DoorRow {
  door: string;
  label: string;
  state: DoorState;
  detail: string;
  traffic: boolean;
  requests: number | null;
  failures: number | null;
  client_errors: number | null;
  error_rate: number | null;
  requests_per_min: number | null;
  /** Whole-request percentiles: information only, never judged. */
  latency_ms: RequestLatency | null;
  /** Always null now: no door is judged on its whole-request time. */
  within_budget: boolean | null;
  over_budget_by_ms: number | null;
  latency_role?: "information";
  /** LLM Server 503 answers (try again later), counted apart from served requests. */
  retry_later: number | null;
  coverage: DoorCoverage | null;
}

/**
 * Every door's requests combined (`requests` in the snapshot). The combined
 * percentiles are computed from all doors' measured per-request times pooled
 * (`latency_basis: "pooled-per-request"`), never averaged across doors.
 */
export interface RequestHealth {
  traffic: boolean;
  requests: number;
  // Server-side failures (5xx) only; a request the model refused as the
  // caller's error is counted in `client_errors` instead.
  failures: number;
  client_errors?: number;
  requests_per_min: number;
  error_rate: number | null;
  /** Whole-request percentiles, pooled: information only, never judged. */
  latency_ms: RequestLatency | null;
  latency_role?: "information";
  /** The time-to-first-word budget the verdict is judged against. */
  budget_ms: number;
  budget_metric: "ttft-p95";
  /** The model-timings verdict (`generation.within_budget`); null when not judged. */
  within_budget: boolean | null;
  over_budget_by_ms: number | null;
  /**
   * The verdict's basis: this window's model timings. Absent on a lifetime
   * panel (they would describe a different span) and from an older payload.
   */
  generation?: GenerationHealth;
  tokens_per_sec: number | null;
  prompt_tokens: number;
  completion_tokens: number;
  note: string;
  // "window" for the windowed reading, "lifetime" when the selected window had
  // no requests and the panel fell back to lifetime totals. Optional and
  // tolerant: an older payload without it reads as the windowed default.
  window_scope?: "window" | "lifetime";
  // On a lifetime-scoped panel: when the newest recorded request was made
  // (epoch seconds) and how long ago. A lifetime panel carries no budget
  // verdict, because its requests may be days old.
  last_request_at?: number | null;
  last_request_age_seconds?: number | null;
  doors: DoorRow[];
  latency_basis?: "pooled-per-request";
  /** Doors whose traffic is not known, so the combined figure leaves them out. */
  unmeasured?: string[];
  /** LLM Server 503 answers in the span, left out of latency and failures. */
  retry_later?: number;
  /** Why token figures leave a door out, when one that cannot see tokens had traffic. */
  tokens_note?: string;
}

export interface NodeSaturation {
  ceiling: "cpu" | "gpu" | "memory" | null;
  value: number | null;
  saturated: boolean;
  verdict: string;
}

export interface NodeGpuTemperature {
  value_c: number | null;
  sensor: GpuTemperatureSensor | null;
  bands: { warning_c: number | null; critical_c: number | null; source: "gpu" | "machine class" } | null;
  note: string;
}

export interface NodeUse {
  id: string;
  name: string;
  role: string;
  reporting: boolean;
  // Epoch seconds of the node's newest stored sample (the store reads it with
  // epoch='s'); null for the controller, whose reading is live.
  last_sample_at: number | null;
  last_sample_age_seconds: number | null;
  use: Record<string, number | null>;
  /**
   * The one GPU temperature shown for this node
   * (`vaelor.platforms.gpu_temperature.gpu_temperature_block`): the value, the
   * sensor it came from, this node's own warning and critical marks and where
   * they came from, and the sentence that goes with them. Absent on an older
   * payload.
   */
  gpu_temperature?: NodeGpuTemperature;
  saturation: NodeSaturation;
  health: { status: string; reasons: string[]; checked: string[] };
}

export type WhySignal =
  | "within_budget" | "latency" | "errors" | "no_traffic" | "engine_only" | "retry_later" | "not_measured";

export interface PerformanceWhy {
  signal: WhySignal;
  headline: string;
  detail: string;
}

/**
 * The live serving gauges the Phase E′ controller scrape collected from the GPU
 * llama.cpp engine's `/metrics`. Every value is `number | null` (never a
 * substituted zero for a gauge the engine did not emit), mirroring
 * `vaelor.performance_snapshot.serving_section`.
 */
export interface ServingMetrics {
  // Per-stream decode speed: how fast a request's tokens arrive while it is
  // generating (llama.cpp's own gauge, or vLLM's inter-token latency).
  decode_tokens_per_second: number | null;
  prompt_tokens_per_second: number | null;
  // vLLM cluster only: tokens counted over the last scrape interval and summed
  // across replicas - aggregate throughput, not the speed of one stream.
  generation_throughput_tokens_per_second?: number | null;
  prompt_throughput_tokens_per_second?: number | null;
  // vLLM cluster only: how many replicas the aggregate covers.
  replicas_read?: number | null;
  replicas_total?: number | null;
  requests_processing: number | null;
  requests_deferred: number | null;
  busy_slots_per_decode: number | null;
  // The deeper vLLM-cluster signals, keyed onto the sample by the backend only
  // when serving on a vLLM cluster whose /metrics scrape feeds them. Optional
  // because the single-node llama.cpp engine never emits them (absent, not 0).
  kv_cache_fraction?: number | null;
  ttft_seconds?: number | null;
  tpot_seconds?: number | null;
  request_success_total?: number | null;
  prefix_cache_hit_rate?: number | null;
  preemptions_total?: number | null;
  goodput_ratio?: number | null;
  goodput_budget_seconds?: number | null;
}

/**
 * One model's RED (requests, errors, duration) breakdown on a vLLM cluster,
 * mirroring `vaelor.serving_metrics` per-model rows. `avg_e2e_seconds` is `null`
 * when the engine served the model no completed request in the window (absent,
 * never a fabricated 0), and the array is empty when no model carries one.
 */
export interface ServingModelRed {
  model: string;
  requests_total: number;
  errors_total: number;
  error_rate: number;
  avg_e2e_seconds: number | null;
}

/** `vaelor.performance_serving.CLUSTER_*`; "unknown" is a record that could not be read. */
export type ClusterServingState = "serving" | "unloaded" | "loading" | "none" | "unknown";

export interface ServingSection {
  collected: boolean;
  // True when a sample exists but is too old (or undated) to be shown as live.
  stale?: boolean;
  // What the fleet's GPU cluster record says, as the backend read it when it
  // chose `reason` (the words are `vaelor.performance_serving.CLUSTER_*`);
  // "unknown" when the record could not be read (null on an older payload).
  // `cluster_cause` says why an unloaded record is unloaded ("idle"
  // scale-to-zero or "manual").
  cluster_state?: ClusterServingState | null;
  cluster_cause?: string;
  reason: string;
  sampled_at?: number | null;
  age_seconds?: number | null;
  metrics: Partial<ServingMetrics>;
  models?: ServingModelRed[];
  // The backend's sentence when the aggregate covers fewer replicas than exist.
  coverage_note?: string;
  // How the deployment is laid out, in the record's own word: "replicated"
  // (one full copy per machine), "distributed" (one model split across
  // machines), or "" (single machine, nothing served, not known).
  placement?: string;
  // The backend's sentence when one model is split across machines: the
  // figures are the whole model's and there is no per-machine breakdown.
  scope_note?: string;
}

/**
 * The Phoenix trace-collector status (VD-128), mirroring
 * `vaelor.performance_snapshot.traces_section`. `running` drives the affordance:
 * the "view request traces" link shows only when the collector is up, and the
 * honest-degrade `reason` shows otherwise — never a dead link.
 */
export interface TracesSection {
  enabled: boolean;
  running: boolean;
  collected: boolean;
  otlp_endpoint: string;
  ui_port: number;
  /** The address the Phoenix UI answers on, as the backend's launch publishes it; "" when not reported (W6-6). */
  ui_host?: string;
  reason: string;
}

/**
 * How the owner reaches Phoenix from the computer this console is open on
 * (W5-D3, W6-6). Phoenix answers only on the address the backend reports
 * (`ui_host`, the controller's loopback), so it is reached through an SSH
 * forward - unless this console is itself open on localhost, which means the
 * owner is on the controller or already tunnelled to it. No address is
 * invented: without `ui_host` there is no command.
 */
export type PhoenixReach =
  | { kind: "unknown" }
  | { kind: "local"; where: string; url: string; forward: string }
  | { kind: "tunnel"; where: string; url: string; command: string };

const bracketed = (host: string) => (host.includes(":") ? `[${host}]` : host);
const LOCAL_CONSOLE = /^(localhost|127(\.\d{1,3}){3}|::1)$/i;

export function phoenixReach(traces: Pick<TracesSection, "ui_host" | "ui_port">, consoleHost: string): PhoenixReach {
  const host = (traces.ui_host ?? "").trim();
  const port = traces.ui_port;
  if (!host || !(port > 0)) return { kind: "unknown" };
  // location.hostname brackets an IPv6 literal; ssh's destination takes it bare.
  const opened = consoleHost.trim().replace(/^\[(.*)\]$/, "$1");
  const where = `${bracketed(host)}:${port}`;
  const forward = `-L ${port}:${bracketed(host)}:${port}`;
  const url = `http://localhost:${port}`;
  if (LOCAL_CONSOLE.test(opened)) return { kind: "local", where, url, forward };
  return { kind: "tunnel", where, url, command: `ssh ${forward} <user>@${opened || "<controller>"}` };
}

export interface UncollectedSignal {
  id: string;
  label: string;
  reason: string;
}

export interface UncollectedBlock {
  collected: false;
  next_step: string;
  signals: UncollectedSignal[];
}

/**
 * Per-deployment cumulative inference usage (Phase G, VD-128). Keyed on the lease
 * credential id, these are lifetime totals - not windowed like the gateway RED
 * beside them - so the tab labels them cumulative.
 */
/**
 * One served model's usage, counted by the model's own counters
 * (`vaelor/model_usage.deployment_usage`). `requests` is null for an engine that
 * does not count them (llama.cpp), never a substituted 0; `live` is whether the
 * model was read in the last 30 seconds.
 */
export interface DeploymentUsage {
  identity: string;
  name: string;
  model: string;
  engine: string;
  prompt_tokens: number;
  completion_tokens: number;
  requests: number | null;
  counting_since: number | null;
  last_used_at: number | null;
  /** When the model's counters were last read (epoch seconds). */
  last_read_at?: number | null;
  live: boolean;
}

/**
 * What the gateway panel's per-request detail covers: `partial` once the
 * detail was pruned past the span the panel reports, with `since` the oldest
 * request still held (`vaelor/inference_metrics.detail_coverage`, ACC-047).
 */
export interface DoorCoverage {
  partial: boolean;
  since: number | null;
  kept: number;
}

/** The sentence a door whose detail is partial carries: what it covers, from when. */
export function doorDetailNote(label: string, detail: DoorCoverage, lifetime: boolean): string {
  const since = detail.since ? exactTime(detail.since * 1000) : "an unknown time";
  return `${label}: requests are kept one by one only since ${since} (the newest ${detail.kept.toLocaleString()} at most), so its figures cover that span, not ${lifetime ? "everything since metering began" : "the whole selected window"}.`;
}

export interface PerformanceSnapshot {
  window_seconds: number;
  baseline_window_seconds: number;
  generated_at: number;
  requested_window: string;
  requests: RequestHealth;
  baseline_requests: {
    traffic: boolean;
    requests: number;
    error_rate: number | null;
    p95_ms: number | null;
  };
  nodes: NodeUse[];
  serving: ServingSection;
  traces: TracesSection;
  deployments: DeploymentUsage[];
  /** False when the model-usage record could not be read (absent on an older backend). */
  deployments_readable?: boolean;
  health: { status: string; reasons: string[]; checked: string[] };
  why: PerformanceWhy;
  uncollected: UncollectedBlock;
}

/**
 * The on-demand serving-profile result (VD-128 §6b), the typed shape
 * `POST /api/v2/cluster/performance/profile` returns. It mirrors
 * `vaelor.serving_profiler.build_profile`, and the honesty rules are in the
 * types: a GPU field is optional (absent, never a substituted 0, when the
 * snapshot did not carry it), and a capture that could not run has NO `result`
 * and carries its plain-language `reason` instead — never a fabricated profile.
 */
export interface PerfSymbol {
  symbol: string;
  overhead_percent: number;
  shared_object: string;
}

export interface CpuProfileResult {
  symbols: PerfSymbol[];
  note?: string;
}

export interface GpuProcessProfile {
  pid: number;
  name?: string;
  /** GTT residency in BYTES (amd-smi reports `unit: "B"`); format with formatBytes. */
  gtt?: number;
  compute_percent?: number;
}

/**
 * What kind of power `power_watts` is (`vaelor.serving_profiler_gpu.POWER_*`):
 * the graphics engine's own draw, a discrete card's own draw, or a figure from
 * a part Vaelor could not identify. Package power is never `power_watts`.
 */
export type GpuPowerKind = "graphics-engine" | "gpu" | "unidentified";

export type { GpuTemperatureSensor } from "./gpuTemperature";

export interface GpuSnapshot {
  power_watts?: number;
  power_kind?: GpuPowerKind;
  /** The whole chip's power (processor and graphics together); never the GPU's own. */
  package_power_watts?: number;
  gpu_temperature_c?: number;
  gpu_temperature_sensor?: GpuTemperatureSensor;
  gfx_clock_mhz?: number;
  gfx_activity_percent?: number;
  process?: GpuProcessProfile;
}

export interface GpuProfileResult {
  snapshot: GpuSnapshot;
  note?: string;
}

/** One thread's OFF-CPU blocking time, in microseconds, keyed by its comm. */
export interface EbpfOffCpuRow {
  label: string;
  value_us: number;
}

/**
 * The on-demand eBPF (bpftrace) OFF-CPU trace result: where the serving threads
 * BLOCK - the signal perf's on-CPU sampling cannot see - attributed by thread
 * comm. `rows` is empty (with a truthful `note`) when the window observed no
 * blocking, so an idle window reads as "captured, nothing blocked", never a
 * fabricated row.
 */
export interface EbpfProfileResult {
  kind: "offcpu";
  window_seconds: number;
  total_us: number;
  rows: EbpfOffCpuRow[];
  note?: string;
}

/** One GPU kernel from the vLLM torch profiler: its long name, total device
 * time in microseconds, and how many times it ran in the window. */
export interface GpuKernelRow {
  name: string;
  total_us: number;
  calls: number;
}

/**
 * One plain-English kernel category: a share of GPU device time with a
 * one-line explanation, so the trace can lead with where the time goes in
 * human terms rather than raw ISA kernel names. `pct` is the category's share
 * of the total device time, rounded to one decimal.
 */
export interface GpuKernelCategory {
  id: string;
  label: string;
  description: string;
  total_us: number;
  pct: number;
}

/**
 * The on-demand GPU kernel trace from vLLM's built-in torch profiler: the
 * hottest device kernels by total time, and a plain-English `categories`
 * breakdown of where that device time went. `rows` is empty (with a truthful
 * `note`) when the profiling window observed no kernels, so an idle window
 * reads as "captured, no kernels", never a fabricated trace. `categories` is
 * optional so an older payload without it still reads.
 */
export interface GpuKernelResult {
  kind: "gpu_kernels";
  window_seconds: number;
  total_us: number;
  rows: GpuKernelRow[];
  categories?: GpuKernelCategory[];
  note?: string;
}

export interface ProfileCapture {
  id: string;
  label: string;
  available: boolean;
  reason: string;
  result?: CpuProfileResult | GpuProfileResult | EbpfProfileResult | GpuKernelResult;
}

export interface ServingProfile {
  seconds: number;
  serving_pid: number | null;
  as_root: boolean;
  perf_event_paranoid: number | null;
  generated_at: number;
  captures: ProfileCapture[];
}

/**
 * Trigger ONE bounded on-demand profile. The route is administrator + CSRF and
 * rate-limited, so the caller passes its CSRF token; a 429 (a profile already
 * running or one moments ago) or a 503 (no privileged bridge) surfaces as the
 * appliance's own message, never a fabricated result.
 */
export function runServingProfile(csrfToken: string): Promise<ServingProfile> {
  return apiRequest<ServingProfile>(
    "/cluster/performance/profile",
    { method: "POST", body: "{}", cache: "no-store" },
    csrfToken,
  );
}

/** The windows offered by the tab — a "now" view, an hour, and a day. */
export const PERFORMANCE_WINDOWS = [
  { value: "15m", label: "Last 15 min" },
  { value: "1h", label: "Last hour" },
  { value: "24h", label: "Last 24 h" },
] as const;

export interface PerformanceState {
  snapshot: PerformanceSnapshot | null;
  loading: boolean;
  error: string;
  reload: () => void;
  /** When the snapshot on screen arrived (ms), so a failed refresh can say how old it is. */
  receivedAt: number | null;
  /** When the newest refresh failed (ms); null once one succeeds. */
  failedAt: number | null;
}

/**
 * How often an open, visible Performance tab re-reads the snapshot. The
 * serving engine is scraped every 10 s, so a state change (a model unloaded,
 * a request served) reaches the screen within one poll without a Reload.
 */
export const PERFORMANCE_REFRESH_MS = 10000;

/**
 * Fetch the snapshot for a window and keep it current while the tab is shown.
 *
 * The first read shows loading; after it, the snapshot is re-read every
 * {@link PERFORMANCE_REFRESH_MS} while the page is visible (and once when it
 * becomes visible again), never with two reads in flight. Every read carries a
 * sequence number, so an older answer that lands after a newer one - a slow
 * read overtaken by a window change or a Reload - is dropped rather than
 * shown. A failed refresh keeps the last good snapshot on screen with the
 * time it arrived (`receivedAt`) and the failure (`failedAt`); it is never
 * blanked. Switching window or leaving the tab aborts the read in flight.
 */
export function useClusterPerformance(window: string | null): PerformanceState {
  const [snapshot, setSnapshot] = useState<PerformanceSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [receivedAt, setReceivedAt] = useState<number | null>(null);
  const [failedAt, setFailedAt] = useState<number | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const reload = useCallback(() => setReloadKey((value) => value + 1), []);
  const sequence = useRef(0);
  // Only a snapshot of the window on screen is shown: after a change of window
  // the old window's figures never sit under the new selection (nor does the
  // "still showing the reading from" notice refer to another window).
  // With no window chosen yet the backend's default answers, whatever it names.
  const current = snapshot && (!window || snapshot.requested_window === window) ? snapshot : null;

  useEffect(() => {
    // No window yet (the dashboard has not said which range it shows): nothing to read.
    if (window === null) return undefined;
    let controller: AbortController | null = null;
    let inFlight = false;
    const read = (first: boolean) => {
      if (!first && (inFlight || document.hidden)) return;
      const mine = ++sequence.current;
      const current = new AbortController();
      controller = current;
      inFlight = true;
      if (first) setLoading(true);
      apiRequest<PerformanceSnapshot>(
        window ? `/cluster/performance?window=${encodeURIComponent(window)}` : "/cluster/performance",
        { signal: current.signal, cache: "no-store" },
      )
        .then((result) => {
          if (mine !== sequence.current) return;
          setSnapshot(result);
          setReceivedAt(Date.now());
          setFailedAt(null);
          setError("");
        })
        .catch((reason: unknown) => {
          if (current.signal.aborted || mine !== sequence.current) return;
          setFailedAt(Date.now());
          setError(reason instanceof Error ? reason.message : "The performance snapshot is unavailable.");
        })
        .finally(() => {
          if (mine !== sequence.current) return;
          inFlight = false;
          setLoading(false);
        });
    };
    read(true);
    const timer = globalThis.setInterval(() => read(false), PERFORMANCE_REFRESH_MS);
    const visibilityChanged = () => {
      if (!document.hidden) read(false);
    };
    document.addEventListener("visibilitychange", visibilityChanged);
    return () => {
      // Anything still in flight belongs to a window or a reload that is gone.
      sequence.current += 1;
      controller?.abort();
      globalThis.clearInterval(timer);
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, [window, reloadKey]);

  return {
    snapshot: current, loading, error, reload,
    receivedAt: current ? receivedAt : null, failedAt: current ? failedAt : null,
  };
}
