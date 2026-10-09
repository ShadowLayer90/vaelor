import { useEffect, useRef, useState } from "react";
import { apiRequest } from "./api";
import { statusTones, type StatusTone } from "../components/ui/status";

/**
 * The Performance dashboard's two layers (VD-147 S6, spec 3 and 4.4), as
 * `vaelor/performance_dashboard.py` (now) and `performance_dashboard_panels.py`
 * (range) send them, and the two hooks that poll them.
 *
 * THE BROWSER HOLDS NO WORD. Every word shown (titles, captions, labels,
 * badges, reasons, the range list, unit suffixes) arrives in the payload and
 * every colour as a `tone`, so state fields are `string` and nothing compares
 * one to a literal. The browser only fills a local clock time into a backend
 * `{time}` / `{since}` template, and formats numbers from their base-unit
 * identifiers (`lib/dashboardSeries.ts`).
 */

/** A backend tone, or neutral when it sent none (an older payload) or one this screen has no colour for. */
export function toneOf(tone: unknown): StatusTone {
  return (statusTones as readonly unknown[]).includes(tone) ? (tone as StatusTone) : statusTones[0];
}

/** Put a local 24-hour clock time ("14:02") where a backend template says `{time}` (or `{since}`). */
export function fillTime(template: string, epochSeconds: number | null | undefined, slot = "{time}"): string {
  if (typeof epochSeconds !== "number" || !Number.isFinite(epochSeconds)) return template.replace(slot, "").trim();
  const date = new Date(epochSeconds * 1000);
  const two = (value: number) => String(value).padStart(2, "0");
  return template.replace(slot, two(date.getHours()) + ":" + two(date.getMinutes()));
}

/** `freshness_projection()`'s block: the word, its label, its tone, and what dates it. */
export interface Freshness {
  state: string;
  label: string;
  tone?: string;
  since?: number | null;
  since_template?: string; // "Not reporting since {time}"
  reason?: string;
}

export interface DashboardNode {
  key: string; // internal: never rendered
  name: string;
  role: string;
  slot: number | null;
  freshness: Freshness;
  clock_note?: string;
  gpu_status_reason?: string;
}

export interface EngineBlock {
  kind: string | null;
  mode?: string | null;
  state?: string;
  cause?: string;
  placement: string | null;
  routing: string;
  routing_note?: string;
  badge?: string;
  badge_tone?: string;
  reason?: string;
  coverage_note?: string;
  scope_note: string;
}

export interface TemperatureMark { value?: number | null; label?: string } // "warning 97 °C"
export interface TemperatureBands {
  warning_c: number | null;
  critical_c: number | null;
  source: string;
  labels?: { warning?: string; critical?: string };
  marks?: TemperatureMark[];
}

export interface TileNode {
  key?: string | null;
  name?: string | null;
  state?: string;
  state_label?: string;
  value: number | null;
  sensor?: string | null;
  bands?: TemperatureBands | null;
  note?: string;
}

export interface DashboardTile {
  title?: string;
  state: string;
  state_label?: string;
  tone?: string;
  unit: string;
  cluster: { value: number | null; name?: string; sensor?: string | null; bands?: TemperatureBands | null; note?: string; partial?: boolean; covered_seconds?: number } | null;
  nodes: TileNode[];
  reason: string;
  /** The whole caption line; may hold `{since}`, filled from `scope_since`. */
  caption?: string;
  scope?: string;
  scope_since?: number | null;
  note?: string;
  excluded?: (string | { name?: string | null; reason: string })[];
}

export interface NowLayer {
  generated_at: number | null;
  refresh_seconds: number;
  engine: EngineBlock;
  nodes: DashboardNode[];
  tiles: Record<string, DashboardTile>;
  panels_freshness: Record<string, Freshness>;
}

/** Per bucket: the typical and slowest-10 % request speed bands, (slowest, fastest) in tokens per second. */
export type WireBand = { p50: [number | null, number | null] | null; p90: [number | null, number | null] | null } | { changed: string } | null;

export interface PanelSeries {
  key: string;
  name?: string;
  kind: string;
  slot?: number | null;
  values: (number | null)[];
  coverage?: (number | null)[];
  partial?: boolean[];
  state?: string;
  state_label?: string;
  tone?: string;
  reason?: string;
  bands?: WireBand[];
  bucket_reasons?: (string | null)[];
  sensor?: string | null;
  metric?: string; // host CPU and memory: which of a machine's two lines
  aggregation?: string;
  off_label?: string; // a line drawn off until chosen: "Cluster total (off - select to show)"
}

export interface DashboardPanel {
  title?: string;
  /** What the lines measure ("Per stream"), shown under the title; empty means show the range. */
  scope?: string;
  state: string;
  state_label?: string;
  tone?: string;
  unit: string;
  reason: string;
  series: PanelSeries[];
  caption?: string;
  bands?: WireBand[] | null;
  bands_note?: string;
  thresholds?: TemperatureBands | TemperatureMark[] | null;
  bands_by_node?: (TemperatureBands | null)[] | null;
  hidden_by_default?: string[];
  /** The prompt-tokens panel: each layer again, one line per machine ({cached: [...], computed: [...]}). */
  by_node?: Record<string, PanelSeries[]>;
}

export interface DashboardEvent { t: number; kind: string; node?: string; label: string }
export interface RangeOption { value: string; label: string }

export interface RangeLayer {
  range: string;
  range_seconds: number;
  start: number;
  step: number;
  count: number;
  generated_at: number | null;
  /** "Last 1 hour". */
  caption_lead?: string;
  labels?: { range_options?: RangeOption[]; units?: Record<string, string> };
  current_model: { model: string; engine: string; since: number } | null;
  engine: Pick<EngineBlock, "kind" | "placement" | "routing" | "scope_note">;
  events: DashboardEvent[];
  tiles: Record<string, DashboardTile>;
  panels: Record<string, DashboardPanel>;
  host_source?: string;
  host_note?: string;
}

// --------------------------------------------------------------------------
// The range this browser last chose (spec 4.4, owner decision D4)
// --------------------------------------------------------------------------

export const RANGE_STORAGE_KEY = "vaelor.performance.range";

/**
 * The range this browser last chose, or null: then the dashboard asks with no
 * range and the backend answers with its default. A stored range the backend
 * does not offer is dropped once its list of options is known.
 */
export function readStoredRange(): string | null {
  try {
    const stored = globalThis.localStorage?.getItem(RANGE_STORAGE_KEY);
    return stored ? stored : null;
  } catch {
    return null;
  }
}

/** Remember a choice (or forget it). A browser that refuses storage simply does not remember. */
export function storeRange(range: string | null): void {
  try {
    if (range) globalThis.localStorage?.setItem(RANGE_STORAGE_KEY, range);
    else globalThis.localStorage?.removeItem(RANGE_STORAGE_KEY);
  } catch {
    // Nothing to do: the choice lasts for this page only.
  }
}

const NOW_FALLBACK_SECONDS = 10; // until the payload names `refresh_seconds`
const RANGE_FLOOR_SECONDS = 20; // the range layer's fastest period, whatever its step (spec 4.4)

export interface Polled<T> {
  data: T | null;
  error: string;
  /** The backend refused the request (a 400: a range it does not offer), not a failed transport. */
  refused: boolean;
  loading: boolean;
  receivedAt: number | null; // when the page last received an answer (ms), for "Updated N s ago"
}

/**
 * Read `path` now, then every `periodMs` while `auto` is on; paused while hidden, read on return, aborted on
 * unmount. A new `path` or `reloadKey` aborts the read in flight; a late answer to an older read is dropped.
 */
function usePolled<T>(path: string, periodMs: number, auto: boolean, reloadKey: number): Polled<T> {
  const [state, setState] = useState<Polled<T> & { path: string }>({ data: null, error: "", refused: false, loading: true, receivedAt: null, path });
  const sequence = useRef(0);
  const readRef = useRef<() => void>(() => undefined);

  useEffect(() => {
    let controller: AbortController | null = null;
    let inFlight = false;
    const read = () => {
      if (inFlight) return;
      const mine = ++sequence.current;
      const current = new AbortController();
      controller = current;
      inFlight = true;
      setState((previous) => ({ ...previous, loading: true }));
      apiRequest<T>(path, { signal: current.signal, cache: "no-store" })
        .then((data) => {
          if (mine !== sequence.current) return;
          setState({ data, error: "", refused: false, loading: false, receivedAt: Date.now(), path });
        })
        .catch((reason: unknown) => {
          if (current.signal.aborted || mine !== sequence.current) return;
          const error = reason instanceof Error ? reason.message : "The dashboard could not be read.";
          const refused = (reason as { status?: unknown } | null)?.status === 400;
          // The answer is for this path now: never "Reading…" forever. A new path keeps no old figures.
          setState((previous) => ({ ...previous, data: previous.path === path ? previous.data : null, error, refused, loading: false, path }));
        })
        .finally(() => {
          if (mine === sequence.current) inFlight = false;
        });
    };
    readRef.current = read;
    read();
    const visibilityChanged = () => { if (!document.hidden) read(); };
    document.addEventListener("visibilitychange", visibilityChanged);
    return () => {
      sequence.current += 1;
      controller?.abort();
      readRef.current = () => undefined;
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, [path, reloadKey]);

  // The period can change (the payload names it) without a read of its own.
  useEffect(() => {
    if (!auto) return undefined;
    const timer = globalThis.setInterval(() => { if (!document.hidden) readRef.current(); }, periodMs);
    return () => globalThis.clearInterval(timer);
  }, [periodMs, auto, path, reloadKey]);

  // Only an answer for the path on screen is shown.
  return state.path === path
    ? { data: state.data, error: state.error, refused: state.refused, loading: state.loading, receivedAt: state.receivedAt }
    : { data: null, error: "", refused: false, loading: true, receivedAt: null };
}

/** The now layer, every `refresh_seconds` (10 s). */
export function useDashboardNow(auto: boolean, reloadKey: number): Polled<NowLayer> {
  const [period, setPeriod] = useState(NOW_FALLBACK_SECONDS);
  const polled = usePolled<NowLayer>("/cluster/performance/dashboard/now", period * 1000, auto, reloadKey);
  const named = polled.data?.refresh_seconds;
  useEffect(() => {
    if (typeof named === "number" && named > 0 && named !== period) setPeriod(named);
  }, [named, period]);
  return polled;
}

/** The range layer, every `max(20 s, step)`; with no range, the backend's default. */
export function useDashboardRange(range: string | null, auto: boolean, reloadKey: number): Polled<RangeLayer> {
  const [step, setStep] = useState(RANGE_FLOOR_SECONDS);
  const polled = usePolled<RangeLayer>(
    "/cluster/performance/dashboard" + (range ? `?range=${encodeURIComponent(range)}` : ""),
    Math.max(RANGE_FLOOR_SECONDS, step) * 1000, auto, reloadKey,
  );
  const named = polled.data?.step;
  useEffect(() => {
    if (typeof named === "number" && named > 0 && named !== step) setStep(named);
  }, [named, step]);
  return polled;
}
