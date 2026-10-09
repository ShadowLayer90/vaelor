import type { ReactNode } from "react";
import { StatusPill } from "./StatusPill";
import { UnavailableValue, type StatusTone } from "./ui";
import { phoenixReach, type PhoenixReach } from "../lib/clusterPerformance";
import { exactTime, formatBytes, formatPercent, formatTemperature, timeAgo } from "../lib/format";
import { healthReassurance } from "../lib/health";
import type { Health } from "../types";
import type {
  NodeUse,
  PerformanceSnapshot,
  ServingMetrics,
  ServingModelRed,
  ServingSection,
  TracesSection,
  WhySignal,
} from "../lib/clusterPerformance";

/**
 * The Performance tab's diagnostics (VD-200, the ClusterPerformanceDiagnostics
 * and ClusterPerformanceAdvanced boards): the serving engine's live gauges,
 * tracing, where the ceiling is, health, and in Advanced the per-machine use
 * table, the percentiles, the request counts and the signals not collected
 * yet. Every card is the boards' card - a title over a quiet second line and
 * one pill - and every figure sits in the one key/value cell size. Every value
 * here is what the backend sent or the honest unavailable mark, never a
 * substitute.
 */

export const WHY_TONE: Record<WhySignal, string> = {
  within_budget: "good",
  latency: "warn",
  errors: "bad",
  no_traffic: "idle",
  engine_only: "idle",
  retry_later: "idle",
  not_measured: "idle",
};

/** This page's four tones as the shared status tones. */
export const PILL_TONE: Record<string, StatusTone> = { good: "success", warn: "warning", bad: "danger", idle: "neutral" };

/** A node's words when it is plain text, for a `title` that keeps what an ellipsis cuts (polish audit, cause 14). */
const plainText = (node: ReactNode): string | undefined =>
  typeof node === "string" || typeof node === "number" ? String(node) : undefined;

/** One diagnostics card: a title, a quiet second line, at most one pill or action, then its body. */
export function PerfCard({ title, subtitle, trailing, label, className, children }: {
  title: ReactNode; subtitle?: ReactNode; trailing?: ReactNode; label: string; className?: string; children: ReactNode;
}) {
  return (
    <section aria-label={label} className={["ui-card", "perf-card", className].filter(Boolean).join(" ")}>
      <header className="perf-card__head">
        <div className="perf-card__titles">
          <h3 title={plainText(title)}>{title}</h3>
          {subtitle && <p title={plainText(subtitle)}>{subtitle}</p>}
        </div>
        {trailing}
      </header>
      <div className="perf-card__body">{children}</div>
    </section>
  );
}

/** The one stat cell: a label of at most two lines over its value, every cell the same height. */
export function KvGrid({ cells, className }: { cells: { label: ReactNode; value: ReactNode; key?: string }[]; className?: string }) {
  return (
    <dl className={["perf-kv", className].filter(Boolean).join(" ")}>
      {cells.map((cell, index) => (
        <div key={cell.key ?? index}>
          <dt>{cell.label}</dt>
          <dd title={plainText(cell.value)}>{cell.value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** One serving gauge: its value, or the honest mark saying why there is none. */
function ServingValue({ value, render, label, reason = "The serving engine did not emit this gauge." }: { value: number | null | undefined; render: (value: number) => string; label: string; reason?: string }) {
  return value === null || value === undefined
    ? <UnavailableValue label={`${label} unavailable`} reason={reason} />
    : <>{render(value)}</>;
}

const tokensPerSecond = (value: number) => `${Math.round(value)} tok/s`;

/** Average end-to-end latency: milliseconds under a second, else seconds. */
function formatServingLatency(seconds: number): string {
  const milliseconds = seconds * 1000;
  return milliseconds < 1000 ? `${Math.round(milliseconds)} ms` : `${seconds.toFixed(1)} s`;
}

/**
 * The deep vLLM-cluster signals the sample carries, each only when it is
 * there: the row is omitted whole on the llama.cpp engine, never empty.
 */
function deepCells(metrics: Partial<ServingMetrics>): { label: string; value: string }[] {
  const goodputLabel = metrics.goodput_budget_seconds != null
    ? `Goodput · e2e < ${Math.round(metrics.goodput_budget_seconds)}s`
    : "Goodput";
  const cells: { label: string; value: string | null }[] = [
    { label: "KV cache", value: metrics.kv_cache_fraction != null ? `${Math.round(metrics.kv_cache_fraction * 100)}%` : null },
    { label: "TTFT", value: metrics.ttft_seconds != null ? `${Math.round(metrics.ttft_seconds * 1000)} ms` : null },
    { label: "TPOT", value: metrics.tpot_seconds != null ? `${Math.round(metrics.tpot_seconds * 1000)} ms` : null },
    { label: "Prefix cache", value: metrics.prefix_cache_hit_rate != null ? `${Math.round(metrics.prefix_cache_hit_rate * 100)}%` : null },
    { label: "Requests served", value: metrics.request_success_total != null ? metrics.request_success_total.toLocaleString() : null },
    { label: "Preemptions", value: metrics.preemptions_total != null ? Math.round(metrics.preemptions_total).toLocaleString() : null },
    { label: goodputLabel, value: metrics.goodput_ratio != null ? `${Math.round(metrics.goodput_ratio * 100)}%` : null },
    {
      label: "Prompt throughput",
      value: metrics.prompt_throughput_tokens_per_second != null ? tokensPerSecond(metrics.prompt_throughput_tokens_per_second) : null,
    },
  ];
  return cells.filter((cell): cell is { label: string; value: string } => cell.value !== null);
}

/**
 * The per-model RED (requests, errors, duration) a vLLM cluster's scrape
 * carries: one line per served model. A model the engine served no completed
 * request shows the honest unavailable mark for its latency, never a 0.
 */
function ServingModelBreakdown({ models }: { models: ServingModelRed[] }) {
  return (
    <ul aria-label="Per-model RED" className="perf-models">
      {models.map((row) => (
        <li
          key={row.model}
          title={`Per-model RED · ${row.model} · Requests ${row.requests_total.toLocaleString()} · Errors ${row.errors_total.toLocaleString()} (${formatPercent(row.error_rate * 100)}) · e2e ${row.avg_e2e_seconds === null ? "unavailable" : formatServingLatency(row.avg_e2e_seconds)}`}
        >
          <span className="perf-muted">Per-model RED {"·"}</span>{" "}
          <span className="perf-mono" title={row.model}>{row.model}</span>{" "}
          <span className="perf-muted">
            {"·"} Requests {row.requests_total.toLocaleString()} {"·"} Errors {row.errors_total.toLocaleString()} ({formatPercent(row.error_rate * 100)}) {"·"} e2e{" "}
            {row.avg_e2e_seconds === null
              ? <UnavailableValue label="Average latency unavailable" reason="The engine served this model no completed request in the window." />
              : formatServingLatency(row.avg_e2e_seconds)}
          </span>
        </li>
      ))}
    </ul>
  );
}

/**
 * The GPU serving engine's live gauges: per-stream decode speed, prefill
 * (llama.cpp) or output throughput (vLLM cluster), and the running-vs-waiting
 * queue. Only a fresh sample is shown as live; the backend owns every "why
 * not" sentence and this card renders it verbatim. Pill: "Serving" for a live
 * sample, "Serving - no live metrics" when a GPU cluster deployment is serving
 * without one, "No live reading" when the last sample is too old, "Unknown"
 * when the fleet's record could not be read, and "Not serving" otherwise.
 */
export function ServingOverview({ serving, clusterServing, className }: { serving: ServingSection; clusterServing: boolean; className?: string }) {
  const metrics = serving.metrics;
  const models = serving.models ?? [];
  // The backend's own reading of the cluster record wins: it chose `reason`
  // from it, so the pill and the sentence cannot disagree. The page's pooled
  // list is the fallback for an older payload or an unreadable record.
  const clusterState = serving.cluster_state ?? (clusterServing ? "serving" : null);
  const servingWithoutMetrics = !serving.collected && clusterState === "serving";
  const badgeTone = serving.collected || servingWithoutMetrics ? "good" : "idle";
  const badgeLabel = serving.collected
    ? "Serving"
    : clusterState === "unloaded"
      ? "Unloaded"
      : clusterState === "loading"
        ? "Loading"
        : servingWithoutMetrics
          ? "Serving - no live metrics"
          : serving.stale
            ? "No live reading"
            : clusterState === "unknown"
              ? "Unknown"
              : "Not serving";
  // A vLLM cluster sample carries counted throughput; llama.cpp carries its own
  // prefill speed instead, under the label that matches what it measured.
  const clusterSample = metrics.generation_throughput_tokens_per_second != null || metrics.replicas_total != null;
  const deep = deepCells(metrics);
  return (
    <PerfCard
      className={className}
      label="GPU serving engine"
      subtitle="GPU serving engine"
      title="Live throughput and queue"
      trailing={<StatusPill className={`perf-pill--${badgeTone}`} label={badgeLabel} tone={PILL_TONE[badgeTone]} />}
    >
      {serving.scope_note ? <p className="perf-note" role="status">{serving.scope_note}</p> : null}
      {serving.collected ? (
        <>
          <KvGrid
            cells={[
              {
                label: "Decode speed",
                value: (
                  <ServingValue
                    label="Decode speed"
                    reason={clusterSample
                      ? "Nothing was generated since the previous reading, so there is no decode speed to measure."
                      : "The serving engine did not emit this gauge."}
                    render={tokensPerSecond}
                    value={metrics.decode_tokens_per_second}
                  />
                ),
              },
              clusterSample
                ? { label: "Output throughput", value: <ServingValue label="Output throughput" reason="No earlier reading to count tokens against yet." render={tokensPerSecond} value={metrics.generation_throughput_tokens_per_second} /> }
                : { label: "Prefill", value: <ServingValue label="Prefill rate" render={tokensPerSecond} value={metrics.prompt_tokens_per_second} /> },
              { label: "Running", value: <ServingValue label="Running requests" render={(value) => `${Math.round(value)}`} value={metrics.requests_processing} /> },
              { label: "Waiting", value: <ServingValue label="Waiting requests" render={(value) => `${Math.round(value)}`} value={metrics.requests_deferred} /> },
            ]}
          />
          {serving.coverage_note ? <p className="perf-note" role="status">{serving.coverage_note}</p> : null}
          {deep.length > 0 && (
            <>
              <p className="perf-section-label">vLLM cluster signals</p>
              <KvGrid cells={deep.map((cell) => ({ ...cell, key: cell.label }))} />
            </>
          )}
          {models.length > 0 && <ServingModelBreakdown models={models} />}
        </>
      ) : (
        <p className="perf-empty">{serving.reason}</p>
      )}
    </PerfCard>
  );
}

/**
 * Per-request tracing (Phoenix). When the collector is running it says where
 * Phoenix answers and how to reach it - it binds the controller's loopback,
 * so never a LAN link that would be dead. When it is not running it says why.
 */
export function TracesOverview({ traces, className }: { traces: TracesSection; className?: string }) {
  return (
    <PerfCard
      className={className}
      label="Request tracing"
      subtitle="Request tracing"
      title="Per-request traces (Phoenix)"
      trailing={<StatusPill label={traces.running ? "Capturing" : "Off"} tone={traces.running ? "success" : "neutral"} />}
    >
      <p className="perf-lead">{traces.reason}</p>
      {/* W5-D3 / W6-6 (LESSONS 19, 10): the address is the backend's (ui_host),
          and the command fits how this console was opened (phoenixReach). */}
      {traces.running && <PhoenixReachNote reach={phoenixReach(traces, window.location.hostname)} />}
    </PerfCard>
  );
}

function PhoenixReachNote({ reach }: { reach: PhoenixReach }) {
  if (reach.kind === "unknown") {
    return (
      <p className="perf-muted">
        Phoenix answers only on the controller itself, not on your network. The controller did not report the
        address it answers on, so no tunnel command is given here.
      </p>
    );
  }
  if (reach.kind === "local") {
    return (
      <div className="perf-muted">
        <p>
          Phoenix answers only on the controller itself ({reach.where}), not on your network. This console is open
          on localhost, so you are on the controller or already tunnelled to it: on the controller, open {reach.url};
          through a tunnel, add this forward to the one you use, then open {reach.url}.
        </p>
        <code className="perf-code">{reach.forward}</code>
      </div>
    );
  }
  return (
    <div className="perf-muted">
      <p>
        Phoenix answers only on the controller itself ({reach.where}), not on your network. To view the traces from
        this computer, forward its port over SSH and open {reach.url}:
      </p>
      <code className="perf-code">{reach.command}</code>
    </div>
  );
}

/** Which machine is the ceiling, and an honest count of any not reporting. */
export function FleetUseSummary({ nodes, className }: { nodes: NodeUse[]; className?: string }) {
  const reporting = nodes.filter((node) => node.reporting);
  const silent = nodes.filter((node) => !node.reporting);
  const ceiling = reporting
    .filter((node) => node.saturation.value !== null)
    .sort((a, b) => (b.saturation.value ?? 0) - (a.saturation.value ?? 0))[0];
  return (
    <PerfCard className={className} label="Fleet utilisation" subtitle="Fleet utilisation" title="Where the ceiling is">
      {ceiling ? (
        <p className="perf-lead">{ceiling.name}: {ceiling.saturation.verdict}</p>
      ) : (
        <p className="perf-empty">No node is reporting a utilisation signal in this window.</p>
      )}
      {silent.length > 0 && (
        <p className="perf-muted">
          {silent.length} of {nodes.length} {nodes.length === 1 ? "machine is" : "machines are"} not reporting telemetry.
        </p>
      )}
    </PerfCard>
  );
}

/** Overall health, read through the same reassurance wording Home uses. */
export function HealthOverview({ health, className }: { health: PerformanceSnapshot["health"]; className?: string }) {
  const asHealth: Health = {
    status: (health.status as Health["status"]) ?? "offline",
    reasons: health.reasons,
    checked: health.checked,
    sampled_at: 0,
  };
  return (
    <PerfCard className={className} label="Health" subtitle="Health" title={<span className="perf-card__status">{health.status}</span>}>
      <p className="perf-lead">{healthReassurance(asHealth)}</p>
    </PerfCard>
  );
}

/**
 * The GPU temperature a machine's row shows: the backend's chosen reading and
 * its sensor. An older payload with no such block has only the edge sensor's
 * field, shown as what it is.
 */
function nodeGpuTemperature(node: NodeUse): { value: number | null; sensor: string | null } {
  if (node.gpu_temperature) return { value: node.gpu_temperature.value_c, sensor: node.gpu_temperature.sensor };
  const edge = node.use.gpu_temperature_c ?? null;
  return { value: edge, sensor: edge === null ? null : "edge" };
}

const NOT_IN_WINDOW = "This machine did not report this signal in the window.";
const NOT_REPORTING = "This machine is not reporting telemetry.";

/** A single USE signal: its value, or the board's dash with the reason on it - never a 0. */
function UseValue({ value, render, label, reason }: { value: number | null | undefined; render: (value: number) => string; label: string; reason: string }) {
  return value === null || value === undefined
    ? <UnavailableValue label={`${label} unavailable`} mark={"—"} reason={reason} />
    : <>{render(value)}</>;
}

/** Advanced: one row per machine - its use, its temperatures and the verdict on where its headroom is. */
export function UtilisationTable({ nodes, className }: { nodes: NodeUse[]; className?: string }) {
  const columns = ["Machine", "Reporting", "CPU", "GPU", "Memory", "NPU", "GPU memory", "CPU temp", "GPU temp (sensor)", "Verdict"];
  return (
    <PerfCard className={className} label="Per-node utilisation" subtitle="Per-node use" title="Utilisation and saturation">
      <div className="perf-table-scroll">
        <table className="perf-table">
          <caption className="sr-only">Utilisation and saturation</caption>
          <thead><tr>{columns.map((column) => <th key={column} scope="col">{column}</th>)}</tr></thead>
          <tbody>
            {nodes.map((node) => {
              const reason = node.reporting ? NOT_IN_WINDOW : NOT_REPORTING;
              const temperature = nodeGpuTemperature(node);
              const sensor = temperature.sensor ?? "no sensor read";
              const lastSeen = node.last_sample_at
                ? `Not reporting telemetry; last seen ${node.last_sample_age_seconds != null ? `${timeAgo(0, node.last_sample_age_seconds * 1000)} ` : ""}(${exactTime(node.last_sample_at * 1000)}).`
                : NOT_REPORTING;
              // A machine that is not reporting shows none of its old figures as current.
              const use: Record<string, number | null> = node.reporting ? node.use : {};
              return (
                <tr key={node.id}>
                  <th scope="row">{node.name}</th>
                  <td><StatusPill label={node.reporting ? "Reporting" : "Not reporting"} tone={node.reporting ? "success" : "neutral"} /></td>
                  <td><UseValue label="CPU" reason={reason} render={formatPercent} value={use.cpu_percent} /></td>
                  <td><UseValue label="GPU" reason={reason} render={formatPercent} value={use.gpu_busy_percent} /></td>
                  <td><UseValue label="Memory" reason={reason} render={formatPercent} value={use.memory_percent} /></td>
                  <td><UseValue label="NPU" reason={reason} render={formatPercent} value={use.npu_activity_percent} /></td>
                  <td><UseValue label="GPU memory" reason={reason} render={(value) => formatBytes(value)} value={use.gpu_gtt_used_bytes} /></td>
                  <td><UseValue label="CPU temperature" reason={reason} render={(value) => formatTemperature(value)} value={use.cpu_temperature_c} /></td>
                  <td data-sensor={sensor}>
                    {/* The sensor is named on every reading: two machines can read two different sensors. */}
                    <UseValue
                      label={`GPU temp (${sensor})`}
                      reason={node.reporting && temperature.sensor === null ? "No graphics temperature sensor was read on this machine." : reason}
                      render={(value) => `${formatTemperature(value)} (${sensor})`}
                      value={node.reporting ? temperature.value : null}
                    />
                  </td>
                  <td className="perf-table__verdict">{node.reporting ? node.saturation.verdict : lastSeen}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </PerfCard>
  );
}

/** Advanced: the whole-request percentiles, shown for information and never judged. */
export function PercentilesCard({ health, className }: { health: PerformanceSnapshot["requests"]; className?: string }) {
  const latency = health.latency_ms;
  const ms = (value: number) => `${Math.round(value).toLocaleString()} ms`;
  return (
    <PerfCard className={className} label="Latency percentiles" subtitle="Latency detail" title="Whole-request percentiles, for information">
      {latency ? (
        <>
          {health.window_scope === "lifetime" && <p className="perf-note">{health.note}</p>}
          <KvGrid
            cells={[
              { label: "p50", value: ms(latency.p50) },
              { label: "p95", value: ms(latency.p95) },
              { label: "p99", value: ms(latency.p99) },
              { label: "max", value: ms(latency.max) },
            ]}
          />
          <p className="perf-muted">
            A whole request lasts as long as its answer, so these are not judged. Tokens:{" "}
            {typeof health.tokens_per_sec === "number"
              ? `${health.tokens_per_sec}/s over the window.`
              : <UnavailableValue label="Token rate unavailable" reason="No token count was recorded for the requests in this window." />}{" "}
            These percentiles pool every request from the inference gateway, the LLM Server and AI Chat.
          </p>
        </>
      ) : (
        <p className="perf-empty">No requests in this window, so there are no percentiles to show.</p>
      )}
    </PerfCard>
  );
}

/** Advanced: how many requests, failures and tokens the window held. */
export function RequestCountsCard({ health, className }: { health: PerformanceSnapshot["requests"]; className?: string }) {
  const rejected = typeof health.client_errors === "number" ? health.client_errors : null;
  return (
    <PerfCard className={className} label="Request counts" subtitle="Request counts" title="Requests, errors and tokens">
      <KvGrid
        cells={[
          { label: "Requests", value: health.requests.toLocaleString() },
          // Requests the model refused as the caller's error are counted apart from failures.
          rejected === null
            ? { label: "Failures", value: health.failures.toLocaleString() }
            : { label: "Failures + rejected", value: `${health.failures.toLocaleString()} + ${rejected.toLocaleString()}` },
          { label: "Prompt tokens", value: health.prompt_tokens.toLocaleString() },
          { label: "Completion tokens", value: health.completion_tokens.toLocaleString() },
        ]}
      />
      {health.tokens_note && <p className="perf-muted">{health.tokens_note}</p>}
    </PerfCard>
  );
}

/**
 * The deeper serving-signals state. While the backend still lists signals it
 * has not collected, the second line says so; once the list is empty it says
 * every signal is collected, so it never contradicts the sentence beneath it.
 */
export function UncollectedPanel({ uncollected, className }: { uncollected: PerformanceSnapshot["uncollected"]; className?: string }) {
  const hasSignals = uncollected.signals.length > 0;
  return (
    <PerfCard
      className={["perf-uncollected", className].filter(Boolean).join(" ")}
      label="Deeper serving signals"
      subtitle={hasSignals ? "Not collected yet" : "All signals collected"}
      title="Deeper serving signals"
    >
      {hasSignals && (
        <ul className="perf-uncollected__list">
          {uncollected.signals.map((signal) => (
            <li key={signal.id}>
              <strong>{signal.label}</strong> <span className="perf-muted">{"·"} {signal.reason}</span>
            </li>
          ))}
        </ul>
      )}
      <p className="perf-muted">{uncollected.next_step}</p>
    </PerfCard>
  );
}
