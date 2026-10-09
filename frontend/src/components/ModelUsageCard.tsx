import { StatusPill } from "./StatusPill";
import { UnavailableValue } from "./ui";
import { KvGrid, PerfCard } from "./PerformanceDiagnostics";
import { exactTime, timeAgo } from "../lib/format";
import type { DeploymentUsage } from "../lib/clusterPerformance";

/**
 * Model usage (Performance; VD-200, the ClusterPerformanceAdvanced board): how
 * much each served model did, counted by the model itself (ACC-044/045).
 *
 * The serving engine's own counters see every request whichever way it came
 * in - the LLM Server, AI Chat, agents, the inference gateway. Rows are the
 * deployments the owner made, by name, and continue across a redeploy.
 * llama.cpp keeps no request counter, so its requests are shown as not
 * counted rather than as 0; a record Vaelor could not read says so rather
 * than showing an empty fleet.
 */

const ENGINE_LABEL: Record<string, string> = {
  vllm: "vLLM cluster",
  "llama.cpp": "llama.cpp on this machine",
};

function engineLine(row: DeploymentUsage): string {
  const engine = ENGINE_LABEL[row.engine] ?? row.engine;
  return row.model && row.model !== row.name ? `${row.model} · ${engine}` : engine;
}

function ModelUsageRow({ row }: { row: DeploymentUsage }) {
  const used = row.last_used_at ? `Last used ${timeAgo(row.last_used_at * 1000)}.` : "No request served yet.";
  // How fresh the COUNT is: whether the model is serving is the serving card's answer, not this one's.
  const freshness = row.live
    ? "Counting"
    : row.last_read_at ? `Last read ${timeAgo(row.last_read_at * 1000)}` : "Not read yet";
  return (
    <article aria-label={row.name} className="perf-usage">
      <header className="perf-usage__head">
        <span className="perf-usage__name">
          <strong>{row.name}</strong> <span className="perf-muted">{engineLine(row)}</span>
        </span>
        <StatusPill label={freshness} tone={row.live ? "info" : "neutral"} />
      </header>
      <KvGrid
        cells={[
          {
            label: "Completed requests",
            value: row.requests === null
              ? <UnavailableValue label="Requests not counted" reason="llama.cpp does not count requests; the tokens it processed are shown." />
              : row.requests.toLocaleString(),
          },
          { label: "Prompt tokens", value: row.prompt_tokens.toLocaleString() },
          { label: "Completion tokens", value: row.completion_tokens.toLocaleString() },
          { label: "Counted since", value: row.counting_since ? exactTime(row.counting_since * 1000) : "Not read" },
        ]}
      />
      <p className="perf-muted">{used}</p>
    </article>
  );
}

export function ModelUsageCard({ deployments, readable = true, className }: { deployments: DeploymentUsage[]; readable?: boolean; className?: string }) {
  return (
    <PerfCard className={className} label="Model usage" subtitle="Model usage" title="Requests and tokens each model served">
      {!readable ? (
        <p className="perf-empty">Vaelor could not read its record of model usage, so no totals are shown.</p>
      ) : deployments.length ? (
        deployments.map((row) => <ModelUsageRow key={row.identity} row={row} />)
      ) : (
        <p className="perf-empty">No counted model has served a request since Vaelor began counting.</p>
      )}
      <p className="perf-muted">
        Counted by each model itself, so every way in is included: the LLM Server, AI Chat, agents and the inference
        gateway. Totals since counting began, not the selected window. Not counted: a model shared with the Assistant
        on a single-model machine, clusters led by a worker, and the NPU Assistant.
      </p>
    </PerfCard>
  );
}
