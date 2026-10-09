import { StatusPill } from "./StatusPill";
import { UnavailableValue } from "./ui";
import { KvGrid, PerfCard } from "./PerformanceDiagnostics";
import { formatPercent } from "../lib/format";
import {
  doorDetailNote,
  formatDuration,
  type DoorRow,
  type DoorState,
  type GenerationHealth,
  type RequestHealth,
} from "../lib/clusterPerformance";

/**
 * Request health (Performance; VD-200, the ClusterPerformanceDiagnostics
 * board): how fast the model starts and writes its answers, and the rate and
 * errors for every way a request reaches it - the inference gateway, the LLM
 * Server and AI Chat - combined, with one row each.
 *
 * The verdict is the backend's (`vaelor/generation_health`, owner decision
 * 2026-09-29): time to first word and speed while answering, both measured by
 * the model itself. Whole-request times are shown as information and never
 * judged - a long answer takes long to finish however healthy the model is -
 * so no row turns red for them. A figure that was not measured shows why,
 * never a 0. The card spans the grid's width.
 */

/** The short word for a way in whose requests are not in the figures. */
const DOOR_STATE_WORD: Record<Exclude<DoorState, "measured">, string> = {
  "not-known": "Not known",
  off: "Off",
  "not-timed": "Not timed",
};

/** What each judged figure is called on the card and in the pill. */
const TTFT_LABEL = "Time to first word";
const DECODE_LABEL = "Speed while answering";

/** The verdict as one pill: the model's timings, never whole-request time. */
function BudgetBadge({ health }: { health: RequestHealth }) {
  if (health.traffic && health.window_scope === "lifetime") {
    return (
      <span className="perf-pill-group">
        <StatusPill label="No recent traffic" tone="neutral" />
        <StatusPill label="lifetime" tone="neutral" />
      </span>
    );
  }
  if (!health.traffic) return <StatusPill label="No traffic" tone="neutral" />;
  const generation = health.generation;
  if (!generation || generation.within_budget === null) return <StatusPill label="Not judged" tone="neutral" />;
  if (!generation.within_budget) {
    const over = [
      generation.ttft.within_budget === false ? TTFT_LABEL : null,
      generation.decode.within_budget === false ? DECODE_LABEL : null,
    ].filter(Boolean);
    return <StatusPill label={`Over budget: ${over.join(", ")}`} tone="danger" />;
  }
  return <StatusPill label="Within budget" tone="success" />;
}

/** Time to first word: the share that began within the budget, or why there is none. */
function TtftValue({ generation }: { generation: GenerationHealth }) {
  const ttft = generation.ttft;
  if (ttft.state !== "measured") return <span className="perf-muted">Not measured</span>;
  // Measured, and no answer started in the window: nothing to judge.
  if (!ttft.answers) return <span className="perf-muted">No answer started</span>;
  if (ttft.share_within_budget !== null) return <>{formatPercent(ttft.share_within_budget * 100)} within {formatDuration(ttft.budget_ms)}</>;
  const band = ttft.p95_band_ms;
  return <>95% within {band && band[1] !== null ? formatDuration(band[1]) : "the largest timer bucket"}</>;
}

/** Speed while answering: tokens a second on average while writing (all answers pooled). */
function DecodeValue({ generation }: { generation: GenerationHealth }) {
  const decode = generation.decode;
  if (decode.state !== "measured") return <span className="perf-muted">Not measured</span>;
  if (decode.tokens_per_second === null) return <span className="perf-muted">Nothing written</span>;
  return <>{decode.tokens_per_second} tokens/s</>;
}

/** The reasons a timing was not judged, each said once, in the backend's words. */
function generationNotes(generation: GenerationHealth | undefined): string[] {
  if (!generation) return [];
  const notes: string[] = [];
  for (const detail of [generation.ttft.detail, generation.decode.detail]) {
    if (detail && !notes.includes(detail)) notes.push(detail);
  }
  return notes;
}

/** The figures line for a way in that was read and has requests. */
function DoorFigures({ row, requests }: { row: DoorRow; requests: number }) {
  const latency = row.latency_ms;
  return (
    <span className="perf-door__figures">
      Requests {requests.toLocaleString()}
      {latency && <> {"·"} Whole request p95 {formatDuration(latency.p95)}</>}
      {" · "}Errors{" "}
      {typeof row.failures === "number"
        ? row.failures
        : <UnavailableValue label="Error count unavailable" reason="This way in did not report how many of its requests failed." />}
      {typeof row.error_rate === "number" && ` (${formatPercent(row.error_rate * 100)})`}
      {row.retry_later ? <> {"·"} Asked to retry {row.retry_later}</> : null}
    </span>
  );
}

/**
 * One way in: its own figures; "No requests in this span." when it was read
 * and saw none; or, when its requests are not in the figures, the state word
 * and the backend's reason - never a count. A row's own sentence is shown
 * whenever the backend sent one, with or without a state word (ACC-168).
 */
function DoorLine({ row }: { row: DoorRow }) {
  const stateWord = row.state === "measured" ? null : DOOR_STATE_WORD[row.state];
  const recorded = row.requests !== null && (row.requests > 0 || (row.retry_later ?? 0) > 0);
  let body;
  if (row.requests === null) body = <span className="perf-door__figures">{row.detail || "These requests could not be read."}</span>;
  else if (!recorded) body = <span className="perf-door__figures">No requests in this span.</span>;
  else body = <DoorFigures requests={row.requests} row={row} />;
  return (
    <li className="perf-door">
      <span className="perf-door__name">{row.label}</span>
      <span className="perf-door__text">
        {body}
        {row.requests !== null && row.detail && <span className="perf-door__detail">{row.detail}</span>}
      </span>
      {stateWord && <StatusPill className="perf-door__state" label={stateWord} tone="neutral" />}
    </li>
  );
}

export function RequestHealthCard({ health, className }: { health: RequestHealth; className?: string }) {
  const lifetime = health.window_scope === "lifetime";
  const partial = health.doors.filter((row) => row.coverage?.partial && row.traffic);
  const generation = health.generation;
  const notes = health.traffic && !lifetime ? generationNotes(generation) : [];
  const cells = health.traffic && health.latency_ms ? [
    ...(generation ? [
      { label: `${TTFT_LABEL} · budget ${Math.round(generation.ttft.budget_share * 100)}%`, value: <TtftValue generation={generation} /> },
      { label: `${DECODE_LABEL} · floor ${generation.decode.floor_tokens_per_second}`, value: <DecodeValue generation={generation} /> },
    ] : []),
    { label: "Request rate", value: `${health.requests_per_min}/min` },
    {
      label: "Error rate",
      value: (
        <>
          {typeof health.error_rate === "number"
            ? formatPercent(health.error_rate * 100)
            : <UnavailableValue label="Error rate unavailable" reason="No error rate was reported for these requests." />}{" "}
          ({health.failures.toLocaleString()}/{health.requests.toLocaleString()})
        </>
      ),
    },
  ] : [];
  return (
    <PerfCard
      className={className}
      label="Request health"
      subtitle={"Requests · by way in"}
      title={"Requests – speed and errors"}
      trailing={<BudgetBadge health={health} />}
    >
      {lifetime && <p className="perf-note">{health.note}</p>}
      {partial.map((row) => row.coverage && (
        <p className="perf-note" key={row.door}>{doorDetailNote(row.label, row.coverage, lifetime)}</p>
      ))}
      {health.traffic && health.latency_ms ? (
        <>
          <KvGrid cells={cells} />
          {notes.map((note) => <p className="perf-muted" key={note}>{note}</p>)}
          <p className="perf-muted">
            Whole request time, for information: {formatDuration(health.latency_ms.p50)} typical,{" "}
            {formatDuration(health.latency_ms.p95)} at p95. It is mostly how long each answer was, so it is not judged.
          </p>
        </>
      ) : (
        <p className="perf-empty">{health.note} There is no speed to report.</p>
      )}
      <ul aria-label="Requests by way in" className="perf-doors">
        {health.doors.map((row) => <DoorLine key={row.door} row={row} />)}
      </ul>
    </PerfCard>
  );
}
