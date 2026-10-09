import type { ReactNode } from "react";
import { UnavailableValue } from "./ui";
import { exactTime } from "../lib/format";

/**
 * External API usage (Settings > External API access): the per-key line under
 * each inference-gateway API key, and the 24-hour figures of the gateway
 * status block. Kept in its own module so the large Administration surface
 * stays under the line ceiling.
 */

/**
 * Per-key cumulative gateway usage (Phase G, VD-128), joined onto each token by
 * id. Counts only: when a key was last used is the token row's own
 * `last_used_at`, one answer per key (ACC-098).
 */
export interface ApiTokenUsage {
  request_count: number;
  prompt_tokens: number;
  completion_tokens: number;
}

/**
 * The per-key inference usage line: real counts under the token, an honest
 * "none yet" for an inference key that has not been called, and "could not be
 * read" when the meter did not answer - never a zero for a count nobody took.
 * Only inference keys reach the metered gateway, so an assistant-only key
 * shows no usage line.
 */
export function ApiUsageLine({ token }: { token: { scopes: string[]; usage?: ApiTokenUsage | null } }) {
  if (!token.scopes.includes("inference")) return null;
  const usage = token.usage;
  if (usage === null) {
    return <small className="credential-row__usage">Cluster inference: usage could not be read</small>;
  }
  if (!usage || usage.request_count === 0) {
    return <small className="credential-row__usage">Cluster inference: no requests yet</small>;
  }
  return (
    <small className="credential-row__usage">
      {`Cluster inference: ${usage.request_count.toLocaleString()} requests, ${usage.prompt_tokens.toLocaleString()} prompt + ${usage.completion_tokens.toLocaleString()} completion tokens`}
    </small>
  );
}

/** What reached the model from outside Vaelor in the last 24 hours (ACC-046/048). */
export interface ExternalUsage {
  window_seconds: number;
  /** Epoch seconds: where the counted span really starts (it is kept in 10-minute steps). */
  since: number;
  requests: number;
  failures: number;
  average_latency_ms: number | null;
  /** Each door with the start of the span it really covers (epoch seconds). */
  gateway: { requests: number; failures: number; since?: number };
  /** Null when the LLM Server's usage is not known (not read, or not being logged). */
  llm_server: { requests: number; failures: number; refused?: number; since?: number } | null;
  llm_server_state: string;
  llm_server_detail: string;
  /**
   * The model's own token count over the window, less AI Chat's where they can
   * be told apart; `excludes_ai_chat` false when this machine's model served
   * in the window (AI Chat's tokens there cannot be separated).
   */
  model_tokens: {
    prompt_tokens: number; completion_tokens: number; counting_since: number | null;
    excludes_ai_chat?: boolean;
  } | null;
}

export interface InferenceGatewayStatus {
  healthy: boolean;
  /** `vaelor.inference_gateway.GATEWAY_STATE_SENTENCES` keys (ACC-097). */
  state?: string;
  message: string;
  target: { label: string; base_url: string } | null;
  metrics_24h: {
    requests: number; failures: number; streaming_requests: number;
    average_latency_ms: number; prompt_tokens: number; completion_tokens: number;
  };
  /** Absent from a backend older than the per-door figures. */
  external_24h?: ExternalUsage;
  endpoints: { models: string; chat_completions: string; openapi: string };
}

/** "since <time>" for one door when it differs from the card's own start. */
function doorSince(since: number | undefined, cardSince: number): string {
  return since && Math.abs(since - cardSince) > 600 ? ` since ${exactTime(since * 1000)}` : "";
}

/**
 * The sentence under the gateway's name that says what the four figures
 * count: which doors, since when, what was refused, and what the token figure
 * does and does not include.
 */
export function ExternalApiScope({ status }: { status: InferenceGatewayStatus | null }) {
  const external = status?.external_24h;
  if (!external) return null;
  const gateway = `${external.gateway.requests.toLocaleString()} through the gateway${doorSince(external.gateway.since, external.since)}`;
  const llm = external.llm_server
    ? `${external.llm_server.requests.toLocaleString()} through the LLM Server${doorSince(external.llm_server.since, external.since)}`
    : `the LLM Server's are not known here (${external.llm_server_detail || "its usage log was not read"})`;
  const refused = external.llm_server?.refused
    ? ` The LLM Server refused at least ${external.llm_server.refused.toLocaleString()} for a wrong or missing key.`
    : "";
  const tokens = external.model_tokens && external.model_tokens.excludes_ai_chat === false
    ? "Model tokens are the model's own count and include AI Chat on this machine's model."
    : "Model tokens are the model's own count, less AI Chat's.";
  return (
    <small>
      {`Since ${exactTime(external.since * 1000)}: ${gateway}, ${llm}. Only apps using an API key are counted.${refused} ${tokens}`}
    </small>
  );
}

type Figure = [string, (external: ExternalUsage) => ReactNode];

const FIGURES: Figure[] = [
  ["Requests · 24h", (external) => external.requests.toLocaleString()],
  ["Failures", (external) => external.failures.toLocaleString()],
  ["Average latency", (external) => external.average_latency_ms === null
    ? <UnavailableValue label="Average latency unavailable" reason="No external request reached the model in this span, so there is nothing to time." />
    : `${Math.round(external.average_latency_ms)} ms`],
  ["Model tokens · 24h", (external) => external.model_tokens
    ? (external.model_tokens.prompt_tokens + external.model_tokens.completion_tokens).toLocaleString()
    : <UnavailableValue label="Model tokens unavailable" reason="Vaelor has not read the serving model's own token counters yet." />],
];

/**
 * The four 24-hour figures. A figure that was not measured is shown as
 * unavailable with its reason, never as 0; before the status answered, every
 * figure is unavailable.
 */
export function ExternalApiFigures({ status }: { status: InferenceGatewayStatus | null }) {
  const external = status ? status.external_24h ?? fromGatewayOnly(status) : null;
  return (
    <dl>
      {FIGURES.map(([term, read]) => (
        <div key={term}>
          <dt>{term}</dt>
          <dd>{external
            ? read(external)
            : <UnavailableValue label={`${term} unavailable`} reason="Vaelor has not read the gateway yet" />}</dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * An older backend's gateway-only figures in the same shape (tokens are then
 * not the model's); counters that are not numbers were not read, so null.
 */
function fromGatewayOnly(status: InferenceGatewayStatus): ExternalUsage | null {
  const metrics = status.metrics_24h;
  if (typeof metrics?.requests !== "number" || typeof metrics.failures !== "number") return null;
  return {
    window_seconds: 86400, since: 0, requests: metrics.requests, failures: metrics.failures,
    average_latency_ms: metrics.requests ? metrics.average_latency_ms : null,
    gateway: { requests: metrics.requests, failures: metrics.failures },
    llm_server: null, llm_server_state: "", llm_server_detail: "",
    model_tokens: null,
  };
}
