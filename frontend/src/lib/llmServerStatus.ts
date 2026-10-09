import { statusTone, type StatusTone } from "../components/ui/status";
import type { EndpointKey, KeyUsageState, ServedEndpoint } from "./endpoints";
import { timeAgo } from "./format";

/**
 * Every word the backend's `runtime.state` may carry. The owner is
 * `vaelor/api_llm_server_routes.py` `RUNTIME_STATES`; this is the one
 * TypeScript copy, marked for `tests/test_wire_vocabularies.py` and pinned by
 * `tests/test_gpu_cluster_mode_llm_server.py`, so a state added on either side
 * alone fails the suite instead of reaching the badge as an unknown word.
 */
export const LLM_SERVER_RUNTIME_STATES = [ // vocabulary: llm-server-runtime-state
  "serving",
  "not-running",
  "model-not-answering",
  "applying-keys",
  "no-keys",
  "paused",
  "starting",
  "unknown",
  "off",
  "still-open",
] as const;

export type LlmServerRuntimeState = (typeof LLM_SERVER_RUNTIME_STATES)[number];

/**
 * The LLM Server's badge, from what its LAN door is DOING rather than what it
 * was told (LESSONS pattern 1). The backend's `runtime.state` is the reading:
 * the proxy container's status over the root bridge, a `/health` request
 * through it to the model, and the key set the running proxy carries. "Serving"
 * is shown for `serving` and nothing else - the console once showed it from the
 * enabled flag alone for eight days while the port refused every connection
 * (2026-09-28).
 *
 * One mapping, read by both surfaces that show the badge: the endpoints panel
 * under Cluster › Deployments › Models and the compact summary in AI Chat.
 */
export interface LlmServerBadge {
  label: string;
  tone: StatusTone;
}

export function llmServerBadge(endpoint: ServedEndpoint): LlmServerBadge {
  const runtime = endpoint.runtime?.state;
  // A paused or starting cluster model is named as such even while a parked AI
  // Chat lease leaves nothing "available" to expose. The door is shut, or -
  // after an idle unload, while the wake responder answers - open to wake it;
  // the backend's detail says which, and the note shows it.
  if (endpoint.enabled && runtime === "paused") return { label: "Paused", tone: statusTone("paused") };
  if (endpoint.enabled && runtime === "starting") return { label: "Starting", tone: "info" };
  // A disabled server whose port still answers is the fact to show, whatever
  // is (or is not) behind it.
  if (!endpoint.enabled && runtime === "still-open") return { label: "Disabled \u2014 still open", tone: "warning" };
  if (!endpoint.available) return { label: "No GPU model", tone: "warning" };
  if (!endpoint.enabled) return { label: "Disabled", tone: "neutral" };
  switch (runtime) {
    case "serving":
      return { label: "Serving", tone: "success" };
    case "applying-keys":
      return { label: "Enabled \u2014 applying key change", tone: "warning" };
    case "unknown":
      return { label: "Enabled \u2014 unverified", tone: "warning" };
    case "no-keys":
      return { label: "Enabled \u2014 no API keys", tone: "warning" };
    case "model-not-answering":
      return { label: "Enabled \u2014 model not answering", tone: "danger" };
    default:
      return { label: "Enabled \u2014 not running", tone: "danger" };
  }
}

/**
 * The backend's own sentence for a door that is not serving, or "" when there
 * is nothing to explain (serving, or disabled and closed).
 */
export function llmServerRuntimeNote(endpoint: ServedEndpoint): string {
  if (endpoint.runtime?.state === "serving") return "";
  return endpoint.runtime?.detail ?? "";
}

/**
 * Whether the keys' use figures are current (`usage.state` on `GET /llm-server`).
 * The owner is `vaelor/llm_gate_usage.py` `USAGE_STATES`; this is the one
 * TypeScript copy, pinned by `tests/test_wire_vocabularies.py`.
 */
export const KEY_USAGE_STATES = ["counting", "unreadable", "not-reading", "not-logging"] as const; // vocabulary: llm-gate-usage-state
export type KeyUsageStateWord = (typeof KEY_USAGE_STATES)[number];
const COUNTING: KeyUsageStateWord = KEY_USAGE_STATES[0];

/** Whether the gate's log is being read and written right now. */
export function keyUseIsCounting(usage?: KeyUsageState): boolean {
  return !usage || usage.state === COUNTING;
}

/**
 * The "Last used" cell of one key (ACC-043). "Never used" only while the
 * gate's log is being read and no request was counted - a reader that is
 * failing or stopped, or a gate that is not writing its log, makes absence
 * look like disuse (LESSONS pattern 8), so then it says "Not known", or keeps
 * the last recorded use with a note that it may be older than the truth.
 */
export function keyUseLabel(key: EndpointKey, usage?: KeyUsageState, now = Date.now()): string {
  const counting = keyUseIsCounting(usage);
  const requests = typeof key.requests === "number" ? key.requests : null;
  const count = requests === null
    ? ""
    : ` · ${requests.toLocaleString()} ${requests === 1 ? "request" : "requests"}`;
  const stale = counting ? "" : " (may be out of date)";
  if (key.last_used_at) return `${timeAgo(key.last_used_at * 1000, now)}${count}${stale}`;
  // Counted requests with no time yet: the broker has not taken the stamp.
  if (requests) return `Time not recorded yet${count}${stale}`;
  return counting ? "Never used" : "Not known";
}

/**
 * The sentence under the key table, or "" when there is nothing to add: why
 * the figures are not current, requests lost to a missed log rotation, and
 * refused requests (a wrong or missing key) in the last day.
 */
export function keyUseNote(usage?: KeyUsageState, now = Date.now()): string {
  if (!usage) return "";
  const parts: string[] = [];
  if (!keyUseIsCounting(usage) && usage.detail) parts.push(usage.detail);
  if (usage.last_gap_at && now - usage.last_gap_at * 1000 < 7 * 86_400_000) {
    parts.push(`Some requests around ${timeAgo(usage.last_gap_at * 1000, now)} were not counted: the usage log turned over before Vaelor read it.`);
  }
  if (usage.refused_24h) {
    // A lower bound: refusals are capped on their own so they can never crowd out a key's use.
    parts.push(`At least ${usage.refused_24h.toLocaleString()} ${usage.refused_24h === 1 ? "request was" : "requests were"} refused for a wrong or missing key in the last 24 hours.`);
  }
  return parts.join(" ");
}

/**
 * Home's words for the same state (VD-200, the Main board): the pill says it
 * in a few words ("Needs a key"), and the long label moves to the row's own
 * line, so nothing is lost. The Endpoints panel and AI Chat's summary keep
 * `llmServerBadge` as it is.
 */
export function llmServerHomeBadge(endpoint: ServedEndpoint): LlmServerBadge & { line: string } {
  const badge = llmServerBadge(endpoint);
  const [head, tail] = badge.label.split(" — ");
  if (!tail) return { ...badge, line: "" };
  const label = endpoint.runtime?.state === "no-keys"
    ? "Needs a key"
    : `${tail.charAt(0).toUpperCase()}${tail.slice(1)}`;
  return { label, tone: badge.tone, line: `${head}, ${tail}` };
}
