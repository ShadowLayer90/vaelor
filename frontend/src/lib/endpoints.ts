import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "./api";
import type { AgentRuntime } from "./agentRuntimeStatus";
import { listClusterAgents, type AgentDeployment } from "./clusterAgents";
import { jobIsTerminal } from "./jobPresentation";
import type { LlmServerRuntimeState } from "./llmServerStatus";
import { NOT_ANSWERING, type StatusTone } from "../components/ui/status";

/**
 * Served endpoints and their API keys (F3c): the typed data behind the
 * Cluster › Deployments › Models surface that mints, rotates and revokes the inbound keys guarding Vaelor's served
 * OpenAI-compatible endpoints.
 *
 * Only one endpoint is live today — the LLM Server, id `llm-server` — but the
 * surface is a LIST so cluster-serving and agent endpoints can join it without
 * a second shape. The one live endpoint is still read from `GET /llm-server`
 * (its own surface predates the registry), and mapped into the generic
 * {@link ServedEndpoint} the panel renders.
 *
 * The honesty rule the backend enforces is encoded here: a stored key is a
 * NON-SECRET profile (fingerprint + last4 + label), never a plaintext value.
 * The plaintext lives only in a {@link KeyReveal}, returned exactly once by a
 * mint, a rotate, or the enable that generates the first key — never on a GET.
 */

/** The external LLM Server endpoint. */
export const LLM_SERVER_ENDPOINT_ID = "llm-server";

/**
 * The internal cluster serving key endpoint (F3e). It gates the balancer->worker
 * hop of a replicated Mode-B deployment and is NEVER presented by a client, so
 * it is displayed and rotated in place, never minted or revealed.
 */
export const CLUSTER_SERVING_ENDPOINT_ID = "cluster-serving";

/** The non-secret profile of one active key. Never carries a plaintext value. */
export interface EndpointKey {
  credential_id: string;
  label: string;
  key_fingerprint: string;
  last4: string;
  created_at: number | null;
  last_used_at: number | null;
  /**
   * LLM Server keys: how many requests the gate admitted with this key, from
   * its access log; null when that log has not been read (never a guessed 0).
   */
  requests?: number | null;
  /** Present only if a surface ever lists revoked keys; today's GET omits them. */
  revoked?: boolean;
}

/**
 * Whether the LLM Server's key-use figures are current: `counting` when the
 * control plane is reading the gate's log, otherwise the reason it is not.
 */
export interface KeyUsageState {
  state: string;
  detail: string;
  /** Epoch seconds: when requests were last lost to a missed log rotation. */
  last_gap_at?: number | null;
  /** Requests the gate refused for a wrong or missing key in the last day. */
  refused_24h?: number | null;
}

/**
 * What the LLM Server's LAN door is actually doing, read by the backend from
 * the proxy container's status, a `/health` request through it and the key set
 * it carries. `state` is one word of the backend-owned vocabulary
 * (`LLM_SERVER_RUNTIME_STATES`); `detail` is the backend's sentence for any
 * state but `serving`.
 */
export interface EndpointRuntime {
  state: LlmServerRuntimeState;
  reason: string;
  detail: string;
}

/**
 * One served endpoint, as the panel renders it. `available` gates whether there
 * is anything to serve (a GPU model must be loaded); `enabled` is whether the
 * LAN gate is armed; `base_url` is populated by the backend ONLY while enabled,
 * so an empty string means "no address to advertise yet", never a guess.
 */
export interface ServedEndpoint {
  id: string;
  label: string;
  enabled: boolean;
  available: boolean;
  state: string;
  unavailable_reason: string;
  target_kind?: string;
  model: string;
  /** The LLM Server only: false when which model it serves could not be read (ACC-209). */
  modelKnown?: boolean;
  /** The LLM Server only: the backend's line under the card's name - the model, or why none is named (B3). */
  modelLine?: string;
  /** The LLM Server only: the backend's lead when there is nothing to expose, or "" (B3). */
  unavailableNote?: string;
  /** The LLM Server only: what revoking its last key does (VD-159), in the backend's words, or "". */
  lastKeyNote?: string;
  base_url: string;
  keys: EndpointKey[];
  /**
   * LLM Server only: the door's live state. `enabled` is the owner's choice;
   * this is whether the port is actually answering.
   */
  runtime?: EndpointRuntime;
  /** LLM Server only: whether its keys' use figures are current. */
  usage?: KeyUsageState;
  /**
   * Cluster-serving only (F3e): the key is INTERNAL - it gates the internal
   * balancer->worker hop and is never presented by a client, so it is shown
   * with a fingerprint and rotated in place, never minted or revealed.
   */
  internal?: boolean;
  /** Cluster-serving only: a replicated deployment is keyed, a distributed one is not. */
  replicated?: boolean;
  /** Cluster-serving only: whether the deployment carries an internal key at all. */
  key_present?: boolean;
  /** Cluster-serving only: the internal key's fingerprint (never the key itself). */
  key_fingerprint?: string;
  /**
   * Cluster-serving only: a plain sentence when the replica balancer still runs
   * another Vaelor release's routing (the bridge and the executor are on
   * different releases); empty otherwise.
   */
  balancer_note?: string;
  /** Cluster-serving only: the deployment name a rotate targets. */
  deployment_name?: string;
  /**
   * Deployed-agent only: what the agent is actually doing (its own vocabulary,
   * `AGENT_RUNTIME_STATES`), or null from a backend older than that reading.
   * Kept apart from `runtime`, which is the LLM Server's door.
   */
  agentRuntime?: AgentRuntime | null;
  /** Deployed-agent only: false when the broker could not list its keys. */
  keysKnown?: boolean;
  /**
   * Cluster-serving only (ACC-055): the backend's MEASURED reading of the
   * deployment - the mode watch checks a replicated row's copies and a split
   * row's parts every 30 s - never a fixed "Serving".
   */
  serving?: ClusterServingReading;
}

/** `GET /cluster/serving`'s `serving`: one word and the sentence behind it. */
export interface ClusterServingReading {
  state: string;
  detail: string;
}

/** The card's pill for each word `gpu_pool_serving_health.serving_reading` sends. */
const CLUSTER_SERVING_PILLS: Record<string, { label: string; tone: StatusTone }> = {
  serving: { label: "Serving", tone: "success" },
  "partly-serving": { label: "Partly serving", tone: "warning" },
  degraded: { label: "Degraded", tone: "warning" },
  "not-answering": NOT_ANSWERING,
  "not-checked": { label: "Not checked yet", tone: "neutral" },
  paused: { label: "Paused", tone: "neutral" },
  starting: { label: "Starting", tone: "info" },
  failed: { label: "Failed", tone: "danger" },
};

/** The cluster card's pill, from the measured reading; an unknown word is said so. */
export function clusterServingPill(reading: ClusterServingReading | undefined): { label: string; tone: StatusTone } {
  if (!reading) return { label: "Not reported", tone: "neutral" };
  return CLUSTER_SERVING_PILLS[reading.state] ?? { label: "Unknown state", tone: "warning" };
}

/**
 * The one-time plaintext reveal. `key` is the only place a plaintext value ever
 * appears; the caller shows it once and then discards it. Returned by mint,
 * rotate, and the enable that generates an endpoint's first key.
 */
export interface KeyReveal {
  credential_id: string;
  endpoint_id: string;
  label?: string;
  key_fingerprint?: string;
  last4?: string;
  key: string;
  /**
   * When the change reaches the running gate. LLM Server: the re-key job was
   * `queued`, or is `pending` on Vaelor's reconcile because none could be.
   * Deployed agent: the gate was re-keyed in the request (`applied`), or is
   * `pending` on the reconcile, within about 30 seconds.
   */
  apply?: string;
}

/** A revoke's answer; `apply` as on {@link KeyReveal}. */
export interface KeyRevocation {
  revoked: boolean;
  apply?: string;
}

/** The wire shape of `GET /llm-server` (F3a/F3b). */
export interface LlmServerSurface {
  enabled: boolean;
  state: string;
  available: boolean;
  unavailable_reason: string;
  target_kind?: string;
  model: string;
  /** False when which model the door serves could not be read (absent from an older backend). */
  model_known?: boolean;
  /** The card's line and lead, in the backend's words (B3; absent from an older backend). */
  model_line?: string;
  unavailable_note?: string;
  /** VD-159: the revoke warning for the last key, in the backend's words (absent from an older backend). */
  last_key_note?: string;
  port: number;
  base_url: string;
  keys?: EndpointKey[];
  /** The door's live state (absent from a backend older than this field). */
  runtime?: EndpointRuntime;
  /** Whether the keys' use figures are current (absent from an older backend). */
  usage?: KeyUsageState;
  /** Present on an enable/disable response; the freshly minted plaintext, once. */
  minted_key?: string;
}

/** The wire shape of a mint/rotate response — the plaintext is in `key`. */
interface KeyRevealWire {
  credential_id: string;
  endpoint_id: string;
  label?: string;
  key_fingerprint?: string;
  last4?: string;
  key: string;
  apply?: string;
}

export function toEndpoint(surface: LlmServerSurface): ServedEndpoint {
  return {
    id: LLM_SERVER_ENDPOINT_ID,
    label: "LLM Server",
    enabled: surface.enabled === true,
    available: surface.available === true,
    state: surface.state,
    unavailable_reason: surface.unavailable_reason,
    target_kind: surface.target_kind,
    model: surface.model,
    modelKnown: surface.model_known !== false,
    modelLine: surface.model_line ?? surface.model,
    unavailableNote: surface.unavailable_note ?? "",
    lastKeyNote: surface.last_key_note ?? "",
    base_url: surface.base_url,
    keys: Array.isArray(surface.keys) ? surface.keys : [],
    runtime: surface.runtime,
    usage: surface.usage,
  };
}

/** The read model of `GET /cluster/serving` (F3e). */
interface ClusterServingEndpoint {
  id: string;
  label: string;
  name: string;
  model: string;
  base_url: string;
  internal: boolean;
  replicated: boolean;
  key_present: boolean;
  key_fingerprint: string;
  state: string;
  serving?: ClusterServingReading;
  balancer_note?: string;
}

/** The wire shape of `GET /cluster/serving`: an endpoint, or absent. */
interface ClusterServingSurface {
  present: boolean;
  endpoint: ClusterServingEndpoint | null;
}

function toClusterEndpoint(view: ClusterServingEndpoint): ServedEndpoint {
  return {
    id: CLUSTER_SERVING_ENDPOINT_ID,
    label: view.label || "Cluster serving",
    // No enable toggle - the lifecycle is the deployment's, so the surface
    // never offers to arm it. `state` is the row's own; `serving` is what was
    // measured (ACC-055), and the card's pill reads that.
    enabled: true,
    available: true,
    state: view.state || "",
    unavailable_reason: "",
    target_kind: "cluster",
    model: view.model || "",
    base_url: view.base_url || "",
    keys: [],
    internal: view.internal === true,
    replicated: view.replicated === true,
    key_present: view.key_present === true,
    key_fingerprint: view.key_fingerprint || "",
    deployment_name: view.name || "",
    balancer_note: typeof view.balancer_note === "string" ? view.balancer_note : "",
    serving: view.serving && typeof view.serving.state === "string"
      ? { state: view.serving.state, detail: typeof view.serving.detail === "string" ? view.serving.detail : "" }
      : undefined,
  };
}

/** The `target_kind` that marks a deployed cluster agent in the endpoints list. */
export const AGENT_TARGET_KIND = "agent";

/**
 * Map one deployed agent (F6d) into the generic {@link ServedEndpoint} the panel
 * already renders, so it reads as a natural entry in the same list.
 *
 * An agent is reached behind its own LAN gate, which accepts ANY of the
 * endpoint's active keys. `keys` are those keys exactly as the broker lists
 * them (fingerprint, last four, label - never a value), with `keysKnown` false
 * when the broker could not answer: an empty list then means "unknown", not
 * "none" (ACC-071). The card's badge comes from `agentRuntime`, what the agent
 * is doing, never from the stored deploy row.
 */
export function toAgentEndpoint(agent: AgentDeployment): ServedEndpoint {
  return {
    id: agent.endpoint_id,
    label: agent.name || "Cluster agent",
    // An agent has no enable toggle - its lifecycle is the deployment's - so
    // `enabled`/`available` carry no meaning for it; the card reads
    // `agentRuntime` instead.
    enabled: true,
    available: true,
    state: agent.state || "",
    unavailable_reason: "",
    target_kind: AGENT_TARGET_KIND,
    model: agent.backing.model_deployment_name || "",
    base_url: agent.endpoint || "",
    keys: agent.keys,
    keysKnown: agent.keys_known,
    agentRuntime: agent.runtime,
  };
}

export interface EndpointsState {
  endpoints: ServedEndpoint[];
  loading: boolean;
  error: string;
  /**
   * Why the deployed agents could not be read, or "". The other endpoints are
   * still listed; the panel says the agents' endpoints and keys are missing
   * rather than letting an unreadable list pass for "no agents" (ACC-081).
   */
  agentsError: string;
  reload: () => void;
}

/**
 * Read the served endpoints, exposing the three states the surface needs: a
 * first load, a served list, and a failure carrying the appliance's own
 * message. The request is abortable so leaving the tab does not land a stale
 * answer, and `reload` re-reads after a mint, rotate, revoke or toggle.
 *
 * `pageKey` is the hosting page's own freshness: when it changes (its Reload,
 * an operation's Done, a deployment's polled state) the cards are read again.
 * Without it they were read once per mount, so after an Unload they said
 * SERVING until a browser reload (ACC-202).
 */
/**
 * How soon an enabled LLM Server door that is not settled is read again (W4-D5).
 * Its state can change with nothing on the page changing: after a Remove the
 * GPU watch brings the door back on its next pass, after the page's last read.
 */
export const UNSETTLED_DOOR_REREAD_MS = 5_000;

/**
 * How many re-reads in a row one unchanged door gets (F1): two minutes at
 * {@link UNSETTLED_DOOR_REREAD_MS}, more than the W4-D5 window (about 40 s). A
 * door still unchanged then is not coming back by itself; the page's Reload,
 * an operation finishing, or a change of state reads it again.
 */
export const UNSETTLED_DOOR_MAX_REREADS = 24;

/**
 * How many re-reads one mount gets in one visible period, whatever the door
 * reads (B5): five minutes at {@link UNSETTLED_DOOR_REREAD_MS}. The per-reading
 * count above starts again on every change, so a door flapping between two
 * states was otherwise read every 5 s for as long as the page stayed open
 * (LESSONS 16). Hiding the page and seeing it again starts a new period.
 */
export const UNSETTLED_DOOR_MAX_TOTAL_REREADS = 60;

/** The door states that change on their own, so the card reads them again. */
const UNSETTLED_DOOR_STATES = new Set<LlmServerRuntimeState>([
  "not-running", "model-not-answering", "starting", "unknown", "applying-keys", "still-open",
]);

/**
 * Whether an endpoint is the LLM Server's door in a state that changes on its
 * own (F1): enabled and not-running for a reason other than having nothing to
 * front (`no-target`, which no wait changes) - or disabled while its port still
 * answers. A disabled, closed door is the owner's settled choice (FE9), but
 * `still-open` is the disable not yet applied: W4d-D23 left the card on it 30 s
 * after the port refused, because a disabled door was never read again.
 */
export function doorUnsettled(door: ServedEndpoint | undefined): boolean {
  if (!door || door.id !== LLM_SERVER_ENDPOINT_ID || !door.runtime) return false;
  if (!door.enabled) return door.runtime.state === "still-open";
  const { state, reason } = door.runtime;
  if (state === "not-running" && reason === "no-target") return false;
  return UNSETTLED_DOOR_STATES.has(state);
}

export function useEndpoints(pageKey: string | number = ""): EndpointsState {
  const [endpoints, setEndpoints] = useState<ServedEndpoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [agentsError, setAgentsError] = useState("");
  const [reloadKey, setReloadKey] = useState(0);
  const reload = useCallback(() => setReloadKey((value) => value + 1), []);
  const door = endpoints[0];
  const unsettled = doorUnsettled(door);
  // The reading the re-reads are counted against: a change of state starts
  // the count again, the same state does not.
  const doorReading = door?.runtime ? `${door.runtime.state}|${door.runtime.reason}` : "";
  const rereads = useRef({ reading: "", count: 0 });
  // B5: the re-reads this visible period, and which period that is.
  const [visiblePeriod, setVisiblePeriod] = useState(0);
  const total = useRef({ period: 0, count: 0, returned: false });
  useEffect(() => {
    const seen = () => {
      if (!document.hidden) setVisiblePeriod((value) => value + 1);
    };
    document.addEventListener("visibilitychange", seen);
    return () => document.removeEventListener("visibilitychange", seen);
  }, []);

  // W4-D5 / F1: while the door is changing, read it again on a short clock -
  // a cheap GET - only while the page is visible, at most
  // UNSETTLED_DOOR_MAX_REREADS times for one unchanged reading, and never
  // once it settles. Each mount keeps its own clock: the two (the
  // Deployments tab's endpoint cards and AI Chat's summary) live on
  // different pages, so they are not polling at the same time.
  useEffect(() => {
    if (rereads.current.reading !== doorReading) rereads.current = { reading: doorReading, count: 0 };
    // A page seen again starts a new period, which reads at once (B5).
    if (total.current.period !== visiblePeriod) total.current = { period: visiblePeriod, count: 0, returned: true };
    if (!unsettled || loading || document.hidden || rereads.current.count >= UNSETTLED_DOOR_MAX_REREADS
      || total.current.count >= UNSETTLED_DOOR_MAX_TOTAL_REREADS) return undefined;
    const reread = () => {
      rereads.current.count += 1;
      total.current.count += 1;
      total.current.returned = false;
      reload();
    };
    // Hidden, nothing is scheduled; the period effect above notices the
    // return to view and this runs again.
    const timer = window.setTimeout(reread, total.current.returned ? 0 : UNSETTLED_DOOR_REREAD_MS);
    return () => window.clearTimeout(timer);
  }, [unsettled, loading, reload, endpoints, doorReading, visiblePeriod]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    const llmServer = apiRequest<LlmServerSurface>("/llm-server", { cache: "no-store", signal: controller.signal });
    // The cluster-serving endpoint is ADDITIVE: a box with no cluster (or a
    // read that fails) simply has no such endpoint, and that must never fail
    // the whole surface, so this read degrades to "absent" on any error.
    const clusterServing = apiRequest<ClusterServingSurface>("/cluster/serving", { cache: "no-store", signal: controller.signal })
      .catch(() => ({ present: false, endpoint: null }) as ClusterServingSurface);
    // The deployed agents are additive too - a failed read must not blank the
    // LLM Server - but NOT silent: an unreadable agent list is reported as
    // such, never passed off as "no agents" (ACC-081).
    let agentsFailure = "";
    const agents = listClusterAgents(controller.signal).catch((reason: unknown) => {
      agentsFailure = reason instanceof Error && reason.message
        ? reason.message
        : "The request did not complete.";
      return [] as AgentDeployment[];
    });
    Promise.all([llmServer, clusterServing, agents])
      .then(([surface, cluster, agentList]) => {
        const list = [toEndpoint(surface)];
        if (cluster && cluster.present && cluster.endpoint) {
          list.push(toClusterEndpoint(cluster.endpoint));
        }
        for (const agent of agentList) {
          // Skip a malformed agent with no addressing id: its key actions
          // could not be routed, so it has no place in this surface.
          if (agent.endpoint_id) list.push(toAgentEndpoint(agent));
        }
        setEndpoints(list);
        setError("");
        setAgentsError(agentsFailure);
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setError(reason instanceof Error ? reason.message : "The served endpoints could not be read.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [reloadKey, pageKey]);

  return { endpoints, loading, error, agentsError, reload };
}

/** Mint a fresh key for an endpoint. Returns the plaintext ONCE. */
export function mintEndpointKey(
  endpointId: string,
  label: string,
  csrfToken: string,
): Promise<KeyReveal> {
  return apiRequest<KeyRevealWire>(
    `/endpoints/${encodeURIComponent(endpointId)}/keys`,
    { method: "POST", body: JSON.stringify({ label }), cache: "no-store" },
    csrfToken,
  );
}

/** Rotate one key in place. Returns the new plaintext ONCE. */
export function rotateEndpointKey(
  endpointId: string,
  credentialId: string,
  csrfToken: string,
): Promise<KeyReveal> {
  return apiRequest<KeyRevealWire>(
    `/endpoints/${encodeURIComponent(endpointId)}/keys/${encodeURIComponent(credentialId)}/rotate`,
    { method: "POST", body: "{}", cache: "no-store" },
    csrfToken,
  );
}

/**
 * Rotate the INTERNAL cluster serving key in place (F3e). This never reveals a
 * plaintext key: the key gates the balancer->worker hop and is never presented
 * by a client, so a rotation only re-keys the balancer and every worker gate.
 * It enqueues an overlap-tolerant rotation job and returns the queued job's id.
 */
export function rotateClusterKey(
  csrfToken: string,
): Promise<{ job_id: string; apply: string; name: string }> {
  return apiRequest<{ job_id: string; apply: string; name: string }>(
    "/cluster/serving/rotate-key",
    { method: "POST", body: "{}", cache: "no-store" },
    csrfToken,
  );
}

/** A key job as {@link waitForJobEnd} reads it: its state and its own sentence. */
export interface KeyJob {
  id: string;
  state: string;
  message?: string;
}

/** How often, and how many times, {@link waitForJobEnd} reads a key job. */
export const KEY_JOB_POLL_MS = 2_000;
export const KEY_JOB_MAX_READS = 150;

/**
 * Read a queued key job until it ends (W4d-D15): the cluster-key rotate swaps
 * the fingerprint when its job finishes, not when it is queued, so a card read
 * at queue time keeps the old one. Bounded (five minutes), stopped by
 * `signal`; a read that fails is retried, and the last job seen (or null) is
 * returned either way, so the caller always reads its card once more.
 */
export async function waitForJobEnd(
  jobId: string,
  signal: AbortSignal,
): Promise<KeyJob | null> {
  let last: KeyJob | null = null;
  for (let read = 0; read < KEY_JOB_MAX_READS && !signal.aborted; read += 1) {
    try {
      last = await apiRequest<KeyJob>(
        `/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store", signal },
      );
      if (jobIsTerminal(last)) return last;
    } catch {
      if (signal.aborted) break;
    }
    await new Promise((resolve) => window.setTimeout(resolve, KEY_JOB_POLL_MS));
  }
  return last;
}

/**
 * Revoke one key; the row is kept for audit. It stops being accepted once the
 * endpoint's gate is re-keyed - `apply` on the answer says whether that has
 * happened (`applied`) or is left to Vaelor's reconcile (`pending`/`queued`).
 */
export function revokeEndpointKey(
  endpointId: string,
  credentialId: string,
  csrfToken: string,
): Promise<KeyRevocation> {
  return apiRequest<KeyRevocation>(
    `/endpoints/${encodeURIComponent(endpointId)}/keys/${encodeURIComponent(credentialId)}`,
    { method: "DELETE", cache: "no-store" },
    csrfToken,
  );
}

/**
 * Arm or disarm the LLM Server's LAN gate. Enabling generates the endpoint's
 * first key when it has none, and the response carries that plaintext once in
 * `minted_key`; disabling keeps every key so re-enabling restores them. The
 * route is LLM-Server-specific because it also relaunches the auth proxy.
 */
export async function setLlmServerEnabled(
  enabled: boolean,
  csrfToken: string,
): Promise<{ endpoint: ServedEndpoint; reveal: string }> {
  const surface = await apiRequest<LlmServerSurface>(
    enabled ? "/llm-server/enable" : "/llm-server/disable",
    { method: "POST", body: "{}", cache: "no-store" },
    csrfToken,
  );
  return { endpoint: toEndpoint(surface), reveal: surface.minted_key ?? "" };
}
