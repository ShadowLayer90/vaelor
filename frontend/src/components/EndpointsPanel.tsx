import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { ApiError } from "../lib/api";
import {
  AGENT_TARGET_KIND,
  CLUSTER_SERVING_ENDPOINT_ID,
  clusterServingPill,
  LLM_SERVER_ENDPOINT_ID,
  mintEndpointKey,
  revokeEndpointKey,
  rotateClusterKey,
  rotateEndpointKey,
  waitForJobEnd,
  setLlmServerEnabled,
  useEndpoints,
  type EndpointKey,
  type ServedEndpoint,
} from "../lib/endpoints";
import { agentRuntimeBadge, agentRuntimeNote } from "../lib/agentRuntimeStatus";
import { exactTime, timeAgo } from "../lib/format";
import { jobIsSuccessful, jobIsTerminal } from "../lib/jobPresentation";
import { keyUseLabel, keyUseNote, llmServerBadge, llmServerRuntimeNote } from "../lib/llmServerStatus";
import type { Session } from "../types";
import { ClusterCard } from "./ClusterPrimitives";
import { RevealKeyModal } from "./RevealKeyModal";
import { StatusPill } from "./StatusPill";
import { Button, EmptyState, Input, LoadingLines, Notice } from "./ui";
import { ClusterConfirm } from "./ClusterConfirm";
import "../styles/cluster-deployments.css";

/**
 * The served endpoints and their API keys (F3c), shown under the Models filter
 * of Cluster › Deployments (VD-200): the surface that manages the inbound keys
 * guarding Vaelor's served OpenAI-compatible endpoints.
 *
 * For each endpoint it shows the base URL and a ready-to-paste curl snippet
 * (each with a copy button), whether it is serving, and a table of its ACTIVE
 * keys — label, fingerprint, last four, created and last-used — with a rotate
 * and a revoke per key. Mint is the one primary action per endpoint; rotate and
 * the toggle are secondary; revoke is destructive and confirmed. A stored key
 * never shows its plaintext: minting, rotating and the enable that generates
 * the first key each surface it once through {@link RevealKeyModal}, and it is
 * discarded when that modal closes.
 *
 * The list is built to hold more than one endpoint, but only `llm-server` is
 * live today, so the toggle is wired to its LAN-gate route.
 *
 * VD-200 (the ClusterDeploymentsModels board): each endpoint is a card under
 * Deployments > Models - the LLM Server, then the model library the tab hands
 * in as `library`, then the cluster balancer and each deployed agent - and the
 * key confirmations are the ClusterDialogsManage2 board's "API keys" dialogs.
 */

/** The one-time reveal currently on screen, or `null`. Holds the plaintext. */
interface Reveal {
  title: string;
  description: string;
  key: string;
}

/** A key queued for a confirmed revoke, with the endpoint it belongs to. */
interface RevokeTarget {
  endpointId: string;
  isAgent: boolean;
  key: EndpointKey;
  /** The backend's warning when this is the LLM Server's last active key (VD-159), else "". */
  lastKeyNote: string;
}

/**
 * When a key change reaches the LLM Server. Its proxy renders the key SET into
 * its config, so a mint, rotate or revoke takes effect when the proxy is
 * re-keyed - by the job the change queues, or by Vaelor's reconcile - not the
 * moment the key store changes. Said here once, so no copy claims "immediately".
 */
const LLM_SERVER_REKEY_WINDOW = "within about 30 seconds, once its proxy is re-keyed";
const REKEY_PENDING_NOTE =
  "The key change is saved, but Vaelor could not queue the job that re-keys the LLM Server, so its next reconcile pass applies it instead.";

// VD-158: a request that names its conversation is kept on one machine, so
// its follow-up turns reuse what that machine already worked out. A client
// that sends no header still works; its requests are spread, turn by turn.
const SESSION_HEADER_EXAMPLE = "X-Session-Id: my-conversation-1";
// Said directly only while the cluster serves one copy of the model per
// machine behind its balancer: that is the only state in which the header
// changes anything (re-review R3). Everywhere else it is said as a condition,
// and no example row is offered.
const SESSION_HEADER_RULE =
  "Any name of your own works: letters, numbers and . _ : - up to 128 characters. It is optional.";
const PASTE_NOTE = "Paste these into an OpenAI-compatible app on your LAN.";
const SESSION_HEADER_NOTE =
  `${PASTE_NOTE} Send one X-Session-Id per conversation so follow-up replies start faster. `
  + SESSION_HEADER_RULE;
const SESSION_HEADER_CONDITIONAL_NOTE =
  `${PASTE_NOTE} When the model runs as one copy per machine, send one X-Session-Id per `
  + "conversation so follow-up replies start faster. " + SESSION_HEADER_RULE;

/** Whether the cluster serves one copy per machine behind its balancer, now. */
function servesOneCopyPerMachine(endpoints: ServedEndpoint[]): boolean {
  return endpoints.some((endpoint) => (
    endpoint.id === CLUSTER_SERVING_ENDPOINT_ID && endpoint.replicated === true && endpoint.state === "healthy"
  ));
}

const CLUSTER_INTERNAL_NOTE =
  "Internal: it authenticates the balancer to each worker replica and is never presented by a client, so it is shown by fingerprint and rotated in place, never revealed.";
const CLUSTER_KEYLESS_NOTE =
  "This is a single-head (distributed) deployment, so it has no client key. Its loopback endpoint is reached only through the LLM Server proxy.";
/** Why rotate is off: `POST /cluster/serving/rotate-key` refuses a row that is not healthy. */
const CLUSTER_ROTATE_OFF_NOTE =
  "The internal key can be rotated only while the model is serving; this deployment is not serving now.";
const CLUSTER_REVOKE_NOTE =
  "There is no separate revoke; its lifecycle is the deployment's. To remove it, tear down the replicated deployment.";
/**
 * When a key change reaches a deployed agent. Its gate is re-keyed inside the
 * mint, rotate or revoke request, and when that cannot happen Vaelor's
 * reconcile does it - so the honest timing is "at once, or within about 30
 * seconds", never "immediately" (ACC-081).
 */
const AGENT_REKEY_WINDOW = "as soon as the agent's gate is re-keyed - normally at once, at most about 30 seconds";
const AGENT_REKEY_PENDING_NOTE =
  "The key change is saved, but the agent's gate could not be re-keyed during the request, so Vaelor's reconcile applies it within about 30 seconds.";
const AGENT_KEY_NOTE =
  "Each agent accepts any of its active keys. Its first key was shown once at deploy; issue another for each app that calls it.";

/** What each key's line says, in order, named once beside the "API keys" heading. */
const KEY_COLUMNS = ["Label", "Fingerprint", "Last 4", "Created", "Last used"] as const;

/** A key named for a dialog: its label and last four, or "This key". */
function keyName(key: EndpointKey): string {
  const label = key.label || "This key";
  return key.last4 ? `${label} (ending ${key.last4})` : label;
}

function statePill(endpoint: ServedEndpoint, busyToggle: boolean) {
  if (busyToggle) return { label: "Applying…", tone: "info" as const };
  return llmServerBadge(endpoint);
}

export function EndpointsPanel({
  library,
  session,
  reloadKey = "",
}: {
  /** The model library card, drawn beside the LLM Server card (the board's grid). */
  library?: ReactNode;
  session: Session;
  reloadKey?: string;
}) {
  const { endpoints, loading, error, agentsError, reload } = useEndpoints(reloadKey);
  const sessionRouting = servesOneCopyPerMachine(endpoints);
  const [busy, setBusy] = useState("");
  const [actionError, setActionError] = useState("");
  // N1 (LESSONS 8, VD-189): the cluster rotate's outcome lands minutes after
  // its dialog closed, perhaps while another dialog is open, so it has its own
  // state that no other action clears - only the next cluster rotate.
  const [rotateOutcome, setRotateOutcome] = useState<{ severity: "danger" | "info"; text: string } | null>(null);
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [reveal, setReveal] = useState<Reveal | null>(null);
  const [revokeTarget, setRevokeTarget] = useState<RevokeTarget | null>(null);
  // W6-5 (VD-159): keys this page revoked whose reload has not landed yet. The
  // last-key count leaves them out, so a second quick revoke still warns.
  const [revokedHere, setRevokedHere] = useState<ReadonlySet<string>>(() => new Set());
  const [copied, setCopied] = useState("");
  const [rotateClusterOpen, setRotateClusterOpen] = useState(false);
  // W4d-D15: a per-key rotate confirms first, as revoke does.
  const [rotateTarget, setRotateTarget] = useState<{ endpoint: ServedEndpoint; key: EndpointKey } | null>(null);
  // The rotate job's wait ends with the panel.
  const lifetime = useRef<AbortController | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    lifetime.current = controller;
    return () => controller.abort();
  }, []);
  const [rekeyNote, setRekeyNote] = useState("");
  // R-F1 (LESSONS 19 / VD-189): a confirm the server refuses stays open and
  // says why INSIDE the dialog, never on the inert page behind it.
  const [dialogError, setDialogError] = useState("");

  const fail = useCallback((cause: unknown, fallback: string) => {
    setActionError(cause instanceof ApiError ? cause.message : fallback);
  }, []);
  const refuse = useCallback((cause: unknown, fallback: string) => {
    setDialogError(cause instanceof ApiError ? cause.message : fallback);
  }, []);

  const copy = useCallback((field: string, value: string) => {
    if (!value || !navigator.clipboard) return;
    void navigator.clipboard.writeText(value).then(
      () => {
        setCopied(field);
        window.setTimeout(() => setCopied((current) => (current === field ? "" : current)), 1500);
      },
      () => setCopied(""),
    );
  }, []);

  const mint = useCallback(async (endpoint: ServedEndpoint) => {
    setBusy(`mint:${endpoint.id}`);
    setActionError("");
    setDialogError("");
    const isAgent = endpoint.target_kind === AGENT_TARGET_KIND;
    try {
      const label = (labels[endpoint.id] ?? "").trim();
      const result = await mintEndpointKey(endpoint.id, label, session.csrf_token);
      setLabels((current) => ({ ...current, [endpoint.id]: "" }));
      const accepted = endpoint.id === LLM_SERVER_ENDPOINT_ID
        ? ` The LLM Server starts accepting it ${LLM_SERVER_REKEY_WINDOW}.`
        : isAgent
          ? ` The agent accepts it ${AGENT_REKEY_WINDOW}.`
          : "";
      setReveal({
        title: "New API key",
        description: `A new key for ${endpoint.label} was created${result.label ? ` and named “${result.label}”` : ""}.${accepted}`,
        key: result.key,
      });
      setRekeyNote(result.apply === "pending" ? (isAgent ? AGENT_REKEY_PENDING_NOTE : REKEY_PENDING_NOTE) : "");
      reload();
    } catch (cause) {
      fail(cause, "The key could not be minted.");
    } finally {
      setBusy("");
    }
  }, [fail, labels, reload, session.csrf_token]);

  const rotate = useCallback(async (endpoint: ServedEndpoint, key: EndpointKey) => {
    setBusy(`rotate:${key.credential_id}`);
    setActionError("");
    setDialogError("");
    const isAgent = endpoint.target_kind === AGENT_TARGET_KIND;
    try {
      const result = await rotateEndpointKey(endpoint.id, key.credential_id, session.csrf_token);
      setReveal({
        title: "Key rotated",
        description: endpoint.id === LLM_SERVER_ENDPOINT_ID
          ? `${key.label || "This key"} now has a new value. The LLM Server switches from the old value to the new one ${LLM_SERVER_REKEY_WINDOW}.`
          : isAgent
            ? `${key.label || "This key"} now has a new value. The old value stops working ${AGENT_REKEY_WINDOW}.`
            : `${key.label || "This key"} now has a new value. Its old value stopped working the moment it was rotated.`,
        key: result.key,
      });
      setRekeyNote(result.apply === "pending" ? (isAgent ? AGENT_REKEY_PENDING_NOTE : REKEY_PENDING_NOTE) : "");
      setRotateTarget(null);
      reload();
    } catch (cause) {
      refuse(cause, "The key could not be rotated.");
    } finally {
      setBusy("");
    }
  }, [refuse, reload, session.csrf_token]);

  const toggle = useCallback(async (endpoint: ServedEndpoint) => {
    setBusy(`toggle:${endpoint.id}`);
    setActionError("");
    setDialogError("");
    try {
      const { reveal: minted } = await setLlmServerEnabled(!endpoint.enabled, session.csrf_token);
      if (minted) {
        setReveal({
          title: "Endpoint enabled",
          description: `${endpoint.label} is now exposed on your LAN, protected by this new key.`,
          key: minted,
        });
      }
      reload();
    } catch (cause) {
      fail(cause, "That change could not be applied.");
    } finally {
      setBusy("");
    }
  }, [fail, reload, session.csrf_token]);

  const confirmRevoke = useCallback(async () => {
    if (!revokeTarget) return;
    setBusy(`revoke:${revokeTarget.key.credential_id}`);
    setActionError("");
    setDialogError("");
    try {
      const result = await revokeEndpointKey(revokeTarget.endpointId, revokeTarget.key.credential_id, session.csrf_token);
      setRekeyNote(result.apply === "pending" ? (revokeTarget.isAgent ? AGENT_REKEY_PENDING_NOTE : REKEY_PENDING_NOTE) : "");
      setRevokedHere((current) => new Set(current).add(revokeTarget.key.credential_id));
      setRevokeTarget(null);
      reload();
    } catch (cause) {
      refuse(cause, "The key could not be revoked.");
    } finally {
      setBusy("");
    }
  }, [refuse, reload, revokeTarget, session.csrf_token]);

  const rotateCluster = useCallback(async () => {
    setBusy("rotate-cluster");
    setActionError("");
    setRotateOutcome(null);
    setDialogError("");
    try {
      const queued = await rotateClusterKey(session.csrf_token);
      setRotateClusterOpen(false);
      reload();
      // W4d-D15: the fingerprint changes when the job ends; read it then.
      const signal = lifetime.current?.signal;
      if (queued.job_id && signal) {
        void waitForJobEnd(queued.job_id, signal).then((job) => {
          if (signal.aborted) return;
          // R-F12 (LESSONS 8): a rotation whose job did not succeed is said,
          // never reloaded silently - the card then shows the key still in use.
          // N1: three ends, not two - waitForJobEnd gives up after five
          // minutes and hands back a job that may still be running.
          if (!job) {
            setRotateOutcome({ severity: "danger", text: "Vaelor could not read whether the cluster key rotation finished. The card shows the key it reads now." });
          } else if (!jobIsTerminal(job)) {
            setRotateOutcome({ severity: "info", text: `The cluster key rotation is still running${job.message ? ` (${job.message})` : ""}. Follow it in Activity; this card shows the new key once it finishes and is read again.` });
          } else if (!jobIsSuccessful(job)) {
            setRotateOutcome({ severity: "danger", text: `The cluster key was not rotated${job.message ? `: ${job.message}` : "."} The card shows the key the cluster still uses.` });
          }
          reload();
        });
      }
    } catch (cause) {
      refuse(cause, "The cluster key could not be rotated.");
    } finally {
      setBusy("");
    }
  }, [refuse, reload, session.csrf_token]);


  const copyButton = (field: string, value: string, label?: string) => (
    <Button aria-label={label} onClick={() => copy(field, value)} type="button" variant="quiet">
      {copied === field ? "Copied" : "Copy"}
    </Button>
  );

  const clusterCard = (endpoint: ServedEndpoint) => {
    const pill = clusterServingPill(endpoint.serving);
    const rotateOff = endpoint.state !== "healthy";
    return (
      <ClusterCard
        actions={<><StatusPill label="Internal" tone="info" /><StatusPill {...pill} /></>}
        aria-label={endpoint.label}
        as="article"
        description={endpoint.model || "Internal cluster serving"}
        icon="key"
        key={endpoint.id}
        title={endpoint.label}
      >
        {endpoint.balancer_note ? <Notice severity="warning">{endpoint.balancer_note}</Notice> : null}
        {endpoint.serving && endpoint.serving.state !== "serving" && endpoint.serving.detail ? (
          <Notice severity={pill.tone === "danger" ? "danger" : pill.tone === "warning" ? "warning" : "info"}>
            {endpoint.serving.detail}
          </Notice>
        ) : null}
        {endpoint.base_url && (
          <p>
            Base URL <code className="cl-ep-mono">{endpoint.base_url}</code>
            {endpoint.key_present && (
              <> · Internal key <code className="cl-ep-mono">…{(endpoint.key_fingerprint || "").slice(-8)}</code></>
            )}
          </p>
        )}
        {endpoint.key_present ? (
          <>
            <p className="cl-meta">{CLUSTER_INTERNAL_NOTE} {CLUSTER_REVOKE_NOTE}</p>
            <div className="cl-actions cl-actions--end">
              <Button
                disabled={busy !== "" || rotateOff}
                disabledReason={rotateOff ? CLUSTER_ROTATE_OFF_NOTE : undefined}
                onClick={() => setRotateClusterOpen(true)}
                type="button"
                variant="secondary"
              >
                {busy === "rotate-cluster" ? "Rotating…" : "Rotate cluster key"}
              </Button>
            </div>
          </>
        ) : (
          <Notice severity="info">{CLUSTER_KEYLESS_NOTE}</Notice>
        )}
      </ClusterCard>
    );
  };

  const servedCard = (endpoint: ServedEndpoint, firstAgent = false) => {
    const isAgent = endpoint.target_kind === AGENT_TARGET_KIND;
    const pill = isAgent
      ? agentRuntimeBadge(endpoint.agentRuntime, endpoint.state)
      : statePill(endpoint, busy === `toggle:${endpoint.id}`);
    const runtimeNote = isAgent ? agentRuntimeNote(endpoint.agentRuntime) : llmServerRuntimeNote(endpoint);
    // An agent's key list is only a fact when the broker answered; an
    // unreadable list is never shown as "0 active".
    const keysKnown = !isAgent || endpoint.keysKnown === true;
    const curl = endpoint.base_url
      ? `curl ${endpoint.base_url}/models -H "Authorization: Bearer $VAELOR_KEY"`
      : "";
    const activeCount = endpoint.keys.length;
    return (
      <ClusterCard
        actions={(
          <>
            <StatusPill label={pill.label} tone={pill.tone} />
            {!isAgent && endpoint.available && (
              <Button
                className={endpoint.enabled ? "cl-danger-outline" : undefined}
                disabled={busy !== ""}
                onClick={() => void toggle(endpoint)}
                type="button"
                variant="secondary"
              >
                {busy === `toggle:${endpoint.id}`
                  ? (endpoint.enabled ? "Disabling…" : "Enabling…")
                  : (endpoint.enabled ? "Disable" : "Enable")}
              </Button>
            )}
          </>
        )}
        aria-label={endpoint.label}
        as="article"
        // B3: the LLM Server's line is the backend's sentence (model_line).
        description={isAgent ? <>Cluster agent · <span>{endpoint.model || "model not recorded"}</span></> : endpoint.modelLine}
        icon={isAgent ? "assistant" : "server"}
        key={endpoint.id}
        title={endpoint.label}
      >
        {/* The agent key rule, said once: on the first agent's card (the
            board's place), not repeated on every card after it. */}
        {firstAgent && <p className="cl-meta">{AGENT_KEY_NOTE}</p>}
        {/* One notice at a time: the backend sends its lead only when the
            runtime has no sentence of its own saying why (B3). */}
        {!isAgent && endpoint.unavailableNote && <Notice severity="info">{endpoint.unavailableNote}</Notice>}
        {runtimeNote && (
          <Notice severity={pill.tone === "danger" || pill.tone === "warning" ? "warning" : "info"}>{runtimeNote}</Notice>
        )}

        {endpoint.enabled && endpoint.base_url && !(isAgent && keysKnown && !endpoint.keys.length) ? (
          <>
            <dl className="cl-ep-rows">
              <div>
                <dt>LAN base URL</dt>
                <dd><code>{endpoint.base_url}</code>{copyButton(`base:${endpoint.id}`, endpoint.base_url)}</dd>
              </div>
              <div>
                <dt>Example request</dt>
                <dd><code>{curl}</code>{copyButton(`curl:${endpoint.id}`, curl)}</dd>
              </div>
              {sessionRouting && (
                <div>
                  <dt>Conversation header</dt>
                  <dd>
                    <code>{SESSION_HEADER_EXAMPLE}</code>
                    {copyButton(`session:${endpoint.id}`, SESSION_HEADER_EXAMPLE, "Copy conversation header")}
                  </dd>
                </div>
              )}
            </dl>
            <p className="cl-meta">{sessionRouting ? SESSION_HEADER_NOTE : SESSION_HEADER_CONDITIONAL_NOTE}</p>
          </>
        ) : isAgent && endpoint.base_url ? (
          // No active key: the gate is closed, so there is nothing to call.
          <p className="cl-meta">This agent's LAN gate is closed until it has a key, so there is no address or example request to copy yet.</p>
        ) : isAgent ? (
          // An agent has no Enable control: without an address it is still
          // deploying or its deploy failed, which the runtime note explains
          // when the backend gave one.
          <p className="cl-meta">{runtimeNote
            ? "This agent has no LAN address yet, so there is no example request to copy. The note above says why."
            : "This agent has no LAN address yet, so there is no example request to copy."}</p>
        ) : endpoint.available ? (
          <p className="cl-meta">Enable this endpoint to get its base URL and example request.</p>
        ) : null}

        <div className="cl-ep-section-head">
          <h3>API keys</h3>
          <span className="cl-meta">
            {!keysKnown ? "Keys could not be read" : (
              <>
                <span>{`${activeCount} active`}</span>
                {activeCount > 0 && <span>{` · ${KEY_COLUMNS.join(", ").toLowerCase()}`}</span>}
              </>
            )}
          </span>
        </div>
        {endpoint.keys.length ? (
          <>
            <ul aria-label={`${endpoint.label} API keys`} className="cl-ep-keys">
              {endpoint.keys.map((key) => (
                <li key={key.credential_id}>
                  <div className="cl-rows__text">
                    <strong>{key.label || "Unnamed key"}</strong>
                    <span>{keyFacts(key, isAgent ? agentKeyUse(key) : llmKeyUse(keyUseLabel(key, endpoint.usage)))}</span>
                  </div>
                  <div className="cl-actions">
                    <Button
                      disabled={busy !== ""}
                      onClick={() => setRotateTarget({ endpoint, key })}
                      type="button"
                      variant="secondary"
                    >
                      {busy === `rotate:${key.credential_id}` ? "Rotating…" : "Rotate"}
                    </Button>
                    <Button
                      aria-label={`Revoke ${key.label || "unnamed key"}`}
                      className="cl-danger-outline"
                      disabled={busy !== ""}
                      onClick={() => setRevokeTarget({ endpointId: endpoint.id, isAgent, key, lastKeyNote: endpoint.keys.filter((item) => !item.revoked && !revokedHere.has(item.credential_id)).length === 1 ? endpoint.lastKeyNote ?? "" : "" })}
                      type="button"
                      variant="secondary"
                    >
                      Revoke
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
            {keyUseNote(endpoint.usage) && <p className="cl-meta">{keyUseNote(endpoint.usage)}</p>}
          </>
        ) : (
          // A keyless agent's closed gate is already said above the keys.
          isAgent && keysKnown ? null : (
            <p className="cl-meta">{!keysKnown
              ? "Vaelor could not read this agent's keys, so they are not listed here. Reload the page to try again."
              : "No keys yet. Mint one to let an external client authenticate."}</p>
          )
        )}

        <div className="cl-ep-mint">
          <Input
            id={`endpoint-key-label-${endpoint.id}`}
            label="Key name"
            maxLength={80}
            onChange={(event) => setLabels((current) => ({ ...current, [endpoint.id]: event.target.value }))}
            placeholder={isAgent ? "Example: Dashboard" : "Example: My desktop chat app"}
            value={labels[endpoint.id] ?? ""}
          />
          <Button
            disabled={busy !== ""}
            onClick={() => void mint(endpoint)}
            type="button"
            variant={isAgent ? "secondary" : "primary"}
          >
            {busy === `mint:${endpoint.id}`
              ? (isAgent ? "Issuing…" : "Minting…")
              : (isAgent ? "Issue key" : "Mint key")}
          </Button>
        </div>
      </ClusterCard>
    );
  };

  // The board's order: the LLM Server (and any other served endpoint) first,
  // the model library beside it, then the cluster balancer and the agents.
  const served = endpoints.filter((item) => item.id !== CLUSTER_SERVING_ENDPOINT_ID && item.target_kind !== AGENT_TARGET_KIND);
  const balancers = endpoints.filter((item) => item.id === CLUSTER_SERVING_ENDPOINT_ID);
  const agentCards = endpoints.filter((item) => item.target_kind === AGENT_TARGET_KIND);

  return (
    <section aria-label="Endpoints and API keys" className="cl-stack">
      {actionError && <Notice severity="danger">{actionError}</Notice>}
      {rotateOutcome && (
        <Notice severity={rotateOutcome.severity}>
          {rotateOutcome.text}{rotateOutcome.severity === "info" ? <> <a href="#/activity">Open Activity</a></> : null}
        </Notice>
      )}
      {rekeyNote && <Notice severity="info">{rekeyNote}</Notice>}
      {agentsError && (
        <Notice severity="warning">{`The deployed agents could not be read, so their endpoints and keys are not listed: ${agentsError}`}</Notice>
      )}

      <div className="cl-grid">
        {loading && !endpoints.length ? (
          <ClusterCard icon="server" title="Served endpoints">
            <LoadingLines label="Reading the served endpoints…" />
          </ClusterCard>
        ) : error && !endpoints.length ? (
          <ClusterCard icon="server" title="Served endpoints">
            <Notice severity="danger">{error}</Notice>
          </ClusterCard>
        ) : !endpoints.length ? (
          <ClusterCard icon="server" title="Served endpoints">
            <EmptyState title="No served endpoints" text="Nothing is exposing a model on this appliance yet." />
          </ClusterCard>
        ) : (
          served.map((item) => servedCard(item))
        )}
        {library}
        {balancers.map(clusterCard)}
        {agentCards.map((item, index) => servedCard(item, index === 0))}
      </div>

      {reveal && (
        <RevealKeyModal
          description={reveal.description}
          keyValue={reveal.key}
          onClose={() => setReveal(null)}
          title={reveal.title}
        />
      )}
      <ClusterConfirm
        busyLabel="Sending…"
        eyebrow="API keys"
        eyebrowTone="warning"
        busy={Boolean(revokeTarget && busy === `revoke:${revokeTarget.key.credential_id}`)}
        confirmLabel="Revoke key"
        description={revokeTarget
          ? revokeTarget.endpointId === LLM_SERVER_ENDPOINT_ID
            ? `${revokeTarget.lastKeyNote ? `${revokeTarget.lastKeyNote} ` : ""}${keyName(revokeTarget.key)} will stop being accepted by the LLM Server ${LLM_SERVER_REKEY_WINDOW}. Any client using it will need a new one.`
            : `${keyName(revokeTarget.key)} will stop working ${AGENT_REKEY_WINDOW}. Any client using it will need a new one.`
          : ""}
        error={dialogError}
        onCancel={() => { if (!busy) { setRevokeTarget(null); setDialogError(""); } }}
        onConfirm={() => void confirmRevoke()}
        open={Boolean(revokeTarget)}
        title="Revoke this API key?"
      />
      <ClusterConfirm
        busyLabel="Sending…"
        eyebrow="API keys"
        eyebrowTone="warning"
        busy={Boolean(rotateTarget && busy === `rotate:${rotateTarget.key.credential_id}`)}
        confirmLabel="Rotate key"
        error={dialogError}
        description={rotateTarget
          ? rotateTarget.endpoint.id === LLM_SERVER_ENDPOINT_ID
            ? `${keyName(rotateTarget.key)} gets a new value, shown once. The LLM Server stops accepting the old value ${LLM_SERVER_REKEY_WINDOW}, so every client using it needs the new one.`
            : rotateTarget.endpoint.target_kind === AGENT_TARGET_KIND
              ? `${keyName(rotateTarget.key)} gets a new value, shown once. The old value stops working ${AGENT_REKEY_WINDOW}, so every client using it needs the new one.`
              : `${keyName(rotateTarget.key)} gets a new value, shown once. The old value stops working at once, so every client using it needs the new one.`
          : ""}
        onCancel={() => { if (!busy) { setRotateTarget(null); setDialogError(""); } }}
        onConfirm={() => rotateTarget && void rotate(rotateTarget.endpoint, rotateTarget.key)}
        open={Boolean(rotateTarget)}
        title="Rotate this API key?"
      />
      <ClusterConfirm
        busyLabel="Sending…"
        eyebrow="API keys"
        eyebrowTone="warning"
        busy={busy === "rotate-cluster"}
        confirmLabel="Rotate the key"
        description="This re-keys the balancer and every worker gate. Serving stays authenticated throughout — the gates accept both keys before the balancer switches, then the old key is retired — though a few in-flight requests can drop across the brief gate and balancer restarts."
        error={dialogError}
        onCancel={() => { if (!busy) { setRotateClusterOpen(false); setDialogError(""); } }}
        onConfirm={() => void rotateCluster()}
        open={rotateClusterOpen}
        title="Rotate the internal cluster key?"
      />
    </section>
  );
}

/** A key's facts on one line: fingerprint, last four, created, and its use. */
function keyFacts(key: EndpointKey, use: string): ReactNode {
  return (
    <>
      {key.key_fingerprint ? <code>…{key.key_fingerprint.slice(-8)}</code> : "fingerprint not shown"}
      {" · "}
      {key.last4 ? <code>····{key.last4}</code> : "last 4 not shown"}
      {" · "}
      {key.created_at ? `created ${exactTime(key.created_at * 1000)}` : "created not recorded"}
      {` · ${use}`}
    </>
  );
}

/**
 * An LLM Server key's use, from the gate's log (`keyUseLabel`): a time reads
 * "last used 4 min ago"; the other words keep what they say.
 */
function llmKeyUse(label: string): string {
  if (/^(\d|just now)/.test(label)) return `last used ${label}`;
  if (label === "Never used") return "never used";
  if (label === "Not known") return "use not known";
  return label;
}

/** An agent key's use is recorded by record_use alone (the agent's gate keeps no usage log). */
function agentKeyUse(key: EndpointKey): string {
  return key.last_used_at ? `last used ${timeAgo(key.last_used_at * 1000)}` : "use not tracked";
}

