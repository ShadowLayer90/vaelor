import { useState } from "react";
import {
  ApiUsageLine, ExternalApiFigures, ExternalApiScope,
  type ApiTokenUsage, type InferenceGatewayStatus,
} from "./ApiUsageLine";
import { Icon } from "./Icon";
import { RecordConfirm, RecordFacts } from "./RecordKit";
import { ReadPill } from "./SettingsAccounts";
import { StatusPill } from "./StatusPill";
import { Button, Card, Input, Notice } from "./ui";
import { NOT_ANSWERING, type StatusTone } from "./ui/status";
import { credentialStatus, removalConsequence, type CredentialListing, type StoredCredential } from "../lib/credentialStatus";
import { CONNECTION_FORM_PATH, openAiChatConnectionForm } from "../lib/connectionFormHandoff";
import { timeAgo } from "../lib/format";
import "../styles/settings.css";

/*
 * Settings › Connections (VD-200, the Connections board): secure remote
 * access, the external API keys, and the AI connections this appliance holds.
 * Administration fetches everything in its one refresh and owns the busy and
 * message state; these components only present it.
 */

/** What `/security/transport` says about the household authority (VD-212). */
export interface TransportAuthority {
  /** `ready`, `not-set-up`, or `unreadable` (a file Vaelor could not read). */
  state: string; message: string;
  /** Colon-separated upper-case hex, as certificate dialogs and phones show it. */
  fingerprint_sha256: string; fingerprint_sha1: string;
  /** ISO 8601. */
  created_at: string | null;
  /**
   * What the console serves (`tls_authority_pki.classify`): `household`,
   * `legacy-self-signed`, `previous-household`, `custom`, `none`, or
   * `unreadable` when the file was there and could not be read.
   */
  kind: string;
}

export interface TransportStatus {
  secure: boolean; scheme: string; certificate_managed: boolean;
  certificate_fingerprint: string; vnc_secure: boolean; remote_ready: boolean;
  /** Absent from a controller older than VD-212: read as "not read", never as "none". */
  authority?: TransportAuthority;
  /** `expires_at` is ISO 8601. */
  leaf?: { expires_at: string; names: string[] } | null;
  /** The switch to the household certificate, as `tls_authority_status` states it; null when not reported. */
  migration?: { state: "none" | "waiting" | "blocked" | "done"; reason: string | null } | null;
  /** Each name the root could not cover, and why; null when not reported. */
  names_left_out?: Array<{ name: string; reason: string }> | null;
}

const CERTIFICATE_KIND: Record<string, string> = {
  household: "Issued by this Vaelor's household authority",
  "legacy-self-signed": "Earlier self-signed certificate",
  "previous-household": "Issued by this Vaelor's previous authority",
  custom: "Your own certificate (Vaelor leaves it alone)",
  none: "Not configured",
  unreadable: "Could not be read",
};

function certificateFact(transport: TransportStatus | null) {
  if (!transport) return "Not read";
  const kind = transport.authority?.kind;
  if (kind) return CERTIFICATE_KIND[kind] ?? "Could not be read";
  return transport.certificate_managed ? "Local appliance certificate" : "Not configured";
}

export interface AgentApiToken {
  id: string; label: string; prefix: string; enabled: boolean;
  created_at: number; last_used_at?: number | null;
  /** When the key was revoked; null for a key revoked before this was recorded. */
  revoked_at?: number | null;
  scopes: Array<"assistant" | "inference">;
  usage?: ApiTokenUsage | null; // cumulative gateway usage by token id; null when unreadable
}

/** The gateway status the card reads; its one definition is beside the usage figures. */
export type { InferenceGatewayStatus };

/*
 * ACC-097: the badge for each state the gateway reports. A cluster model that
 * is asleep (scaled to zero, wakes on the next request) or was unloaded by hand
 * is resting, not broken, so neither wears the warning colour; only a model
 * that should answer and does not, or a status Vaelor could not read, does.
 */
const GATEWAY_BADGES: Record<string, { label: string; tone: StatusTone }> = {
  reachable: { label: "Reachable", tone: "success" },
  asleep: { label: "Asleep", tone: "neutral" },
  unloaded: { label: "Unloaded", tone: "neutral" },
  "unloaded-unknown": { label: "Unloaded", tone: "neutral" },
  starting: { label: "Starting", tone: "info" },
  "not-answering": NOT_ANSWERING,
  "no-model": { label: "Not set up", tone: "neutral" },
  unknown: { label: "Status unknown", tone: "warning" },
};

function gatewayBadge(gateway: InferenceGatewayStatus | null) {
  if (!gateway) return { label: "Reading", tone: "neutral" as StatusTone };
  return GATEWAY_BADGES[gateway.state ?? ""] ?? (gateway.healthy
    ? GATEWAY_BADGES.reachable
    : { label: "Unavailable", tone: "warning" as StatusTone });
}

/** How a browser and Remote Desktop reach this machine, read from `/security/transport`. */
export function SecureRemoteAccess({ transport }: { transport: TransportStatus | null }) {
  return (
    <Card
      actions={<ReadPill notLabel="LAN only" ok={Boolean(transport?.remote_ready)} okLabel="HTTPS + WSS ready" read={Boolean(transport)} />}
      as="section"
      className="stg-transport"
      description="How browsers and Remote Desktop reach this machine"
      heading="Secure remote access"
    >
      <RecordFacts className="stg-transport__facts" facts={[
        { key: "console", label: "Console", value: transport ? (transport.secure ? "Encrypted HTTPS" : "Unencrypted HTTP") : "Not read" },
        { key: "desktop", label: "Remote Desktop", value: transport ? (transport.vnc_secure ? "Encrypted WSS" : "Trusted LAN only") : "Not read" },
        { key: "certificate", label: "Certificate", value: certificateFact(transport) },
        ...(transport?.leaf ? [{ key: "renews", label: "Valid until", value: new Date(transport.leaf.expires_at).toLocaleDateString() }] : []),
      ]} />
      {/* VD-212: a device trusts the household authority, not this one
          certificate, so its fingerprint and download live in Trust this
          Vaelor below; the leaf's own were dropped from here. */}
      {/* Only once the transport has actually been read: warning about an
          unencrypted link Vaelor has not looked at is a guess wearing an
          alarm's clothes, and it trains the reader to ignore the real one. */}
      {transport && !transport.remote_ready && <Notice severity="warning">Do not expose ports 34001 or 34002 directly to the internet until HTTPS and WSS are active. Use a trusted LAN or VPN.</Notice>}
    </Card>
  );
}

/** One stored credential as a row: what it is, what uses it, its last recorded fact, and what you can do. */
function CredentialRow({ busy, item, onRemove, onTest }: {
  busy: string;
  item: StoredCredential;
  onRemove: (item: StoredCredential) => void;
  onTest: (item: StoredCredential) => Promise<void>;
}) {
  const status = credentialStatus(item);
  const users = item.used_by ?? [];
  const rowBusy = busy === `credential-${item.id}`;
  return (
    <div className="stg-row stg-row--credential">
      <div className="stg-row__text">
        <strong>{item.label}</strong>
        <small>{item.kind_label ?? item.provider}{item.sends_off_machine ? " · prompts and files leave this machine" : ""} · fingerprint …{item.fingerprint.slice(-6)}{status.detail ? ` · ${status.detail}` : ""}</small>
        {item.manage_note && <small className="stg-row__reason">{item.manage_note}</small>}
      </div>
      <div className="stg-row__controls record-card-button">
        <span className="stg-tag">{users.length ? users.join(" · ") : "Not in use"}</span>
        <StatusPill label={status.label} tone={status.tone} />
        {/* B3 / ACC-107: every kind with a real test can be tested from here;
            S1: only a secret nothing uses is offered for removal, behind a
            confirmation. One Vaelor manages is said so instead. */}
        {item.testable && <Button aria-label={`Test ${item.label}`} disabled={rowBusy} onClick={() => void onTest(item)}>Test</Button>}
        {item.can_disconnect
          ? <Button aria-label={`Remove ${item.label}`} className="record-ghost" disabled={rowBusy} onClick={() => onRemove(item)} variant="quiet">Remove</Button>
          : item.managed_by ? <span className="ui-muted ui-small">Managed by Vaelor</span> : null}
      </div>
    </div>
  );
}

/**
 * The AI connections card: every model server AI Chat can use and what uses
 * it, then the other secrets this appliance holds (cluster sign-ins, app
 * secrets). Only metadata is shown; a secret cannot be read back.
 */
export function EncryptedConnectionsPanel({ listing, busy, onTest, onRemove }: {
  listing: CredentialListing | null;
  busy: string;
  onTest: (item: StoredCredential) => Promise<void>;
  onRemove: (item: StoredCredential) => Promise<void>;
}) {
  const [removing, setRemoving] = useState<StoredCredential | null>(null);
  const credentials = listing?.credentials ?? [];
  const ai = credentials.filter((item) => item.ai_connection === true);
  const other = credentials.filter((item) => item.ai_connection !== true);
  const inbound = listing?.endpoint_keys;
  const inboundTotal = inbound ? inbound.active + inbound.revoked : 0;
  const removingBusy = Boolean(removing && busy === `credential-${removing.id}`);
  return (
    <Card
      actions={<Button onClick={openAiChatConnectionForm} variant="primary">Add a connection</Button>}
      as="section"
      className="stg-connections"
      description="Every model server AI Chat can use, and what uses it"
      flush
      heading="AI connections"
    >
      {listing === null && <p className="stg-empty">Not read yet.</p>}
      {listing !== null && ai.length === 0 && (
        // VD-200: this pointed at "Workloads → Improve the assistant". The page
        // is Apps and AI, and a connection added there is AI Chat's (VD-201).
        <p className="stg-empty">No AI connections yet. Connect a hosted provider or a model server on your network for AI Chat: Add a connection opens the form in {CONNECTION_FORM_PATH}.</p>
      )}
      {ai.map((item) => <CredentialRow busy={busy} item={item} key={item.id} onRemove={setRemoving} onTest={onTest} />)}
      {other.length > 0 && (
        <>
          <h3 className="stg-subhead">Other stored secrets</h3>
          {other.map((item) => <CredentialRow busy={busy} item={item} key={item.id} onRemove={setRemoving} onTest={onTest} />)}
        </>
      )}
      {inboundTotal > 0 && inbound && (
        <p className="stg-empty">
          {`${inbound.active} key${inbound.active === 1 ? "" : "s"} that apps and agents use to reach this appliance${inbound.revoked ? ` (and ${inbound.revoked} revoked)` : ""} ${inboundTotal === 1 ? "is" : "are"} not listed here. Rotate or revoke them under Cluster → Deployments → Models.`}
        </p>
      )}
      {removing && (
        <RecordConfirm
          busy={removingBusy}
          confirmLabel="Remove and delete secret"
          eyebrow="Can't be undone"
          onCancel={() => { if (!busy) setRemoving(null); }}
          onConfirm={() => void onRemove(removing).then(() => setRemoving(null))}
          title={`Remove ${removing.label}?`}
        >
          <p>{`Vaelor deletes the stored secret for ${removing.label} from this appliance; it cannot be read back or restored. ${removalConsequence(removing)}`}</p>
        </RecordConfirm>
      )}
    </Card>
  );
}

function revokedWhen(token: AgentApiToken) {
  return token.revoked_at ? `revoked ${new Date(token.revoked_at * 1000).toLocaleString()}` : "revoked · date not recorded";
}

function keyHistory(token: AgentApiToken) {
  const created = `Created ${timeAgo(token.created_at * 1000)}`;
  return `${created} · ${token.last_used_at ? `last used ${timeAgo(token.last_used_at * 1000)}` : "never used"}`;
}

/**
 * External API access: the keys apps use for this appliance's inference API,
 * the gateway they reach, and what reached it in the last day.
 */
export function ExternalApiPanel({
  tokens, gateway, busy, label, onLabel, onCreate, newToken, onDismissToken, onRevoke, onRemoveRecord,
}: {
  tokens: AgentApiToken[];
  gateway: InferenceGatewayStatus | null;
  busy: string;
  label: string;
  onLabel: (value: string) => void;
  onCreate: () => void;
  newToken: string;
  onDismissToken: () => void;
  onRevoke: (token: AgentApiToken) => void;
  onRemoveRecord: (token: AgentApiToken) => Promise<void>;
}) {
  const [removing, setRemoving] = useState<AgentApiToken | null>(null);
  const active = tokens.filter((token) => token.enabled);
  const revoked = tokens.filter((token) => !token.enabled);
  const badge = gatewayBadge(gateway);
  return (
    <Card
      actions={<StatusPill label={active.length ? `${active.length} active key${active.length === 1 ? "" : "s"}` : "No active keys"} tone={active.length ? "success" : "neutral"} />}
      as="section"
      className="stg-api"
      description="Revocable keys for the cluster inference API on this appliance"
      flush
      heading="External API access"
    >
      <div className="stg-gateway">
        <div className="stg-row__text">
          <span className="record-eyebrow">Inference gateway</span>
          <strong>{gateway?.target?.label ?? (gateway?.state === "unknown" ? "Cluster model not read" : "No cluster model")}</strong>
          <small>{gateway?.message ?? "Reading the gateway status."}</small>
        </div>
        <StatusPill label={badge.label} tone={badge.tone} />
      </div>
      {/* External requests from both doors, and the model's own tokens
          (ACC-046/048); unread figures show as unavailable, never 0. */}
      <details className="stg-usage record-disclosure">
        <summary>Requests in the last 24 hours</summary>
        <ExternalApiScope status={gateway} />
        <ExternalApiFigures status={gateway} />
      </details>
      <form className="stg-api__create" onSubmit={(event) => { event.preventDefault(); if (label.trim() && busy !== "api-token") onCreate(); }}>
        <Input id="api-token-label" label="Connection name" maxLength={100} onChange={(event) => onLabel(event.target.value)} placeholder="e.g. Laptop editor" value={label} />
        <span className="record-card-button">
          <Button disabled={busy === "api-token"} disabledReason={!label.trim() ? "Name this key so it can be recognised later." : undefined} type="submit" variant="primary">Create one-time API key</Button>
        </span>
      </form>
      {newToken && (
        <div className="stg-token" role="status">
          <Icon name="key" size={18} />
          <div><strong>Copy this key now</strong><code className="record-mono">{newToken}</code><small>{`Inference: ${window.location.origin}/inference/v1`}</small></div>
          <span className="record-card-button"><Button onClick={() => void navigator.clipboard.writeText(newToken)}>Copy key</Button><Button className="record-ghost" onClick={onDismissToken} variant="quiet">I saved it</Button></span>
        </div>
      )}
      {active.map((item) => (
        <div className="stg-row stg-row--key" key={item.id}>
          <div className="stg-row__text">
            <strong>{item.label}</strong>
            <small>{keyHistory(item)} · <span className="record-mono">{item.prefix}…</span></small>
            <ApiUsageLine token={item} />
          </div>
          <div className="stg-row__controls record-card-button">
            <StatusPill label="Active" tone="success" />
            <Button aria-label={`Revoke ${item.label}`} className="record-danger-outline" disabled={busy === `api-${item.id}`} onClick={() => onRevoke(item)}>Revoke</Button>
          </div>
        </div>
      ))}
      {/* ACC-115: a revoked key says when it was revoked, and its record can
          be cleared - the audit log keeps its history. */}
      {revoked.map((item) => (
        <div className="stg-row stg-row--key" key={item.id}>
          <div className="stg-row__text">
            <strong>{item.label}</strong>
            <small>Inference · <span className="record-mono">{item.prefix}…</span> · {revokedWhen(item)}</small>
          </div>
          <div className="stg-row__controls record-card-button">
            <StatusPill label="Revoked" tone="neutral" />
            <Button aria-label={`Remove the record of ${item.label}`} className="record-ghost" disabled={busy === `api-${item.id}`} onClick={() => setRemoving(item)} variant="quiet">Remove record</Button>
          </div>
        </div>
      ))}
      {removing && (
        <RecordConfirm
          busy={busy === `api-${removing.id}`}
          confirmLabel="Remove record"
          eyebrow="Already revoked"
          eyebrowTone="plain"
          onCancel={() => { if (!busy) setRemoving(null); }}
          onConfirm={() => void onRemoveRecord(removing).then(() => setRemoving(null))}
          title="Remove revoked key record?"
        >
          <p>{`${removing.label} (${removing.prefix}…) is already revoked and cannot be used. Its record leaves this list; the audit log keeps when it was created, revoked and removed.`}</p>
        </RecordConfirm>
      )}
    </Card>
  );
}
