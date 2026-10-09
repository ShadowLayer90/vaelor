import { useCallback, useState, type ReactNode } from "react";
import { ApiError } from "../lib/api";
import { exactTime, timeAgo } from "../lib/format";
import {
  deleteMcpServer,
  parseApprovedTools,
  probeMcpServerHealth,
  registerMcpServer,
  summarizeHealth,
  updateMcpServer,
  useMcpCatalog,
  type McpServer,
} from "../lib/mcpCatalog";
import type { Session } from "../types";
import { OptionCard, Tag } from "./ClusterPrimitives";
import { ClusterDialog } from "./ClusterDialog";
import { ClusterConfirm } from "./ClusterConfirm";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button, EmptyState, Input, LoadingLines, Notice, Select, Textarea } from "./ui";
import "../styles/cluster-agents.css";

/**
 * MCP catalog (F6a): the "Agents & tools" surface that lists the Model Context
 * Protocol servers an agent may call, and lets an administrator register, enable,
 * re-probe and remove them.
 *
 * Each server shows its transport kind and endpoint, whether it is enabled, its
 * last health probe, and the exact tools it exposes to agents. A server never
 * shows a secret: `credential_id` is an id only. Mint has no analogue here - a
 * credential is created elsewhere and only referenced by id.
 *
 * Reading the catalog is an operator capability; every change is an
 * administrator one, so a non-administrator sees the catalog in full but the
 * register action and per-row controls are withheld.
 *
 * VD-200 (the ClusterAgentsMcp and ClusterDialogsTools boards): the servers are
 * cards in a three-up grid, and the dialogs are the Cluster page's own.
 */

/** The register form's own field state, kept as strings until it is submitted. */
interface RegisterForm {
  name: string;
  kind: string;
  endpoint: string;
  credentialId: string;
  enabled: boolean;
  approvedTools: string;
}

const EMPTY_FORM: RegisterForm = {
  name: "",
  kind: "url",
  endpoint: "",
  credentialId: "",
  enabled: true,
  approvedTools: "",
};

const ADMIN_ONLY_NOTE =
  "You can view the catalog. Registering, enabling, re-probing or removing an MCP server needs an administrator account.";
const UNCONFIGURED_NOTE =
  "No MCP catalog is configured on this appliance yet. Once it is, registered servers and the tools they expose to agents appear here.";

/** A server's last health check, relative, with the exact time on hover. */
function CheckedAt({ server }: { server: McpServer }) {
  if (server.checked_at === null) return <span className="cl-tool__muted">Never checked</span>;
  const at = server.checked_at * 1000;
  return <time dateTime={new Date(at).toISOString()} title={exactTime(at)}>{`Checked ${timeAgo(at)}`}</time>;
}

/** The colour a probe's own sentence takes: red for a failure, amber for a degraded one. */
function detailClass(tone: string): string | undefined {
  if (tone === "danger") return "cl-bad-text";
  if (tone === "warning") return "cl-warn-text";
  return undefined;
}

/** A list of short names as mono tags, or the sentence that says there are none. */
export function ToolTags({ empty, items }: { empty: string; items: string[] }) {
  if (!items.length) return <p className="cl-tool__muted">{empty}</p>;
  return (
    <ul className="cl-tags cl-tool__tags">
      {items.map((item) => <li key={item}><Tag mono>{item}</Tag></li>)}
    </ul>
  );
}

export function McpCatalogPanel({ session }: { session: Session }) {
  const { catalog, loading, error, reload } = useMcpCatalog({ poll: true });
  const isAdmin = session.user.role === "administrator";
  const [busy, setBusy] = useState("");
  const [actionError, setActionError] = useState("");
  const [registerOpen, setRegisterOpen] = useState(false);
  const [form, setForm] = useState<RegisterForm>(EMPTY_FORM);
  const [removeTarget, setRemoveTarget] = useState<McpServer | null>(null);

  const fail = useCallback((cause: unknown, fallback: string) => {
    if (!(cause instanceof ApiError)) console.error(fallback, cause);
    setActionError(cause instanceof ApiError ? cause.message : fallback);
  }, []);

  const closeRegister = useCallback(() => {
    setRegisterOpen(false);
    setForm(EMPTY_FORM);
    setActionError("");
  }, []);

  const submitRegister = useCallback(async () => {
    setBusy("register");
    setActionError("");
    try {
      await registerMcpServer(
        {
          name: form.name.trim(),
          kind: form.kind.trim(),
          endpoint: form.endpoint.trim(),
          credential_id: form.credentialId.trim(),
          enabled: form.enabled,
          approved_tools: parseApprovedTools(form.approvedTools),
        },
        session.csrf_token,
      );
      closeRegister();
      reload();
    } catch (cause) {
      fail(cause, "The MCP server could not be registered.");
    } finally {
      setBusy("");
    }
  }, [closeRegister, fail, form, reload, session.csrf_token]);

  const toggle = useCallback(async (server: McpServer) => {
    setBusy(`toggle:${server.id}`);
    setActionError("");
    try {
      await updateMcpServer(server.id, { enabled: !server.enabled }, session.csrf_token);
      reload();
    } catch (cause) {
      fail(cause, "That change could not be applied.");
    } finally {
      setBusy("");
    }
  }, [fail, reload, session.csrf_token]);

  const probe = useCallback(async (server: McpServer) => {
    setBusy(`probe:${server.id}`);
    setActionError("");
    try {
      await probeMcpServerHealth(server.id, session.csrf_token);
      reload();
    } catch (cause) {
      fail(cause, "The server could not be re-probed.");
    } finally {
      setBusy("");
    }
  }, [fail, reload, session.csrf_token]);

  const confirmRemove = useCallback(async () => {
    if (!removeTarget) return;
    setBusy(`remove:${removeTarget.id}`);
    setActionError("");
    try {
      await deleteMcpServer(removeTarget.id, session.csrf_token);
      setRemoveTarget(null);
      reload();
    } catch (cause) {
      fail(cause, "The MCP server could not be removed.");
    } finally {
      setBusy("");
    }
  }, [fail, reload, removeTarget, session.csrf_token]);

  const registerInvalid = !form.name.trim() || !form.endpoint.trim();
  const servers = catalog.servers;

  let body: ReactNode;
  if (loading && !servers.length) {
    body = (
      <div className="ui-card cl-tool__loading">
        <p aria-hidden="true">Reading the MCP catalog...</p>
        <LoadingLines label="Reading the MCP catalog" />
      </div>
    );
  } else if (error && !servers.length) {
    body = <Notice severity="danger">{error}</Notice>;
  } else if (!catalog.configured && !servers.length) {
    body = <Notice severity="info">{catalog.note || UNCONFIGURED_NOTE}</Notice>;
  } else if (!servers.length) {
    body = (
      <div className="ui-card cl-tool__empty">
        <EmptyState
          icon={<Icon name="server" />}
          text={isAdmin
            ? "Register a server to let agents call its tools."
            : "An administrator can register a server to let agents call its tools."}
          title="No MCP servers registered"
        />
      </div>
    );
  } else {
    body = (
      <>
        {catalog.note && <Notice severity="info">{catalog.note}</Notice>}
        <div className="cl-grid cl-grid--three cl-tool__grid">
          {servers.map((server) => {
            const health = summarizeHealth(server.health);
            const name = server.name || "Unnamed server";
            return (
              <article aria-label={server.name || "MCP server"} className="ui-card cl-tool" key={server.id}>
                <div className="cl-tool__top">
                  <div className="cl-tool__name">
                    <strong>{name}</strong>
                    <small>{server.kind || "MCP server"}</small>
                  </div>
                  <div className="cl-tool__pills">
                    <StatusPill label={server.enabled ? "Enabled" : "Disabled"} tone={server.enabled ? "success" : "neutral"} />
                    <StatusPill label={health.label} tone={health.tone} />
                  </div>
                </div>

                <dl className="cl-tool__facts">
                  <div>
                    <dt>Endpoint</dt>
                    <dd>{server.endpoint ? <code>{server.endpoint}</code> : <span className="cl-tool__muted">Not set</span>}</dd>
                  </div>
                  <div>
                    <dt>Credential</dt>
                    <dd>{server.credential_id ? <code>{server.credential_id}</code> : <span className="cl-tool__muted">None</span>}</dd>
                  </div>
                  <div>
                    <dt>Last check</dt>
                    <dd>
                      <CheckedAt server={server} />
                      {server.health_refreshing && <span className="cl-tool__muted"> · checking again now</span>}
                    </dd>
                  </div>
                  <div>
                    <dt>Tools found</dt>
                    <dd>{server.tools_count === null ? <span className="cl-tool__muted">Not reported</span> : server.tools_count}</dd>
                  </div>
                  {health.detail && (
                    <div>
                      <dt>Last probe</dt>
                      <dd className={detailClass(health.tone)}>{health.detail}</dd>
                    </div>
                  )}
                </dl>

                <div className="cl-tool__section">
                  <div className="cl-tool__section-head">
                    <h4>Approved tools</h4>
                    <span>{server.approved_tools.length}</span>
                  </div>
                  <ToolTags empty="No tools approved - this server exposes nothing to agents yet." items={server.approved_tools} />
                </div>

                {isAdmin && (
                  <div className="cl-actions cl-tool__actions">
                    <Button disabled={busy !== ""} onClick={() => void toggle(server)}>
                      {busy === `toggle:${server.id}`
                        ? (server.enabled ? "Disabling..." : "Enabling...")
                        : (server.enabled ? "Disable" : "Enable")}
                    </Button>
                    <Button disabled={busy !== ""} onClick={() => void probe(server)} variant="quiet">
                      {busy === `probe:${server.id}` ? "Checking..." : "Re-probe health"}
                    </Button>
                    <Button
                      aria-label={`Remove ${server.name || "server"}`}
                      className="cl-danger-outline"
                      disabled={busy !== ""}
                      onClick={() => { setActionError(""); setRemoveTarget(server); }}
                    >
                      Remove
                    </Button>
                  </div>
                )}
              </article>
            );
          })}
        </div>
      </>
    );
  }

  return (
    <section aria-labelledby="mcp-catalog-title" className="cl-stack cl-tools">
      <div className="cl-tools__head">
        <div className="cl-tools__head-text">
          <h3 id="mcp-catalog-title">MCP servers</h3>
          <p>Model Context Protocol servers an agent may call. Each exposes only the tools you approve.</p>
        </div>
        {isAdmin && (
          <div className="cl-actions">
            <Button
              disabled={busy !== ""}
              onClick={() => { setActionError(""); setRegisterOpen(true); }}
              variant="primary"
            >
              <Icon name="add" />Register MCP server
            </Button>
          </div>
        )}
      </div>

      {/* VD-189: while a dialog is open its own refusal is shown in it; the page beneath is inert. */}
      {actionError && !registerOpen && !removeTarget && <Notice severity="danger">{actionError}</Notice>}
      {!isAdmin && <Notice severity="info">{ADMIN_ONLY_NOTE}</Notice>}
      {error && servers.length > 0 && (
        <Notice severity="warning">{`The list could not be read again, so it may be out of date: ${error}`}</Notice>
      )}

      {body}

      {registerOpen && (
        <ClusterDialog
          error={actionError}
          eyebrow="MCP catalog"
          footer={<>
            <Button disabled={busy !== ""} onClick={closeRegister}>Cancel</Button>
            <Button
              disabled={busy !== ""}
              disabledReason={registerInvalid ? "Enter at least a name and an endpoint to register the server." : undefined}
              form="mcp-register-form"
              type="submit"
              variant="primary"
            >
              {busy === "register" ? "Registering..." : "Register server"}
            </Button>
          </>}
          onClose={() => { if (busy === "") closeRegister(); }}
          title="Register MCP server"
          titleId="mcp-register-title"
        >
          <p>Point Vaelor at an MCP server and choose which of its tools agents may call.</p>
          <form
            className="cl-tool-form"
            id="mcp-register-form"
            onSubmit={(event) => {
              event.preventDefault();
              if (!registerInvalid) void submitRegister();
            }}
          >
            <Input
              id="mcp-name"
              label="Name"
              maxLength={80}
              onChange={(event) => setForm((current) => ({ ...current, name: event.target.value }))}
              placeholder="Example: filesystem-tools"
              value={form.name}
            />
            <Select
              hint="url = an external MCP server addressed by URL; builtin = a server Vaelor provides."
              id="mcp-kind"
              label="Kind"
              onChange={(event) => setForm((current) => ({ ...current, kind: event.target.value }))}
              value={form.kind}
            >
              <option value="url">URL</option>
              <option value="builtin">Built-in</option>
            </Select>
            <Input
              hint="https:// with a public address, or http:// with a loopback address (localhost or 127.0.0.1) for a server on this host. An https:// address on your private network is refused."
              id="mcp-endpoint"
              label="Endpoint"
              maxLength={300}
              onChange={(event) => setForm((current) => ({ ...current, endpoint: event.target.value }))}
              placeholder="https://example.com/mcp"
              value={form.endpoint}
            />
            <Input
              hint="Optional. The id of a stored credential - never the secret itself."
              id="mcp-credential"
              label="Credential ID"
              maxLength={120}
              onChange={(event) => setForm((current) => ({ ...current, credentialId: event.target.value }))}
              placeholder="Example: cred_filesystem"
              value={form.credentialId}
            />
            <Textarea
              hint="Separate tool names with commas or spaces. Only these tools are exposed to agents."
              id="mcp-tools"
              label="Approved tools"
              maxLength={2000}
              onChange={(event) => setForm((current) => ({ ...current, approvedTools: event.target.value }))}
              placeholder="read_file, list_dir, search"
              rows={3}
              value={form.approvedTools}
            />
            <OptionCard
              checked={form.enabled}
              onChange={(event) => setForm((current) => ({ ...current, enabled: event.target.checked }))}
              title="Enable this server now"
              type="checkbox"
            />
          </form>
        </ClusterDialog>
      )}

      <ClusterConfirm
        busy={Boolean(removeTarget && busy === `remove:${removeTarget.id}`)}
        confirmLabel="Remove server"
        description={removeTarget
          ? `${removeTarget.name || "This server"} will be removed from the catalog and agents will no longer be able to call its tools. Any stored credential it referenced is left untouched.`
          : ""}
        error={actionError}
        eyebrow="Remove"
        onCancel={() => { if (busy === "") { setRemoveTarget(null); setActionError(""); } }}
        onConfirm={() => void confirmRemove()}
        open={Boolean(removeTarget)}
        title="Remove this MCP server?"
      />
    </section>
  );
}
