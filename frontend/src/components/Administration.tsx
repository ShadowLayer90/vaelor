import { useCallback, useEffect, useState } from "react";
import { apiRequest, downloadApiRequest } from "../lib/api";
import type { Role, Session } from "../types";
import { Icon } from "./Icon";
import { PaginatedItems } from "./PaginatedItems";
import { StatusPill } from "./StatusPill";
import { ConfirmDialog } from "./ConfirmDialog";
import { BackupSettings } from "./BackupSettings";
import { Button, Checkbox, Input, Notice, Select, UnavailableValue } from "./ui";
import { destinations } from "../lib/destinations";
import { readDraft, writeDraft } from "../lib/draftStorage";
import { timeAgo } from "../lib/format";

/**
 * A verdict about something that has to be fetched first.
 *
 * Every pill on this page read `value?.field ? good : bad` off a `null`
 * initial state, so before the request answered each one announced a settled
 * conclusion about a fact nobody had looked at yet: "LAN only", "Unavailable",
 * and — the reassuring version of the same mistake — "Ready". Two of those are
 * false alarms and one is a false all-clear.
 *
 * Three states. Not knowing is its own answer and is shown as itself.
 */
function ReadStatusPill({ read, ok, okLabel, notLabel }: {
  read: boolean;
  ok: boolean;
  okLabel: string;
  notLabel: string;
}) {
  if (!read) return <StatusPill label="Reading" status="neutral" />;
  return <StatusPill label={ok ? okLabel : notLabel} status={ok ? "healthy" : "degraded"} />;
}

interface ManagedUser {
  username: string;
  role: Role;
  enabled: boolean;
  created_at: number;
  active_sessions: number;
}

interface ManagedSession {
  id: string;
  username: string;
  created_at: number;
  expires_at: number;
  last_seen_at: number;
  remote_addr: string;
  user_agent: string;
  current: boolean;
}

interface Credential {
  id: string;
  provider: string;
  label: string;
  fingerprint: string;
  last_test_status?: string;
  last_tested_at?: number | null;
  active_for?: string[];
}
interface AgentApiToken {
  id: string; label: string; prefix: string; enabled: boolean;
  created_at: number; last_used_at?: number | null;
  scopes: Array<"assistant" | "inference">;
}
interface InferenceGatewayStatus {
  healthy: boolean;
  message: string;
  target: { label: string; base_url: string } | null;
  metrics_24h: {
    requests: number; failures: number; streaming_requests: number;
    average_latency_ms: number; prompt_tokens: number; completion_tokens: number;
  };
  endpoints: { models: string; chat_completions: string; openapi: string };
}
interface TransportStatus {
  secure: boolean; scheme: string; certificate_managed: boolean;
  certificate_fingerprint: string; vnc_secure: boolean; remote_ready: boolean;
}
interface RecoveryStatus {
  confirmation: string;
  erases: string[];
  retains: string[];
}
interface PortableStateStatus {
  staged: boolean;
  confirmation: string;
  plan: {
    id: string;
    source_version?: string;
    created_at: number;
    expires_at: number;
    approved_at?: number;
  } | null;
  scope: { includes: string[]; excludes: string[] };
  last_result?: {
    ok: boolean;
    completed_at: number;
    imported?: number;
    error?: string;
  } | null;
}

const FINAL_ADMINISTRATOR_REASON =
  "This is the last enabled administrator. Add another enabled administrator before disabling, changing the role, or deleting this account.";
const CURRENT_ACCOUNT_REASON = "You cannot disable or delete the account you are using.";

export type AdministrationSection = "users" | "sessions" | "secrets" | "recovery";

export function administrationSectionFromHash(hash: string): AdministrationSection {
  const section = hash.match(/^#\/admin\/(accounts|users|sessions|connections|secrets|recovery)(?:[/?].*)?$/)?.[1];
  if (section === "sessions" || section === "recovery") return section;
  if (section === "connections" || section === "secrets") return "secrets";
  return "users";
}

function administrationHashForSection(section: AdministrationSection) {
  const slug = section === "users" ? "accounts" : section === "secrets" ? "connections" : section;
  return `#/admin/${slug}`;
}

export function Administration({ session, onBack }: { session: Session; onBack?: () => void }) {
  const [tab, setTab] = useState<AdministrationSection>(() => administrationSectionFromHash(window.location.hash));
  const [users, setUsers] = useState<ManagedUser[]>([]);
  const [sessions, setSessions] = useState<ManagedSession[]>([]);
  const [credentials, setCredentials] = useState<Credential[]>([]);
  const [apiTokens, setApiTokens] = useState<AgentApiToken[]>([]);
  const [apiTokenLabel, setApiTokenLabel] = useState("");
  const [apiTokenScopes, setApiTokenScopes] = useState<Array<"assistant" | "inference">>(["assistant"]);
  const [newApiToken, setNewApiToken] = useState("");
  const [newApiTokenScopes, setNewApiTokenScopes] = useState<Array<"assistant" | "inference">>([]);
  const [inferenceGateway, setInferenceGateway] = useState<InferenceGatewayStatus | null>(null);
  const [transport, setTransport] = useState<TransportStatus | null>(null);
  const [totpEnabled, setTotpEnabled] = useState(false);
  const [totpSecret, setTotpSecret] = useState("");
  const [totpCode, setTotpCode] = useState("");
  /*
   * #150: leaving Settings unmounts this component and discarded a
   * part-completed Add-an-account form without warning, while switching
   * between Settings sub-tabs preserved it — inconsistent as well as lossy.
   * Username and role survive the round trip, keyed to the signed-in account
   * and storage-failure-safe (see draftStorage); the passphrase deliberately
   * does not survive — a credential is never written to browser storage, and
   * re-typing it is the correct cost.
   */
  const [username, setUsername] = useState(() =>
    readDraft("vaelor.admin.new-account.username", session.user.username),
  );
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<Role>(() => {
    const saved = readDraft("vaelor.admin.new-account.role", session.user.username);
    return saved === "operator" || saved === "administrator" ? saved : "viewer";
  });
  useEffect(() => {
    writeDraft("vaelor.admin.new-account.username", session.user.username, username);
  }, [session.user.username, username]);
  useEffect(() => {
    writeDraft("vaelor.admin.new-account.role", session.user.username, role);
  }, [session.user.username, role]);
  const [busy, setBusy] = useState("");
  const [message, setMessageText] = useState("");
  // #149: every outcome — including failed provider connections — rendered in
  // the informational blue Notice, visually identical to the advice notice
  // above it. The severity travels with the message so a failure cannot be
  // dressed as information.
  const [messageFailed, setMessageFailed] = useState(false);
  const setMessage = (text: string) => { setMessageFailed(false); setMessageText(text); };
  const reportFailure = (error: unknown, fallback: string) => {
    setMessageFailed(true);
    setMessageText(error instanceof Error && error.message ? error.message : fallback);
  };
  const [recovery, setRecovery] = useState<RecoveryStatus | null>(null);
  const [portableState, setPortableState] = useState<PortableStateStatus | null>(null);
  const [resetConfirmation, setResetConfirmation] = useState("");
  const [exportPassphrase, setExportPassphrase] = useState("");
  const [exportPassphraseAgain, setExportPassphraseAgain] = useState("");
  const [importArchive, setImportArchive] = useState<File | null>(null);
  const [importPassphrase, setImportPassphrase] = useState("");
  const [importConfirmation, setImportConfirmation] = useState("");
  const [deleteUser, setDeleteUser] = useState<ManagedUser | null>(null);
  const [revokeToken, setRevokeToken] = useState<AgentApiToken | null>(null);

  const refresh = useCallback(async () => {
    const [nextUsers, nextSessions, secretData, nextApiTokens, gateway, mfa, nextTransport, nextRecovery, nextPortable] =
      await Promise.allSettled([
        apiRequest<ManagedUser[]>("/admin/users"),
        apiRequest<ManagedSession[]>("/admin/sessions"),
        apiRequest<{ credentials: Credential[] }>("/credentials"),
        apiRequest<AgentApiToken[]>("/admin/agent-api-tokens"),
        apiRequest<InferenceGatewayStatus>("/admin/inference-gateway"),
        apiRequest<{ enabled: boolean }>("/auth/totp"),
        apiRequest<TransportStatus>("/security/transport"),
        apiRequest<RecoveryStatus>("/admin/recovery/factory-reset"),
        apiRequest<PortableStateStatus>("/admin/portable-state"),
      ]);
    if (nextUsers.status === "fulfilled") setUsers(nextUsers.value);
    if (nextSessions.status === "fulfilled") setSessions(nextSessions.value);
    if (secretData.status === "fulfilled") setCredentials(secretData.value.credentials);
    if (nextApiTokens.status === "fulfilled") setApiTokens(nextApiTokens.value);
    if (gateway.status === "fulfilled") setInferenceGateway(gateway.value);
    if (mfa.status === "fulfilled") setTotpEnabled(mfa.value.enabled);
    if (nextTransport.status === "fulfilled") setTransport(nextTransport.value);
    if (nextRecovery.status === "fulfilled") setRecovery(nextRecovery.value);
    if (nextPortable.status === "fulfilled") setPortableState(nextPortable.value);
    const unavailable = [
      nextUsers, nextSessions, secretData, nextApiTokens, gateway, mfa, nextTransport, nextRecovery, nextPortable,
    ].filter((result) => result.status === "rejected").length;
    if (unavailable > 0) {
      // A failed refresh is a failure, not information (#149 review).
      setMessageFailed(true);
      setMessageText(
        unavailable === 9
          ? "Administration data is temporarily unavailable."
          : `${unavailable} administration panel${unavailable === 1 ? "" : "s"} could not refresh. The available panels are still shown.`,
      );
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    const restoreSection = () => setTab(administrationSectionFromHash(window.location.hash));
    window.addEventListener("hashchange", restoreSection);
    window.addEventListener("popstate", restoreSection);
    return () => {
      window.removeEventListener("hashchange", restoreSection);
      window.removeEventListener("popstate", restoreSection);
    };
  }, []);

  const selectTab = (
    next: AdministrationSection,
    { replaceHistory = false }: { replaceHistory?: boolean } = {},
  ) => {
    setTab(next);
    setMessage("");
    // Arrow-key roving replaces the entry rather than pushing one: walking
    // Accounts → Recovery used to stuff three history entries, so Back
    // revisited every tab before leaving the page.
    if (replaceHistory) {
      window.history.replaceState(null, "", administrationHashForSection(next));
    } else {
      window.history.pushState(null, "", administrationHashForSection(next));
    }
  };

  const createUser = async () => {
    setBusy("create");
    setMessage("");
    try {
      await apiRequest("/admin/users", {
        method: "POST", body: JSON.stringify({ username: normalizedUsername, password, role }),
      }, session.csrf_token);
      const renamed = normalizedUsername !== username.trim();
      setUsername("");
      setPassword("");
      setRole("viewer");
      // Normalising in silence is what made the field feel like it was fighting
      // the typist. If the stored name differs from what was typed, say so.
      setMessage(renamed
        ? `Account created as ${normalizedUsername}. Usernames are stored in lowercase, so that is the name to sign in with.`
        : "Account created. The user can sign in immediately.");
      await refresh();
    } catch (error) {
      reportFailure(error,"Account could not be created.");
    } finally {
      setBusy("");
    }
  };

  const updateUser = async (name: string, patch: { role?: Role; enabled?: boolean }) => {
    setBusy(`user-${name}`);
    setMessage("");
    try {
      await apiRequest(`/admin/users/${encodeURIComponent(name)}`, {
        method: "PATCH", body: JSON.stringify(patch),
      }, session.csrf_token);
      setMessage(`${name} updated.`);
      await refresh();
    } catch (error) {
      reportFailure(error,"Account could not be updated.");
    } finally {
      setBusy("");
    }
  };

  const revokeSession = async (id: string) => {
    setBusy(`session-${id}`);
    setMessage("");
    try {
      await apiRequest(`/admin/sessions/${id}`, { method: "DELETE" }, session.csrf_token);
      setMessage("Remote session revoked.");
      await refresh();
    } catch (error) {
      reportFailure(error,"Session could not be revoked.");
    } finally {
      setBusy("");
    }
  };

  const removeUser = async () => {
    if (!deleteUser) return;
    const name = deleteUser.username;
    setBusy(`user-${name}`);
    setMessage("");
    try {
      await apiRequest(
        `/admin/users/${encodeURIComponent(name)}`,
        { method: "DELETE" },
        session.csrf_token,
      );
      setDeleteUser(null);
      setMessage(`${name} and its active sessions were deleted.`);
      await refresh();
    } catch (error) {
      reportFailure(error,"Account could not be deleted.");
    } finally {
      setBusy("");
    }
  };

  const createApiToken = async () => {
    setBusy("api-token"); setMessage("");
    try {
      const created = await apiRequest<AgentApiToken & { token: string }>(
        "/admin/agent-api-tokens",
        {
          method: "POST",
          body: JSON.stringify({ label: apiTokenLabel, scopes: apiTokenScopes }),
        },
        session.csrf_token,
      );
      setNewApiToken(created.token);
      setNewApiTokenScopes(created.scopes);
      setApiTokenLabel("");
      setMessage("API key created. Copy it now; Vaelor will not show it again.");
      await refresh();
    } catch (error) {
      reportFailure(error,"API connection could not be created.");
    } finally { setBusy(""); }
  };

  const revokeApiToken = async (id: string) => {
    setBusy(`api-${id}`); setMessage("");
    try {
      await apiRequest(`/admin/agent-api-tokens/${id}`, { method: "DELETE" }, session.csrf_token);
      setMessage("External API connection revoked.");
      setRevokeToken(null);
      await refresh();
    } catch (error) {
      reportFailure(error,"API connection could not be revoked.");
    } finally { setBusy(""); }
  };

  const startTotp = async () => {
    setBusy("totp"); setMessage("");
    try {
      const setup = await apiRequest<{ secret: string }>(
        "/auth/totp/setup", { method: "POST", body: "{}" }, session.csrf_token,
      );
      setTotpSecret(setup.secret);
      setMessage("Add this Vaelor node to your authenticator app, then enter its code.");
    } catch (error) {
      reportFailure(error,"Two-factor setup could not start.");
    } finally { setBusy(""); }
  };

  const confirmTotp = async () => {
    setBusy("totp"); setMessage("");
    try {
      await apiRequest("/auth/totp/confirm", {
        method: "POST", body: JSON.stringify({ code: totpCode }),
      }, session.csrf_token);
      setTotpSecret(""); setTotpCode(""); setTotpEnabled(true);
      setMessage("Two-factor authentication is now required for your account.");
      await refresh();
    } catch (error) {
      reportFailure(error,"The authenticator code was not accepted.");
    } finally { setBusy(""); }
  };

  const disableTotp = async () => {
    setBusy("totp"); setMessage("");
    try {
      await apiRequest("/auth/totp", {
        method: "DELETE", body: JSON.stringify({ code: totpCode }),
      }, session.csrf_token);
      setTotpCode(""); setTotpEnabled(false);
      setMessage("Two-factor authentication disabled.");
      await refresh();
    } catch (error) {
      reportFailure(error,"The authenticator code was not accepted.");
    } finally { setBusy(""); }
  };

  const applyFactoryReset = async () => {
    setBusy("reset-apply"); setMessage("");
    try {
      await apiRequest(
        "/admin/recovery/factory-reset",
        {
          method: "POST",
          body: JSON.stringify({ confirmation: resetConfirmation }),
        },
        session.csrf_token,
      );
      setMessage("Factory reset accepted. This session will end and first-run setup will return.");
    } catch (error) {
      reportFailure(error,"Factory reset was not accepted.");
    } finally { setBusy(""); }
  };

  const exportPortableState = async () => {
    setBusy("portable-export"); setMessage("");
    try {
      await downloadApiRequest(
        "/admin/portable-state/export",
        "vaelor-state.vaelor",
        {
          method: "POST",
          body: JSON.stringify({ passphrase: exportPassphrase }),
        },
        session.csrf_token,
      );
      setExportPassphrase("");
      setExportPassphraseAgain("");
      setMessage("Encrypted Vaelor state downloaded. Store the file and passphrase separately.");
    } catch (error) {
      reportFailure(error,"Portable state could not be exported.");
    } finally { setBusy(""); }
  };

  const stagePortableImport = async () => {
    if (!importArchive) return;
    setBusy("portable-stage"); setMessage("");
    const form = new FormData();
    form.set("archive", importArchive);
    form.set("passphrase", importPassphrase);
    try {
      const status = await apiRequest<PortableStateStatus>(
        "/admin/portable-state/import/stage",
        { method: "POST", body: form },
        session.csrf_token,
      );
      setPortableState(status);
      setMessage("Transfer verified. Review the replacement scope before approving it.");
    } catch (error) {
      reportFailure(error,"The transfer archive could not be verified.");
    } finally { setBusy(""); }
  };

  const cancelPortableImport = async () => {
    setBusy("portable-cancel"); setMessage("");
    try {
      const status = await apiRequest<PortableStateStatus>(
        "/admin/portable-state/import/stage",
        { method: "DELETE" },
        session.csrf_token,
      );
      setPortableState(status);
      setImportArchive(null);
      setImportPassphrase("");
      setImportConfirmation("");
      setMessage("Staged transfer removed. No appliance data changed.");
    } catch (error) {
      reportFailure(error,"The staged transfer could not be removed.");
    } finally { setBusy(""); }
  };

  const applyPortableImport = async () => {
    setBusy("portable-import"); setMessage("");
    try {
      await apiRequest(
        "/admin/portable-state/import",
        {
          method: "POST",
          body: JSON.stringify({ confirmation: importConfirmation }),
        },
        session.csrf_token,
      );
      setMessage("Import accepted. Vaelor will replace portable state, restart services, and end this sign-in.");
    } catch (error) {
      reportFailure(error,"The portable import was not accepted.");
    } finally { setBusy(""); }
  };

  /*
   * The username field used to lowercase capitals inside `onChange` while
   * letting spaces and punctuation through to be rejected later. Half the input
   * was rewritten under the typist's hands and the other half was flagged, so
   * neither behaviour was learnable. The field now keeps exactly what was
   * typed, states up front that the name is stored in lowercase, and normalises
   * once on submit — where the confirmation says which name was created.
   */
  const normalizedUsername = username.trim().toLowerCase();
  const usernameValid = /^[a-z0-9][a-z0-9_.-]{1,63}$/.test(normalizedUsername);
  /*
   * #149: typing `a` used to be answered with "Use letters, numbers, dots,
   * dashes, or underscores…" — a rule `a` satisfies in every clause. The rule
   * actually broken was the 2-character minimum, and an error that names an
   * unbroken rule teaches the reader the form is wrong, not the input. Each
   * branch below names only the rule the current value breaks.
   */
  const usernameError = !username
    ? undefined
    : normalizedUsername.length < 2
      ? "Usernames need at least 2 characters."
      : normalizedUsername.length > 64
        ? "Usernames can have at most 64 characters."
        : !/^[a-z0-9]/.test(normalizedUsername)
          ? "Start the username with a letter or number."
          : !usernameValid
            ? "Use only letters, numbers, dots, dashes, or underscores. Spaces and other punctuation are not allowed."
            : undefined;
  const usernameWillChange = Boolean(username) && usernameValid && normalizedUsername !== username;
  // The button points at the field rather than repeating its sentence — the
  // field error already names the broken rule beside the input.
  const createAccountBlocked = !username
    ? "Enter a username to continue."
    : usernameError
      ? "Fix the username to continue."
      : password.length < 12
        ? "Enter a temporary passphrase of at least 12 characters to continue."
        : undefined;
  const enabledAdministratorCount = users.filter((user) => user.role === "administrator" && user.enabled).length;
  const exportPassphrasesMismatch = Boolean(
    exportPassphraseAgain && exportPassphrase !== exportPassphraseAgain,
  );

  return (
    <div className="admin-page">
      <div className="page-heading">
        <div>
          <h1>{destinations.admin.name}</h1>
          <p>Manage who can use this Vaelor node, where sessions are active, and which encrypted provider connections exist.</p>
        </div>
        <div className="workspace-route-actions">
          {onBack && <Button onClick={onBack} type="button" variant="quiet">Back to overview</Button>}
          <Button disabled={Boolean(busy)} onClick={() => void refresh()} type="button" variant="quiet">Reload</Button>
          <StatusPill label="Administrator" status="healthy" />
        </div>
      </div>
      {/* #150: this tablist was half-built — the roles were declared, but
          Right arrow did nothing, every tab sat in the Tab order, and no
          panel was wired to its tab. It now follows the same pattern the
          System tabs use: roving tabindex, arrow keys, aria-controls. */}
      <div className="assistant-tabs" role="tablist" aria-label="Settings sections">
        {([
          ["users", "Accounts", "shield"],
          ["sessions", "Sessions", "network"],
          ["secrets", "Connections", "lock"],
          ["recovery", "Recovery", "alert"],
        ] as const).map(([id, label, icon], index, sections) => (
          <Button
            // Only the active panel is mounted, so only the selected tab may
            // reference one — an aria-controls to an id that is not in the
            // document is a broken reference three tabs out of four.
            aria-controls={tab === id ? `settings-panel-${id}` : undefined}
            aria-selected={tab === id}
            className="assistant-tab"
            id={`settings-tab-${id}`}
            key={id}
            onClick={() => selectTab(id)}
            onKeyDown={(event) => {
              if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
              // Alt+Left is the browser's own Back; a focused tab must not
              // swallow it — or any other modified press.
              if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
              event.preventDefault();
              const next = sections[(index + (event.key === "ArrowRight" ? 1 : -1) + sections.length) % sections.length][0];
              selectTab(next, { replaceHistory: true });
              document.getElementById(`settings-tab-${next}`)?.focus();
            }}
            role="tab"
            tabIndex={tab === id ? 0 : -1}
          >
            <Icon name={icon} />{label}
          </Button>
        ))}
      </div>
      {message && <Notice severity={messageFailed ? "danger" : "info"}><Icon name="shield" />{message}</Notice>}

      {tab === "users" && (
        <div aria-labelledby="settings-tab-users" className="admin-layout" id="settings-panel-users" role="tabpanel" tabIndex={-1}>
          <section className="data-panel admin-create" aria-labelledby="new-account-title">
            <div className="panel-heading"><div><h2 id="new-account-title">Add an account</h2><p>Start with the least access the person needs.</p></div><Icon name="lock" /></div>
            {/*
              * #149: the hint and error were hand-rolled <small> siblings, so
              * the error rendered byte-identical to the help text above it and
              * the field's described-by never named it. The Input primitive
              * already draws the red border, the danger error line, and the
              * aria wiring — Case lighting has used it correctly all along.
              * maxLength is gone too: it silently truncated a 104-character
              * paste at 64, where the validation now says so instead.
              */}
            <Input
              autoCapitalize="none"
              error={usernameError}
              hint="Use 2–64 letters, numbers, dots, dashes, or underscores. The name is saved in lowercase."
              id="account-username"
              label="Username"
              onChange={(event) => setUsername(event.target.value)}
              value={username}
            />
            {usernameWillChange && <small id="account-username-normalized">This account will be created as {normalizedUsername}.</small>}
            <Input
              autoComplete="new-password"
              hint="Use at least 12 characters. This passphrase is not shown again."
              id="account-password"
              label="Temporary passphrase"
              maxLength={256}
              minLength={12}
              onChange={(event) => setPassword(event.target.value)}
              type="password"
              value={password}
            />
            <Select id="account-role" label="Access level" onChange={(event) => setRole(event.target.value as Role)} value={role}>
              <option value="viewer">Viewer — see status</option>
              <option value="operator">Operator — control and deploy</option>
              <option value="administrator">Administrator — manage access</option>
            </Select>
            {/* A disabled primary action has to say what is missing. The Button
                primitive renders `disabledReason` as visible, described-by text;
                this call site simply never passed one. */}
            <Button
              disabledReason={createAccountBlocked}
              disabled={busy === "create"}
              onClick={() => void createUser()}
              variant="primary"
            >
              {busy === "create" ? "Creating…" : "Create account"}
            </Button>
          </section>
          <section className="data-panel admin-users" aria-labelledby="accounts-title">
            <div className="panel-heading"><div><h2 id="accounts-title">Local accounts</h2><p>Disabling an account also ends its sessions.</p></div><span>{users.length} total</span></div>
            <PaginatedItems items={users} label="Local accounts" pageSize={8} render={(user) => (
              <div className="admin-user-row" key={user.username}>
                <span className="avatar">{user.username[0].toUpperCase()}</span>
                <div><strong>{user.username}{user.username === session.user.username ? " · you" : ""}</strong><small>{user.active_sessions} active session{user.active_sessions === 1 ? "" : "s"}</small>{(user.role === "administrator" && user.enabled && enabledAdministratorCount === 1) || user.username === session.user.username ? <span id={`admin-user-${user.username}-restriction`}>{user.role === "administrator" && user.enabled && enabledAdministratorCount === 1 ? FINAL_ADMINISTRATOR_REASON : CURRENT_ACCOUNT_REASON}</span> : null}</div>
                <select aria-describedby={user.role === "administrator" && user.enabled && enabledAdministratorCount === 1 ? "admin-user-" + user.username + "-restriction" : undefined} aria-label={user.username + " role"} className="ui-control ui-control--select" disabled={busy === "user-" + user.username || (user.role === "administrator" && user.enabled && enabledAdministratorCount === 1)} onChange={(event) => void updateUser(user.username, { role: event.target.value as Role })} value={user.role}>
                  <option value="viewer">Viewer</option><option value="operator">Operator</option><option value="administrator">Administrator</option>
                </select>
                {/* #150: each action names its verb and its account — the
                    restriction text stays a description, never the name, so
                    a screen reader can tell Disable from Delete on exactly
                    the two controls where confusing them matters most. */}
                <div className="admin-user-row__actions">
                  <Button variant="quiet" aria-label={`${user.enabled ? "Disable" : "Enable"} ${user.username}`} aria-describedby={user.role === "administrator" && user.enabled && enabledAdministratorCount === 1 || user.username === session.user.username ? `admin-user-${user.username}-restriction` : undefined} disabled={busy === `user-${user.username}` || user.username === session.user.username || (user.role === "administrator" && user.enabled && enabledAdministratorCount === 1)} onClick={() => void updateUser(user.username, { enabled: !user.enabled })}>
                    {user.enabled ? "Disable" : "Enable"}
                  </Button>
                  <Button variant="danger" aria-label={`Delete ${user.username}`} aria-describedby={user.role === "administrator" && user.enabled && enabledAdministratorCount === 1 || user.username === session.user.username ? `admin-user-${user.username}-restriction` : undefined} disabled={busy === `user-${user.username}` || user.username === session.user.username || (user.role === "administrator" && user.enabled && enabledAdministratorCount === 1)} onClick={() => setDeleteUser(user)}>
                    Delete
                  </Button>
                </div>
              </div>
            )} />
          </section>
          <section className="data-panel admin-create mfa-panel" aria-labelledby="mfa-title">
            <div className="panel-heading"><div><h2 id="mfa-title">Your two-factor sign-in</h2><p>Add an authenticator code after your password.</p></div><StatusPill label={totpEnabled ? "Enabled" : "Optional"} status={totpEnabled ? "healthy" : "neutral"} /></div>
            {!totpEnabled && !totpSecret && <Button variant="primary" disabled={busy === "totp"} onClick={() => void startTotp()}>Set up authenticator</Button>}
            {totpSecret && <><div className="totp-secret"><small>Manual setup key</small><code>{totpSecret}</code><Button variant="quiet" onClick={() => void navigator.clipboard.writeText(totpSecret)}>Copy</Button></div><Input autoComplete="one-time-code" id="totp-code" inputMode="numeric" label="Six-digit verification code" maxLength={6} onChange={(event) => setTotpCode(event.target.value.replace(/\D/g, ""))} value={totpCode} /><Button variant="primary" disabled={busy === "totp"} disabledReason={totpCode.length === 6 ? undefined : "Enter the six-digit code from your authenticator app to continue."} onClick={() => void confirmTotp()}>Verify and enable</Button></>}
            {totpEnabled && <><Input autoComplete="one-time-code" id="totp-code-current" inputMode="numeric" label="Current authenticator code" maxLength={6} onChange={(event) => setTotpCode(event.target.value.replace(/\D/g, ""))} value={totpCode} /><Button variant="danger" disabled={totpCode.length !== 6 || busy === "totp"} onClick={() => void disableTotp()}>Verify and disable</Button></>}
          </section>
        </div>
      )}
      <ConfirmDialog
        busy={Boolean(deleteUser && busy === `user-${deleteUser.username}`)}
        confirmLabel="Delete account"
        description={deleteUser ? `${deleteUser.username}, its authenticator setup, and every active session will be permanently removed.` : ""}
        onCancel={() => !busy && setDeleteUser(null)}
        onConfirm={() => void removeUser()}
        open={Boolean(deleteUser)}
        title="Delete local account?"
      />
      <ConfirmDialog
        busy={Boolean(revokeToken && busy === `api-${revokeToken.id}`)}
        confirmLabel="Revoke connection"
        description={revokeToken ? `${revokeToken.label} (${revokeToken.prefix}…) will stop working immediately. Any app using this key will need a new one.` : ""}
        onCancel={() => !busy && setRevokeToken(null)}
        onConfirm={() => { if (revokeToken) void revokeApiToken(revokeToken.id); }}
        open={Boolean(revokeToken)}
        title="Revoke API connection?"
      />

      {tab === "sessions" && (
        <section aria-labelledby="settings-tab-sessions" className="data-panel admin-wide" id="settings-panel-sessions" role="tabpanel" tabIndex={-1}>
          <div className="panel-heading"><div><h2 id="sessions-title">Signed-in devices</h2><p>End access you do not recognize. Session tokens are never shown.</p></div><Icon name="network" /></div>
          <PaginatedItems items={sessions} label="Signed-in devices" pageSize={8} render={(item) => (
            <div className="admin-session-row" key={item.id}>
              <span><Icon name={item.current ? "shield" : "display"} /></span>
              <div><strong>{item.username}{item.current ? " · this device" : ""}</strong><small>{item.remote_addr || "Local"} · last active {new Date(item.last_seen_at * 1000).toLocaleString()}</small><small>{item.user_agent || "Unknown client"}</small></div>
              <Button variant="danger" disabled={item.current || busy === `session-${item.id}`} onClick={() => void revokeSession(item.id)}>{item.current ? "Current" : "End session"}</Button>
            </div>
          )} />
        </section>
      )}

      {tab === "secrets" && (
        <div aria-labelledby="settings-tab-secrets" className="settings-tabpanel" id="settings-panel-secrets" role="tabpanel" tabIndex={-1}>
        <section className="data-panel admin-wide" aria-labelledby="transport-title">
          <div className="panel-heading">
            <div><h2 id="transport-title">Secure remote access</h2><p>Encryption protects sign-in, API, and Remote Desktop traffic.</p></div>
            <ReadStatusPill notLabel="LAN only" ok={Boolean(transport?.remote_ready)} okLabel="HTTPS + WSS ready" read={Boolean(transport)} />
          </div>
          <div className="transport-security">
            <span><Icon name="shield" /><div><small>Dashboard</small><strong>{transport?.secure ? "Encrypted HTTPS" : "Unencrypted HTTP"}</strong></div></span>
            <span><Icon name="display" /><div><small>Remote Desktop</small><strong>{transport?.vnc_secure ? "Encrypted WSS" : "Trusted LAN only"}</strong></div></span>
            <span><Icon name="lock" /><div><small>Certificate</small><strong>{transport?.certificate_managed ? "Local appliance certificate" : "Not configured"}</strong></div></span>
          </div>
          {transport?.certificate_fingerprint && (
            <div className="certificate-fingerprint">
              <span><small>SHA-256 fingerprint</small><code>{transport.certificate_fingerprint.match(/.{1,2}/g)?.join(":")}</code></span>
              <a className="ui-button ui-button--quiet" download href="/api/v2/security/certificate">Download certificate</a>
            </div>
          )}
          {/* Only once the transport has actually been read: warning about an
              unencrypted link Vaelor has not looked at is a guess wearing an
              alarm's clothes, and it trains the reader to ignore the real one. */}
          {transport && !transport.remote_ready && <Notice severity="warning">Do not expose ports 34001 or 34002 directly to the internet until HTTPS and WSS are active. Use a trusted LAN or VPN.</Notice>}
        </section>
        <section className="data-panel admin-wide" aria-labelledby="connections-title">
          <div className="panel-heading"><div><h2 id="connections-title">Encrypted AI connections</h2><p>Only metadata is visible. Secret values cannot be read back.</p></div><Icon name="lock" /></div>
          {credentials.length ? <PaginatedItems items={credentials} label="AI connections" pageSize={8} render={(item) => (
            <div className="admin-session-row" key={item.id}>
              <span><Icon name="lock" /></span>
              <div><strong>{item.label}</strong><small>{item.provider} · fingerprint …{item.fingerprint.slice(-6)}</small></div>
              {/*
                * #143: this pill said "success" — a raw status code — in
                * green, indefinitely. A connection that passed weeks ago and
                * has offered no models since wore the same green as one that
                * passed a minute ago. A test result is only readable next to
                * when it was measured, and a failed one is a failure, not a
                * neutral fact.
                */}
              <StatusPill
                label={item.last_test_status === "success"
                  ? `Passed ${item.last_tested_at ? timeAgo(item.last_tested_at * 1000) : "· date not recorded"}`
                  : item.last_test_status === "failed"
                    ? `Failed ${item.last_tested_at ? timeAgo(item.last_tested_at * 1000) : "· date not recorded"}`
                    : "Not tested"}
                status={item.last_test_status === "success" ? "healthy" : item.last_test_status === "failed" ? "degraded" : "neutral"}
              />
            </div>
          )} /> : <div className="empty-state"><Icon name="lock" /><strong>No provider connections</strong><span>Connect one from Workloads → Improve the assistant.</span></div>}
        </section>
        <section className="data-panel admin-wide" aria-labelledby="agent-api-title">
          <div className="panel-heading"><div><h2 id="agent-api-title">External API access</h2><p>Create revocable keys for Vaelor Assistant, managed cluster inference, or both.</p></div><Icon name="network" /></div>
          <div className="gateway-status">
            <div>
              <span className="page-eyebrow">Inference gateway</span>
              <strong>{inferenceGateway?.target?.label ?? "No active cluster model"}</strong>
              <small>{inferenceGateway?.message ?? "Reading the gateway status."}</small>
            </div>
            <ReadStatusPill notLabel="Unavailable" ok={Boolean(inferenceGateway?.healthy)} okLabel="Reachable" read={Boolean(inferenceGateway)} />
            {/* Four counters that defaulted to 0 before the request answered.
                A zero is a measurement, and none of these had been taken. */}
            <dl>
              {([
                ["Requests · 24h", (metrics) => String(metrics.requests)],
                ["Failures", (metrics) => String(metrics.failures)],
                ["Average latency", (metrics) => `${Math.round(metrics.average_latency_ms)} ms`],
                ["Tokens", (metrics) => String(metrics.prompt_tokens + metrics.completion_tokens)],
              ] as Array<[string, (metrics: InferenceGatewayStatus["metrics_24h"]) => string]>)
                .map(([term, read]) => (
                  <div key={term}>
                    <dt>{term}</dt>
                    <dd>{inferenceGateway
                      ? read(inferenceGateway.metrics_24h)
                      : <UnavailableValue label={`${term} unavailable`} reason="Vaelor has not read the gateway yet" />}</dd>
                  </div>
                ))}
            </dl>
          </div>
          <div className="agent-api-create">
            <Input id="api-token-label" label="Connection name" maxLength={100} onChange={(event) => setApiTokenLabel(event.target.value)} placeholder="Example: My desktop chat app" value={apiTokenLabel} />
            <fieldset>
              <legend>Allowed APIs</legend>
              {(["assistant", "inference"] as const).map((scope) => (
  <Checkbox
    checked={apiTokenScopes.includes(scope)}
    id={"api-scope-" + scope}
    key={scope}
    label={scope === "assistant" ? "Assistant" : "Cluster inference"}
    onChange={(event) => setApiTokenScopes(event.target.checked ? [...apiTokenScopes, scope] : apiTokenScopes.filter((item) => item !== scope))}
  />
))}
            </fieldset>
            <Button variant="primary" disabled={busy === "api-token"} disabledReason={!apiTokenLabel.trim() ? "Name this key so it can be recognised later." : !apiTokenScopes.length ? "Select at least one scope this key may use." : undefined} onClick={() => void createApiToken()}>Create one-time API key</Button>
          </div>
          {newApiToken && <div className="one-time-token" role="status"><Icon name="shield" /><div><strong>Copy this key now</strong><code>{newApiToken}</code><small>{newApiTokenScopes.includes("assistant") && `Assistant: ${window.location.origin}/v1 · model vaelor-assistant`}{newApiTokenScopes.includes("assistant") && newApiTokenScopes.includes("inference") && " · "}{newApiTokenScopes.includes("inference") && `Inference: ${window.location.origin}/inference/v1`}</small></div><Button onClick={() => void navigator.clipboard.writeText(newApiToken)}>Copy key</Button><Button variant="quiet" onClick={() => setNewApiToken("")}>I saved it</Button></div>}
          <div className="credential-list"><PaginatedItems items={apiTokens} label="External API connections" pageSize={8} render={(item) => <div className="credential-row" key={item.id}><span className="credential-row__icon"><Icon name="network" /></span><span><strong>{item.label}</strong><small>{item.scopes.map((scope) => scope === "assistant" ? "Assistant" : "Inference").join(" + ")} · {item.prefix}… · {item.last_used_at ? `used ${new Date(item.last_used_at * 1000).toLocaleString()}` : "never used"}</small></span><StatusPill label={item.enabled ? "Active" : "Revoked"} status={item.enabled ? "healthy" : "neutral"} />{item.enabled && <Button variant="danger" disabled={busy === `api-${item.id}`} onClick={() => setRevokeToken(item)}>Revoke</Button>}</div>} /></div>
        </section>
        </div>
      )}

      {tab === "recovery" && (
        <div aria-labelledby="settings-tab-recovery" className="settings-tabpanel" id="settings-panel-recovery" role="tabpanel" tabIndex={-1}>
        <section className="data-panel admin-wide portable-state" aria-labelledby="portable-state-title">
          <div className="panel-heading">
            <div>
              <span className="page-eyebrow">Encrypted state transfer</span>
              <h2 id="portable-state-title">Move this Vaelor</h2>
              <p>Carry accounts, conversations, agents, and app definitions to another Vaelor installation.</p>
            </div>
            <ReadStatusPill notLabel="Transfer staged" ok={Boolean(portableState) && !portableState?.staged} okLabel="Ready" read={Boolean(portableState)} />
          </div>

          <div className="portable-state__manifest">
            <div>
              <span>Moves with the archive</span>
              <p>{portableState?.scope.includes.join(" · ") ?? "Loading portable data scope…"}</p>
            </div>
            <div>
              <span>Stays on this hardware</span>
              <p>{portableState?.scope.excludes.join(" · ") ?? "Loading protected data scope…"}</p>
            </div>
          </div>

          <div className="portable-state__actions">
            <article>
              <div className="portable-state__action-heading">
                <Icon name="download" />
                <div><h3>Export encrypted state</h3><p>Create a versioned archive for another Vaelor node.</p></div>
              </div>
<Input autoComplete="new-password" id="export-passphrase" label="Transfer passphrase" minLength={16} onChange={(event) => setExportPassphrase(event.target.value)} type="password" value={exportPassphrase} />
              <Input aria-invalid={exportPassphrasesMismatch} autoComplete="new-password" id="export-passphrase-again" label="Confirm passphrase" minLength={16} onChange={(event) => setExportPassphraseAgain(event.target.value)} type="password" value={exportPassphraseAgain} />
              {exportPassphrasesMismatch && <small role="alert">The transfer passphrases do not match.</small>}
              <Button variant="primary"

                disabled={Boolean(busy) || exportPassphrase.length < 16 || exportPassphrase !== exportPassphraseAgain}
                onClick={() => void exportPortableState()}
                type="button"
              >
                {busy === "portable-export" ? "Encrypting…" : "Download encrypted archive"}
              </Button>
            </article>

            <article>
              <div className="portable-state__action-heading">
                <Icon name="upload" />
                <div><h3>Import on this node</h3><p>Verify first; replacement starts only after exact approval.</p></div>
              </div>
              {!portableState?.staged ? (
                <>
                  <label>Vaelor transfer archive
                    <input accept=".vaelor,application/octet-stream" className="ui-control ui-control--file-picker" onChange={(event) => setImportArchive(event.target.files?.[0] ?? null)} type="file" />
                  </label>
<Input autoComplete="current-password" id="import-passphrase" label="Transfer passphrase" minLength={16} onChange={(event) => setImportPassphrase(event.target.value)} type="password" value={importPassphrase} />
                  <Button disabled={Boolean(busy) || !importArchive || importPassphrase.length < 16} onClick={() => void stagePortableImport()} type="button">
                    {busy === "portable-stage" ? "Verifying…" : "Review transfer"}
                  </Button>
                </>
              ) : (
                <div className="portable-state__approval">
                  <div><small>Verified source</small><strong>Vaelor {portableState.plan?.source_version ?? "compatible archive"}</strong><span>Approval expires {portableState.plan ? new Date(portableState.plan.expires_at * 1000).toLocaleTimeString() : "soon"}.</span></div>
<Input autoComplete="off" id="import-confirmation" label={<span>Type <strong>{portableState.confirmation}</strong></span>} onChange={(event) => setImportConfirmation(event.target.value)} value={importConfirmation} />
                  <div>
                    <Button variant="quiet" disabled={Boolean(busy) || Boolean(portableState.plan?.approved_at)} onClick={() => void cancelPortableImport()} type="button">Cancel transfer</Button>
                    <Button variant="danger" disabled={Boolean(busy) || Boolean(portableState.plan?.approved_at) || importConfirmation !== portableState.confirmation} onClick={() => void applyPortableImport()} type="button">
                      {busy === "portable-import" ? "Starting import…" : "Replace portable state"}
                    </Button>
                  </div>
                </div>
              )}
            </article>
          </div>
          <p className="portable-state__warning"><Icon name="alert" />Import ends active sessions. Provider keys, SSH credentials, models, container volumes, TLS keys, and hardware identity are never transferred.</p>
        </section>
        <BackupSettings session={session} />
        <section className="data-panel admin-wide factory-reset" aria-labelledby="factory-reset-title">
          <div className="panel-heading">
            <div>
              <span className="page-eyebrow">Last-resort recovery</span>
              <h2 id="factory-reset-title">Reset Vaelor</h2>
              <p>Use this only when the control plane cannot be repaired. Resetting returns the app to first-run setup.</p>
            </div>
            <StatusPill label="Approval required" status="neutral" />
          </div>

          <div className="factory-reset__scope">
            <section>
              <h3>Will be permanently erased</h3>
              <ul>{recovery?.erases.map((item) => <li key={item}>{item}</li>)}</ul>
            </section>
            <section>
              <h3>Will remain</h3>
              <ul>{recovery?.retains.map((item) => <li key={item}>{item}</li>)}</ul>
            </section>
          </div>

          <div className="factory-reset__confirmation">
              <div>
                <strong>Review complete</strong>
                <small>No data is erased until the final action is approved.</small>
              </div>
<Input autoComplete="off" id="reset-confirmation" label={<span>Type <strong>{recovery?.confirmation ?? "RESET VAELOR"}</strong> to unlock the final action</span>} onChange={(event) => setResetConfirmation(event.target.value)} value={resetConfirmation} />
              <div className="factory-reset__actions">
                <Button variant="danger"

                  disabled={Boolean(busy) || !recovery || resetConfirmation !== recovery.confirmation}
                  onClick={() => void applyFactoryReset()}
                  type="button"
                >
                  {busy === "reset-apply" ? "Starting reset…" : "Approve and reset application"}
                </Button>
              </div>
          </div>
        </section>
        </div>
      )}
    </div>
  );
}
