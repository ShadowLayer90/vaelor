import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest, SESSION_CHANGED_EVENT } from "../lib/api";
import type { Role, Session } from "../types";
import { EncryptedConnectionsPanel, ExternalApiPanel, SecureRemoteAccess, type AgentApiToken, type InferenceGatewayStatus, type TransportStatus } from "./ConnectionsSettings";
import { Icon } from "./Icon";
import { RecordConfirm } from "./RecordKit";
import { SettingsAccounts, type ManagedUser } from "./SettingsAccounts";
import { SettingsRecovery, type PortableStateStatus, type RecoveryStatus, type RemoveVaelorStatus } from "./SettingsRecovery";
import { SettingsSessions, type ManagedSession } from "./SettingsSessions";
import { SettingsTwoFactor } from "./SettingsTwoFactor";
import { StatusPill } from "./StatusPill";
import { TrustThisVaelor } from "./TrustThisVaelor";
import { Button, Notice, TabSet } from "./ui";
import type { CredentialListing, StoredCredential } from "../lib/credentialStatus";
import { destinations } from "../lib/destinations";
import { readDraft, writeDraft } from "../lib/draftStorage";
import { TopbarPageActions, usePagePlace } from "../lib/topbarSlot";
import "../styles/settings.css";

/*
 * Settings (VD-200: the Settings, Connections, Recovery, SettingsSessions and
 * SettingsBackup boards). This module reads everything in one refresh and owns
 * every request and its outcome; each tab's cards are drawn by their own
 * module (SettingsAccounts, SettingsSessions, ConnectionsSettings,
 * SettingsRecovery with BackupSettings).
 */

/** CR2: sent only after the own-role confirmation (`api_auth_routes.SELF_DEMOTION_CONFIRM`). */
const SELF_DEMOTION_CONFIRM = "leave-administrator";
const ROLE_NAMES: Record<Role, string> = { viewer: "Viewer", operator: "Operator", administrator: "Administrator" };
/** How long the "you are leaving Settings" message stays before the shell re-routes. */
const LEAVE_SETTINGS_AFTER_MS = 2500;

export type AdministrationSection = "users" | "sessions" | "secrets" | "recovery";

/** Settings' tabs, in the strip and in the top bar breadcrumb ("Settings / Sessions"). */
const SECTIONS: ReadonlyArray<{ id: AdministrationSection; label: string }> = [
  { id: "users", label: "Accounts" },
  { id: "sessions", label: "Sessions" },
  { id: "secrets", label: "Connections" },
  { id: "recovery", label: "Recovery" },
];

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

export function Administration({ session }: { session: Session; /** The shell passes it; the rail and breadcrumb are the way back now (VD-200). */ onBack?: () => void }) {
  const [tab, setTab] = useState<AdministrationSection>(() => administrationSectionFromHash(window.location.hash));
  const [users, setUsers] = useState<ManagedUser[]>([]);
  const [sessions, setSessions] = useState<ManagedSession[] | null>(null);
  const [credentials, setCredentials] = useState<CredentialListing | null>(null);
  const [apiTokens, setApiTokens] = useState<AgentApiToken[]>([]);
  const [apiTokenLabel, setApiTokenLabel] = useState("");
  const [newApiToken, setNewApiToken] = useState("");
  const [inferenceGateway, setInferenceGateway] = useState<InferenceGatewayStatus | null>(null);
  const [transport, setTransport] = useState<TransportStatus | null>(null);
  // null until `/auth/totp` answers: the card says "Reading", not "Optional".
  const [totpEnabled, setTotpEnabled] = useState<boolean | null>(null);
  /** A two-factor setup in progress survives a tab change (S-Y7); see SettingsTwoFactor. */
  const [totpSecret, setTotpSecret] = useState("");
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
  // VD-189: a refusal of a confirmation dialog's action stays in that dialog; the page under it is inert.
  const [dialogError, setDialogError] = useState("");
  const reportFailure = (error: unknown, fallback: string, inDialog = false) => {
    const text = error instanceof Error && error.message ? error.message : fallback;
    if (inDialog) setDialogError(text); else { setMessageFailed(true); setMessageText(text); }
  };
  const [recovery, setRecovery] = useState<RecoveryStatus | null>(null);
  const [removeVaelor, setRemoveVaelor] = useState<RemoveVaelorStatus | null>(null);
  const [portableState, setPortableState] = useState<PortableStateStatus | null>(null);
  const [deleteUser, setDeleteUser] = useState<ManagedUser | null>(null);
  const [revokeToken, setRevokeToken] = useState<AgentApiToken | null>(null);
  // CR2: a role change on your OWN row waits here for an explicit confirmation.
  const [ownRoleChange, setOwnRoleChange] = useState<Role | null>(null);
  const leaveTimer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(leaveTimer.current), []);

  const refresh = useCallback(async () => {
    const [nextUsers, nextSessions, secretData, nextApiTokens, gateway, mfa, nextTransport, nextRecovery, nextRemove, nextPortable] =
      await Promise.allSettled([
        apiRequest<ManagedUser[]>("/admin/users"),
        apiRequest<ManagedSession[]>("/admin/sessions"),
        apiRequest<CredentialListing>("/credentials"),
        apiRequest<AgentApiToken[]>("/admin/agent-api-tokens"),
        apiRequest<InferenceGatewayStatus>("/admin/inference-gateway"),
        apiRequest<{ enabled: boolean }>("/auth/totp"),
        apiRequest<TransportStatus>("/security/transport"),
        apiRequest<RecoveryStatus>("/admin/recovery/factory-reset"),
        apiRequest<RemoveVaelorStatus>("/admin/recovery/remove-vaelor"),
        apiRequest<PortableStateStatus>("/admin/portable-state"),
      ]);
    if (nextUsers.status === "fulfilled") setUsers(nextUsers.value);
    if (nextSessions.status === "fulfilled") setSessions(nextSessions.value);
    if (secretData.status === "fulfilled") setCredentials(secretData.value);
    if (nextApiTokens.status === "fulfilled") setApiTokens(nextApiTokens.value);
    if (gateway.status === "fulfilled") setInferenceGateway(gateway.value);
    if (mfa.status === "fulfilled") setTotpEnabled(mfa.value.enabled);
    if (nextTransport.status === "fulfilled") setTransport(nextTransport.value);
    if (nextRecovery.status === "fulfilled") setRecovery(nextRecovery.value);
    if (nextRemove.status === "fulfilled") setRemoveVaelor(nextRemove.value);
    if (nextPortable.status === "fulfilled") setPortableState(nextPortable.value);
    const unavailable = [
      nextUsers, nextSessions, secretData, nextApiTokens, gateway, mfa, nextTransport, nextRecovery, nextRemove, nextPortable,
    ].filter((result) => result.status === "rejected").length;
    if (unavailable > 0) {
      // A failed refresh is a failure, not information (#149 review).
      setMessageFailed(true);
      setMessageText(
        unavailable === 10
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


  usePagePlace([SECTIONS.find((section) => section.id === tab)?.label ?? SECTIONS[0].label]);

  const selectTab = (next: AdministrationSection, keyboard = false) => {
    if (next === tab) return;
    setTab(next);
    setMessage("");
    // Each tab is an address, pushed as Activity pushes its tabs, so Back
    // returns to the tab before (the System review's NIT: the two pages
    // disagreed). Arrow-key roving replaces the entry instead: walking
    // Accounts to Recovery by key once left an entry per tab passed, so Back
    // revisited each before leaving the page (as the pre-redesign page had it).
    if (keyboard) window.history.replaceState(null, "", administrationHashForSection(next));
    else window.history.pushState(null, "", administrationHashForSection(next));
  };

  /** A tab's own outcome, worded by the tab; a failure is never shown as information (#149). */
  const report = (text: string, failed = false) => { setMessageFailed(failed); setMessageText(text); };

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

  const updateUser = async (name: string, patch: { role?: Role; enabled?: boolean; confirm?: string }) => {
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
    setMessage(""); setDialogError("");
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
      reportFailure(error, "Account could not be deleted.", true);
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
          body: JSON.stringify({ label: apiTokenLabel, scopes: ["inference"] }),
        },
        session.csrf_token,
      );
      setNewApiToken(created.token);
      setApiTokenLabel("");
      setMessage("API key created. Copy it now; Vaelor will not show it again.");
      await refresh();
    } catch (error) {
      reportFailure(error,"API connection could not be created.");
    } finally { setBusy(""); }
  };

  const changeRole = (user: ManagedUser, next: Role) => {
    // Your own role is never changed on a stray select: say what it costs first.
    if (user.username === session.user.username && user.role === "administrator" && next !== "administrator") {
      setOwnRoleChange(next);
      return;
    }
    void updateUser(user.username, { role: next });
  };

  const confirmOwnRoleChange = async () => {
    if (!ownRoleChange) return;
    const next = ownRoleChange;
    setBusy(`user-${session.user.username}`); setMessage("");
    try {
      await apiRequest(`/admin/users/${encodeURIComponent(session.user.username)}`, {
        method: "PATCH", body: JSON.stringify({ role: next, confirm: SELF_DEMOTION_CONFIRM }),
      }, session.csrf_token);
      // Re-read who we are now: these screens are for administrators only.
      const active = await apiRequest<Session>("/auth/session");
      setOwnRoleChange(null);
      setMessage(`Your access is now ${ROLE_NAMES[active.user.role] ?? active.user.role}. Settings is for administrators only, so Vaelor is taking you back to Home.`);
      leaveTimer.current = window.setTimeout(() => {
        window.history.replaceState(null, "", "#/");
        window.dispatchEvent(new CustomEvent(SESSION_CHANGED_EVENT, { detail: active }));
      }, LEAVE_SETTINGS_AFTER_MS);
    } catch (error) {
      setOwnRoleChange(null);
      reportFailure(error, "Your access could not be changed.");
    } finally { setBusy(""); }
  };

  const testStoredCredential = async (item: StoredCredential) => {
    setBusy(`credential-${item.id}`); setMessage("");
    try {
      const result = await apiRequest<{ ok: boolean; message: string; tested?: boolean }>(`/credentials/${item.id}/test`, { method: "POST" }, session.csrf_token);
      // `tested: false` = the cluster model is resting on purpose: information, not a failure.
      if (result.ok || result.tested === false) setMessage(`${item.label}: ${result.message}`);
      else reportFailure(new Error(`${item.label}: ${result.message}`), "The connection test failed.");
      await refresh();
    } catch (error) {
      reportFailure(error, "The connection test could not run.");
    } finally { setBusy(""); }
  };

  const removeStoredCredential = async (item: StoredCredential) => {
    setBusy(`credential-${item.id}`); setMessage("");
    try {
      await apiRequest(`/credentials/${item.id}`, { method: "DELETE" }, session.csrf_token);
      setMessage(`${item.label} was removed from this appliance.`);
      await refresh();
    } catch (error) {
      reportFailure(error, "It could not be removed.");
    } finally { setBusy(""); }
  };

  const removeApiTokenRecord = async (token: AgentApiToken) => {
    setBusy(`api-${token.id}`); setMessage("");
    try {
      await apiRequest(`/admin/agent-api-tokens/${token.id}/record`, { method: "DELETE" }, session.csrf_token);
      setMessage(`The record of ${token.label} was removed. The audit log keeps its history.`);
      await refresh();
    } catch (error) {
      reportFailure(error, "The revoked key record could not be removed.");
    } finally { setBusy(""); }
  };

  const revokeApiToken = async (id: string) => {
    setBusy(`api-${id}`); setMessage(""); setDialogError("");
    try {
      await apiRequest(`/admin/agent-api-tokens/${id}`, { method: "DELETE" }, session.csrf_token);
      setMessage("External API connection revoked.");
      setRevokeToken(null);
      await refresh();
    } catch (error) {
      reportFailure(error, "API connection could not be revoked.", true);
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
  const toggleUser = (user: ManagedUser) => {
    if (user.username === session.user.username) return;
    if (user.role === "administrator" && user.enabled && enabledAdministratorCount === 1) return;
    void updateUser(user.username, { enabled: !user.enabled });
  };

  return (
    <div className="stg-page">
      {/* The Recovery board draws no page header, only the tabs; the name stays the page's heading for a screen reader. */}
      {tab === "recovery" ? <h1 className="sr-only">{destinations.admin.name}</h1> : (
        <div className="ui-page-header">
          <div className="record-page-title">
            <h1 className="ui-page-header__title">{destinations.admin.name}</h1>
            <p className="ui-page-header__title record-page-title__line">Who can use this machine, where they're signed in, and its connections.</p>
          </div>
        </div>
      )}
      <TopbarPageActions>
        <StatusPill label="Administrator" tone="info" />
        <Button disabled={Boolean(busy)} onClick={() => void refresh()} type="button"><Icon name="refresh" size={16} />Reload</Button>
      </TopbarPageActions>
      <TabSet items={SECTIONS} label="Settings sections" onSelect={(id, how) => selectTab(id as AdministrationSection, how?.keyboard)} selectedId={tab}>
        <div className="stg-stack">
          {message && <Notice severity={messageFailed ? "danger" : "success"}>{message}</Notice>}

          {tab === "users" && (
            <SettingsAccounts
              busy={busy}
              form={{
                username, password, role,
                usernameError,
                normalizedName: usernameWillChange ? normalizedUsername : undefined,
                blockedReason: createAccountBlocked,
                busy: busy === "create",
                onUsername: setUsername,
                onPassword: setPassword,
                onRole: setRole,
                onCreate: () => void createUser(),
              }}
              onDelete={setDeleteUser}
              onRoleChange={changeRole}
              onToggle={toggleUser}
              session={session}
              twoFactor={<SettingsTwoFactor enabled={totpEnabled} onChanged={setTotpEnabled} onSecret={setTotpSecret} secret={totpSecret} session={session} />}
              users={users}
            />
          )}

          {tab === "sessions" && <SettingsSessions busy={busy} onEnd={(id) => void revokeSession(id)} sessions={sessions} />}

          {tab === "secrets" && (
            <>
              <SecureRemoteAccess transport={transport} />
              <TrustThisVaelor transport={transport} />
              <div className="stg-two-col stg-two-col--even">
                <ExternalApiPanel
                  busy={busy}
                  gateway={inferenceGateway}
                  label={apiTokenLabel}
                  newToken={newApiToken}
                  onCreate={() => void createApiToken()}
                  onDismissToken={() => setNewApiToken("")}
                  onLabel={setApiTokenLabel}
                  onRemoveRecord={removeApiTokenRecord}
                  onRevoke={setRevokeToken}
                  tokens={apiTokens}
                />
                <EncryptedConnectionsPanel busy={busy} listing={credentials} onRemove={removeStoredCredential} onTest={testStoredCredential} />
              </div>
            </>
          )}

          {tab === "recovery" && (
            <SettingsRecovery
              onReport={report}
              portable={portableState}
              recovery={recovery}
              refresh={refresh}
              removeVaelor={removeVaelor}
              session={session}
              setPortable={setPortableState}
            />
          )}
        </div>
      </TabSet>

      {deleteUser && (
        <RecordConfirm
          busy={busy === `user-${deleteUser.username}`}
          confirmLabel="Delete account"
          error={dialogError}
          eyebrow="Can't be undone"
          onCancel={() => { if (!busy) { setDialogError(""); setDeleteUser(null); } }}
          onConfirm={() => void removeUser()}
          title="Delete local account?"
        >
          <p>{`${deleteUser.username}, its authenticator setup, and every active session will be permanently removed.`}</p>
        </RecordConfirm>
      )}
      {ownRoleChange && (
        <RecordConfirm
          busy={busy === `user-${session.user.username}`}
          confirmLabel={`Change my access to ${ROLE_NAMES[ownRoleChange]}`}
          eyebrow="Your own access"
          onCancel={() => { if (!busy) setOwnRoleChange(null); }}
          onConfirm={() => void confirmOwnRoleChange()}
          title="Remove your own administrator access?"
        >
          <p>{`You are signed in as ${session.user.username}. Changing your own access level to ${ROLE_NAMES[ownRoleChange]} takes away your administrator access as soon as it is saved: you will no longer be able to open Settings, manage accounts or change this back yourself. Another administrator would have to restore it.`}</p>
        </RecordConfirm>
      )}
      {revokeToken && (
        <RecordConfirm
          busy={busy === `api-${revokeToken.id}`}
          confirmLabel="Revoke connection"
          error={dialogError}
          eyebrow="Can't be undone"
          onCancel={() => { if (!busy) { setDialogError(""); setRevokeToken(null); } }}
          onConfirm={() => void revokeApiToken(revokeToken.id)}
          title="Revoke API connection?"
        >
          <p>{`${revokeToken.label} (${revokeToken.prefix}…) will stop working immediately. Any app using this key will need a new one.`}</p>
        </RecordConfirm>
      )}
    </div>
  );
}
