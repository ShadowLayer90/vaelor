import type { FormEvent } from "react";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { SectionCard } from "./systemUi";
import { Button, EmptyState, Input, ListRow, Notice } from "./ui";

/*
 * Remote console's cards (VD-200, the Console and ConsoleRemoteLogin boards):
 * this machine's Remote Desktop, the browser desktop's details, and the app
 * desktops. RemoteConsole.tsx reads the appliance and owns every action; these
 * draw what it hands them.
 */

export type Outcome = { text: string; refused: boolean } | null;

/** Why a viewer's desktop buttons are locked, said beside each (S-Y4). */
const OPERATOR_REQUIRED = "Operator access is required to open a desktop session.";

/**
 * What the appliance says it is serving, so the owner can check the warning
 * their client shows instead of accepting a certificate they cannot verify.
 * `algorithm` is empty when the appliance could not name the digest, and
 * `detail` carries the reason when there is no fingerprint to show.
 */
export interface RdpCertificate {
  fingerprint: string;
  algorithm?: string;
  detail?: string;
}

export interface RemoteApp {
  id: string;
  name: string;
  image: string;
  running: boolean;
  capabilities: { remote_desktop: boolean };
}

export interface RemoteDesktopFacts {
  /** "ready" RDP, a browser desktop, or the SSH console where there is no desktop. */
  rdpReady: boolean;
  rdpSetupSupported: boolean;
  browserReady: boolean;
  /** The appliance can install the browser desktop here (it is not installed yet when `browserReady` is false). */
  browserSetupSupported: boolean;
  consoleFallback: boolean;
  consoleReady: boolean;
  hostOsName: string;
  remoteLoginName: string;
  rdpAddress: string;
  sshCommand: string;
  certificate?: RdpCertificate;
  /** The appliance's own sentence about the desktop, when it gave one. */
  desktopDetail?: string;
  /**
   * `unread`: `/host/remote-desktop` has not answered yet, so every fact above
   * is a default. `stale`: the last read failed and these are the previous
   * answer's facts. Either way nothing here is green (LESSONS 1, S-Y3).
   */
  reading?: "unread" | "stale";
}

export function remoteDesktopPill(facts: RemoteDesktopFacts): { label: string; tone: "success" | "neutral"; reading?: "unread" | "stale" } {
  if (facts.reading === "unread") return { label: "Not read", tone: "neutral", reading: "unread" };
  if (facts.reading === "stale") return { label: "Old reading", tone: "neutral", reading: "stale" };
  if (facts.consoleFallback) return facts.consoleReady
    ? { label: "SSH console online", tone: "success" as const }
    : { label: "Console offline", tone: "neutral" as const };
  if (facts.rdpReady) return { label: "RDP online", tone: "success" as const };
  if (facts.browserReady) return { label: "Browser desktop ready", tone: "success" as const };
  return { label: "Not configured", tone: "neutral" as const };
}

interface CredentialForm {
  canAdminister: boolean;
  busy: string;
  username: string;
  usernameError?: string;
  usernameValid: boolean;
  password: string;
  showPassword: boolean;
  credentialsSaved: boolean;
  outcome: Outcome;
  onUsername: (value: string) => void;
  onPassword: (value: string) => void;
  onToggleShow: () => void;
  onGenerate: () => void;
  onSubmit: () => void;
  onCopySaved: () => void;
  onDisable: () => void;
}

/**
 * This machine's Remote Desktop card. `summary` is the Console board's (the
 * address, its three actions, the fingerprint); `settings` is the
 * ConsoleRemoteLogin board's, the same card with the dedicated credentials
 * beside it.
 */
export function RemoteDesktopCard({
  busy,
  canControl,
  facts,
  form,
  onCopyAddress,
  onCopyConsole,
  onDownloadProfile,
  onOpenBrowser,
  onOpenSettings,
  outcome,
  variant,
}: {
  busy: string;
  canControl: boolean;
  facts: RemoteDesktopFacts;
  form?: CredentialForm;
  onCopyAddress: () => void;
  onCopyConsole: () => void;
  onDownloadProfile: () => void;
  onOpenBrowser: () => void;
  onOpenSettings: () => void;
  /** The last download or copy, said in this card. */
  outcome: Outcome;
  variant: "summary" | "settings";
}) {
  const pill = remoteDesktopPill(facts);
  const settings = variant === "settings";
  const { certificate } = facts;
  const unread = facts.reading === "unread";
  const connectionSentence = unread
    ? "Vaelor has not read this machine's remote access yet."
    : facts.consoleFallback
    ? "The graphical desktop is unavailable, so console access is the safe default."
    : facts.rdpReady
      ? settings
        ? `Encrypted native ${facts.hostOsName} remote login`
        : "Use any Remote Desktop app. Sign in with the dedicated RDP account, not your console account."
      // Two paths are described here and only one needs setting up, so the
      // sentence names which (a bare "Configure dedicated credentials" read as
      // a limit on the ready browser desktop beside it).
      : facts.rdpSetupSupported
        ? "Native RDP login is not set up. Configure dedicated RDP credentials to enable it."
        : "Native RDP setup is unavailable on this OS";
  const showSettingsEntry = !settings && !unread && !facts.consoleFallback && facts.rdpSetupSupported;
  /*
   * LESSONS 19: the browser desktop's install lives on the Remote Login view,
   * so the summary must always lead there while it can be installed - also
   * where native RDP cannot be set up and on the console fallback, the two
   * hosts that need it most (Y1).
   */
  const showBrowserEntry = !settings && !unread && !showSettingsEntry && facts.browserSetupSupported && !facts.browserReady;

  const access = (
    <div className="console-access">
      <div className="console-address">
        {/*
          * LESSONS 8 / VD-200 system verify: before the host is read there is no
          * address to give. The fallback was this browser's own host and 3389
          * ("127.0.0.1:3389"), shown and copyable as the machine's. Say it is
          * not read; draw no address and no Copy until it is.
          */}
        {!unread && <span className="sys-eyebrow">{facts.consoleFallback ? "Console command" : "RDP address"}</span>}
        {!unread && <strong>{facts.consoleFallback ? facts.sshCommand : facts.rdpAddress}</strong>}
        <span>{connectionSentence}</span>
      </div>
      <div className="console-actions">
        {facts.consoleFallback ? (
          <Button disabled={!facts.consoleReady} onClick={onCopyConsole} type="button" variant="primary">Copy console command</Button>
        ) : facts.rdpReady ? (
          <Button disabled={Boolean(busy)} onClick={onDownloadProfile} type="button" variant="primary">
            {busy === "rdp-download" ? "Downloading…" : settings ? "Download optimized RDP profile" : "Download RDP profile"}
          </Button>
        ) : null}
        {facts.browserReady && (
          <Button
            disabled={!canControl || Boolean(busy)}
            disabledReason={!canControl ? OPERATOR_REQUIRED : undefined}
            onClick={onOpenBrowser}
            type="button"
            variant={facts.rdpReady || facts.consoleFallback ? "secondary" : "primary"}
          >
            {busy === "host-browser" ? "Opening…" : "Open browser desktop"}
          </Button>
        )}
        {!unread && !facts.consoleFallback && <Button onClick={onCopyAddress} type="button" variant="quiet">Copy address</Button>}
        {showSettingsEntry && (
          <Button onClick={onOpenSettings} type="button" variant={facts.rdpReady ? "quiet" : "secondary"}>
            {facts.rdpReady ? "Remote Login settings" : "Set up Remote Login"}
          </Button>
        )}
        {showBrowserEntry && (
          <Button onClick={onOpenSettings} type="button" variant={facts.consoleFallback ? "secondary" : "primary"}>
            Set up browser desktop
          </Button>
        )}
        {!unread && !facts.consoleFallback && !facts.rdpReady && !facts.rdpSetupSupported && !facts.browserReady && (
          <span className="sys-muted">Use browser desktop or physical KVM</span>
        )}
      </div>
      {outcome && <Notice severity={outcome.refused ? "danger" : "info"}>{outcome.text}</Notice>}
      {/*
        * The appliance signs its own RDP certificate, so every client warns;
        * the fingerprint turns the accept into a comparison. When it could not
        * be read, the reason is printed: "no fingerprint" and "we could not
        * ask" must not look the same.
        */}
      {facts.rdpReady && !facts.consoleFallback && (certificate?.fingerprint || certificate?.detail) && (
        <div className="console-fingerprint">
          <span className="sys-eyebrow">
            {settings ? "TLS fingerprint on this appliance" : "TLS fingerprint"}
            {certificate.algorithm ? ` · ${certificate.algorithm}` : ""}
          </span>
          {certificate.fingerprint ? <code>{certificate.fingerprint}</code> : <strong>Not available</strong>}
          <span className="sys-muted">
            {certificate.fingerprint
              ? settings
                ? "Vaelor signs this certificate itself, so your client will warn you. Check this value against the one the warning shows before you accept it."
                : "Compare this with the warning your Remote Desktop app shows the first time."
              : certificate.detail}
          </span>
        </div>
      )}
    </div>
  );

  return (
    <section aria-labelledby="host-access-title" className={settings ? "card ui-card sys-card console-desktop console-desktop--settings" : "card ui-card sys-card console-desktop"}>
      <header className="ui-card__header">
        <div className="ui-card__titles">
          <span className="sys-eyebrow">This machine · {facts.hostOsName}</span>
          <h2 id="host-access-title">
            {facts.consoleFallback ? `${facts.hostOsName} console fallback` : settings ? facts.remoteLoginName : "Remote Desktop"}
          </h2>
          {facts.consoleFallback && facts.desktopDetail && <p>{facts.desktopDetail}</p>}
        </div>
        <div className="ui-card__actions"><StatusPill label={pill.label} reading={pill.reading} tone={pill.tone} /></div>
      </header>
      {settings && form && !facts.consoleFallback && facts.rdpSetupSupported ? (
        <div className="console-desktop__split">
          <div className="ui-card__body">{access}</div>
          <RdpCredentials facts={facts} form={form} />
        </div>
      ) : <div className="ui-card__body">{access}</div>}
    </section>
  );
}

/** The dedicated Remote Login credentials (the ConsoleRemoteLogin board). */
function RdpCredentials({ facts, form }: { facts: RemoteDesktopFacts; form: CredentialForm }) {
  const tooShort = form.password.length < 12;
  /*
   * One visible sentence for every control a non-administrator cannot use,
   * each button pointing at it, rather than a bare greyed row (S-Y4).
   */
  const adminReasonId = "rdp-admin-required";
  const adminOnly = form.canAdminister ? undefined : adminReasonId;
  const submit = (event: FormEvent) => {
    // A real form, so Enter submits from either field and password managers
    // recognise a credential-change flow.
    event.preventDefault();
    if (!form.canAdminister || !form.usernameValid || tooShort || form.busy) return;
    form.onSubmit();
  };
  return (
    <div aria-labelledby="rdp-settings-title" className="console-credentials" id="rdp-settings" role="region" tabIndex={-1}>
      <div className="console-credentials__titles">
        <span className="sys-eyebrow">{facts.rdpReady ? "Remote Login settings" : "Set up Remote Login"}</span>
        <h3 id="rdp-settings-title">{facts.rdpReady ? "Rotate dedicated RDP credentials" : "Create dedicated RDP credentials"}</h3>
        <p>These credentials are only for Remote Login. They do not change your {facts.hostOsName} or Vaelor password.</p>
      </div>
      <form className="cns-form" onSubmit={submit}>
        <Input
          aria-invalid={Boolean(form.username) && !form.usernameValid}
          autoComplete="username"
          disabled={!form.canAdminister || Boolean(form.busy)}
          error={form.usernameError}
          label="RDP user name"
          maxLength={64}
          minLength={3}
          onChange={(event) => form.onUsername(event.target.value)}
          value={form.username}
        />
        <div className="cns-password">
          <Input
            autoComplete="new-password"
            disabled={!form.canAdminister || Boolean(form.busy)}
            id="rdp-password"
            label="Dedicated RDP password"
            maxLength={128}
            minLength={12}
            onChange={(event) => form.onPassword(event.target.value)}
            type={form.showPassword ? "text" : "password"}
            value={form.password}
          />
          <Button aria-pressed={form.showPassword} onClick={form.onToggleShow} variant="quiet">
            {form.showPassword ? "Hide password" : "Show password"}
          </Button>
        </div>
        {form.outcome && <Notice severity={form.outcome.refused ? "danger" : form.credentialsSaved ? "success" : "info"}>{form.outcome.text}</Notice>}
        <div className="cns-form__actions">
          <Button
            aria-describedby={adminOnly}
            disabled={!form.canAdminister || !form.usernameValid || tooShort || Boolean(form.busy)}
            /* #149: a disabled primary action states what is missing. */
            disabledReason={!form.canAdminister
              ? undefined
              : !form.username
                ? "Enter an RDP user name to continue."
                : !form.usernameValid
                  ? "Fix the RDP user name to continue."
                  : tooShort
                    ? "Enter a dedicated password of at least 12 characters to continue."
                    : undefined}
            type="submit"
            variant="primary"
          >
            {form.busy === "rdp-setup" ? "Saving and verifying…" : facts.rdpReady ? "Save RDP credentials" : "Enable and save credentials"}
          </Button>
          <Button aria-describedby={adminOnly} disabled={!form.canAdminister || Boolean(form.busy)} onClick={form.onGenerate} type="button" variant="quiet">
            Generate new password
          </Button>
          {form.credentialsSaved && (
            <Button disabled={Boolean(form.busy)} onClick={form.onCopySaved} type="button" variant="secondary">Copy saved credentials</Button>
          )}
          {facts.rdpReady && (
            <Button aria-describedby={adminOnly} disabled={!form.canAdminister || Boolean(form.busy)} onClick={form.onDisable} type="button" variant="danger">
              {form.busy === "rdp-disable" ? "Disabling…" : "Disable RDP"}
            </Button>
          )}
        </div>
        {!form.canAdminister && (
          <p className="ui-button__disabled-reason" id={adminReasonId}>
            Administrator access is required to change or disable RDP credentials.
          </p>
        )}
        <small className={form.credentialsSaved ? "cns-form__state cns-form__state--saved" : "cns-form__state"}>
          {form.credentialsSaved
            ? "✓ These exact credentials were saved and verified by GNOME Remote Login."
            : "Changes in these fields are not active until you save them."}
        </small>
      </form>
    </div>
  );
}

/**
 * The browser desktop's details (the ConsoleRemoteLogin board). Its install
 * control is never hidden in a collapsed disclosure: a live tester read the
 * hidden button as a dead one.
 */
export function BrowserDesktopCard({
  busy,
  canAdminister,
  detail,
  hostOsName,
  onInstall,
  onRecheck,
  outcome,
  ready,
  setupRequested,
  setupSupported,
}: {
  busy: string;
  canAdminister: boolean;
  detail?: string;
  hostOsName: string;
  onInstall: () => void;
  onRecheck: () => void;
  outcome: Outcome;
  ready: boolean;
  setupRequested: boolean;
  setupSupported: boolean;
}) {
  const pill = ready
    ? { label: "Browser desktop ready", tone: "success" as const }
    : setupRequested
      ? { label: "Setup running", tone: "info" as const }
      : setupSupported ? { label: "Optional", tone: "neutral" as const } : { label: "Unavailable", tone: "neutral" as const };
  return (
    <SectionCard
      actions={<StatusPill label={pill.label} tone={pill.tone} />}
      className="console-browser"
      description={`${hostOsName} inside this browser`}
      title={ready ? "Browser desktop details" : `Optional: open ${hostOsName} inside this browser`}
    >
      <div className="console-browser__body">
        <p>This installs an isolated TigerVNC session on loopback. Vaelor protects each browser connection with a short-lived, one-use ticket.</p>
        <div className="console-browser__state">
          {ready ? (
            <p role="status">Ready now. Use “Open browser desktop” above to create a protected one-use session.</p>
          ) : setupRequested ? (
            <>
              <Notice severity="info">Optional browser desktop setup is running. This page will detect it automatically.</Notice>
              <Button disabled type="button" variant="secondary">Setup in progress…</Button>
            </>
          ) : setupSupported ? (
            <Button
              disabled={!canAdminister || Boolean(busy)}
              disabledReason={!canAdminister ? "Administrator access is required to install the browser desktop." : undefined}
              onClick={onInstall}
              type="button"
              variant="secondary"
            >
              {busy === "browser-setup" ? "Starting…" : "Install browser desktop"}
            </Button>
          ) : (
            <>
              <p>{detail || "Browser desktop setup is unavailable on this operating system. Use RDP or the SSH console above."}</p>
              <Button onClick={onRecheck} type="button" variant="quiet">Recheck browser desktop</Button>
            </>
          )}
          {outcome?.refused && <Notice severity="danger">{outcome.text}</Notice>}
        </div>
      </div>
    </SectionCard>
  );
}

/** App desktops (the Console board): one-use browser sessions for installed apps that expose VNC. */
export function AppDesktopsCard({
  apps,
  busy,
  canControl,
  onOpen,
}: {
  apps: RemoteApp[];
  busy: string;
  canControl: boolean;
  onOpen: (app: RemoteApp) => void;
}) {
  return (
    <SectionCard
      actions={apps.length > 0 && <span>Compatible installed apps appear here automatically</span>}
      flush
      title="App desktops"
    >
      {apps.length ? apps.map((app) => (
        <ListRow
          detail={`${app.image} · ${app.running ? "Running" : "Stopped"}`}
          icon={<Icon name="apps" size={18} />}
          key={app.id}
          title={app.name}
          trailing={(
            <span className="console-app__actions">
              <StatusPill label={app.running ? "Running" : "Stopped"} tone={app.running ? "success" : "neutral"} />
              <Button
                disabled={!canControl || !app.running || Boolean(busy)}
                disabledReason={!canControl
                  ? OPERATOR_REQUIRED
                  : !app.running ? "Start the app to open its desktop." : undefined}
                onClick={() => onOpen(app)}
                type="button"
                variant="secondary"
              >
                {busy === `app-${app.id}` ? "Opening…" : "Open desktop"}
              </Button>
            </span>
          )}
        />
      )) : (
        <EmptyState
          icon={<Icon name="display" size={18} />}
          text="Compatible installed apps appear here automatically."
          title="No app desktops available"
        />
      )}
    </SectionCard>
  );
}
