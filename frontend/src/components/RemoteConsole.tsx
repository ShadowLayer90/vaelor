import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest, downloadApiFile } from "../lib/api";
import { useDialogFocus } from "../hooks/useDialogFocus";
import type { Session } from "../types";
import { Icon } from "./Icon";
import { ConsoleLadder, consoleSessionAvailable } from "./ConsoleLadder";
import { ConsoleReadinessPanel } from "./ConsoleReadinessPanel";
import { PhysicalKvmStage, physicalKvmState, type KvmCapabilities } from "./PhysicalKvmStage";
import { StatusPill } from "./StatusPill";
import { Button, Input, Notice } from "./ui";
import { destinations } from "../lib/destinations";
import { sessionStateLabel, type RemoteSessionState } from "./remoteSessionState";

interface RemoteApp {
  id: string;
  name: string;
  image: string;
  running: boolean;
  capabilities: { remote_desktop: boolean };
}

interface RemoteTransport {
  available: boolean;
  port: number;
  detail?: string;
  setup_supported?: boolean;
}

/**
 * What the appliance says it is serving, so the owner can check the warning
 * their client shows instead of accepting a certificate they cannot verify.
 * `algorithm` is empty when the appliance could not name the digest, and
 * `detail` carries the reason when there is no fingerprint to show.
 */
interface RdpCertificate {
  fingerprint: string;
  algorithm?: string;
  detail?: string;
}

interface HostDesktop {
  available: boolean;
  port: number;
  address: string;
  name: string;
  kind: string;
  detail: string;
  rdp: RemoteTransport & { certificate?: RdpCertificate };
  browser_vnc: RemoteTransport;
  preferred_access?: "rdp" | "console";
  desktop?: {
    available: boolean;
    display_manager: boolean;
    graphical_target: boolean;
    session: boolean;
    detail: string;
  };
  console?: RemoteTransport & { kind?: string };
  os?: { id: string; name: string; support_level: string; support_label: string };
}

const commissioningFallback = [
  {
    id: "capture", complete: false,
    title: "Connect a supported USB HDMI capture adapter",
    detail: "Vaelor will detect it automatically; no Linux commands are required.",
  },
  {
    id: "hid", complete: false,
    title: "Commission isolated keyboard and mouse emulation",
    detail: "Vaelor will verify the USB device controller before enabling input.",
  },
  {
    id: "atx", complete: false,
    title: "Connect isolated ATX power leads",
    detail: "Optional; enables audited target power and reset controls.",
  },
];

function generateRdpPassword() {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%";
  const values = crypto.getRandomValues(new Uint8Array(22));
  return Array.from(values, (value) => alphabet[value % alphabet.length]).join("");
}

export function RemoteConsole({ session, onBack }: { session: Session; onBack?: () => void }) {
  const [capability, setCapability] = useState<KvmCapabilities | null>(null);
  /**
   * "The check did not answer" and "the hardware is not fitted" are different
   * facts, and the readiness list must not render the second when it means the
   * first.
   */
  const [capabilityFailed, setCapabilityFailed] = useState(false);
  const [apps, setApps] = useState<RemoteApp[]>([]);
  const [hostDesktop, setHostDesktop] = useState<HostDesktop | null>(null);
  const [remoteUrl, setRemoteUrl] = useState("");
  const [remoteSessionId, setRemoteSessionId] = useState("");
  const [remoteName, setRemoteName] = useState("");
  const [rdpUsername, setRdpUsername] = useState("vaelor");
  const [rdpPassword, setRdpPassword] = useState("");
  const [credentialsSaved, setCredentialsSaved] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState("");
  const [setupRequested, setSetupRequested] = useState(false);
  const [remoteSessionState, setRemoteSessionState] = useState<RemoteSessionState>("ended");
  const [sessionTarget, setSessionTarget] = useState<"host" | { app: RemoteApp } | null>(null);
  /* Rendered inside the session overlay — see `endDesktopSession`. */
  const [endError, setEndError] = useState("");
  const remoteDialogRef = useRef<HTMLDivElement>(null);
  const remoteCloseRef = useRef<HTMLButtonElement>(null);
  const remoteOpenerRef = useRef<HTMLElement | null>(null);
  useDialogFocus({
    active: Boolean(remoteUrl),
    containerRef: remoteDialogRef,
    initialFocusRef: remoteCloseRef,
    // Escape stops watching. It must never be the key that kills a desktop.
    onEscape: () => stopViewingRemoteSession(),
  });
  const canControl = session.user.role !== "viewer";
  const canAdminister = session.user.role === "administrator";
  const rdpReady = Boolean(hostDesktop?.rdp?.available ?? hostDesktop?.available);
  const desktopReady = hostDesktop?.desktop?.available !== false;
  const consoleFallback = hostDesktop?.preferred_access === "console" || !desktopReady;
  const consoleReady = Boolean(hostDesktop?.console?.available);
  const browserReady = Boolean(hostDesktop?.browser_vnc?.available);
  const browserSetupSupported = Boolean(hostDesktop?.browser_vnc?.setup_supported);
  const kvmState = physicalKvmState(capability, setupRequested);
  /*
   * The header pill is a claim about what this page can do, so it reads the
   * ladder rather than the older `console_ready` summary. "Hardware KVM ready"
   * beside three rows that all say "not available" was the same class of
   * mismatch the ladder exists to remove.
   */
  const kvmReady = consoleSessionAvailable(capability?.ladder);
  const certificate = hostDesktop?.rdp?.certificate;
  const rdpAddress = `${hostDesktop?.address || window.location.hostname}:${hostDesktop?.rdp?.port || 3389}`;
  const hostOsName = hostDesktop?.os?.name || "Linux host";
  const remoteLoginName = hostDesktop?.name || `${hostOsName} Remote Desktop`;
  const sshCommand = `ssh <linux-user>@${hostDesktop?.address || window.location.hostname}`;
  const rdpUsernameValid = /^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$/.test(rdpUsername);
  // #149: name only the rule the current value breaks — a compound message
  // that recites every rule tells the reader nothing about their input.
  const rdpUsernameError = !rdpUsername
    ? undefined
    : rdpUsername.length < 3
      ? "RDP user names need at least 3 characters."
      : rdpUsername.length > 64
        ? "RDP user names can have at most 64 characters."
        : !/^[A-Za-z0-9]/.test(rdpUsername)
          ? "Start the RDP user name with a letter or number."
          : !rdpUsernameValid
            ? "Use only letters, numbers, dots, dashes, or underscores."
            : undefined;

  const refresh = useCallback(async () => {
    setMessage("");
    const [capabilityResult, managedResult, hostResult] = await Promise.allSettled([
      apiRequest<KvmCapabilities>("/kvm/capabilities"),
      apiRequest<{ apps: RemoteApp[] }>("/managed"),
      apiRequest<HostDesktop>("/host/remote-desktop"),
    ]);
    if (capabilityResult.status === "fulfilled") {
      setCapability(capabilityResult.value);
      setCapabilityFailed(false);
    } else {
      setCapabilityFailed(true);
    }
    if (managedResult.status === "fulfilled") {
      setApps(managedResult.value.apps.filter((app) => app.capabilities.remote_desktop));
    }
    if (hostResult.status === "fulfilled") setHostDesktop(hostResult.value);
    if (
      capabilityResult.status === "rejected"
      || managedResult.status === "rejected"
      || hostResult.status === "rejected"
    ) {
      setMessage("Some remote-access checks did not respond. Refresh the status and try again.");
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    if (!setupRequested) return;
    const timer = window.setInterval(() => {
      void refresh();
    }, 5000);
    if (browserReady) setSetupRequested(false);
    return () => window.clearInterval(timer);
  }, [browserReady, refresh, setupRequested]);

  const control = async (action: "acquire" | "release") => {
    setBusy(action);
    setMessage("");
    try {
      await apiRequest("/kvm/control", { method: action === "acquire" ? "POST" : "DELETE", body: "{}" }, session.csrf_token);
      setMessage(action === "acquire" ? "You now own keyboard and mouse control." : "Keyboard and mouse control released.");
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Control ownership could not be changed.");
    } finally {
      setBusy("");
    }
  };

  const openDesktop = async (app: RemoteApp) => {
    if (!remoteUrl && document.activeElement instanceof HTMLElement) {
      remoteOpenerRef.current = document.activeElement;
    }
    setBusy(`app-${app.id}`);
    setSessionTarget({ app });
    setRemoteSessionState("connecting");
    setMessage("");
    try {
      const result = await apiRequest<{ url: string; session_id: string }>(
        `/managed/apps/${app.id}/remote-desktop`,
        { method: "POST", body: "{}" },
        session.csrf_token,
      );
      setRemoteName(app.name);
      setRemoteUrl(result.url);
      setRemoteSessionId(result.session_id);
      setRemoteSessionState("connecting");
    } catch (error) {
      setRemoteSessionState("failed");
      setMessage(error instanceof Error ? error.message : "Remote Desktop could not start.");
    } finally {
      setBusy("");
    }
  };

  const openHostBrowserDesktop = async () => {
    if (!remoteUrl && document.activeElement instanceof HTMLElement) {
      remoteOpenerRef.current = document.activeElement;
    }
    setBusy("host-browser");
    setSessionTarget("host");
    setRemoteSessionState("connecting");
    setMessage("");
    try {
      const result = await apiRequest<{ url: string; session_id: string }>(
        "/host/remote-desktop/browser-session",
        { method: "POST", body: "{}" },
        session.csrf_token,
      );
      setRemoteName(`${hostOsName} browser desktop`);
      setRemoteUrl(result.url);
      setRemoteSessionId(result.session_id);
      setRemoteSessionState("connecting");
    } catch (error) {
      setRemoteSessionState("failed");
      const detail = error instanceof Error ? error.message : "The browser desktop could not start.";
      setMessage(`${detail} Refresh the status or try opening it again.`);
    } finally {
      setBusy("");
    }
  };

  const retrySession = () => {
    if (sessionTarget === "host") void openHostBrowserDesktop();
    else if (sessionTarget) void openDesktop(sessionTarget.app);
  };

  /*
   * **This stops watching. It does not stop the desktop**, and the label now
   * says so. It used to read "Close session" while doing only this — clearing
   * three pieces of React state — so an owner whose desktop had locked itself
   * pressed it, reopened, and landed straight back in the same broken
   * session. The only way out was SSH, which is not a way out an owner has.
   */
  const stopViewingRemoteSession = () => {
    setRemoteUrl("");
    setRemoteSessionId("");
    setRemoteSessionState("ended");
    setMessage("Stopped viewing. The desktop is still running on the appliance.");
    window.requestAnimationFrame(() => {
      if (remoteOpenerRef.current?.isConnected) remoteOpenerRef.current.focus();
    });
  };

  /*
   * **An end failure belongs to the session it happened to, and dies with
   * it.** `endError` was cleared only when "End desktop" was pressed again,
   * so a failed end, then "Stop viewing", then a fresh open re-mounted the
   * overlay with the previous session's warning sitting over a session nobody
   * had tried to end — a reported failure that had stopped being true (#191).
   * Keyed on the session id rather than cleared in each handler, so a future
   * path that changes the viewed session cannot forget to do it.
   */
  useEffect(() => { setEndError(""); }, [remoteSessionId]);

  /* Ends the desktop on the appliance, so the next one starts clean. */
  const endDesktopSession = async () => {
    setBusy("end-desktop");
    setEndError("");
    try {
      const result = await apiRequest<{
        ended?: boolean; detail?: string; screen_lock_disabled?: boolean;
        desktop_listening?: boolean;
      }>("/remote-desktop/browser-sessions/end", {
        method: "POST",
      }, session.csrf_token);
      setRemoteUrl("");
      setRemoteSessionId("");
      setRemoteSessionState("ended");
      /*
       * **Report what the appliance said, not what the button hoped.** The
       * broker used to answer `ended: true` for a machine with no browser
       * desktop at all — creating a system account on the way — and this
       * message repeated the claim. Both halves are fixed: the appliance
       * tells the truth, and the screen stops overwriting it.
       *
       * **"Opening it again starts a fresh one" is a claim about the
       * appliance, and it used to be printed unconditionally** — including
       * when the restart had failed and nothing was listening on 5901, which
       * the reply said in `service_restarted` and this screen ignored.
       *
       * The appliance's readings decide, and `detail` only supplies the
       * words. Written the other way round — `detail` first, the readings as
       * a fallback — the two disagree about which is in charge, and a reply
       * carrying a reason makes the readings unreachable, so the branch that
       * withholds the promise stops being exercised while still looking like
       * it guards something.
       */
      const confirmed = result?.desktop_listening !== false
        && result?.screen_lock_disabled !== false;
      setMessage(
        result?.ended === false
          ? result.detail || "There was no desktop session on the appliance to end."
          : confirmed
            ? "The desktop session on the appliance was ended. Opening it again starts a fresh one."
            : result?.detail
              || "The desktop session on the appliance was ended, but the appliance did not confirm it is ready to open again.",
      );
    } catch (error) {
      /*
       * **Into the overlay, not behind it.** `setMessage` renders on the page
       * underneath this modal, so the first version of this reported its
       * failure somewhere the owner could not see while the session sat
       * unchanged in front of them — indistinguishable from a button that
       * does nothing, which is exactly how it was reported.
       */
      setEndError(
        error instanceof Error
          ? error.message
          : "The desktop session could not be ended.",
      );
    } finally {
      setBusy("");
      window.requestAnimationFrame(() => {
        if (remoteOpenerRef.current?.isConnected) remoteOpenerRef.current.focus();
      });
    }
  };

  useEffect(() => {
    if (!remoteUrl || !remoteSessionId) return;
    let active = true;
    let timer = 0;
    const check = async () => {
      try {
        const result = await apiRequest<{ state: RemoteSessionState; message: string }>(
          `/remote-desktop/browser-sessions/${encodeURIComponent(remoteSessionId)}`,
          { cache: "no-store" },
        );
        if (!active) return;
        setRemoteSessionState(result.state);
        if (result.state === "connecting") timer = window.setTimeout(() => void check(), 750);
      } catch {
        if (active) setRemoteSessionState("failed");
      }
    };
    void check();
    return () => { active = false; window.clearTimeout(timer); };
  }, [remoteSessionId, remoteUrl]);

  const configureRdp = async () => {
    setBusy("rdp-setup");
    setMessage("");
    try {
      await apiRequest(
        "/host/remote-desktop/rdp",
        {
          method: "POST",
          body: JSON.stringify({ username: rdpUsername, password: rdpPassword }),
        },
        session.csrf_token,
      );
      setCredentialsSaved(true);
      setMessage(`${remoteLoginName} is ready. Copy the saved credentials below, then download the connection profile.`);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : `${remoteLoginName} could not be configured.`);
    } finally {
      setBusy("");
    }
  };

  const downloadRdpProfile = async () => {
    setBusy("rdp-download");
    setMessage("");
    try {
      await downloadApiFile(
        "/host/remote-desktop/profile?mode=responsive",
        "vaelor-responsive-remote-login.rdp",
      );
      setMessage("Optimized RDP profile downloaded at 1600 × 900 with audio disabled for a smoother remote session.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "The RDP profile could not be downloaded.");
    } finally {
      setBusy("");
    }
  };

  const copyRdpCredentials = async () => {
    try {
      await navigator.clipboard.writeText(`Username: ${rdpUsername}\nPassword: ${rdpPassword}`);
      setMessage("Saved RDP username and password copied.");
    } catch {
      setMessage("Clipboard access was blocked. Use the visible username and password fields.");
    }
  };

  const disableRdp = async () => {
    setBusy("rdp-disable");
    setMessage("");
    try {
      await apiRequest(
        "/host/remote-desktop/rdp",
        { method: "DELETE", body: "{}" },
        session.csrf_token,
      );
      setMessage(`${remoteLoginName} is disabled.`);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : `${remoteLoginName} could not be disabled.`);
    } finally {
      setBusy("");
    }
  };

  const enableBrowserDesktop = async () => {
    setBusy("browser-setup");
    setMessage("");
    try {
      await apiRequest(
        "/jobs",
        {
          method: "POST",
          body: JSON.stringify({
            type: "host.vnc.enable",
            payload: { confirm: "enable-host-vnc" },
          }),
        },
        session.csrf_token,
      );
      setSetupRequested(true);
      setMessage("Optional browser desktop setup is running. This page will detect it automatically.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "The browser desktop setup could not start.");
    } finally {
      setBusy("");
    }
  };

  return (
    <div className="console-page">
      <div className="page-heading">
        <div>
          <h1>{destinations.kvm.name}</h1>
          <p>Connect to {hostOsName} using the remote-access services available on this OS, open app desktops, or commission the optional physical KVM path.</p>
        </div>
        <div className="workspace-route-actions">
          {onBack && <Button onClick={onBack} type="button" variant="quiet">Back to overview</Button>}
          <Button disabled={Boolean(busy)} onClick={() => void refresh()} type="button" variant="quiet">Reload</Button>
          <StatusPill
            label={
              consoleFallback
                ? (consoleReady ? "Console ready" : "Console unavailable")
                : rdpReady
                  ? "RDP ready"
                  : browserReady
                    ? "Remote access available"
                    : kvmState === "ready"
                      ? "Hardware KVM ready"
                      : hostDesktop?.rdp.setup_supported
                        ? "RDP setup available"
                        : "Remote access setup available"
            }
            status={rdpReady || browserReady || kvmReady || (consoleFallback && consoleReady) ? "healthy" : "neutral"}
          />
        </div>
      </div>

      {message && (
        <Notice severity="info">
          <Icon name="shield" />
          <span>{message}</span>
          <Button onClick={() => void refresh()} variant="quiet">Reload</Button>
        </Notice>
      )}

      <section className={`host-access ${rdpReady ? "host-access--ready" : ""}`} aria-labelledby="host-access-title">
        <div className="host-access__identity">
          <span className="host-access__icon"><Icon name={consoleFallback ? "terminal" : "display"} size={30} /></span>
          <div>
            <small>THIS VAELOR NODE · {hostOsName.toUpperCase()}</small>
            <h2 id="host-access-title">{consoleFallback ? `${hostOsName} console fallback` : remoteLoginName}</h2>
            <p>{consoleFallback ? hostDesktop?.desktop?.detail : hostDesktop?.detail || "Checking GNOME Remote Desktop…"}</p>
          </div>
          <StatusPill
            label={consoleFallback ? (consoleReady ? "SSH console online" : "Console offline") : rdpReady ? "RDP online" : browserReady ? "Browser desktop ready" : "Not configured"}
            status={consoleFallback && consoleReady || rdpReady || browserReady ? "healthy" : "neutral"}
          />
        </div>

        <div className="host-access__connection">
          <div>
            <small>{consoleFallback ? "CONSOLE COMMAND" : "RDP ADDRESS"}</small>
            <strong>{consoleFallback ? sshCommand : rdpAddress}</strong>
            {/*
              * "Configure dedicated credentials to enable it" — with no
              * subject. It sits under RDP ADDRESS and is about RDP, but a
              * tester read it beside a `Browser desktop ready` pill and an
              * enabled "Open browser desktop" and reported the screen as
              * contradicting itself. Two paths are described here and only one
              * of them needs setting up, so the sentence names which.
              */}
            <span>{consoleFallback ? "The graphical desktop is unavailable, so console access is the safe default." : rdpReady ? `Encrypted native ${hostOsName} remote login` : hostDesktop?.rdp.setup_supported ? "Native RDP login is not set up. Configure dedicated RDP credentials to enable it." : "Native RDP setup is unavailable on this OS"}</span>
          </div>
          <div className="host-access__actions">
            {browserReady && (
              <Button variant="primary"

                disabled={!canControl || Boolean(busy)}
                onClick={() => void openHostBrowserDesktop()}
                type="button"
              >
                {busy === "host-browser" ? "Opening…" : "Open browser desktop"}
              </Button>
            )}
            {consoleFallback ? (
              <Button variant="primary"

                disabled={!consoleReady}
                onClick={() => void navigator.clipboard.writeText(sshCommand).then(
                  () => setMessage("Console command copied. Replace <linux-user> with your Ubuntu account name."),
                  () => setMessage(`Console command: ${sshCommand}`),
                )}
                type="button"
              >
                Copy console command
              </Button>
            ) : rdpReady ? (
              <Button variant="primary"

                disabled={Boolean(busy)}
                onClick={() => void downloadRdpProfile()}
                type="button"
              >
                {busy === "rdp-download" ? "Downloading…" : "Download optimized RDP profile"}
              </Button>
            ) : hostDesktop?.rdp.setup_supported ? (
              <a className="ui-button ui-button--quiet" href="#rdp-settings">Set up Remote Login</a>
            ) : (
              <span className="host-access__unavailable">Use browser desktop or physical KVM</span>
            )}
            {!consoleFallback && <Button variant="quiet"

              onClick={() => void navigator.clipboard.writeText(rdpAddress).then(
                () => setMessage("RDP address copied."),
                () => setMessage(`RDP address: ${rdpAddress}`),
              )}
              type="button"
            >
              Copy address
            </Button>}
          </div>
          {/*
            * **The owner is asked to trust something they cannot check.** The
            * appliance signs its own RDP certificate, so every client shows a
            * warning; without this the only available answer is to click Yes
            * and hope. The appliance knows what it is serving, so it says so,
            * and the accept becomes a comparison. Shown only beside the
            * profile download, which is the moment it is needed.
            *
            * When the fingerprint could not be read, that reason is printed
            * rather than nothing — "no fingerprint" and "we could not ask"
            * must not look the same.
            */}
          {rdpReady && !consoleFallback && (certificate?.fingerprint || certificate?.detail) && (
            <div className="host-access__certificate">
              <small>
                TLS FINGERPRINT ON THIS APPLIANCE
                {certificate.algorithm ? ` · ${certificate.algorithm}` : ""}
              </small>
              {certificate.fingerprint
                ? <code>{certificate.fingerprint}</code>
                : <strong>Not available</strong>}
              <span>
                {certificate.fingerprint
                  ? "Vaelor signs this certificate itself, so your client will warn you. Check this value against the one the warning shows before you accept it."
                  : certificate.detail}
              </span>
            </div>
          )}
        </div>

        {!consoleFallback && hostDesktop?.rdp.setup_supported !== false && <details className="host-access__settings" id="rdp-settings" open={!rdpReady}>
          <summary>{rdpReady ? "Remote Login settings" : "Set up Remote Login"}</summary>
          <div className="host-access__settings-body">
            <div>
              <h3>{rdpReady ? "Rotate dedicated RDP credentials" : "Create dedicated RDP credentials"}</h3>
              <p>These credentials are only for Remote Login. They do not change your {hostOsName} or Vaelor password.</p>
            </div>
            <form
              className="rdp-credential-form"
              onSubmit={(event) => {
                // A real form so Enter submits from either field and password
                // managers recognise a credential-change flow.
                event.preventDefault();
                if (!canAdminister || !rdpUsernameValid || rdpPassword.length < 12 || busy) return;
                void configureRdp();
              }}
            >
              <Input
                aria-invalid={Boolean(rdpUsername) && !rdpUsernameValid}
                autoComplete="username"
                disabled={!canAdminister || Boolean(busy)}
                error={rdpUsernameError}
                label="RDP user name"
                maxLength={64}
                minLength={3}
                onChange={(event) => {
                  setRdpUsername(event.target.value);
                  setCredentialsSaved(false);
                }}
                value={rdpUsername}
              />
              <div className="rdp-form-field">
                <div className="rdp-password-field">
                  <Input
                    autoComplete="new-password"
                    disabled={!canAdminister || Boolean(busy)}
                    id="rdp-password"
                    label="Dedicated RDP password"
                    maxLength={128}
                    minLength={12}
                    onChange={(event) => {
                      setRdpPassword(event.target.value);
                      setCredentialsSaved(false);
                    }}
                    type={showPassword ? "text" : "password"}
                    value={rdpPassword}
                  />
                  <Button
                    aria-label={showPassword ? "Hide password" : "Show password"}
                    aria-pressed={showPassword}
                    onClick={() => setShowPassword((value) => !value)}
                    variant="quiet"
                  >
                    {showPassword ? "Hide password" : "Show password"}
                  </Button>
                </div>
              </div>
              <div className="rdp-credential-form__actions">
                <Button variant="quiet"

                  disabled={!canAdminister || Boolean(busy)}
                  onClick={() => {
                    setRdpPassword(generateRdpPassword());
                    setShowPassword(true);
                    setCredentialsSaved(false);
                    setMessage("A new password was generated but is not saved yet. Select “Save RDP credentials” next.");
                  }}
                  type="button"
                >
                  Generate new password
                </Button>
                <Button variant="primary"

                  disabled={!canAdminister || !rdpUsernameValid || rdpPassword.length < 12 || Boolean(busy)}
                  /* #149: an empty user name disabled this button with no
                     message anywhere on the form. A disabled primary action
                     states what is missing. */
                  disabledReason={!canAdminister
                    ? "Administrator access is required to change RDP credentials."
                    : !rdpUsername
                      ? "Enter an RDP user name to continue."
                      : !rdpUsernameValid
                        ? "Fix the RDP user name to continue."
                        : rdpPassword.length < 12
                          ? "Enter a dedicated password of at least 12 characters to continue."
                          : undefined}
                  type="submit"
                >
                  {busy === "rdp-setup" ? "Saving and verifying…" : rdpReady ? "Save RDP credentials" : "Enable and save credentials"}
                </Button>
                {credentialsSaved && (
                  <Button variant="quiet"

                    disabled={Boolean(busy)}
                    onClick={() => void copyRdpCredentials()}
                    type="button"
                  >
                    Copy saved credentials
                  </Button>
                )}
                {rdpReady && (
                  <Button variant="danger"

                    disabled={!canAdminister || Boolean(busy)}
                    onClick={() => void disableRdp()}
                    type="button"
                  >
                    {busy === "rdp-disable" ? "Disabling…" : "Disable RDP"}
                  </Button>
                )}
              </div>
              <small className={credentialsSaved ? "rdp-credential-state rdp-credential-state--saved" : "rdp-credential-state"}>
                {credentialsSaved
                  ? "✓ These exact credentials were saved and verified by GNOME Remote Login."
                  : "Changes in these fields are not active until you save them."}
              </small>
              {!canAdminister && <small>Administrator access is required to change Remote Login.</small>}
            </form>
          </div>
        </details>}

        {/*
          * `open={!browserReady}` because "Install browser desktop" is the
          * only control that commissions the desktop, and a plain <details>
          * kept it collapsed by default - so the owner reached this page,
          * never opened the disclosure, and the one button that would install
          * the desktop stayed hidden. A live tester read that as a dead
          * control: hidden, it queues no job because it is never clickable.
          * The RDP settings disclosure above already does this with
          * `open={!rdpReady}`; this is the same rule for the same reason -
          * expand while there is an action to take, collapse to "details" once
          * the desktop is ready.
          */}
        <details className="host-access__browser" open={!browserReady}>
          <summary>{browserReady ? "Browser desktop details" : `Optional: open ${hostOsName} inside this browser`}</summary>
          <div>
            <p>This installs an isolated TigerVNC session on loopback. Vaelor protects each browser connection with a short-lived, one-use ticket.</p>
            {browserReady ? (
              <p role="status">Ready now. Use “Open browser desktop” above to create a protected one-use session.</p>
            ) : browserSetupSupported ? (
              <Button

                disabled={!canAdminister || Boolean(busy) || setupRequested}
                onClick={() => void enableBrowserDesktop()}
                type="button"
              >
                {setupRequested ? "Setup in progress…" : busy === "browser-setup" ? "Starting…" : "Install browser desktop"}
              </Button>
            ) : (
              <div>
                <p>{hostDesktop?.browser_vnc?.detail || "Browser desktop setup is unavailable on this operating system. Use RDP or the SSH console above."}</p>
                <Button onClick={() => void refresh()} variant="quiet">
                  Recheck browser desktop
                </Button>
              </div>
            )}
          </div>
        </details>
      </section>

      <PhysicalKvmStage
        busy={busy}
        canControl={canControl}
        capability={capability}
        onControl={(action) => void control(action)}
        state={kvmState}
        username={session.user.username}
      />

      <div className="console-grid">
        {/*
          * Three siblings, discovered independently and never inheriting each
          * other's rung: seeing the screen, driving the keyboard, and powering
          * the machine out of band fail for different reasons and are fixed by
          * different people. Each row names its own remediation owner, which is
          * what stops out-of-band power reading as a promise of a screen.
          */}
        <ConsoleLadder rows={capability?.ladder} />

        {/*
          * Moved here from Home, whole and on both machine classes. Its three
          * rows are a commissioning checklist, and this is where commissioning
          * is the subject.
          */}
        <ConsoleReadinessPanel capability={capability} failed={capabilityFailed} />

        <section className="data-panel" id="physical-kvm-setup">
          <div className="panel-heading">
            <div><h2>Physical KVM setup</h2><p>Hardware discovery checks each requirement automatically.</p></div>
            <Icon name="usb" />
          </div>
          <ol className="commission-list">
            {(capability?.commissioning?.length ? capability.commissioning : commissioningFallback).map((step, index) => (
              <li className={step.complete ? "commission-list__complete" : ""} key={step.id}>
                <span>{step.complete ? "✓" : index + 1}</span>
                <div><strong>{step.title}</strong><small>{step.detail}</small></div>
              </li>
            ))}
          </ol>
          <div className="tool-explainer">
            <Icon name="shield" />
            <span><strong>Protected by discovery</strong><small>Streaming and input stay disabled until real capture and isolated HID hardware pass verification.</small></span>
          </div>
        </section>

        <section className="data-panel">
          <div className="panel-heading">
            <div><h2>App remote desktops</h2><p>One-use browser sessions for installed apps that expose VNC.</p></div>
            <Icon name="display" />
          </div>
          {apps.length ? (
            <div className="remote-app-list">
              {apps.map((app) => (
                <article key={app.id}>
                  <span><Icon name="display" /></span>
                  <div><strong>{app.name}</strong><small>{app.image} · {app.running ? "Running" : "Stopped"}</small></div>
                  <Button

                    disabled={!canControl || !app.running || Boolean(busy)}
                    onClick={() => void openDesktop(app)}
                    type="button"
                  >
                    {busy === `app-${app.id}` ? "Opening…" : "Open desktop"}
                  </Button>
                </article>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <Icon name="display" />
              <strong>No app desktops available</strong>
              <span>Compatible installed apps appear here automatically.</span>
            </div>
          )}
        </section>
      </div>

      {remoteSessionState === "failed" && !remoteUrl && (
        <Notice severity="danger">
          <Icon name="shield" />
          <span>The remote desktop session did not start. The page is still available; retry the session or choose another access path.</span>
          <Button disabled={Boolean(busy)} onClick={retrySession} type="button" variant="primary">Retry session</Button>
        </Notice>
      )}

      {remoteUrl && (
        <div className="remote-console-modal" ref={remoteDialogRef} role="dialog" aria-modal="true" aria-labelledby="remote-console-title">
          <section>
            <header>
              <div><span className="page-eyebrow">Protected one-use session</span><h2 id="remote-console-title">{remoteName}</h2></div>
              <div className="workspace-route-actions">
                <StatusPill label={sessionStateLabel(remoteSessionState)} status={remoteSessionState === "connected" ? "healthy" : remoteSessionState === "failed" ? "degraded" : "neutral"} />
                {/*
                  * Two controls, because they do different things and one
                  * label covering both is what stranded the owner. "Stop
                  * viewing" leaves the desktop running; "End desktop" is the
                  * recovery route when the session itself is wrong.
                  */}
                <Button ref={remoteCloseRef} onClick={stopViewingRemoteSession} type="button" variant="quiet">Stop viewing</Button>
                {canControl && sessionTarget === "host" && (
                  <Button disabled={busy === "end-desktop"} onClick={() => void endDesktopSession()} type="button" variant="quiet">
                    {busy === "end-desktop" ? "Ending…" : "End desktop"}
                  </Button>
                )}
              </div>
            </header>
            {endError && <Notice severity="warning" heading="The desktop session was not ended."><span>{endError}</span></Notice>}
            {remoteSessionState === "failed" && <Notice severity="info"><span>The visible desktop session could not be reached.</span><Button disabled={Boolean(busy)} onClick={retrySession} type="button" variant="primary">Retry session</Button></Notice>}
            <iframe
              allow="clipboard-read; clipboard-write; fullscreen"
              sandbox="allow-forms allow-same-origin allow-scripts"
              src={remoteUrl}
              title={`${remoteName} remote desktop`}
              onError={() => setRemoteSessionState("failed")}
            />
          </section>
        </div>
      )}
    </div>
  );
}
