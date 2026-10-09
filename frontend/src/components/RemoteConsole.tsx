import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest, downloadApiFile } from "../lib/api";
import { useDialogFocus } from "../hooks/useDialogFocus";
import type { Session } from "../types";
import { consoleSessionAvailable, ConsoleLadder } from "./ConsoleLadder";
import { ConsoleReadinessPanel } from "./ConsoleReadinessPanel";
import {
  AppDesktopsCard,
  BrowserDesktopCard,
  RemoteDesktopCard,
  type Outcome,
  type RdpCertificate,
  type RemoteApp,
  type RemoteDesktopFacts,
} from "./ConsoleRemoteDesktop";
import { ConsoleSessionDialog } from "./ConsoleSessionDialog";
import { KvmChecklistCard, KvmVideoCard, PhysicalKvmStage, physicalKvmState, type KvmCapabilities } from "./PhysicalKvmStage";
import { StatusPill } from "./StatusPill";
import { Button, Notice } from "./ui";
import { destinations } from "../lib/destinations";
import { TopbarPageActions, usePagePlace } from "../lib/topbarSlot";
import type { RemoteSessionState } from "./remoteSessionState";

interface RemoteTransport {
  available: boolean;
  port: number;
  detail?: string;
  setup_supported?: boolean;
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

/**
 * Remote console's three views (VD-200): the Console board's first view, the
 * ConsoleRemoteLogin board's Remote Login and the ConsoleKvm board's Physical
 * KVM. Each is a route of its own under `#/kvm`, so Back returns and a link
 * can name one; none of them is a fragment, which in this hash-routed console
 * would replace the route (W4d-D1).
 */
export type ConsoleView = "main" | "remote-login" | "hardware";

export function consoleViewFromHash(hash: string): ConsoleView {
  const view = hash.match(/^#\/[^/?]+\/(remote-login|hardware)(?:[/?].*)?$/)?.[1];
  return view === "remote-login" || view === "hardware" ? view : "main";
}

function hashForView(view: ConsoleView) {
  return view === "main" ? "#/kvm" : `#/kvm/${view}`;
}

const VIEW_NAMES: Record<Exclude<ConsoleView, "main">, string> = {
  "remote-login": "Remote Login",
  hardware: "Physical KVM",
};

function generateRdpPassword() {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%";
  const values = crypto.getRandomValues(new Uint8Array(22));
  return Array.from(values, (value) => alphabet[value % alphabet.length]).join("");
}

export function RemoteConsole({ session }: {
  session: Session;
  /** Accepted for the shell's signature; the boards draw no Back control (the rail is the way back). */
  onBack?: () => void;
}) {
  const [view, setView] = useState<ConsoleView>(() => consoleViewFromHash(window.location.hash));
  const [capability, setCapability] = useState<KvmCapabilities | null>(null);
  /**
   * "The check did not answer" and "the hardware is not fitted" are different
   * facts, and the readiness list must not render the second when it means the
   * first.
   */
  const [capabilityFailed, setCapabilityFailed] = useState(false);
  const [apps, setApps] = useState<RemoteApp[]>([]);
  const [hostDesktop, setHostDesktop] = useState<HostDesktop | null>(null);
  /** The last `/host/remote-desktop` read failed: what is shown is the previous answer, or nothing (S-Y3). */
  const [hostReadFailed, setHostReadFailed] = useState(false);
  const [remoteUrl, setRemoteUrl] = useState("");
  const [remoteSessionId, setRemoteSessionId] = useState("");
  const [remoteName, setRemoteName] = useState("");
  const [rdpUsername, setRdpUsername] = useState("vaelor");
  const [rdpPassword, setRdpPassword] = useState("");
  const [credentialsSaved, setCredentialsSaved] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState("");
  /** The page's own outcome: a session that stopped, ended or did not start, or checks that did not answer. */
  const [message, setMessageText] = useState("");
  // VD-189: a refusal is an alert, never an info notice.
  const [messageRefused, setMessageRefused] = useState(false);
  const setMessage = useCallback((text: string, refused = false) => { setMessageText(text); setMessageRefused(refused); }, []);
  /** Each card's own outcome, said where its button is. */
  const [deskOutcome, setDeskOutcome] = useState<Outcome>(null);
  const [rdpOutcome, setRdpOutcome] = useState<Outcome>(null);
  const [browserOutcome, setBrowserOutcome] = useState<Outcome>(null);
  const [kvmOutcome, setKvmOutcome] = useState<Outcome>(null);
  const [setupRequested, setSetupRequested] = useState(false);
  const [remoteSessionState, setRemoteSessionState] = useState<RemoteSessionState>("ended");
  const [sessionTarget, setSessionTarget] = useState<"host" | { app: RemoteApp } | null>(null);
  /* Rendered inside the session overlay — see `endDesktopSession`. */
  const [endError, setEndError] = useState("");
  /* VD-189: why a retry from inside the overlay failed; the page under the overlay is inert. */
  const [retryError, setRetryError] = useState("");
  const remoteDialogRef = useRef<HTMLDivElement>(null);
  const remoteCloseRef = useRef<HTMLButtonElement>(null);
  const remoteOpenerRef = useRef<HTMLElement | null>(null);
  /** Set by a move between views, so the arriving view takes focus once it renders. */
  const focusOnArrival = useRef(false);
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
   * ladder rather than the older `console_ready` summary.
   */
  const kvmReady = consoleSessionAvailable(capability?.ladder);
  const rdpAddress = `${hostDesktop?.address || window.location.hostname}:${hostDesktop?.rdp?.port || 3389}`;
  const hostOsName = hostDesktop?.os?.name || "Linux host";
  const remoteLoginName = hostDesktop?.name || `${hostOsName} Remote Desktop`;
  const sshCommand = `ssh <linux-user>@${hostDesktop?.address || window.location.hostname}`;
  const rdpUsernameValid = /^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$/.test(rdpUsername);
  // #149: name only the rule the current value breaks.
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

  useEffect(() => {
    const restore = () => setView(consoleViewFromHash(window.location.hash));
    window.addEventListener("hashchange", restore);
    window.addEventListener("popstate", restore);
    return () => {
      window.removeEventListener("hashchange", restore);
      window.removeEventListener("popstate", restore);
    };
  }, []);

  // "Remote console / Remote Login", "Remote console / Physical KVM" in the top bar.
  usePagePlace(view === "main" ? null : [VIEW_NAMES[view]]);

  const navigateView = (next: ConsoleView) => {
    if (window.location.hash !== hashForView(next)) window.history.pushState(null, "", hashForView(next));
    focusOnArrival.current = true;
    setView(next);
  };

  useEffect(() => {
    if (!focusOnArrival.current) return;
    focusOnArrival.current = false;
    const target = view === "remote-login"
      ? document.getElementById("rdp-settings") ?? document.getElementById("console-view-title")
      : document.getElementById(view === "hardware" ? "console-view-title" : "console-page-title");
    target?.scrollIntoView?.({ behavior: "smooth", block: "start" });
    target?.focus({ preventScroll: true });
  }, [view]);

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
    if (hostResult.status === "fulfilled") {
      setHostDesktop(hostResult.value);
      setHostReadFailed(false);
    } else {
      setHostReadFailed(true);
    }
    if (
      capabilityResult.status === "rejected"
      || managedResult.status === "rejected"
      || hostResult.status === "rejected"
    ) {
      setMessage("Some remote-access checks did not respond. Refresh the status and try again.", true);
    }
  }, [setMessage]);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    if (!setupRequested) return;
    const timer = window.setInterval(() => { void refresh(); }, 5000);
    if (browserReady) setSetupRequested(false);
    return () => window.clearInterval(timer);
  }, [browserReady, refresh, setupRequested]);

  const control = async (action: "acquire" | "release") => {
    setBusy(action);
    setKvmOutcome(null);
    try {
      await apiRequest("/kvm/control", { method: action === "acquire" ? "POST" : "DELETE", body: "{}" }, session.csrf_token);
      setKvmOutcome({ text: action === "acquire" ? "You now own keyboard and mouse control." : "Keyboard and mouse control released.", refused: false });
      await refresh();
    } catch (error) {
      setKvmOutcome({ text: error instanceof Error ? error.message : "Control ownership could not be changed.", refused: true });
    } finally {
      setBusy("");
    }
  };

  const rememberOpener = () => {
    if (!remoteUrl && document.activeElement instanceof HTMLElement) remoteOpenerRef.current = document.activeElement;
  };

  const openDesktop = async (app: RemoteApp) => {
    rememberOpener();
    setBusy(`app-${app.id}`);
    setSessionTarget({ app });
    setRemoteSessionState("connecting");
    setMessage(""); setRetryError("");
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
      const refusal = error instanceof Error ? error.message : "Remote Desktop could not start.";
      if (remoteUrl) setRetryError(refusal);
      else setMessage(refusal, true);
    } finally {
      setBusy("");
    }
  };

  const openHostBrowserDesktop = async () => {
    rememberOpener();
    setBusy("host-browser");
    setSessionTarget("host");
    setRemoteSessionState("connecting");
    setMessage(""); setRetryError("");
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
      if (remoteUrl) setRetryError(`${detail} Refresh the status or try opening it again.`);
      else setMessage(`${detail} Refresh the status or try opening it again.`, true);
    } finally {
      setBusy("");
    }
  };

  const retrySession = () => {
    if (sessionTarget === "host") void openHostBrowserDesktop();
    else if (sessionTarget) void openDesktop(sessionTarget.app);
  };

  const returnFocus = () => window.requestAnimationFrame(() => {
    if (remoteOpenerRef.current?.isConnected) remoteOpenerRef.current.focus();
  });

  /*
   * **This stops watching. It does not stop the desktop**, and the label says
   * so. It once read "Close session" while only clearing React state, so an
   * owner whose desktop had locked pressed it, reopened, and landed back in
   * the same broken session.
   */
  const stopViewingRemoteSession = () => {
    setRemoteUrl("");
    setRemoteSessionId("");
    setRemoteSessionState("ended");
    setMessage("Stopped viewing. The desktop is still running on the appliance.");
    returnFocus();
  };

  /*
   * **An end failure belongs to the session it happened to, and dies with
   * it** (#191). Keyed on the session id rather than cleared in each handler,
   * so a future path that changes the viewed session cannot forget to.
   */
  useEffect(() => { setEndError(""); setRetryError(""); }, [remoteSessionId]);

  /* Ends the desktop on the appliance, so the next one starts clean. */
  const endDesktopSession = async () => {
    setBusy("end-desktop");
    setEndError("");
    try {
      const result = await apiRequest<{
        ended?: boolean; detail?: string; screen_lock_disabled?: boolean;
        desktop_listening?: boolean;
      }>("/remote-desktop/browser-sessions/end", { method: "POST" }, session.csrf_token);
      setRemoteUrl("");
      setRemoteSessionId("");
      setRemoteSessionState("ended");
      /*
       * **Report what the appliance said, not what the button hoped.** The
       * appliance's readings decide whether "a fresh one" is promised, and
       * `detail` only supplies the words.
       */
      const confirmed = result?.desktop_listening !== false && result?.screen_lock_disabled !== false;
      setMessage(
        result?.ended === false
          ? result.detail || "There was no desktop session on the appliance to end."
          : confirmed
            ? "The desktop session on the appliance was ended. Opening it again starts a fresh one."
            : result?.detail
              || "The desktop session on the appliance was ended, but the appliance did not confirm it is ready to open again.",
      );
    } catch (error) {
      // **Into the overlay, not behind it**: the page under the modal is inert.
      setEndError(error instanceof Error ? error.message : "The desktop session could not be ended.");
    } finally {
      setBusy("");
      returnFocus();
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
    setRdpOutcome(null);
    try {
      await apiRequest(
        "/host/remote-desktop/rdp",
        { method: "POST", body: JSON.stringify({ username: rdpUsername, password: rdpPassword }) },
        session.csrf_token,
      );
      setCredentialsSaved(true);
      setRdpOutcome({ text: `${remoteLoginName} is ready. Copy the saved credentials below, then download the connection profile.`, refused: false });
      await refresh();
    } catch (error) {
      setRdpOutcome({ text: error instanceof Error ? error.message : `${remoteLoginName} could not be configured.`, refused: true });
    } finally {
      setBusy("");
    }
  };

  const downloadRdpProfile = async () => {
    setBusy("rdp-download");
    setDeskOutcome(null);
    try {
      await downloadApiFile("/host/remote-desktop/profile?mode=responsive", "vaelor-responsive-remote-login.rdp");
      setDeskOutcome({ text: "Optimized RDP profile downloaded at 1600 × 900 with audio disabled for a smoother remote session.", refused: false });
    } catch (error) {
      setDeskOutcome({ text: error instanceof Error ? error.message : "The RDP profile could not be downloaded.", refused: true });
    } finally {
      setBusy("");
    }
  };

  const copyRdpCredentials = async () => {
    try {
      await navigator.clipboard.writeText(`Username: ${rdpUsername}\nPassword: ${rdpPassword}`);
      setRdpOutcome({ text: "Saved RDP username and password copied.", refused: false });
    } catch {
      setRdpOutcome({ text: "Clipboard access was blocked. Use the visible username and password fields.", refused: false });
    }
  };

  const disableRdp = async () => {
    setBusy("rdp-disable");
    setRdpOutcome(null);
    try {
      await apiRequest("/host/remote-desktop/rdp", { method: "DELETE", body: "{}" }, session.csrf_token);
      setRdpOutcome({ text: `${remoteLoginName} is disabled.`, refused: false });
      await refresh();
    } catch (error) {
      setRdpOutcome({ text: error instanceof Error ? error.message : `${remoteLoginName} could not be disabled.`, refused: true });
    } finally {
      setBusy("");
    }
  };

  const enableBrowserDesktop = async () => {
    setBusy("browser-setup");
    setBrowserOutcome(null);
    try {
      await apiRequest(
        "/jobs",
        { method: "POST", body: JSON.stringify({ type: "host.vnc.enable", payload: { confirm: "enable-host-vnc" } }) },
        session.csrf_token,
      );
      setSetupRequested(true);
    } catch (error) {
      setBrowserOutcome({ text: error instanceof Error ? error.message : "The browser desktop setup could not start.", refused: true });
    } finally {
      setBusy("");
    }
  };

  const copyText = (text: string, done: string, fallback: string) => {
    void navigator.clipboard.writeText(text).then(
      () => setDeskOutcome({ text: done, refused: false }),
      () => setDeskOutcome({ text: fallback, refused: false }),
    );
  };

  const hostReading = hostDesktop === null ? "unread" as const : hostReadFailed ? "stale" as const : undefined;
  const facts: RemoteDesktopFacts = {
    rdpReady,
    rdpSetupSupported: hostDesktop?.rdp?.setup_supported !== false,
    browserReady,
    browserSetupSupported,
    consoleFallback,
    consoleReady,
    hostOsName,
    remoteLoginName,
    rdpAddress,
    sshCommand,
    certificate: hostDesktop?.rdp?.certificate,
    desktopDetail: hostDesktop?.desktop?.detail,
    reading: hostReading,
  };

  /*
   * LESSONS 1 / S-Y3: before the first answer, and after a failed re-read,
   * the page pill says so in grey rather than repeating a default or the last
   * good answer as if it were current.
   */
  const pageLabel = hostReading === "unread"
    ? (hostReadFailed ? "Not read" : "Checking remote access")
    : hostReading === "stale"
      ? "Old reading"
      : consoleFallback
        ? (consoleReady ? "Console ready" : "Console unavailable")
        : rdpReady
          ? "RDP ready"
          : browserReady
            ? "Remote access available"
            : kvmState === "ready"
              ? "Hardware KVM ready"
              : hostDesktop?.rdp.setup_supported
                ? "RDP setup available"
                : "Remote access setup available";
  const pageHealthy = rdpReady || browserReady || kvmReady || (consoleFallback && consoleReady);

  const desktopCard = (variant: "summary" | "settings") => (
    <RemoteDesktopCard
      busy={busy}
      canControl={canControl}
      facts={facts}
      form={variant === "settings" ? {
        canAdminister,
        busy,
        username: rdpUsername,
        usernameError: rdpUsernameError,
        usernameValid: rdpUsernameValid,
        password: rdpPassword,
        showPassword,
        credentialsSaved,
        outcome: rdpOutcome,
        onUsername: (value) => { setRdpUsername(value); setCredentialsSaved(false); },
        onPassword: (value) => { setRdpPassword(value); setCredentialsSaved(false); },
        onToggleShow: () => setShowPassword((value) => !value),
        onGenerate: () => {
          setRdpPassword(generateRdpPassword());
          setShowPassword(true);
          setCredentialsSaved(false);
          setRdpOutcome({ text: "A new password was generated but is not saved yet. Select “Save RDP credentials” next.", refused: false });
        },
        onSubmit: () => void configureRdp(),
        onCopySaved: () => void copyRdpCredentials(),
        onDisable: () => void disableRdp(),
      } : undefined}
      onCopyAddress={() => copyText(rdpAddress, "RDP address copied.", `RDP address: ${rdpAddress}`)}
      onCopyConsole={() => copyText(sshCommand, "Console command copied. Replace <linux-user> with your Ubuntu account name.", `Console command: ${sshCommand}`)}
      onDownloadProfile={() => void downloadRdpProfile()}
      onOpenBrowser={() => void openHostBrowserDesktop()}
      onOpenSettings={() => navigateView("remote-login")}
      outcome={deskOutcome}
      variant={variant}
    />
  );

  const header = (
    <div className="page-heading system-page-heading console-heading">
      <div>
        <h1 id="console-page-title" tabIndex={-1}>{destinations.kvm.name}</h1>
        <p>Connect with Remote Desktop, open an app's desktop, or use a hardware KVM.</p>
      </div>
    </div>
  );

  const crumb = view !== "main" && (
    <nav aria-label="Remote console views" className="console-crumb">
      <Button onClick={() => navigateView("main")} type="button" variant="quiet">{destinations.kvm.name}</Button>
      <span aria-hidden="true">/</span>
      {/* The Physical KVM view has no page header, so its name is the page's heading. */}
      {view === "hardware"
        ? <h1 id="console-view-title" tabIndex={-1}>{VIEW_NAMES[view]}</h1>
        : <h2 id="console-view-title" tabIndex={-1}>{VIEW_NAMES[view]}</h2>}
    </nav>
  );

  // The page's own outcome, with the way forward beside it (the ConsoleSession board's third row).
  const pageNotice = (
    <>
      {message && (
        <Notice className="console-notice" severity={messageRefused ? "danger" : "info"}>
          {message}
          <Button onClick={() => void refresh()} variant="secondary">Reload</Button>
        </Notice>
      )}
      {remoteSessionState === "failed" && !remoteUrl && (
        <Notice className="console-notice" severity="danger">
          The remote desktop session did not start. The page is still available; retry the session or choose another access path.
          <Button disabled={Boolean(busy)} onClick={retrySession} type="button" variant="primary">Retry session</Button>
        </Notice>
      )}
    </>
  );

  return (
    <div className="console-page sys-page">
      {/* The page's pill and Reload, in the top bar on every view (the Console boards). */}
      <TopbarPageActions>
        <StatusPill label={pageLabel} reading={hostReading} tone={pageHealthy ? "success" : "neutral"} />
        <Button disabled={Boolean(busy)} onClick={() => void refresh()} type="button" variant="secondary">Reload</Button>
      </TopbarPageActions>
      {crumb}
      {view !== "hardware" && header}
      {pageNotice}

      {view === "main" && (
        <div className="sys-section">
          <div className="sys-grid-2 console-pair">
            {desktopCard("summary")}
            <KvmVideoCard capability={capability} onOpenChecklist={() => navigateView("hardware")} state={kvmState} />
          </div>
          <AppDesktopsCard apps={apps} busy={busy} canControl={canControl} onOpen={(app) => void openDesktop(app)} />
          {/*
            * Three siblings, discovered independently and never inheriting each
            * other's rung: seeing the screen, driving the keyboard, and powering
            * the machine out of band fail for different reasons and are fixed
            * by different people.
            */}
          <ConsoleLadder rows={capability?.ladder} />
        </div>
      )}

      {view === "remote-login" && (
        <div className="sys-section">
          {desktopCard("settings")}
          <BrowserDesktopCard
            busy={busy}
            canAdminister={canAdminister}
            detail={hostDesktop?.browser_vnc?.detail}
            hostOsName={hostOsName}
            onInstall={() => void enableBrowserDesktop()}
            onRecheck={() => void refresh()}
            outcome={browserOutcome}
            ready={browserReady}
            setupRequested={setupRequested}
            setupSupported={browserSetupSupported}
          />
        </div>
      )}

      {view === "hardware" && (
        <div className="sys-section">
          <PhysicalKvmStage
            busy={busy}
            canControl={canControl}
            capability={capability}
            onControl={(action) => void control(action)}
            outcome={kvmOutcome}
            state={kvmState}
            username={session.user.username}
          />
          <div className="sys-grid-2">
            {/* Moved here whole: its three rows are a commissioning checklist. */}
            <ConsoleReadinessPanel capability={capability} failed={capabilityFailed} />
            <KvmChecklistCard capability={capability} failed={capabilityFailed} />
          </div>
        </div>
      )}

      {remoteUrl && (
        <ConsoleSessionDialog
          busy={busy}
          canEnd={canControl && sessionTarget === "host"}
          closeRef={remoteCloseRef}
          dialogRef={remoteDialogRef}
          endError={endError}
          name={remoteName}
          onEnd={() => void endDesktopSession()}
          onFrameError={() => setRemoteSessionState("failed")}
          onRetry={retrySession}
          onStop={stopViewingRemoteSession}
          retryError={retryError}
          state={remoteSessionState}
          url={remoteUrl}
        />
      )}
    </div>
  );
}
