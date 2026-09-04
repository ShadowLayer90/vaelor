import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import type {
  AuditEvent,
  Device,
  Health,
  Metrics,
  Session,
  TelemetrySample,
} from "../types";
import { apiRequest } from "../lib/api";
import { brand } from "../lib/brand";
import {
  formatQuantity,
  formatPercent,
  formatTemperature,
  formatUptime,
  timeAgo,
} from "../lib/format";
import {
  connectionStatus,
  sampleTimeMs,
  STALE_AFTER_MS,
  type TelemetryOutcome,
} from "../lib/connectionState";
import { metricNumber, metricSum } from "../lib/metrics";
import { machineNoun, thermalLimits, unknownMachine } from "../lib/machine";
import { healthClaim } from "../lib/health";
import { headlineFan, parseBoardSensors } from "../lib/wmiSensors";
import { useMachineProfile } from "../hooks/useMachineProfile";
import { overviewCards } from "./overviewCards";
import { OverviewStoragePanel } from "./OverviewStoragePanel";
import { retentionNote, type RetentionStatus } from "../lib/retention";
import { OverviewHardwareChips } from "./OverviewHardwareChips";
import { EnclosurePanel } from "./EnclosurePanel";
import { PiPowerPanel } from "./PiPowerPanel";
import { ConfirmDialog } from "./ConfirmDialog";
import { AmbientBackground } from "./AmbientBackground";
import { Icon, type IconName } from "./Icon";
import { MetricCard } from "./MetricCard";
import { Sidebar, type StorageSummary } from "./Sidebar";
import {
  hashForPage,
  hashTargetsPage,
  navigationPages,
  resolveHash,
  standaloneRouteFromHash,
  type NavigationPage,
  type StandaloneRoute,
} from "../lib/navigation";
import { destinationDescriptorFor, destinations, documentTitleForPage } from "../lib/destinations";
import { auditActionLabel } from "../lib/auditLabels";
import { OverviewQuickActions } from "./OverviewQuickActions";
import { StatusPill } from "./StatusPill";
import { ProductMark } from "./ProductMark";
import { PironmanDeviceIcon } from "./PironmanDeviceIcon";
import { DeviceHeroIcon } from "./DeviceHeroIcon";
import { ModalShell } from "./ModalShell";
import { MemoryOptimizer } from "./MemoryOptimizer";
import { WorkspaceErrorBoundary } from "./WorkspaceErrorBoundary";
import { Button, Notice, Select, UnavailableValue } from "./ui";
import {
  ActivityCenter,
  Administration,
  AgentCenter,
  AiChat,
  FanControl,
  FleetCenter,
  MemoryCenter,
  RemoteConsole,
  useRouteChunkPrefetch,
  Workloads,
} from "./overviewRoutes";

type PowerAction = "restart_service" | "reboot" | "shutdown";
type PowerCapabilities = {
  actions: Record<PowerAction, { available: boolean; reason: string }>;
};

const actionCopy: Record<
  PowerAction,
  {
    title: string;
    description: string;
    label: string;
    confirmation: string;
    /** Shown on the control itself, before the confirmation dialog opens. */
    consequence: string;
  }
> = {
  restart_service: {
    // The control and its confirmation have to be the same action. A card
    // labelled "Restart service" that opens "Restart hardware service?" reads
    // as a second, broader thing happening.
    title: "Restart service?",
    description:
      "Live telemetry will disconnect briefly. The device will remain powered on.",
    label: "Restart service",
    confirmation: "restart-service",
    consequence: "Interrupts live telemetry for a few seconds",
  },
  reboot: {
    title: "Reboot this device?",
    description:
      "All active sessions and services will stop while the device restarts.",
    label: "Reboot device",
    confirmation: "reboot-device",
    consequence: "Stops every session and running app until it restarts",
  },
  shutdown: {
    title: "Shut down this device?",
    description:
      "Remote access will be lost and physical power may be required to start it again.",
    label: "Shut down",
    confirmation: "shutdown-device",
    consequence: "Ends remote access; restarting may need physical power",
  },
};

function WorkspacePage({
  page,
  session,
  onBack,
}: {
  page: Exclude<NavigationPage, "overview">;
  session: Session;
  onBack: () => void;
}) {
  switch (page) {
    case "system":
      return <FanControl onBack={onBack} session={session} />;
    case "kvm":
      return <RemoteConsole onBack={onBack} session={session} />;
    case "workloads":
      return <Workloads session={session} />;
    case "fleet":
      return <FleetCenter onBack={onBack} session={session} />;
    case "assistant":
      return <AgentCenter session={session} />;
    case "ai-chat":
      return <AiChat session={session} />;
    case "activity":
      return <ActivityCenter session={session} />;
    case "admin":
      return <Administration onBack={onBack} session={session} />;
  }
}

export function Overview({
  session,
  onLogout,
}: {
  session: Session;
  onLogout: () => Promise<void>;
}) {
  const [device, setDevice] = useState<Device | null>(null);
  const [health, setHealth] = useState<Health>({
    status: "offline",
    reasons: [],
    sampled_at: 0,
  });
  const [history, setHistory] = useState<TelemetrySample[]>([]);
  const [audit, setAudit] = useState<AuditEvent[]>([]);
  /*
   * The two facts the connection indicator is allowed to be built from. They
   * are kept apart on purpose: `telemetryOutcome` says what the last request
   * did, `polling` says whether anything is being requested at all. Collapsing
   * them into one `isLive` boolean is exactly what let a request that resolved
   * after the tab was hidden repaint the dot green while the sentence beside
   * it correctly read "Paused" (#52).
   */
  const [telemetryOutcome, setTelemetryOutcome] = useState<TelemetryOutcome>("pending");
  const [polling, setPolling] = useState(() => !document.hidden);
  const [clock, setClock] = useState(() => Date.now());
  const [selectedAction, setSelectedAction] = useState<PowerAction | null>(null);
  const [actionBusy, setActionBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const pageAllowed = useCallback((page: NavigationPage) =>
    !(
      ((page === "assistant" || page === "admin") && session.user.role !== "administrator")
      || (page === "ai-chat" && session.user.role === "viewer")
      || (page === "activity" && session.user.role === "viewer")
    ), [session.user.role]);
  /*
   * Every memory endpoint is administrator-only, so the standalone route is
   * gated the same way the chip that offers it is. An operator who types the
   * URL lands on Home rather than on a page that 403s on load.
   */
  const standaloneAllowed = useCallback(
    (route: StandaloneRoute) => route !== "memory" || session.user.role === "administrator",
    [session.user.role],
  );
  const [standaloneRoute, setStandaloneRoute] = useState<StandaloneRoute | null>(() => {
    const requested = standaloneRouteFromHash(window.location.hash);
    return requested && standaloneAllowed(requested) ? requested : null;
  });
  const [activePage, setActivePage] = useState<NavigationPage>(() => {
    const requested = resolveHash(window.location.hash).page;
    return pageAllowed(requested) ? requested : "overview";
  });
  const [deviceChoice, setDeviceChoice] = useState("");
  const [deviceChoiceBusy, setDeviceChoiceBusy] = useState(false);
  const [showDeviceSelector, setShowDeviceSelector] = useState(false);
  const [storage, setStorage] = useState<StorageSummary | null>(null);
  const [retention, setRetention] = useState<RetentionStatus | null>(null);
  const [connectivity, setConnectivity] = useState<{
    dns: boolean;
    internet: boolean;
    latency_ms: number | null;
  } | null>(null);
  const [showMemoryOptimizer, setShowMemoryOptimizer] = useState(false);
  const [powerCapabilities, setPowerCapabilities] = useState<PowerCapabilities | null>(null);
  /**
   * `null` until discovery answers. Nothing that depends on a capability may
   * render an interactive control before this resolves, because the honest
   * default for "we have not asked yet" is not "yes".
   */
  const machine = useMachineProfile();

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
    const main = document.getElementById("main-content");
    main?.focus({ preventScroll: true });
  }, [activePage, standaloneRoute]);

  // Warm the lazy route chunks on idle after first paint (see overviewRoutes),
  // so the first navigation is not a cold fetch while the initial load stays light.
  useRouteChunkPrefetch();

  const navigateTo = useCallback((page: NavigationPage, addHistory = true) => {
    if (!navigationPages.includes(page) || !pageAllowed(page)) return;
    setStandaloneRoute(null);
    if (addHistory && page !== activePage) {
      window.history.pushState(null, "", hashForPage(page));
    } else if (!addHistory && !hashTargetsPage(window.location.hash, page)) {
      window.history.replaceState(null, "", hashForPage(page));
    }
    setActivePage(page);
  }, [activePage, pageAllowed]);

  /**
   * Resolve whatever is in the address bar and make the address bar agree with
   * what is on screen. An alias such as `#/settings`, an unknown path, or a
   * page this account may not open used to fall through to Home while the URL
   * kept claiming otherwise, so the link simply looked broken.
   */
  const applyLocation = useCallback(() => {
    const standalone = standaloneRouteFromHash(window.location.hash);
    if (standalone && standaloneAllowed(standalone)) {
      setStandaloneRoute(standalone);
      return;
    }
    setStandaloneRoute(null);
    const { page, redirect } = resolveHash(window.location.hash);
    const target = pageAllowed(page) ? page : "overview";
    if (redirect || target !== page || !hashTargetsPage(window.location.hash, target)) {
      window.history.replaceState(null, "", hashForPage(target));
    }
    setActivePage(target);
  }, [pageAllowed, standaloneAllowed]);

  useEffect(() => {
    applyLocation();
    const navigate = (event: Event) => {
      navigateTo((event as CustomEvent<NavigationPage>).detail);
    };
    window.addEventListener("popstate", applyLocation);
    window.addEventListener("hashchange", applyLocation);
    window.addEventListener("pironman:navigate", navigate);
    return () => {
      window.removeEventListener("popstate", applyLocation);
      window.removeEventListener("hashchange", applyLocation);
      window.removeEventListener("pironman:navigate", navigate);
    };
  }, [applyLocation, navigateTo]);

  // WCAG 2.4.2 Page Titled: tabs, history entries and the screen-reader page
  // announcement each name the destination before the product.
  useEffect(() => {
    document.title = standaloneRoute === "memory"
      ? `What Vaelor remembers · ${brand.name}`
      : documentTitleForPage(activePage);
  }, [activePage, standaloneRoute]);

  const refreshSummary = useCallback(async () => {
    const [nextDevice, nextHealth, sample, nextStorage, nextPower] = await Promise.all([
      apiRequest<Device>("/device"),
      apiRequest<Health>("/health"),
      apiRequest<TelemetrySample>("/telemetry/current"),
      apiRequest<StorageSummary>("/system/storage"),
      apiRequest<PowerCapabilities>("/power/capabilities"),
    ]);
    setDevice(nextDevice);
    setHealth(nextHealth);
    setStorage(nextStorage);
    setPowerCapabilities(nextPower);
    setHistory((current) => [...current, sample].slice(-30));
    // Retention (#186) is a soft add: an appliance whose control plane predates
    // this route must still render the summary, so its failure is swallowed and
    // the note simply stays absent.
    void apiRequest<RetentionStatus>("/telemetry/retention")
      .then(setRetention)
      .catch(() => undefined);
  }, []);

  /*
   * A one-shot read on demand. Polling is suspended while the tab is in the
   * background (#52); this gives the reader who came back to a "Paused" strip a
   * fresh reading without a full reload. It fetches once regardless of the pause
   * and ages the clock to the answer, but does not touch `polling` — a manual
   * refresh, not a defeat of the pause.
   */
  const refreshTelemetryNow = useCallback(() => {
    setClock(Date.now());
    void refreshSummary()
      .then(() => setTelemetryOutcome("ok"))
      .catch(() => setTelemetryOutcome("failed"));
  }, [refreshSummary]);

  const saveDeviceChoice = async () => {
    if (!deviceChoice) return;
    setDeviceChoiceBusy(true);
    setNotice("");
    try {
      await apiRequest(
        "/device/model",
        { method: "PATCH", body: JSON.stringify({ model: deviceChoice }) },
        session.csrf_token,
      );
      const selectedDevice = await apiRequest<Device>("/device");
      setDevice(selectedDevice);
      setShowDeviceSelector(false);
      setNotice("Pironman model saved. Hardware labels and the product image now match your selection.");
      void refreshSummary().catch(() => {
        setNotice("Pironman model saved. Some unrelated live telemetry is still refreshing.");
      });
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The Pironman model could not be saved.");
    } finally {
      setDeviceChoiceBusy(false);
    }
  };

  useEffect(() => {
    if (activePage === "overview" && session.user.role !== "viewer") {
      void apiRequest<AuditEvent[]>("/audit?limit=6").then(setAudit);
    }
  }, [activePage, session.user.role]);

  useEffect(() => {
    if (session.user.role === "viewer") return;
    void apiRequest<{ dns: boolean; internet: boolean; latency_ms: number | null }>(
      "/system/network/test",
      { method: "POST", body: "{}" },
      session.csrf_token,
    ).then(setConnectivity).catch(() => setConnectivity({
      dns: false, internet: false, latency_ms: null,
    }));
  }, [session.csrf_token, session.user.role]);

  useEffect(() => {
    setPolling(!document.hidden);
    void refreshSummary()
      .then(() => setTelemetryOutcome("ok"))
      .catch(() => setTelemetryOutcome("failed"));
    let telemetryPending = false;
    const pollTelemetry = () => {
      if (document.hidden || telemetryPending) return;
      telemetryPending = true;
      void apiRequest<TelemetrySample>("/telemetry/current")
        .then((sample) => {
          setHistory((current) => [...current, sample].slice(-30));
          setTelemetryOutcome("ok");
        })
        .catch(() => setTelemetryOutcome("failed"))
        .finally(() => { telemetryPending = false; });
    };
    const visibilityChanged = () => {
      /*
       * `polling` is written from the one place that decides whether requests
       * are being made, and never from a request handler. A response that
       * lands after this point cannot claim the connection is live, which is
       * the whole defect in #52: the last in-flight poll always won the race
       * because nothing ran afterwards to correct it.
       */
      setPolling(!document.hidden);
      if (!document.hidden) {
        setClock(Date.now());
        pollTelemetry();
        void refreshSummary();
      }
    };
    document.addEventListener("visibilitychange", visibilityChanged);
    const telemetryInterval = window.setInterval(
      pollTelemetry,
      activePage === "overview" ? 2_500 : 30_000,
    );
    const healthInterval = window.setInterval(() => {
      if (!document.hidden) void apiRequest<Health>("/health").then(setHealth);
    }, activePage === "overview" ? 30_000 : 60_000);
    return () => {
      document.removeEventListener("visibilitychange", visibilityChanged);
      window.clearInterval(telemetryInterval);
      window.clearInterval(healthInterval);
    };
  }, [activePage, refreshSummary]);

  /*
   * The connection sentence ages against this clock, and the top bar is on
   * every screen — so the clock has to run on every screen. It used to stop
   * outside Home, which froze `now` at the moment that page was opened while
   * `lastSample` kept advancing: the age went negative, clamped to zero, and
   * the bar read "Live · updated just now" indefinitely on Workloads, Chat and
   * Settings. A frozen clock is a second way for this indicator to lie.
   */
  /*
   * The clock keeps running while the page is hidden, and that is the point.
   * It used to skip the tick on `document.hidden`, which froze the age at the
   * moment the tab went away — so a reader coming back after ten minutes was
   * told "Paused · last updated 15 sec ago" (measured live 2026-08-11, #162).
   * Polling is correctly suspended while hidden; the *age of what we already
   * have* is the one number that must keep moving, because it is the only
   * thing telling the reader how stale a paused reading has become. It costs
   * a `setState` on a timer browsers already throttle in background tabs.
   */
  useEffect(() => {
    const timer = window.setInterval(
      () => setClock(Date.now()),
      document.hidden ? 10_000 : activePage === "overview" ? 1_000 : 5_000,
    );
    return () => window.clearInterval(timer);
  }, [activePage, polling]);

  const latest = history.at(-1);
  const metrics = latest?.metrics ?? {};
  const lastSample = latest?.sampled_at ?? 0;
  const telemetryAge = lastSample ? clock - sampleTimeMs(lastSample) : Number.POSITIVE_INFINITY;
  const telemetryStale = telemetryAge > STALE_AFTER_MS;
  /*
   * One value, four states. Every indicator on this screen and in the rail is
   * painted from `connection.state`, and the sentence beside each of them is
   * `connection.label`. There is no second opinion available to render.
   */
  const connectionInput = { lastSample, now: clock, outcome: telemetryOutcome, polling };
  const connection = connectionStatus(connectionInput);
  const heroSignal = connectionStatus(connectionInput, "signal-orb");
  const busPulse = connectionStatus(connectionInput, "telemetry-bus__pulse");
  const networkTotal = metricSum(metrics, ["network_download_speed", "network_upload_speed"]);
  const resolvedMachine = machine ?? unknownMachine;
  const isAppliance = resolvedMachine.machine_class === "pi-appliance";
  const noun = machineNoun(resolvedMachine.machine_class);
  /*
   * The appliance publishes one PWM fan under `pwm_fan_speed`; the workstation
   * publishes several through the board. Either is a fan reading, and neither
   * is a fan control.
   */
  const boardFan = headlineFan(parseBoardSensors(metrics));
  const fanRpm = metricNumber(metrics, "pwm_fan_speed") ?? boardFan?.rpm ?? null;
  const gpuAvailable = resolvedMachine.capabilities.gpu.available;

  const cards = useMemo(
    () => overviewCards({
      metrics,
      history,
      machine: resolvedMachine,
      onTuneMemory: () => setShowMemoryOptimizer(true),
    }),
    [history, metrics, resolvedMachine],
  );

  const performAction = async () => {
    if (!selectedAction) return;
    setActionBusy(true);
    try {
      await apiRequest(
        "/power/actions",
        {
          method: "POST",
          body: JSON.stringify({
            action: selectedAction,
            confirmation: actionCopy[selectedAction].confirmation,
          }),
        },
        session.csrf_token,
      );
      setNotice(`${actionCopy[selectedAction].label} request accepted.`);
      setSelectedAction(null);
      window.setTimeout(() => setNotice(""), 5000);
      if (session.user.role !== "viewer") {
        window.setTimeout(() => {
          void apiRequest<AuditEvent[]>("/audit?limit=6").then(setAudit);
        }, 500);
      }
    } finally {
      setActionBusy(false);
    }
  };

  /*
   * The hero's mark, its sentence and the pill in the page heading are one
   * value. During a telemetry outage the hero read "All systems operational"
   * beside a red dot while the pill still said OPERATIONAL — the dot from the
   * connection state, the words from the last /health answer. Losing contact
   * with a machine is not evidence that its systems are operational.
   */
  const claim = healthClaim(health, connection.state, noun);

  return (
    <div className="app-shell">
      <AmbientBackground />
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <Sidebar
        activePage={activePage}
        connection={connection.state}
        connectivity={connectivity}
        health={health}
        machineClass={resolvedMachine.machine_class}
        thermalWarningC={thermalLimits(resolvedMachine).cpuWarn}
        metrics={metrics}
        onNavigate={navigateTo}
        storage={storage}
        user={session.user}
      />

      <div className="workspace">
        <header className="topbar">
          <div className="topbar__identity">
            <span className="topbar__mark" aria-hidden="true"><ProductMark /></span>
            <strong>{brand.name}</strong>
          </div>
          <div className="topbar__actions">
            <div
              aria-label="Connection status"
              aria-live="polite"
              className="connection-state"
              data-connection={connection.state}
            >
              <span className={connection.indicatorClassName} />
              {connection.label}
            </div>
            <span className="topbar__divider" aria-hidden="true" />
            <div className="operator-chip">
              <span>{session.user.username.slice(0, 1).toUpperCase()}</span>
              <div>
                <strong>{session.user.username}</strong>
                <small>{session.user.role}</small>
              </div>
            </div>
            <Button
              aria-label="Sign out"
              className="icon-button"
              onClick={() => void onLogout()}
              title="Sign out"
            >
              <Icon name="logout" />
            </Button>
          </div>
        </header>

        <main className="main" id="main-content" tabIndex={-1}>
          {standaloneRoute === "memory" ? (
            <WorkspaceErrorBoundary
              onBack={() => navigateTo("overview", false)}
              workspaceKey="memory"
            >
              <Suspense fallback={<div className="page-loading" role="status">Loading workspace…</div>}>
                <MemoryCenter session={session} />
              </Suspense>
            </WorkspaceErrorBoundary>
          ) : activePage !== "overview" ? (
            <WorkspaceErrorBoundary
               onBack={() => navigateTo("overview", false)}
               workspaceKey={activePage}
             >
               <Suspense fallback={<div className="page-loading" role="status">Loading workspace…</div>}>
                 <WorkspacePage onBack={() => navigateTo("overview", false)} page={activePage} session={session} />
               </Suspense>
            </WorkspaceErrorBoundary>
          ) : (
            <>
          <div className="page-heading">
            <div>
              <h1>{destinations.overview.name}</h1>
              <p>{destinationDescriptorFor("overview", resolvedMachine.machine_class)} · {device?.name ?? "Detecting hardware"}</p>
            </div>
            <StatusPill status={claim.pillStatus} />
          </div>

          {notice && (
            <Notice severity="info">
              <Icon name="shield" size={17} />
              {notice}
            </Notice>
          )}

          <section className="device-hero" aria-label="System summary">
            <div className="device-hero__content">
              <div className="system-strip__lead">
                <span className={heroSignal.indicatorClassName} />
                <div>
                  <strong>{claim.title}</strong>
                  {/*
                    * Four alert categories were enumerated here and two were
                    * evaluated. `/health` computes from `cpu_temperature` and
                    * `memory_percent` alone, so a workstation with its
                    * graphics processor at 100 °C read "No thermal … alerts"
                    * on its landing page. The sentence now names what was
                    * checked and nothing else, and grows by itself the moment
                    * the route reports a wider `checked` list.
                    */}
                  <span>{claim.detail}</span>
                </div>
              </div>
              <dl className="system-strip__facts">
                <div>
                  <dt>Uptime</dt>
                  {/*
                    * A permanent em dash reads as a value. `boot_time` is only
                    * emitted by some telemetry providers, so when it is absent
                    * the fact says it is absent.
                    */}
                  <dd>{metricNumber(metrics, "boot_time") === null
                    ? <UnavailableValue label="Uptime unavailable" reason={`Boot time is not reported by this ${noun}`} />
                    : formatUptime(metrics.boot_time)}</dd>
                </div>
                <div>
                  <dt>Version</dt>
                  <dd>{device?.version ?? "—"}</dd>
                </div>
                <div>
                  <dt>Access</dt>
                  <dd>{session.user.role}</dd>
                </div>
                <div className="system-strip__os">
                  <dt>Operating system</dt>
                  <dd title={device?.platform?.os.name}>
                    {device?.platform?.os.name ?? "Detecting"}
                  </dd>
                </div>
              </dl>
              <OverviewHardwareChips device={device} machine={resolvedMachine} storage={storage} />
              {/*
                * Offering a Pironman model on a machine that is not one lets a
                * reader give a workstation a Pi enclosure's artwork, NVMe slot
                * count, OLED and RGB chips and a two-fan case-fan count. The
                * picker belongs to the appliance class and nowhere else.
                */}
              {device && isAppliance && !device.platform?.product.confident && (
                <div className="device-identification">
                  {/*
                    * What discovery actually found, before asking the reader to
                    * decide. This was framed as a chooser with no statement of
                    * what had been detected or how sure it was, so the one
                    * option offered read as an arbitrary menu of one rather
                    * than as a reading the reader is being asked to confirm.
                    */}
                  <div>
                    <strong>Which Pironman enclosure is this?</strong>
                    <small>
                      {device.platform?.product.detected_from
                        ? `Detected as ${device.platform.product.name} from ${device.platform.product.detected_from}, but not confidently. `
                        : "Automatic identification was inconclusive. "}
                      Your choice controls hardware labels, available controls, and the product image.
                    </small>
                  </div>
                  <Select
                    aria-label="Pironman enclosure model"
                    label={<span className="sr-only">Pironman enclosure model</span>}
                    onChange={(event) => setDeviceChoice(event.target.value)}
                    value={deviceChoice}
                  >
                    <option value="">Choose a model</option>
                    {device.model_choices?.map((choice) => <option key={choice.id} value={choice.id}>{choice.name}</option>)}
                  </Select>
                  <Button disabled={!deviceChoice || deviceChoiceBusy} onClick={() => void saveDeviceChoice()} variant="primary">
                    {deviceChoiceBusy ? "Saving…" : "Use this model"}
                  </Button>
                </div>
              )}
            </div>
            <figure className="device-hero__visual">
              <div className="device-hero__glow" aria-hidden="true" />
              <DeviceHeroIcon device={device} isAppliance={isAppliance} machine={resolvedMachine} />
              <figcaption>
                <span>{isAppliance ? "Active appliance" : "This machine"}</span>
                <strong>{device?.name ?? (isAppliance ? "Local appliance" : "This machine")}</strong>
                {session.user.role === "administrator" && isAppliance && device?.platform?.board.is_raspberry_pi && <Button onClick={() => {
                  setDeviceChoice(device?.id ?? "");
                  setShowDeviceSelector(true);
                }} variant="quiet">Change enclosure</Button>}
              </figcaption>
            </figure>
          </section>

          {showDeviceSelector && device && <ModalShell labelledBy="device-selector-title" onClose={() => setShowDeviceSelector(false)} size="standard">
            <section className="device-selector" aria-labelledby="device-selector-title">
              <div className="panel-heading">
                <div><span className="page-eyebrow">Hardware identity</span><h2 id="device-selector-title">Choose your Pironman enclosure</h2><p>This changes the product image, labels, fan layout, NVMe count, and available hardware controls.</p></div>
                <Button onClick={() => setShowDeviceSelector(false)} variant="quiet">Close</Button>
              </div>
              {/*
                * A single-select list, marked up as one. These carried
                * `aria-pressed`, which announces a toggle that happens to be
                * on — nothing said the options were alternatives or which one
                * is currently in force, and with one option offered there was
                * no visible answer either.
                */}
              <div aria-labelledby="device-selector-title" className="device-selector__grid" role="radiogroup">
                {device.model_choices?.map((choice) => (
                  <Button
                    aria-checked={deviceChoice === choice.id}
                    key={choice.id}
                    onClick={() => setDeviceChoice(choice.id)}
                    role="radio"
                  >
                    <span><PironmanDeviceIcon model={choice.id} size={126} /></span>
                    <strong>{choice.name}</strong>
                    {device.platform?.product.id === choice.id && (
                      <small>{device.platform.product.confident ? "Detected" : "Best match from discovery"}</small>
                    )}
                  </Button>
                ))}
              </div>
              <div className="device-selector__actions">
                <span><Icon name="shield" /> You can change this later without reinstalling.</span>
                <Button disabled={!deviceChoice || deviceChoiceBusy} onClick={() => void saveDeviceChoice()} variant="primary">{deviceChoiceBusy ? "Saving…" : "Use selected enclosure"}</Button>
              </div>
            </section>
          </ModalShell>}
          {showMemoryOptimizer && <ModalShell labelledBy="memory-optimizer-title" onClose={() => setShowMemoryOptimizer(false)}>
            <MemoryOptimizer onClose={() => setShowMemoryOptimizer(false)} session={session} />
          </ModalShell>}

          <OverviewQuickActions machineClass={resolvedMachine.machine_class} pageAllowed={pageAllowed} />

          <section className={telemetryStale ? "telemetry-bus telemetry-bus--stale" : "telemetry-bus"} aria-label="Telemetry polling status">
            <div className="telemetry-bus__state">
              <span className={busPulse.indicatorClassName} />
              <span>
                <small>Telemetry polling</small>
                <strong>{busPulse.label}</strong>
              </span>
            </div>
            <div className="telemetry-bus__channels">
              <span><small>CPU</small><strong>{formatPercent(metrics.cpu_percent)}</strong></span>
              {/*
                * One reading, one name. This strip called it "Core" while the
                * chart below called it "CPU temperature"; both read
                * `cpu_temperature` from the same sample.
                */}
              <span><small>CPU temperature</small><strong>{formatTemperature(metrics.cpu_temperature)}</strong></span>
              {/*
                * The `Fan` channel could never carry a value on a machine with
                * no readable fan, so it is a channel only where one was found.
                * Where a GPU was found it takes the slot, because that is the
                * reading a workstation owner actually watches.
                */}
              {/*
                * Gated on being able to *read* a fan, not on being able to set
                * one. This machine has three readable tachometers and no
                * writable control, and while the channel keyed off `cpu_fan`
                * the product showed nothing — which a reader takes as "no
                * fans", on a chassis that is plainly cooling itself.
                *
                * The channel names the fan it is quoting. A machine with three
                * of them has no single "fan speed", and presenting one of
                * three under that word would be the same class of claim.
                */}
              {resolvedMachine.capabilities.fan_readings.available && (
                <span>
                  <small>{boardFan?.label ?? "Fan"}</small>
                  <strong>{fanRpm === null
                    ? <UnavailableValue label="Fan speed unavailable" reason="This cooling controller does not report fan RPM" />
                    : `${Math.round(fanRpm)} RPM`}</strong>
                </span>
              )}
              {gpuAvailable && (
                <span><small>GPU</small><strong>{formatPercent(metrics.gpu_busy_percent)}</strong></span>
              )}
              {gpuAvailable && (
                <span><small>GPU temperature</small><strong>{formatTemperature(metrics.gpu_temperature_c)}</strong></span>
              )}
              <span><small>Memory</small><strong>{formatPercent(metrics.memory_percent)}</strong></span>
              <span><small>Network</small><strong>{formatQuantity(networkTotal, "transfer", "/s")}</strong></span>
            </div>
            {/*
              * A suspended poller is intentional, not a fault — and it used to
              * have no exit but a full page reload. The strip now says the pause
              * is automatic and reversible, and offers a one-shot read on
              * demand. "Refresh now" is not a resume switch: the interval still
              * owns the steady state and returns to it when the tab is active.
              */}
            {connection.state === "paused" && (
              <div className="telemetry-bus__resume">
                <small>
                  Live updates pause while this tab is in the background and resume
                  automatically when it is active.
                </small>
                <Button onClick={refreshTelemetryNow} type="button" variant="quiet">
                  Refresh now
                </Button>
              </div>
            )}
            <span className="telemetry-bus__sweep" aria-hidden="true" />
          </section>

          <section aria-labelledby="live-metrics-heading" className="section-block">
            <div className="section-heading">
              <div>
                <h2 id="live-metrics-heading">Live telemetry</h2>
                {/*
                  * The tile colours group readings; they are not a severity
                  * scale. A tester read the pink trace under a 33 °C processor
                  * as an alarm — on a page whose own health line said "All
                  * systems operational" — and nothing on screen contradicted
                  * that reading, because there was no legend and no tooltip.
                  * The colours cannot carry severity without a thermal band
                  * per series, which is the health strip's job, so the caption
                  * says what they are instead of implying what they are not.
                  */}
                <p>Thirty-second rolling window · colours group the readings and do not indicate health</p>
              </div>
              <span className="section-heading__meta">
                Updated {lastSample ? timeAgo(lastSample) : "when connected"}
              </span>
            </div>
            {/* The span rules are tuned for the four-tile layout; the GPU pair
                makes six, which tiles evenly as two rows of three instead. */}
            <div
              className={telemetryStale ? "metric-grid metric-grid--stale" : "metric-grid"}
              data-cards={cards.length}
            >
              {cards.map((card) => (
                <MetricCard {...card} key={card.label} />
              ))}
            </div>
          </section>

          {/*
            * The enclosure is the reason this product exists and it was not on
            * Home at all: "is the fan about to be loud", "are the lights on",
            * "how long will the UPS hold" were each two clicks away, while the
            * UPS itself was a 13-pixel chip beside the NVMe slot count. Both
            * panels are appliance-only — there is no enclosure and no PiPower
            * on a workstation, and a panel that rendered "—" on one would be
            * the defect these fixes remove.
            */}
          {isAppliance && (
            <div className="lower-grid">
              <EnclosurePanel machine={resolvedMachine} />
              <PiPowerPanel device={device} />
            </div>
          )}

          <div className="lower-grid">
            <OverviewStoragePanel machineNoun={noun} metrics={metrics} storage={storage} retentionNote={retentionNote(retention)} />

            {/*
              * The KVM readiness list used to sit here. It is genuinely wired
              * to `/kvm/capabilities` — that fix was worth making — but on a
              * machine that will never have a capture card fitted, three rows
              * reading "Not connected" forever are a commissioning checklist
              * occupying a third of the landing page. It moved, whole, to
              * `Remote console`, where commissioning is the subject.
              */}
            <section className="data-panel data-panel--actions" aria-labelledby="actions-heading">
              <div className="panel-heading">
                <div>
                  <h2 id="actions-heading">Power controls</h2>
                  <p>Each one interrupts service and asks you to confirm first</p>
                </div>
                <Icon name="power" />
              </div>
              {session.user.role === "viewer" ? (
                <div className="empty-state">
                  <Icon name="lock" />
                  <strong>Viewer access</strong>
                  <span>Operator permission is required.</span>
                </div>
              ) : (
                // A destructive action must not look like a link. These carry
                // the danger palette, an alert glyph instead of a navigation
                // chevron, the consequence in plain words, and `aria-haspopup`
                // so assistive technology announces that a confirmation step
                // follows rather than a page change.
                <div className="action-list action-list--destructive">
                  {(Object.keys(actionCopy) as PowerAction[]).map((action) => {
                    const available = Boolean(powerCapabilities?.actions[action]?.available);
                    return (
                      <Button
                        aria-haspopup="dialog"
                        disabled={!available}
                        key={action}
                        onClick={() => setSelectedAction(action)}
                        title={powerCapabilities?.actions[action]?.reason || undefined}
                        variant="danger"
                      >
                        <span className="action-list__copy">
                          <strong>{actionCopy[action].label}</strong>
                          <small>
                            {available
                              ? actionCopy[action].consequence
                              : powerCapabilities?.actions[action]?.reason || "Checking platform support"}
                          </small>
                        </span>
                        <Icon name="alert" size={17} />
                      </Button>
                    );
                  })}
                </div>
              )}
            </section>
          </div>

          <section className="activity-panel" aria-labelledby="activity-heading">
            <div className="panel-heading">
              <div>
                <h2 id="activity-heading">Recent activity</h2>
                <p>Authenticated actions on this {noun}</p>
              </div>
              <Icon name="activity" />
            </div>
            {audit.length ? (
              <div className="activity-table-wrap">
                <table className="activity-table">
                  <thead>
                    <tr>
                      <th>Event</th>
                      <th>Operator</th>
                      <th>Time</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {audit.slice(0, 6).map((event) => (
                      <tr key={event.id}>
                        <td title={event.action}>{auditActionLabel(event.action)}</td>
                        <td>{event.actor}</td>
                        <td>{timeAgo(event.created_at * 1000)}</td>
                        <td>
                          <span className={`event-state event-state--${event.result}`}>
                            {event.result}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state">
                <Icon name="shield" />
                <strong>No operator activity yet</strong>
                <span>Privileged actions will appear here.</span>
              </div>
            )}
          </section>
            </>
          )}
        </main>
      </div>

      {selectedAction && (
        <ConfirmDialog
          busy={actionBusy}
          confirmLabel={actionCopy[selectedAction].label}
          description={actionCopy[selectedAction].description}
          onCancel={() => !actionBusy && setSelectedAction(null)}
          onConfirm={() => void performAction()}
          open
          title={actionCopy[selectedAction].title}
        />
      )}
    </div>
  );
}
