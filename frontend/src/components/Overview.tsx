import { Fragment, Suspense, useCallback, useEffect, useRef, useState } from "react";
import type {
  Device,
  Health,
  Session,
  TelemetrySample,
} from "../types";
import { apiRequest } from "../lib/api";
import { brand } from "../lib/brand";
import {
  appendSample,
  connectionStatus,
  sampleTimeMs,
  STALE_AFTER_MS,
  type TelemetryOutcome,
} from "../lib/connectionState";
import { machineNoun, thermalLimits, unknownMachine } from "../lib/machine";
import { healthClaim } from "../lib/health";
import { useMachineProfile } from "../hooks/useMachineProfile";
import { CLUSTER_SUMMARY_REFRESH_MS, clusterData, clusterRoleOf, useClusterSummary } from "../hooks/useClusterSummary";
import { SUMMARY_POLL_MS } from "../hooks/useHomeSummary";
import {
  coolingSectionFromHash,
  resolvedSystemSection,
  SYSTEM_SECTION_EVENT,
  SYSTEM_SECTION_LABELS,
} from "../lib/systemSections";
import { CLUSTER_SECTION_EVENT, CLUSTER_SECTION_LABELS, clusterSectionFrom } from "../lib/clusterSections";
import { TOPBAR_PAGE_SLOT_ID, useReportedPagePlace } from "../lib/topbarSlot";
import { AmbientBackground } from "./AmbientBackground";
import { BuildReloadNotice } from "./BuildReloadNotice";
import { Icon, ICON_SIZE } from "./Icon";
import { Sidebar, type StorageSummary } from "./Sidebar";
import { CommandSearch } from "./CommandSearch";
import { HomePage } from "./HomePage";
import { PowerMenu, type PowerCapabilities } from "./PowerMenu";
import type { ShellTelemetry } from "./SystemCompute";
import {
  hashForPage,
  hashTargetsPage,
  memoryRailItem,
  navigationPages,
  resolveHash,
  standaloneRouteFromHash,
  type NavigationPage,
  type StandaloneRoute,
} from "../lib/navigation";
import { destinations, documentTitleForPage } from "../lib/destinations";
import { SkipLink } from "./SkipLink";
import { ProductMark } from "./ProductMark";
import { WorkspaceErrorBoundary } from "./WorkspaceErrorBoundary";
import { Button, Notice } from "./ui";
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

function WorkspacePage({
  page,
  session,
  onBack,
  telemetry,
}: {
  page: Exclude<NavigationPage, "overview">;
  session: Session;
  onBack: () => void;
  /** The shell's live sample, which System › Compute shows (VD-200). */
  telemetry: ShellTelemetry;
}) {
  switch (page) {
    case "system":
      return <FanControl onBack={onBack} session={session} telemetry={telemetry} />;
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
  // The address bar, for the breadcrumb's System and Cluster sections; set wherever the location is applied.
  const [locationHash, setLocationHash] = useState(() => window.location.hash);
  const [locationSearch, setLocationSearch] = useState(() => window.location.search);
  const [health, setHealth] = useState<Health>({
    status: "offline",
    reasons: [],
    sampled_at: 0,
  });
  const [history, setHistory] = useState<TelemetrySample[]>([]);
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
  const [notice, setNoticeText] = useState(""), [noticeRefused, setNoticeRefused] = useState(false);
  // VD-189 (N3): a refusal is an alert, never an info notice.
  const setNotice = useCallback((message: string, refused = false) => { setNoticeText(message); setNoticeRefused(refused); }, []);
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
  // The console's one `/cluster` read: at Home's pace on Home, slower elsewhere.
  const clusterRead = useClusterSummary(session.user.role, activePage === "overview" ? SUMMARY_POLL_MS : CLUSTER_SUMMARY_REFRESH_MS);
  const cluster = clusterData(clusterRead);
  const [storage, setStorage] = useState<StorageSummary | null>(null);
  const [connectivity, setConnectivity] = useState<{
    dns: boolean;
    internet: boolean;
    latency_ms: number | null;
  } | "unread" | null>(null);
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
    setLocationHash(window.location.hash);
    setLocationSearch(window.location.search);
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
    // System and Cluster move their tab with pushState, which fires no popstate.
    const sectionMoved = () => {
      setLocationHash(window.location.hash);
      setLocationSearch(window.location.search);
    };
    window.addEventListener(SYSTEM_SECTION_EVENT, sectionMoved);
    window.addEventListener(CLUSTER_SECTION_EVENT, sectionMoved);
    return () => {
      window.removeEventListener(SYSTEM_SECTION_EVENT, sectionMoved);
      window.removeEventListener(CLUSTER_SECTION_EVENT, sectionMoved);
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
    setHistory((current) => appendSample(current, sample));
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

  const deviceSaved = useCallback(async (saved: Device) => {
    setDevice(saved);
    await refreshSummary();
  }, [refreshSummary]);

  useEffect(() => {
    if (session.user.role === "viewer") return;
    // The console's own probe is a status read, never audited; the owner's
    // Test press is the POST, always audited (W8-4: not a body flag).
    void apiRequest<{ dns: boolean; internet: boolean; latency_ms: number | null }>(
      "/system/network/status",
    // LESSONS 8: a failed read is "not read", never {internet: false}.
    ).then(setConnectivity).catch(() => setConnectivity("unread"));
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
          setHistory((current) => appendSample(current, sample));
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
    // Home's tiles and System's live readings read every 2.5 s; elsewhere
    // only the top bar and the rail read it.
    const telemetryInterval = window.setInterval(
      pollTelemetry,
      activePage === "overview" || activePage === "system" ? 2_500 : 30_000,
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
      document.hidden ? 10_000 : activePage === "overview" || activePage === "system" ? 1_000 : 5_000,
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
  const resolvedMachine = machine ?? unknownMachine;
  const isAppliance = resolvedMachine.machine_class === "pi-appliance";
  const noun = machineNoun(resolvedMachine.machine_class);
  /*
   * The hero's mark, its sentence and the pill in the page heading are one
   * value. During a telemetry outage the hero read "All systems operational"
   * beside a red dot while the pill still said OPERATIONAL — the dot from the
   * connection state, the words from the last /health answer. Losing contact
   * with a machine is not evidence that its systems are operational.
   */
  const claim = healthClaim(health, connection.state, noun);
  const clusterRole = clusterRoleOf(cluster);
  const shellTelemetry: ShellTelemetry = {
    claim,
    cluster,
    connectionState: connection.state,
    device,
    healthSampledAt: health.sampled_at,
    history,
    metrics,
    noun,
    onDeviceSaved: deviceSaved,
    onNotice: setNotice,
    role: clusterRole,
    sampledAt: lastSample,
    stale: telemetryStale,
    storage,
  };
  /*
   * The board's breadcrumb (VD-200): the page, then where you are on it -
   * "Home / This controller", "System / Compute", "Cluster / Fleet",
   * "Assistant / Routines / Agents". The rail group ("Overview /") named the
   * menu the page sits in, which the rail already shows. Cluster and System
   * keep their place in the address; every other page says its own
   * (`usePagePlace`), in the words of the tab strip it owns.
   */
  const reportedPlace = useReportedPagePlace();
  const crumbPlace: readonly string[] = standaloneRoute === "memory"
    ? [`Used by ${destinations.assistant.name} and ${destinations["ai-chat"].name}`]
    : activePage === "overview"
      ? [`This ${clusterRole ?? noun}`]
      : activePage === "system"
        ? [SYSTEM_SECTION_LABELS[resolvedSystemSection(coolingSectionFromHash(locationHash), resolvedMachine)]]
        : activePage === "fleet"
          ? [CLUSTER_SECTION_LABELS[clusterSectionFrom(locationSearch)]]
          : reportedPlace ?? [];

  return (
    <div className="app-shell">
      <AmbientBackground />
      <SkipLink />
      <Sidebar
        activePage={activePage}
        connection={connection.state}
        connectivity={connectivity}
        health={health}
        machineClass={resolvedMachine.machine_class}
        thermalWarningC={thermalLimits(resolvedMachine).cpuWarn}
        metrics={metrics}
        onNavigate={navigateTo}
        cluster={cluster}
        onSignOut={() => void onLogout()}
        storage={storage}
        user={session.user}
      />

      <div className="workspace">
        <header className="topbar">
          <div className="topbar__identity">
            <span className="topbar__mark" aria-hidden="true"><ProductMark /></span>
            <strong>{brand.name}</strong>
          </div>
          {/*
            * Where you are, as the redesign's breadcrumb: the rail group, then
            * the page. On Home the page name is the page's level-one heading
            * (the greeting below is not a heading); every other page owns its
            * own.
            */}
          <nav aria-label="Breadcrumb" className="topbar__crumb">
            {standaloneRoute === null && activePage === "overview"
              ? <h1 className="topbar__title">{destinations.overview.name}</h1>
              : <strong aria-current={crumbPlace.length ? undefined : "page"}>{standaloneRoute === "memory" ? memoryRailItem.label : destinations[activePage].name}</strong>}
            {/* The whole place is one box that shortens behind one ellipsis: with
                each part its own flex item, only the last could give way, and a
                three-part place ("Assistant / Ask about this machine / Skills")
                scrolled the bar sideways at 768 (VD-200 assist verify S2). */}
            {crumbPlace.length > 0 && (
              <span className="topbar__place">
                {crumbPlace.map((part, index) => (
                  <Fragment key={`${index}-${part}`}>
                    <span aria-hidden="true" className="topbar__sep">/</span>
                    <span aria-current={index === crumbPlace.length - 1 ? "page" : undefined}>{part}</span>
                  </Fragment>
                ))}
              </span>
            )}
          </nav>
          <div className="topbar__actions">
            {/* The open page's pill and actions (`TopbarPageActions`); the shell draws nothing in it. */}
            <div className="topbar__page" id={TOPBAR_PAGE_SLOT_ID} style={{ display: "contents" }} />
            <div
              aria-label="Connection status"
              aria-live="polite"
              className="connection-state"
              data-connection={connection.state}
            >
              <span className={connection.indicatorClassName} />
              <span className="connection-state__label">{connection.label}</span>
              {/* A phone shows the state's one word (the PhoneHome board); a screen reader still hears the sentence and its age. */}
              <span aria-hidden="true" className="connection-state__short">{connection.short}</span>
            </div>
            {/*
              * A paused poller is deliberate (the tab was in the background);
              * this is a one-shot read on demand, not a resume switch. It
              * replaces old Home's polling strip and sits beside the pill, so
              * the pill's own sentence stays one value.
              */}
            {connection.state === "paused" && (
              <Button
                aria-label="Refresh now — read live telemetry once"
                className="connection-state__refresh"
                onClick={refreshTelemetryNow}
                title="Live updates pause while this tab is in the background and resume automatically when it is active."
                variant="quiet"
              >
                Refresh now
              </Button>
            )}
            {activePage === "system" && standaloneRoute === null && (
              <PowerMenu capabilities={powerCapabilities} onNotice={setNotice} session={session} />
            )}
            <CommandSearch allowed={pageAllowed} machine={resolvedMachine} memoryAllowed={standaloneAllowed("memory")} />
            <UserMenu onLogout={onLogout} role={session.user.role} username={session.user.username} />
          </div>
        </header>

        <main className="main" id="main-content" tabIndex={-1}>
          <BuildReloadNotice />
          {/* The outcome of a power request or an enclosure choice, on the page it was made from. */}
          {notice && (
            <Notice className="shell-notice" severity={noticeRefused ? "danger" : "info"}>
              <Icon name="shield" size={ICON_SIZE.nav} />
              {notice}
            </Notice>
          )}
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
                 <WorkspacePage onBack={() => navigateTo("overview", false)} page={activePage} session={session} telemetry={shellTelemetry} />
               </Suspense>
            </WorkspaceErrorBoundary>
          ) : (
            <HomePage
              cluster={clusterRead}
              device={device}
              health={health}
              live={lastSample > 0 && !telemetryStale}
              machine={resolvedMachine}
              metrics={metrics}
              noun={noun}
              pageAllowed={pageAllowed}
              session={session}
            />
          )}
        </main>
      </div>
    </div>
  );
}

/**
 * Who is signed in (VD-200, the board's top bar): the name as an outline pill
 * that opens a menu with the role and Sign out. Sign out was a separate icon
 * button beside it, which on a phone wrapped the top bar onto a second row;
 * on a phone the bottom bar's More sheet carries Sign out instead.
 */
function UserMenu({ onLogout, role, username }: {
  onLogout: () => Promise<void>;
  role: string;
  username: string;
}) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return undefined;
    const close = (event: PointerEvent) => {
      if (!wrap.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [open]);
  return (
    <div
      className="user-menu"
      onKeyDown={(event) => {
        if (event.key !== "Escape" || !open) return;
        event.preventDefault();
        setOpen(false);
        trigger.current?.focus();
      }}
      ref={wrap}
    >
      <Button
        aria-controls="user-menu-list"
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label={`${username}, account`}
        className="operator-chip"
        onClick={() => setOpen((value) => !value)}
        ref={trigger}
        title={`Signed in as ${username} (${role})`}
      >
        <span className="operator-chip__name">{username}</span>
      </Button>
      {open && (
        <div aria-label="Account" className="user-menu__list" id="user-menu-list" role="menu">
          <p className="user-menu__who">Signed in as <strong>{username}</strong> · {role}</p>
          <Button
            autoFocus
            className="user-menu__item"
            onClick={() => { setOpen(false); void onLogout(); }}
            role="menuitem"
            variant="quiet"
          >
            <Icon name="logout" size={16} />
            Sign out
          </Button>
        </div>
      )}
    </div>
  );
}
