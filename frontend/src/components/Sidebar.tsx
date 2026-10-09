import { Button } from "./ui";
import type { FleetSummary } from "./fleetTypes";
import { useEffect, useRef, useState } from "react";
import { useMobileNavClearance } from "../hooks/useMobileNavClearance";
import type { Health, Metrics, User } from "../types";
import { formatQuantity } from "../lib/format";
import { storagePercent } from "../lib/storage";
import { brand } from "../lib/brand";
import {
  connectionNeedsAttention,
  hardwareSignal,
  type ConnectionState,
} from "../lib/connectionState";
import { hashForPage, memoryRailItem, standaloneRouteFromHash, type NavigationPage } from "../lib/navigation";
import { destinationDescriptorFor, destinationList } from "../lib/destinations";
import { thermalPolicy, type MachineClass } from "../lib/machine";
import { Icon, ICON_SIZE, type IconName } from "./Icon";
import { ProductMark } from "./ProductMark";

export interface StorageVolume {
  id: string;
  device_id: string;
  model: string;
  kind: "nvme" | "usb" | "microsd" | "sata" | "other";
  mountpoint: string;
  free_bytes: number;
  total_bytes: number;
  used_percent: number;
  /**
   * Served by `linux_storage.py` and, until now, not read: the client rebuilt
   * it as `total - free`, which counts the root-reserved blocks as used and put
   * the byte figure ~5 points out of step with the percentage beside it.
   */
  used_bytes?: number;
  /**
   * The filesystem's own reserved blocks: `total - used - free`, served rather
   * than derived. `free_bytes` is `f_bavail`, what an unprivileged process may
   * still write; `total - used` is `f_bfree`, which also counts the reserve. A
   * live tester read 220.7 GB free off one surface and 231.3 GB off another for
   * the same volume in the same second, and the reserve is exactly that
   * difference — about 4.6%, which is close enough to `1/1.048576 ≈ 4.63%`
   * that it reads as a GiB/GB unit error and is not one.
   */
  reserved_bytes?: number;
}

export interface StorageSummary {
  volumes: StorageVolume[];
  media_counts: Record<string, number>;
  media_presence: Record<string, { present: boolean; count: number }>;
}

const storageLabel = (volume: StorageVolume) => {
  const medium = {
    nvme: "NVMe",
    usb: "USB storage",
    microsd: "microSD",
    sata: "SATA storage",
    other: "Storage",
  }[volume.kind];
  return `${medium} · ${volume.mountpoint}`;
};

// The sidebar never invents a name. It reads the one canonical name for each
// destination so the item the user clicks, the heading they land on, and the
// browser tab all say the same word.
// The approved Hugeicons glyph for each destination (VD-200, IconsHugeicons).
const navigationIcons: Record<NavigationPage, IconName> = {
  overview: "home",
  kvm: "console",
  system: "system",
  workloads: "apps",
  fleet: "cluster",
  /*
   * Every destination needs its own glyph. Assistant and AI Chat both used
   * "memory", and the rail drops its labels on short viewports - so the two
   * places the app is at pains to say are different became indistinguishable
   * exactly where the label could no longer tell them apart. Assistant reads
   * this machine; AI Chat works over your documents.
   */
  assistant: "assistant",
  "ai-chat": "chat",
  activity: "activity",
  admin: "settings",
};

function navigationItems(machineClass: MachineClass): Array<{
  page: NavigationPage;
  label: string;
  descriptor: string;
  icon: IconName;
  planned?: boolean;
}> {
  return destinationList.map((destination) => ({
    page: destination.page,
    label: destination.name,
    descriptor: destinationDescriptorFor(destination.page, machineClass),
    icon: navigationIcons[destination.page],
  }));
}

/**
 * The desktop rail's three groups, in the order the redesign mockup shows
 * them (VD-200). Every destination is in exactly one group; Sidebar.test.tsx
 * holds that, so a new destination cannot silently drop out of the rail. The
 * phone bar keeps today's order and does not use the groups.
 */
export const navigationGroups: ReadonlyArray<{ label: string; pages: readonly NavigationPage[] }> = [
  { label: "Overview", pages: ["overview", "fleet", "system", "kvm"] },
  { label: "AI", pages: ["workloads", "ai-chat", "assistant"] },
  { label: "Manage", pages: ["activity", "admin"] },
];

/** Whether the address bar is on the Memory page, followed as it changes. */
function useOnMemoryRoute(): boolean {
  const read = () => standaloneRouteFromHash(window.location.hash) === memoryRailItem.route;
  const [onMemory, setOnMemory] = useState(read);
  useEffect(() => {
    const update = () => setOnMemory(read());
    window.addEventListener("hashchange", update);
    window.addEventListener("popstate", update);
    return () => {
      window.removeEventListener("hashchange", update);
      window.removeEventListener("popstate", update);
    };
  }, []);
  return onMemory;
}

/** The worker software states the board words as "<name> software differs from its profile". */
const PROFILE_DIFFERS_STATES = new Set(["full-appliance-installed", "full-appliance-partial"]);

/** Where a storage row opens: System › Hardware and services, whose first card is Storage. */
const STORAGE_HASH = "#/system/hardware";

/**
 * The rail's warning card for one worker, or null when its software is not
 * a warning. The board's sentence only where it is true (the full appliance on
 * the machine, or part of it); every other state keeps the backend's own
 * label (VD-173). An old reading is grey and gives its age (LESSONS 10).
 */
export function workerAlert(node: FleetSummary["enrolled_nodes"][number]): { text: string; tone: "warning" | "danger" | "stale" } | null {
  const software = node.worker_software;
  if (!software || (software.tone !== "warning" && software.tone !== "danger")) return null;
  const words = PROFILE_DIFFERS_STATES.has(software.state)
    ? `${node.name} software differs from its profile`
    : `${node.name}: ${software.label}`;
  if (software.stale) return { text: `${words} · ${software.checked || "reading is old"}`, tone: "stale" };
  return { text: words, tone: software.tone };
}

/** Why a role cannot open a destination: a short word for the row, the full sentence for the tooltip. */
interface NavRestriction { who: string; full: string }
const ADMIN_ONLY: NavRestriction = { who: "admin", full: "administrator access required" };
const OPERATOR_OR_ADMIN: NavRestriction = { who: "operator", full: "operator or administrator access required" };

/**
 * The row's reason, "Needs admin" / "Needs operator". On the 92 px rail
 * (721-1024 px) it wrapped to two lines under every closed row and doubled
 * each one's height (88 px against 42); there it draws as a lock and the one
 * role word on a single line, and "Needs" stays in the text for a screen
 * reader (shell.css, polish audit). The text is the row's description either way.
 */
function NavReason({ id, restriction }: { id: string; restriction: NavRestriction }) {
  return (
    <span className="nav-item__reason" id={id}>
      <Icon className="nav-item__reason-icon" name="lock" size={ICON_SIZE.inline} />
      <span className="nav-item__reason-verb">Needs</span>{" "}{restriction.who}
    </span>
  );
}

const mobilePrimaryPages = new Set<NavigationPage>([
  "overview",
  "kvm",
  "system",
  "workloads",
]);

export function Sidebar({
  activePage,
  cluster = null,
  connection,
  connectivity,
  health,
  machineClass = "pi-appliance",
  thermalWarningC,
  metrics,
  onNavigate,
  onSignOut,
  storage,
  user,
}: {
  activePage: NavigationPage;
  /** The shell's `/cluster` read: each worker's software state, for the warning card (VD-200). */
  cluster?: FleetSummary | null;
  /**
   * The one connection state the whole shell paints from (#52). A boolean here
   * forced "paused" into either "Healthy" or "Offline", and the rail chose
   * "Offline" — announcing a machine as unreachable because the reader had
   * collapsed the tab group it was sitting in.
   */
  connection: ConnectionState;
  /** null while the read is in flight; "unread" when it failed (LESSONS 8: not "Offline"). */
  connectivity: { dns: boolean; internet: boolean; latency_ms: number | null } | "unread" | null;
  health: Health;
  /**
   * Defaults to the appliance because that is the machine Vaelor shipped on;
   * Home overrides it as soon as discovery answers.
   */
  machineClass?: MachineClass;
  /** The served warning temperature, when discovery states one. */
  thermalWarningC?: number;
  metrics: Metrics;
  onNavigate: (page: NavigationPage) => void;
  /** Sign out, for the phone's More sheet: the top bar has no room for it there. */
  onSignOut?: () => void;
  storage: StorageSummary | null;
  user: User;
}) {
  const mobileMore = useRef<HTMLDetailsElement>(null);
  /*
   * Memory is the shell's standalone route, so while it is open the shell's
   * `activePage` still names the page before it. The rail marks Memory alone
   * then; only an administrator reaches it (the shell sends anyone else Home).
   */
  const memoryAllowed = user.role === "administrator";
  const onMemory = useOnMemoryRoute() && memoryAllowed;
  const currentPage: NavigationPage | null = onMemory ? null : activePage;
  const bar = useRef<HTMLElement>(null);
  useMobileNavClearance(bar);
  const temperature = typeof metrics.cpu_temperature === "number" ? metrics.cpu_temperature : null;
  const telemetryStorageValues = Object.keys(metrics)
    .filter((key) => /^disk_.+_percent$/.test(key) && typeof metrics[key] === "number")
    .map((key) => Number(metrics[key]));
  const physicalStorage = Array.from(
    (storage?.volumes ?? []).reduce((devices, volume) => {
      const current = devices.get(volume.device_id);
      if (!current || volume.total_bytes > current.total_bytes) devices.set(volume.device_id, volume);
      return devices;
    }, new Map<string, StorageVolume>()).values(),
  );
  // Derived from the same bytes Home shows, not from the served percentage,
  // so the rail and the card cannot state two different numbers for one disk.
  const storageValues = physicalStorage
    .map((volume) => storagePercent(volume))
    .filter((value): value is number => value !== null);
  if (!storageValues.length) storageValues.push(...telemetryStorageValues);
  const storageUsed = storageValues.length ? Math.max(...storageValues) : null;
  const storageCount = physicalStorage.length || telemetryStorageValues.length;
  const navigation = navigationItems(machineClass);
  /*
   * 70 °C is a Pi 5 constant. A workstation processor that boosts to ~95 °C by
   * design sat permanently in "Attention" under it, which is alarm fatigue
   * caused entirely by a threshold borrowed from another machine.
   */
  const thermalLimit = thermalWarningC ?? thermalPolicy(machineClass).cpuWarn;
  const thermalAttention = temperature !== null && temperature >= thermalLimit;
  const storageAttention = storageUsed !== null && storageUsed >= 85;
  /*
   * A suspended poller is not an unhealthy machine, so `paused` and `unknown`
   * do not raise attention — they say plainly that the rail does not currently
   * know, which is the honest third answer. Only `error` means readings have
   * stopped arriving while Vaelor was still asking for them.
   */
  const needsAttention = connectionNeedsAttention(connection)
    || health.status !== "healthy" || thermalAttention || storageAttention;
  const signal = hardwareSignal(connection, needsAttention);
  /**
   * Why a destination is closed to this role, or null when it is open. The
   * reason is drawn on the row itself (VD-200 S-H7: a disabled control shows
   * its reason beside it, not only in a tooltip a touch or keyboard reader
   * never sees); the tooltip repeats it in full.
   */
  const restriction = (page: NavigationPage): NavRestriction | null => {
    if ((page === "assistant" || page === "admin") && user.role !== "administrator") return ADMIN_ONLY;
    if ((page === "ai-chat" || page === "activity") && user.role === "viewer") return OPERATOR_OR_ADMIN;
    return null;
  };
  const navButton = (
    item: (typeof navigation)[number],
    closeMore = false,
  ) => {
    const restricted = restriction(item.page);
    // The accessible name is the canonical name, unchanged, so voice control
    // ("click Settings") reaches the item and a screen reader announces exactly
    // the word on screen (WCAG 2.5.3 Label in Name). The descriptor is support
    // text in the tooltip and never a second name for the same place.
    const accessibleName = item.label;
    const className = currentPage === item.page ? "nav-item nav-item--active" : "nav-item";
    const content = (
      <>
        <span className="nav-item__icon">
          <Icon name={item.icon} size={ICON_SIZE.nav} />
        </span>
        <span className="nav-item__label">{item.label}</span>
        {item.planned && <span className="nav-item__state">Queued</span>}
        {!item.planned && restricted && <NavReason id={`nav-reason-${item.page}${closeMore ? "-more" : ""}`} restriction={restricted} />}
      </>
    );
    // Every tooltip keeps the visible label in it, including the restricted
    // case, so hovering never contradicts what is on screen.
    if (item.planned || restricted) {
      return (
        <Button
          aria-describedby={restricted && !item.planned ? `nav-reason-${item.page}${closeMore ? "-more" : ""}` : undefined}
          aria-label={accessibleName}
          className={className}
          disabled
          key={item.label}
          title={item.planned || !restricted
            ? `${accessibleName} is not commissioned yet`
            : `${accessibleName} — ${restricted.full}`}
        >
          {content}
        </Button>
      );
    }
    // A real link, not a button (#150): these are places, the hash routes
    // already exist, and Home's content cards are links — so middle-click,
    // open-in-new-tab, and copy-link work here too. The hashchange listener
    // performs the navigation; the click handler only dismisses the mobile
    // sheet.
    return (
      <a
        aria-current={currentPage === item.page ? "page" : undefined}
        aria-label={accessibleName}
        className={className}
        href={hashForPage(item.page)}
        key={item.label}
        onClick={() => {
          if (closeMore && mobileMore.current) mobileMore.current.open = false;
        }}
        title={`${accessibleName} — ${item.descriptor}`}
      >
        {content}
      </a>
    );
  };
  /** The Memory item, placed after the Assistant in the rail and in the phone's More sheet. */
  const memoryItem = (closeMore = false) => {
    const content = (
      <>
        <span className="nav-item__icon"><Icon name="aiMemory" size={ICON_SIZE.nav} /></span>
        <span className="nav-item__label">{memoryRailItem.label}</span>
        {!memoryAllowed && <NavReason id={`nav-reason-memory${closeMore ? "-more" : ""}`} restriction={ADMIN_ONLY} />}
      </>
    );
    if (!memoryAllowed) {
      return (
        <Button
          aria-describedby={`nav-reason-memory${closeMore ? "-more" : ""}`}
          aria-label={memoryRailItem.label}
          className="nav-item"
          disabled
          key={memoryRailItem.route}
          title={`${memoryRailItem.label} — ${ADMIN_ONLY.full}`}
        >
          {content}
        </Button>
      );
    }
    return (
      <a
        aria-current={onMemory ? "page" : undefined}
        aria-label={memoryRailItem.label}
        className={onMemory ? "nav-item nav-item--active" : "nav-item"}
        href={memoryRailItem.hash}
        key={memoryRailItem.route}
        onClick={() => {
          if (closeMore && mobileMore.current) mobileMore.current.open = false;
        }}
        title={`${memoryRailItem.label} — ${memoryRailItem.descriptor}`}
      >
        {content}
      </a>
    );
  };
  const withMemory = (items: typeof navigation, closeMore = false) => items.flatMap((item) => (
    item.page === memoryRailItem.after ? [navButton(item, closeMore), memoryItem(closeMore)] : [navButton(item, closeMore)]
  ));
  const mobileSecondary = navigation.filter((item) => !mobilePrimaryPages.has(item.page));
  const mobileMoreActive = onMemory || mobileSecondary.some((item) => item.page === activePage);

  return (
    <aside className="sidebar" ref={bar}>
      <div className="sidebar__brand" aria-label={brand.controlPlane}>
        <div className="brand-glyph" aria-hidden="true">
          <ProductMark />
        </div>
        <div className="brand-copy">
          <strong>{brand.name}</strong>
          <span>Control plane</span>
        </div>
      </div>

      <nav aria-label="Primary navigation" className="sidebar__nav sidebar__nav--desktop">
        {navigationGroups.map((group) => (
          <div aria-labelledby={`nav-group-${group.label.toLowerCase()}`} className="nav-group" key={group.label} role="group">
            <span className="nav-group__label" id={`nav-group-${group.label.toLowerCase()}`}>{group.label}</span>
            {withMemory(group.pages.flatMap((page) => navigation.filter((item) => item.page === page)))}
          </div>
        ))}
      </nav>
      <nav aria-label="Primary navigation" className="sidebar__nav sidebar__nav--mobile">
        {/* Navigating from the bar behind the sheet must dismiss the sheet. */}
        {navigation.filter((item) => mobilePrimaryPages.has(item.page)).map((item) => navButton(item, true))}
        {/*
          * A <details> ignores Escape, so this full-screen sheet stayed open
          * over whatever you navigated to next. Close it the way every other
          * overlay in the app closes.
          */}
        <details
          className="mobile-nav-more"
          onKeyDown={(event) => {
            if (event.key !== "Escape" || !mobileMore.current?.open) return;
            event.stopPropagation();
            mobileMore.current.open = false;
            mobileMore.current.querySelector("summary")?.focus();
          }}
          ref={mobileMore}
        >
          <summary
            aria-label="More navigation"
            className={mobileMoreActive ? "nav-item nav-item--active" : "nav-item"}
          >
            <span className="nav-item__icon"><Icon name="settings" size={ICON_SIZE.nav} /></span>
            <span className="nav-item__label">More</span>
          </summary>
          <div className="mobile-nav-more__menu">
            <strong>More</strong>
            {withMemory(mobileSecondary, true)}
            {onSignOut && (
              <Button
                className="nav-item mobile-nav-more__sign-out"
                onClick={() => {
                  if (mobileMore.current) mobileMore.current.open = false;
                  onSignOut();
                }}
              >
                <span className="nav-item__icon"><Icon name="logout" size={ICON_SIZE.nav} /></span>
                <span className="nav-item__label">Sign out</span>
              </Button>
            )}
          </div>
        </details>
      </nav>

      <section className={`sidebar__hardware ${needsAttention ? "sidebar__hardware--attention" : ""}`} aria-labelledby="hardware-signals-title">
        <div className="hardware-signals__heading">
          <span id="hardware-signals-title">{machineClass === "pi-appliance" ? "Appliance health" : "Machine health"}</span>
          {/*
            * Task #74: the light and the word are one value. They used to be
            * two — the light from `connection`, the colour of the word from a
            * stylesheet default that was green unless `needsAttention` took it
            * away — so the rail read PAUSED in success green beside a grey
            * light. Nothing here can be sourced separately any more.
            */}
          <strong className={signal.labelClassName}>
            <span className={signal.markClassName} />
            {signal.word}
          </strong>
        </div>
        <Button className="hardware-signal" onClick={() => onNavigate("system")} type="button">
          <Icon name="temperature" size={ICON_SIZE.nav} />
          <span>
            <strong>CPU temperature</strong>
            <small>{temperature === null ? "Waiting for sensor" : `${temperature.toFixed(1)}°C${thermalAttention ? ` · above ${thermalLimit}°C` : " · normal"}`}</small>
          </span>
          <Icon name="chevron" size={ICON_SIZE.inline} />
        </Button>
        {physicalStorage.length ? physicalStorage.slice(0, 3).map((volume) => {
          const used = storagePercent(volume) ?? volume.used_percent;
          return (
            <Button className="hardware-signal" key={volume.id} onClick={() => { window.location.hash = STORAGE_HASH; }} type="button">
              <Icon name="drive" size={ICON_SIZE.nav} />
              <span>
                <strong>{storageLabel(volume)}</strong>
                {/* The same percentage as the words under it, drawn (VD-200 board). */}
                <span aria-hidden="true" className="hardware-signal__bar"><i style={{ width: `${Math.min(100, Math.max(0, used))}%` }} /></span>
                <small>{used.toFixed(0)}% used · {formatQuantity(volume.free_bytes, "free")} free</small>
              </span>
              <Icon name="chevron" size={ICON_SIZE.inline} />
            </Button>
          );
        }) : (
          <Button className="hardware-signal" onClick={() => { window.location.hash = STORAGE_HASH; }} type="button">
            <Icon name="drive" size={ICON_SIZE.nav} />
            <span>
              <strong>{storageCount ? `${storageCount} monitored volume${storageCount === 1 ? "" : "s"}` : "Storage"}</strong>
              <small>{storageUsed === null ? "Waiting for disk data" : `${storageUsed.toFixed(0)}% highest use · ${storageAttention ? "space low" : "capacity okay"}`}</small>
            </span>
            <Icon name="chevron" size={ICON_SIZE.inline} />
          </Button>
        )}
        {physicalStorage.length > 3 && <small className="hardware-storage-more">+{physicalStorage.length - 3} more device{physicalStorage.length - 3 === 1 ? "" : "s"} in System</small>}
        <Button className="hardware-signal" onClick={() => onNavigate("system")} type="button">
          <Icon name="network" size={ICON_SIZE.nav} />
          <span>
            <strong>Internet</strong>
            <small>{connectivity === null ? "Checking connectivity" : connectivity === "unread" ? "Connectivity not read" : connectivity.internet ? `Online${connectivity.latency_ms !== null ? ` · ${connectivity.latency_ms} ms` : ""}` : connectivity.dns ? "DNS works · internet unavailable" : "Offline"}</small>
          </span>
          <Icon name="chevron" size={ICON_SIZE.inline} />
        </Button>
        {/*
          * The board ends Machine health with a warning card for a worker
          * whose software differs from its profile or whose check failed -
          * the same warning and danger tones Home's attention list keys on,
          * in the backend's own words (VD-173). It opens that machine's card
          * on Cluster.
          */}
        {(cluster?.enrolled_nodes ?? []).map((node) => ({ node, alert: workerAlert(node) })).filter(({ alert }) => alert !== null).map(({ node, alert }) => (
          <a
            className={`hardware-alert hardware-alert--${alert!.tone}`}
            href={`?cluster=fleet&machine=${encodeURIComponent(node.id)}#/fleet`}
            key={node.id}
          >
            <Icon name="alert" size={ICON_SIZE.inline} />
            <span>{alert!.text}</span>
          </a>
        ))}
        {health.reasons.length > 0 && (
          <Button className="hardware-alert" onClick={() => onNavigate("activity")} type="button">
            <Icon name="alert" size={ICON_SIZE.inline} />
            <span>{health.reasons[0]}</span>
          </Button>
        )}
      </section>

    </aside>
  );
}
