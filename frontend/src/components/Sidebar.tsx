import { Button } from "./ui";
import { useRef } from "react";
import type { Health, Metrics, User } from "../types";
import { formatQuantity } from "../lib/format";
import { storagePercent } from "../lib/storage";
import { brand } from "../lib/brand";
import {
  connectionNeedsAttention,
  hardwareSignal,
  type ConnectionState,
} from "../lib/connectionState";
import { hashForPage, type NavigationPage } from "../lib/navigation";
import { destinationDescriptorFor, destinationList } from "../lib/destinations";
import { thermalPolicy, type MachineClass } from "../lib/machine";
import { Icon, type IconName } from "./Icon";
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
const navigationIcons: Record<NavigationPage, IconName> = {
  overview: "grid",
  kvm: "hdmi",
  system: "fan",
  workloads: "database",
  fleet: "network",
  /*
   * Every destination needs its own glyph. Assistant and AI Chat both used
   * "memory", and the rail drops its labels on short viewports - so the two
   * places the app is at pains to say are different became indistinguishable
   * exactly where the label could no longer tell them apart. Assistant reads
   * this machine; AI Chat works over your documents.
   */
  assistant: "cpu",
  "ai-chat": "memory",
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

const mobilePrimaryPages = new Set<NavigationPage>([
  "overview",
  "kvm",
  "system",
  "workloads",
]);

export function Sidebar({
  activePage,
  connection,
  connectivity,
  health,
  machineClass = "pi-appliance",
  thermalWarningC,
  metrics,
  onNavigate,
  storage,
  user,
}: {
  activePage: NavigationPage;
  /**
   * The one connection state the whole shell paints from (#52). A boolean here
   * forced "paused" into either "Healthy" or "Offline", and the rail chose
   * "Offline" — announcing a machine as unreachable because the reader had
   * collapsed the tab group it was sitting in.
   */
  connection: ConnectionState;
  connectivity: { dns: boolean; internet: boolean; latency_ms: number | null } | null;
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
  storage: StorageSummary | null;
  user: User;
}) {
  const mobileMore = useRef<HTMLDetailsElement>(null);
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
  const isRestricted = (page: NavigationPage) =>
    ((page === "assistant" || page === "admin") && user.role !== "administrator") ||
    (page === "ai-chat" && user.role === "viewer") ||
    (page === "activity" && user.role === "viewer");
  const navButton = (
    item: (typeof navigation)[number],
    closeMore = false,
  ) => {
    const restricted = isRestricted(item.page);
    // The accessible name is the canonical name, unchanged, so voice control
    // ("click Settings") reaches the item and a screen reader announces exactly
    // the word on screen (WCAG 2.5.3 Label in Name). The descriptor is support
    // text in the tooltip and never a second name for the same place.
    const accessibleName = item.label;
    const className = activePage === item.page ? "nav-item nav-item--active" : "nav-item";
    const content = (
      <>
        <span className="nav-item__icon">
          <Icon name={item.icon} size={18} />
        </span>
        <span className="nav-item__label">{item.label}</span>
        {item.planned && <span className="nav-item__state">Queued</span>}
      </>
    );
    // Every tooltip keeps the visible label in it, including the restricted
    // case, so hovering never contradicts what is on screen.
    if (item.planned || restricted) {
      return (
        <Button
          aria-label={accessibleName}
          className={className}
          disabled
          key={item.label}
          title={item.planned
            ? `${accessibleName} is not commissioned yet`
            : `${accessibleName} — administrator access required`}
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
        aria-current={activePage === item.page ? "page" : undefined}
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
  const mobileSecondary = navigation.filter((item) => !mobilePrimaryPages.has(item.page));
  const mobileMoreActive = mobileSecondary.some((item) => item.page === activePage);

  return (
    <aside className="sidebar">
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
        {navigation.map((item) => navButton(item))}
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
            <span className="nav-item__icon"><Icon name="settings" size={18} /></span>
            <span className="nav-item__label">More</span>
          </summary>
          <div className="mobile-nav-more__menu">
            <strong>More</strong>
            {mobileSecondary.map((item) => navButton(item, true))}
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
          <Icon name={machineClass === "pi-appliance" ? "fan" : "activity"} size={15} />
          <span>
            <strong>CPU temperature</strong>
            <small>{temperature === null ? "Waiting for sensor" : `${temperature.toFixed(1)}°C${thermalAttention ? ` · above ${thermalLimit}°C` : " · normal"}`}</small>
          </span>
          <Icon name="chevron" size={13} />
        </Button>
        {physicalStorage.length ? physicalStorage.slice(0, 3).map((volume) => (
          <Button className="hardware-signal" key={volume.id} onClick={() => onNavigate("system")} type="button">
            <Icon name={volume.kind === "nvme" ? "nvme" : "database"} size={15} />
            <span>
              <strong>{storageLabel(volume)}</strong>
              <small>{(storagePercent(volume) ?? volume.used_percent).toFixed(0)}% used · {formatQuantity(volume.free_bytes, "free")} free</small>
            </span>
            <Icon name="chevron" size={13} />
          </Button>
        )) : (
          <Button className="hardware-signal" onClick={() => onNavigate("overview")} type="button">
            <Icon name="database" size={15} />
            <span>
              <strong>{storageCount ? `${storageCount} monitored volume${storageCount === 1 ? "" : "s"}` : "Storage"}</strong>
              <small>{storageUsed === null ? "Waiting for disk data" : `${storageUsed.toFixed(0)}% highest use · ${storageAttention ? "space low" : "capacity okay"}`}</small>
            </span>
            <Icon name="chevron" size={13} />
          </Button>
        )}
        {physicalStorage.length > 3 && <small className="hardware-storage-more">+{physicalStorage.length - 3} more device{physicalStorage.length - 3 === 1 ? "" : "s"} in System</small>}
        <Button className="hardware-signal" onClick={() => onNavigate("system")} type="button">
          <Icon name="network" size={15} />
          <span>
            <strong>Internet</strong>
            <small>{connectivity === null ? "Checking connectivity" : connectivity.internet ? `Online${connectivity.latency_ms !== null ? ` · ${connectivity.latency_ms} ms` : ""}` : connectivity.dns ? "DNS works · internet unavailable" : "Offline"}</small>
          </span>
          <Icon name="chevron" size={13} />
        </Button>
        {health.reasons.length > 0 && (
          <Button className="hardware-alert" onClick={() => onNavigate("activity")} type="button">
            <Icon name="alert" size={14} />
            <span>{health.reasons[0]}</span>
          </Button>
        )}
      </section>

    </aside>
  );
}
