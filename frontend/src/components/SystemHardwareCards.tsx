import { useState } from "react";
import { formatQuantity, timeAgo } from "../lib/format";
import { storageFigures, storagePercent } from "../lib/storage";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { SectionCard } from "./systemUi";
import { Button, ListRow, MeterBar } from "./ui";

/*
 * System › Hardware and services, its first row (VD-200, the System board):
 * Storage beside Network. The data is read by SystemInventoryPanel and handed
 * down; these cards only draw it.
 */

export interface StorageDevice {
  name: string;
  path?: string;
  type?: string;
  size?: number;
  model?: string;
  tran?: string;
  mountpoints?: Array<string | null>;
  children?: StorageDevice[];
}

export interface StorageVolume {
  id: string;
  device_id: string;
  name: string;
  path: string;
  model: string;
  kind: "nvme" | "usb" | "microsd" | "sata" | "other";
  mountpoint: string;
  total_bytes: number;
  used_bytes: number;
  /** The filesystem's reserved blocks (`linux_storage.py`); older appliances omit it. */
  reserved_bytes?: number;
  free_bytes: number;
  used_percent: number;
  removable: boolean;
}

export interface NetworkInterface {
  name: string;
  state: string;
  addresses: Array<{ family: string; address: string; prefix: number }>;
  rx_bytes: number;
  tx_bytes: number;
  rx_errors: number;
  tx_errors: number;
}

export interface Connectivity {
  dns: boolean;
  internet: boolean;
  /** `null` where the probe could not time a round trip. */
  latency_ms: number | null;
}

const mediumLabel = (kind: string) => ({
  nvme: "NVMe",
  usb: "USB storage",
  microsd: "microSD",
  sata: "SATA storage",
  other: "Storage",
}[kind] ?? "Storage");

/**
 * One volume in words: used, free and total, with the rail's percentage -
 * every figure the reserve note says adds up, drawn (inventory review S2).
 */
function volumeLine(volume: StorageVolume): string {
  const used = storageFigures(volume).usedBytes;
  const percent = (storagePercent(volume) ?? volume.used_percent).toFixed(0);
  return [
    used === null ? `${percent}% used` : `${formatQuantity(used, "used")} used (${percent}%)`,
    `${formatQuantity(volume.free_bytes, "free")} free`,
    `${formatQuantity(volume.total_bytes, "capacity")} total`,
  ].join(" · ");
}

const flattenDevices = (devices: StorageDevice[]): StorageDevice[] =>
  devices.flatMap((device) => [device, ...flattenDevices(device.children ?? [])]);

export function StorageCard({
  devices,
  loading,
  mediaSlots,
  notes,
  temperatures,
  volumes,
}: {
  devices: StorageDevice[];
  loading: boolean;
  mediaSlots: Array<{ kind: string; label: string; state: string }>;
  /** The filesystem reserve and the telemetry retention sentences, when they apply. */
  notes: string[];
  temperatures: Record<string, number>;
  volumes: StorageVolume[];
}) {
  const [expandedId, setExpandedId] = useState("");
  const groups = Object.values(volumes.reduce<Record<string, StorageVolume[]>>((byDevice, volume) => {
    const key = volume.device_id || volume.path || volume.id;
    (byDevice[key] ??= []).push(volume);
    return byDevice;
  }, {}));
  const disks = flattenDevices(devices).filter((device) => device.type === "disk");
  const count = groups.length || disks.length;

  return (
    <SectionCard
      actions={!loading && <span>{count} device{count === 1 ? "" : "s"}</span>}
      className="sys-storage"
      flush
      footer={(notes.length > 0 || mediaSlots.length > 0) && (
        <div className="sys-footnotes">
          {notes.map((note) => <p key={note}>{note}</p>)}
          {mediaSlots.length > 0 && (
            <dl aria-label="Removable media slots" className="sys-slots">
              {mediaSlots.map((slot) => (
                <div key={slot.kind}><dt>{slot.label}</dt><dd>{slot.state}</dd></div>
              ))}
            </dl>
          )}
        </div>
      )}
      title="Storage"
    >
      {groups.length ? groups.map((group) => {
        const primary = [...group].sort((left, right) => right.total_bytes - left.total_bytes)[0];
        const storageId = primary.device_id || primary.path || primary.id;
        const detailsId = `storage-details-${storageId.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
        const expanded = expandedId === storageId;
        const temperature = Object.entries(temperatures)
          .find(([key]) => key.includes(storageId) || key.includes(primary.name))?.[1];
        const mountCount = new Set(group.map((volume) => volume.mountpoint).filter(Boolean)).size;
        const percent = storagePercent(primary) ?? primary.used_percent;
        return (
          <div className="sys-drive" key={storageId}>
            {/* The row is the disclosure for the drive's mounts and facts. */}
            <Button
              aria-controls={expanded ? detailsId : undefined}
              aria-expanded={expanded}
              className="sys-drive__trigger"
              onClick={() => setExpandedId(expanded ? "" : storageId)}
              type="button"
              variant="quiet"
            >
              <span className="sys-drive__title">
                <strong>{mediumLabel(primary.kind)} · {primary.model || primary.name}</strong>
                <small>{formatQuantity(primary.total_bytes, "capacity")} total · {formatQuantity(primary.free_bytes, "free")} free · {mountCount} mount{mountCount === 1 ? "" : "s"}</small>
              </span>
              <span className="sys-drive__temp">{temperature === undefined ? "" : `${temperature} °C`}</span>
              <Icon aria-hidden="true" className="sys-drive__chevron" name="chevron" size={16} />
            </Button>
            <MeterBar fraction={Number.isFinite(percent) ? percent / 100 : null} label={`${Number.isFinite(percent) ? percent.toFixed(0) : "—"}% used`} />
            {expanded && (
              <div className="sys-drive__details" id={detailsId}>
                <div className="sys-drive__mounts">
                  <strong>Mounted volumes</strong>
                  {group.map((volume) => (
                    <div className="sys-volume" key={volume.id}>
                      <span>{volume.mountpoint || "Not mounted"}</span>
                      {/* The rail's percentage, from the same bytes (lib/storage). */}
                      <small>{volumeLine(volume)}</small>
                    </div>
                  ))}
                </div>
                <dl className="sys-drive__facts">
                  <div><dt>Device</dt><dd>{primary.device_id || primary.path || primary.name}</dd></div>
                  <div><dt>Connection</dt><dd>{mediumLabel(primary.kind)}</dd></div>
                  <div><dt>Drive temperature</dt><dd>{temperature === undefined ? "Not reported" : `${temperature}°C`}</dd></div>
                </dl>
              </div>
            )}
          </div>
        );
      }) : disks.length ? disks.map((device) => (
        <ListRow
          detail={`${device.path ?? device.name} · ${formatQuantity(device.size, "capacity")}`}
          icon={<Icon name="drive" size={18} />}
          key={device.path ?? device.name}
          title={device.model?.trim() || device.name}
        />
      )) : <p className="sys-empty">{loading ? "Loading storage devices…" : "No block-device details were returned."}</p>}
    </SectionCard>
  );
}

function interfaceState(state: string): { label: string; tone: "success" | "neutral" } {
  if (state === "up") return { label: "Connected", tone: "success" };
  if (state === "down") return { label: "Down", tone: "neutral" };
  return { label: state ? state.charAt(0).toUpperCase() + state.slice(1) : "Not reported", tone: "neutral" };
}

export function NetworkCard({
  busy,
  canTest,
  checkedAt,
  connectivity,
  interfaces,
  loading,
  onTest,
  refusal,
}: {
  busy: boolean;
  canTest: boolean;
  checkedAt: number;
  connectivity: Connectivity | null;
  interfaces: NetworkInterface[];
  loading: boolean;
  onTest: () => void;
  /** Why the last test could not run, in the card it was pressed in. */
  refusal: string;
}) {
  const shown = interfaces.filter((item) => item.name !== "lo");
  return (
    <SectionCard
      actions={(
        <Button
          disabled={!canTest || busy}
          disabledReason={!canTest ? "Operator access is required to run the connection test." : undefined}
          onClick={onTest}
        >
          {busy ? "Testing…" : connectivity ? "Test again" : "Test internet connection"}
        </Button>
      )}
      className="sys-network"
      flush
      title="Network"
    >
      {/*
        * The outcome sits directly under the button that produced it, above
        * the interface list: a Docker host lists dozens of interfaces, and a
        * result drawn after them landed below the fold, so a tester pressed
        * three times and saw nothing. The time makes a repeat run visible.
        */}
      {connectivity && (
        <div aria-live="polite" className="sys-connectivity" role="status">
          <StatusPill label={connectivity.internet ? "Internet reachable" : "Internet unavailable"} tone={connectivity.internet ? "success" : "warning"} />
          <span>
            DNS {connectivity.dns ? "working" : "failed"}
            {connectivity.latency_ms === null ? "" : ` · round trip to the internet ${connectivity.latency_ms} ms`}
            {checkedAt ? ` · checked ${timeAgo(checkedAt)}` : ""}
          </span>
        </div>
      )}
      {refusal && <p className="sys-refusal" role="alert">{refusal}</p>}
      {shown.map((item) => {
        const state = interfaceState(item.state);
        const addresses = item.addresses.map((address) => address.address).join(", ");
        return (
          <ListRow
            detail={[
              addresses || "No address",
              `Received ${formatQuantity(item.rx_bytes, "transfer")} · Sent ${formatQuantity(item.tx_bytes, "transfer")} · ${item.rx_errors + item.tx_errors} errors`,
            ].join(" · ")}
            icon={<Icon name="network" size={18} />}
            key={item.name}
            title={item.name}
            trailing={<StatusPill label={state.label} tone={state.tone} />}
          />
        );
      })}
      {loading && <p className="sys-empty">Loading network interfaces…</p>}
      {/* Without this the card resolved to its heading alone, which reads as broken. */}
      {!loading && !shown.length && (
        <p className="sys-empty">
          No network interfaces were reported. The connection test still works, and the sidebar shows current internet status.
        </p>
      )}
    </SectionCard>
  );
}
