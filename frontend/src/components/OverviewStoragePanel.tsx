import type { Metrics } from "../types";
import { formatPercent, formatQuantity } from "../lib/format";
import { metricNumber } from "../lib/metrics";
import { storageFigures, storagePercent } from "../lib/storage";
import { Icon } from "./Icon";
import type { StorageSummary, StorageVolume } from "./Sidebar";
import { UnavailableValue } from "./ui";

interface StorageReading {
  percent: number;
  usedBytes: number | null;
  totalBytes: number | null;
  freeBytes: number | null;
  /** `null` when the volume did not report one; zero means it reported none. */
  reservedBytes: number | null;
  deviceCount: number;
  source: "devices" | "telemetry";
}

/**
 * Home's storage panel was built on `disk_*_percent` telemetry keys, which
 * only the Pironman bridge emits. On any other host it read a permanent "—"
 * while the sidebar, on the same screen, showed the same NVMe correctly from
 * `/system/storage` — two contradictory storage answers side by side. Worse,
 * the progressbar under it set `aria-valuenow={0}`, so a screen-reader user
 * was told "Storage 0%": a real-sounding reading for a panel that had no
 * reading at all.
 *
 * The discovered device list is the primary source now, exactly as the sidebar
 * uses it, and the telemetry keys are kept only as a fallback for hosts that
 * do emit them. When neither answers, the panel says so.
 */
export function storageReading(
  storage: StorageSummary | null,
  metrics: Metrics,
): StorageReading | null {
  const byDevice = (storage?.volumes ?? []).reduce((devices, volume) => {
    const current = devices.get(volume.device_id);
    if (!current || volume.total_bytes > current.total_bytes) devices.set(volume.device_id, volume);
    return devices;
  }, new Map<string, StorageVolume>());
  const devices = [...byDevice.values()];
  if (devices.length) {
    // Ranked on the same derived figure the card then prints, so "highest use"
    // and the number under it are the same measurement.
    const busiest = devices.reduce((highest, volume) =>
      (storagePercent(volume) ?? -1) > (storagePercent(highest) ?? -1) ? volume : highest);
    const figures = storageFigures(busiest);
    return {
      percent: figures.percent ?? 0,
      usedBytes: figures.usedBytes,
      totalBytes: figures.totalBytes,
      freeBytes: Number.isFinite(busiest.free_bytes) ? busiest.free_bytes : null,
      reservedBytes: figures.reservedBytes,
      deviceCount: devices.length,
      source: "devices",
    };
  }

  const telemetryKeys = Object.keys(metrics)
    .filter((key) => /^disk_.+_percent$/.test(key) && metricNumber(metrics, key) !== null);
  if (!telemetryKeys.length) return null;
  const busiestKey = telemetryKeys.reduce((highest, key) =>
    (metricNumber(metrics, key) ?? 0) > (metricNumber(metrics, highest) ?? 0) ? key : highest);
  return {
    percent: metricNumber(metrics, busiestKey) ?? 0,
    usedBytes: metricNumber(metrics, busiestKey.replace("_percent", "_used")),
    totalBytes: metricNumber(metrics, busiestKey.replace("_percent", "_total")),
    freeBytes: metricNumber(metrics, busiestKey.replace("_percent", "_free")),
    // The telemetry keys carry no reserve, and a bridge that never mentioned
    // one has not told us it has none.
    reservedBytes: null,
    deviceCount: telemetryKeys.length,
    source: "telemetry",
  };
}

/**
 * What the filesystem holds back, said out loud.
 *
 * "201 GB used" and "220.7 GB free" on a 1.01 TB volume leave about 10 GB
 * unaccounted for, and a reader looking at two correct numbers with a hole
 * between them concludes one of them is wrong. `statvfs` reports `f_bavail` —
 * what can still be written — while `total - used` is `f_bfree`, which also
 * counts the root reserve. **There is no unit error here**, and the reason
 * that is worth saying is that the reserve is about 4.6% and `1/1.048576` is
 * about 4.63%: a reserve looks exactly like a GiB/GB mistake, so the only way
 * to stop it being read as one is to name it.
 */
export function reserveNote(reading: StorageReading | null): string {
  if (!reading?.reservedBytes) return "";
  return (
    `${formatQuantity(reading.reservedBytes, "capacity")} of this volume is reserved by the `
    + "filesystem, so it counts as neither used nor free. Used, free and reserved add up to "
    + "the total."
  );
}

/**
 * The media slots this machine has, and whether anything is in them.
 *
 * `media_presence` has been computed and returned by `linux_storage.py` since
 * it landed and nothing read it, so the microSD slot on a Pironman — a slot
 * with a card in it or without one, which is a fact an owner checks — never
 * appeared anywhere in the product. A slot the machine does not have stays
 * absent; a slot it has with nothing in it says "Not fitted", which is a
 * reading, not a gap.
 */
/*
 * `MEDIA_KINDS` in `linux_storage.py` is
 * `("microsd", "nvme", "usb", "sata", "other")` and this map named four of
 * them. So an appliance whose Home badge reads "2 NVMe slots" listed microSD,
 * SATA and USB underneath and **no NVMe row at all** — leaving empty slots and
 * invisible drives looking identical, on the one medium the badge had just
 * promised. The rows are only worth having if they cover what the badge claims.
 *
 * `other` stays out: it is a catch-all, not a slot a reader can go and look at.
 */
const mediaLabels: Record<string, string> = {
  nvme: "NVMe",
  microsd: "microSD",
  usb: "USB storage",
  sata: "SATA storage",
};

export function mediaSlotRows(
  storage: StorageSummary | null,
): Array<{ kind: string; label: string; state: string }> {
  const presence = storage?.media_presence ?? {};
  // Ordered by the map rather than by object key order, so the medium the
  // hardware badge names is the first row a reader looks at.
  return Object.keys(mediaLabels)
    .filter((kind) => kind in presence)
    .map((kind) => ({
      kind,
      label: mediaLabels[kind],
      state: presence[kind].present ? `${presence[kind].count} fitted` : "Not fitted",
    }));
}

export function OverviewStoragePanel({
  storage,
  metrics,
  machineNoun,
  retentionNote = "",
}: {
  storage: StorageSummary | null;
  metrics: Metrics;
  machineNoun: string;
  /**
   * A one-line honest note about telemetry history retention, or "" when there
   * is nothing to flag (#186). It lives on the storage panel because retention
   * is what governs how much of the telemetry store is kept on this disk.
   */
  retentionNote?: string;
}) {
  const reading = storageReading(storage, metrics);
  const slots = mediaSlotRows(storage);
  const unavailableReason = `No mounted volume has been reported by this ${machineNoun} yet`;

  return (
    <section className="data-panel" aria-labelledby="storage-heading">
      <div className="panel-heading">
        <div>
          <h2 id="storage-heading">Storage utilization</h2>
          <p>
            {reading
              ? `${reading.deviceCount} ${reading.source === "devices" ? "storage device" : "monitored volume"}${reading.deviceCount === 1 ? "" : "s"} · highest use`
              : unavailableReason}
          </p>
        </div>
        <Icon name="database" />
      </div>
      <div className="storage-summary">
        <strong>
          {reading
            ? formatPercent(reading.percent)
            : <UnavailableValue label="Storage use unavailable" reason={unavailableReason} />}
        </strong>
        <span>used</span>
      </div>
      <div
        aria-label={reading ? `Storage ${formatPercent(reading.percent)}` : "Storage utilization"}
        className="progress-track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        // Deliberately indeterminate rather than zero when nothing was
        // measured: `aria-valuenow={0}` announced "Storage 0%".
        aria-valuenow={reading ? reading.percent : undefined}
        aria-valuetext={reading ? undefined : "Not reported by this device"}
      >
        {reading && <span style={{ width: `${Math.min(100, reading.percent)}%` }} />}
      </div>
      <div className="storage-meta">
        <span>
          {reading?.usedBytes != null
            ? `${formatQuantity(reading.usedBytes, "used")} used`
            : <><UnavailableValue label="Used capacity unavailable" reason={unavailableReason} /> used</>}
        </span>
        {reading?.freeBytes != null && (
          <span>{formatQuantity(reading.freeBytes, "free")} free</span>
        )}
        {/*
          * The reserve, between the two figures it sits between. Without it
          * the used and free figures do not add up to the total and a reader
          * concludes one of them is wrong — which is what "10 GB missing"
          * looked like on the tester's 1.01 TB volume.
          */}
        {Boolean(reading?.reservedBytes) && (
          <span>{formatQuantity(reading!.reservedBytes!, "capacity")} reserved</span>
        )}
        <span>
          {reading?.totalBytes != null
            ? `${formatQuantity(reading.totalBytes, "capacity")} total`
            : <><UnavailableValue label="Total capacity unavailable" reason={unavailableReason} /> total</>}
        </span>
      </div>
      {reserveNote(reading) && <p className="storage-reserve-note">{reserveNote(reading)}</p>}
      {retentionNote && <p className="storage-reserve-note">{retentionNote}</p>}
      {slots.length > 0 && (
        <dl className="storage-slots" aria-label="Removable media slots">
          {slots.map((slot) => (
            <div key={slot.kind}>
              <dt>{slot.label}</dt>
              <dd>{slot.state}</dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}
