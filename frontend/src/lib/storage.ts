import type { StorageSummary, StorageVolume } from "../components/Sidebar";
import type { Metrics } from "../types";
import { formatQuantity } from "./format";
import { metricNumber } from "./metrics";

/**
 * One definition of "used", for every surface that shows storage.
 *
 * Home showed a headline **"20% used"** with a bar at 20% and, directly
 * beneath it, **"253.1 GB used / 1.01 TB total"** — which is 25%. The sidebar
 * agreed with the percentage and disagreed with the bytes. Both figures were
 * "right" and they came from different definitions:
 *
 * - `used_percent` is served by `linux_storage.py` as `usage.used / usage.total`.
 * - the byte figure was reconstructed in the client as `total_bytes - free_bytes`.
 *
 * On Linux those are not the same quantity. `total - free` includes the
 * root-reserved blocks — about 5% on a default ext4 filesystem — which are
 * neither in use nor available. On the tester's 1.01 TB volume that is ~52 GB,
 * and it is the whole discrepancy: 201 GB genuinely used, 253 GB once the
 * reservation is counted as if it were.
 *
 * The client was reconstructing a figure the backend already served
 * (`used_bytes`) and reconstructing it wrongly, because `StorageVolume` did not
 * carry the field. Both numbers now come from the same one, so the headline and
 * the detail cannot drift again: whatever "used" means, the bar, the
 * percentage and the byte figure mean it together.
 */

export interface StorageVolumeLike {
  free_bytes: number;
  total_bytes: number;
  used_percent: number;
  /**
   * What the filesystem reports as in use. Optional only for an appliance too
   * old to serve it; where it is absent the fallback is derived once, here,
   * rather than differently at each call site.
   */
  used_bytes?: number;
  /** The filesystem's reserved blocks, served by `linux_storage.py`. */
  reserved_bytes?: number;
}

export interface StorageFigures {
  usedBytes: number | null;
  totalBytes: number | null;
  /** Derived from the bytes above, never taken from a second source. */
  percent: number | null;
  /**
   * What the filesystem holds back, so `used + free + reserved == total`.
   *
   * `null` means the volume did not say. Zero means it said none, and the two
   * are not the same answer: a filesystem with no reserve and a payload that
   * never mentioned one look identical the moment they are collapsed.
   */
  reservedBytes: number | null;
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * The used/total/percent triple for one volume, internally consistent.
 *
 * Reserved blocks are excluded where the filesystem told us about them, and
 * included only when it did not — in which case the percentage is derived from
 * that same inflated figure, so the two still agree with each other.
 */
export function storageFigures(volume: StorageVolumeLike): StorageFigures {
  const total = finite(volume.total_bytes);
  const free = finite(volume.free_bytes);
  const served = finite(volume.used_bytes);
  const used = served ?? (total !== null && free !== null ? Math.max(0, total - free) : null);
  /*
   * The reserve is served, and derived only when the served figure is absent
   * *and* all three of the others are present. Where `used` had to be
   * reconstructed as `total - free` there is nothing left over by
   * construction, so no reserve is claimed: the gap has already been counted
   * as used, and printing a zero would assert that the filesystem holds none.
   */
  const reserved = finite(volume.reserved_bytes)
    ?? (served !== null && total !== null && free !== null
      ? Math.max(0, total - served - free)
      : null);
  if (used === null || total === null || total <= 0) {
    return {
      usedBytes: used,
      totalBytes: total,
      percent: finite(volume.used_percent),
      reservedBytes: reserved,
    };
  }
  return {
    usedBytes: used,
    totalBytes: total,
    percent: Math.round((used / total) * 1000) / 10,
    reservedBytes: reserved,
  };
}

/** The percentage a volume should be shown at, from its own byte figures. */
export function storagePercent(volume: StorageVolumeLike): number | null {
  return storageFigures(volume).percent;
}

export interface StorageReading {
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
 * The storage panel was built on `disk_*_percent` telemetry keys, which
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
