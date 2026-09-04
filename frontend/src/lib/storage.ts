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
