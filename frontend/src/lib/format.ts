export function formatPercent(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${Math.round(value)}%`
    : "—";
}

export function formatTemperature(value: unknown, unit = "C"): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const converted = unit === "F" ? value * (9 / 5) + 32 : value;
  return `${Math.round(converted)}°${unit}`;
}

export type ByteMode = "si" | "iec";
export type ByteQuantity =
  | "capacity"
  | "used"
  | "free"
  | "transfer"
  | "model"
  | "checkpoint";

export interface ByteFormatOptions {
  mode?: ByteMode;
  suffix?: string;
}

const BYTE_DIVISORS: Record<ByteMode, number> = {
  si: 1_000,
  iec: 1_024,
};

const BYTE_UNITS: Record<ByteMode, readonly string[]> = {
  si: ["B", "KB", "MB", "GB", "TB"],
  iec: ["B", "KiB", "MiB", "GiB", "TiB"],
};

const QUANTITY_MODES: Record<ByteQuantity, ByteMode> = {
  capacity: "si",
  used: "si",
  free: "si",
  transfer: "si",
  model: "iec",
  checkpoint: "iec",
};

export function formatBytes(
  value: unknown,
  options: string | ByteFormatOptions = "",
): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return "—";
  const normalized = typeof options === "string"
    ? { mode: "si" as const, suffix: options }
    : options;
  const mode = normalized.mode ?? "si";
  const suffix = normalized.suffix ?? "";
  const divisor = BYTE_DIVISORS[mode];
  const units = BYTE_UNITS[mode];
  let amount = value;
  let index = 0;
  while (amount >= divisor && index < units.length - 1) {
    amount /= divisor;
    index += 1;
  }
  const precision = index === 0 ? 0 : amount >= 10 ? 1 : 2;
  return `${amount.toFixed(precision)} ${units[index]}${suffix}`;
}

export function formatQuantity(
  value: unknown,
  quantity: ByteQuantity,
  suffix = "",
): string {
  return formatBytes(value, { mode: QUANTITY_MODES[quantity], suffix });
}

export function formatUptime(bootTime: unknown, now = Date.now()): string {
  if (typeof bootTime !== "number" || !Number.isFinite(bootTime)) return "—";
  const bootMilliseconds = bootTime > 10_000_000_000 ? bootTime : bootTime * 1000;
  const totalMinutes = Math.max(0, Math.floor((now - bootMilliseconds) / 60_000));
  const days = Math.floor(totalMinutes / 1440);
  const hours = Math.floor((totalMinutes % 1440) / 60);
  const minutes = totalMinutes % 60;
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

/**
 * Relative age with unambiguous units. Uppercased single letters read badly:
 * "4M AGO" is equally 4 minutes or 4 months.
 */
export function timeAgo(timestamp: number, now = Date.now()): string {
  const elapsedSeconds = Math.max(0, Math.floor((now - timestamp) / 1000));
  if (elapsedSeconds < 10) return "just now";
  if (elapsedSeconds < 60) return `${elapsedSeconds} sec ago`;
  const minutes = Math.floor(elapsedSeconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} ${hours === 1 ? "hour" : "hours"} ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days} ${days === 1 ? "day" : "days"} ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return `${months} ${months === 1 ? "month" : "months"} ago`;
  const years = Math.floor(months / 12);
  return `${years} ${years === 1 ? "year" : "years"} ago`;
}

/** Full local date and time, for the tooltip behind a relative age. */
export function exactTime(timestamp: number): string {
  return new Date(timestamp).toLocaleString();
}

export function timeUntil(timestamp: number, now = Date.now()): string {
  const remainingSeconds = Math.max(0, Math.ceil((timestamp - now) / 1000));
  if (remainingSeconds < 10) return "now";
  if (remainingSeconds < 60) return `in ${remainingSeconds}s`;
  const minutes = Math.ceil(remainingSeconds / 60);
  if (minutes < 60) return `in ${minutes}m`;
  const hours = Math.ceil(minutes / 60);
  if (hours < 24) return `in ${hours}h`;
  return `in ${Math.ceil(hours / 24)}d`;
}
