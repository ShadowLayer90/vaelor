import type { Metrics } from "../types";

/**
 * The board's own sensors, as `hp-wmi` reports them.
 *
 * Vaelor told this workstation it had no fans. That was the only sentence the
 * client could form, because it had one fan capability — controllability — and
 * the honest answer to *that* is no: HP keeps the curve in the embedded
 * controller and exposes nothing writable. Meanwhile the same chassis reports
 * three tachometers and seven labelled temperature channels, and the product
 * showed none of them.
 *
 * Reading a fan and setting its speed are different capabilities. The absence
 * of the second must never be rendered as the absence of the first.
 */

export interface FanReading {
  label: string;
  rpm: number | null;
  /** The board's own fault flag for this fan. */
  fault: boolean;
}

export interface TemperatureChannel {
  label: string;
  celsius: number | null;
  fault: boolean;
}

export interface SourceAgreement {
  agrees: boolean;
  deltaC: number | null;
  /** Present only when the backend has something to warn about. */
  warning: string | null;
}

export interface BoardSensors {
  fans: FanReading[];
  temperatures: TemperatureChannel[];
  /**
   * Channels the backend read but will not vouch for. Reported separately so
   * a corroborated channel and an unproven one are never mixed in one list.
   */
  uncorroborated: string[];
  agreement: SourceAgreement | null;
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function list(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

export function parseBoardSensors(metrics: Metrics): BoardSensors {
  const fans = list(metrics.wmi_fans)
    .map((entry) => {
      if (!entry || typeof entry !== "object") return null;
      const record = entry as Record<string, unknown>;
      const label = text(record.label);
      if (!label) return null;
      return { label, rpm: finite(record.rpm), fault: record.fault === true };
    })
    .filter((entry): entry is FanReading => entry !== null);

  /*
   * Every channel the backend vouches for, at the value it reported.
   *
   * Deliberately unfiltered. Ambient and M.2 move 1-2 °C under a load that
   * moves the processor 28 °C, and a range filter would have dropped exactly
   * those — they are measuring a different thing, slowly, not measuring badly.
   * A channel that barely moves is a fact about the board, and suppressing it
   * would be this client deciding which of the machine's own sensors count.
   */
  const temperatures = list(metrics.wmi_temperatures)
    .map((entry) => {
      if (!entry || typeof entry !== "object") return null;
      const record = entry as Record<string, unknown>;
      const label = text(record.label);
      if (!label) return null;
      return { label, celsius: finite(record.celsius), fault: record.fault === true };
    })
    .filter((entry): entry is TemperatureChannel => entry !== null);

  const uncorroborated = list(metrics.wmi_temperatures_uncorroborated)
    .map(text)
    .filter((entry): entry is string => entry !== null);

  const agrees = metrics.cpu_temperature_sources_agree;
  const agreement: SourceAgreement | null = typeof agrees === "boolean"
    ? {
      agrees,
      deltaC: finite(metrics.cpu_temperature_source_delta_c),
      warning: text(metrics.cpu_temperature_source_warning),
    }
    : null;

  return { fans, temperatures, uncorroborated, agreement };
}

/** Fans the board says are turning, for a one-line summary. */
export function spinningFans(sensors: BoardSensors): FanReading[] {
  return sensors.fans.filter((fan) => fan.rpm !== null && fan.rpm > 0);
}

/**
 * The single fan figure a strip channel can show, and which fan it belongs to.
 *
 * A machine with three fans has no one "fan speed", so the channel names the
 * fan it is quoting rather than presenting one of three as though it were the
 * whole answer.
 */
export function headlineFan(sensors: BoardSensors): FanReading | null {
  const turning = spinningFans(sensors);
  if (!turning.length) return null;
  const processor = turning.find((fan) => /cpu|processor/i.test(fan.label));
  return processor ?? turning[0];
}
