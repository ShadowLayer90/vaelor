import type { Metrics } from "../types";

/**
 * Where the processor temperature came from.
 *
 * `linux_sensors.cpu_temperature()` prefers a labelled sensor — `k10temp`'s
 * `Tctl` on this AMD workstation, `cpu_thermal` on a Pi — and falls back to the
 * hottest thermal zone, flagging that case as unlabelled precisely so nothing
 * downstream mistakes a board or chipset reading for a processor one. It has
 * emitted `cpu_temperature_source` and `cpu_temperature_labelled` on every
 * sample since it landed and the client displayed neither, so a real k10temp
 * reading and an ACPI board sensor standing in for one looked identical on
 * screen. The backend did the hard part; this is the client stopping throwing
 * the answer away.
 */
export interface CpuTemperatureProvenance {
  /** `Tctl · k10temp`, or the raw source when it is not `driver/label`. */
  label: string;
  labelled: boolean;
  /** Set only for an unlabelled fallback; shown as the caveat. */
  caveat: string | null;
}

const UNLABELLED_CAVEAT =
  "Unlabelled sensor — this may not be the processor";

export function cpuTemperatureProvenance(
  metrics: Metrics,
): CpuTemperatureProvenance | null {
  const raw = metrics.cpu_temperature_source;
  if (typeof raw !== "string" || !raw.trim()) return null;
  const source = raw.trim();
  const labelled = metrics.cpu_temperature_labelled === true;
  const [driver, ...rest] = source.split("/");
  const reading = rest.join("/");
  // `k10temp/Tctl` reads best as "Tctl · k10temp": the measurement first, the
  // driver that produced it second. A source with no `/` is shown verbatim
  // rather than reshaped into a form the backend never sent.
  const label = reading ? `${reading} · ${driver}` : source;
  return {
    label,
    labelled,
    caveat: labelled ? null : UNLABELLED_CAVEAT,
  };
}
