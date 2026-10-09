import type { ReactNode } from "react";
import type { Metrics, TelemetrySample } from "../types";
import { formatPercent, formatQuantity, formatTemperature } from "../lib/format";
import { metricNumber, metricSeries, metricSum, metricSumSeries } from "../lib/metrics";
import type { MachineProfile } from "../lib/machine";
import { cpuTemperatureProvenance } from "../lib/sensorSource";
import type { IconName } from "./Icon";

export interface OverviewCard {
  icon: IconName;
  label: string;
  value: string;
  /**
   * `ReactNode`, not `string`, precisely so an absent component of the detail
   * line can be an `UnavailableValue` rather than a fabricated number. A
   * string-typed slot is what pushed `?? 0` into this file in the first place.
   */
  detail: ReactNode;
  values: Array<number | null>;
  tone: "blue" | "green" | "amber" | "pink";
  unavailableReason?: string | null;
  actionLabel?: string;
  actionDescription?: string;
  onAction?: () => void;
}

/**
 * The tile grid, derived from what this machine can actually measure.
 *
 * It used to be a literal array of four, one of which — CPU temperature —
 * carried a fan glyph and a "Thermal monitoring" filler line on hosts with no
 * readable fan. The GPU tiles appear only where a GPU was discovered, so a Pi
 * and a GPU-less host keep exactly the four they have today rather than
 * gaining two cards reading `0%`.
 */
export function overviewCards({
  metrics,
  history,
  machine,
  onTuneMemory,
}: {
  metrics: Metrics;
  history: TelemetrySample[];
  machine: MachineProfile;
  onTuneMemory: () => void;
}): OverviewCard[] {
  const cpuFrequency = metricNumber(metrics, "cpu_freq");
  const fanRpm = metricNumber(metrics, "pwm_fan_speed");
  const memoryUsed = metricNumber(metrics, "memory_used");
  const memoryTotal = metricNumber(metrics, "memory_total");
  const networkKeys = ["network_download_speed", "network_upload_speed"];
  const networkTotal = metricSum(metrics, networkKeys);
  const provenance = cpuTemperatureProvenance(metrics);

  /*
   * The tile used to spend its detail line, on every poll, restating what the
   * machine has not got: "No controllable processor fan detected". Correct,
   * and dead by construction — a permanent negative statement in a live
   * telemetry slot. Absence of fan control is a capability fact and is stated
   * once, on the control that would have used it.
   *
   * What belongs here is where the number came from. A reading labelled "CPU
   * temperature" that will not name its sensor cannot be told apart from an
   * ACPI board sensor standing in for one.
   */
  const thermalDetail = [
    fanRpm !== null ? `${Math.round(fanRpm)} RPM fan` : null,
    provenance?.caveat ?? provenance?.label ?? null,
  ].filter(Boolean).join(" · ")
    || (metricNumber(metrics, "cpu_temperature") === null
      ? "No sensor reported"
      : "Sensor source not reported");

  const cards: OverviewCard[] = [
    {
      icon: "cpu",
      label: "Processor load",
      value: formatPercent(metrics.cpu_percent),
      detail: cpuFrequency !== null
        ? `${(cpuFrequency / 1000).toFixed(2)} GHz`
        : "Frequency unavailable",
      values: metricSeries(history, "cpu_percent"),
      tone: "blue",
      unavailableReason: metricNumber(metrics, "cpu_percent") === null
        ? "Processor load has not been reported yet"
        : null,
    },
    {
      // The fan glyph asserted a fan. Temperature is a processor reading and
      // is measured on every supported machine; the fan is not.
      icon: "activity",
      label: "CPU temperature",
      value: formatTemperature(metrics.cpu_temperature),
      detail: thermalDetail,
      values: metricSeries(history, "cpu_temperature"),
      tone: "pink",
      unavailableReason: metricNumber(metrics, "cpu_temperature") === null
        ? "No processor temperature sensor was reported by this machine"
        : null,
    },
  ];

  // VD-200 (the SystemCompute board): the GPU's busy, temperature, power and
  // clock are the Accelerators card's now, so Live readings is one row of four.
  cards.push(
    {
      icon: "memory",
      label: "Memory in use",
      /*
       * Gigabytes, not a percentage. On a machine where a single resident
       * model occupies 5.7–22.6 GB, "9%" hides the largest consumer of memory
       * behind a figure that cannot be compared with anything the user is
       * about to load. The headline is the amount; the percentage survives as
       * the sparkline series it always was.
       */
      value: memoryUsed !== null && memoryTotal !== null
        ? `${formatQuantity(memoryUsed, "used")} of ${formatQuantity(memoryTotal, "capacity")}`
        : formatPercent(metrics.memory_percent),
      detail: memoryUsed !== null && memoryTotal !== null
        ? `${formatQuantity(Math.max(0, memoryTotal - memoryUsed), "free")} free`
        : "Memory total not reported",
      values: metricSeries(history, "memory_percent"),
      tone: "green",
      unavailableReason: metricNumber(metrics, "memory_percent") === null
        && memoryUsed === null
        ? "Memory use has not been reported yet"
        : null,
      // "OPTIMIZE" said nothing about what it would do, and floated on the
      // chart as if it acted immediately. It names its destination now, and
      // the panel it opens still requires an explicit reviewed approval
      // before any kernel setting is changed.
      actionLabel: "Tune memory",
      actionDescription:
        "Open memory settings. Nothing changes until you review and approve a profile.",
      onAction: onTuneMemory,
    },
    {
      icon: "network",
      label: "Network throughput",
      value: formatQuantity(networkTotal, "transfer", "/s"),
      detail: `Down ${formatQuantity(metrics.network_download_speed, "transfer", "/s")}`,
      values: metricSumSeries(history, networkKeys),
      tone: "amber",
      unavailableReason: networkTotal === null
        ? "No network interface reported a throughput figure"
        : null,
    },
  );

  return cards;
}
