import { useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import type { Device, Metrics, TelemetrySample } from "../types";
import { formatQuantity, formatTemperature } from "../lib/format";
import { metricNumber } from "../lib/metrics";
import type { MachineProfile } from "../lib/machine";
import type { AccelerationReading } from "../lib/acceleration";
import { cpuTemperatureProvenance } from "../lib/sensorSource";
import { parseBoardSensors, type BoardSensors } from "../lib/wmiSensors";
import { AcceleratorCard, type EngineFact } from "./AcceleratorCard";
import { AccelerationVerdict } from "./AccelerationVerdict";

/**
 * `System → Compute` — what this machine is made of.
 *
 * A workstation has three compute engines and the product represented one of
 * them. There is no Cooling workspace here because nothing is controllable:
 * HP, Dell and Lenovo business machines keep the fan curve in the embedded
 * controller and the in-tree interface exposes no writable attribute at all.
 * That absence is a capability fact, so it is stated once, here, on the
 * processor it belongs to — rather than on every poll in a live telemetry
 * slot, which is where it used to live.
 */

/** A string metric, or `null`. The scalar-string sibling of `metricNumber`:
 * every reason and provenance string the backend publishes is carried on the
 * telemetry sample, and the card must read those rather than hard-code a
 * sentence that outlives the fact it describes (VD-040). */
function metricText(metrics: Metrics, key: string): string | null {
  const value = metrics[key];
  return typeof value === "string" && value ? value : null;
}

export function processorFacts(
  device: Device | null,
  metrics: Metrics,
): EngineFact[] {
  const cores = device?.platform?.board.cpu_cores ?? metricNumber(metrics, "cpu_cores");
  // `cpu_cores` is PHYSICAL cores and `cpu_threads` is the logical count; on an
  // SMT part they differ (16 vs 32 on this Z2) and this row labelled the core
  // count "threads", so a 16-core machine read "16 threads" (#107/#113/VD-099).
  // `cpu_cores_are_threads` is set by the backend when the core count could not
  // be established and `cpu_cores` is standing in for the logical count — the
  // one case where calling the number "threads" is the honest label.
  const threads = metricNumber(metrics, "cpu_threads");
  const coresAreThreads = metrics["cpu_cores_are_threads"] === true;
  // Package energy is root-only, which is a reason to ask the privileged bridge
  // — which the backend does — not to tell the owner the machine cannot be
  // read. The watt figure and, when there is none, the *true* reason both ride
  // the telemetry sample. This row hard-coded "Vaelor cannot read them" and so
  // reported "?" on a machine whose package power the bridge was reading (#85).
  const packageWatts = metricNumber(metrics, "cpu_package_power_watts");
  const packageReason = metricText(metrics, "cpu_package_power_reason");
  const packageSource = metricText(metrics, "cpu_package_power_source");
  const frequency = metricNumber(metrics, "cpu_freq");
  const temperature = metricNumber(metrics, "cpu_temperature");
  const provenance = cpuTemperatureProvenance(metrics);
  const agreement = parseBoardSensors(metrics).agreement;
  return [
    {
      label: "Cores",
      value: cores === null || cores === undefined
        ? null
        : coresAreThreads
          ? `${cores} threads`
          : threads !== null && threads !== cores
            ? `${cores} cores · ${threads} threads`
            : `${cores} cores`,
      reason: "No processor topology has been reported for this machine",
    },
    {
      label: "Frequency",
      value: frequency !== null ? `${(frequency / 1000).toFixed(2)} GHz` : null,
      reason: "This machine does not report a current processor frequency",
    },
    {
      label: "Temperature",
      value: temperature !== null ? formatTemperature(temperature) : null,
      reason: "No processor temperature sensor was reported by this machine",
      note: temperature === null
        ? undefined
        : provenance
          ? provenance.labelled
            ? `from ${provenance.label}`
            : `from ${provenance.label} — an unlabelled fallback, so this may not be the processor`
          : "the sensor source was not reported",
    },
    {
      label: "Package power",
      value: packageWatts !== null ? `${packageWatts.toFixed(1)} W` : null,
      reason: packageReason ?? "Package power has not been reported yet",
      note: packageWatts !== null && packageSource ? `from ${packageSource}` : undefined,
    },
    /*
     * Two independent paths to one reading, and whether they still agree.
     *
     * `k10temp` reads the die; `hp-wmi` reads the board's own CPU channel.
     * They tracked each other to within 1.4 °C across a 29 °C swing, which is
     * worth stating — and a future disagreement would say something neither
     * reading says alone, which is the reason to put it on screen rather than
     * to check it once and forget it.
     */
    ...(agreement ? [{
      label: "Sensor cross-check",
      value: agreement.agrees ? "Two sources agree" : "Sources disagree",
      note: agreement.warning
        ?? (agreement.deltaC === null
          ? "The die sensor and the board's own channel are compared each sample."
          : `The die sensor and the board's own channel differ by ${agreement.deltaC.toFixed(1)} °C.`),
    }] : []),
  ];
}

export function graphicsFacts(
  machine: MachineProfile,
  metrics: Metrics,
): EngineFact[] {
  const adapter = machine.hardware.accelerators[0];
  const inventory = machine.hardware.graphics;
  const gttUsed = metricNumber(metrics, "gpu_gtt_used_bytes");
  const gttTotal = metricNumber(metrics, "gpu_gtt_total_bytes");
  const vramUsed = metricNumber(metrics, "gpu_vram_used_bytes");
  const vramTotal = metricNumber(metrics, "gpu_vram_total_bytes");
  const busy = metricNumber(metrics, "gpu_busy_percent");
  const temperature = metricNumber(metrics, "gpu_temperature_c");
  const watts = metricNumber(metrics, "gpu_power_watts");
  const clock = metricNumber(metrics, "gpu_clock_mhz");
  return [
    {
      label: "Driver",
      value: adapter?.driver ?? null,
      reason: "No kernel driver name was reported for this adapter",
    },
    {
      label: "PCI id",
      value: adapter?.pciId ?? null,
      reason: "No PCI identity was reported for this adapter",
    },
    {
      label: "ROCm",
      value: inventory?.rocmVersion ?? null,
      reason: inventory?.rocmNote || "No ROCm installation was found",
    },
    {
      label: "Graphics userspace (Mesa)",
      value: inventory?.mesaVersion ?? null,
      reason: inventory?.mesaNote || "The Mesa version was not readable",
    },
    {
      label: "Busy",
      value: [
        busy !== null ? `${Math.round(busy)}%` : null,
        temperature !== null ? formatTemperature(temperature) : null,
        watts !== null ? `${watts.toFixed(0)} W` : null,
        clock !== null ? `${(clock / 1000).toFixed(2)} GHz` : null,
      ].filter(Boolean).join(" · ") || null,
      reason: "This adapter does not report utilisation",
    },
    {
      // Both halves must be present. A total with no used figure is not a
      // ratio, and `?? 0` here would report an aperture holding 19 GB as empty
      // — the same fabrication the GPU tile on Home had.
      label: "Shared with the system",
      value: gttTotal !== null && gttUsed !== null
        ? `${formatQuantity(gttUsed, "used")} of ${formatQuantity(gttTotal, "capacity")}`
        : null,
      reason: gttTotal !== null
        ? `This adapter publishes a shared aperture of ${formatQuantity(gttTotal, "capacity")} but not how much of it is in use`
        : "This adapter reports no shared memory aperture",
    },
    {
      label: "Reserved video memory",
      value: vramTotal !== null && vramUsed !== null
        ? `${formatQuantity(vramUsed, "used")} of ${formatQuantity(vramTotal, "capacity")}`
        : null,
      reason: vramTotal !== null
        ? `This adapter publishes ${formatQuantity(vramTotal, "capacity")} of reserved video memory but not how much of it is in use`
        : "This adapter reports no reserved video memory",
      note: adapter?.unifiedMemory
        ? "This adapter takes memory from the shared pool rather than from the reserved block, so the reserved block sits nearly unused. Vaelor cannot change that; it is a firmware setting."
        : undefined,
    },
  ];
}

export function neuralFacts(
  machine: MachineProfile,
  metrics: Metrics,
): EngineFact[] {
  const npu = machine.hardware.neuralAccelerators[0];
  // Activity, power and clock all come from `amd-smi` and are omitted entirely
  // when the reading cannot be taken; the backend then publishes one reason for
  // the gap. This card used to hard-code "amd-smi, which is not installed" — a
  // sentence that stayed on screen after amd-smi was installed, and was false
  // on a Z2 where it is installed twice over (VD-040). The reason is now
  // whatever the probe actually found: not installed, installed-but-no-APU-
  // metrics, or a failure — never a guess written at the call site.
  const activity = metricNumber(metrics, "npu_activity_percent");
  const power = metricNumber(metrics, "npu_power_watts");
  const clock = metricNumber(metrics, "npu_clock_mhz");
  const reason = metricText(metrics, "npu_reading_reason");
  return [
    {
      label: "Driver",
      value: npu?.driver ?? null,
      reason: "No kernel driver name was reported for this device",
    },
    {
      label: "Device",
      value: npu?.deviceNode ?? null,
      reason: "No device node was reported",
    },
    {
      label: "Firmware version",
      value: npu?.firmwareVersion ?? null,
      reason: npu?.firmwareNote || "No firmware version was readable for this device",
    },
    {
      // Never zero on the strength of a missing reading. A genuine 0% with the
      // device idle is a real reading and is shown; an absent one is absent.
      label: "Activity",
      value: activity !== null ? `${Math.round(activity)}%` : null,
      reason: reason ?? "Neural-processor activity has not been reported yet",
      note: activity !== null && clock !== null
        ? `${(clock / 1000).toFixed(2)} GHz`
        : undefined,
    },
    {
      label: "Power",
      value: power !== null ? `${power.toFixed(2)} W` : null,
      reason: reason ?? "Neural-processor power has not been reported yet",
    },
    {
      // When Vaelor is actually serving the Assistant on this device, say so
      // and by which model. The driver's `reason` is a hardware fact ("no
      // backend targets this device") that stays true of the silicon while the
      // Assistant runs on it, so serving wins over it and drops that note — the
      // panel used to read "nothing uses the NPU" with the Assistant live on it.
      label: "Used by",
      value: npu?.servingAssistant
        ? `Serving the Assistant${npu.servingModel ? ` (${npu.servingModel})` : ""}`
        : npu?.runtimeDetected
          ? "A userspace runtime is present"
          : "Not in use",
      // No action is rendered on an engine Vaelor cannot act on; a served device
      // needs no such note, so it is dropped in favour of what is serving it.
      note: npu?.servingAssistant ? undefined : (npu?.reason ?? undefined),
    },
  ];
}

/**
 * What the board reports about its own cooling.
 *
 * This card exists because the product told a three-fan workstation it had no
 * fans. The statement it had was true and remains here word for word — HP
 * keeps the curve in the embedded controller and exposes nothing writable —
 * but it was the *only* statement, so "Vaelor cannot set this fan" was read as
 * "there is no fan". Reading and controlling are separate capabilities and are
 * now answered separately.
 */
export function coolingFacts(
  sensors: BoardSensors,
  machine: MachineProfile,
): EngineFact[] {
  const fan = machine.capabilities.cpu_fan;
  const readings = machine.capabilities.fan_readings;
  const facts: EngineFact[] = sensors.fans.map((entry) => ({
    label: entry.label,
    value: entry.fault
      ? "Fault reported"
      : entry.rpm === null
        ? null
        : entry.rpm === 0 ? "Not turning" : `${Math.round(entry.rpm)} RPM`,
    reason: "This fan reports no tachometer reading",
    note: entry.fault ? "The board flagged a fault on this fan." : undefined,
  }));
  if (!facts.length) {
    facts.push({
      label: "Fan speed",
      value: null,
      reason: readings.reason ?? "No fan tachometer is readable on this machine",
    });
  }
  facts.push({
    // Unchanged, deliberately. Both testers praised this sentence, and its
    // second clause is now demonstrably true rather than merely plausible.
    label: "Fan control",
    value: fan.available ? "Available" : "Not available",
    note: fan.available ? undefined : fan.reason ?? undefined,
  });
  return facts;
}

/** The board's labelled temperature channels, exactly as served. */
export function boardTemperatureFacts(sensors: BoardSensors): EngineFact[] {
  return sensors.temperatures.map((channel) => ({
    label: channel.label,
    value: channel.fault
      ? "Fault reported"
      : channel.celsius === null ? null : formatTemperature(channel.celsius),
    reason: "This channel reported no reading on this sample",
  }));
}

export function memoryFacts(metrics: Metrics): EngineFact[] {
  const used = metricNumber(metrics, "memory_used");
  const total = metricNumber(metrics, "memory_total");
  // Whether the memory corrects errors, established rather than assumed. The
  // machine is *asked* — the SMBIOS records are root-only, so the privileged
  // bridge reads them — and it answers "None" on this Z2, which has no
  // host-visible ECC. This row hard-coded "Not reported by this machine", which
  // reads as "there is no ECC" while claiming the machine stayed silent; it was
  // neither asked-and-silent nor without an answer (#85/#228). The value and
  // its one-sentence basis both ride the telemetry sample.
  const eccValue = metricText(metrics, "memory_ecc_value");
  const eccSummary = metricText(metrics, "memory_ecc_summary");
  return [
    {
      label: "Fitted",
      value: total !== null ? formatQuantity(total, "capacity") : null,
      reason: "Total memory has not been reported yet",
    },
    {
      label: "In use",
      value: used !== null ? formatQuantity(used, "used") : null,
      reason: "Memory use has not been reported yet",
    },
    {
      label: "Free",
      value: used !== null && total !== null
        ? formatQuantity(Math.max(0, total - used), "free")
        : null,
      reason: "Free memory cannot be derived without both a total and a usage figure",
    },
    {
      label: "ECC",
      value: eccValue,
      reason: eccSummary ?? "ECC status has not been reported yet",
      // The summary carries the on-die caveat, which is the part a bare "None"
      // would drop: SMBIOS reports no host-visible ECC, but LPDDR5X commonly
      // corrects within the die and never says so.
      note: eccValue !== null ? (eccSummary ?? undefined) : undefined,
    },
  ];
}

/** One engine as `/inference/status` reports it. Only the GPU tier can fall
 * back to the CPU without saying so, so only that tier carries a reading. */
export interface InferenceEngine {
  kind: string;
  acceleration?: AccelerationReading | null;
}

/**
 * The GPU engine's acceleration reading, if this machine has one to report.
 *
 * Written as a function so the choice is testable: an engine list with no GPU
 * tier, and a GPU tier that reported nothing, are both "no reading" and
 * neither is a fault.
 */
export function gpuAcceleration(
  engines: InferenceEngine[] | null,
): AccelerationReading | null {
  return engines?.find((engine) => engine.kind === "gpu")?.acceleration ?? null;
}

export function ComputePanel({ machine }: { machine: MachineProfile }) {
  const [device, setDevice] = useState<Device | null>(null);
  const [metrics, setMetrics] = useState<Metrics>({});
  /*
   * Whether the accelerator is actually serving, which is a different question
   * from whether the adapter is present — and the one the rest of this panel
   * could not answer. A llama.cpp that cannot resolve its accelerator library
   * loads the CPU backend and serves normally, nine times slower, with no
   * error anywhere.
   */
  const [engines, setEngines] = useState<InferenceEngine[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    void apiRequest<{ engines: InferenceEngine[] }>("/inference/status")
      .then((status) => { if (!cancelled) setEngines(status.engines ?? []); })
      // A reading that could not be taken says nothing. It is not a fault, and
      // it must not render as one, so nothing is put in its place.
      .catch(() => { if (!cancelled) setEngines(null); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const read = () => {
      void apiRequest<TelemetrySample>("/telemetry/current")
        .then((sample) => { if (!cancelled) setMetrics(sample.metrics ?? {}); })
        .catch(() => undefined);
    };
    void apiRequest<Device>("/device")
      .then((next) => { if (!cancelled) setDevice(next); })
      .catch(() => undefined);
    read();
    const timer = window.setInterval(read, 5_000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, []);

  const adapter = machine.hardware.accelerators[0];
  const npu = machine.hardware.neuralAccelerators[0];
  const sensors = parseBoardSensors(metrics);

  return (
    <div className="engine-grid">
      <AcceleratorCard
        icon="cpu"
        title="Processor"
        subtitle={device?.platform?.board.cpu_model ?? null}
        facts={processorFacts(device, metrics)}
      />
      {(adapter || machine.capabilities.gpu.available) && (
        <AcceleratorCard
          icon="gpu"
          title="Graphics"
          subtitle={adapter?.name ?? "Graphics processor"}
          facts={graphicsFacts(machine, metrics)}
        >
          {/*
            * On the engine it is a fact about. Every row above says what the
            * adapter *is*; this says whether the model server is using it, and
            * it is the one reading on this panel that a silent 9x degradation
            * would show up in.
            */}
          <AccelerationVerdict
            acceleration={gpuAcceleration(engines)}
            headingLevel="h4"
            title="Local AI on this adapter"
          />
        </AcceleratorCard>
      )}
      {(npu || machine.capabilities.npu.available) && (
        <AcceleratorCard
          icon="npu"
          title="Neural processor"
          subtitle={npu?.name ?? "Neural accelerator"}
          facts={neuralFacts(machine, metrics)}
        />
      )}
      {/*
        * Cooling gets a card of its own on a machine with no cooling *controls*
        * — because it has cooling *readings*, and the product previously had
        * nowhere to say so.
        */}
      <AcceleratorCard
        icon="fan"
        title="Cooling"
        subtitle={machine.capabilities.fan_readings.available
          ? "Reported by the board; not adjustable from here"
          : null}
        facts={coolingFacts(sensors, machine)}
      />
      <AcceleratorCard icon="memory" title="Memory" facts={memoryFacts(metrics)} />
      {sensors.temperatures.length > 0 && (
        <AcceleratorCard
          icon="activity"
          title="Board temperatures"
          subtitle="Labelled channels this board publishes"
          facts={boardTemperatureFacts(sensors)}
        />
      )}
    </div>
  );
}
