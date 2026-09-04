import type { Device } from "../types";
import {
  noMachineHardware,
  parseMachineHardware,
  type MachineHardware,
} from "./accelerators";

/**
 * What kind of computer Vaelor is running on.
 *
 * Vaelor was written for one machine: a Raspberry Pi 5 inside a SunFounder
 * Pironman enclosure, with a case fan, RGB strips, an OLED and a PiPower UPS.
 * Every one of those is an *enclosure* fact, not a computer fact, and on a
 * workstation none of them exist. The UI used to assert them anyway — five
 * enabled airflow profiles over "0 enclosure fans", a live RGB console with a
 * working Save on a machine with no LEDs, and an Apply button that reported
 * success for work it had not done.
 *
 * The class is discovered once, server-side, and every surface that used to
 * hardcode a Pi fact reads it from here instead. Both platforms stay
 * supported: this is conditional behaviour, not a Pi removal.
 */
export type MachineClass = "pi-appliance" | "workstation" | "generic";

export type MachineCapabilityKey =
  | "case_fan"
  | "case_lighting"
  | "oled"
  | "cpu_fan"
  /**
   * Whether a fan tachometer can be **read**, which is a different question
   * from whether a fan can be **set**. Keeping them apart is the whole point:
   * this workstation exposes three readable fans through `hp-wmi` and no
   * writable one, and while the client had a single `cpu_fan` key, "Vaelor
   * cannot control the fans" was the only sentence available and it rendered
   * as "this machine has no fans".
   */
  | "fan_readings"
  | "battery"
  | "gpu"
  /**
   * The third compute engine. `workstation.py` has served this key since the
   * driver landed and this list did not name it, so `parseMachineProfile`
   * dropped it on every payload and the neural accelerator was invisible to
   * the whole client.
   */
  | "npu";

/**
 * `reason` is user-facing prose saying *why* this is unavailable on this
 * machine, and is `null` exactly when the capability is available. It is shown
 * on the disabled control itself — the pattern `/power/capabilities` already
 * uses for restart/reboot/shutdown, which is the one correct unavailable state
 * the app already shipped.
 */
export interface MachineCapability {
  available: boolean;
  reason: string | null;
}

export interface MachineProfile {
  machine_class: MachineClass;
  capabilities: Record<MachineCapabilityKey, MachineCapability>;
  /**
   * The server's own health band, when it states one. Discovery knows which
   * silicon this is; duplicating a second table here would be a second place
   * for the numbers to drift, so the served policy wins where it exists and
   * the class default only fills a gap.
   */
  thermal?: { cpuWarn: number; cpuCritical: number };
  /**
   * The parts discovery actually found. Served alongside the capability map
   * and read by nothing until now, which is why Home could only say "GPU
   * telemetry live" — a fact about Vaelor — where it could have named the
   * adapter discovery had already identified.
   */
  hardware: MachineHardware;
}

export const machineCapabilityKeys: MachineCapabilityKey[] = [
  "case_fan",
  "case_lighting",
  "oled",
  "cpu_fan",
  "fan_readings",
  "battery",
  "gpu",
  "npu",
];

function unavailable(reason: string): MachineCapability {
  return { available: false, reason };
}

const available: MachineCapability = { available: true, reason: null };

/**
 * The conservative profile. Nothing is claimed that has not been discovered,
 * because a claim the machine cannot honour is worse than a missing feature.
 */
export const unknownMachine: MachineProfile = {
  machine_class: "generic",
  capabilities: {
    case_fan: unavailable("No enclosure fan controller has been reported for this machine"),
    case_lighting: unavailable("No addressable lighting controller has been reported for this machine"),
    oled: unavailable("No front display has been reported for this machine"),
    cpu_fan: unavailable("No controllable processor fan has been reported for this machine"),
    fan_readings: unavailable("No fan tachometer has been reported for this machine"),
    battery: unavailable("No battery or UPS has been reported for this machine"),
    gpu: unavailable("No graphics processor telemetry has been reported for this machine"),
    npu: unavailable("No neural processing unit has been reported for this machine"),
  },
  hardware: noMachineHardware,
};

function isMachineClass(value: unknown): value is MachineClass {
  return value === "pi-appliance" || value === "workstation" || value === "generic";
}

function readCapability(source: unknown): MachineCapability | null {
  if (!source || typeof source !== "object") return null;
  const entry = source as { available?: unknown; reason?: unknown };
  if (typeof entry.available !== "boolean") return null;
  const reason = typeof entry.reason === "string" && entry.reason.trim() ? entry.reason.trim() : null;
  // An available capability has no reason; an unavailable one must carry one,
  // because a control that goes dead without saying why reads as broken.
  return entry.available ? { available: true, reason: null } : { available: false, reason };
}

/**
 * Parse `GET /api/v2/system/machine`. Anything the payload does not state is
 * taken from the conservative profile rather than guessed at.
 */
export function parseMachineProfile(payload: unknown): MachineProfile {
  if (!payload || typeof payload !== "object") return unknownMachine;
  const source = payload as { machine_class?: unknown; capabilities?: unknown };
  const machineClass = isMachineClass(source.machine_class) ? source.machine_class : "generic";
  const capabilitySource = (source.capabilities ?? {}) as Record<string, unknown>;
  const policy = (source as { thermal_policy?: unknown }).thermal_policy;
  const thermal = policy && typeof policy === "object"
    ? (() => {
      const entry = policy as { warning_c?: unknown; critical_c?: unknown };
      return typeof entry.warning_c === "number" && typeof entry.critical_c === "number"
        ? { cpuWarn: entry.warning_c, cpuCritical: entry.critical_c }
        : undefined;
    })()
    : undefined;
  const capabilities = { ...unknownMachine.capabilities };
  for (const key of machineCapabilityKeys) {
    const parsed = readCapability(capabilitySource[key]);
    if (parsed) {
      capabilities[key] = parsed.available
        ? available
        : { available: false, reason: parsed.reason ?? unknownMachine.capabilities[key].reason };
    }
  }
  return {
    machine_class: machineClass,
    capabilities,
    thermal,
    hardware: parseMachineHardware(payload),
  };
}

/**
 * The temperatures at which an operator should actually act on this machine,
 * preferring what the server discovered over the class default.
 */
export function thermalLimits(machine: MachineProfile): { cpuWarn: number; cpuCritical: number } {
  const fallback = thermalPolicy(machine.machine_class);
  return machine.thermal ?? { cpuWarn: fallback.cpuWarn, cpuCritical: fallback.cpuCritical };
}

/**
 * Derive a profile from `/device` when `/system/machine` is not served yet.
 *
 * The enclosure product profile already carries every fact this needs — a
 * Pironman reports its fan count and an `oled`/`rgb` capability list, and the
 * generic product reports `fan_count: 0` and an empty list — so a Pi keeps
 * exactly the surfaces it has today and a workstation stops claiming hardware
 * it has never had.
 */
export function machineProfileFromDevice(device: Device | null): MachineProfile {
  const platform = device?.platform;
  if (!platform) return unknownMachine;
  const isPi = platform.board.is_raspberry_pi;
  const product = platform.product;
  const enclosure = isPi && product.id !== "generic";
  const machineClass: MachineClass = isPi
    ? "pi-appliance"
    : /^(x86_64|amd64|i[3-6]86)$/i.test(platform.board.architecture)
      ? "workstation"
      : "generic";
  const machineNounText = machineNoun(machineClass);
  return {
    machine_class: machineClass,
    capabilities: {
      case_fan: enclosure && product.fan_count > 0
        ? available
        : unavailable(`No enclosure fan controller was detected on this ${machineNounText}`),
      case_lighting: enclosure && product.capabilities.includes("rgb")
        ? available
        : unavailable(`No addressable case lighting was detected on this ${machineNounText}`),
      oled: enclosure && product.capabilities.includes("oled")
        ? available
        : unavailable(`No front display was detected on this ${machineNounText}`),
      cpu_fan: isPi
        ? available
        : unavailable(`No controllable processor fan was detected on this ${machineNounText}`),
      // `/device` carries no tachometer probe, so this stays conservative off-Pi
      // rather than inferring a reading from the absence of a control.
      fan_readings: isPi
        ? available
        : unavailable(`No fan tachometer was detected on this ${machineNounText}`),
      battery: platform.power.battery.available
        ? available
        : unavailable(`No battery or UPS was detected on this ${machineNounText}`),
      gpu: unavailable(`No graphics processor telemetry was reported for this ${machineNounText}`),
      npu: unavailable(`No neural processing unit was reported for this ${machineNounText}`),
    },
    hardware: noMachineHardware,
  };
}

/**
 * Whether a neural accelerator was actually discovered on this machine.
 *
 * Machine *class* is the wrong signal for this and must never be used: a
 * workstation is not guaranteed an NPU, and drawing one because the chassis is
 * x86 would be the illustration promising an engine the machine has not got.
 * Both served signals mean the same thing — `workstation.py` derives the `npu`
 * capability from `neural_accelerators()` — so either one answering is
 * discovery having answered.
 */
export function hasNeuralAccelerator(machine: MachineProfile): boolean {
  return machine.hardware.neuralAccelerators.length > 0
    || machine.capabilities.npu.available;
}

/**
 * The chassis form factors that draw the laptop illustration — a laptop, not a
 * desktop. Read from the DMI chassis string the backend already serves, so the
 * hero illustration can draw the right form factor: a mobile workstation has an
 * internal display and a battery a compact desktop does not.
 *
 * Every value here is one the backend actually emits (see `CHASSIS_TYPES` in
 * `vaelor/platforms/base.py`) *and* a machine with a keyboard deck, which is
 * what the laptop drawing depicts. A pure "tablet" (SMBIOS 30) has no keyboard
 * or trackpad, so it is deliberately absent — drawing it as a laptop would
 * assert hardware it does not have, and the drawing keeps that promise.
 */
const portableChassisNames = new Set([
  "portable",
  "laptop",
  "notebook",
  "convertible",
  "detachable",
]);

/**
 * Whether this machine's chassis is a portable/notebook form factor. Undefined
 * or unrecognised chassis strings are not portable, so the desktop drawing
 * stays the conservative default.
 */
export function isPortableChassis(chassis: string | null | undefined): boolean {
  const normalized = chassis?.trim().toLowerCase();
  return normalized ? portableChassisNames.has(normalized) : false;
}

/**
 * The noun for the computer itself. "Appliance" is right for a Pi in a case
 * and reads as a downgrade for a workstation; "machine" carries both, so it is
 * the fallback whenever the class is not the appliance.
 */
export function machineNoun(machineClass: MachineClass): string {
  return machineClass === "pi-appliance" ? "appliance" : "machine";
}

export interface ThermalPolicy {
  /** °C at which the sidebar and health start asking for attention. */
  cpuWarn: number;
  cpuCritical: number;
  gpuWarn: number;
  gpuCritical: number;
  /** Bounds a hand-authored cooling curve may use. */
  curveMinimum: number;
  curveMaximum: number;
}

/**
 * A Pi 5 throttles near 80 °C, so 70/80 is a real warning band for it. A Ryzen
 * AI Max+ 395 boosts to about 95 °C *by design*, so the same numbers put a
 * workstation permanently in "Attention" from the first minute of load — alarm
 * fatigue caused entirely by a constant borrowed from another machine.
 */
export function thermalPolicy(machineClass: MachineClass): ThermalPolicy {
  if (machineClass === "pi-appliance") {
    return { cpuWarn: 70, cpuCritical: 80, gpuWarn: 70, gpuCritical: 80, curveMinimum: 35, curveMaximum: 79 };
  }
  if (machineClass === "workstation") {
    return { cpuWarn: 92, cpuCritical: 100, gpuWarn: 95, gpuCritical: 105, curveMinimum: 40, curveMaximum: 99 };
  }
  return { cpuWarn: 85, cpuCritical: 95, gpuWarn: 90, gpuCritical: 100, curveMinimum: 35, curveMaximum: 94 };
}
