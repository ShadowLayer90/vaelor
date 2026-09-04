/**
 * The compute engines this machine actually has.
 *
 * `GET /api/v2/system/machine` has served `accelerators`, `neural_accelerators`
 * and `graphics` since the platform drivers landed, and the client read none of
 * them: `parseMachineProfile` looked only at `machine_class` and six capability
 * keys, so a discovered neural accelerator — the third compute engine on this
 * hardware — existed nowhere in the product. Everything here is parsed from
 * what the driver reported and nothing is inferred: a field the payload does
 * not carry stays `null`, and the reason the driver gave travels with it.
 */

export interface AcceleratorSummary {
  /** The discovered name, verbatim: "AMD Radeon 8060S (Strix Halo)". */
  name: string;
  /** The same name with the vendor word and any parenthetical removed. */
  shortName: string;
  driver: string | null;
  /** `vendor:device`, as the inventory reports it. */
  pciId: string | null;
  linkSpeed: string | null;
  vramTotalBytes: number | null;
  gttTotalBytes: number | null;
  /** True where the part has both a carve-out and a shared aperture. */
  unifiedMemory: boolean;
}

export interface NeuralAcceleratorSummary {
  name: string;
  shortName: string;
  driver: string | null;
  deviceNode: string | null;
  firmwareVersion: string | null;
  /** Why the firmware version is absent, from the driver. Never invented. */
  firmwareNote: string | null;
  runtimeDetected: boolean;
  /** Why Vaelor cannot submit work to it, from the driver. */
  reason: string | null;
  /**
   * True when Vaelor is serving the Assistant on this device. The driver's
   * hardware discovery cannot know this; it is layered on server-side from the
   * running deployment, so `reason` (a hardware fact) can still be false while
   * this is true.
   */
  servingAssistant: boolean;
  /** The served model tag, when serving. Never invented. */
  servingModel: string | null;
}

export interface GraphicsInventory {
  rocmVersion: string | null;
  rocmNote: string | null;
  mesaVersion: string | null;
  /** Mesa is genuinely unreadable headless; the note says which. */
  mesaNote: string | null;
}

export interface MachineHardware {
  accelerators: AcceleratorSummary[];
  neuralAccelerators: NeuralAcceleratorSummary[];
  graphics: GraphicsInventory | null;
}

export const noMachineHardware: MachineHardware = {
  accelerators: [],
  neuralAccelerators: [],
  graphics: null,
};

const VENDOR_WORDS = ["AMD", "NVIDIA", "Intel", "Advanced Micro Devices"];

/**
 * A chip has room for the part, not for the vendor's full marketing string.
 *
 * This only removes; it never substitutes. A short name that named a part the
 * payload did not name would be the same class of defect as the "GPU telemetry
 * live" chip it replaces — a sentence Vaelor wrote about itself rather than a
 * fact about the machine.
 */
export function shortAcceleratorName(name: string): string {
  let trimmed = name.replace(/\s*\([^)]*\)\s*$/, "").trim();
  for (const vendor of VENDOR_WORDS) {
    if (trimmed.toLowerCase().startsWith(`${vendor.toLowerCase()} `)) {
      trimmed = trimmed.slice(vendor.length + 1).trim();
      break;
    }
  }
  trimmed = trimmed.replace(/\bNeural Processing Unit\b/i, "NPU");
  return trimmed || name;
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function count(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readAccelerator(source: unknown): AcceleratorSummary | null {
  if (!source || typeof source !== "object") return null;
  const entry = source as Record<string, unknown>;
  const name = text(entry.name);
  if (!name) return null;
  const vendorId = text(entry.vendor_id);
  const deviceId = text(entry.device_id);
  return {
    name,
    shortName: shortAcceleratorName(name),
    driver: text(entry.driver),
    pciId: text(entry.pci_id)
      ?? (vendorId && deviceId
        ? `${vendorId.replace(/^0x/, "")}:${deviceId.replace(/^0x/, "")}`
        : null),
    linkSpeed: text(entry.link_speed),
    vramTotalBytes: count(entry.vram_total_bytes),
    gttTotalBytes: count(entry.gtt_total_bytes),
    unifiedMemory: entry.unified_memory === true,
  };
}

function readNeural(source: unknown): NeuralAcceleratorSummary | null {
  if (!source || typeof source !== "object") return null;
  const entry = source as Record<string, unknown>;
  const name = text(entry.name);
  if (!name) return null;
  return {
    name,
    shortName: shortAcceleratorName(name),
    driver: text(entry.driver),
    deviceNode: text(entry.device_node),
    firmwareVersion: text(entry.firmware_version),
    firmwareNote: text(entry.firmware_note),
    runtimeDetected: entry.runtime_detected === true,
    reason: text(entry.reason),
    servingAssistant: entry.serving_assistant === true,
    servingModel: text(entry.serving_model),
  };
}

function readGraphics(source: unknown): GraphicsInventory | null {
  if (!source || typeof source !== "object") return null;
  const entry = source as Record<string, unknown>;
  return {
    rocmVersion: text(entry.rocm_version),
    rocmNote: text(entry.rocm_note),
    mesaVersion: text(entry.mesa_version),
    mesaNote: text(entry.mesa_note),
  };
}

function list(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

/** Parse the accelerator half of `GET /api/v2/system/machine`. */
export function parseMachineHardware(payload: unknown): MachineHardware {
  if (!payload || typeof payload !== "object") return noMachineHardware;
  const source = payload as Record<string, unknown>;
  const graphics = readGraphics(source.graphics);
  const adapters = list(source.accelerators)
    .map(readAccelerator)
    .filter((entry): entry is AcceleratorSummary => entry !== null);
  // `graphics.adapters` carries the PCI id and ROCm-adjacent facts that the
  // raw accelerator record does not, so it wins where both describe the same
  // part. Neither is preferred blindly: whichever names the adapter is used.
  const inventoryAdapters = list(
    (source.graphics as Record<string, unknown> | undefined)?.adapters,
  )
    .map(readAccelerator)
    .filter((entry): entry is AcceleratorSummary => entry !== null);
  const neural = list(source.neural_accelerators)
    .map(readNeural)
    .filter((entry): entry is NeuralAcceleratorSummary => entry !== null);
  const inventoryNeural = list(
    (source.graphics as Record<string, unknown> | undefined)?.neural_accelerators,
  )
    .map(readNeural)
    .filter((entry): entry is NeuralAcceleratorSummary => entry !== null);
  return {
    accelerators: inventoryAdapters.length ? inventoryAdapters : adapters,
    neuralAccelerators: mergeNeural(neural, inventoryNeural),
    graphics,
  };
}

/**
 * `neural_accelerators` carries the driver's `reason`; the graphics inventory
 * carries the `firmware_note`. Both matter and neither is a superset, so a
 * device present in both is reported with the union rather than with whichever
 * list happened to be read second.
 */
function mergeNeural(
  primary: NeuralAcceleratorSummary[],
  inventory: NeuralAcceleratorSummary[],
): NeuralAcceleratorSummary[] {
  if (!primary.length) return inventory;
  return primary.map((device) => {
    const match = inventory.find((entry) => entry.name === device.name);
    if (!match) return device;
    return {
      ...device,
      deviceNode: device.deviceNode ?? match.deviceNode,
      firmwareVersion: device.firmwareVersion ?? match.firmwareVersion,
      firmwareNote: device.firmwareNote ?? match.firmwareNote,
      reason: device.reason ?? match.reason,
    };
  });
}
