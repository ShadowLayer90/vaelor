import type { MachineCapabilityKey, MachineClass } from "../lib/machine";

/**
 * The `GET /api/v2/system/machine` payload, as the wire delivers it.
 *
 * Tests build the machine they are testing against rather than inheriting a
 * default, because "what does this computer have" is the question every
 * capability gate now asks and a fixture that quietly answers "everything"
 * would hide exactly the defects these gates exist to prevent.
 */
export interface MachinePayload {
  machine_class: MachineClass;
  capabilities: Record<MachineCapabilityKey, { available: boolean; reason: string | null }>;
  accelerators?: unknown[];
  neural_accelerators?: unknown[];
  graphics?: unknown;
}

const reasons: Record<MachineCapabilityKey, string> = {
  case_fan: "No enclosure fan controller detected",
  case_lighting: "No addressable lighting controller detected",
  oled: "No front display detected",
  cpu_fan: "No controllable processor fan detected",
  fan_readings: "No fan tachometer detected",
  battery: "No battery or UPS detected",
  gpu: "No graphics processor telemetry detected",
  npu: "No neural processing unit detected",
};

export function machinePayload(
  machineClass: MachineClass,
  availability: Partial<Record<MachineCapabilityKey, boolean>> = {},
): MachinePayload {
  const capabilities = {} as MachinePayload["capabilities"];
  for (const key of Object.keys(reasons) as MachineCapabilityKey[]) {
    const isAvailable = availability[key] ?? false;
    capabilities[key] = isAvailable
      ? { available: true, reason: null }
      : { available: false, reason: reasons[key] };
  }
  return { machine_class: machineClass, capabilities };
}

/** A Raspberry Pi 5 in a Pironman 5 Max: everything the product shipped for. */
export const pironmanMachine = machinePayload("pi-appliance", {
  case_fan: true,
  case_lighting: true,
  oled: true,
  cpu_fan: true,
  fan_readings: true,
  battery: true,
});

/** An HP Z2 Mini G1a: a GPU, and no enclosure of any kind. */
export const workstationMachine = machinePayload("workstation", { gpu: true });

/**
 * What `discover_accelerators` and `graphics_inventory` return on the Z2,
 * verbatim from `vaelor/platforms/accelerators.py` — the `KNOWN_DEVICES` names,
 * the `unified_memory` flag, and the `firmware_note`/`reason` strings the
 * driver writes when a value is genuinely unreadable.
 */
export const z2Accelerators = [{
  id: "card1",
  name: "AMD Radeon 8060S (Strix Halo)",
  vendor: "AMD",
  vendor_id: "0x1002",
  device_id: "0x1586",
  driver: "amdgpu",
  kind: "gpu",
  vram_total_bytes: 17_179_869_184,
  gtt_total_bytes: 24_493_297_664,
  link_speed: "16.0 GT/s PCIe",
  unified_memory: true,
}];

export const z2NeuralAccelerators = [{
  id: "accel0",
  name: "AMD Strix Halo Neural Processing Unit",
  vendor: "AMD",
  vendor_id: "0x1022",
  device_id: "0x17f0",
  driver: "amdxdna",
  kind: "npu",
  device_node: "/dev/accel/accel0",
  firmware_version: null,
  runtime_detected: false,
  usable_for_inference: false,
  reason:
    "No userspace runtime for this device was found, and no Vaelor inference "
    + "backend targets it.",
}];

export const z2Graphics = {
  adapters: [{
    name: "AMD Radeon 8060S (Strix Halo)",
    vendor: "AMD",
    driver: "amdgpu",
    pci_id: "1002:1586",
    vram_total_bytes: 17_179_869_184,
    gtt_total_bytes: 24_493_297_664,
    link_speed: "16.0 GT/s PCIe",
    unified_memory: true,
    firmware: null,
  }],
  neural_accelerators: [{
    name: "AMD Strix Halo Neural Processing Unit",
    driver: "amdxdna",
    device_node: "/dev/accel/accel0",
    runtime_detected: false,
    firmware_version: null,
    firmware_note:
      "No firmware version was readable at accel0/device/fw_version.",
  }],
  rocm_version: "6.4.1",
  rocm_source: "/opt/rocm/.info/version",
  rocm_note: "",
  mesa_version: null,
  mesa_source: null,
  mesa_note: "Not readable without a display connection.",
  adapter_firmware: null,
};
