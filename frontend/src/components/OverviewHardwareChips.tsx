import type { Device } from "../types";
import type { MachineProfile } from "../lib/machine";
import { Icon, type IconName } from "./Icon";
import type { StorageSummary } from "./Sidebar";

/**
 * The parts this machine has, named.
 *
 * Two of these chips used to describe the plumbing rather than the machine:
 * "GPU telemetry live" and "Storage discovered live" both name *the fact that
 * Vaelor has data*, which is a sentence about Vaelor. Meanwhile the adapter
 * name discovery had already resolved — `KNOWN_DEVICES` in
 * `platforms/accelerators.py` knows it exactly — appeared nowhere on Home, and
 * the neural accelerator appeared nowhere in the product at all.
 *
 * Every chip here is a discovered fact or it is absent. Nothing is a status
 * report on the control plane's own wiring.
 */
export interface HardwareChip {
  icon: IconName;
  label: string;
  title?: string;
}

function nvmeDriveCount(storage: StorageSummary | null): number {
  if (!storage?.volumes?.length) return 0;
  const devices = new Set<string>();
  for (const volume of storage.volumes) {
    if (volume.kind === "nvme") devices.add(volume.device_id || volume.id);
  }
  return devices.size;
}

export function hardwareChips({
  device,
  machine,
  storage,
}: {
  device: Device | null;
  machine: MachineProfile;
  storage: StorageSummary | null;
}): HardwareChip[] {
  const chips: HardwareChip[] = [];
  const isAppliance = machine.machine_class === "pi-appliance";

  // A GPIO header glyph on a workstation names pins it has not got.
  chips.push({
    icon: isAppliance ? "gpio" : "cpu",
    label: device?.platform?.board.model || "Computer",
  });

  const adapter = machine.hardware.accelerators[0];
  if (adapter) {
    chips.push({ icon: "gpu", label: adapter.shortName, title: adapter.name });
  } else if (machine.capabilities.gpu.available) {
    // Telemetry arrived but the inventory did not name the part. Saying
    // "Graphics processor" is weaker than a model number and stronger than a
    // claim about what Vaelor can see.
    chips.push({ icon: "gpu", label: "Graphics processor" });
  }

  const neural = machine.hardware.neuralAccelerators[0];
  if (neural) {
    chips.push({
      icon: "npu",
      label: neural.shortName,
      // The chip states the part. Why Vaelor cannot submit work to it is the
      // driver's own sentence and travels with it rather than being invented.
      title: neural.reason ?? neural.name,
    });
  } else if (machine.capabilities.npu.available) {
    chips.push({ icon: "npu", label: "Neural accelerator" });
  }

  const product = device?.platform?.product;
  if (product && product.id !== "generic") {
    chips.push({
      icon: "nvme",
      label: `${product.nvme_slots} NVMe slot${product.nvme_slots === 1 ? "" : "s"}`,
    });
    chips.push({
      icon: "oled",
      label: `${product.capabilities.includes("oled") ? "OLED" : "No OLED"} · ${product.capabilities.includes("rgb") ? "RGB" : "No RGB"}`,
    });
  } else {
    const drives = nvmeDriveCount(storage);
    if (drives > 0) {
      chips.push({
        icon: "nvme",
        label: `${drives} NVMe drive${drives === 1 ? "" : "s"}`,
      });
    }
  }

  return chips;
}

export function OverviewHardwareChips({
  device,
  machine,
  storage,
}: {
  device: Device | null;
  machine: MachineProfile;
  storage: StorageSummary | null;
}) {
  const os = device?.platform?.os;
  return (
    <div className="device-capabilities" aria-label="Installed platform capabilities">
      {os && (
        <span
          className={`os-support os-support--${os.support_level}`}
          title={os.support_note}
        >
          {["verified", "compatible"].includes(os.support_level)
            ? <b aria-hidden="true">✓</b>
            : <i aria-hidden="true" />}
          OS {os.support_label}
        </span>
      )}
      {hardwareChips({ device, machine, storage }).map((chip) => (
        <span key={chip.label} title={chip.title}>
          <Icon name={chip.icon} size={13} />{chip.label}
        </span>
      ))}
      {/*
        * The battery and the input voltage used to be chips here, at the same
        * 13-pixel weight as the NVMe slot count. On an appliance whose whole
        * purpose is surviving a power cut, "how long will it hold" is a
        * first-order question, so those readings moved to `PiPowerPanel` where
        * they get a panel of their own and a runtime line. Repeating them here
        * would put the same fact on the page twice at two different weights.
        */}
    </div>
  );
}
