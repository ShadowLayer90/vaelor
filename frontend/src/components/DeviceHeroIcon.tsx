import { hasNeuralAccelerator, isPortableChassis, type MachineProfile } from "../lib/machine";
import type { Device } from "../types";
import { PironmanDeviceIcon } from "./PironmanDeviceIcon";
import { WorkstationDeviceIcon } from "./WorkstationDeviceIcon";
import { LaptopDeviceIcon } from "./LaptopDeviceIcon";

interface Props {
  device: Device | null;
  machine: MachineProfile;
  isAppliance: boolean;
}

/**
 * The hero illustration of the machine you are on.
 *
 * An illustration is a claim about a machine, so it has to be a claim that
 * holds. Off-Pi the artwork used to fall through to `Generic` — a plain box
 * with three coloured dots — and the objection was never to artwork, it was
 * that a generic drawing standing in for a specific machine is clip art of
 * something this is not. There is now a drawing of *this* class — a Pironman
 * enclosure for the appliance, a laptop for a portable chassis, otherwise the
 * compact workstation — so the objection is answered where it arose.
 *
 * The neural mark follows discovery, never the machine class: a workstation is
 * not guaranteed an accelerator, and a drawing that showed one anyway would
 * promise an engine the rest of the product then correctly refuses to report.
 */
export function DeviceHeroIcon({ device, machine, isAppliance }: Props) {
  if (isAppliance) {
    return (
      <PironmanDeviceIcon
        label={`${device?.name ?? "Local appliance"} technical enclosure illustration`}
        model={device?.id ?? "generic"}
        size={260}
      />
    );
  }
  const label = `${device?.name ?? "This machine"} technical hardware illustration`;
  const neural = hasNeuralAccelerator(machine);
  if (isPortableChassis(device?.platform?.board.chassis)) {
    return <LaptopDeviceIcon label={label} neural={neural} size={260} />;
  }
  return <WorkstationDeviceIcon label={label} neural={neural} size={260} />;
}
