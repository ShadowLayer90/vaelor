import { Icon, ICON_SIZE } from "./Icon";
import { DeviceHeroIcon } from "./DeviceHeroIcon";
import { unknownMachine } from "../lib/machine";
import type { Device } from "../types";
import type { FleetNode, NodeCapacity } from "./fleetTypes";

/**
 * Which machine a Fleet node is, and its drawing - the helpers the machine
 * cards use.
 *
 * This module used to also hold a `ClusterCapacity` placement view that no
 * screen rendered any more, and whose per-machine "free" figure was the
 * placement ledger's Swarm reservation, not what the machine measured. The
 * Fleet tab's machine cards and headline replaced it (they read the measured
 * memory), so the view was removed rather than left as a second, reservation-
 * based answer to "how much is free" (review nit, ACC-120).
 */

/**
 * The machine class a node's *discovered* architecture implies, and only that.
 *
 * A Vaelor cluster runs one architecture (VD-031), and the two it supports map
 * to two form factors: ARM is a Raspberry Pi appliance, x86-64 is a compact
 * workstation (the Z2 controller and the ZBook worker are both x86-64). The
 * value is read from what the node actually probed (`uname -m`, or the
 * controller's own `platform.machine()`), never from a class a joining node
 * could assert. An architecture this appliance does not recognise returns
 * `null`, so the caller draws the generic mark rather than guess a machine the
 * node is not — "we could not tell" is its own answer.
 *
 * The laptop form factor turns on a portable *chassis*, which the capacity
 * ledger and the fleet inventory do not carry for a remote node, so it is not
 * drawn here: a workstation drawing for an x86 node claims only the silicon the
 * node reported, and never an internal display or battery it did not.
 */
export function machineClassForArchitecture(
  architecture: string | undefined | null,
): "pi-appliance" | "workstation" | null {
  const normalized = (architecture ?? "").trim().toLowerCase();
  if (!normalized) return null;
  if (/^(?:aarch64|arm64|armv8[0-9a-z]*|arm)$/.test(normalized)) return "pi-appliance";
  if (/^(?:amd64|x86[_-]?64|i[3-6]86)$/.test(normalized)) return "workstation";
  return null;
}

/**
 * The architecture a capacity node probed, resolved from the fleet's own
 * records. The controller reports its class through the fleet summary; a worker
 * carries it on its enrolment inventory. Matched on any identifier the two
 * sides share so a node named on one and keyed on the other still lines up.
 */
export function architectureForNode(
  node: NodeCapacity,
  fleetNodes: FleetNode[],
  controllerArchitecture: string | undefined,
): string | undefined {
  if (node.role === "head-controller" || node.node_id === "controller") {
    return controllerArchitecture;
  }
  const match = fleetNodes.find(
    (fleet) =>
      fleet.id === node.node_id
      || fleet.labels?.swarm_node_id === node.node_id
      || fleet.name === node.name,
  );
  return match?.inventory.architecture;
}

/** The per-node machine drawing, or the generic mark when its class is unknown. */
export function NodeMachineMark({ node, architecture }: { node: NodeCapacity; architecture: string | undefined }) {
  const machineClass = machineClassForArchitecture(architecture);
  if (!machineClass) {
    return (
      <span className="cluster-capacity__mark">
        <Icon name="cpu" size={ICON_SIZE.standalone} />
      </span>
    );
  }
  const isAppliance = machineClass === "pi-appliance";
  const device: Device = {
    name: node.name,
    id: isAppliance ? "generic" : node.node_id,
    version: "",
    peripherals: [],
  };
  return (
    <span className="cluster-capacity__mark cluster-capacity__mark--machine" data-machine={machineClass}>
      <DeviceHeroIcon device={device} machine={unknownMachine} isAppliance={isAppliance} />
    </span>
  );
}
