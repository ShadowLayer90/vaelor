import type { MachineProfile } from "./machine";

/**
 * The System page's sections, named once (VD-200): the tab strip, the page's
 * subtitle and the top bar's breadcrumb ("System / Compute") all read these,
 * so the three can never name one place two ways.
 */
export type CoolingSection = "compute" | "cooling" | "lighting" | "hardware";

export const SYSTEM_SECTION_LABELS: Record<CoolingSection, string> = {
  cooling: "Cooling",
  compute: "Compute",
  lighting: "Case lighting",
  // The Assistant names this tab too, from HARDWARE_PANEL in
  // vaelor/assistant_console_places.py; a test holds the two equal, so a
  // rename changes both or neither.
  hardware: "Hardware and services",
};

/** Compute's own line on the SystemCompute board; the other sections keep the page's. */
export const COMPUTE_SUBTITLE = "Processor, graphics, memory and the readings behind them";

/** Compute's line on a Pi appliance (the SystemComputePi board): no graphics engine, an enclosure. */
export const COMPUTE_SUBTITLE_APPLIANCE = "Processor, memory, the enclosure and the readings behind them";


/** Fired after the System page moves between sections with `pushState`, which raises no event of its own. */
export const SYSTEM_SECTION_EVENT = "vaelor:system-section";

export function coolingSectionFromHash(hash: string): CoolingSection | null {
  const section = hash.match(/^#\/system\/(compute|cooling|lighting|hardware)(?:[/?].*)?$/)?.[1];
  // `null` means "the address bar did not ask for one", which is different
  // from "it asked for cooling": the default depends on the machine class and
  // is not knowable at the moment the hash is parsed.
  return section === "compute" || section === "lighting"
    || section === "hardware" || section === "cooling"
    ? section
    : null;
}

/** The section a machine actually shows for a requested one: an unavailable request falls back to the class's first. */
export function resolvedSystemSection(requested: CoolingSection | null, machine: MachineProfile): CoolingSection {
  const isAppliance = machine.machine_class === "pi-appliance";
  const available = (candidate: CoolingSection) =>
    candidate === "hardware"
    || (candidate === "lighting" && machine.capabilities.case_lighting.available)
    || (candidate === "cooling" && isAppliance)
    // VD-200: Compute (This machine, Live readings) is every machine's now; a
    // Pi keeps Cooling as its first section.
    || candidate === "compute";
  return requested && available(requested) ? requested : isAppliance ? "cooling" : "compute";
}
