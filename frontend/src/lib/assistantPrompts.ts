import type { MachineProfile } from "./machine";

/**
 * The questions the welcome screen offers, per machine class.
 *
 * The Assistant can now read this machine's compute engines —
 * `assistant_machine_tools.py` registers `machine.brief`, `gpu.status`,
 * `npu.status` and `inference.status` — and nothing on screen invited a single
 * one of those questions. The suggestions were a capability-filtered list
 * built around the enclosure, so on a workstation the reader was offered "Is
 * anything running hot?" and "Are updates ready to install?" while the two
 * accelerators sat there unmentioned.
 *
 * A tool that exists and is never asked for is, from the reader's side, a tool
 * that does not exist. This is the smallest change with the largest effect on
 * whether the machine-reading work is reachable at all.
 */
export function suggestedAssistantPrompts(machine: MachineProfile): string[] {
  const capabilities = machine.capabilities;
  if (machine.machine_class === "pi-appliance") {
    /*
     * On the appliance the enclosure *is* the product, so the enclosure
     * questions are the right ones — still gated on what was discovered, so a
     * Pi with no PiPower is never asked about a battery it has not got.
     */
    return [
      ...(capabilities.case_fan.available || capabilities.cpu_fan.available
        ? ["Are my fans working?"] : []),
      ...(capabilities.battery.available
        ? ["Is the battery holding charge?"] : []),
      ...(capabilities.oled.available
        ? ["Why is the front screen off?"] : []),
      "Is anything running hot?",
      "Are updates ready to install?",
    ].slice(0, 3);
  }
  /*
   * On a workstation the questions worth asking are about work, not about
   * parts: "why is this slow" is answered by reading the engines, and it is
   * the question a person actually has.
   */
  return [
    "Why is this slow?",
    "What is using my memory?",
    ...(capabilities.gpu.available ? ["Is the graphics processor busy?"] : []),
    ...(capabilities.npu.available ? ["Is the neural accelerator being used?"] : []),
    "Is anything running hot?",
  ].slice(0, 3);
}
