import { useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import {
  machineProfileFromDevice,
  parseMachineProfile,
  type MachineProfile,
} from "../lib/machine";
import type { Device } from "../types";

/**
 * What this computer actually has, for any surface that gates a control on it.
 *
 * Returns `null` while discovery is outstanding. Callers must treat that as
 * "not yet known" and keep capability-gated controls disabled, because the
 * honest default for an unanswered question is not "yes" — an enabled Apply
 * button that turns out to have no hardware behind it is the defect this
 * exists to remove.
 */
export function useMachineProfile(): MachineProfile | null {
  const [machine, setMachine] = useState<MachineProfile | null>(null);

  useEffect(() => {
    let cancelled = false;
    const settle = (next: MachineProfile) => { if (!cancelled) setMachine(next); };
    void apiRequest<unknown>("/system/machine")
      .then((payload) => settle(parseMachineProfile(payload)))
      .catch(() =>
        // `/system/machine` is the authority, but until it is served
        // everywhere the enclosure product profile on `/device` carries the
        // same facts, so a Pi keeps every surface it has today.
        apiRequest<Device>("/device")
          .then((device) => settle(machineProfileFromDevice(device)))
          .catch(() => settle(machineProfileFromDevice(null))),
      );
    return () => { cancelled = true; };
  }, []);

  return machine;
}
