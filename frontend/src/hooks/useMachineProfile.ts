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
    /*
     * The discovery lives exactly as long as the surface that asked. A bare
     * `cancelled` flag used to guard only the state write: an unmounted hook
     * still issued `/device`, and the transport's own dead-socket retry still
     * re-sent `/system/machine` 200 ms after the surface had gone — wasted work
     * in the product, and in the test suite a straggler that landed during
     * whichever test ran next and was charged to that test's fetch mock.
     * Withdrawing through the signal reaches the retry too, and costs nothing
     * for the surfaces that stay: the transport still shares one wire request
     * and one cached answer among every mounted asker.
     */
    const discovery = new AbortController();
    const { signal } = discovery;
    const settle = (next: MachineProfile) => { if (!signal.aborted) setMachine(next); };
    void apiRequest<unknown>("/system/machine", { signal })
      .then((payload) => settle(parseMachineProfile(payload)))
      .catch(() => {
        if (signal.aborted) return;
        // `/system/machine` is the authority, but until it is served
        // everywhere the enclosure product profile on `/device` carries the
        // same facts, so a Pi keeps every surface it has today.
        return apiRequest<Device>("/device", { signal })
          .then((device) => settle(machineProfileFromDevice(device)))
          .catch(() => { if (!signal.aborted) settle(machineProfileFromDevice(null)); });
      });
    return () => discovery.abort();
  }, []);

  return machine;
}
