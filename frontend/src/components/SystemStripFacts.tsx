import type { Device, Metrics } from "../types";
import { formatUptime } from "../lib/format";
import { metricNumber } from "../lib/metrics";
import { UnavailableValue } from "./ui";

/**
 * This machine's four headline facts (uptime, Vaelor version, operating
 * system, your access), in the order and words of the SystemCompute board.
 * Its own component so the layout contract can render the real list at a
 * phone width under a raised text size (W8-D2: "Administrator" broke mid-word).
 */
export function SystemStripFacts({ device, metrics, noun, role }: {
  device: Device | null;
  metrics: Metrics;
  /** What the machine is called in a sentence ("appliance", "workstation"). */
  noun: string;
  role: string;
}) {
  return (
    <dl className="system-strip__facts">
      <div>
        <dt>Uptime</dt>
        {/*
          * A permanent em dash reads as a value. `boot_time` is only
          * emitted by some telemetry providers, so when it is absent
          * the fact says it is absent.
          */}
        <dd>{metricNumber(metrics, "boot_time") === null
          ? <UnavailableValue label="Uptime unavailable" reason={`Boot time is not reported by this ${noun}`} />
          : formatUptime(metrics.boot_time)}</dd>
      </div>
      <div>
        <dt>Vaelor</dt>
        <dd>{device?.version ?? "—"}</dd>
      </div>
      <div className="system-strip__os">
        <dt>Operating system</dt>
        <dd title={device?.platform?.os.name}>
          {device?.platform?.os.name ?? "Detecting"}
        </dd>
      </div>
      <div>
        <dt>Your access</dt>
        <dd>{role}</dd>
      </div>
    </dl>
  );
}
