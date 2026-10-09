import type { Device, Metrics, Session, TelemetrySample } from "../types";
import type { ConnectionState } from "../lib/connectionState";
import type { HealthClaim } from "../lib/health";
import type { MachineProfile } from "../lib/machine";
import { Accelerators } from "./Accelerators";
import { ComputePanel } from "./ComputePanel";
import type { FleetSummary } from "./fleetTypes";
import { EnclosurePanel } from "./EnclosurePanel";
import { PiPowerPanel } from "./PiPowerPanel";
import type { StorageSummary } from "./Sidebar";
import { SystemLiveReadings } from "./SystemLiveReadings";
import { SystemThisMachine } from "./SystemThisMachine";

/**
 * What the console shell already reads every few seconds and hands to the
 * System page, so System › Compute shows the same sample the top bar and the
 * rail show rather than polling a second time.
 */
export interface ShellTelemetry {
  claim: HealthClaim;
  /** The shell's `/cluster` read, for what each accelerator runs (VD-200); null when not read. */
  cluster?: FleetSummary | null;
  /** The connection state the claim and its mark were both computed from. */
  connectionState: ConnectionState;
  device: Device | null;
  /** When `/health` last gave its verdict (seconds or milliseconds), so an old verdict can say its age. */
  healthSampledAt: number;
  history: TelemetrySample[];
  metrics: Metrics;
  noun: string;
  onDeviceSaved: (device: Device) => Promise<void>;
  onNotice: (message: string, refused?: boolean) => void;
  /** "controller" when this machine leads a cluster; omitted otherwise. */
  role?: string;
  /** The newest sample's own time (seconds or milliseconds); 0 before the first one. */
  sampledAt: number;
  stale: boolean;
  storage: StorageSummary | null;
}

/**
 * System › Compute (VD-200, the SystemCompute and SystemComputePi boards):
 * This machine, the Pi appliance's enclosure panels under it, and Live
 * readings; then, except on a Pi, Accelerators and the Hardware list.
 */
export function SystemCompute({
  machine,
  session,
  telemetry,
}: {
  machine: MachineProfile;
  session: Session;
  telemetry?: ShellTelemetry;
}) {
  const isAppliance = machine.machine_class === "pi-appliance";
  return (
    <div className="system-compute">
      {telemetry && (
        <SystemThisMachine
          claim={telemetry.claim}
          connectionState={telemetry.connectionState}
          device={telemetry.device}
          healthSampledAt={telemetry.healthSampledAt}
          machine={machine}
          metrics={telemetry.metrics}
          noun={telemetry.noun}
          onDeviceSaved={telemetry.onDeviceSaved}
          onNotice={telemetry.onNotice}
          role={telemetry.role}
          session={session}
          storage={telemetry.storage}
        />
      )}
      {/* The enclosure and its PiPower board exist on a Pi appliance only. */}
      {telemetry && isAppliance && (
        <div className="system-compute__pair">
          <EnclosurePanel machine={machine} />
          <PiPowerPanel device={telemetry.device} />
        </div>
      )}
      {telemetry && (
        <SystemLiveReadings
          history={telemetry.history}
          machine={machine}
          metrics={telemetry.metrics}
          session={session}
          stale={telemetry.stale}
        />
      )}
      {/*
        * On a Pi appliance the tab ends at Live readings (the SystemComputePi
        * board). Elsewhere: what runs the AI models, a card per graphics or
        * neural processor, then the Hardware list - processor, memory,
        * cooling and board temperatures, a row each.
        */}
      {!isAppliance && (
        <>
          <Accelerators
            cluster={telemetry?.cluster}
            history={telemetry?.history}
            machine={machine}
            metrics={telemetry?.metrics}
            role={session.user.role}
            sampledAt={telemetry?.sampledAt}
            stale={telemetry?.stale}
          />
          <ComputePanel machine={machine} />
        </>
      )}
    </div>
  );
}
