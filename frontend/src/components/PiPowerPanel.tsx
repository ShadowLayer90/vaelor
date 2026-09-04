import type { Device } from "../types";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { UnavailableValue } from "./ui";

/**
 * PiPower, at the weight the fact deserves.
 *
 * The UPS appeared on Home as a 13-pixel chip reading `Battery 96%`, sitting
 * beside "2 NVMe slots" at identical weight. On an appliance whose whole point
 * is that it keeps running when the mains does not, "how long will it hold" is
 * a first-order question and there was no runtime estimate, no charge
 * direction, and no discharge history anywhere in the product.
 *
 * The runtime follows the provenance rule for readings: it is shown only where
 * a discharge has genuinely been observed. A UPS panel that computes a
 * plausible runtime from a capacity it has never watched drain is exactly the
 * class of invention this work exists to remove — so where nothing has been
 * observed the panel says so, in those words.
 */
export function batteryRuntimeText(
  battery: NonNullable<Device["platform"]>["power"]["battery"],
): { minutes: number } | { reason: string } {
  const minutes = battery.runtime_minutes;
  if (
    battery.discharge_observed === true
    && typeof minutes === "number"
    && Number.isFinite(minutes)
    && minutes > 0
  ) {
    return { minutes };
  }
  return {
    reason: "No discharge has been observed on this appliance yet, so the runtime is unknown",
  };
}

export function formatRuntime(minutes: number): string {
  const whole = Math.round(minutes);
  if (whole < 60) return `About ${whole} min`;
  const hours = Math.floor(whole / 60);
  const rest = whole % 60;
  return rest ? `About ${hours} h ${rest} min` : `About ${hours} h`;
}

export function PiPowerPanel({ device }: { device: Device | null }) {
  const power = device?.platform?.power;
  const battery = power?.battery;
  const mains = power?.input_voltage != null || power?.output_watts != null;
  const runtime = battery ? batteryRuntimeText(battery) : { reason: "No battery has been reported" };
  const charging = battery?.charging;

  return (
    <section className="data-panel enclosure-panel" aria-labelledby="pi-power-heading">
      <div className="panel-heading">
        <div>
          <h2 id="pi-power-heading">Power</h2>
          <p>Mains, battery, and how long it would hold</p>
        </div>
        <StatusPill
          status={power?.undervoltage_now ? "degraded" : mains ? "healthy" : "neutral"}
          label={power?.undervoltage_now ? "Undervoltage" : mains ? "Mains connected" : "Checking"}
        />
      </div>
      <dl className="enclosure-facts">
        <div>
          <dt><Icon name="bolt" size={15} />Input</dt>
          <dd>
            {power?.input_voltage != null
              ? `${power.input_voltage.toFixed(2)} V${power.output_watts != null ? ` · ${power.output_watts.toFixed(1)} W` : ""}`
              : <UnavailableValue
                label="Input voltage unavailable"
                reason="This appliance reports no input voltage measurement"
              />}
          </dd>
        </div>
        <div>
          <dt><Icon name="bolt" size={15} />Battery</dt>
          <dd>
            {battery?.available && battery.percentage != null
              ? `${Math.round(battery.percentage)}%${charging === true ? " · charging" : charging === false ? " · on battery" : ""}`
              : <UnavailableValue
                label="Battery charge unavailable"
                reason="No battery or UPS has been reported for this appliance"
              />}
          </dd>
        </div>
        <div>
          <dt><Icon name="activity" size={15} />If mains is lost</dt>
          <dd>
            {"minutes" in runtime
              ? formatRuntime(runtime.minutes)
              : <UnavailableValue label="Battery runtime unknown" reason={runtime.reason} />}
          </dd>
        </div>
      </dl>
      {!("minutes" in runtime) && battery?.available && (
        <p className="enclosure-panel__note">
          Vaelor will estimate the runtime once it has seen this appliance run on
          battery. It will not guess one before then.
        </p>
      )}
    </section>
  );
}
