import type { Device } from "../types";
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
  const batteryRead = Boolean(battery?.available && battery.percentage != null);
  const pill = power?.undervoltage_now
    ? { label: "Undervoltage", tone: "warning" as const }
    : mains ? { label: "Mains connected", tone: "success" as const } : { label: "Checking", tone: "neutral" as const };

  /* The SystemComputePi board's Power card: input, battery, and how long it would hold. */
  return (
    <section aria-labelledby="pi-power-heading" className="card ui-card sys-card enclosure-panel">
      <header className="ui-card__header">
        <div className="ui-card__titles">
          <h2 id="pi-power-heading">Power</h2>
          <p>Mains, battery, and how long it would hold</p>
        </div>
        <div className="ui-card__actions">
          <StatusPill label={pill.label} reading={pill.tone === "neutral" ? "unread" : undefined} tone={pill.tone} />
        </div>
      </header>
      <div className="ui-card__body sys-stack">
        <dl className="sys-kv sys-kv--readings">
          <div className="sys-kv__cell">
            <dt>Input</dt>
            <dd className="sys-kv__reading">
              {power?.input_voltage != null
                ? `${power.input_voltage.toFixed(2)} V`
                : <UnavailableValue label="Input voltage unavailable" mark="—" reason="This appliance reports no input voltage measurement" />}
            </dd>
            <dd className="sys-kv__detail">
              {power?.input_voltage == null
                ? "This appliance reports no input voltage measurement"
                : power.output_watts != null ? `${power.output_watts.toFixed(1)} W` : "Power draw not reported"}
            </dd>
          </div>
          <div className="sys-kv__cell">
            <dt>Battery</dt>
            <dd className="sys-kv__reading">
              {batteryRead
                ? `${Math.round(battery!.percentage!)}%`
                : <UnavailableValue label="Battery charge unavailable" mark="—" reason="No battery or UPS has been reported for this appliance" />}
            </dd>
            <dd className="sys-kv__detail">
              {!batteryRead
                ? "No battery or UPS has been reported for this appliance"
                : charging === true ? "charging" : charging === false ? "on battery" : "charge direction not reported"}
            </dd>
          </div>
          <div className="sys-kv__cell">
            <dt>If mains is lost</dt>
            <dd className="sys-kv__reading">
              {"minutes" in runtime
                ? formatRuntime(runtime.minutes)
                : <><UnavailableValue className="sys-unread-dot" label="Battery runtime unknown" mark="" reason={runtime.reason} />Not known yet</>}
            </dd>
            <dd className="sys-kv__detail">{"minutes" in runtime ? "From an observed discharge" : "Battery runtime unknown"}</dd>
          </div>
        </dl>
        {!("minutes" in runtime) && battery?.available && (
          <p className="sys-note enclosure-panel__note">
            {runtime.reason}. Vaelor will estimate the runtime once it has seen this appliance run on
            battery. It will not guess one before then.
          </p>
        )}
      </div>
    </section>
  );
}
