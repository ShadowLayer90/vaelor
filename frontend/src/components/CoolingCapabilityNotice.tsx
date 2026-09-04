import { Icon } from "./Icon";
import { UnavailableValue } from "./ui";
import type { MachineProfile } from "../lib/machine";

/**
 * The absent enclosure, stated once and in one place.
 *
 * Silently hiding the enclosure panels would leave the Cooling tab looking
 * half-built — a reader who knows Vaelor has case-fan controls would go
 * looking for the setting that vanished. Naming each absent piece, with the
 * reason discovery gave, makes the absence legible instead: there is nothing
 * missing, there is nothing there.
 */
export function CoolingCapabilityNotice({ machine }: { machine: MachineProfile }) {
  const rows = ([
    ["case_fan", "Enclosure fans"],
    ["case_lighting", "Case lighting"],
    ["oled", "Front display"],
  ] as const)
    .filter(([key]) => !machine.capabilities[key].available)
    .map(([key, label]) => ({ key, label, reason: machine.capabilities[key].reason }));

  if (!rows.length) return null;

  return (
    <section className="data-panel cooling-absent" aria-labelledby="cooling-absent-title">
      <div className="panel-heading">
        <div>
          <h2 id="cooling-absent-title">Enclosure controls</h2>
          <p>
            Vaelor found no enclosure controller here, so there is nothing on this
            machine for these controls to command. They are shown, and disabled,
            rather than hidden so that nothing looks missing.
          </p>
        </div>
        <Icon name="alert" />
      </div>
      <dl className="cooling-absent__list">
        {rows.map((row) => (
          <div key={row.key}>
            <dt>
              <UnavailableValue label={`${row.label} unavailable`} reason={row.reason ?? "Not reported by this device"} />
              <span>{row.label}</span>
            </dt>
            <dd>{row.reason}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
