import { StatusPill } from "./StatusPill";
import { SectionCard } from "./systemUi";
import { ListRow, UnavailableValue } from "./ui";
import type { MachineProfile } from "../lib/machine";

/**
 * The absent enclosure, stated once and in one place.
 *
 * Silently hiding the enclosure cards would leave the Cooling tab looking
 * half-built — a reader who knows Vaelor has case-fan controls would go
 * looking for the setting that vanished. Naming each absent piece, with the
 * reason discovery gave, makes the absence legible instead: there is nothing
 * missing, there is nothing there. Drawn as the boards draw an absent
 * controller (the SystemLighting board's "No lighting controller").
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
    <SectionCard
      className="cooling-absent"
      description="Vaelor found no enclosure controller here, so there is nothing on this machine for these controls to command."
      flush
      title="Enclosure controls"
      titleId="cooling-absent-title"
    >
      {rows.map((row) => (
        <ListRow
          detail={row.reason ?? "Not reported by this device"}
          key={row.key}
          title={<>{row.label} <UnavailableValue label={`${row.label} unavailable`} mark="—" reason={row.reason ?? "Not reported by this device"} /></>}
          trailing={<StatusPill label="Not detected" tone="neutral" />}
        />
      ))}
    </SectionCard>
  );
}
