import { useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import type { MachineProfile } from "../lib/machine";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { UnavailableValue } from "./ui";

/**
 * The enclosure, on the landing page, because the enclosure *is* the product.
 *
 * "Is the fan about to get loud", "are the lights on", "is the front screen
 * awake" are the three things an owner of this appliance actually wonders, and
 * every one of them was two clicks away inside `System`. Home showed a picture
 * of the case and none of its state.
 *
 * Each row is gated on the capability that was discovered, so a Pi without a
 * lighting controller gets no lighting row rather than a row reading "off".
 */

interface CaseFan {
  id: string;
  name: string;
  detected: boolean;
  rpm?: number | null;
  running?: boolean | null;
  profile?: number;
  mode?: string;
  current_state?: number | null;
  fan_count?: number;
}

interface FanSnapshot {
  profiles: Array<{ id: number; name: string }>;
  fans: CaseFan[];
}

interface LightingSnapshot {
  detected: boolean;
  enabled: boolean;
  color: string;
  style: string;
  styles: Array<{ id: string; name: string }>;
}

interface DisplaySnapshot {
  detected: boolean;
  enabled: boolean;
  pages: string[];
}

/**
 * What has happened to one enclosure read.
 *
 * A bare `T | null` could not tell "not asked yet" from "asked and the
 * controller never answered", so a failed `/fans` left the panel reading
 * "Reading" for as long as the page stayed open. Three outcomes, three values.
 */
export type Read<T> =
  | { state: "pending" }
  | { state: "ok"; value: T }
  | { state: "failed" };

const value = <T,>(read: Read<T>): T | null => (read.state === "ok" ? read.value : null);

export interface EnclosureRow {
  icon: "fan" | "gpio" | "oled";
  label: string;
  value: string | null;
  detail: string;
  /** Set when the value could not be read; carries the reason, never a zero. */
  unavailable?: string;
  /**
   * True only while the endpoint behind this row has not answered.
   *
   * Separate from `unavailable` because they are different facts with the same
   * appearance: "we have not asked yet" ends, and "this controller does not
   * report that" does not. The panel used to derive its whole status from
   * `unavailable` alone, so a Pironman whose cooling controller reports no RPM
   * — a permanent, correct answer — sat under a "Reading" pill forever.
   */
  pending?: boolean;
}

/**
 * The three states this panel can be in, which is one more than it had.
 *
 * `reading` and `settled` were the only two, and everything that was neither
 * landed in `reading`. Saying "All good" about readings that were never taken
 * would be the opposite error, so the middle answer gets its own display: the
 * panel has finished, and some of what it asked for is not reported by this
 * hardware.
 */
export type EnclosureState = "reading" | "partial" | "settled";

export function enclosureState(rows: EnclosureRow[]): EnclosureState {
  if (rows.some((row) => row.pending)) return "reading";
  return rows.some((row) => row.unavailable) ? "partial" : "settled";
}

/** The reason a row has no value, given what happened to the read behind it. */
function silence<T>(read: Read<T>, unreported: string, unanswered: string): string {
  return read.state === "ok" ? unreported : unanswered;
}

export function enclosureRows({
  fans,
  lighting,
  display,
  machine,
}: {
  fans: Read<FanSnapshot>;
  lighting: Read<LightingSnapshot>;
  display: Read<DisplaySnapshot>;
  machine: MachineProfile;
}): EnclosureRow[] {
  const rows: EnclosureRow[] = [];
  const fanState = value(fans);
  const lightingState = value(lighting);
  const displayState = value(display);
  const caseFan = fanState?.fans.find((fan) => fan.id === "case-gpio");
  const cpuFan = fanState?.fans.find((fan) => fan.id === "cpu-pwm");

  if (machine.capabilities.case_fan.available) {
    const profile = fanState?.profiles.find((entry) => entry.id === caseFan?.profile);
    rows.push({
      icon: "fan",
      label: "Case fans",
      value: caseFan?.running == null
        ? null
        : caseFan.running ? "Running" : "Idle",
      detail: profile ? `${profile.name} profile` : "Airflow profile not reported",
      pending: fans.state === "pending",
      unavailable: caseFan?.running == null
        ? silence(
          fans,
          "This enclosure controller does not report whether the case fans are turning",
          "The cooling controller did not answer",
        )
        : undefined,
    });
  }

  if (machine.capabilities.cpu_fan.available) {
    const rpm = typeof cpuFan?.rpm === "number" ? cpuFan.rpm : null;
    rows.push({
      icon: "fan",
      label: "Processor fan",
      value: rpm === null ? null : `${Math.round(rpm)} RPM`,
      detail: cpuFan?.mode === "automatic"
        ? `Automatic curve${cpuFan.current_state != null ? `, level ${cpuFan.current_state}` : ""}`
        : cpuFan?.mode
          ? `${cpuFan.mode.charAt(0).toUpperCase()}${cpuFan.mode.slice(1)}`
          : "Cooling mode not reported",
      pending: fans.state === "pending",
      unavailable: rpm === null
        ? silence(
          fans,
          "This cooling controller does not report fan RPM",
          "The cooling controller did not answer",
        )
        : undefined,
    });
  }

  if (machine.capabilities.case_lighting.available) {
    const style = lightingState?.styles.find((entry) => entry.id === lightingState.style);
    rows.push({
      icon: "gpio",
      label: "Lights",
      value: lightingState ? (lightingState.enabled ? style?.name ?? "On" : "Off") : null,
      detail: lightingState?.enabled ? lightingState.color : "Case lighting is switched off",
      pending: lighting.state === "pending",
      unavailable: lightingState
        ? undefined
        : silence(lighting, "", "The lighting controller did not answer"),
    });
  }

  if (machine.capabilities.oled.available) {
    rows.push({
      icon: "oled",
      label: "Front screen",
      value: displayState ? (displayState.enabled ? "On" : "Off") : null,
      detail: displayState?.enabled
        ? `${displayState.pages.length || "default"} page${displayState.pages.length === 1 ? "" : "s"}`
        : "The front display is asleep",
      pending: display.state === "pending",
      unavailable: displayState
        ? undefined
        : silence(display, "", "The front display did not answer"),
    });
  }

  return rows;
}

export function EnclosurePanel({ machine }: { machine: MachineProfile }) {
  const [fans, setFans] = useState<Read<FanSnapshot>>({ state: "pending" });
  const [lighting, setLighting] = useState<Read<LightingSnapshot>>({ state: "pending" });
  const [display, setDisplay] = useState<Read<DisplaySnapshot>>({ state: "pending" });

  const wantsFans = machine.capabilities.case_fan.available
    || machine.capabilities.cpu_fan.available;
  const wantsLighting = machine.capabilities.case_lighting.available;
  const wantsDisplay = machine.capabilities.oled.available;

  useEffect(() => {
    let cancelled = false;
    // Nothing is asked for that this machine has not got. An endpoint queried
    // for hardware the reader does not have is how a workstation ended up
    // fetching `/lighting` at all.
    /*
     * A swallowed rejection used to leave the read pending for the life of the
     * page, and the panel reported that as "Reading" - indefinitely, on a
     * controller that had already refused to answer.
     */
    if (wantsFans) {
      void apiRequest<FanSnapshot>("/fans")
        .then((next) => { if (!cancelled) setFans({ state: "ok", value: next }); })
        .catch(() => { if (!cancelled) setFans({ state: "failed" }); });
    }
    if (wantsLighting) {
      void apiRequest<LightingSnapshot>("/lighting")
        .then((next) => { if (!cancelled) setLighting({ state: "ok", value: next }); })
        .catch(() => { if (!cancelled) setLighting({ state: "failed" }); });
    }
    if (wantsDisplay) {
      void apiRequest<DisplaySnapshot>("/system/display")
        .then((next) => { if (!cancelled) setDisplay({ state: "ok", value: next }); })
        .catch(() => { if (!cancelled) setDisplay({ state: "failed" }); });
    }
    return () => { cancelled = true; };
  }, [wantsDisplay, wantsFans, wantsLighting]);

  const rows = enclosureRows({ fans, lighting, display, machine });
  if (!rows.length) return null;
  /*
   * One state value, three answers. The pill and its tone come from the same
   * call, so the panel cannot show a settled tone beside an unsettled word.
   */
  const state = enclosureState(rows);
  const pill = {
    reading: { tone: "neutral" as const, label: "Reading" },
    partial: { tone: "neutral" as const, label: "Partly reported" },
    settled: { tone: "success" as const, label: "All good" },
  }[state];

  return (
    <section
      aria-labelledby="enclosure-heading"
      className="data-panel enclosure-panel"
      data-enclosure-state={state}
    >
      <div className="panel-heading">
        <div>
          <h2 id="enclosure-heading">Enclosure</h2>
          <p>Cooling, lighting, and the front screen</p>
        </div>
        <StatusPill className="enclosure-panel__state" label={pill.label} tone={pill.tone} />
      </div>
      <dl className="enclosure-facts">
        {rows.map((row) => (
          <div key={row.label}>
            <dt><Icon name={row.icon} size={15} />{row.label}</dt>
            <dd>
              {row.value === null
                ? <UnavailableValue
                  label={`${row.label} state unavailable`}
                  reason={row.unavailable ?? "Not reported by this enclosure"}
                />
                : row.value}
              <small>{row.detail}</small>
            </dd>
          </div>
        ))}
      </dl>
      <a className="ui-button ui-button--quiet" href="#/system">Adjust in System</a>
    </section>
  );
}
