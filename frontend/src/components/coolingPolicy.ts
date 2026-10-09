import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import type { MachineProfile } from "../lib/machine";
import type { Session } from "../types";
import { useVisiblePoll } from "./systemUi";

/*
 * System › Cooling's state and its one coordinated apply, apart from the
 * screen that draws it (SystemCooling.tsx). The System page reads the same
 * state for its header pill, so the page and the tab cannot disagree about
 * how many fans were found.
 */

export interface FanProfile {
  id: number;
  name: string;
  description: string;
}

export interface FanDevice {
  id: "cpu-pwm" | "case-gpio";
  name: string;
  kind: "pwm" | "gpio";
  detected: boolean;
  control: "system-managed" | "firmware-with-overrides" | "profile" | "unavailable";
  rpm?: number | null;
  mode?: "automatic" | "boost" | "custom";
  current_state?: number | null;
  max_state?: number | null;
  boost_level?: number | null;
  boost_expires_at?: number | null;
  temperature?: number | null;
  writable?: boolean;
  curve?: CurvePoint[];
  safety_limit?: number;
  running?: boolean | null;
  profile?: number;
  led?: LedMode;
  fan_count?: number;
  shared_control?: boolean;
  rpm_available?: boolean;
}

export interface FanState {
  profiles: FanProfile[];
  fans: FanDevice[];
}

export type LedMode = "follow" | "on" | "off";
export type CpuMode = "automatic" | "boost" | "custom";
export interface CurvePoint { temperature: number; percent: number; state: number }

export const COOLING_SAFE_BASELINE = {
  profile: 2,
  led: "follow" as const,
  cpuMode: "automatic" as const,
};

/** The board's words for the three indicator modes (the Cooling board). */
export const LED_MODE_LABELS: Record<LedMode, string> = {
  follow: "Follow the fans",
  on: "Always on",
  off: "Always off",
};

export const CPU_MODE_LABELS: Record<CpuMode, string> = {
  automatic: "Automatic",
  custom: "Custom curve",
  boost: "Boost",
};

/**
 * A hand-authored curve could not name a threshold above 79 °C, which is a Pi
 * ceiling: a processor that boosts to ~95 °C by design cannot be given a
 * usable curve inside it. The bounds come from the machine class now and
 * default to the appliance's, so a Pi is unchanged.
 *
 * #149 split the verdict per field: one form-level "use increasing
 * temperatures" never said which of the four levels was wrong. The step is one
 * decimal place because that is the precision the backend actually stores
 * (`round(float(temperature), 1)` in fan_control.py).
 */
export function fanCurveFaults(
  curve: Array<{ temperature: number; state?: number }>,
  bounds: { curveMinimum: number; curveMaximum: number } = { curveMinimum: 35, curveMaximum: 79 },
): Array<string | null> {
  return curve.map((point, index) => {
    if (!Number.isFinite(point.temperature)) return "Enter a temperature.";
    if (point.temperature < bounds.curveMinimum || point.temperature > bounds.curveMaximum) {
      return `Use ${bounds.curveMinimum}–${bounds.curveMaximum}°C.`;
    }
    if (Math.abs(point.temperature * 10 - Math.round(point.temperature * 10)) > 1e-6) {
      return "Use one decimal place, like 45.3.";
    }
    const previous = index > 0 ? curve[index - 1] : null;
    if (previous && Number.isFinite(previous.temperature) && !(point.temperature > previous.temperature)) {
      return `Start above level ${previous.state ?? index}'s ${previous.temperature}°C.`;
    }
    return null;
  });
}

export function isValidFanCurve(
  curve: Array<{ temperature: number; state?: number }>,
  bounds: { curveMinimum: number; curveMaximum: number } = { curveMinimum: 35, curveMaximum: 79 },
) {
  return fanCurveFaults(curve, bounds).every((fault) => fault === null);
}

const sameCurve = (left: CurvePoint[], right: CurvePoint[] | undefined) =>
  Boolean(right) && left.length === right!.length
  && left.every((point, index) => point.temperature === right![index].temperature);

/**
 * What the drafts start from when the appliance leaves a field out. The seed
 * in `useCoolingPolicy` and the count below read the same defaults, or an
 * absent field is one permanent phantom "change".
 */
const CASE_DEFAULTS = { profile: 0, led: "follow" as LedMode };
/** The boost length the drafts start with; the appliance does not report a running boost's length. */
export const DEFAULT_BOOST_MINUTES = 15;

/**
 * How many settings the drafts change, counted against what the appliance
 * reported (the policy bar's "2 changes"). Only hardware discovery found is
 * counted: a draft for an absent controller is never sent, so it is no change.
 * `appliedBoostMinutes` is the length last sent: a running boost re-applied
 * for a different length is a change the appliance cannot report back.
 */
export function coolingChangeCount({
  appliedBoostMinutes = DEFAULT_BOOST_MINUTES,
  caseFan,
  cpuFan,
  caseAvailable,
  cpuAvailable,
  drafts,
}: {
  appliedBoostMinutes?: number;
  caseFan?: FanDevice;
  cpuFan?: FanDevice;
  caseAvailable: boolean;
  cpuAvailable: boolean;
  drafts: { profile: number; led: LedMode; cpuMode: CpuMode; cpuLevel: number; cpuDuration?: number; customCurve: CurvePoint[] };
}): number {
  let changes = 0;
  if (caseAvailable && caseFan) {
    if ((caseFan.profile ?? CASE_DEFAULTS.profile) !== drafts.profile) changes += 1;
    if ((caseFan.led ?? CASE_DEFAULTS.led) !== drafts.led) changes += 1;
  }
  if (cpuAvailable && cpuFan) {
    if ((cpuFan.mode ?? "automatic") !== drafts.cpuMode) changes += 1;
    else if (drafts.cpuMode === "custom" && !sameCurve(drafts.customCurve, cpuFan.curve)) changes += 1;
    else if (drafts.cpuMode === "boost" && (
      (cpuFan.boost_level ?? null) !== drafts.cpuLevel
      || (drafts.cpuDuration ?? appliedBoostMinutes) !== appliedBoostMinutes
    )) changes += 1;
  }
  return changes;
}

export function useCoolingPolicy({ active = true, machine, session }: {
  /** Cooling is on screen: `/fans` is read only then (it was polled every 3 s on every System tab). */
  active?: boolean;
  machine: MachineProfile;
  session: Session;
}) {
  const [state, setState] = useState<FanState | null>(null);
  /**
   * Why the last `/fans` read failed, and when the shown state was read. A
   * failed poll used to be swallowed, so the last reading stayed on screen,
   * green, for as long as the controller was silent (LESSONS 1, S-Y3).
   */
  const [readError, setReadError] = useState("");
  const [readAt, setReadAt] = useState(0);
  const [profile, setProfile] = useState(0);
  const [led, setLed] = useState<LedMode>("follow");
  const [cpuMode, setCpuMode] = useState<CpuMode>("automatic");
  const [customCurve, setCustomCurve] = useState<CurvePoint[]>([
    { temperature: 50, percent: 30, state: 1 },
    { temperature: 60, percent: 50, state: 2 },
    { temperature: 67.5, percent: 70, state: 3 },
    { temperature: 75, percent: 100, state: 4 },
  ]);
  const [cpuLevel, setCpuLevel] = useState(2);
  const [cpuDuration, setCpuDuration] = useState(DEFAULT_BOOST_MINUTES);
  const [appliedBoostMinutes, setAppliedBoostMinutes] = useState(DEFAULT_BOOST_MINUTES);
  const [policyBusy, setPolicyBusy] = useState(false);
  const [partialApplication, setPartialApplication] = useState(false);
  const [message, setMessage] = useState("");
  // #149: a failed apply rendered in the informational blue Notice.
  const [applyFailed, setApplyFailed] = useState(false);
  const draftsInitialized = useRef(false);
  /**
   * Set the moment the reader changes any cooling draft, and never cleared.
   * The first `/fans` answer can land after the reader has begun editing; a
   * poll may seed state nobody has touched, never overwrite state somebody has.
   */
  const draftsTouched = useRef(false);
  const edit = <T,>(setter: (value: T) => void) => (value: T) => {
    draftsTouched.current = true;
    setter(value);
  };

  const refresh = useCallback(async () => {
    let next: FanState;
    try {
      next = await apiRequest<FanState>("/fans");
    } catch (error) {
      setReadError(error instanceof Error && error.message ? error.message : "The fan controller did not answer.");
      throw error;
    }
    setState(next);
    setReadError("");
    setReadAt(Date.now());
    if (!draftsInitialized.current && !draftsTouched.current) {
      const nextCaseFan = next.fans.find((fan) => fan.id === "case-gpio");
      const nextCpuFan = next.fans.find((fan) => fan.id === "cpu-pwm");
      setProfile(nextCaseFan?.profile ?? CASE_DEFAULTS.profile);
      setLed(nextCaseFan?.led ?? CASE_DEFAULTS.led);
      setCpuMode(nextCpuFan?.mode ?? "automatic");
      if (nextCpuFan?.curve?.length) setCustomCurve(nextCpuFan.curve);
      setCpuLevel(nextCpuFan?.boost_level ?? Math.min(2, nextCpuFan?.max_state ?? 4));
      draftsInitialized.current = true;
    }
  }, []);

  useEffect(() => {
    if (active) void refresh().catch(() => undefined);
  }, [active, refresh]);
  useVisiblePoll(() => void refresh().catch(() => undefined), 3_000, active);

  const caseFanCapability = machine.capabilities.case_fan;
  const cpuFanCapability = machine.capabilities.cpu_fan;
  const caseFan = state?.fans.find((fan) => fan.id === "case-gpio");
  const cpuFan = state?.fans.find((fan) => fan.id === "cpu-pwm");

  /** A result belongs to the surface that produced it; leaving the section retires it. */
  const clearOutcome = () => {
    setMessage("");
    setPartialApplication(false);
    setApplyFailed(false);
  };

  const apply = async (restoreBaseline = false) => {
    const selectedProfile = restoreBaseline ? COOLING_SAFE_BASELINE.profile : profile;
    const selectedLed = restoreBaseline ? COOLING_SAFE_BASELINE.led : led;
    const selectedMode = restoreBaseline ? COOLING_SAFE_BASELINE.cpuMode : cpuMode;
    setPolicyBusy(true);
    clearOutcome();
    const previousCase = caseFan
      ? { profile: caseFan.profile ?? COOLING_SAFE_BASELINE.profile, led: caseFan.led ?? COOLING_SAFE_BASELINE.led }
      : null;
    let caseApplied = false;
    try {
      /*
       * A request is only sent for a controller discovery actually found: an
       * unconditional enclosure PATCH once audited `fan.case.update` as a
       * success on a machine with no enclosure board.
       */
      if (caseFanCapability.available) {
        await apiRequest<FanState>(
          "/fans/case",
          { method: "PATCH", body: JSON.stringify({ profile: selectedProfile, led: selectedLed }) },
          session.csrf_token,
        );
        caseApplied = true;
      }
      if (cpuFanCapability.available) {
        const body = selectedMode === "boost"
          ? { mode: selectedMode, level: cpuLevel, duration_minutes: cpuDuration }
          : selectedMode === "custom" ? { mode: selectedMode, curve: customCurve } : { mode: selectedMode };
        await apiRequest<FanState>("/fans/cpu", { method: "PATCH", body: JSON.stringify(body) }, session.csrf_token);
        if (selectedMode === "boost") setAppliedBoostMinutes(cpuDuration);
      }
      await refresh();
      if (restoreBaseline) {
        setProfile(COOLING_SAFE_BASELINE.profile);
        setLed(COOLING_SAFE_BASELINE.led);
        setCpuMode(COOLING_SAFE_BASELINE.cpuMode);
      }
      setMessage(restoreBaseline ? "Safe cooling baseline restored and verified." : "Cooling policy applied and current state refreshed.");
    } catch (error) {
      let recovery = "No compensating enclosure restore was available.";
      if (caseApplied && previousCase) {
        try {
          await apiRequest<FanState>("/fans/case", { method: "PATCH", body: JSON.stringify(previousCase) }, session.csrf_token);
          recovery = "The previous enclosure setting was restored.";
        } catch {
          recovery = "The previous enclosure setting could not be restored; use Restore safe baseline after checking the live state.";
        }
      }
      const refreshed = await refresh().then(() => true).catch(() => false);
      setApplyFailed(true);
      if (caseApplied) {
        setPartialApplication(true);
        setMessage(`Cooling policy was only partially applied: the enclosure setting succeeded, but the CPU fan setting failed. ${recovery} ${refreshed ? "Live state was refreshed." : "Live state could not be refreshed."}`);
      } else {
        setMessage(error instanceof Error ? `${error.message} ${refreshed ? "Live cooling state was refreshed." : "Live cooling state could not be refreshed."}` : `Cooling policy failed. ${refreshed ? "Live cooling state was refreshed." : "Live cooling state could not be refreshed."}`);
      }
    } finally {
      setPolicyBusy(false);
    }
  };

  const changeCurveTemperature = (index: number, temperature: number) => {
    draftsTouched.current = true;
    setCustomCurve((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, temperature } : item));
  };

  return {
    state,
    /** `unread`: nothing answered yet; `stale`: the last read failed and `state` is the one before. */
    reading: state === null ? "unread" as const : readError ? "stale" as const : undefined,
    readError,
    readAt,
    appliedBoostMinutes,
    caseFan,
    cpuFan,
    drafts: { profile, led, cpuMode, customCurve, cpuLevel, cpuDuration },
    setProfile: edit(setProfile),
    setLed: edit(setLed),
    setCpuMode: edit(setCpuMode),
    setCpuLevel: edit(setCpuLevel),
    setCpuDuration: edit(setCpuDuration),
    changeCurveTemperature,
    policyBusy,
    partialApplication,
    applyFailed,
    message,
    refresh,
    apply,
    clearOutcome,
  };
}

export type CoolingPolicy = ReturnType<typeof useCoolingPolicy>;
