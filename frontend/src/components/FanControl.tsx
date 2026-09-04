import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { apiRequest } from "../lib/api";
import type { Session } from "../types";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { LightingControl } from "./LightingControl";
import { SystemInventoryPanel } from "./SystemInventoryPanel";
import { ConfirmDialog } from "./ConfirmDialog";
import { Button, Notice, TabSet, UnavailableValue } from "./ui";
import { destinationDescriptorFor, destinations } from "../lib/destinations";
import { useMachineProfile } from "../hooks/useMachineProfile";
import { thermalPolicy, unknownMachine } from "../lib/machine";
import { CoolingCapabilityNotice } from "./CoolingCapabilityNotice";
import { ComputePanel } from "./ComputePanel";

interface FanProfile {
  id: number;
  name: string;
  description: string;
}

interface FanDevice {
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
  curve?: Array<{ temperature: number; percent: number; state: number }>;
  safety_limit?: number;
  running?: boolean | null;
  profile?: number;
  led?: "follow" | "on" | "off";
  fan_count?: number;
  shared_control?: boolean;
  rpm_available?: boolean;
}

interface FanState {
  profiles: FanProfile[];
  fans: FanDevice[];
}

/**
 * `compute` is the workstation's first section and the appliance never has it;
 * `cooling` is the appliance's and the workstation never has it. Nothing on an
 * x86 chassis is controllable — the fan curve lives in the embedded controller
 * — so a Cooling workspace there was a whole tab whose only content was a
 * statement of absence. That statement moved onto the processor it is about,
 * in Compute, and the tab it used to occupy now describes the three engines
 * this machine genuinely has.
 */
export type CoolingSection = "compute" | "cooling" | "lighting" | "hardware";

export const COOLING_SAFE_BASELINE = {
  profile: 2,
  led: "follow" as const,
  cpuMode: "automatic" as const,
};

export function coolingSectionFromHash(hash: string): CoolingSection | null {
  const section = hash.match(/^#\/system\/(compute|cooling|lighting|hardware)(?:[/?].*)?$/)?.[1];
  // `null` means "the address bar did not ask for one", which is different
  // from "it asked for cooling": the default depends on the machine class and
  // is not knowable at the moment the hash is parsed.
  return section === "compute" || section === "lighting"
    || section === "hardware" || section === "cooling"
    ? section
    : null;
}

function coolingHashForSection(section: CoolingSection) {
  return `#/system/${section}`;
}

function FanProfileButton({
  item,
  active,
  disabled,
  onSelect,
}: {
  item: FanProfile;
  active: boolean;
  disabled: boolean;
  onSelect: (id: number) => void;
}) {
  return (
    <Button
      aria-pressed={active}
      className={active ? "fan-profile fan-profile--active" : "fan-profile"}
      disabled={disabled}
      onClick={() => onSelect(item.id)}
      type="button"
    >
      <span>{String(item.id).padStart(2, "0")}</span>
      <strong>{item.name}</strong>
      <small>{item.description}</small>
    </Button>
  );
}

/**
 * A hand-authored curve could not name a threshold above 79 °C, which is a Pi
 * ceiling: a processor that boosts to ~95 °C by design cannot be given a
 * usable curve inside it. The bounds come from the machine class now and
 * default to the appliance's, so a Pi is unchanged.
 *
 * #149 split the verdict per field: one form-level "use increasing
 * temperatures" never said which of the four levels was wrong, and the step
 * the form declared was checked by the browser's `checkValidity()` and by
 * nothing the Apply button could see — 45.3 drew no error while the button
 * stayed enabled. The step is one decimal place because that is the
 * precision the backend actually stores (`round(float(temperature), 1)` in
 * fan_control.py) — a stricter rule here would retroactively invalidate
 * curves the appliance itself saved.
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

export function FanControl({
  session,
  onBack,
}: {
  session: Session;
  onBack?: () => void;
}) {
  const [state, setState] = useState<FanState | null>(null);
  const [profile, setProfile] = useState(0);
  const [led, setLed] = useState<"follow" | "on" | "off">("follow");
  const [cpuMode, setCpuMode] = useState<"automatic" | "boost" | "custom">("automatic");
  const [customCurve, setCustomCurve] = useState([
    { temperature: 50, percent: 30, state: 1 },
    { temperature: 60, percent: 50, state: 2 },
    { temperature: 67.5, percent: 70, state: 3 },
    { temperature: 75, percent: 100, state: 4 },
  ]);
  const [cpuLevel, setCpuLevel] = useState(2);
  const [cpuDuration, setCpuDuration] = useState(15);
  const [section, setSection] = useState<CoolingSection | null>(() => coolingSectionFromHash(window.location.hash));
  const [policyBusy, setPolicyBusy] = useState(false);
  const [reviewCooling, setReviewCooling] = useState(false);
  const [partialApplication, setPartialApplication] = useState(false);
  const [message, setMessage] = useState("");
  // #149: a failed apply rendered in the informational blue Notice. Only the
  // partial-application case carried danger, so a complete failure looked
  // calmer than a partial success.
  const [applyFailed, setApplyFailed] = useState(false);
  const draftsInitialized = useRef(false);
  /**
   * Set the moment the reader changes any cooling draft, and never cleared.
   *
   * `draftsInitialized` alone was not enough, and this is the same defect the
   * lighting console had: the panel renders as soon as discovery answers,
   * which is a different request from `/fans`, so the first fan response can
   * land *after* the reader has begun editing a curve - and it reseeded every
   * draft on top of them. A poll may seed state nobody has touched; it may not
   * overwrite state somebody has.
   */
  const draftsTouched = useRef(false);
  const editDraft = <T,>(setter: (value: T) => void) => (value: T) => {
    draftsTouched.current = true;
    setter(value);
  };
  const machine = useMachineProfile();
  /*
   * Until discovery answers, nothing is claimed: the conservative profile
   * reports every enclosure capability unavailable, so no control can be
   * pressed in the window before the answer arrives.
   */
  const resolvedMachine = machine ?? unknownMachine;
  const caseFanCapability = resolvedMachine.capabilities.case_fan;
  const cpuFanCapability = resolvedMachine.capabilities.cpu_fan;
  const lightingCapability = resolvedMachine.capabilities.case_lighting;
  const isAppliance = resolvedMachine.machine_class === "pi-appliance";
  const thermal = thermalPolicy(resolvedMachine.machine_class);

  const refresh = useCallback(async () => {
    const next = await apiRequest<FanState>("/fans");
    setState(next);
    if (!draftsInitialized.current && !draftsTouched.current) {
      const nextCaseFan = next.fans.find((fan) => fan.id === "case-gpio");
      const nextCpuFan = next.fans.find((fan) => fan.id === "cpu-pwm");
      setProfile(nextCaseFan?.profile ?? 0);
      setLed(nextCaseFan?.led ?? "follow");
      setCpuMode(nextCpuFan?.mode ?? "automatic");
      if (nextCpuFan?.curve?.length) setCustomCurve(nextCpuFan.curve);
      setCpuLevel(nextCpuFan?.boost_level ?? Math.min(2, nextCpuFan?.max_state ?? 4));
      draftsInitialized.current = true;
    }
  }, []);

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => void refresh(), 3_000);
    return () => window.clearInterval(interval);
  }, [refresh]);

  useEffect(() => {
    const restoreSection = () => setSection(coolingSectionFromHash(window.location.hash));
    window.addEventListener("hashchange", restoreSection);
    window.addEventListener("popstate", restoreSection);
    return () => {
      window.removeEventListener("hashchange", restoreSection);
      window.removeEventListener("popstate", restoreSection);
    };
  }, []);

  /*
   * A bookmarked section must not land on a workspace that this machine has no
   * hardware for. It resolves to the class's own first section — Cooling on an
   * appliance, Compute on a workstation — rather than rendering an empty or
   * fabricated console.
   */
  const defaultSection: CoolingSection = isAppliance ? "cooling" : "compute";
  const sectionAvailable = (candidate: CoolingSection) =>
    candidate === "hardware"
    || (candidate === "lighting" && lightingCapability.available)
    || (candidate === "cooling" && isAppliance)
    || (candidate === "compute" && !isAppliance);
  const activeSection: CoolingSection =
    section && sectionAvailable(section) ? section : defaultSection;

  const navigateSection = (next: CoolingSection) => {
    if (next === section && window.location.hash === coolingHashForSection(next)) return;
    window.history.pushState(null, "", coolingHashForSection(next));
    // A result belongs to the surface that produced it. Leaving the section
    // retires it, so returning later does not present a stale outcome as if
    // it had just happened.
    setMessage("");
    setPartialApplication(false);
    setApplyFailed(false);
    setSection(next);
  };

  const applyCoolingPolicy = async (restoreBaseline = false) => {
    const selectedProfile = restoreBaseline ? COOLING_SAFE_BASELINE.profile : profile;
    const selectedLed = restoreBaseline ? COOLING_SAFE_BASELINE.led : led;
    const selectedMode = restoreBaseline ? COOLING_SAFE_BASELINE.cpuMode : cpuMode;
    setPolicyBusy(true);
    setMessage("");
    setReviewCooling(false);
    setPartialApplication(false);
    setApplyFailed(false);
    const previousCase = caseFan
      ? { profile: caseFan.profile ?? COOLING_SAFE_BASELINE.profile, led: caseFan.led ?? COOLING_SAFE_BASELINE.led }
      : null;
    let caseApplied = false;
    try {
      /*
       * The enclosure PATCH used to be unconditional. On a machine with no
       * enclosure board it still returned 200, wrote `gpio_fan_mode` and
       * `gpio_fan_led` to config, audited `fan.case.update` as a success, and
       * the screen said "Cooling policy applied" — a confirmation that work
       * had been done to hardware that does not exist. A request is only sent
       * for a controller discovery actually found.
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
        await apiRequest<FanState>(
          "/fans/cpu",
          { method: "PATCH", body: JSON.stringify(body) },
          session.csrf_token,
        );
      }
      await refresh();
      setMessage(restoreBaseline ? "Safe cooling baseline restored and verified." : "Cooling policy applied and current state refreshed.");
    } catch (error) {
      let recovery = "No compensating enclosure restore was available.";
      if (caseApplied && previousCase) {
        try {
          await apiRequest<FanState>(
            "/fans/case",
            { method: "PATCH", body: JSON.stringify(previousCase) },
            session.csrf_token,
          );
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

  const canControl = session.user.role !== "viewer";
  const cpuFan = state?.fans.find((fan) => fan.id === "cpu-pwm");
  const caseFan = state?.fans.find((fan) => fan.id === "case-gpio");
  const curveFaults = fanCurveFaults(customCurve, thermal);
  const customCurveValid = curveFaults.every((fault) => fault === null);
  /*
   * `canApply` carried no capability term at all, so Apply was enabled on
   * every host. A control that cannot act must not offer to act.
   */
  const commandableTargets = [
    caseFanCapability.available,
    cpuFanCapability.available,
  ].filter(Boolean).length;
  const applyBlockedReason = !canControl
    ? "Operator access is required to change the cooling policy."
    : !machine
      ? "Checking what cooling hardware this machine has."
      : commandableTargets === 0
        ? caseFanCapability.reason ?? cpuFanCapability.reason
          ?? "No controllable cooling hardware was detected on this machine."
        : null;
  const canApply = canControl && Boolean(state) && !policyBusy && commandableTargets > 0;
  const coolingState = cpuFan?.current_state;
  const maximumCoolingState = cpuFan?.max_state ?? 4;
  const cpuExplanation = cpuFan?.rpm === 0 && cpuFan.mode === "automatic"
    ? "CPU fan stopped by the protected automatic curve until its next threshold."
    : cpuFan?.rpm == null
      ? "CPU fan RPM telemetry is unavailable; temperature protection remains active."
      : "CPU fan RPM is live from the cooling controller.";
  const caseExplanation = !caseFanCapability.available
    ? caseFanCapability.reason ?? "No enclosure fan controller was detected on this machine."
    : caseFan?.running === true && !caseFan.rpm_available
      ? "Enclosure fans are commanded on; this enclosure does not expose case-fan RPM telemetry."
      : caseFan?.running === false && !caseFan.rpm_available
        ? "Enclosure airflow is stopped by the selected profile; no case-fan RPM sensor is available."
        : "Enclosure airflow state is reported by the selected profile.";
  /*
   * The tab strip was a literal of three. "Case lighting" is an entire
   * workspace for hardware a workstation does not have, and there is nothing
   * to explain inside it, so it is not offered; the absent lighting is named
   * once on Cooling instead, next to the other absent enclosure parts.
   */
  const systemTabs: Array<{ id: CoolingSection; label: string; icon: "fan" | "bolt" | "activity" | "cpu" }> = [
    ...(isAppliance
      ? [{ id: "cooling" as const, label: "Cooling", icon: "fan" as const }]
      : [{ id: "compute" as const, label: "Compute", icon: "cpu" as const }]),
    ...(lightingCapability.available
      ? [{ id: "lighting" as const, label: "Case lighting", icon: "bolt" as const }]
      : []),
    { id: "hardware", label: "Hardware & services", icon: "activity" },
  ];
  return (
    <div className="fan-page">
      <div className="page-heading">
        <div>
          <h1>{destinations.system.name}</h1>
          <p>{destinationDescriptorFor("system", resolvedMachine.machine_class)}.</p>
        </div>
        <div className="workspace-route-actions">
          {onBack && <Button variant="quiet" onClick={onBack} type="button">Back to overview</Button>}
          <Button disabled={policyBusy} onClick={() => void refresh()} variant="quiet">Reload</Button>
          {/*
            * "Detecting hardware" read as still-in-progress and never
            * resolved, so a reader with no enclosure waited forever. Once
            * discovery has answered the pill states the answer.
            */}
          {activeSection === "cooling" && <StatusPill
            label={!machine
              ? "Detecting hardware"
              : cpuFan?.detected && caseFan?.detected
                ? `${1 + (caseFan.fan_count ?? 0)} physical fans detected`
                : commandableTargets === 0
                  ? "No cooling controller"
                  : cpuFan?.detected ? "Processor fan detected" : "No cooling controller"}
            status={cpuFan?.detected ? "healthy" : "neutral"}
          />}
        </div>
      </div>

      {/* The section tabs were a hand-rolled `role="tablist"` of <Button>s with
          ArrowLeft/Right but no roving tabIndex and no Home/End. The shared
          `TabSet` primitive implements the whole ARIA tabs pattern once — roving
          focus, Arrow/Home/End, and panel wiring — and fits cleanly here because
          this component already renders exactly one active panel, which becomes
          TabSet's single swapped child. */}
      <TabSet
        items={systemTabs.map((tab) => ({
          id: tab.id,
          label: <><Icon name={tab.icon} />{tab.label}</>,
        }))}
        label="System sections"
        onSelect={(id) => navigateSection(id as CoolingSection)}
        selectedId={activeSection}
      >
      {activeSection === "cooling" && <>
        <section className="cooling-hero" aria-label="Cooling status" id="cooling-controls">
          <div className="cooling-hero__fan" aria-hidden="true"><span><Icon name="fan" size={58} /></span></div>
          <div className="cooling-hero__reading">
            <small>{cpuFanCapability.available ? "CPU fan speed" : "Processor temperature"}</small>
            <strong>{cpuFanCapability.available
              ? <>{cpuFan?.rpm == null
                  ? <UnavailableValue label="Fan speed unavailable" reason="This cooling controller does not report fan RPM" />
                  : Math.round(cpuFan.rpm).toLocaleString()} <span>RPM</span></>
              : <>{cpuFan?.temperature?.toFixed(1)
                  ?? <UnavailableValue label="Processor temperature unavailable" reason="No processor temperature sensor was reported" />} <span>°C</span></>}</strong>
            {/* Only a Pi has Raspberry Pi firmware. */}
            <p>{cpuFanCapability.available
              ? `${isAppliance ? "Raspberry Pi firmware" : "The host firmware"} owns the base curve. A timed boost can raise its minimum cooling level without disabling thermal protection.`
              : `${cpuFanCapability.reason}. Temperature is still read and reported; there is nothing here to command.`}</p>
          </div>
          <div className="cooling-hero__state">
            {cpuFanCapability.available && <span><small>CPU control</small><strong>{cpuFan?.mode === "boost" ? "Boost active" : cpuFan?.mode === "custom" ? "Custom curve" : "Automatic"}</strong></span>}
            {/* "0 case fans · Status unavailable" is a reading of a fan that is not there. */}
            {caseFanCapability.available && <span><small>{caseFan?.fan_count ?? 0} case fan{caseFan?.fan_count === 1 ? "" : "s"}</small><strong>{caseFan?.running == null ? "Status unavailable" : caseFan.running ? "Running" : "Standby"}</strong></span>}
            <span><small>Cooling state</small><strong>{cpuFanCapability.available
              ? <>{coolingState ?? "—"} / {cpuFan?.max_state ?? "—"}</>
              : <UnavailableValue label="Cooling state unavailable" reason={cpuFanCapability.reason ?? "No cooling controller"} />}</strong></span>
          </div>
        </section>

        <section className="cooling-telemetry" aria-label="Cooling telemetry explanations">
          <div><strong>CPU fan telemetry</strong><span>{cpuFanCapability.available ? cpuExplanation : cpuFanCapability.reason}</span></div>
          <div><strong>Case-fan telemetry</strong><span>{caseExplanation}</span></div>
          {/* The 80 °C override is a Pi safety constant, not a universal one. */}
          <div><strong>Cooling state scale</strong><span>{!cpuFanCapability.available
            ? "There is no cooling level to report on this machine."
            : coolingState == null
              ? "Waiting for the controller."
              : `Level ${coolingState} of ${maximumCoolingState}; level ${maximumCoolingState} is maximum cooling and is forced at ${cpuFan?.safety_limit ?? thermal.cpuCritical}°C.`}</span></div>
        </section>

        <div className="cooling-grid">
          {/*
            * These two panels described an enclosure: "0 enclosure fans share
            * one on/off threshold" with five enabled profile buttons, and "the
            * two white fan-status LEDs on GPIO 5 · 2 white GPIO indicators"
            * with three enabled mode buttons. Both are literals about a
            * Pironman case, and neither was gated on anything.
            */}
          {!caseFanCapability.available && <CoolingCapabilityNotice machine={resolvedMachine} />}
          {caseFanCapability.available && <section className="data-panel cooling-policy" aria-labelledby="cooling-policy-title">
            <div className="panel-heading"><div><h2 id="cooling-policy-title">Enclosure airflow start point</h2><p>{caseFan?.fan_count ?? 0} enclosure fan{caseFan?.fan_count === 1 ? "" : "s"} share one on/off threshold. Case-fan RPM is not reported by this enclosure.</p></div><Icon name="gpio" /></div>
            <div className="fan-profile-grid">{state?.profiles.map((item) => <FanProfileButton active={profile === item.id} disabled={!canControl || policyBusy} item={item} key={item.id} onSelect={editDraft(setProfile)} />)}</div>
          </section>}

          {caseFanCapability.available && <section className="data-panel fan-led-panel" aria-labelledby="fan-led-title">
            <div className="panel-heading"><div><h2 id="fan-led-title">Case-fan indicator lights</h2><p>Control the two white fan-status LEDs on GPIO 5. These are separate from the RGB case lighting.</p></div><Icon name="bolt" /></div>
            <div className="fan-led-preview" aria-hidden="true"><span className={`fan-led-preview__light fan-led-preview__light--${led}`} /><div><strong>{led === "follow" ? "Following both fans" : `Always ${led}`}</strong><small>2 white GPIO indicators</small></div></div>
            <div className="segmented-control" aria-label="Fan LED mode">{(["follow", "on", "off"] as const).map((mode) => <Button aria-pressed={led === mode} className="fan-led-mode" disabled={!canControl || policyBusy} key={mode} onClick={() => editDraft(setLed)(mode)} type="button">{mode}</Button>)}</div>
          </section>}

          {cpuFanCapability.available && <section className="data-panel cpu-policy-panel" aria-labelledby="cpu-policy-title">
            <div className="panel-heading"><div><h2 id="cpu-policy-title">CPU temperature control</h2><p>Use the protected firmware curve, define your own thresholds, or apply a temporary boost.</p></div><StatusPill label={cpuFan?.mode === "boost" ? "Boost active" : cpuFan?.mode === "custom" ? "Custom curve active" : "Protected automatic"} status={cpuFan?.mode === "boost" ? "degraded" : "healthy"} /></div>
            <div className="cpu-policy-layout">
              <div className="cpu-policy-controls">
                <div><span className="control-label">Operating mode</span><div className="segmented-control" aria-label="CPU fan operating mode">{(["automatic", "custom", "boost"] as const).map((mode) => <Button aria-pressed={cpuMode === mode} className="fan-cpu-mode" disabled={!canControl || policyBusy || (mode !== "automatic" && !cpuFan?.writable)} key={mode} onClick={() => editDraft(setCpuMode)(mode)} type="button">{mode}</Button>)}</div></div>
                {cpuMode === "boost" && <><div><span className="control-label">Minimum cooling level</span><div className="boost-levels">{Array.from({ length: cpuFan?.max_state ?? 4 }, (_, index) => index + 1).map((level) => { const curvePoint = cpuFan?.curve?.find((point) => point.state === level); return <Button aria-pressed={cpuLevel === level} className="fan-boost-level" disabled={!canControl || policyBusy || !cpuFan?.writable} key={level} onClick={() => editDraft(setCpuLevel)(level)} type="button"><strong>{curvePoint?.percent ?? Math.round(level / (cpuFan?.max_state ?? 4) * 100)}%</strong><small>Level {level}</small></Button>; })}</div></div><div><span className="control-label">Automatic return</span><div className="segmented-control" aria-label="CPU fan boost duration">{[5, 15, 30].map((duration) => <Button aria-pressed={cpuDuration === duration} className="fan-boost-duration" disabled={!canControl || policyBusy || !cpuFan?.writable} key={duration} onClick={() => editDraft(setCpuDuration)(duration)} type="button">{duration} min</Button>)}</div></div></>}
                {/*
                  * #149: each level owns its error now — one form-level alert
                  * styled like the permanent hint never said which of the four
                  * levels was wrong, no field carried aria-invalid, and an
                  * emptied field became a literal 0 (then "065" when 65 was
                  * typed over it) because Number("") is 0. An empty draft is
                  * held as NaN and rendered empty; Apply stays blocked on it.
                  */}
                {cpuMode === "custom" && <div className="custom-curve-editor"><span className="control-label">Start each cooling level at</span>{customCurve.map((point, index) => {
                  const fault = curveFaults[index];
                  const faultId = `curve-level-${point.state}-error`;
                  return <div className="custom-curve-row" key={point.state}><label><span>Level {point.state} · {point.percent}%</span><span><input className="fan-temperature-input" aria-describedby={fault ? faultId : undefined} aria-invalid={fault ? true : undefined} aria-label={`Level ${point.state} · ${point.percent}% temperature`} disabled={policyBusy} max={thermal.curveMaximum} min={thermal.curveMinimum} onChange={(event) => { const raw = event.target.value; const temperature = raw === "" ? Number.NaN : Number(raw); draftsTouched.current = true; setCustomCurve((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, temperature } : item)); }} step={0.1} type="number" value={Number.isFinite(point.temperature) ? point.temperature : ""} /> °C</span></label>{fault && <small className="ui-field__error" id={faultId} role="alert">{fault}</small>}</div>;
                })}<small>Thresholds must rise from level to level. Maximum cooling is always forced at {cpuFan?.safety_limit ?? thermal.cpuCritical}°C.</small></div>}
                {cpuFan?.detected && !cpuFan?.writable && <p className="control-warning">Boost needs write access to the Linux cooling device. Automatic monitoring remains available.</p>}
              </div>
              <div className="fan-curve" aria-label={isAppliance ? "Raspberry Pi default CPU fan curve" : "Default CPU fan curve for this machine"}><div className="fan-curve__header"><div><span className="control-label">{cpuMode === "custom" ? "Custom cooling curve" : "Firmware cooling curve"}</span><strong>{cpuMode === "custom" ? "Your thresholds" : "Default thresholds"}</strong></div><span>{cpuFan?.temperature?.toFixed(1) ?? "—"}°C now</span></div><div className="fan-curve__track" aria-hidden="true">{(cpuMode === "custom" ? customCurve : cpuFan?.curve)?.map((point) => <span key={point.state} style={{ "--curve-height": `${point.percent}%` } as CSSProperties}><i /><strong>{point.percent}%</strong><small>{Number.isFinite(point.temperature) ? point.temperature : "—"}°</small></span>)}</div><p><Icon name="shield" /> Automatic, custom, and boost modes all retain the {cpuFan?.safety_limit ?? thermal.cpuCritical}°C maximum-cooling safety override.</p></div>
            </div>
          </section>}

          {!cpuFanCapability.available && <section className="data-panel cpu-policy-panel" aria-labelledby="cpu-policy-title">
            <div className="panel-heading"><div><h2 id="cpu-policy-title">CPU temperature control</h2><p>{cpuFanCapability.reason}.</p></div><StatusPill label="No controllable fan" status="neutral" /></div>
          </section>}
        </div>

        <section className="data-panel cooling-policy-actions" aria-labelledby="cooling-action-title">
          <div><span className="control-label">Cooling policy</span><h2 id="cooling-action-title">Review and apply one coordinated policy</h2><p>{applyBlockedReason && commandableTargets === 0
            ? applyBlockedReason
            : "Nothing on this tab takes effect until you apply it. Case airflow and CPU fan mode are submitted together; boost and custom curves are reviewed first because they change thermal behavior."}</p></div>
          {/*
            * Apply used to be enabled on every host and reported "Cooling
            * policy applied and current state refreshed" after commanding
            * nothing. It now carries the same shape the power controls already
            * use: disabled, with the reason readable on the control itself.
            */}
          <div className="cooling-policy-actions__buttons"><Button disabled={!canApply || (cpuMode !== "automatic" && !cpuFan?.writable) || (cpuMode === "custom" && !customCurveValid)} disabledReason={applyBlockedReason ?? (cpuMode === "custom" && !customCurveValid ? "Fix the highlighted temperature thresholds to continue." : undefined)} onClick={() => { if (cpuMode !== "automatic") setReviewCooling(true); else void applyCoolingPolicy(); }} title={applyBlockedReason ?? undefined} variant="primary">{policyBusy ? "Applying…" : canControl ? "Apply cooling policy" : "Operator access required"}</Button>{!partialApplication && <Button disabled={!canApply} onClick={() => void applyCoolingPolicy(true)} title={applyBlockedReason ?? undefined} variant="quiet">Restore safe baseline</Button>}</div>
          {/*
            * The outcome, beside the control that caused it.
            *
            * This banner used to render at the very top of the page while
            * "Apply cooling policy" sits at the bottom of a long one, so a
            * reader pressed Apply, saw nothing change, and found
            * "Cooling policy applied" only by scrolling back up — the same
            * defect class as a button that reports success somewhere the
            * person is not looking. It is also scoped to this section now:
            * a cooling result has no business appearing above the lighting
            * panel, and it used to sit there alongside lighting's own banner.
            */}
          {message && (
            <Notice severity={applyFailed || partialApplication ? "danger" : "info"}>
              <Icon name="fan" />
              <span>{message}</span>
              {partialApplication && <Button disabled={policyBusy} onClick={() => void applyCoolingPolicy(true)} type="button" variant="primary">Restore safe baseline</Button>}
            </Notice>
          )}
        </section>
      </>}

      {activeSection === "compute" && <ComputePanel machine={resolvedMachine} />}
      {activeSection === "lighting" && <LightingControl session={session} />}
      {activeSection === "hardware" && <SystemInventoryPanel session={session} />}
      </TabSet>

      <ConfirmDialog
        busy={policyBusy}
        confirmLabel="Apply reviewed policy"
        description={cpuMode === "boost" ? `This starts CPU fan boost at level ${cpuLevel} for ${cpuDuration} minutes before returning to automatic control. Case airflow will also use the selected profile.` : `This custom CPU curve changes when cooling levels start. The ${cpuFan?.safety_limit ?? thermal.cpuCritical}°C maximum-cooling safety override remains active; case airflow will also use the selected profile.`}
        onCancel={() => !policyBusy && setReviewCooling(false)}
        onConfirm={() => void applyCoolingPolicy()}
        open={reviewCooling}
        title="Review cooling policy change"
      />
    </div>
  );
}
