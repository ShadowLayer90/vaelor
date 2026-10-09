import type { CSSProperties } from "react";
import { thermalPolicy, type MachineProfile } from "../lib/machine";
import type { Session } from "../types";
import { CoolingCapabilityNotice } from "./CoolingCapabilityNotice";
import { Icon, ICON_SIZE } from "./Icon";
import { StatusPill } from "./StatusPill";
import {
  coolingChangeCount,
  CPU_MODE_LABELS,
  fanCurveFaults,
  LED_MODE_LABELS,
  type CoolingPolicy,
} from "./coolingPolicy";
import { KvGrid, SectionCard } from "./systemUi";
import { Button, Notice, SegmentedControl, UnavailableValue } from "./ui";

/**
 * System › Cooling (VD-200, the Cooling board): the fan reading, the CPU
 * temperature control beside the cooling curve, the enclosure's airflow start
 * point beside its indicator lights, and one policy bar that applies them
 * together. Only a Pi appliance has this tab.
 */
export function SystemCooling({
  cooling,
  machine,
  onReviewPolicy,
  session,
}: {
  cooling: CoolingPolicy;
  machine: MachineProfile;
  /** Boost and a custom curve change thermal behaviour, so they are reviewed in a dialog first. */
  onReviewPolicy: () => void;
  session: Session;
}) {
  const { caseFan, cpuFan, drafts, policyBusy, state } = cooling;
  const caseFanCapability = machine.capabilities.case_fan;
  const cpuFanCapability = machine.capabilities.cpu_fan;
  const isAppliance = machine.machine_class === "pi-appliance";
  const thermal = thermalPolicy(machine.machine_class);
  const canControl = session.user.role !== "viewer";
  const safetyLimit = cpuFan?.safety_limit ?? thermal.cpuCritical;
  const curveFaults = fanCurveFaults(drafts.customCurve, thermal);
  const customCurveValid = curveFaults.every((fault) => fault === null);
  const maximumCoolingState = cpuFan?.max_state ?? 4;
  const coolingState = cpuFan?.current_state;
  const commandableTargets = [caseFanCapability.available, cpuFanCapability.available].filter(Boolean).length;
  const applyBlockedReason = !canControl
    ? "Operator access is required to change the cooling policy."
    : commandableTargets === 0
      ? caseFanCapability.reason ?? cpuFanCapability.reason
        ?? "No controllable cooling hardware was detected on this machine."
      : null;
  const canApply = canControl && Boolean(state) && !policyBusy && commandableTargets > 0;
  const changes = coolingChangeCount({
    appliedBoostMinutes: cooling.appliedBoostMinutes,
    caseFan,
    cpuFan,
    caseAvailable: caseFanCapability.available,
    cpuAvailable: cpuFanCapability.available,
    drafts,
  });

  /* What the CPU fan reading means when it is not an ordinary live speed. */
  const cpuExplanation = cpuFan?.rpm === 0 && cpuFan.mode === "automatic"
    ? "CPU fan stopped by the protected automatic curve until its next threshold."
    : cpuFan?.rpm == null
      ? "CPU fan RPM telemetry is unavailable; temperature protection remains active."
      : null;
  const caseExplanation = caseFan?.running === true && !caseFan.rpm_available
    ? "Enclosure fans are commanded on; this enclosure does not expose case-fan RPM telemetry."
    : caseFan?.running === false && !caseFan.rpm_available
      ? "Enclosure airflow is stopped by the selected profile; no case-fan RPM sensor is available."
      : "Enclosure airflow state is reported by the selected profile.";
  const temperature = typeof cpuFan?.temperature === "number" ? cpuFan.temperature : null;
  const caseFanCount = caseFan?.fan_count ?? 0;

  const heroItems = [
    ...(cpuFanCapability.available ? [{
      label: "CPU control",
      value: cpuFan?.mode === "boost" ? "Boost active" : cpuFan?.mode === "custom" ? "Custom curve" : "Automatic",
    }] : []),
    // "0 case fans · Status unavailable" would be a reading of a fan that is not there.
    ...(caseFanCapability.available ? [{
      label: `${caseFanCount} case fan${caseFanCount === 1 ? "" : "s"}`,
      value: caseFan?.running == null ? "Status unavailable" : caseFan.running ? "Running" : "Standby",
    }] : []),
    {
      label: "Cooling state",
      value: !cpuFanCapability.available
        ? <UnavailableValue label="Cooling state unavailable" reason={cpuFanCapability.reason ?? "No cooling controller"} />
        : coolingState == null
          ? "Not read"
          : `Level ${coolingState} of ${maximumCoolingState}`,
    },
  ];

  const curve = drafts.cpuMode === "custom" ? drafts.customCurve : cpuFan?.curve ?? [];
  /*
   * The Cooling board's short line for an airflow profile ("Level 1 · about
   * 50 °C"), from the CPU curve the profile follows; the appliance's own
   * sentence stays as the button's description. Profile 0 runs at every level.
   */
  const profileShortLabel = (id: number) => {
    if (id === 0) return "At every fan level";
    const point = cpuFan?.curve?.find((entry) => entry.state === id);
    if (!point) return null;
    return `${id === maximumCoolingState ? "Maximum" : `Level ${id}`} · about ${point.temperature} °C`;
  };
  const firstThreshold = curve[0]?.temperature;

  return (
    <div className="sys-section sys-cooling">
      {/*
        * LESSONS 1 / S-Y3: a fan controller that stopped answering left its
        * last speed and "Protected automatic" on screen in green. Say which
        * reading this is, and grey what came from it.
        */}
      {cooling.reading === "stale" && (
        <Notice severity="warning">
          {`${cooling.readError} The fan readings below are from ${new Date(cooling.readAt).toLocaleTimeString()}, the last time it answered.`}
        </Notice>
      )}
      <section aria-label="Cooling status" className="card ui-card sys-hero" id="cooling-controls">
        <span aria-hidden="true" className="sys-hero__tile"><Icon name="fan" size={ICON_SIZE.standalone} /></span>
        <div className="sys-hero__reading">
          <small>{cpuFanCapability.available ? "CPU fan speed" : "Processor temperature"}</small>
          <strong>
            {cpuFanCapability.available
              ? <>{cpuFan?.rpm == null
                  ? <UnavailableValue label="Fan speed unavailable" reason="This cooling controller does not report fan RPM" />
                  : Math.round(cpuFan.rpm).toLocaleString()} <span>RPM</span></>
              : <>{temperature?.toFixed(1)
                  ?? <UnavailableValue label="Processor temperature unavailable" reason="No processor temperature sensor was reported" />} <span>°C</span></>}
          </strong>
          {/* Only a Pi has Raspberry Pi firmware. */}
          <p>{cpuFanCapability.available
            ? [
              temperature === null ? "Processor temperature not read" : `Processor at ${Math.round(temperature)} °C`,
              `${isAppliance ? "Raspberry Pi firmware" : "the host firmware"} owns the base curve`,
            ].join(" · ")
            : `${cpuFanCapability.reason}. Temperature is still read and reported; there is nothing here to command.`}</p>
          {cpuFanCapability.available && cpuExplanation && <p>{cpuExplanation}</p>}
        </div>
        <KvGrid className="sys-hero__facts" items={heroItems} label="Cooling state" />
      </section>

      <div className="sys-grid-2">
        {cpuFanCapability.available ? (
          <SectionCard
            actions={<StatusPill
              label={cpuFan?.mode === "boost" ? "Boost active" : cpuFan?.mode === "custom" ? "Custom curve active" : "Protected automatic"}
              tone={cpuFan?.mode === "boost" ? "warning" : cpuFan ? "success" : "neutral"}
              reading={cpuFan ? cooling.reading : "unread"}
            />}
            description={`Every mode keeps the ${safetyLimit} °C maximum-cooling override.`}
            title="CPU temperature control"
            titleId="cpu-policy-title"
          >
            <div className="sys-stack">
              <div className="sys-field">
                <span className="sys-field__label">Operating mode</span>
                <SegmentedControl
                  label="CPU fan operating mode"
                  onChange={cooling.setCpuMode}
                  options={(["automatic", "custom", "boost"] as const).map((mode) => ({
                    value: mode,
                    label: CPU_MODE_LABELS[mode],
                    disabled: !canControl || policyBusy || (mode !== "automatic" && !cpuFan?.writable),
                  }))}
                  value={drafts.cpuMode}
                />
              </div>
              {drafts.cpuMode === "boost" && <>
                <div className="sys-field">
                  <span className="sys-field__label" id="boost-level-label">Minimum cooling level</span>
                  <div aria-labelledby="boost-level-label" className="sys-choice-grid sys-choice-grid--4" role="group">
                    {Array.from({ length: maximumCoolingState }, (_, index) => index + 1).map((level) => {
                      const point = cpuFan?.curve?.find((entry) => entry.state === level);
                      return (
                        <Button
                          aria-pressed={drafts.cpuLevel === level}
                          className="sys-choice sys-choice--center"
                          disabled={!canControl || policyBusy || !cpuFan?.writable}
                          key={level}
                          onClick={() => cooling.setCpuLevel(level)}
                          type="button"
                          variant="secondary"
                        >
                          <strong>{point?.percent ?? Math.round(level / maximumCoolingState * 100)}%</strong>
                          <small>Level {level}</small>
                        </Button>
                      );
                    })}
                  </div>
                </div>
                <div className="sys-field">
                  <span className="sys-field__label">Return to automatic</span>
                  <SegmentedControl
                    label="CPU fan boost duration"
                    onChange={(value) => cooling.setCpuDuration(Number(value))}
                    options={[5, 15, 30].map((minutes) => ({
                      value: String(minutes),
                      label: `${minutes} min`,
                      disabled: !canControl || policyBusy || !cpuFan?.writable,
                    }))}
                    value={String(drafts.cpuDuration)}
                  />
                </div>
              </>}
              {/*
                * #149: each level owns its error; an emptied field is held as
                * NaN and rendered empty, and Apply stays blocked on it.
                */}
              {drafts.cpuMode === "custom" && (
                <div className="sys-field">
                  <span className="sys-field__label">Start each cooling level at</span>
                  {drafts.customCurve.map((point, index) => {
                    const fault = curveFaults[index];
                    const faultId = `curve-level-${point.state}-error`;
                    return (
                      <div className="sys-curve-row" key={point.state}>
                        <label>
                          <span>Level {point.state} · {point.percent}%</span>
                          <span>
                            <input
                              aria-describedby={fault ? faultId : undefined}
                              aria-invalid={fault ? true : undefined}
                              aria-label={`Level ${point.state} · ${point.percent}% temperature`}
                              className="fan-temperature-input"
                              disabled={policyBusy}
                              max={thermal.curveMaximum}
                              min={thermal.curveMinimum}
                              onChange={(event) => {
                                const raw = event.target.value;
                                cooling.changeCurveTemperature(index, raw === "" ? Number.NaN : Number(raw));
                              }}
                              step={0.1}
                              type="number"
                              value={Number.isFinite(point.temperature) ? point.temperature : ""}
                            /> °C
                          </span>
                        </label>
                        {fault && <small className="ui-field__error" id={faultId} role="alert">{fault}</small>}
                      </div>
                    );
                  })}
                  <small className="sys-muted">Thresholds must rise from level to level. Maximum cooling is always forced at {safetyLimit}°C.</small>
                </div>
              )}
              {cpuFan?.detected && !cpuFan.writable && (
                <p className="sys-muted">Boost needs write access to the Linux cooling device. Automatic monitoring remains available.</p>
              )}
            </div>
          </SectionCard>
        ) : (
          <SectionCard
            actions={<StatusPill label="No controllable fan" tone="neutral" />}
            description={`${cpuFanCapability.reason}.`}
            title="CPU temperature control"
            titleId="cpu-policy-title"
          />
        )}

        {cpuFanCapability.available && (
          <SectionCard
            // LESSONS 1 / VD-200 system verify: the accent says "read just now"; a
            // temperature from before the controller stopped answering is grey and Old.
            actions={<span className={temperature !== null && !cooling.reading ? "sys-now" : "sys-now sys-now--old"}>
              {temperature === null ? "Not read" : `${Math.round(temperature)} °C ${cooling.reading ? "· Old" : "now"}`}
            </span>}
            title={drafts.cpuMode === "custom" ? "Custom cooling curve" : "Firmware cooling curve"}
          >
            <div
              aria-label={isAppliance ? "Raspberry Pi default CPU fan curve" : "Default CPU fan curve for this machine"}
              className={cooling.reading ? "sys-curve sys-curve--old" : "sys-curve"}
              role="img"
            >
              {/* Below the first threshold the fan is at cooling level 0. */}
              {firstThreshold !== undefined && (
                <span className={coolingState === 0 && drafts.cpuMode !== "custom" ? "is-current" : undefined} style={{ "--curve-height": "0%" } as CSSProperties}>
                  <strong>0%</strong><i /><small>&lt; {Number.isFinite(firstThreshold) ? firstThreshold : "—"}°</small>
                </span>
              )}
              {curve.map((point) => (
                <span
                  className={coolingState === point.state && drafts.cpuMode !== "custom" ? "is-current" : undefined}
                  key={point.state}
                  style={{ "--curve-height": `${point.percent}%` } as CSSProperties}
                >
                  <strong>{point.percent}%</strong><i /><small>{Number.isFinite(point.temperature) ? point.temperature : "—"}°</small>
                </span>
              ))}
            </div>
          </SectionCard>
        )}

        {/*
          * The two enclosure cards describe a Pironman case; on a machine with
          * no enclosure board the absence is named once instead.
          */}
        {!caseFanCapability.available && <CoolingCapabilityNotice machine={machine} />}
        {caseFanCapability.available && (
          <SectionCard
            description={`${caseFanCount} enclosure fan${caseFanCount === 1 ? "" : "s"} share one on/off threshold.`}
            footer={<span className="sys-muted">{caseExplanation}</span>}
            title="Enclosure airflow start point"
            titleId="cooling-policy-title"
          >
            <div className="sys-choice-grid sys-choice-grid--3">
              {state?.profiles.map((item) => {
                const short = profileShortLabel(item.id);
                return (
                  <Button
                    aria-pressed={drafts.profile === item.id}
                    className="sys-choice"
                    disabled={!canControl || policyBusy}
                    key={item.id}
                    onClick={() => cooling.setProfile(item.id)}
                    title={short ? item.description : undefined}
                    type="button"
                    variant="secondary"
                  >
                    <strong>{item.name}</strong>
                    <small>{short ?? item.description}</small>
                  </Button>
                );
              })}
            </div>
          </SectionCard>
        )}
        {caseFanCapability.available && (
          <SectionCard
            description="2 white indicators on the case fans"
            title="Case-fan indicator lights"
            titleId="fan-led-title"
          >
            <div className="sys-stack">
              <SegmentedControl
                label="Fan LED mode"
                onChange={cooling.setLed}
                options={(["follow", "on", "off"] as const).map((mode) => ({
                  value: mode,
                  label: LED_MODE_LABELS[mode],
                  disabled: !canControl || policyBusy,
                }))}
                value={drafts.led}
              />
              <p className="sys-muted">{drafts.led === "follow"
                ? "Lights are on while either fan spins."
                : drafts.led === "on" ? "Lights stay on whether or not the fans spin." : "Lights stay off whether or not the fans spin."}</p>
            </div>
          </SectionCard>
        )}
      </div>

      <section aria-labelledby="cooling-action-title" className="card ui-card sys-savebar">
        <div className="sys-savebar__text">
          <span className="sys-eyebrow">Cooling policy · {changes === 0 ? "no changes" : `${changes} change${changes === 1 ? "" : "s"}`}</span>
          <h2 id="cooling-action-title">Review and apply one coordinated policy</h2>
          <p>{applyBlockedReason && commandableTargets === 0
            ? applyBlockedReason
            : "Nothing on this tab takes effect until you apply it. Boost and custom curves are reviewed first because they change thermal behaviour."}</p>
        </div>
        <div className="sys-savebar__actions">
          {!cooling.partialApplication && (
            <Button disabled={!canApply} onClick={() => void cooling.apply(true)} title={applyBlockedReason ?? undefined} variant="secondary">Restore safe baseline</Button>
          )}
          {/* Disabled with the reason readable on the control itself. */}
          <Button
            disabled={!canApply || (drafts.cpuMode !== "automatic" && !cpuFan?.writable) || (drafts.cpuMode === "custom" && !customCurveValid)}
            disabledReason={applyBlockedReason ?? (drafts.cpuMode === "custom" && !customCurveValid ? "Fix the highlighted temperature thresholds to continue." : undefined)}
            onClick={() => { if (drafts.cpuMode !== "automatic") onReviewPolicy(); else void cooling.apply(); }}
            title={applyBlockedReason ?? undefined}
            variant="primary"
          >
            {policyBusy ? "Applying…" : canControl ? "Apply cooling policy" : "Operator access required"}
          </Button>
        </div>
        {/* The outcome, beside the control that caused it. */}
        {cooling.message && (
          <Notice className="sys-savebar__outcome" severity={cooling.applyFailed || cooling.partialApplication ? "danger" : "success"}>
            {cooling.message}
            {cooling.partialApplication && (
              <Button disabled={policyBusy} onClick={() => void cooling.apply(true)} type="button" variant="primary">Restore safe baseline</Button>
            )}
          </Notice>
        )}
      </section>
    </div>
  );
}
