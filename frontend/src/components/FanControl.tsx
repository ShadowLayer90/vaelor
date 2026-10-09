import { useEffect, useState } from "react";
import type { Session } from "../types";
import { StatusPill } from "./StatusPill";
import { LightingControl } from "./LightingControl";
import { SystemInventoryPanel } from "./SystemInventoryPanel";
import { ConfirmDialog } from "./ConfirmDialog";
import { SystemCooling } from "./SystemCooling";
import { useCoolingPolicy } from "./coolingPolicy";
import { Button, TabSet } from "./ui";
import type { StatusTone } from "./ui/status";
import { destinationDescriptorFor, destinations } from "../lib/destinations";
import { TopbarPageActions } from "../lib/topbarSlot";
import { useMachineProfile } from "../hooks/useMachineProfile";
import { thermalPolicy, unknownMachine } from "../lib/machine";
import { SystemCompute, type ShellTelemetry } from "./SystemCompute";
import {
  COMPUTE_SUBTITLE,
  COMPUTE_SUBTITLE_APPLIANCE,
  coolingSectionFromHash,
  resolvedSystemSection,
  SYSTEM_SECTION_EVENT,
  SYSTEM_SECTION_LABELS,
  type CoolingSection,
} from "../lib/systemSections";

/**
 * `compute` is every machine's section (VD-200: This machine and Live readings
 * live there); `cooling` and `lighting` are the appliance's. Nothing on an x86
 * chassis is controllable — the fan curve lives in the embedded controller —
 * so the processor's cooling is stated on Compute rather than in a Cooling tab
 * whose only content would be an absence.
 */
export type { CoolingSection } from "../lib/systemSections";
export { coolingSectionFromHash } from "../lib/systemSections";
export { COOLING_SAFE_BASELINE, fanCurveFaults, isValidFanCurve } from "./coolingPolicy";

/** What a section says about itself in the page header (the boards put it beside Reload). */
export interface SectionStatus {
  label: string;
  tone: StatusTone;
  reading?: "unread" | "stale";
}

function coolingHashForSection(section: CoolingSection) {
  return `#/system/${section}`;
}

export function FanControl({
  session,
  telemetry,
}: {
  session: Session;
  /** Accepted for the shell's signature; the boards draw no Back control (the rail's Home is the way back). */
  onBack?: () => void;
  /** The shell's live sample, for System › Compute (absent in a bare render). */
  telemetry?: ShellTelemetry;
}) {
  const [section, setSection] = useState<CoolingSection | null>(() => coolingSectionFromHash(window.location.hash));
  const [reviewCooling, setReviewCooling] = useState(false);
  const [reloadToken, setReloadToken] = useState(0);
  const [lightingStatus, setLightingStatus] = useState<SectionStatus | null>(null);
  const [hardwareStatus, setHardwareStatus] = useState<SectionStatus | null>(null);
  const machine = useMachineProfile();
  /*
   * Until discovery answers, nothing is claimed: the conservative profile
   * reports every enclosure capability unavailable, so no control can be
   * pressed in the window before the answer arrives.
   */
  const resolvedMachine = machine ?? unknownMachine;
  const isAppliance = resolvedMachine.machine_class === "pi-appliance";
  /*
   * A bookmarked section must not land on a workspace this machine has no
   * hardware for: it resolves to the class's own first section.
   *
   * Only Cooling reads `/fans`, so the poll runs only while Cooling is the
   * open section (it once ran every 3 s behind Compute, Lighting and
   * Hardware).
   */
  const activeSection: CoolingSection = resolvedSystemSection(section, resolvedMachine);
  /*
   * Before discovery answers, a Pi's default (or a bookmarked Cooling) is
   * still unknown, so the first read starts then rather than after it: the
   * fans arrive with the page instead of a beat behind its controls.
   */
  const coolingMayOpen = activeSection === "cooling" || (!machine && (section === null || section === "cooling"));
  const cooling = useCoolingPolicy({ active: coolingMayOpen, machine: resolvedMachine, session });
  const lightingCapability = resolvedMachine.capabilities.case_lighting;
  const thermal = thermalPolicy(resolvedMachine.machine_class);

  useEffect(() => {
    const restoreSection = () => setSection(coolingSectionFromHash(window.location.hash));
    window.addEventListener("hashchange", restoreSection);
    window.addEventListener("popstate", restoreSection);
    return () => {
      window.removeEventListener("hashchange", restoreSection);
      window.removeEventListener("popstate", restoreSection);
    };
  }, []);


  const navigateSection = (next: CoolingSection) => {
    if (next === section && window.location.hash === coolingHashForSection(next)) return;
    window.history.pushState(null, "", coolingHashForSection(next));
    // pushState raises nothing; the top bar's breadcrumb names the section too.
    window.dispatchEvent(new Event(SYSTEM_SECTION_EVENT));
    // A result belongs to the surface that produced it: leaving the section
    // retires it, so returning later does not present a stale outcome.
    cooling.clearOutcome();
    setSection(next);
  };

  const { caseFan, cpuFan } = cooling;
  const commandableTargets = [
    resolvedMachine.capabilities.case_fan.available,
    resolvedMachine.capabilities.cpu_fan.available,
  ].filter(Boolean).length;
  /*
   * "Detecting hardware" read as still-in-progress and never resolved, so a
   * reader with no enclosure waited forever. Once discovery has answered the
   * pill states the answer.
   */
  const coolingStatus: SectionStatus = cooling.reading ? {
    // LESSONS 1 / S-Y3: no fan count is claimed from a reading that did not answer.
    label: cooling.reading === "stale" ? "Old reading" : cooling.readError ? "Not read" : "Reading fans",
    tone: "neutral",
    reading: cooling.reading,
  } : {
    label: !machine
      ? "Detecting hardware"
      : cpuFan?.detected && caseFan?.detected
        ? `${1 + (caseFan.fan_count ?? 0)} physical fans detected`
        : commandableTargets === 0
          ? "No cooling controller"
          : cpuFan?.detected ? "Processor fan detected" : "No cooling controller",
    tone: cpuFan?.detected ? "success" : "neutral",
  };
  const sectionStatus = activeSection === "cooling"
    ? coolingStatus
    : activeSection === "lighting" ? lightingStatus : activeSection === "hardware" ? hardwareStatus : null;

  /*
   * Case lighting is a whole workspace for hardware a workstation does not
   * have, so it is offered only where discovery found a lighting controller.
   */
  const shown = { cooling: isAppliance, lighting: lightingCapability.available };
  // VD-200 decision 4 (the SystemComputePi board): Compute first on every
  // machine, then the appliance's Cooling and Case lighting, then Hardware.
  const systemTabs: Array<{ id: CoolingSection; label: string }> = [
    { id: "compute", label: SYSTEM_SECTION_LABELS.compute },
    ...(shown.cooling ? [{ id: "cooling" as const, label: SYSTEM_SECTION_LABELS.cooling }] : []),
    ...(shown.lighting ? [{ id: "lighting" as const, label: SYSTEM_SECTION_LABELS.lighting }] : []),
    { id: "hardware", label: SYSTEM_SECTION_LABELS.hardware },
  ];
  const subtitle = activeSection === "compute"
    ? isAppliance ? COMPUTE_SUBTITLE_APPLIANCE : COMPUTE_SUBTITLE
    : destinationDescriptorFor("system", resolvedMachine.machine_class);

  return (
    <div className="fan-page sys-page">
      <div className="page-heading system-page-heading">
        <div>
          <h1>{destinations.system.name}</h1>
          {/* The subtitle follows the tab: Compute has the board's own line. */}
          <p>{subtitle}.</p>
        </div>
      </div>
      {/* In the top bar (the System boards). Compute reads the shell's live
          sample; the other sections read their own and say what they found
          beside Reload. */}
      <TopbarPageActions>
        {sectionStatus && <StatusPill label={sectionStatus.label} reading={sectionStatus.reading} tone={sectionStatus.tone} />}
        {activeSection !== "compute" && (
          <Button
            disabled={cooling.policyBusy}
            onClick={() => {
              if (activeSection === "cooling") void cooling.refresh().catch(() => undefined);
              else setReloadToken((value) => value + 1);
            }}
            variant="secondary"
          >
            Reload
          </Button>
        )}
      </TopbarPageActions>

      {/* The shared TabSet implements the whole ARIA tabs pattern; this page
          renders exactly one active panel, its single swapped child. */}
      <TabSet
        items={systemTabs.map((tab) => ({ id: tab.id, label: tab.label }))}
        label="System sections"
        onSelect={(id) => navigateSection(id as CoolingSection)}
        selectedId={activeSection}
      >
        {activeSection === "cooling" && (
          <SystemCooling cooling={cooling} machine={resolvedMachine} onReviewPolicy={() => setReviewCooling(true)} session={session} />
        )}
        {activeSection === "compute" && <SystemCompute machine={resolvedMachine} session={session} telemetry={telemetry} />}
        {activeSection === "lighting" && <LightingControl onStatus={setLightingStatus} reloadToken={reloadToken} session={session} />}
        {activeSection === "hardware" && <SystemInventoryPanel onStatus={setHardwareStatus} reloadToken={reloadToken} session={session} />}
      </TabSet>

      <ConfirmDialog
        busy={cooling.policyBusy}
        confirmLabel="Apply reviewed policy"
        description={cooling.drafts.cpuMode === "boost"
          ? `This starts CPU fan boost at level ${cooling.drafts.cpuLevel} for ${cooling.drafts.cpuDuration} minutes before returning to automatic control. Case airflow will also use the selected profile.`
          : `This custom CPU curve changes when cooling levels start. The ${cpuFan?.safety_limit ?? thermal.cpuCritical}°C maximum-cooling safety override remains active; case airflow will also use the selected profile.`}
        onCancel={() => !cooling.policyBusy && setReviewCooling(false)}
        onConfirm={() => { setReviewCooling(false); void cooling.apply(); }}
        open={reviewCooling}
        title="Review cooling policy change"
      />
    </div>
  );
}
