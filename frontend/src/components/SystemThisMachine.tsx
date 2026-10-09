import { useState } from "react";
import type { Device, Metrics, Session } from "../types";
import { apiRequest } from "../lib/api";
import { sampleTimeMs, type ConnectionState } from "../lib/connectionState";
import { timeAgo } from "../lib/format";
import type { HealthClaim } from "../lib/health";
import type { MachineProfile } from "../lib/machine";
import { DeviceHeroIcon } from "./DeviceHeroIcon";
import { Icon } from "./Icon";
import { ModalShell } from "./ModalShell";
import { OverviewHardwareChips } from "./OverviewHardwareChips";
import { PironmanDeviceIcon } from "./PironmanDeviceIcon";
import { StatusPill } from "./StatusPill";
import type { StorageSummary } from "./Sidebar";
import { SystemStripFacts } from "./SystemStripFacts";
import { Button, Card, Select } from "./ui";
import type { StatusTone } from "./ui/status";

/**
 * System › Compute › This machine (VD-200, the SystemCompute board): the
 * facts (uptime, Vaelor version, operating system, your access), the hardware
 * chips and the drawing of the machine. On a Pi appliance the enclosure
 * chooser sits under it. It was Home's hero before the redesign.
 */
export function SystemThisMachine({
  claim,
  connectionState,
  role,
  device,
  healthSampledAt,
  machine,
  metrics,
  noun,
  onDeviceSaved,
  onNotice,
  session,
  storage,
}: {
  claim: HealthClaim;
  connectionState: ConnectionState;
  /** "controller" when this machine leads a cluster; omitted otherwise. */
  role?: string;
  device: Device | null;
  /** When `/health` last gave its verdict, so a paused page can say how old it is. */
  healthSampledAt: number;
  machine: MachineProfile;
  metrics: Metrics;
  noun: string;
  onDeviceSaved: (device: Device) => Promise<void>;
  onNotice: (message: string, refused?: boolean) => void;
  session: Session;
  storage: StorageSummary | null;
}) {
  const isAppliance = machine.machine_class === "pi-appliance";
  const [deviceChoice, setDeviceChoice] = useState("");
  const [busy, setBusy] = useState(false);
  const [showSelector, setShowSelector] = useState(false);
  // VD-189: the selector's refusal is shown in the selector.
  const [selectorError, setSelectorError] = useState("");

  const saveDeviceChoice = async () => {
    if (!deviceChoice) return;
    setBusy(true);
    onNotice(""); setSelectorError("");
    try {
      await apiRequest(
        "/device/model",
        { method: "PATCH", body: JSON.stringify({ model: deviceChoice }) },
        session.csrf_token,
      );
      const selected = await apiRequest<Device>("/device");
      setShowSelector(false);
      onNotice("Pironman model saved. Hardware labels and the product image now match your selection.");
      await onDeviceSaved(selected).catch(() => {
        onNotice("Pironman model saved. Some unrelated live telemetry is still refreshing.");
      });
    } catch (error) {
      const text = error instanceof Error ? error.message : "The Pironman model could not be saved.";
      if (showSelector) setSelectorError(text); else onNotice(text, true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card
      actions={(
        <>
          <HealthPill claim={claim} connectionState={connectionState} sampledAt={healthSampledAt} />
          <span>{[device?.name, role].filter(Boolean).join(" · ")}</span>
        </>
      )}
      as="section"
      aria-label="System summary"
      className="this-machine device-hero"
      flush
      heading="This machine"
    >
      <div className="this-machine__grid">
      <div className="device-hero__content">
        <SystemStripFacts device={device} metrics={metrics} noun={noun} role={session.user.role} />
        <OverviewHardwareChips device={device} machine={machine} storage={storage} />
        {/*
          * Offering a Pironman model on a machine that is not one lets a reader
          * give a workstation a Pi enclosure's artwork. The picker belongs to
          * the appliance class and nowhere else.
          */}
        {device && isAppliance && !device.platform?.product.confident && (
          <div className="device-identification">
            <div>
              <strong>Which Pironman enclosure is this?</strong>
              <small>
                {device.platform?.product.detected_from
                  ? `Detected as ${device.platform.product.name} from ${device.platform.product.detected_from}, but not confidently. `
                  : "Automatic identification was inconclusive. "}
                Your choice controls hardware labels, available controls, and the product image.
              </small>
            </div>
            <Select
              aria-label="Pironman enclosure model"
              label={<span className="sr-only">Pironman enclosure model</span>}
              onChange={(event) => setDeviceChoice(event.target.value)}
              value={deviceChoice}
            >
              <option value="">Choose a model</option>
              {device.model_choices?.map((choice) => <option key={choice.id} value={choice.id}>{choice.name}</option>)}
            </Select>
            <Button disabled={!deviceChoice || busy} onClick={() => void saveDeviceChoice()} variant="primary">
              {busy ? "Saving…" : "Use this model"}
            </Button>
          </div>
        )}
      </div>
      <figure className="device-hero__visual">
        <div className="device-hero__glow" aria-hidden="true" />
        <DeviceHeroIcon device={device} isAppliance={isAppliance} machine={machine} />
        <figcaption>
          <span>{isAppliance ? "Active appliance" : "This machine"}</span>
          <strong>{device?.name ?? (isAppliance ? "Local appliance" : "This machine")}</strong>
          {session.user.role === "administrator" && isAppliance && device?.platform?.board.is_raspberry_pi && <Button onClick={() => {
            setDeviceChoice(device?.id ?? ""); setSelectorError("");
            setShowSelector(true);
          }} variant="quiet">Change enclosure</Button>}
        </figcaption>
      </figure>
      {showSelector && device && <ModalShell error={selectorError} labelledBy="device-selector-title" onClose={() => setShowSelector(false)} size="standard">
        <section className="device-selector" aria-labelledby="device-selector-title">
          <div className="panel-heading">
            <div><span className="page-eyebrow">Hardware identity</span><h2 id="device-selector-title">Choose your Pironman enclosure</h2><p>This changes the product image, labels, fan layout, NVMe count, and available hardware controls.</p></div>
            <Button onClick={() => setShowSelector(false)} variant="quiet">Close</Button>
          </div>
          {/* A single-select list, marked up as one (radios, not toggles). */}
          <div aria-labelledby="device-selector-title" className="device-selector__grid" role="radiogroup">
            {device.model_choices?.map((choice) => (
              <Button
                aria-checked={deviceChoice === choice.id}
                key={choice.id}
                onClick={() => setDeviceChoice(choice.id)}
                role="radio"
              >
                <span><PironmanDeviceIcon model={choice.id} size={126} /></span>
                <strong>{choice.name}</strong>
                {device.platform?.product.id === choice.id && (
                  <small>{device.platform.product.confident ? "Detected" : "Best match from discovery"}</small>
                )}
              </Button>
            ))}
          </div>
          <div className="device-selector__actions">
            <span><Icon name="shield" /> You can change this later without reinstalling.</span>
            <Button disabled={!deviceChoice || busy} onClick={() => void saveDeviceChoice()} variant="primary">{busy ? "Saving…" : "Use selected enclosure"}</Button>
          </div>
        </section>
      </ModalShell>}
      </div>
    </Card>
  );
}

/**
 * The health sentence as This machine's header pill (VD-200, the SystemCompute
 * board). It names what was checked and nothing else (`/health` reads
 * `cpu_temperature` and `memory_percent`): a reported verdict prints the
 * reassurance itself ("No processor or memory alerts"), never "All systems
 * operational". The word, the tone and the dot are one value, drawn by the
 * one StatusPill: a withdrawn verdict ("Health not known right now") is grey,
 * because not knowing is not a failure; a paused page shows the last verdict
 * as old, grey, with its age.
 */
export function healthPill(claim: HealthClaim, connectionState: ConnectionState, sampledAt: number, now = Date.now()): {
  label: string;
  tone: StatusTone;
  reading?: "unread" | "stale";
} {
  const word = claim.reported ? claim.detail.replace(/\.$/, "") : claim.title;
  if (connectionState === "error" || connectionState === "unknown" || !claim.reported || claim.checkedNothing) {
    return { label: word, tone: "neutral", reading: "unread" };
  }
  if (connectionState === "paused") {
    return { label: sampledAt > 0 ? `${word} · checked ${timeAgo(sampleTimeMs(sampledAt), now)}` : word, tone: "neutral", reading: "stale" };
  }
  return { label: word, tone: claim.pillStatus === "healthy" ? "success" : claim.pillStatus === "critical" ? "danger" : "warning" };
}

function HealthPill({ claim, connectionState, sampledAt }: {
  claim: HealthClaim;
  connectionState: ConnectionState;
  sampledAt: number;
}) {
  const pill = healthPill(claim, connectionState, sampledAt);
  return (
    <StatusPill
      className="health-pill"
      description={claim.reported ? undefined : claim.detail}
      label={pill.label}
      reading={pill.reading}
      tone={pill.tone}
    />
  );
}
