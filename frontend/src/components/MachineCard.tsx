import { useEffect, useId, useRef } from "react";
import { RecheckOutcome, RecheckTrigger, useRecheck, type RecheckResult } from "./RecheckButton";
import { Button } from "./ui";
import { StatusPill } from "./StatusPill";
import { Icon } from "./Icon";
import { ClusterCard, KeyValues, type KeyValue } from "./ClusterPrimitives";
import { MachineMetrics, freshnessLabel, notInstalledSentence } from "./MachineMetrics";
import { WorkerSoftwareRow } from "./WorkerSoftwareRow";
import { residentInUseBytes, unmeasuredPhrase } from "./fleetMemory";
import { exactTime, formatBytes, formatTemperature } from "../lib/format";
import { nodeStateWord, pillToneForRow } from "../lib/fleetNodeState";
import type { ClusterMode } from "../lib/clusterMode";
import type { StatusTone } from "./ui/status";
import type { RowStatus } from "./clusterAppStatus";
import type { FleetLiveReading } from "../hooks/useFleetLiveMemory";
import {
  formatClusterLink,
  type ClusterInterface,
  type FleetNode,
  type NodeCapacity,
} from "./fleetTypes";
import "../styles/cluster-fleet.css";

/**
 * One machine card on the Fleet tab (VD-200, the ClusterFleet board), and the
 * same card opened in place (the ClusterFleetDetail board).
 *
 * Closed, it leads with four readings - processor, memory, unified memory and
 * CPU temperature - each with a meter and a foot line, then the workloads
 * placed here, then "This machine" on the controller or the worker's software,
 * and a foot row that opens the card. Opened, it shows the workloads, the
 * machine's facts and cluster connection, the worker software in full, the
 * telemetry tiles with their trends, and the machine actions.
 *
 * The workloads it shows are the *real* per-node set the caller computes from
 * both deployment sources — pooled models placed here (via `pooled_deployments`
 * node_ids) as well as single-node-pinned services (the ledger's `reserved.from`)
 * — never from `reserved.from` alone, which the backend only fills for pinned
 * services. So a machine whose only workload is a pooled model shard shows that
 * model and is never mislabelled "Idle" (VD-033). A model that failed or was
 * unloaded stays on the card with its state, but only a running workload
 * counts toward the machine's Ready/Idle pill (ACC-085).
 */

/** A workload placed on this machine: a model (pooled here) or a pinned app. */
export interface MachineWorkload {
  name: string;
  kind: "model" | "app";
  /** Whether it is actually running here; a failed or unloaded model is not. */
  running: boolean;
  /**
   * Placed here and still being brought up (a `deploying` model). Not running
   * yet, so it is not counted as running, but the machine is not idle either.
   */
  starting?: boolean;
  /** A model's state word and tone, the same one its Deployments row shows. */
  status?: RowStatus;
  /** The model it serves, and which copy this machine holds ("Qwen3-30B-A3B · copy 1 of 2"). */
  detail?: string;
}

/**
 * The card's memory figure when none is current: why there is no figure. The
 * total follows in the foot line, as "of 64 GB".
 */
function unmeasuredLead(reading: FleetLiveReading | null | undefined): string {
  if (!reading?.stale && !reading?.readFailed) return "Memory use not reported";
  const phrase = unmeasuredPhrase(reading);
  return phrase.charAt(0).toUpperCase() + phrase.slice(1);
}

/** Memory used-fraction is amber past this, the same "nearly full" the bar shows. */
const NEARLY_FULL_PERCENT = 85;

/** A temperature meter is drawn as a share of 100 °C, as the boards draw it. */
const TEMPERATURE_METER_TOP_C = 100;

type MachinePill = { label: string; tone: StatusTone };

/**
 * The labelled status a machine reports (ACC-091). The controller says so
 * first — that is the machine's role, not a health verdict. Every other word
 * comes from the capacity ledger's ONE per-node `state` (contract 3): a worker
 * that is enrolled but never joined reads "Not joined", an operator's drain or
 * pause reads "Drained"/"Paused", a machine the cluster cannot reach reads
 * "Offline", and a node whose state could not be read — or a state this
 * console has no word for — reads "Unknown" (`nodeStateWord`), never Ready or
 * Idle. Only a READY node is then described by what it holds: nearly out of
 * memory is "Low space", running nothing is "Idle", the rest "Ready". "Idle"
 * counts only running workloads, so a node holding a pooled model shard is
 * never called idle and one whose only model failed is never called Ready.
 * A node whose only workload is still being deployed reads "Starting": it is
 * not running anything yet, and it is not idle either (SC8).
 * `usedPercent` is null when the machine's memory use was not measured, which
 * says nothing about space either way.
 */
export function machinePill(
  node: NodeCapacity,
  usedPercent: number | null,
  runningCount: number,
  startingCount = 0,
): MachinePill {
  if (node.role === "head-controller") return { label: "Controller", tone: "info" };
  if (node.state !== "ready") {
    const word = nodeStateWord(node.state);
    return { label: word.label, tone: word.tone };
  }
  if (usedPercent !== null && usedPercent >= NEARLY_FULL_PERCENT) {
    return { label: "Low space", tone: "warning" };
  }
  if (!runningCount && startingCount) return { label: "Starting", tone: "info" };
  if (!runningCount) return { label: "Idle", tone: "neutral" };
  return { label: "Ready", tone: "success" };
}

/**
 * The GPU's total addressable memory. Phase 1 tracks no GPU reservation, so a
 * "free" figure would always claim the whole accelerator is free even with a
 * model resident. Until reservation is tracked, the card states the size it can
 * back — the total — and never a free figure it cannot.
 *
 * `addressable_bytes` is the ledger's one ceiling and is taken as it comes. On
 * a unified part the ledger's ceiling is the shared aperture, and falling back
 * to the VRAM carve-out beside it is the 0.5 GiB figure VD-125 stopped showing.
 * A zero means the ledger has no ceiling for this node.
 */
function graphicsTotal(node: NodeCapacity): number {
  return node.capacity.gpu.addressable_bytes;
}

/** The processor family in words ("x86-64"); the raw word when it is not one this card knows. */
function architectureWords(architecture: string | undefined): string {
  const value = (architecture ?? "").trim().toLowerCase();
  if (value === "x86_64" || value === "amd64") return "x86-64";
  if (value === "aarch64" || value === "arm64") return "64-bit ARM";
  return architecture ?? "";
}

/** "Worker · x86-64 · 16 cores · GPU 96 GB": what this machine is, in one line. */
function machineLine(node: NodeCapacity, architecture: string | undefined): string {
  const role = node.role === "head-controller" ? "Cluster controller" : "Worker";
  const cores = node.capacity.cpu ? `${node.capacity.cpu} cores` : "";
  const gpu = node.capacity.gpu.present ? `GPU ${formatBytes(graphicsTotal(node))}` : "No GPU";
  return [role, architectureWords(architecture), cores, gpu].filter(Boolean).join(" · ");
}

/** When a reading stopped, short enough for a quarter-width foot: "13:42" within 12 hours, the full date and time before that. */
function sinceWords(epochSeconds: number): string {
  const when = new Date(epochSeconds * 1000);
  if (Date.now() - when.getTime() >= 12 * 3600 * 1000) return exactTime(when.getTime());
  return when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/** A share drawn as the board's 6 px meter; never drawn for a reading not taken. */
function Meter({ fraction, warn = false }: { fraction: number; warn?: boolean }) {
  const width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  return (
    <span aria-hidden="true" className={warn ? "cf-meter cf-meter--warn" : "cf-meter"}>
      <i style={{ width }} />
    </span>
  );
}

/** One of the card's four readings: its name, the figure, a meter, and a foot line. */
function Reading({ id, label, value, fraction, foot, warn }: {
  /** Which reading this is, for a test or a script to find it. */
  id: string;
  label: string;
  /** Null when not read: the cell then says so in words and draws no meter. */
  value: string | null;
  fraction: number | null;
  foot: string;
  warn?: boolean;
}) {
  return (
    <div className="cf-reading" data-reading={id}>
      <span className="cf-reading__label">{label}</span>
      {value === null
        ? <span className="cf-reading__value cf-reading__value--unread">Not read</span>
        : <span className="cf-reading__value">{value}</span>}
      {value !== null && fraction !== null && <Meter fraction={fraction} warn={warn} />}
      <span className="cf-reading__foot">{foot}</span>
    </div>
  );
}

/** The CPU temperature cell's foot: how old the readings are, or why there are none. */
function readingAge(reading: FleetLiveReading | null | undefined, measured: boolean): string {
  if (reading?.clockRefused) return "Readings refused";
  if (reading?.stale) return typeof reading.since === "number" ? `Not reporting since ${sinceWords(reading.since)}` : "Not reporting";
  return measured && typeof reading?.ageSeconds === "number" ? freshnessLabel(reading.ageSeconds) : "Not read";
}

/**
 * The closed card's four readings (the ClusterFleet board), all four from the
 * fleet's one measured read (`useFleetLiveMemory`, shared with the Free memory
 * tile, ACC-120): the controller's live snapshot, a worker's newest raw sample
 * while fresh (ACC-119). A closed card starts no poll of its own; the opened
 * card's tiles read on their own while it is open, as at 68ccdfb.
 */
function MachineReadings({ node, reading }: {
  node: NodeCapacity;
  reading: FleetLiveReading | null | undefined;
}) {
  const total = node.capacity.memory_bytes;
  const inUse = residentInUseBytes(node, reading);
  const free = inUse === null ? null : Math.max(0, total - inUse);
  const usedFraction = inUse === null || total <= 0 ? null : inUse / total;
  const gpu = node.capacity.gpu;
  // The GPU aperture's CURRENT residency: the fresh telemetry reading only. The
  // ledger's own figure is a snapshot from join or Recheck (ACC-084).
  const unifiedUsed = reading?.gttUsedBytes ?? null;
  const unifiedTotal = gpu.gtt_total_bytes || graphicsTotal(node);

  const cpu = reading?.cpuPercent ?? null;
  const temperature = reading?.cpuTemperatureC ?? null;
  const age = readingAge(reading, cpu !== null || temperature !== null);
  const readFailed = free !== null && reading?.readFailed;

  return (
    <>
      <div className={gpu.present ? "cf-readings" : "cf-readings cf-readings--three"}>
        <Reading
          id="processor"
          foot="Processor load"
          fraction={cpu === null ? null : cpu / 100}
          label="Processor"
          value={cpu === null ? null : `${Math.round(cpu)}%`}
        />
        <div className="cf-reading" data-reading="memory">
          <span className="cf-reading__label">Memory</span>
          {inUse === null || free === null ? (
            <>
              <span className="cf-reading__value cf-reading__value--unread">{unmeasuredLead(reading)}</span>
              <span className="cf-reading__foot">of {formatBytes(total)}</span>
            </>
          ) : (
            <>
              <span className="cf-reading__value">{formatBytes(inUse)}</span>
              {usedFraction !== null && <Meter fraction={usedFraction} warn={usedFraction * 100 >= NEARLY_FULL_PERCENT} />}
              <span className="cf-reading__foot">{formatBytes(free)} free of {formatBytes(total)}</span>
            </>
          )}
        </div>
        {gpu.present && (
          <Reading
            id="unified"
            foot={`of ${formatBytes(unifiedTotal)} shared aperture`}
            fraction={unifiedUsed === null || unifiedTotal <= 0 ? null : unifiedUsed / unifiedTotal}
            label="Unified memory"
            value={unifiedUsed === null ? null : formatBytes(unifiedUsed)}
          />
        )}
        <Reading
          id="temperature"
          foot={age}
          fraction={temperature === null ? null : temperature / TEMPERATURE_METER_TOP_C}
          label="CPU temp"
          value={temperature === null ? null : formatTemperature(temperature)}
        />
      </div>
      {/* SC5: a failed read keeps the last figures, and says so. */}
      {readFailed && <p className="cf-note cf-readings__note">The latest reading failed; showing the previous one.</p>}
    </>
  );
}

/** The workloads placed on this machine, one bordered row each, or why there are none. */
function Workloads({ workloads, isHead }: { workloads: MachineWorkload[]; isHead: boolean }) {
  if (!workloads.length) {
    return (
      <p className="cf-note">
        {isHead
          ? "No cluster workloads are placed on this machine."
          : "Nothing is running on this machine yet."}
      </p>
    );
  }
  return (
    <div className="cf-workloads">
      {workloads.map((workload) => (
        <span className="cf-workload" data-kind={workload.kind} key={`${workload.kind}-${workload.name}`}>
          <Icon aria-hidden="true" name={workload.kind === "model" ? "server" : "apps"} size={16} />
          <span className="cf-workload__name">{workload.name}</span>
          <span className="cf-workload__detail">{workload.detail ?? (workload.kind === "app" ? "App" : "")}</span>
          {workload.status && (
            <StatusPill tone={pillToneForRow(workload.status)} label={workload.status.label} />
          )}
        </span>
      ))}
    </div>
  );
}

/** "enp1s0 (192.0.2.20) · 2.5 GbE, wired", or why the link is not recorded. */
function connectionWords(link: ClusterInterface): string {
  if (!link.name) return link.reason || "not recorded";
  // VD-125: the ADDRESS beside the name, never the name alone - the fact is
  // captured against the address the control plane manages the machine on.
  const named = link.address ? `${link.name} (${link.address})` : link.name;
  return `${named} · ${formatClusterLink(link)}`;
}

interface Props {
  node: NodeCapacity;
  /**
   * This machine's current MEASURED memory and unified (GPU) memory, from
   * `useFleetLiveMemory` (ACC-083/119/120). Undefined/null when no fresh
   * reading is in hand, when the card says its memory is not reported.
   */
  reading?: FleetLiveReading | null;
  architecture: string | undefined;
  /** The fleet record for this machine, when it is an enrolled worker. */
  fleetNode: FleetNode | undefined;
  /**
   * VD-125. The controller's OWN cluster link, for the head card — which has no
   * `fleetNode` record (the controller is the Swarm manager, not one of its
   * members), so it shows its real connection the way a worker does.
   */
  controllerInterface?: ClusterInterface;
  /** The real workloads on this machine: pooled models here plus pinned apps. */
  workloads: MachineWorkload[];
  expanded: boolean;
  /** The Fleet page's Easy/Advanced detail level, threaded to the metrics. */
  mode: ClusterMode;
  /** Whether the viewer may act on the architecture-mismatch removal (admin). */
  administrator: boolean;
  onToggle: () => void;
  onNodeAction: (action: string, nodeId: string) => void;
  onRecheck: (nodeId: string) => Promise<RecheckResult>;
  /** E2b: install the telemetry agent on a machine awaiting its join (administrator). */
  onWorkerTelemetry: (action: "install", nodeId: string) => void;
}

/** The states in which the cluster has no live reading of the machine to show. */
const UNREACHABLE = new Set(["not-joined", "offline", "missing"]);

export function MachineCard({
  node,
  reading,
  architecture,
  fleetNode,
  controllerInterface,
  workloads,
  expanded,
  mode,
  administrator,
  onToggle,
  onNodeAction,
  onRecheck,
  onWorkerTelemetry,
}: Props) {
  const total = node.capacity.memory_bytes;
  // ACC-120: the memory the machine MEASURED as in use (the figure Home shows),
  // never the placement ledger's reservation; one derivation, shared with the
  // fleet headline. Null when no fresh reading is in hand.
  const inUse = residentInUseBytes(node, reading);
  const usedPercent =
    inUse === null ? null : total > 0 ? Math.min(100, (inUse / total) * 100) : 0;
  const running = workloads.filter((workload) => workload.running).length;
  const starting = workloads.filter((workload) => !workload.running && workload.starting).length;
  const pill = machinePill(node, usedPercent, running, starting);
  const isHead = node.role === "head-controller";
  const gpu = node.capacity.gpu;
  const unifiedUsed = reading?.gttUsedBytes ?? null;
  const joined = Boolean(fleetNode?.labels?.swarm_node_id);
  const drained = fleetNode?.runtime?.availability?.toLowerCase() === "drain";
  const unreachable = !isHead && UNREACHABLE.has(String(node.state ?? ""));
  const offline = pill.label === "Offline";
  const notInstalled = !isHead && fleetNode?.telemetry_provisioned === false;
  // VD-031/033. A worker that does not share this controller's architecture
  // cannot run its workloads at all, so the fact is stated on the machine
  // itself rather than hidden behind a green pill. Only a confirmed mismatch is
  // removable; an architecture that could not be read is reported and left.
  const arch = fleetNode?.architecture;
  const mismatched = Boolean(arch && !arch.matches_controller);
  // A worker carries its link on its fleet record; the controller has no such
  // record, so its live link arrives on `controllerInterface` (VD-125).
  const clusterInterface = fleetNode?.inventory?.cluster_interface ?? controllerInterface;
  const recheck = useRecheck(() => onRecheck(fleetNode?.id ?? ""));
  const detailId = "cf-detail-" + useId().replaceAll(":", "");
  const openButton = useRef<HTMLButtonElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const toggled = useRef(false);

  /*
   * The rail's worker-software card links here (VD-200):
   * `?cluster=fleet&machine=<enrolled id>#/fleet` brings this machine's card
   * into view once it has rendered.
   */
  const enrolledId = fleetNode?.id;
  const cardId = enrolledId ? `machine-${enrolledId}` : undefined;
  useEffect(() => {
    if (!cardId || new URLSearchParams(window.location.search).get("machine") !== enrolledId) return;
    document.getElementById(cardId)?.scrollIntoView?.({ block: "start" });
  }, [cardId, enrolledId]);

  // Opening or closing moves focus to the control that undoes it, so a
  // keyboard user is never left on a button that has gone.
  useEffect(() => {
    if (!toggled.current) return;
    (expanded ? closeButton : openButton).current?.focus();
  }, [expanded]);
  const toggle = () => {
    toggled.current = true;
    onToggle();
  };

  const pillNode = <StatusPill className="cf-card__pill" label={pill.label} tone={pill.tone} />;
  const mismatch = mismatched && arch && (
    <p className="cf-note cf-card__mismatch cl-warn-text" role="status">{arch.removable ? arch.removal_reason : arch.reason}</p>
  );
  const recheckControl = administrator
    ? <RecheckTrigger busy={recheck.busy} run={recheck.run} />
    : <p className="cf-actions__note">Only an administrator can Recheck this machine.</p>;
  const removeControls = fleetNode && (
    <>
      {arch?.removable && administrator && (
        <Button className="cl-danger-outline" onClick={() => onNodeAction("evict-mismatched", fleetNode.id)}>
          Review drain and removal
        </Button>
      )}
      <Button className="cl-danger-outline" onClick={() => onNodeAction("remove-node", fleetNode.id)}>
        Remove
      </Button>
    </>
  );
  const installTelemetry = fleetNode && !joined && administrator && !fleetNode.telemetry_provisioned && (
    <Button variant="quiet" onClick={() => onWorkerTelemetry("install", fleetNode.id)}>
      Install telemetry
    </Button>
  );

  if (!expanded) {
    const notJoined = unreachable && node.state === "not-joined";
    return (
      <ClusterCard
        as="article"
        actions={pillNode}
        className="cf-card"
        description={machineLine(node, architecture)}
        flush
        icon="server"
        iconAccent={isHead}
        id={cardId}
        title={node.name}
      >
        {unreachable ? (
          <div className="cf-section">
            <p className="cf-lead">
              <strong>{unmeasuredLead(reading)}</strong> of {formatBytes(total)}
            </p>
            {offline && (
              <p className="cf-note">Live metrics are unavailable while this machine is {pill.label.toLowerCase()}.</p>
            )}
            {/* ACC-091: why a machine cannot take work, in the ledger's words. */}
            {node.state_reason && <p className="cf-note">{node.state_reason}</p>}
            {notJoined && notInstalled && <p className="cf-note">{notInstalledSentence(administrator, joined)}</p>}
            {mismatch}
          </div>
        ) : (
          <>
            <MachineReadings node={node} reading={reading} />
            {(mismatch || node.state_reason) && (
              <div className="cf-section">
                {node.state_reason && <p className="cf-note">{node.state_reason}</p>}
                {mismatch}
              </div>
            )}
          </>
        )}
        {!notJoined && (
          <div className="cf-section">
            <p className="cf-label">Workloads</p>
            <Workloads isHead={isHead} workloads={workloads} />
          </div>
        )}
        {isHead ? (
          <div className="cf-section">
            <p className="cf-label">This machine</p>
            <p className="cf-text">This is the machine you are on; it is managed here.</p>
          </div>
        ) : fleetNode?.worker_software && !notJoined && (
          <div className="cf-section">
            <WorkerSoftwareRow administrator={administrator} mode={mode} software={fleetNode.worker_software} variant="summary" />
          </div>
        )}
        {notJoined && fleetNode && (
          /* The ClusterStates board: a machine awaiting its join keeps its
             three actions on the card. */
          <div className="cf-section">
            <div className="cf-actions">
              {installTelemetry}
              {recheckControl}
              <span className="cf-actions__spacer" />
              {removeControls}
            </div>
            <RecheckOutcome result={recheck.result} />
          </div>
        )}
        {/* It still opens (owner, 2026-10-07, VD-200 C2): the board drew no
            opener, but its worker software in full, Review conversion, its
            recorded connection and an early agent's trends live only there. */}
        <Button
          aria-controls={detailId}
          aria-expanded={false}
          className="cf-open"
          onClick={toggle}
          ref={openButton}
          variant="quiet"
        >
          <span>
            {notJoined && fleetNode ? "Worker software, connection and trends" : "Cluster connection, trends and machine actions"}{" "}
            <span className="sr-only">{`for ${node.name}`}</span>
          </span>
          <Icon name="chevron" size={16} />
        </Button>
      </ClusterCard>
    );
  }

  const facts: KeyValue[] = [{ label: "Cores", value: node.capacity.cpu ?? "—" }];
  if (gpu.present) {
    facts.push({ label: "GPU memory", value: formatBytes(graphicsTotal(node)) });
    facts.push({
      label: "Unified in use",
      value: unifiedUsed !== null ? `${formatBytes(unifiedUsed)} / ${formatBytes(gpu.gtt_total_bytes)}` : "Not reported",
    });
  }
  if (clusterInterface) facts.push({ label: "Cluster connection", value: connectionWords(clusterInterface) });

  return (
    <ClusterCard
      as="article"
      actions={(
        <div className="cf-head-actions">
          {pillNode}
          <Button aria-controls={detailId} aria-expanded className="cf-close" onClick={toggle} ref={closeButton} variant="quiet">
            <Icon name="chevron" size={16} />
            Close details{" "}
            <span className="sr-only">{`for ${node.name}`}</span>
          </Button>
        </div>
      )}
      className="cf-card is-open"
      description={machineLine(node, architecture)}
      icon="server"
      iconAccent={isHead}
      id={cardId}
      title={node.name}
    >
      <div className="cf-detail" id={detailId}>
        <p className="cf-label">Workloads on this machine</p>
        <Workloads isHead={isHead} workloads={workloads} />
        <KeyValues items={facts} label="Machine facts" />
        {node.state_reason && <p className="cf-note">{node.state_reason}</p>}
        {mismatch}
        <div className="cf-detail__split">
          {isHead ? (
            <div className="cf-software">
              <p className="cf-label">This machine</p>
              <p className="cf-text">This is the machine you are on; it is managed here.</p>
            </div>
          ) : fleetNode?.worker_software ? (
            <WorkerSoftwareRow administrator={administrator} mode={mode} software={fleetNode.worker_software} />
          ) : null}
          {/* E1/E2c: live + short-history metrics for this node. A worker
              reads its OWN node-tagged history by the fleet record's own `id`,
              which is what the history route validates a `node=` against; the
              controller uses the no-node route. An offline node degrades rather
              than drawing a fabricated chart. */}
          <MachineMetrics
            administrator={administrator}
            gpuPresent={gpu.present}
            isController={isHead}
            joined={joined}
            mode={mode}
            nodeId={isHead ? undefined : fleetNode?.id}
            offline={offline}
            stateLabel={pill.label}
            telemetryProvisioned={fleetNode?.telemetry_provisioned}
          />
        </div>
        {fleetNode?.telemetry_repair_note && <p className="cf-note cl-warn-text">{fleetNode.telemetry_repair_note}</p>}
        {fleetNode && (
          <div className="cf-actions-block">
            <div className="cf-actions">
              {joined && !drained && (
                <Button onClick={() => onNodeAction("drain-node", fleetNode.id)}>Drain</Button>
              )}
              {joined && drained && (
                <Button onClick={() => onNodeAction("resume-node", fleetNode.id)}>Return to service</Button>
              )}
              {/* VD-189 / LESSONS 19: the Recheck route is administrator-only, so
                  anyone else is told why rather than offered a press that fails. */}
              {recheckControl}
              {/* Owner, 2026-10-05 (VD-194): a worker's telemetry is part of its
                  worker software. It comes off when the machine leaves the
                  cluster, so no card offers to remove it on its own; a machine
                  still awaiting its join can have it installed ahead of time. */}
              {joined ? (
                <p className="cf-actions__note">
                  Telemetry is part of this worker&apos;s software; it comes off when the machine is removed from the cluster.
                </p>
              ) : (
                installTelemetry || <span className="cf-actions__spacer" />
              )}
              {removeControls}
            </div>
            <RecheckOutcome result={recheck.result} />
          </div>
        )}
      </div>
    </ClusterCard>
  );
}
