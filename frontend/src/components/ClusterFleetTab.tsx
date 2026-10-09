import type { RecheckResult } from "./RecheckButton";
import { useState } from "react";
import { MeterBar, StatTile } from "./ui";
import { MachineCard, type MachineWorkload } from "./MachineCard";
import { pooledStatus, type PooledDeployment } from "./ClusterDeployments";
import { architectureForNode } from "./ClusterCapacity";
import { fleetMemoryHeadline } from "./fleetMemory";
import { useFleetLiveMemory, type FleetMemoryNode } from "../hooks/useFleetLiveMemory";
import { formatBytes } from "../lib/format";
import { shortModelName } from "../lib/homeSummary";
import { listedDeployments } from "../lib/deploymentRows";
import { fillTime, useDashboardRange, type DashboardPanel, type RangeLayer } from "../lib/performanceDashboard";
import type { AgentDeployment } from "../lib/clusterAgents";
import type { ClusterMode } from "../lib/clusterMode";
import type { Session } from "../types";
import type {
  ClusterCapacityLedger,
  FleetNode,
  FleetSummary,
  NodeCapacity,
} from "./fleetTypes";
import "../styles/cluster-fleet.css";

/**
 * The Fleet tab (VD-200, the ClusterFleet board): four stat tiles - free
 * memory across the fleet, the machines, the deployments, and the output
 * throughput the Performance tab reads - then a card per machine that opens
 * in place. Serve a model, Deploy app and Add machine sit in the page's title
 * row on every tab, so this tab carries none of its own; every machine action
 * is wired to the flows FleetCenter already owns.
 */

/** The head controller's own ledger row, keyed the way the ledger names it. */
function isController(node: NodeCapacity): boolean {
  return node.role === "head-controller" || node.node_id === "controller";
}

/** The fleet record for a capacity node, matched on any identifier the two sides share. */
function resolveFleetNode(
  node: NodeCapacity,
  fleetNodes: FleetNode[],
): FleetNode | undefined {
  if (isController(node)) return undefined;
  return fleetNodes.find(
    (fleet) =>
      fleet.id === node.node_id
      || fleet.labels?.swarm_node_id === node.node_id
      || fleet.name === node.name,
  );
}

/**
 * Which copy of a model this machine holds, in the record's own terms: a
 * replicated deployment runs one whole copy per machine ("copy 1 of 2"); a
 * distributed one splits one copy across them. A record from before the word
 * existed says neither.
 */
function placementWords(deployment: PooledDeployment, nodeId: string): string {
  const count = deployment.node_ids.length;
  const mode = deployment.units?.mode;
  if (count < 2 || !mode) return "";
  if (mode === "replicated") return `copy ${deployment.node_ids.indexOf(nodeId) + 1} of ${count}`;
  return `one copy split across ${count} machines`;
}

/**
 * The real workloads on one machine, from BOTH deployment sources.
 *
 * A pooled model records its placement in `pooled_deployments[].node_ids`, keyed
 * by the same placement id the capacity node carries ("controller" for the head,
 * the fleet node id for a worker); a single-node-pinned service is the only kind
 * the backend records in the ledger's `reserved.from`. Replicated services carry
 * no per-node task placement in the `/cluster` summary, so they are deliberately
 * NOT attributed to any node card — they appear only in Deployments. This is why
 * a node whose only workload is a pooled shard must read its model here.
 *
 * Each model carries its record's state through `pooledStatus` - the same word
 * and tone its Deployments row shows - and only a `healthy` record is running
 * (ACC-085). A failed or unloaded deployment keeps its record, and its
 * `node_ids`, until it is removed; it stays on the card, labelled, but it is
 * not counted as running, so it can never make a machine read Ready. A
 * `deploying` model is marked `starting`: not running yet, but the machine is
 * not idle while it comes up (SC8). A pinned app is in `reserved.from` only
 * while Swarm reserves it here, so it counts.
 */
export function workloadsForNode(
  node: NodeCapacity,
  pooled: PooledDeployment[],
): MachineWorkload[] {
  const models: MachineWorkload[] = pooled
    .filter((deployment) => deployment.node_ids.includes(node.node_id))
    .map((deployment) => ({
      name: deployment.name,
      kind: "model",
      running: deployment.state === "healthy",
      starting: deployment.state === "deploying",
      status: pooledStatus(deployment),
      detail: [deployment.model_id ? shortModelName(deployment.model_id) : "", placementWords(deployment, node.node_id)]
        .filter(Boolean).join(" · "),
    }));
  const apps: MachineWorkload[] = node.reserved.from.map((name) => ({
    name,
    kind: "app",
    running: true,
  }));
  return [...models, ...apps];
}

/** "185.0 GB" -> ["185.0", "GB"], so a tile draws the unit small. */
function splitQuantity(text: string): [string, string] {
  const match = text.match(/^([\d.,]+)\s*(.*)$/);
  return match ? [match[1], match[2]] : [text, ""];
}

function plural(count: number, one: string, many = one + "s"): string {
  return `${count} ${count === 1 ? one : many}`;
}

/** A figure for the throughput tile: whole tokens from ten up, one decimal below. */
function throughputFigure(value: number): string {
  return value >= 10 ? String(Math.round(value)) : value.toFixed(1);
}

/** The newest measured bucket in a line, or null when it has none. */
function newestBucket(values: Array<number | null> | undefined): { value: number; index: number } | null {
  for (let index = (values?.length ?? 0) - 1; index >= 0; index -= 1) {
    const value = values?.[index];
    if (typeof value === "number" && Number.isFinite(value)) return { value, index };
  }
  return null;
}

/**
 * A measured bucket counts as current when it is one of the last two of the
 * range (the newest may still be filling). An older one is stated with the
 * time it ended, never shown as the current throughput (review S2).
 */
const CURRENT_BUCKETS = 2;

export interface ThroughputReading {
  value: number | null;
  unit: string;
  /** The foot line: what was read and over which range, or why there is no figure. */
  foot: string;
  /** The figure is an older bucket's, not current: the tile draws it grey and says "Old". */
  old: boolean;
}

/**
 * The Performance tab's output throughput, read the way that tab reads it
 * (owner, 2026-10-06: the Fleet tile reads Performance): the newest real
 * value of the `output_throughput` panel over the last 15 minutes. The
 * cluster's summed line when the backend draws one, else the one machine
 * that serves. With no figure the tile says "Not read" with the backend's own
 * state and reason, never a zero.
 */
export function throughputReading(range: RangeLayer | null, error: string, loading: boolean): ThroughputReading {
  const newestValue = (values: Array<number | null> | undefined) => newestBucket(values)?.value ?? null;
  const panel: DashboardPanel | undefined = range?.panels?.output_throughput;
  const unit = panel?.unit ? range?.labels?.units?.[panel.unit] ?? "" : "";
  if (!range || !panel) {
    if (error) return { value: null, unit, foot: "The Performance page's readings could not be read.", old: false };
    return { value: null, unit, foot: loading ? "Reading the Performance page…" : "The Performance page sent no throughput.", old: false };
  }
  const lead = range.caption_lead ? range.caption_lead.charAt(0).toLowerCase() + range.caption_lead.slice(1) : "";
  const named = [panel.title, lead].filter(Boolean).join(", ");
  const lines = Array.isArray(panel.series) ? panel.series : [];
  const aggregate = lines.find((line) => line.kind === "aggregate");
  const measured = lines.filter((line) => newestValue(line.values) !== null);
  const line = aggregate && newestValue(aggregate.values) !== null ? aggregate : measured.length === 1 ? measured[0] : undefined;
  const bucket = line ? newestBucket(line.values) : null;
  if (bucket !== null) {
    const count = line?.values.length ?? 0;
    if (count - bucket.index <= CURRENT_BUCKETS || !range.step) return { value: bucket.value, unit, foot: named, old: false };
    const ended = range.start + (bucket.index + 1) * range.step;
    return {
      value: bucket.value,
      unit,
      foot: `${panel.title ?? "Output throughput"}, ${fillTime("last measured at {time}", ended)}; nothing since`,
      old: true,
    };
  }
  if (measured.length > 1) return { value: null, unit, foot: "Each machine's throughput is on the Performance page.", old: false };
  const reason = [panel.state_label, panel.reason].filter(Boolean).join(". ").replace(/\.\./g, ".");
  return { value: null, unit, foot: reason || named, old: false };
}

const controllers = (ledger: ClusterCapacityLedger) => ledger.nodes.filter(isController).length;
const workers = (ledger: ClusterCapacityLedger) => ledger.nodes.length - controllers(ledger);

/** A tile's foot when its source was not read: reading, or why it could not be. */
function unreadFoot(what: string, error: string): string {
  return error ? `${what} could not be read: ${error.replace(/\.$/, "")}.` : `Reading ${what.toLowerCase()}…`;
}

/**
 * The Throughput tile as drawn. An old figure is grey and labelled "Old" (the
 * one state vocabulary, VD-200): bright digits read as a live rate.
 */
export function ThroughputStatTile({ reading }: { reading: ThroughputReading }) {
  const tile = (
    <StatTile
      foot={reading.foot}
      label="Throughput"
      side={reading.old ? <span className="cf-tile-old">Old</span> : undefined}
      unit={reading.unit}
      value={reading.value === null ? null : throughputFigure(reading.value)}
    />
  );
  return reading.old ? <div className="cf-tile-wrap--old">{tile}</div> : tile;
}

/** The Throughput tile, read from the Performance tab's own range layer. */
function ThroughputTile({ reloadKey }: { reloadKey: number }) {
  const range = useDashboardRange("15m", true, reloadKey);
  return <ThroughputStatTile reading={throughputReading(range.data, range.error, range.loading)} />;
}

interface Props {
  ledger: ClusterCapacityLedger;
  /** False until the capacity ledger has been read once: the machine count is then Not read, never 0. */
  ledgerRead?: boolean;
  /** Why the capacity ledger could not be read, or "". */
  ledgerError?: string;
  summary: FleetSummary | null;
  /** Why the fleet summary could not be read, or "": the deployment count is then Not read. */
  summaryError?: string;
  /** Why the deployed agents could not be read, or "" (ACC-080): never counted as none. */
  agentsError?: string;
  session: Session;
  mode: ClusterMode;
  onNodeAction: (action: string, nodeId: string) => void;
  onRecheck: (nodeId: string) => Promise<RecheckResult>;
  onWorkerTelemetry: (action: "install", nodeId: string) => void;
  /** The deployed cluster agents, counted in the Deployments tile as Deployments lists them. */
  agents?: AgentDeployment[];
  /** Bumped on every page reload, so the Throughput tile reads again with the page. */
  reloadKey: number;
}

export function ClusterFleetTab({
  ledger,
  ledgerRead = true,
  ledgerError = "",
  summary,
  summaryError = "",
  agentsError = "",
  session,
  mode,
  onNodeAction,
  onRecheck,
  onWorkerTelemetry,
  agents = [],
  reloadKey,
}: Props) {
  const [expanded, setExpanded] = useState("");
  const { cluster } = ledger;
  const fleetNodes = summary?.enrolled_nodes ?? [];
  const services = summary?.runtime.services ?? [];
  const pooled = summary?.pooled_deployments ?? [];
  const appGroups = summary?.runtime.app_groups ?? [];
  const controllerArchitecture = summary?.architecture?.controller;
  const administrator = session.user.role === "administrator";
  // VD-125. The controller's own cluster link, read live by the backend. It has
  // no `enrolled_nodes` record (it is the Swarm manager, not a member), so its
  // link rides on `summary.controller` instead and is handed to the head's card.
  const controllerInterface =
    summary?.controller.placement?.inventory?.cluster_interface
    ?? summary?.controller.inventory?.cluster_interface;

  // The live MEASURED memory per machine (ACC-120), from the same telemetry
  // Home and the machine-detail tiles read. A machine with no fresh reading has none.
  const memoryNodes: FleetMemoryNode[] = ledger.nodes.map((node) => {
    const head = isController(node);
    return {
      key: node.node_id || node.name,
      telemetryId: head ? undefined : resolveFleetNode(node, fleetNodes)?.id,
      isController: head,
    };
  });
  const liveMemory = useFleetLiveMemory(memoryNodes);

  // Summed from the SAME per-node derivation each card renders, over only the
  // machines that can take work and were measured, naming every machine left
  // out (ACC-091, ACC-120), so the tile never disagrees with the cards.
  const headline = fleetMemoryHeadline(
    ledger.nodes,
    (node) => liveMemory.readings[node.node_id || node.name],
  );
  const { total, used, free } = headline;
  const coverage = headline.countedNodes === headline.totalNodes
    ? "across the fleet"
    : `across ${headline.countedNodes} of ${headline.totalNodes} machines`;
  const [freeValue, freeUnit] = splitQuantity(formatBytes(free));
  // Counted exactly as the Deployments tab lists them (ACC-092).
  const listed = listedDeployments({ services, pooled, appGroups, agents });
  const apps = listed.services.length + listed.appGroups.length;
  // LESSONS 8 (review S1): what was not read is said, never counted as none.
  const deploymentFoot = summary ? [
    plural(listed.pooled.length, "model"),
    plural(apps, "app"),
    agentsError ? "agents not read" : plural(listed.agents.length, "agent"),
  ].join(" · ") : unreadFoot("The deployments", summaryError);
  const machinesFoot = ledgerRead ? [
    `${cluster.capacity.gpu.present_nodes} with a GPU`,
    [controllers(ledger) ? plural(controllers(ledger), "controller") : "", workers(ledger) ? plural(workers(ledger), "worker") : ""].filter(Boolean).join(", "),
  ].filter(Boolean).join(" · ") : unreadFoot("The machines", ledgerError);
  const awaitingJoin = ledger.nodes.filter((node) => node.state === "not-joined").length;

  return (
    <div className="cf-tab">
      <section aria-label="Fleet at a glance" className="cf-tiles">
        <StatTile
          foot={(
            <>
              {!ledgerRead ? unreadFoot("The machines", ledgerError) : headline.countedNodes
                ? <MeterBar fraction={total > 0 ? used / total : null} label={`of ${formatBytes(total)} ${coverage}`} />
                : "Free memory is not reported for any machine that can take work"}
              {headline.leftOut.length > 0 && (
                <span className="cf-tile-note" role="note">Not counted: {headline.leftOut.join(", ")}.</span>
              )}
            </>
          )}
          label="Free memory"
          unit={headline.countedNodes ? `${freeUnit} free` : undefined}
          value={headline.countedNodes ? freeValue : null}
        />
        <StatTile
          foot={machinesFoot}
          label="Machines"
          unit="enrolled"
          value={ledgerRead ? String(cluster.node_count) : null}
        />
        <StatTile
          foot={deploymentFoot}
          label="Deployments"
          // Unread agents are not none: the count is a floor, and says so (LESSONS 8).
          unit={agentsError ? "deployed, agents not counted" : "deployed"}
          value={summary ? String(listed.total) : null}
        />
        <ThroughputTile reloadKey={reloadKey} />
      </section>

      <section aria-labelledby="cluster-machines-title" className="cf-tab">
        <h2 className="cf-machines-title" id="cluster-machines-title">
          Machines
          <span>
            {ledgerRead ? ` · ${cluster.node_count} enrolled` : " · not read"}
            {awaitingJoin > 0 && ` · ${awaitingJoin} awaiting join`}
          </span>
        </h2>
        {ledger.nodes.length ? (
          <div className="cf-grid">
            {ledger.nodes.map((node) => {
              const key = node.node_id || node.name;
              return (
                <MachineCard
                  key={key}
                  node={node}
                  reading={liveMemory.readings[key]}
                  architecture={architectureForNode(node, fleetNodes, controllerArchitecture)}
                  fleetNode={resolveFleetNode(node, fleetNodes)}
                  controllerInterface={isController(node) ? controllerInterface : undefined}
                  workloads={workloadsForNode(node, pooled)}
                  expanded={expanded === key}
                  mode={mode}
                  administrator={administrator}
                  onToggle={() => setExpanded((current) => (current === key ? "" : key))}
                  onNodeAction={onNodeAction}
                  onRecheck={onRecheck}
                  onWorkerTelemetry={onWorkerTelemetry}
                />
              );
            })}
          </div>
        ) : (
          <div className="ui-card">
            <p className="cf-empty">{ledgerRead ? "No machine capacity has been reported yet." : unreadFoot("The machines", ledgerError)}</p>
          </div>
        )}
      </section>
    </div>
  );
}
