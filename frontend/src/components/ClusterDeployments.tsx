import type { ReactNode } from "react";
import { Button, EmptyState, Notice, type NoticeSeverity } from "./ui";
import { ClusterServiceManager, type ClusterServiceSummary } from "./ClusterServiceManager";
import { ClusterAppGroupManager } from "./ClusterAppGroupManager";
import { ClusterCard, Tag } from "./ClusterPrimitives";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import "../styles/cluster-deployments.css";
import { type RowStatus, statusForAppState } from "./clusterAppStatus";
import {
  LAN_EXPOSED_PILL_DESCRIPTION,
  LAN_EXPOSED_PILL_LABEL,
  MODE_B_BADGE_DETAIL,
  MODE_B_BADGE_LABEL,
  isLanExposedDeployment,
  isModeBDeployment,
} from "../lib/gpuServingMode";
import type { Session } from "../types";
import type { AppGroup, ClusterCapacityLedger, FleetSummary } from "./fleetTypes";
import type { AgentDeployment } from "../lib/clusterAgents";
import { listedDeployments, pooledStatus, refreshNote, type Replica, ROW_PILL_TONE, servingReplicas, splitFenceNotes } from "../lib/deploymentRows";
import { agentRuntimeBadge, agentRuntimeNote } from "../lib/agentRuntimeStatus";

/**
 * Every app, model and agent across the cluster as one table (VD-200, the
 * ClusterDeployments board's "Deployments" card), and where each one runs: a
 * type tag, the placement in plain words ("on Pi-01", "2 replicas", "sharded
 * across 3 machines", "replicated on 2 machines"), the address, a status
 * pill, and beneath a row the backend's own explanation as a banner across the
 * table. A phone draws each row as a block. The management actions (restart,
 * logs, refresh, rollback, backups, removal) are the exact same server-reviewed
 * ones as before — the app rows delegate to `ClusterServiceManager`'s modal, and
 * the model rows queue the pooled-removal plan.
 */

export type PooledDeployment = NonNullable<FleetSummary["pooled_deployments"]>[number];

type PlacementKind = "pin" | "spread" | "shard";
interface Placement {
  kind: PlacementKind;
  label: string;
}

/** Parses a Swarm replica string like "2/2" — or "2/2 (max 1 per node)", the
 * annotation a spread deploy's `--replicas-max-per-node 1` adds — into the
 * desired replica count (the denominator, read after the slash, not anchored to
 * end so the trailing annotation does not defeat the match). */
function desiredReplicas(replicas: string | undefined): number {
  const match = String(replicas ?? "").match(/\/\s*(\d+)/);
  return match ? Number(match[1]) : 1;
}

/** The machine name a pinned service sits on, read from the capacity ledger's
 *  per-node reservation list (which records the workloads committed to a node). */
function pinnedNodeName(
  name: string,
  ledger: ClusterCapacityLedger,
): string | undefined {
  const candidates = new Set([name, name.replace(/^vaelor-/, "")]);
  const node = ledger.nodes.find((entry) =>
    entry.reserved.from.some((workload) => candidates.has(workload)));
  return node?.name;
}

/** The machine name for a placement id, from the capacity ledger. */
function nodeName(nodeId: string, ledger: ClusterCapacityLedger): string | undefined {
  return ledger.nodes.find((entry) => entry.node_id === nodeId)?.name;
}

/**
 * Where a single Swarm service runs, in plain words. A multi-replica service is
 * "N replicas" — deliberately NOT "spread across N nodes": the `/cluster`
 * summary carries the desired replica count but no per-node task placement, and
 * Swarm can stack replicas on one node, so a node count would be a claim the
 * data cannot back. A single-replica service is pinned to the one machine the
 * capacity ledger reserves it on.
 */
export function servicePlacement(
  service: ClusterServiceSummary,
  ledger: ClusterCapacityLedger,
): Placement {
  const desired = desiredReplicas(service.replicas);
  if (desired > 1) return { kind: "spread", label: `${desired} replicas` };
  const node = pinnedNodeName(String(service.name ?? ""), ledger);
  return { kind: "pin", label: node ? `on ${node}` : "on one machine" };
}

/**
 * Where a pooled model runs: one node pins it (named from the ledger); several
 * either shard one copy across them or, under VD-129's `units.mode =
 * "replicated"`, run one whole copy on each. The mode is read from the record,
 * never inferred from the count: a replicated row must never read "sharded",
 * and a row from before the word existed (no mode) keeps the sharded wording
 * that was true of every multi-node vLLM record then.
 */
export function pooledPlacement(
  deployment: PooledDeployment,
  ledger: ClusterCapacityLedger,
): Placement {
  const count = deployment.node_ids.length;
  if (count > 1) {
    return deployment.units?.mode === "replicated"
      ? { kind: "spread", label: `replicated on ${count} machines` }
      : { kind: "shard", label: `sharded across ${count} machines` };
  }
  if (!count) return { kind: "pin", label: "not yet placed" };
  const name = nodeName(deployment.node_ids[0], ledger);
  return { kind: "pin", label: name ? `on ${name}` : "on one machine" };
}

/** The row status for an app service, from the backend `state`, through the
 *  shared app-state contract (`clusterAppStatus`) the grouped D4d row reuses. A
 *  row the backend could not enrich reads as an honest unknown, never a guessed
 *  Running/Deploying. */
function appStatus(service: ClusterServiceSummary): RowStatus {
  return statusForAppState(service.state);
}

// A pooled row's status word (`pooledStatus`) lives in lib/deploymentRows, so
// Home's AI Chat row reads the same rule without importing this table.
export { pooledStatus };

/**
 * An agent row's status, from what the agent is DOING (`agent.runtime`) through
 * the one mapping the Endpoints panel's agent card also uses - never from the
 * stored deploy row, which read "Healthy" after every key was revoked or its
 * model was removed (ACC-071/080). The badge's tones fold onto the row's dot
 * the way the model rows fold theirs: an in-progress state is a warning.
 */
function agentStatus(agent: AgentDeployment): RowStatus {
  const badge = agentRuntimeBadge(agent.runtime, agent.state);
  const tone = badge.tone === "success"
    ? "ok"
    : badge.tone === "danger"
      ? "danger"
      : badge.tone === "neutral"
        ? "neutral"
        : "warn";
  return { label: badge.label, tone };
}

/** One unit a failed deploy's rollback could not stop, as the row names it. */
function unstoppedLabel(entry: { node_id?: string; node?: string; unit?: string }): string {
  const node = entry.node || entry.node_id || "";
  const unit = entry.unit || "";
  if (node && unit) return `${node} · ${unit}`;
  return node || unit || "a unit the record does not name";
}

/** The Deployments type filter. "Models" are the LLM/pooled deployments; "Apps"
 *  are the cluster services; "Agents" arrive with the agent-deployment phase. */
export type DeploymentFilter = "all" | "models" | "apps" | "agents";

/**
 * The filters this table is drawn under. "Models" draws the endpoints and the
 * model library instead (ClusterDeploymentsTab), so it never reaches here.
 */
export type TableFilter = Exclude<DeploymentFilter, "models">;

interface Props {
  services: ClusterServiceSummary[];
  pooled: PooledDeployment[];
  ledger: ClusterCapacityLedger;
  session: Session;
  onServiceReview: (action: string, payload: Record<string, unknown>) => Promise<void>;
  onNotice: (message: string, refused?: boolean) => void;
  /** Which types to show. Defaults to everything. */
  filter?: TableFilter;
  /**
   * D4d. The researched multi-service apps, each folded to ONE row (aggregate
   * tone + reason, with the per-service breakdown in Manage). Additive: their
   * member services are removed from the per-service `services` list below so a
   * grouped app is not also drawn as N loose rows, while a catalog service —
   * which carries no app-group label and appears in no group — is UNCHANGED.
   */
  appGroups?: AppGroup[];
  /**
   * F6c-1. The deployed cluster agents, shown under the Agents filter. Threaded
   * from FleetCenter's central fleet read alongside the models and apps, so an
   * agent list reloads on the same refresh every other deployment change uses.
   */
  agents?: AgentDeployment[];
  /**
   * Why the deployed agents could not be read, or "". An unreadable list is
   * said as such under the Agents (and All) filter, never as "No agents are
   * running yet" (ACC-080).
   */
  agentsError?: string;
}

const NOTICE_FOR_TONE: Record<RowStatus["tone"], NoticeSeverity> = {
  ok: "info",
  warn: "warning",
  danger: "danger",
  neutral: "info",
};

/** A row's status as the boards' outline pill. */
function RowPill({ label, tone }: RowStatus) {
  return <StatusPill label={label} tone={ROW_PILL_TONE[tone]} />;
}

/** The table's column count, which every explanation row spans. */
const COLUMN_COUNT = 6;

/**
 * An explanation row beneath a deployment: the board's banner across the
 * whole table, in the tone of what it explains. A standing state, so it is
 * announced politely, never as an alert.
 */
function SubRow({ children, severity }: { children: ReactNode; severity: NoticeSeverity }) {
  return (
    <tr className="cl-dep-sub" role="row">
      <td colSpan={COLUMN_COUNT} role="cell">
        <Notice severity={severity} standing>{children}</Notice>
      </td>
    </tr>
  );
}

/**
 * A split's fence notes beneath its row (ACC-187): the clean-up the unload
 * could not finish, as a warning, and the firewall check an older split has
 * not had yet, as information. Both sentences are the backend's, verbatim.
 */
function FenceNotes({ deployment }: { deployment: PooledDeployment }) {
  const { cleanup, fence } = splitFenceNotes(deployment);
  return (
    <>
      {cleanup && <SubRow severity="warning">{cleanup}</SubRow>}
      {fence && <SubRow severity="info">{fence}</SubRow>}
    </>
  );
}

/**
 * What the post-upgrade refresh could not do (W4-D1), as a warning: the row
 * is still on the previous release's units until the step it names is taken.
 * The backend's sentence, verbatim.
 */
function RefreshNote({ deployment }: { deployment: PooledDeployment }) {
  const note = refreshNote(deployment);
  return note ? <SubRow severity="warning">{note}</SubRow> : null;
}

/**
 * What a serving split's last failed health check found and how long the watch
 * waits (W4-D6), as a warning beneath the row: the backend's sentence verbatim.
 */
function CheckingNote({ deployment }: { deployment: PooledDeployment }) {
  const note = deployment.state === "healthy" ? (deployment.checking_note ?? "").trim() : "";
  return note ? <SubRow severity="warning">{note}</SubRow> : null;
}

/**
 * What a degraded record says beneath its row (B1): the backend's sentence,
 * and each machine the deployment lost, with why.
 */
function DegradedNote({ deployment }: { deployment: PooledDeployment }) {
  const reason = deployment.state === "healthy" ? (deployment.degraded_reason ?? "").trim() : "";
  if (!reason) return null;
  const lost = deployment.units?.lost_nodes ?? [];
  return (
    <SubRow severity="warning">
      {reason}
      {lost.length > 0 && (
        <ul>
          {lost.map((entry) => (
            <li key={entry.node_id}>
              <strong>{entry.name || "A removed machine"}</strong>
              {entry.reason ? ` - ${entry.reason}` : ""}
            </li>
          ))}
        </ul>
      )}
    </SubRow>
  );
}

/**
 * What a failed record says beneath its row: the backend's own sentence, and
 * the units its rollback could not stop. Rendered from the record whenever it
 * carries either, so the row never hides a fact the backend wrote for it.
 */
function FailureNote({ deployment }: { deployment: PooledDeployment }) {
  const failure = deployment.units?.failure ?? "";
  const unstopped = deployment.units?.unstopped ?? [];
  if (!failure && !unstopped.length) return null;
  return (
    <SubRow severity="danger">
      {failure}
      {unstopped.length > 0 && (
        <>
          {failure ? " " : ""}The rollback could not stop these, so they may still hold that machine's GPU:
          <ul>
            {unstopped.map((entry, index) => (
              <li key={`${entry.node_id ?? ""}-${entry.unit ?? ""}-${index}`}>
                <code>{unstoppedLabel(entry)}</code>
                {entry.error ? ` — ${entry.error}` : ""}
              </li>
            ))}
          </ul>
        </>
      )}
    </SubRow>
  );
}

/**
 * What the probe established about a replica that is not serving: "not
 * answering" when the machine answered and its replica was unhealthy,
 * "unreachable" when nothing at the address answered at all — a node that
 * left the cluster, was unplugged, or is rebooting (`units.replicas[].reachable`,
 * VD-129). A row written before the field existed reads as reachable, which
 * is what its probe established.
 */
function replicaStateWord(replica: Replica): string {
  return replica.reachable === false ? "unreachable" : "not answering";
}

/**
 * The lead sentence of the note, from what the row will DO. With some replica
 * alive the endpoint answers and requests go to the others. With none alive
 * nothing behind the endpoint answers, and the record stays healthy only for
 * the four consecutive checks (about two minutes) the mode reconcile allows
 * before it tears the cluster down and returns AI Chat to llama.cpp — so the
 * note says that, rather than promising other replicas that are not there.
 */
const SOME_REPLICAS_DOWN = "Not serving, so requests go to the other replicas:";
const NO_REPLICA_SERVING =
  "No replica is serving, so nothing behind the endpoint answers. If none answers for "
  + "four consecutive checks (about two minutes) the cluster is torn down and AI Chat "
  + "returns to llama.cpp:";

/**
 * The units a silent replica is: its server, and on a worker its gate too
 * (`units.replicas[].gate`, VD-129) — the probe goes through the gate, so
 * either may be the one that is down, and the note names both rather than
 * pointing at the replica when it was the door.
 */
function replicaUnits(replica: Replica): string {
  return replica.gate ? `${replica.unit}, ${replica.gate}` : replica.unit;
}

/**
 * Which replicas of a healthy replicated row are not serving, named by
 * machine (from the ledger, else the address the balancer reaches them on) and
 * unit(s), each with what the probe established, so "1 of 2 serving" says
 * WHICH one is down and whether the machine is there without a log. Some
 * replicas still answering is a warning; none is danger, as the pill says.
 */
function ReplicaNote({
  deployment,
  ledger,
}: {
  deployment: PooledDeployment;
  ledger: ClusterCapacityLedger;
}) {
  const replicas = servingReplicas(deployment) ?? [];
  const down = replicas.filter((replica) => replica.alive !== true);
  if (!down.length) return null;
  const anyAlive = down.length < replicas.length;
  return (
    <SubRow severity={anyAlive ? "warning" : "danger"}>
      {anyAlive ? SOME_REPLICAS_DOWN : NO_REPLICA_SERVING}
      <ul>
        {down.map((replica) => (
          <li key={`${replica.node_id}-${replica.unit}`}>
            <code>{nodeName(replica.node_id, ledger) ?? replica.address} · {replicaUnits(replica)}</code>
            {` — ${replicaStateWord(replica)}`}
          </li>
        ))}
      </ul>
    </SubRow>
  );
}

/** The thinking pill's words, or null when the model answers without thinking. */
function thinkingLabel(deployment: { engine?: string; thinking?: boolean | null; thinking_switch?: boolean }): string | null {
  if (deployment.engine !== "vllm" || !deployment.thinking_switch) return null;
  if (deployment.thinking === true) return "Thinks before answering";
  if (deployment.thinking === null || deployment.thinking === undefined) {
    return "Thinks until its next Load";
  }
  return null;
}

/**
 * The label on a scale-to-zero deployment's pill (G3b): how many idle minutes
 * before it unloads itself, rounded from the seconds the read view carries.
 * Floored at 2 minutes to mirror the backend's MINIMUM_IDLE_TIMEOUT_SECONDS
 * (gpu_idle_watch), so the pill never advertises less than the enforced window.
 */
function autoUnloadLabel(idleSeconds: number): string {
  const minutes = Math.max(2, Math.round(idleSeconds / 60));
  return `Auto-unload after ${minutes} min idle`;
}

/**
 * The one-line reason the backend wrote beneath an app row that is not simply
 * running — a rolled-back update's message, a failed update's, or the honest
 * "restore from a backup, this app is not migrated" for a stateful app whose
 * data node is gone — in the tone of the row's own status. A healthy row
 * carries no reason and shows no note.
 */
function ReasonNote({ reason, status }: { reason?: string; status: RowStatus }) {
  const text = (reason ?? "").trim();
  return text ? <SubRow severity={NOTICE_FOR_TONE[status.tone]}>{text}</SubRow> : null;
}

/** One deployment's row: the five cells the board draws, and its actions. */
function MainRow({
  actions,
  address,
  addressTitle,
  name,
  placement,
  status,
  type,
}: {
  actions?: ReactNode;
  address: ReactNode;
  /** The whole address on hover, for a cell the layout cuts to one line. */
  addressTitle?: string;
  name: ReactNode;
  placement: string;
  status: RowStatus;
  type: string;
}) {
  return (
    <tr role="row">
      <td className="cl-dep-cell--name" role="cell">{name}</td>
      <td role="cell"><Tag>{type}</Tag></td>
      <td className="cl-dep-where" role="cell">{placement}</td>
      {/* One line, cut with an ellipsis where the column is narrower than the
          address; the cell's title keeps the whole address. */}
      <td className="cl-dep-address" role="cell" title={addressTitle || undefined}>
        <span className="cl-dep-address__text">{address}</span>
      </td>
      <td role="cell"><RowPill {...status} /></td>
      <td className="cl-dep-cell--actions" role="cell"><div className="cl-dep-actions">{actions}</div></td>
    </tr>
  );
}

const EMPTY_TITLE: Record<TableFilter, string> = {
  all: "No apps, models, or agents are running across the fleet yet.",
  apps: "No apps are running yet.",
  agents: "No agents are running yet.",
};

const EMPTY_TEXT: Record<TableFilter, string> = {
  all: "Serve a model or deploy an app from the header.",
  apps: "Deploy an app from the header.",
  agents: "Deploy an agent with the button above.",
};

export function ClusterDeployments({
  services,
  pooled,
  ledger,
  session,
  onServiceReview,
  onNotice,
  filter = "all",
  appGroups = [],
  agents = [],
  agentsError = "",
}: Props) {
  // The rows listed under the filter, from the ONE derivation the Fleet hero
  // also counts with (ACC-092): a researched app's member services fold into
  // its grouped row, and every model, app and agent is one row.
  const {
    services: visibleServices,
    pooled: visiblePooled,
    appGroups: visibleGroups,
    agents: visibleAgents,
    total,
  } = listedDeployments({ services, pooled, appGroups, agents }, filter);
  // A failed agents read is said, never passed off as an empty list (ACC-080).
  const agentsUnreadable = (filter === "all" || filter === "agents") && Boolean(agentsError);
  const isAdministrator = session.user.role === "administrator";
  // Worded to the active filter: "no apps or models" would contradict the other
  // filter one click away when only that type is deployed.
  const emptyTitle = filter === "all" && agentsUnreadable
    ? "No apps or models are running across the fleet yet."
    : EMPTY_TITLE[filter];
  // Read from the deployments themselves rather than from a row's engine alone:
  // only a HEALTHY vLLM deployment the controller takes part in has entered the
  // mode switch (`gpu_pool_operations.deploy`), so only that is Mode B.
  const clustering = pooled.some(isModeBDeployment);

  return (
    <ClusterCard
      actions={<span className="cl-meta">{total} shown, and where each runs</span>}
      description="Running across the fleet"
      flush
      title="Deployments"
    >
      {/* VD-127. Mode B is a fleet-wide state, not a property of one row:
          while it holds, this controller's llama.cpp AI-Chat model is
          stopped and both AI Chat and the LLM Server answer on the cluster.
          Saying so above the list is the only place a reader meets that
          fact before they act on a row. */}
      {clustering && (
        <p className="cl-dep-mode">
          <StatusPill label={MODE_B_BADGE_LABEL} tone="info" />
          <span>{MODE_B_BADGE_DETAIL}</span>
        </p>
      )}

      {agentsUnreadable && (
        <div className="cl-dep-pad">
          <Notice severity="warning">{`The deployed agents could not be read: ${agentsError.replace(/\.$/, "")}. They may still be running.`}</Notice>
        </div>
      )}

      {total ? (
        <div className="ui-table-scroll cl-dep-scroll">
          <table className="cl-dep-table" role="table">
            <caption className="sr-only">Deployments running across the fleet</caption>
            <thead role="rowgroup">
              <tr role="row">
                <th role="columnheader" scope="col">Workload</th>
                <th role="columnheader" scope="col">Type</th>
                <th role="columnheader" scope="col">Placement</th>
                <th role="columnheader" scope="col">Address</th>
                <th role="columnheader" scope="col">Status</th>
                <th role="columnheader" scope="col"><span className="sr-only">Actions</span></th>
              </tr>
            </thead>
            {visiblePooled.map((deployment) => {
              const thinking = thinkingLabel(deployment);
              const idle = deployment.idle_timeout ?? 0;
              const exposed = isLanExposedDeployment(deployment);
              return (
                <tbody className="cl-dep-group" key={`model-${deployment.name}`} role="rowgroup">
                  <MainRow
                    address={deployment.endpoint || "—"}
                    addressTitle={deployment.endpoint}
                    name={(
                      <div className="cl-dep-name">
                        <strong>{deployment.name}</strong>
                        {/* The model, and for a cluster deployment the vLLM
                            version it runs: a deployment keeps the version it
                            was made with, so two rows can differ. */}
                        <span>
                          {deployment.vllm_version
                            ? `${deployment.model_id} · ${deployment.vllm_version}`
                            : deployment.model_id}
                        </span>
                        {(exposed || idle > 0 || thinking) && (
                          <div className="cl-tags">
                            {/* VD-127 D2, ACC-162: a worker-led split deployed
                                before its lead was put behind a keyed gate still
                                answers on the LAN with no key until it is Loaded
                                again. This is the row where an owner meets it. */}
                            {exposed && (
                              <StatusPill
                                description={LAN_EXPOSED_PILL_DESCRIPTION}
                                label={LAN_EXPOSED_PILL_LABEL}
                                tone="warning"
                              />
                            )}
                            {/* G3b: a scale-to-zero window, so an owner knows it
                                unloads itself when idle and a request wakes it. */}
                            {idle > 0 && <span className="cl-dep-chip">{autoUnloadLabel(idle)}</span>}
                            {/* Owner decision 2026-09-29: a cluster model answers
                                without thinking unless its owner chose otherwise;
                                nothing is said for the default. */}
                            {thinking && <span className="cl-dep-chip">{thinking}</span>}
                          </div>
                        )}
                      </div>
                    )}
                    placement={pooledPlacement(deployment, ledger).label}
                    status={pooledStatus(deployment)}
                    type="Model"
                    actions={(
                      <>
                        {/* G3a: a healthy vLLM deployment can be UNLOADED to
                            reclaim its GPU without removing it, and an unloaded
                            one LOADED back warm. Both go through the same
                            confirm-then-queue path as Remove; the CPU pooled
                            tier has neither. */}
                        {deployment.engine === "vllm" && deployment.state === "healthy" && (
                          <Button
                            variant="quiet"
                            onClick={() => void onServiceReview("unload-gpu", { name: deployment.name })}
                          >
                            Unload
                          </Button>
                        )}
                        {/* W4-D7: Load also starts a failed row again from its
                            record, when the backend says it can. */}
                        {deployment.engine === "vllm"
                          && (deployment.load_offered ?? deployment.state === "unloaded") && (
                          <Button
                            variant="quiet"
                            onClick={() => void onServiceReview("load-gpu", { name: deployment.name })}
                          >
                            Load
                          </Button>
                        )}
                        <Button
                          variant="quiet"
                          onClick={() => void onServiceReview(
                            // A vLLM (GPU) deployment stops through `cluster.gpu.remove`;
                            // every other pooled row is CPU distributed-llama, removed
                            // through `cluster.pooled.remove`. The engine projected onto
                            // the row is the discriminator both backends already use.
                            deployment.engine === "vllm" ? "remove-gpu" : "remove-pooled",
                            { name: deployment.name },
                          )}
                        >
                          Remove
                        </Button>
                      </>
                    )}
                  />
                  <ReplicaNote deployment={deployment} ledger={ledger} />
                  <DegradedNote deployment={deployment} />
                  <FailureNote deployment={deployment} />
                  <FenceNotes deployment={deployment} />
                  <CheckingNote deployment={deployment} />
                  <RefreshNote deployment={deployment} />
                </tbody>
              );
            })}

            {visibleGroups.map((group) => {
              const status = statusForAppState(group.state);
              const count = group.services.length;
              const services = `${count} service${count === 1 ? "" : "s"}`;
              return (
                <tbody className="cl-dep-group" key={`group-${group.app_group}`} role="rowgroup">
                  <MainRow
                    actions={(
                      <ClusterAppGroupManager
                        group={group}
                        session={session}
                        onReview={(action, payload) => onServiceReview(action, payload)}
                      />
                    )}
                    address="—"
                    name={(
                      <div className="cl-dep-name">
                        <strong>{group.app_group}</strong>
                        <span>Researched app · {services}</span>
                      </div>
                    )}
                    placement={services}
                    status={status}
                    type="App"
                  />
                  <ReasonNote reason={group.reason} status={status} />
                </tbody>
              );
            })}

            {visibleServices.map((service) => {
              const status = appStatus(service);
              return (
                <tbody className="cl-dep-group" key={`app-${service.id || service.name}`} role="rowgroup">
                  <MainRow
                    actions={(
                      <ClusterServiceManager
                        service={service}
                        session={session}
                        onNotice={onNotice}
                        onReview={(action, payload) => onServiceReview(action, payload)}
                      />
                    )}
                    address={service.replicas ? `${service.replicas} replicas` : "—"}
                    name={(
                      <div className="cl-dep-name">
                        <strong>{service.name}</strong>
                        <span>{service.image}</span>
                      </div>
                    )}
                    placement={servicePlacement(service, ledger).label}
                    status={status}
                    type="App"
                  />
                  <ReasonNote reason={service.reason} status={status} />
                </tbody>
              );
            })}

            {visibleAgents.map((agent) => {
              const model = agent.backing.model_deployment_name;
              const status = agentStatus(agent);
              // A broken agent says why under its row, the way a failed model
              // row does; a starting or paused one needs no alarm here.
              const note = status.tone === "danger" ? agentRuntimeNote(agent.runtime) : "";
              return (
                <tbody className="cl-dep-group" key={`agent-${agent.name}`} role="rowgroup">
                  <MainRow
                    actions={isAdministrator && (
                      // A cluster.* job needs administrator + CSRF, so an
                      // operator sees the agent but not the Remove control. The
                      // removal runs through the same confirm-then-queue path
                      // as the GPU remove.
                      <Button
                        variant="quiet"
                        onClick={() => void onServiceReview("remove-agent", { name: agent.name })}
                      >
                        Remove
                      </Button>
                    )}
                    address={agent.endpoint || "—"}
                    addressTitle={agent.endpoint}
                    name={(
                      <div className="cl-dep-name">
                        <strong>{agent.name}</strong>
                        {/* The backing model in the model (info) hue, so the
                            dependency reads as a link to a row of that type. An
                            agent whose backing the record did not carry says so
                            rather than inventing one. */}
                        <span>
                          {model ? <>backed by <span className="cl-dep-backing">{model}</span></> : "backing not recorded"}
                        </span>
                      </div>
                    )}
                    // A deployed agent runs on the head controller behind its
                    // inbound gate (placement.runs_on), so the placement is
                    // fixed rather than read from the capacity ledger.
                    placement="on this controller"
                    status={status}
                    type="Agent"
                  />
                  {note && <SubRow severity="danger">{note}</SubRow>}
                </tbody>
              );
            })}
          </table>
        </div>
      ) : filter === "agents" && agentsUnreadable ? null : (
        <EmptyState icon={<Icon name="grid" size={18} />} text={EMPTY_TEXT[filter]} title={emptyTitle} />
      )}
    </ClusterCard>
  );
}
