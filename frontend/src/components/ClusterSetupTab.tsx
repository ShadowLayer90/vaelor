import { useCallback } from "react";
import { RecheckOutcome, RecheckTrigger, useRecheck, type RecheckResult } from "./RecheckButton";
import { Button, Notice } from "./ui";
import type { StatusTone } from "./ui";
import { StatusPill } from "./StatusPill";
import { AddMachineFlow } from "./AddMachineFlow";
import { HostSettingsPanel } from "./HostSettingsPanel";
import { ClusterCard, IconTile, KeyValues } from "./ClusterPrimitives";
import { formatMemory } from "./fleetTypes";
import { timeAgo } from "../lib/format";
import type { ClusterState } from "../lib/clusterState";
import type { Session } from "../types";
import type { FleetNode, FleetSummary } from "./fleetTypes";
import type { ClusterMode } from "../lib/clusterMode";
import "../styles/cluster-setup.css";

/**
 * The Setup tab (VD-200, the ClusterSetup and ClusterSetupMachines boards): set
 * up and *grow* the cluster. The left column is this node (Head controller)
 * and the inline Add a machine flow; the right column is the machines that
 * enrolled but have not joined, the workers this controller removed, and the
 * worker requirements. Below them sit the machine settings (GPU memory pool,
 * cluster link) and, in Advanced, the Cluster settings card.
 *
 * The awaiting-join list holds only nodes with no `swarm_node_id`. The
 * capacity ledger lists every enrolled node, so such a machine also has a
 * Fleet card ("Not joined"), but only this list offers its reviewed join plan,
 * Recheck, and removal.
 *
 * It stays presentational: FleetCenter owns the state and every mutation, and
 * the action gates (whether a worker can be enrolled, and why not) are computed
 * there from the same cluster state and passed in, so a control here can never
 * outrun the sentence beside it (VD-009a).
 */

// ACC-093. Where each cluster-wide setting lives today, read off the shipped
// screens rather than the roadmap (LESSONS 10). A row says "not available"
// only where the product really has no control: owner-defined machine labels
// (the constraint fields match labels already on a machine, and Vaelor's own
// labels are refused) and a cluster-wide placement default (placement is
// chosen per deploy). Backups are described by what the archive holds, which
// does not include the cluster store of enrolled machines and deployments.
export const SETTINGS_ROWS: ReadonlyArray<{ label: string; note: string; unavailable?: boolean }> = [
  {
    label: "API keys",
    note: "Cluster › Deployments › Models, beside each served endpoint. Keys for outside tools: Settings › Connections.",
  },
  {
    label: "Alert channels",
    note: "Cluster › Activity: email, or a webhook such as Slack or Discord.",
  },
  {
    label: "Node labels & constraints",
    note: "Your own machine labels are not available yet. A deploy can require a label a machine already has.",
    unavailable: true,
  },
  {
    label: "Default placement policy",
    note: "Not available yet. Placement is chosen each time you deploy an app.",
    unavailable: true,
  },
  {
    label: "Backup & restore",
    note: "Settings › Recovery. Enrolled machines and cluster deployments are not in the backup yet.",
  },
  {
    label: "Access (RBAC)",
    note: "Settings › Accounts: Viewer, Operator, or Administrator.",
  },
];

/**
 * ACC-094. What the awaiting-join card says about reaching a machine, from the
 * last reading the controller took. A Recheck that could not connect records
 * `reachable: false` with a plain reason and `checked_at` (epoch seconds) while
 * keeping the rest of the last good inventory, so the pill reads the check
 * itself, never the enrolment state: a failed check is never green, and a
 * record with no reading says so rather than looking healthy. The backend's
 * reason is already a whole sentence ("Vaelor could not reach this machine
 * over SSH at its last check: ..."), so it is shown as written, followed by
 * when that check ran.
 */
export function reachabilityAtLastCheck(
  inventory: FleetNode["inventory"] & { unreachable_reason?: string; checked_at?: number | null },
  now = Date.now(),
): { tone: StatusTone; label: string; detail: string } {
  if (inventory.reachable === true) return { tone: "success", label: "Reachable", detail: "" };
  if (inventory.reachable !== false) return { tone: "neutral", label: "Not reported", detail: "" };
  const checkedAt = typeof inventory.checked_at === "number" && inventory.checked_at > 0
    ? inventory.checked_at * 1000
    : null;
  const ago = checkedAt === null ? "" : timeAgo(checkedAt, now);
  const reason = (inventory.unreachable_reason ?? "").trim();
  let detail: string;
  if (!reason) {
    detail = ago
      ? `The last check, ${ago}, could not reach this machine.`
      : "The last check could not reach this machine.";
  } else {
    const sentence = /[.!?]$/.test(reason) ? reason : `${reason}.`;
    detail = ago ? `${sentence} Checked ${ago}.` : sentence;
  }
  return { tone: "warning", label: "Not reachable", detail };
}

/**
 * The Head controller pill. Green only for an active controller; a node that
 * is simply not set up yet is grey (the ClusterStates board), and anything
 * degraded is amber.
 */
export function controllerPill(cluster: ClusterState): StatusTone {
  if (cluster.tone === "healthy") return "success";
  if (cluster.tone === "neutral" || cluster.id === "not-initialized") return "neutral";
  return "warning";
}

interface Props {
  session: Session;
  summary: FleetSummary | null;
  cluster: ClusterState;
  mode: ClusterMode;
  onReviewPlan: (action: string, nodeId?: string, payload?: Record<string, unknown>) => void;
  onInitialize: () => void;
  /** Re-read one node's inventory (for a machine awaiting join). */
  onRecheck: (nodeId: string) => Promise<RecheckResult>;
  /** Reload the fleet after a worker joins. */
  onRefresh: () => void;
  setNotice: (message: string) => void;
  /** Whether enrollment can run now, and why not — computed by FleetCenter. */
  addMachineActionable: boolean;
  addMachineDisabledReason?: string;
}

function AwaitingMachine({
  node,
  isAdministrator,
  onReviewPlan,
  onRecheck,
}: {
  node: FleetNode;
  isAdministrator: boolean;
  onReviewPlan: Props["onReviewPlan"];
  onRecheck: Props["onRecheck"];
}) {
  const recheck = useRecheck(useCallback(() => onRecheck(node.id), [onRecheck, node.id]));
  const mismatched = node.architecture && !node.architecture.matches_controller;
  const reach = reachabilityAtLastCheck(node.inventory);
  const headingId = `cl-awaiting-${node.id}`;
  return (
    <article aria-labelledby={headingId} className="cl-setup__machine">
      <div className="cl-setup__machine-head">
        <IconTile name="cpu" />
        <div className="cl-setup__machine-name">
          <h3 id={headingId}>{node.name}</h3>
          <span className="cl-setup__mono">{node.host}:{node.port}</span>
        </div>
        <StatusPill label={reach.label} tone={reach.tone} />
      </div>
      {reach.detail && (
        <p className="cl-setup__reach cl-warn-text" role="status">{reach.detail}</p>
      )}
      {mismatched && (
        <p className="cl-setup__mismatch cl-warn-text" role="status">
          {/* A confirmed mismatch says it is going; an architecture this
              appliance could not read says only that, because unknown is not
              mismatch (VD-033). */}
          {node.architecture!.removable
            ? node.architecture!.removal_reason
            : node.architecture!.reason}
        </p>
      )}
      <KeyValues
        label={`${node.name} hardware`}
        items={[
          { label: "OS", value: node.inventory.os || "Not reported", mono: false },
          { label: "CPU", value: `${node.inventory.cpu_count ?? "?"} cores · ${node.inventory.architecture || "unknown"}`, mono: false },
          { label: "Memory", value: formatMemory(node.inventory.memory_bytes), mono: false },
          { label: "Docker", value: node.inventory.docker ? "Ready" : "Install at join", mono: false },
        ]}
      />
      <div className="cl-actions">
        <Button onClick={() => onReviewPlan("join-node", node.id)}>Review join plan</Button>
        {/* LESSONS 19 / VD-189: the Recheck route is administrator-only;
            anyone else is told why, as on the Fleet card, rather than offered
            a press that answers 403. */}
        {isAdministrator ? (
          <RecheckTrigger busy={recheck.busy} run={recheck.run} />
        ) : (
          <span className="cl-meta">Only an administrator can Recheck this machine.</span>
        )}
        <span className="cl-spacer" />
        {node.architecture?.removable && isAdministrator && (
          <Button className="cl-danger-outline" onClick={() => onReviewPlan("evict-mismatched", node.id)}>
            Review drain and removal
          </Button>
        )}
        <Button className="cl-danger-outline" onClick={() => onReviewPlan("remove-node", node.id)}>
          Remove worker
        </Button>
      </div>
      <RecheckOutcome result={recheck.result} />
    </article>
  );
}

export function ClusterSetupTab({
  session,
  summary,
  cluster,
  mode,
  onReviewPlan,
  onInitialize,
  onRecheck,
  onRefresh,
  setNotice,
  addMachineActionable,
  addMachineDisabledReason,
}: Props) {
  // Split rather than filtered on "does not match": the conflict list also
  // holds nodes whose architecture could not be read, and those are reported
  // and left alone. Unknown is not mismatch (VD-033).
  const conflicts = summary?.architecture?.conflicts ?? [];
  const removableConflicts = conflicts.filter((conflict) => conflict.removable);
  const unreadableConflicts = conflicts.filter((conflict) => !conflict.removable);
  const removals = summary?.architecture?.removals ?? [];
  const isAdministrator = session.user.role === "administrator";
  // Enrolled but not yet in the swarm (no `swarm_node_id`). The Fleet tab
  // draws them too, as "Not joined" cards from the capacity ledger, but only
  // this list offers the join plan for them.
  const awaitingJoin = (summary?.enrolled_nodes ?? []).filter(
    (node) => !node.labels?.swarm_node_id && node.runtime?.availability !== "drain",
  );
  const requirements = summary?.requirements;
  const requirementRows: Array<{ label: string; value: string }> = [
    { label: "Network", value: requirements?.network || "Not reported" },
    { label: "Ports", value: requirements?.ports?.length ? requirements.ports.join(" · ") : "Not reported" },
  ];
  if (summary?.architecture) {
    requirementRows.push({
      label: "Processor architecture",
      value: [summary.architecture.label ? `${summary.architecture.label}.` : "", summary.architecture.note].filter(Boolean).join(" "),
    });
  }
  if (requirements?.container_runtime) {
    requirementRows.push({ label: "Container runtime", value: requirements.container_runtime });
  }

  const clusterSettings = mode === "advanced" ? (
    <ClusterCard
      actions={<StatusPill label="Advanced only" tone="info" />}
      description="Where each cluster-wide setting lives today"
      flush
      title="Cluster settings"
    >
      <ul className="cl-rows cl-setup__settings">
        {SETTINGS_ROWS.map((row) => (
          <li key={row.label}>
            <span className="cl-rows__text">
              <strong>{row.label}</strong>
              <span>{row.note}</span>
            </span>
            {row.unavailable && <StatusPill label="Not available" tone="neutral" />}
          </li>
        ))}
      </ul>
    </ClusterCard>
  ) : null;

  return (
    <div className="cl-stack cl-setup">
      {!!removableConflicts.length && (
        <Notice severity="warning">
          <span className="cl-notice-row">
            <span>
              {removableConflicts.length === 1
                ? `${removableConflicts[0].name || removableConflicts[0].host} is ${removableConflicts[0].label}; this controller is ${summary?.architecture?.label}. Work placed on it cannot start, so it is drained and removed from the fleet.`
                : `${removableConflicts.length} enrolled workers do not share this controller's processor architecture (${summary?.architecture?.label}). Work placed on them cannot start, so they are drained and removed from the fleet.`}
            </span>
            {isAdministrator && (
              <span className="cl-notice-row__actions">
                <Button className="cl-danger-outline" onClick={() => onReviewPlan("evict-mismatched")}>
                  Review drain and removal
                </Button>
              </span>
            )}
          </span>
        </Notice>
      )}
      {!!unreadableConflicts.length && (
        <Notice severity="info">
          {/* Reported, never removed. A probe that failed is not evidence of a
              mismatch, and treating it as one would destroy an enrollment on
              the strength of a reading nobody took. */}
          {unreadableConflicts.length === 1
            ? "1 enrolled worker reports an architecture this controller does not recognise."
            : `${unreadableConflicts.length} enrolled workers report an architecture this controller does not recognise.`}
          {" "}
          {unreadableConflicts.length === 1 ? "It is" : "They are"} left enrolled and unchanged: an
          architecture that could not be read is not a mismatch. Recheck the worker, or remove it
          yourself if it does not belong.
        </Notice>
      )}

      <div className="cl-setup__columns">
        <div className="cl-stack">
          <ClusterCard
            actions={<StatusPill label={cluster.label} tone={controllerPill(cluster)} />}
            description="This Vaelor node"
            icon="cluster"
            iconAccent
            title="Head controller"
          >
            {/* One statement of what is actually true, instead of describing a
                working Swarm manager beside a pill that says otherwise. */}
            <p className="cl-setup__lead">{cluster.summary}</p>
            {cluster.unavailable && <p className="cl-warn-text">{cluster.unavailable}</p>}
            <KeyValues
              label="This controller"
              items={[
                // LESSONS 8: an unread summary is Not read, never "0 enrolled".
                { label: "Engine", value: !summary ? "Not read" : summary.runtime.available ? "Docker Swarm" : "Not available" },
                { label: "Workers", value: summary ? `${summary.enrolled_nodes.length} enrolled` : "Not read" },
                { label: "Services", value: summary ? `${summary.runtime.services?.length ?? 0} managed` : "Not read" },
              ]}
            />
            {cluster.nextStep && (
              isAdministrator && cluster.id === "not-initialized"
                ? <div className="cl-actions"><Button variant="primary" onClick={onInitialize}>Review controller setup</Button></div>
                : <p className="cl-meta">{cluster.nextStep}</p>
            )}
          </ClusterCard>

          {isAdministrator && (
            <AddMachineFlow
              session={session}
              mode={mode}
              actionable={addMachineActionable}
              disabledReason={addMachineDisabledReason}
              onEnrolled={onRefresh}
              onReviewJoin={(nodeId) => onReviewPlan("join-node", nodeId)}
              addressHint={summary?.enrollment?.address_hint}
              setNotice={setNotice}
            />
          )}
        </div>

        <div className="cl-stack">
          {!!awaitingJoin.length && (
            <ClusterCard
              actions={<StatusPill label={`${awaitingJoin.length} waiting`} tone="info" />}
              description="Enrolled, not yet joined · approve each join plan to bring it in"
              flush
              title="Machines awaiting join"
            >
              {awaitingJoin.map((node) => (
                <AwaitingMachine
                  isAdministrator={isAdministrator}
                  key={node.id}
                  node={node}
                  onRecheck={onRecheck}
                  onReviewPlan={onReviewPlan}
                />
              ))}
            </ClusterCard>
          )}

          {!!removals.length && (
            // A removed node is gone from the Fleet grid, so without this the
            // cluster would simply stop showing it. Silent removal is its own
            // kind of lie (VD-033).
            <ClusterCard description="Architecture enforcement" flush title="Workers this controller removed">
              <ul className="cl-rows cl-setup__removals">
                {removals.map((removal) => (
                  <li key={`${removal.node_id}-${removal.at}`}>
                    <span className="cl-rows__text">
                      <strong>{removal.name || removal.host}</strong>
                      <span>{removal.reason}</span>
                      <span>
                        {removal.drained ? "Drained, then unenrolled" : "Unenrolled"}
                        {removal.forced ? " (forced: the worker did not leave cleanly)" : ""}
                        {" · its stored SSH credential was deleted"}
                      </span>
                    </span>
                    <StatusPill label="Removed" tone="neutral" />
                  </li>
                ))}
              </ul>
            </ClusterCard>
          )}

          <ClusterCard title="Worker requirements">
            <dl className="cl-facts">
              {requirementRows.map((row) => (
                <div className="cl-setup__fact" key={row.label}>
                  <dt>{row.label}</dt>
                  <dd>{row.value}</dd>
                </div>
              ))}
            </dl>
          </ClusterCard>
        </div>
      </div>

      {/* VD-161 / VD-162: each machine's GPU memory pool and the cluster
          link. It reads and changes its own settings; this tab only places
          it, and hands it the Advanced Cluster settings card to sit beside
          the cluster link. */}
      <HostSettingsPanel besideLink={clusterSettings} session={session} />
    </div>
  );
}
