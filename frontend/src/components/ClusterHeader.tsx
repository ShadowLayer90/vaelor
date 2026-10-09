import type { ReactNode } from "react";
import { Button, LoadingLines, Notice, type StatusTone } from "./ui";
import { destinations } from "../lib/destinations";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { ClusterCard } from "./ClusterPrimitives";
import { TopbarPageActions } from "../lib/topbarSlot";
import type { ClusterState } from "../lib/clusterState";

/*
 * What every Cluster tab shares (VD-200, the ClusterHeader board): the
 * status pill and Reload in the top bar, the title row with the page's three
 * actions, and the notices stacked under it, newest concern first.
 */

export const CLUSTER_SUBTITLE = "Your machines, and what runs across them.";

/**
 * The pill's colour for each cluster state, as the ClusterHeader board draws
 * them: still reading is blue, active green, degraded amber, not yet a
 * controller grey, and no cluster engine at all red.
 */
export function clusterPillTone(cluster: Pick<ClusterState, "id" | "tone">): StatusTone {
  if (cluster.id === "ready") return "success";
  if (cluster.id === "control-unavailable") return "warning";
  // A refused or failed read is grey and says "Not read" (the owner's rule):
  // never the blue of a read still under way.
  if (cluster.id === "not-initialized" || cluster.id === "not-read") return "neutral";
  // "runtime-unavailable" is both "Checking cluster" (nothing read yet) and
  // "Cluster engine unavailable"; only the second carries the degraded tone.
  return cluster.tone === "degraded" ? "danger" : "info";
}

/**
 * The cluster's state in one word, beside Reload, in the top bar's page slot
 * (the ClusterHeader board).
 */
export function ClusterStatus({
  busy,
  cluster,
  onReload,
}: {
  busy: boolean;
  cluster: Pick<ClusterState, "id" | "label" | "tone" | "summary">;
  onReload: () => void;
}) {
  return (
    <TopbarPageActions>
      <div className="cl-status">
        <StatusPill description={cluster.summary} label={cluster.label} tone={clusterPillTone(cluster)} />
        <Button className="cl-status__reload" disabled={busy} onClick={onReload} variant="quiet">
          <Icon name="refresh" size={16} />
          <span className="cl-status__reload-label">Reload</span>
        </Button>
      </div>
    </TopbarPageActions>
  );
}

/**
 * "Cluster / Your machines, and what runs across them." with Add machine,
 * Deploy app and Serve a model. A `*DisabledReason` disables its button and
 * sits under it; Add machine is drawn only for an administrator. The
 * callbacks are props of their own (never fields of an object) so the sweep
 * inventory can follow each button to the dialog it opens.
 */
export function ClusterTitleRow({
  addMachineDisabledReason,
  deployAppDisabledReason,
  onAddMachine,
  onDeployApp,
  onServeModel,
  serveModelDisabledReason,
  showActions,
  showAddMachine,
}: {
  addMachineDisabledReason?: string;
  deployAppDisabledReason?: string;
  onAddMachine: () => void;
  onDeployApp: () => void;
  onServeModel: () => void;
  serveModelDisabledReason?: string;
  /** False before the first read: no action is offered against a cluster nobody has read. */
  showActions: boolean;
  showAddMachine: boolean;
}) {
  return (
    <div className="ui-page-header">
      {/* The PageHeader primitive's markup, written out so the page heading
          is read from the destination registry the rail reads. */}
      <div className="cl-title">
        <h1 className="ui-page-header__title">{destinations.fleet.name}</h1>
        <p className="ui-page-header__title cl-title__second">{CLUSTER_SUBTITLE}</p>
      </div>
      {showActions && (
        <div className="ui-page-header__actions">
          {showAddMachine && (
            <Button disabledReason={addMachineDisabledReason} onClick={onAddMachine}>
              <Icon name="add" size={16} />
              Add machine
            </Button>
          )}
          <Button disabledReason={deployAppDisabledReason} onClick={onDeployApp}>Deploy app</Button>
          <Button disabledReason={serveModelDisabledReason} onClick={onServeModel} variant="primary">Serve a model</Button>
        </div>
      )}
    </div>
  );
}

/** The first read, before the cluster has answered: the page frame and one card that says so. */
export function ClusterFirstLoad() {
  return (
    <ClusterCard className="cl-first-load" title="Reading cluster state…">
      <LoadingLines label="Reading cluster state…" />
    </ClusterCard>
  );
}

/** A notice with its own actions at the right end. */
export function ActionNotice({
  actions,
  children,
  severity,
}: {
  actions?: ReactNode;
  children: ReactNode;
  severity: "info" | "warning" | "danger";
}) {
  return (
    <Notice severity={severity}>
      <span className="cl-notice-row">
        <span>{children}</span>
        {actions && <span className="cl-notice-row__actions">{actions}</span>}
      </span>
    </Notice>
  );
}
