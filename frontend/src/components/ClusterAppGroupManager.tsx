import { useState } from "react";
import type { Session } from "../types";
import type { AppGroup } from "./fleetTypes";
import { ClusterDialog } from "./ClusterDialog";
import { StatusPill } from "./StatusPill";
import { Button, Notice } from "./ui";
import { statusForAppState, type RowStatus } from "./clusterAppStatus";

/**
 * The Manage affordance for a researched multi-service app (D4d), drawn as
 * the ClusterDialogsManage board's "Cluster application" dialog (VD-200).
 *
 * The grouped Deployments row folds N member services into one row; opening
 * Manage shows the per-service breakdown — each member's manifest key, honest
 * state, replicas and reason — and, for an administrator, the app-level
 * "Remove app" action. Removal goes through the SAME reviewed `/cluster/plan`
 * step every destructive cluster change uses (`onReview("remove-researched-app",
 * { app_group })`): the plan modal carries the multi-service data-loss token
 * when any stateful member has no backup, and only then is a typed
 * acknowledgement required. The per-service lifecycle controls stay on each
 * member's own `ClusterServiceManager`; this component never duplicates them.
 */
interface Props {
  group: AppGroup;
  session: Session;
  onReview: (action: string, payload: Record<string, unknown>) => Promise<void>;
}

const PILL_TONE = { ok: "success", warn: "warning", danger: "danger", neutral: "neutral" } as const;

function statePill(status: RowStatus) {
  return <StatusPill label={status.label} tone={PILL_TONE[status.tone]} />;
}

export function ClusterAppGroupManager({ group, session, onReview }: Props) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const isAdministrator = session.user.role === "administrator";
  const status = statusForAppState(group.state);
  const count = group.services.length;

  const removeApp = async () => {
    setBusy(true);
    try {
      setOpen(false);
      await onReview("remove-researched-app", { app_group: group.app_group });
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Button disabled={busy} onClick={() => setOpen(true)}>
        Manage
      </Button>

      {open && (
        <ClusterDialog
          aside={statePill(status)}
          eyebrow="Cluster application"
          footer={isAdministrator ? (
            <Button className="cl-danger-outline" disabled={busy} onClick={() => void removeApp()}>
              Remove app
            </Button>
          ) : undefined}
          onClose={() => !busy && setOpen(false)}
          title={group.app_group}
          titleId="cluster-app-group-title"
        >
          <p>{count} service{count === 1 ? "" : "s"} on one overlay network.</p>
          {group.reason && (
            <Notice severity={status.tone === "danger" ? "danger" : status.tone === "warn" ? "warning" : "info"} standing>
              {group.reason}
            </Notice>
          )}
          <div className="cl-svc-section-head"><h3>Services</h3></div>
          <p className="cl-meta">Each service reaches the others by its compose name on the app's private network.</p>
          <ul className="cl-grp-members">
            {group.services.map((member) => (
              <li key={member.name}>
                <div className="cl-grp-members__head">
                  <code>{member.service}</code>
                  {statePill(statusForAppState(member.state))}
                </div>
                <span className="cl-meta">
                  {member.replicas ? `${member.replicas} replicas` : "replicas unknown"}
                </span>
                {member.reason && <span className="cl-meta">{member.reason}</span>}
              </li>
            ))}
          </ul>
        </ClusterDialog>
      )}
    </>
  );
}
