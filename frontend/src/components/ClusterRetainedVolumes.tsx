import { useCallback, useEffect, useState } from "react";
import { useModalAction } from "../hooks/useModalAction";
import { apiRequest } from "../lib/api";
import type { Session } from "../types";
import { ClusterCard, IconTile } from "./ClusterPrimitives";
import { ClusterDialog } from "./ClusterDialog";
import { Button, Input, Notice } from "./ui";
import "../styles/cluster-deployments.css";

/**
 * The data volumes a cluster app removal kept on its worker (W4d-D20).
 *
 * Swarm's `service rm` never deletes a volume, so a removed app's data stays on
 * the machine it ran on. The removal records each one; this lists them, says
 * how to get the data back (deploy the same app under the same name on that
 * machine) and lets an administrator delete one - or forget its entry when the
 * machine has left the fleet. Renders nothing while nothing is retained.
 *
 * VD-200 (the ClusterDeployments board): the "Retained data" card, one row per
 * volume; its delete is the ClusterDialogsConfirm board's typed-name dialog.
 */

export interface RetainedVolume {
  id: string;
  service_name: string;
  volume: string;
  node_id: string;
  node_name: string;
  removed_at: number;
  backups: number;
}

const APP_PREFIX = "vaelor-app-";

/** Typed to forget a controller record (F3): only the owner can say the data is gone. */
const CONTROLLER_REMOVED_ACK = "removed on the controller";

/** "4 Oct 2026, 18:20": when a volume was kept, in the board's own format. */
export function removedAtWords(epochSeconds: number): string {
  const when = new Date(epochSeconds * 1000);
  const two = (value: number) => String(value).padStart(2, "0");
  const day = when.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
  return `${day}, ${two(when.getHours())}:${two(when.getMinutes())}`;
}

/** The deployment name an owner typed, from the managed service name. */
const deploymentName = (service: string) =>
  service.startsWith(APP_PREFIX) ? service.slice(APP_PREFIX.length) : service;

export function ClusterRetainedVolumes({ session, reloadKey }: { session: Session; reloadKey?: string }) {
  const [volumes, setVolumes] = useState<RetainedVolume[]>([]);
  const [readError, setReadError] = useState("");
  const [deleting, setDeleting] = useState<RetainedVolume | null>(null);
  const [confirmation, setConfirmation] = useState("");
  const [removedAck, setRemovedAck] = useState("");
  const [notice, setNotice] = useState("");
  const action = useModalAction();

  const reload = useCallback(async () => {
    try {
      setVolumes(await apiRequest<RetainedVolume[]>("/cluster/retained-volumes"));
      setReadError("");
    } catch (error) {
      setReadError(error instanceof Error ? error.message : "Retained data could not be read.");
    }
  }, []);

  useEffect(() => { void reload(); }, [reload, reloadKey]);

  const close = () => {
    if (action.busy) return;
    setDeleting(null);
    setConfirmation("");
    setRemovedAck("");
    action.clear();
  };

  const remove = (forgetOnly: boolean) => {
    const target = deleting;
    if (!target) return;
    void action.run(async () => {
      const result = await apiRequest<{ already_gone?: boolean; forgotten?: boolean }>(
        `/cluster/retained-volumes/${encodeURIComponent(target.id)}`,
        { method: "DELETE", body: JSON.stringify({
          confirmation, forget_only: forgetOnly,
          ...(forgetOnly && removedAck ? { removed_ack: removedAck } : {}),
        }) },
        session.csrf_token,
      );
      setNotice(
        result.forgotten
          ? `Vaelor no longer lists ${target.volume}.`
          : result.already_gone
            ? `${target.volume} was already gone from ${target.node_name}; it is no longer listed.`
            : `${target.volume} was deleted from ${target.node_name}.`,
      );
    }).then((ok) => {
      if (!ok) return;
      setDeleting(null);
      setConfirmation("");
      setRemovedAck("");
      void reload();
    });
  };


  if (!volumes.length && !readError && !notice) return null;
  const administrator = session.user.role === "administrator";
  // Forget is offered once Delete has said the console cannot reach the data:
  // the machine left, or it is the controller or an unknown machine (F3).
  const machineGone = Boolean(action.error && /no longer enrolled|forget this entry/i.test(action.error));
  const onController = deleting?.node_id === "controller";
  const ackReady = !onController || removedAck.trim().toLowerCase() === CONTROLLER_REMOVED_ACK;

  return (
    <ClusterCard
      description="Removed apps · removing a cluster app keeps its data volume on the machine it ran on. Deploy the same app under the same name there to use it again, or delete it here."
      flush
      icon="database"
      title="Retained data"
    >
      {(readError || notice) && (
        <div className="cl-dep-pad">
          {readError && <Notice severity="warning">{`Retained data could not be read: ${readError}`}</Notice>}
          {notice && <Notice severity="success">{notice}</Notice>}
        </div>
      )}
      {!!volumes.length && (
        <ul className="cl-rows cl-dep-retained">
          {volumes.map((item) => (
            <li aria-label={item.volume} key={item.id}>
              <IconTile name="drive" />
              <div className="cl-rows__text">
                <strong>{item.volume}</strong>
                <span>
                  From {deploymentName(item.service_name)} on {item.node_name}, removed{" "}
                  {removedAtWords(item.removed_at)}
                </span>
                {item.backups ? (
                  <span>{`${item.backups} recovery backup${item.backups === 1 ? "" : "s"} of this app also kept.`}</span>
                ) : (
                  <span className="cl-warn-text">No recovery backup of this app exists; this volume is the only copy.</span>
                )}
              </div>
              {administrator && (
                <Button className="cl-danger-outline" onClick={() => { setNotice(""); setDeleting(item); }}>
                  Delete data
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}

      {deleting && (
        <ClusterDialog
          error={action.error}
          eyebrow="Retained data"
          footer={(
            <>
              <Button disabled={action.busy} onClick={close}>Cancel</Button>
              {machineGone && (
                <Button variant="quiet" disabled={action.busy || confirmation !== deleting.volume || !ackReady} onClick={() => remove(true)}>
                  Forget this entry
                </Button>
              )}
              <Button
                variant="danger"
                disabled={action.busy || confirmation !== deleting.volume}
                onClick={() => remove(false)}
              >
                {action.busy ? "Deleting…" : "Delete data"}
              </Button>
            </>
          )}
          onClose={close}
          title={`Delete ${deleting.volume}?`}
          titleId="retained-delete-title"
          tone="danger"
        >
          <p>
            Vaelor deletes this volume on {deleting.node_name}. The data in it cannot be recovered
            {deleting.backups ? " except from this app's recovery backups." : ", and no backup of it exists."}
          </p>
          <Input
            id="retained-delete-confirmation"
            label={`Type ${deleting.volume} to delete it`}
            onChange={(event) => setConfirmation(event.target.value)}
            value={confirmation}
          />
          {machineGone && onController && (
            <Input
              id="retained-removed-ack"
              label={`After running the command on the controller, type ${CONTROLLER_REMOVED_ACK}`}
              onChange={(event) => setRemovedAck(event.target.value)}
              value={removedAck}
            />
          )}
        </ClusterDialog>
      )}
    </ClusterCard>
  );
}
