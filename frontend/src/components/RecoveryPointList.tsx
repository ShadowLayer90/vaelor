import { useCallback, useEffect, useMemo, useState } from "react";
import { apiRequest } from "../lib/api";
import { formatQuantity } from "../lib/format";
import type { Session } from "../types";
import { Icon } from "./Icon";
import { PaginatedItems } from "./PaginatedItems";
import { Button, Input, OperationFeedback, type OperationState } from "./ui";

export interface RecoveryPoint {
  id: string;
  project: string;
  created_at: number;
  size_bytes: number;
  sha256: string;
  manifest_digest: string;
  manifest_entries: number;
  verified: boolean;
  restorable: boolean;
}

type PendingAction = { kind: "restore" | "delete"; checkpoint: RecoveryPoint } | null;

const VERIFIED_DIGEST = /^[0-9a-f]{64}$/;

function isRestorable(point: RecoveryPoint): boolean {
  return (
    point.verified
    && point.restorable
    && Number.isFinite(point.size_bytes)
    && point.size_bytes > 0
    && VERIFIED_DIGEST.test(point.sha256)
  );
}

export function RecoveryPointList({
  session,
  project,
  refreshSignal = 0,
}: {
  session: Session;
  project?: string;
  refreshSignal?: number;
}) {
  const [points, setPoints] = useState<RecoveryPoint[]>([]);
  const [pending, setPending] = useState<PendingAction>(null);
  const [confirmation, setConfirmation] = useState("");
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [noticeState, setNoticeState] = useState<OperationState>("idle");
  const setFeedback = (message: string, state: OperationState = "success") => {
    setNotice(message);
    setNoticeState(message ? state : "idle");
  };
  const clearFeedback = () => setFeedback("", "idle");

  const refresh = useCallback(async () => {
    const data = await apiRequest<RecoveryPoint[]>("/checkpoints?limit=50");
    setPoints(data.filter(isRestorable));
  }, []);

  useEffect(() => {
    void refresh().catch((error) => {
      setFeedback(error instanceof Error ? error.message : "Restore points could not be loaded.", "error");
    });
  }, [refresh, refreshSignal]);

  const visible = useMemo(
    () => (project ? points.filter((point) => point.project === project) : points),
    [points, project],
  );

  const verify = async (point: RecoveryPoint) => {
    setBusy(point.id);
    clearFeedback();
    try {
      const result = await apiRequest<RecoveryPoint>(
        "/checkpoints/" + encodeURIComponent(point.id) + "/verify",
        { method: "POST", body: "{}" },
        session.csrf_token,
      );
      if (isRestorable(result)) {
        setPoints((current) => current.map((item) => item.id === result.id ? result : item));
      }
      setFeedback(`Verified · SHA-256 ${result.sha256.slice(0, 12)}…`);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "Verification failed.", "error");
    } finally {
      setBusy("");
    }
  };

  const applyPending = async () => {
    if (
      !pending
      || confirmation !== pending.checkpoint.project
      || !isRestorable(pending.checkpoint)
    ) {
      return;
    }
    const point = pending.checkpoint;
    setBusy(point.id);
    clearFeedback();
    try {
      if (pending.kind === "restore") {
        await apiRequest(
          "/checkpoints/" + encodeURIComponent(point.id) + "/restore",
          {
            method: "POST",
            body: JSON.stringify({
              project: point.project,
              sha256: point.sha256,
              confirm: confirmation,
            }),
          },
          session.csrf_token,
        );
        setFeedback("Restore queued. Vaelor will first preserve the current configuration, validate this checkpoint, and verify startup.", "pending");
      } else {
        await apiRequest(
          `/checkpoints/${encodeURIComponent(point.id)}`,
          {
            method: "DELETE",
            body: JSON.stringify({ confirmation }),
          },
          session.csrf_token,
        );
        setFeedback(`Restore point for ${point.project} deleted.`);
        await refresh();
      }
      setPending(null);
      setConfirmation("");
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : "The recovery action could not be queued.", "error");
    } finally {
      setBusy("");
    }
  };

  const choose = (kind: "restore" | "delete", checkpoint: RecoveryPoint) => {
    if (kind === "restore" && !isRestorable(checkpoint)) return;
    setPending({ kind, checkpoint });
    setConfirmation("");
    clearFeedback();
  };

  return (
    <div className="recovery-points">
      <div className="recovery-scope">
        <Icon name="shield" />
        <p>
          <strong>Configuration restore points</strong>
          <span>Only non-empty archives verified against their on-disk bytes are shown. Docker volume data and external model files are not included.</span>
        </p>
        <Button onClick={() => void refresh()} variant="quiet">Reload</Button>
      </div>
      {notice && <OperationFeedback className="recovery-operation-feedback" message={notice} state={noticeState} />}
      {visible.length ? (
        <div className="checkpoint-list">
          <PaginatedItems
            items={visible}
            label={project ? `${project} restore points` : "Recovery restore points"}
            pageSize={6}
            render={(point) => (
              <article key={point.id}>
                <span><Icon name="database" /></span>
                <div>
                  <strong>{point.project}</strong>
                    <small>{new Date(point.created_at * 1000).toLocaleString()} · {formatQuantity(point.size_bytes, "checkpoint")}</small>
                </div>
                <div className="checkpoint-actions">
                  <Button disabled={Boolean(busy)} onClick={() => void verify(point)}>Verify</Button>
                  {session.user.role === "administrator" && (
                    <Button disabled={Boolean(busy)} onClick={() => choose("restore", point)} variant="primary">Restore</Button>
                  )}
                  {session.user.role === "administrator" && (
                    <Button disabled={Boolean(busy)} onClick={() => choose("delete", point)} variant="danger">Delete</Button>
                  )}
                </div>
                {pending?.checkpoint.id === point.id && (
                  <form
                    className="checkpoint-confirm"
                    onSubmit={(event) => {
                      event.preventDefault();
                      void applyPending();
                    }}
                  >
                    <div>
                      <strong>{pending.kind === "restore" ? "Restore this configuration?" : "Delete this restore point?"}</strong>
                      <span>
                        {pending.kind === "restore"
                          ? "The app will briefly stop. Vaelor creates a safety point first and rolls back if startup validation fails."
                          : "This archive will be permanently removed."}
                      </span>
                    </div>
                    <Input
                      autoFocus
                      label={<span>Type <strong>{point.project}</strong> to confirm</span>}
                      onChange={(event) => setConfirmation(event.target.value)}
                      value={confirmation}
                    />
                    <div>
                      <Button onClick={() => setPending(null)} variant="quiet">Cancel</Button>
                      <Button
                        disabled={busy === point.id || confirmation !== point.project}
                        type="submit"
                        variant={pending.kind === "delete" ? "danger" : "primary"}
                      >
                        {pending.kind === "restore" ? "Create safety point and restore" : "Delete restore point"}
                      </Button>
                    </div>
                  </form>
                )}
              </article>
            )}
          />
        </div>
      ) : (
        <div className="empty-state">
          <Icon name="shield" />
          <strong>No restore points yet</strong>
          <span>Create one from the managed app before changing or removing its configuration.</span>
        </div>
      )}
    </div>
  );
}
