import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import "../styles/apps-manage.css";
import "../styles/apps-manage-tools.css";
import { apiRequest } from "../lib/api";
import { formatQuantity, timeAgo } from "../lib/format";
import type { Role, Session } from "../types";
import { AppsIconTile } from "./appsKit";
import { PaginatedItems } from "./PaginatedItems";
import { Button, Input, LoadingLines, Notice, OperationFeedback, type OperationState } from "./ui";

/*
 * Configuration restore points (VD-200, the ManageConsoleRestore board): the
 * app manager's Restore points tab, and the Activity page's recovery list.
 * Restore and delete each confirm in place by typing the app's name.
 */

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
/**
 * How many restore points one read asks for (the server caps it at 200, and
 * Activity and the cluster read the same 50). The list is every app's restore
 * points, newest first, with no per-app filter, so a read that comes back
 * full may have left older points out: counts from it are "at least"
 * (VD-200 apps S3, LESSONS 5).
 */
export const CHECKPOINTS_LIMIT = 50;
const CHECKPOINTS_PATH = `/checkpoints?limit=${CHECKPOINTS_LIMIT}`;

/**
 * The server lists restore points from an operator up (`GET /checkpoints`,
 * api_hardware_routes.py), so a viewer is told why instead of reading into a
 * refusal and being shown "could not be read" (VD-200 apps verify, LESSONS 8).
 * One predicate gates the read and the words for it.
 */
export function recoveryReadRefusal(role: Role): string | undefined {
  return role === "viewer" ? "Operator access is required to read restore points." : undefined;
}

function isRestorable(point: RecoveryPoint): boolean {
  return (
    point.verified
    && point.restorable
    && Number.isFinite(point.size_bytes)
    && point.size_bytes > 0
    && VERIFIED_DIGEST.test(point.sha256)
  );
}

/** One read of the restore points, shared by the Overview tile and the tab. */
export interface RecoveryPointsReading {
  state: "loading" | "ok" | "error";
  /** Restorable points of every app, as read. */
  points: RecoveryPoint[];
  error: string;
  /** The read came back full, so older points may exist that it did not return. */
  truncated: boolean;
  reload: () => Promise<void>;
  replace: (point: RecoveryPoint) => void;
}

type RecoveryRead = Pick<RecoveryPointsReading, "state" | "points" | "error" | "truncated">;

function readRecoveryPoints(): Promise<RecoveryRead> {
  return apiRequest<RecoveryPoint[]>(CHECKPOINTS_PATH).then(
    (data) => ({ state: "ok" as const, points: data.filter(isRestorable), error: "", truncated: data.length >= CHECKPOINTS_LIMIT }),
    (error) => ({ state: "error" as const, points: [], error: error instanceof Error && error.message ? error.message : "Restore points could not be loaded.", truncated: false }),
  );
}

export function useRecoveryPoints(refreshSignal = 0, enabled = true): RecoveryPointsReading {
  const [read, setRead] = useState<RecoveryRead>({ state: "loading", points: [], error: "", truncated: false });
  const reload = useCallback(async () => {
    setRead((current) => ({ ...current, state: "loading" }));
    setRead(await readRecoveryPoints());
  }, []);
  useEffect(() => {
    if (!enabled) return;
    let active = true;
    setRead((current) => ({ ...current, state: "loading" }));
    void readRecoveryPoints().then((next) => { if (active) setRead(next); });
    return () => { active = false; };
  }, [enabled, refreshSignal]);
  const replace = useCallback((point: RecoveryPoint) => {
    if (!isRestorable(point)) return;
    setRead((current) => ({ ...current, points: current.points.map((item) => (item.id === point.id ? point : item)) }));
  }, []);
  return { ...read, reload, replace };
}

/** How many restore points one app has and how old the newest is. */
export interface RecoveryPointSummary {
  /** "refused": this account may not read them, so nothing was asked. */
  state: "loading" | "ok" | "error" | "refused";
  count: number;
  /** Unix seconds of the newest restore point, or null when there is none. */
  newest: number | null;
  /** The read was full: `count` is a lower bound, not the total. */
  atLeast?: boolean;
}

/** The Overview's Restore points tile: the tab's own read, counted for one app. */
export function recoveryPointSummary(reading: RecoveryPointsReading, project: string | null | undefined): RecoveryPointSummary {
  if (reading.state !== "ok") return { state: reading.state, count: 0, newest: null };
  const mine = reading.points.filter((point) => point.project === project);
  return { state: "ok", count: mine.length, newest: mine.length ? Math.max(...mine.map((point) => point.created_at)) : null, atLeast: reading.truncated };
}

/** "3 saved · newest 2 days ago", "None saved yet", or "Not read" - never a zero for a list not read. */
export function recoveryPointSummaryText(summary: RecoveryPointSummary): string {
  if (summary.state === "refused") return "Needs operator access";
  if (summary.state === "error") return "Not read";
  if (summary.state === "loading") return "Reading…";
  if (!summary.count || summary.newest === null) {
    return summary.atLeast ? `None in the newest ${CHECKPOINTS_LIMIT} read` : "None saved yet";
  }
  return `${summary.atLeast ? "At least " : ""}${summary.count} saved · newest ${timeAgo(summary.newest * 1000)}`;
}

export function RecoveryPointList({
  session,
  project,
  refreshSignal = 0,
  headerAction,
  showReload = true,
  reading,
}: {
  session: Session;
  project?: string;
  refreshSignal?: number;
  /** A read the page already holds (the app manager's tile), so the tab does not read again. */
  reading?: RecoveryPointsReading;
  /** What sits in the list's header beside Reload: the app manager's Review checkpoint. */
  headerAction?: ReactNode;
  /**
   * False where the page placing the list has its own Reload, which reloads
   * it through `refreshSignal` (Activity > Restore points): one Reload, not two.
   */
  showReload?: boolean;
}) {
  const isAdministrator = session.user.role === "administrator";
  const refusal = recoveryReadRefusal(session.user.role);
  const own = useRecoveryPoints(refreshSignal, !reading && !refusal);
  const source = reading ?? own;
  const { points, truncated } = source;
  const loadError = !refusal && source.state === "error" ? source.error : "";
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
      source.replace(result);
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
        await source.reload();
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
    <div className="manage-restore">
      <div className="manage-restore__head">
        <div className="manage-restore__titles">
          <h3>Configuration restore points</h3>
          <p>Only non-empty archives verified against their on-disk bytes are shown. Docker volume data and external model files are not included. Restore is available here as soon as a checkpoint finishes.</p>
          {refusal ? <p>{refusal}</p> : !isAdministrator && <p>Verify only. Restore and Delete need an administrator.</p>}
        </div>
        {(headerAction || (showReload && !refusal)) && (
          <div className="manage-restore__actions">
            {headerAction}
            {showReload && !refusal && <Button onClick={() => void source.reload()} variant="quiet">Reload</Button>}
          </div>
        )}
      </div>
      {loadError && <Notice severity="danger">{loadError}</Notice>}
      {truncated && <p className="manage-tool__quiet">Showing the newest {CHECKPOINTS_LIMIT} restore points across all apps; older ones are not listed here.</p>}
      {notice && <OperationFeedback className="manage-restore__feedback" message={notice} state={noticeState} />}
      {refusal ? null : visible.length ? (
        <div className="manage-restore__list">
          <PaginatedItems
            items={visible}
            label={project ? `${project} restore points` : "Recovery restore points"}
            pageSize={6}
            render={(point) => (
              <article className="manage-restore__point" key={point.id}>
                <div className="manage-restore__row">
                  <AppsIconTile name="database" />
                  <div className="manage-row__text">
                    <span className="manage-row__title">{point.project}</span>
                    <span className="manage-row__detail">{new Date(point.created_at * 1000).toLocaleString()} · {formatQuantity(point.size_bytes, "checkpoint")}</span>
                  </div>
                  <div className="manage-restore__buttons">
                    <Button disabled={Boolean(busy)} onClick={() => void verify(point)}>Verify</Button>
                    <Button
                      disabled={Boolean(busy)}
                      disabledReason={isAdministrator ? undefined : "Administrator access is required."}
                      onClick={() => choose("restore", point)}
                      variant="primary"
                    >
                      Restore
                    </Button>
                    {isAdministrator && (
                      <Button disabled={Boolean(busy)} onClick={() => choose("delete", point)} variant="danger">Delete</Button>
                    )}
                  </div>
                </div>
                {pending?.checkpoint.id === point.id && (
                  <form
                    className="manage-restore__confirm"
                    onSubmit={(event) => {
                      event.preventDefault();
                      void applyPending();
                    }}
                  >
                    <div className="manage-restore__confirm-text">
                      <strong>{pending.kind === "restore" ? "Restore this configuration?" : "Delete this restore point?"}</strong>
                      <span>
                        {pending.kind === "restore"
                          ? "The app will briefly stop. Vaelor creates a safety point first and rolls back if startup validation fails."
                          : "This archive will be permanently removed."}
                      </span>
                    </div>
                    <Input
                      autoComplete="off"
                      autoFocus
                      label={<span>Type <strong className="apps-mono">{point.project}</strong> to confirm</span>}
                      onChange={(event) => setConfirmation(event.target.value)}
                      value={confirmation}
                    />
                    <div className="manage-restore__confirm-actions">
                      <Button onClick={() => setPending(null)} variant="quiet">Cancel</Button>
                      <Button
                        busy={busy === point.id}
                        className={pending.kind === "delete" ? "manage-solid-danger" : undefined}
                        disabledReason={confirmation === point.project ? undefined : `Type ${point.project} exactly.`}
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
      ) : source.state === "loading" ? (
        <LoadingLines label="Reading restore points" />
      ) : !loadError && (
        <Notice severity="info" standing>
          <strong>No restore points yet.</strong> Create one from the managed app before changing or removing its configuration.
        </Notice>
      )}
    </div>
  );
}
