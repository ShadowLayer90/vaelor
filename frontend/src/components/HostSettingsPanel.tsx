import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { apiRequest } from "../lib/api";
import { bytesIn, formatTypedGib, timeAgo } from "../lib/format";
import {
  CONFIRM,
  linkKindLabel,
  linkSpeedLabel,
  linkSummary,
  machineInSentence,
  parsePoolSize,
  poolBytes,
  poolPill,
  readHostSettings,
  type HostSettings,
  type PoolMachine,
} from "../lib/hostSettings";
import { jobIsSuccessful, jobIsTerminal, type PresentableJob } from "../lib/jobPresentation";
import type { Session } from "../types";
import { ActionReviewDialog, type ProposedJob } from "./ActionReviewDialog";
import { ClusterCard, KeyValues } from "./ClusterPrimitives";
import { ClusterDialog } from "./ClusterDialog";
import { Icon } from "./Icon";
import { RecheckOutcome, RecheckTrigger, recheckFailureFrom, recheckResultFrom, useRecheck, type RecheckResponse, type RecheckResult } from "./RecheckButton";
import { StatusPill } from "./StatusPill";
import { Button, Input, Notice, Select } from "./ui";
import "../styles/cluster-setup.css";

/**
 * Machine settings (Cluster > Setup, the ClusterSetupMachines board): the GPU
 * memory pool of each machine and the cluster link (VD-161, VD-162).
 *
 * Two settings that used to be hand edits on each box. Both are read from
 * `GET /host-settings`; nothing here is a default or a guess - a machine that
 * cannot have a setting says why in the backend's own sentence, and a figure
 * that was not read is not shown as a number.
 *
 * Every change is reviewed before it is sent, and no change restarts a
 * machine. A new pool size counts only after a restart, which is its own
 * action with its own confirmation, so the reader always chooses when the
 * models and apps on that machine stop.
 *
 * A worker's card is a stored reading. It says when it was read, offers a
 * Recheck, and - when the backend calls the reading stale - asks for that
 * Recheck before a change can be reviewed. A job this panel queued is followed
 * until it finishes, and the settings are read again then, so a card never
 * keeps showing the state from before the job.
 */

type Pending =
  | { kind: "job"; job: ProposedJob; summary: string; label: string }
  | { kind: "restart"; machine: PoolMachine }
  | { kind: "link"; name: string };

/** How long a queued job is followed before the panel stops asking. */
const FOLLOW_LIMIT_MS = 20 * 60 * 1000;

function isController(machine: PoolMachine): boolean {
  return machine.role === "controller";
}

function poolJob(machine: PoolMachine, action: "set" | "revert", size?: number): ProposedJob {
  if (isController(machine)) {
    return {
      type: "host.gpu-memory.apply",
      payload: {
        action,
        ...(action === "set" ? { size_gib: size } : {}),
        confirm: action === "set" ? CONFIRM.setPool : CONFIRM.revertPool,
      },
    };
  }
  return {
    type: "cluster.node.gpu-memory",
    payload: {
      node_id: machine.node_id,
      action,
      ...(action === "set" ? { size_gib: size } : {}),
      confirm: CONFIRM.workerPool,
    },
  };
}

/** The line under a pool card's name: its role and when it was read. */
export function poolReadLine(machine: PoolMachine): string {
  if (isController(machine)) return "Head controller · read live each time this page loads";
  return typeof machine.checked_at === "number"
    ? `Worker · read ${timeAgo(machine.checked_at * 1000)}`
    : "Worker · not read yet";
}

function PoolCard({
  machine,
  administrator,
  onReview,
  onRecheck,
}: {
  machine: PoolMachine;
  administrator: boolean;
  onReview: (pending: Pending) => void;
  onRecheck: (machine: PoolMachine) => Promise<RecheckResult>;
}) {
  const { pool } = machine;
  const pill = poolPill(pool);
  const recheck = useRecheck(useCallback(() => onRecheck(machine), [onRecheck, machine]));
  const [text, setText] = useState("");
  // Review was pressed with nothing usable in the field: say what is needed
  // instead of showing a button that cannot be pressed and does not say why.
  const [asked, setAsked] = useState(false);
  const parsed = parsePoolSize(text, pool);
  const ram = pool.ram_bytes ?? 0;
  const leftAtSize = parsed.size === null ? null : Math.max(0, ram - bytesIn(parsed.size, "GiB"));
  const range = `From ${formatTypedGib(pool.min_gib)}, the kernel's own size rounded up, to ${formatTypedGib(pool.max_gib)}.`;
  const hint = leftAtSize === null
    ? `${range} The rest of this machine's memory stays for the system.`
    : `${range} At ${formatTypedGib(parsed.size)}, about ${poolBytes(leftAtSize)} stays for the system.`;
  const fieldError = parsed.error
    || (asked && parsed.size === null ? `Enter a whole number from ${pool.min_gib} to ${pool.max_gib}.` : "");
  const headingId = `host-pool-${machine.node_id}`;
  const stale = Boolean(machine.stale_reason);
  const worker = !isController(machine);
  const canEdit = administrator && pool.can_change && !stale;
  const canRevert = administrator && Boolean(pool.override && pool.override.state !== "absent") && !stale;
  const canRestart = administrator && pool.restart_pending === true && !stale;
  const reviewSize = () => {
    if (parsed.size === null) {
      setAsked(true);
      return;
    }
    onReview({
      kind: "job",
      job: poolJob(machine, "set", parsed.size),
      summary: `Set the GPU memory pool on ${machineInSentence(machine.name)} to ${formatTypedGib(parsed.size)}, leaving about ${poolBytes(leftAtSize)} for the system. Nothing restarts now: the new size counts after you restart ${machineInSentence(machine.name)}.`,
      label: `GPU memory pool change on ${machineInSentence(machine.name)}`,
    });
  };
  const revertButton = canRevert && (
    <Button
      onClick={() => onReview({
        kind: "job",
        job: poolJob(machine, "revert"),
        summary: `Remove the GPU memory pool setting from ${machineInSentence(machine.name)}. Nothing restarts now: the pool goes back to the kernel's own size, about ${poolBytes(pool.default_bytes)}, after you restart ${machineInSentence(machine.name)}.`,
        label: `Removal of the GPU memory pool setting on ${machineInSentence(machine.name)}`,
      })}
    >
      Remove setting
    </Button>
  );
  return (
    <article className="cl-setup__pool" aria-labelledby={headingId}>
      <div className="cl-setup__pool-head">
        <div className="cl-setup__pool-name">
          {/* One line, so a long name never puts this card's rows below the
              other card's (W6-2); the title keeps the whole name. */}
          <h3 id={headingId} title={machine.name}>{machine.name}</h3>
          <p>{poolReadLine(machine)}</p>
        </div>
        <span className="cl-setup__pool-trail">
          {worker && administrator && (
            // W6-2 (LESSONS 13): the label stays one word, so a long machine
            // name never wraps this row; the machine is named to a screen
            // reader. W6-D4 (LESSONS 6): it is the Fleet card's button and
            // outcome line, so a Recheck says "Rechecked at HH:MM" - or why
            // not, as an alert - on both cards. The outcome line goes at the
            // card's end (FE-W7-4).
            <RecheckTrigger busy={recheck.busy} machineName={machine.name} run={recheck.run} />
          )}
          <StatusPill label={pill.label} tone={pill.tone} />
        </span>
      </div>
      {!pool.supported ? (
        <p className="cl-meta">{pool.reason}</p>
      ) : (
        <>
          <KeyValues
            label={`${machine.name} GPU memory pool`}
            items={[
              { label: "Pool now", value: poolBytes(pool.current_bytes) },
              { label: "Kernel default", value: `About ${poolBytes(pool.default_bytes)}` },
              { label: "System memory", value: poolBytes(pool.ram_bytes) },
              { label: "Left for system", value: poolBytes(pool.system_left_bytes) },
            ]}
          />
          {pool.restart_reason && (
            <p className="cl-warn-text cl-setup__note" role="status">{pool.restart_reason}</p>
          )}
          {!pool.can_change && <p className="cl-meta">{pool.blocked_reason}</p>}
          {stale && <p className="cl-warn-text cl-setup__note" role="status">{machine.stale_reason}</p>}
          {!administrator && pool.can_change && (
            <p className="cl-meta">An administrator can change this machine's GPU memory pool.</p>
          )}
          {canEdit && (
            <div className="cl-setup__pool-edit">
              <Input
                error={fieldError || undefined}
                hint={hint}
                inputMode="numeric"
                label={`New pool size for ${machineInSentence(machine.name)} (GiB)`}
                onChange={(event) => { setText(event.target.value); setAsked(false); }}
                value={text}
              />
              <span className="cl-actions cl-setup__pool-buttons">
                <Button variant="primary" onClick={reviewSize}>Review new size</Button>
                {revertButton}
              </span>
            </div>
          )}
          {(canRestart || (!canEdit && canRevert)) && (
            <div className="cl-actions">
              {canRestart && (
                <Button className="cl-danger-outline" onClick={() => onReview({ kind: "restart", machine })}>
                  <Icon name="power" size={16} />
                  Restart {worker ? machine.name : "this controller"}
                </Button>
              )}
              {!canEdit && revertButton}
            </div>
          )}
        </>
      )}
      <RecheckOutcome result={recheck.result} />
    </article>
  );
}

export function HostSettingsPanel({
  session,
  besideLink,
  followEveryMs = 3000,
}: {
  session: Session;
  /** A card the Setup tab places beside the Cluster link card (the Advanced Cluster settings card). */
  besideLink?: ReactNode;
  /** How often a queued job is asked about; a test shortens it. */
  followEveryMs?: number;
}) {
  const administrator = session.user.role === "administrator";
  const [settings, setSettings] = useState<HostSettings | null>(null);
  const [loadError, setLoadError] = useState("");
  const [notice, setNotice] = useState("");
  const [failure, setFailure] = useState("");
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<Pending | null>(null);
  const [linkChoice, setLinkChoice] = useState<string | null>(null);
  const [following, setFollowing] = useState<{ id: string; label: string } | null>(null);
  const mounted = useRef(true);
  const cancelRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const load = useCallback(async () => {
    try {
      const surface = readHostSettings(await apiRequest<unknown>("/host-settings"));
      if (!mounted.current) return;
      if (!surface) {
        setLoadError("the answer was not in the form this page reads");
        return;
      }
      setSettings(surface);
      setLoadError("");
    } catch (error) {
      if (mounted.current) {
        setLoadError(error instanceof Error ? error.message : "The request failed.");
      }
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  // Follow a job this panel queued until it finishes, then read the settings
  // again: the job changed a machine, so what is on screen is out of date the
  // moment it ends. Bounded, and stopped when the panel goes away.
  useEffect(() => {
    if (!following) return undefined;
    const started = Date.now();
    let timer: number | undefined;
    let stopped = false;
    const ask = async () => {
      if (stopped) return;
      try {
        const job = await apiRequest<PresentableJob>(`/jobs/${following.id}`);
        if (stopped) return;
        if (jobIsTerminal(job)) {
          if (jobIsSuccessful(job)) {
            setNotice(job.message || `${following.label} finished.`);
            setFailure("");
          } else {
            setFailure(job.message || `${following.label} did not finish.`);
            setNotice("");
          }
          setFollowing(null);
          void load();
          return;
        }
      } catch {
        // A missed answer is not the job's outcome; ask again until the limit.
      }
      if (Date.now() - started > FOLLOW_LIMIT_MS) {
        setNotice(`${following.label} is still running. It appears in Activity; use Refresh here when it has finished.`);
        setFollowing(null);
        return;
      }
      timer = window.setTimeout(() => void ask(), followEveryMs);
    };
    timer = window.setTimeout(() => void ask(), followEveryMs);
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [following, followEveryMs, load]);

  const run = async (work: () => Promise<string>) => {
    setBusy(true);
    try {
      const message = await work();
      setNotice(message);
      setFailure("");
    } catch (error) {
      setFailure(error instanceof Error ? error.message : "The request failed.");
    } finally {
      setPending(null);
      setBusy(false);
    }
  };

  const queue = async (body: ProposedJob, label: string, queued: string) => {
    const job = await apiRequest<{ id?: string }>(
      "/jobs", { method: "POST", body: JSON.stringify(body) }, session.csrf_token,
    );
    if (job && typeof job.id === "string") setFollowing({ id: job.id, label });
    return queued;
  };

  const approveJob = () => {
    if (pending?.kind !== "job") return;
    const { job, label } = pending;
    void run(() => queue(job, label, `${label} started. This page shows the result when it finishes; nothing restarts.`));
  };

  const confirmRestart = () => {
    if (pending?.kind !== "restart") return;
    const { machine } = pending;
    void run(async () => {
      if (isController(machine)) {
        await apiRequest("/power/actions", {
          method: "POST",
          body: JSON.stringify({ action: "reboot", confirmation: CONFIRM.controllerRestart }),
        }, session.csrf_token);
        return "This controller is restarting. The console comes back when it has started again.";
      }
      return queue(
        {
          type: "cluster.node.reboot",
          payload: { node_id: machine.node_id, confirm: CONFIRM.workerRestart },
        },
        `Restart of ${machineInSentence(machine.name)}`,
        `${machine.name} was asked to restart. This page reads it again when it is back.`,
      );
    });
  };

  const confirmLink = () => {
    if (pending?.kind !== "link") return;
    const { name } = pending;
    void run(async () => {
      const updated = readHostSettings(await apiRequest<unknown>("/host-settings/cluster-link", {
        method: "POST",
        body: JSON.stringify({ interface: name, confirm: CONFIRM.clusterLink }),
      }, session.csrf_token));
      if (updated) setSettings(updated);
      else void load();
      setLinkChoice(null);
      return name
        ? `${name} is now the cluster link for split models deployed from here on.`
        : "The cluster link was cleared. Split models use each machine's enrolled address.";
    });
  };

  // W6-D4 (LESSONS 6): one reading of the answer, the Fleet card's. A 200
  // whose telemetry check failed is still said as an alert (W6-3, VD-189),
  // beside the button that asked rather than on the page.
  const recheck = async (machine: PoolMachine): Promise<RecheckResult> => {
    try {
      return recheckResultFrom(await apiRequest<RecheckResponse>(
        `/cluster/nodes/${machine.node_id}/refresh`, { method: "POST" }, session.csrf_token,
      ));
    } catch (error) {
      return recheckFailureFrom(error);
    } finally {
      void load();
    }
  };

  const link = settings?.cluster_link;
  const storedLink = link?.chosen?.name ?? "";
  const selectedLink = linkChoice ?? storedLink;
  const usableLinks = (link?.links ?? []).filter((item) => item.usable);
  const restartMachine = pending?.kind === "restart" ? pending.machine : null;
  const restartingController = restartMachine !== null && isController(restartMachine);
  const linkName = pending?.kind === "link" ? pending.name : "";
  const linkNote = linkName ? link?.links.find((item) => item.name === linkName)?.shared_note ?? "" : "";
  const refresh = <Button variant="quiet" onClick={() => void load()}><Icon name="refresh" size={16} />Refresh</Button>;
  const cancelPending = () => { if (!busy) setPending(null); };

  const linkCard = settings && (
    <ClusterCard
      description="Carries traffic between machines when one model is split. Vaelor lists this controller's own links; it does not search the network."
      icon="network"
      title="Cluster link"
    >
      {!link?.available ? (
        <p className="cl-meta">{link?.reason}</p>
      ) : (
        <>
          <div className="cl-setup__chosen">
            <StatusPill
              label={!link.chosen ? "Not chosen" : link.chosen.valid ? "Chosen" : "Needs attention"}
              tone={!link.chosen ? "neutral" : link.chosen.valid ? "success" : "warning"}
            />
            <span className="cl-strong">
              {!link.chosen
                ? "No cluster link is chosen. A split model uses the address each machine was enrolled on."
                : link.chosen.valid
                  ? `${link.chosen.name} (${link.chosen.network}) carries split-model traffic.`
                  : link.chosen.reason}
            </span>
          </div>
          <ul className="cl-setup__links" aria-label="This controller's network links">
            {link.links.map((item) => (
              <li key={item.name}>
                <strong className="cl-setup__mono">{item.name}</strong>
                <span className="cl-setup__link-detail">
                  <span>{linkKindLabel(item.kind)} · {linkSpeedLabel(item.speed_mbps)} · {item.state === "up" ? "Connected" : item.state === "down" ? "Not connected" : "State not reported"}</span>
                  {item.network && <span> · network {item.network}, this machine {item.address}</span>}
                  {!item.usable && <>{" · "}<span className="cl-warn-text">{item.reason}</span></>}
                </span>
                {link.chosen?.valid && link.chosen.name === item.name && <StatusPill label="Chosen" tone="success" />}
              </li>
            ))}
            {!link.links.length && <li><span className="cl-meta">No network links were found on this controller.</span></li>}
          </ul>
          {!!link.machines.length && link.chosen?.valid && (
            <ul className="cl-setup__links" aria-label="Workers on the cluster link">
              {link.machines.map((machine) => (
                <li key={machine.node_id}>
                  <strong>{machine.name}</strong>
                  <span className="cl-setup__link-detail">
                    {machine.on_link === true
                      ? `On this link as ${machine.address} (${machine.interface}).`
                      : machine.reason}
                  </span>
                </li>
              ))}
            </ul>
          )}
          {link.notes.map((note) => (
            <p className="cl-warn-text cl-setup__note" key={note} role="status">{note}</p>
          ))}
          {administrator ? (
            <div className="cl-setup__pool-edit">
              <Select
                hint="Every machine in a split model needs an address on the chosen link's network."
                label="Link for split-model traffic"
                onChange={(event) => setLinkChoice(event.target.value)}
                value={selectedLink}
              >
                <option value="">None: use each machine's enrolled address</option>
                {storedLink && !usableLinks.some((item) => item.name === storedLink) && (
                  <option value={storedLink}>{storedLink} · not available now</option>
                )}
                {usableLinks.map((item) => (
                  <option key={item.name} value={item.name}>{linkSummary(item)}</option>
                ))}
              </Select>
              {/* Shown only when there is a change to review: a button that
                  cannot be pressed and does not say why is worse than none. */}
              {selectedLink !== storedLink ? (
                <span className="cl-actions cl-setup__pool-buttons">
                  <Button variant="primary" onClick={() => setPending({ kind: "link", name: selectedLink })}>
                    Review link change
                  </Button>
                </span>
              ) : (
                <p className="cl-meta cl-setup__pool-buttons">Pick a different link to change it.</p>
              )}
            </div>
          ) : (
            <p className="cl-meta">An administrator can choose the cluster link.</p>
          )}
        </>
      )}
    </ClusterCard>
  );

  return (
    <>
      {loadError && (
        <Notice severity="warning">
          <span className="cl-notice-row">
            <span>Machine settings could not be read ({loadError}). Nothing below is current until it is read again.</span>
            {!settings && <span className="cl-notice-row__actions">{refresh}</span>}
          </span>
        </Notice>
      )}
      {notice && <Notice severity="info">{notice}</Notice>}
      {failure && <Notice severity="danger">{failure}</Notice>}
      {following && (
        <p className="cl-meta" role="status">{following.label} is in progress…</p>
      )}
      {!settings && !loadError && (
        <p className="cl-meta" role="status">Reading machine settings…</p>
      )}

      {settings && (
        <ClusterCard
          actions={refresh}
          description="How much of each machine's memory its GPU may use for models. A larger pool fits a larger model and leaves less for the system; a new size counts after a restart."
          icon="gpu"
          title="GPU memory pool"
        >
          <div className="cl-grid">
            {settings.gpu_memory_pool.machines.map((machine) => (
              <PoolCard
                administrator={administrator}
                key={machine.node_id}
                machine={machine}
                onRecheck={recheck}
                onReview={setPending}
              />
            ))}
          </div>
        </ClusterCard>
      )}

      {(linkCard || besideLink) && (
        <div className="cl-setup__pair">
          {linkCard}
          {besideLink}
        </div>
      )}

      <ActionReviewDialog
        busy={busy}
        job={pending?.kind === "job" ? pending.job : null}
        onApprove={approveJob}
        onCancel={cancelPending}
        summary={pending?.kind === "job" ? pending.summary : ""}
      />
      {restartMachine && (
        <ClusterDialog
          eyebrow="Can't be undone while it restarts"
          footer={(
            <>
              <Button disabled={busy} onClick={cancelPending} ref={cancelRef}>Cancel</Button>
              <Button disabled={busy} onClick={confirmRestart} variant="danger">
                {busy ? "Sending…" : restartingController ? "Restart this controller" : `Restart ${restartMachine.name}`}
              </Button>
            </>
          )}
          initialFocusRef={cancelRef}
          onClose={cancelPending}
          role="alertdialog"
          title={restartingController ? "Restart this controller?" : `Restart ${restartMachine.name}?`}
          tone="warning"
        >
          <p>
            {restartingController
              ? "Every model and app on this controller stops, and this console is unreachable, until it has started again. Models served from other machines through this controller stop answering too."
              : `Every model and app running on ${restartMachine.name} stops until it has started again. A model split across machines stops answering; a model with a copy on another machine keeps answering from there.`}
          </p>
        </ClusterDialog>
      )}
      {pending?.kind === "link" && (
        <ClusterDialog
          eyebrow="Cluster link"
          footer={(
            <>
              <Button disabled={busy} onClick={cancelPending} ref={cancelRef}>Cancel</Button>
              <Button disabled={busy} onClick={confirmLink} variant="primary">
                {busy ? "Sending…" : linkName ? "Use this link" : "Clear the link"}
              </Button>
            </>
          )}
          initialFocusRef={cancelRef}
          onClose={cancelPending}
          role="alertdialog"
          title={linkName ? `Use ${linkName} as the cluster link?` : "Clear the cluster link?"}
          tone="accent"
        >
          <p>
            {linkName
              ? `Split models deployed from now on send the traffic between machines over ${linkName}. Each machine in a split must have an address on that link's network, or the deployment is refused before anything starts. Nothing that is running changes.`
              : "Split models deployed from now on use the address each machine was enrolled on. Nothing that is running changes."}
          </p>
          {linkNote && <Notice severity="info">{linkNote}</Notice>}
        </ClusterDialog>
      )}
    </>
  );
}
