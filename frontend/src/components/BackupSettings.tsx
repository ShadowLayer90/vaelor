import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import { formatQuantity } from "../lib/format";
import type { Session } from "../types";
import { Icon } from "./Icon";
import { usePagination } from "./PaginatedItems";
import { RecordDialog, RecordPager, shortStamp } from "./RecordKit";
import { StatusPill } from "./StatusPill";
import { Button, Card, EmptyState, Input, LoadingLines, Notice, Select } from "./ui";
import type { StatusTone } from "./ui";
import "../styles/settings.css";

interface OffsiteConfig {
  backend?: string;
  endpoint?: string;
  bucket?: string;
  prefix?: string;
  region?: string;
  credential_purpose?: string;
}

interface BackupConfig {
  enabled: boolean;
  interval_seconds: number;
  retention_keep: number;
  retention_max_age_seconds: number;
  passphrase_configured: boolean;
  next_run_at: number | null;
  offsite: OffsiteConfig;
}

interface BackupRun {
  id: string;
  created_at: number;
  trigger: string;
  archive_name: string;
  size_bytes: number;
  status: string;
  error: string;
  offsite_status: string;
  offsite_detail: string;
}

interface BackupArchive {
  name: string;
  size_bytes: number;
  modified_at: number;
  offsite_status: string;
  sha256: string;
}

interface BackupStatus {
  config: BackupConfig;
  runs: BackupRun[];
  archives: BackupArchive[];
}

interface PortableStagedStatus {
  staged: boolean;
  confirmation: string;
  plan: { id: string; expires_at: number } | null;
}

const INTERVAL_CHOICES: Array<[string, number]> = [
  ["Every hour", 3600],
  ["Every 6 hours", 21600],
  ["Every 12 hours", 43200],
  ["Every day", 86400],
  ["Every week", 604800],
];

/**
 * A backup archive's size. This module once divided by 1024 x 1024 and wrote
 * "MB" (LESSONS 5); a backup reads as the Recovery checkpoints do, through the
 * one owner of byte figures (W7-D5).
 */
const formatBytes = (bytes: number) => formatQuantity(bytes, "checkpoint");

/** Rows a page of backups or runs shows (the SettingsBackup board). */
const ROWS_PER_PAGE = 6;

/**
 * The off-site copy's state in the board's words. The wire's "skipped" (and
 * an archive that was never offered) reads "none"; a state this console does
 * not know is said so rather than shown as a slug (LESSONS 5).
 */
function offsiteLabel(status: string): string {
  if (status === "ok") return "Off-site: ok";
  if (status === "failed") return "Off-site: failed";
  if (status === "skipped" || status === "") return "Off-site: none";
  if (status === "pending") return "Off-site: sending";
  return "Off-site: state not recognised";
}

/** The same fact under a run, in a sentence's lower case. */
function runOffsiteWords(status: string): string {
  return offsiteLabel(status).replace("Off-site: ", "off-site ");
}

function offsiteTone(status: string): StatusTone {
  if (status === "ok") return "success";
  if (status === "failed") return "danger";
  if (status === "pending") return "warning";
  return "neutral";
}

export function BackupSettings({ session }: { session: Session }) {
  const [status, setStatus] = useState<BackupStatus | null>(null);
  // Whether the last read of the settings failed: the pill then says "Not read", never "Reading" for good.
  const [unread, setUnread] = useState(false);
  const [busy, setBusy] = useState("");
  const [message, setMessageText] = useState("");
  const [failed, setFailed] = useState(false);
  const setMessage = (text: string) => { setFailed(false); setMessageText(text); };
  const reportFailure = (error: unknown, fallback: string) => {
    setFailed(true);
    setMessageText(error instanceof Error && error.message ? error.message : fallback);
  };

  const [enabled, setEnabled] = useState(false);
  const [interval, setInterval] = useState(86400);
  const [retentionKeep, setRetentionKeep] = useState(7);
  const [retentionMaxAgeDays, setRetentionMaxAgeDays] = useState(0);
  const [passphrase, setPassphrase] = useState("");

  const [offsiteBackend, setOffsiteBackend] = useState("");
  const [offsiteEndpoint, setOffsiteEndpoint] = useState("");
  const [offsiteBucket, setOffsiteBucket] = useState("");
  const [offsitePrefix, setOffsitePrefix] = useState("");
  const [offsiteRegion, setOffsiteRegion] = useState("us-east-1");
  const [offsiteCredentials, setOffsiteCredentials] = useState("");

  const [restoreTarget, setRestoreTarget] = useState<string | null>(null);
  const [restorePassphrase, setRestorePassphrase] = useState("");
  const [restoreStaged, setRestoreStaged] = useState<PortableStagedStatus | null>(null);
  const [restoreConfirmation, setRestoreConfirmation] = useState("");
  // VD-189: a refused restore step stays in its dialog; the page under it is inert.
  const [restoreError, setRestoreError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const next = await apiRequest<BackupStatus>("/admin/backups");
      setStatus(next);
      setUnread(false);
      setEnabled(next.config.enabled);
      setInterval(next.config.interval_seconds);
      setRetentionKeep(next.config.retention_keep);
      setRetentionMaxAgeDays(Math.round(next.config.retention_max_age_seconds / 86400));
      const offsite = next.config.offsite || {};
      setOffsiteBackend(offsite.backend ?? "");
      setOffsiteEndpoint(offsite.endpoint ?? "");
      setOffsiteBucket(offsite.bucket ?? "");
      setOffsitePrefix(offsite.prefix ?? "");
      setOffsiteRegion(offsite.region ?? "us-east-1");
    } catch (error) {
      setUnread(true);
      reportFailure(error, "The backup settings could not be loaded.");
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  const saveSchedule = async () => {
    setBusy("schedule"); setMessage("");
    try {
      await apiRequest("/admin/backups/schedule", {
        method: "PUT",
        body: JSON.stringify({
          enabled,
          interval_seconds: interval,
          retention_keep: retentionKeep,
          retention_max_age_seconds: retentionMaxAgeDays * 86400,
        }),
      }, session.csrf_token);
      setMessage("Backup schedule saved.");
      await refresh();
    } catch (error) {
      reportFailure(error, "The backup schedule was not saved.");
    } finally { setBusy(""); }
  };

  const savePassphrase = async () => {
    setBusy("passphrase"); setMessage("");
    try {
      await apiRequest("/admin/backups/passphrase", {
        method: "POST",
        body: JSON.stringify({ passphrase }),
      }, session.csrf_token);
      setPassphrase("");
      setMessage("Backup password stored securely. Keep a copy — it is required to restore.");
      await refresh();
    } catch (error) {
      reportFailure(error, "The backup passphrase was not stored.");
    } finally { setBusy(""); }
  };

  const saveOffsite = async () => {
    setBusy("offsite"); setMessage("");
    try {
      await apiRequest("/admin/backups/offsite", {
        method: "PUT",
        body: JSON.stringify({
          backend: offsiteBackend,
          endpoint: offsiteEndpoint,
          bucket: offsiteBucket,
          prefix: offsitePrefix,
          region: offsiteRegion,
          credentials: offsiteCredentials,
        }),
      }, session.csrf_token);
      setOffsiteCredentials("");
      setMessage(offsiteBackend ? "Off-site target saved." : "Off-site delivery turned off.");
      await refresh();
    } catch (error) {
      reportFailure(error, "The off-site target was not saved.");
    } finally { setBusy(""); }
  };

  const runNow = async () => {
    setBusy("run"); setMessage("");
    try {
      await apiRequest("/admin/backups", { method: "POST", body: "{}" }, session.csrf_token);
      setMessage("Backup created.");
      await refresh();
    } catch (error) {
      reportFailure(error, "The backup did not complete.");
    } finally { setBusy(""); }
  };

  const pushOffsite = async (name: string) => {
    setBusy(`push-${name}`); setMessage("");
    try {
      const result = await apiRequest<{ run: BackupRun }>(
        `/admin/backups/${encodeURIComponent(name)}/offsite`,
        { method: "POST", body: "{}" },
        session.csrf_token,
      );
      setMessage(
        result.run.offsite_status === "ok"
          ? "Archive delivered off-site."
          : `Off-site delivery failed: ${result.run.offsite_detail}`,
      );
      if (result.run.offsite_status !== "ok") setFailed(true);
      await refresh();
    } catch (error) {
      reportFailure(error, "The off-site delivery could not be started.");
    } finally { setBusy(""); }
  };

  const beginRestore = (name: string) => {
    setRestoreTarget(name);
    setRestorePassphrase("");
    setRestoreStaged(null);
    setRestoreConfirmation("");
    setRestoreError("");
    setMessage("");
  };

  const stageRestore = async () => {
    if (!restoreTarget) return;
    setBusy("restore-stage"); setRestoreError("");
    try {
      const staged = await apiRequest<PortableStagedStatus>(
        `/admin/backups/${encodeURIComponent(restoreTarget)}/restore`,
        { method: "POST", body: JSON.stringify({ passphrase: restorePassphrase }) },
        session.csrf_token,
      );
      setRestoreStaged(staged);
    } catch (error) {
      setRestoreError(error instanceof Error && error.message ? error.message : "The backup could not be verified. Check the passphrase.");
    } finally { setBusy(""); }
  };

  const applyRestore = async () => {
    setBusy("restore-apply"); setRestoreError("");
    try {
      await apiRequest(
        "/admin/portable-state/import",
        { method: "POST", body: JSON.stringify({ confirmation: restoreConfirmation }) },
        session.csrf_token,
      );
      setMessage("Restore accepted. Vaelor will replace state, restart services, and end this sign-in.");
      setRestoreTarget(null);
      setRestoreStaged(null);
    } catch (error) {
      setRestoreError(error instanceof Error && error.message ? error.message : "The restore was not accepted.");
    } finally { setBusy(""); }
  };

  const cancelRestore = () => {
    setRestoreTarget(null);
    setRestoreStaged(null);
    setRestoreConfirmation("");
    setRestoreError("");
  };

  const config = status?.config;
  const archivePages = usePagination(status?.archives ?? [], ROWS_PER_PAGE);
  const runPages = usePagination(status?.runs ?? [], ROWS_PER_PAGE);
  const offsiteOn = Boolean(config?.offsite?.backend);

  return (
    <Card
      actions={<>
        {config
          ? <StatusPill label={config.enabled ? "Scheduled" : "Manual only"} tone={config.enabled ? "success" : "neutral"} />
          : <StatusPill label={unread ? "Not read" : "Reading"} reading={unread ? "unread" : undefined} tone="neutral" />}
        <Button busy={busy === "run"} disabled={Boolean(busy) && busy !== "run"} onClick={() => void runNow()}>
          {busy === "run" ? "Backing up…" : "Back up now"}
        </Button>
      </>}
      as="section"
      className="stg-backup"
      flush
      heading={<><span className="record-eyebrow">Scheduled and off-site backups</span>Back up this Vaelor</>}
    >
      <div className="stg-backup__intro">
        <p>Save an encrypted copy of everything on this Vaelor - accounts, agents, settings, and data - so you can put it all back if something goes wrong. Do it by hand, on a schedule, and (optionally) keep a copy off this machine.</p>
        {message && <Notice severity={failed ? "danger" : "success"}>{message}</Notice>}
      </div>

      <div className="stg-steps record-card-button">
        <section aria-labelledby="backup-step-1" className="stg-step">
          <div className="stg-step__head"><span aria-hidden="true">1</span><div><h3 id="backup-step-1">Choose a backup password</h3><p>Your backups are locked with this. You'll need the SAME password to restore, so save it somewhere safe - it is never shown again.</p></div></div>
          {config && (config.passphrase_configured
            ? <StatusPill label="A backup password is set." tone="success" />
            : <StatusPill label="No backup password set yet." tone="neutral" />)}
          <Input
            autoComplete="new-password"
            error={passphrase.length > 0 && passphrase.length < 16 ? "Use at least 16 characters." : undefined}
            hint="At least 16 characters. Write it down before you store it - it cannot be recovered."
            id="backup-passphrase"
            label="Backup password"
            minLength={16}
            onChange={(event) => setPassphrase(event.target.value)}
            type="password"
            value={passphrase}
          />
          <span>
            <Button
              busy={busy === "passphrase"}
              disabled={Boolean(busy) && busy !== "passphrase"}
              disabledReason={busy ? undefined : passphrase.length === 0
                ? (config?.passphrase_configured ? "Type a new password to replace the stored one." : "Type a password of at least 16 characters.")
                : passphrase.length < 16 ? "Use at least 16 characters." : undefined}
              onClick={() => void savePassphrase()}
            >
              {busy === "passphrase" ? "Storing…" : "Store password"}
            </Button>
          </span>
        </section>

        <section aria-labelledby="backup-step-2" className="stg-step">
          <div className="stg-step__head"><span aria-hidden="true">2</span><div><h3 id="backup-step-2">Back up now, or on a schedule</h3><p>Make a backup this instant, or let Vaelor make one on its own on a timer and keep only the newest.</p></div></div>
          <Select hint="How often Vaelor makes a backup on its own (when the schedule is turned on below)." id="backup-interval" label="How often" value={String(interval)} onChange={(event) => setInterval(Number(event.target.value))}>
            {INTERVAL_CHOICES.map(([label, value]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </Select>
          <div className="stg-form__pair stg-form__pair--tight">
            <Input id="backup-keep" label="How many backups to keep" type="number" min={1} max={365} value={String(retentionKeep)} onChange={(event) => setRetentionKeep(Number(event.target.value))} />
            <Input id="backup-max-age" label="Delete backups older than (days)" type="number" min={0} max={365} value={String(retentionMaxAgeDays)} onChange={(event) => setRetentionMaxAgeDays(Number(event.target.value))} />
          </div>
          <small className="ui-muted">Older backups beyond the count are deleted automatically. 0 days means no age limit.</small>
          <label className="stg-switch">
            <span>Run backups automatically</span>
            <input checked={enabled} className="stg-switch__input" id="backup-enabled" onChange={(event) => setEnabled(event.target.checked)} role="switch" type="checkbox" />
          </label>
          <span>
            <Button
              busy={busy === "schedule"}
              disabled={Boolean(busy) && busy !== "schedule"}
              disabledReason={!busy && enabled && !config?.passphrase_configured ? "Set a backup password before enabling the schedule." : undefined}
              onClick={() => void saveSchedule()}
              variant="primary"
            >
              {busy === "schedule" ? "Saving…" : "Save schedule"}
            </Button>
          </span>
        </section>

        <section aria-labelledby="backup-step-3" className="stg-step">
          <div className="stg-step__head"><span aria-hidden="true">3</span><div><h3 id="backup-step-3">Keep a copy off this machine (optional)</h3><p>Also send each backup to cloud storage or another server, so a copy survives even if this machine is lost or fails. Leave this off if local backups are enough.</p></div></div>
          <Select hint="Cloud storage (Amazon S3, MinIO, Backblaze B2, ...) or a plain HTTPS server that accepts an upload." id="offsite-backend" label="Where to send a copy" value={offsiteBackend} onChange={(event) => setOffsiteBackend(event.target.value)}>
            <option value="">Off - keep backups on this machine only</option>
            <option value="s3">Cloud storage (S3-compatible)</option>
            <option value="webhook">Another server (HTTPS upload)</option>
          </Select>
          {offsiteBackend === "" && <small className="ui-muted">Backups stay on this machine only. Send off-site is not offered on backup rows.</small>}
          {offsiteBackend !== "" && (
            <>
              <Input id="offsite-endpoint" label="HTTPS endpoint" placeholder="https://s3.example.com" value={offsiteEndpoint} onChange={(event) => setOffsiteEndpoint(event.target.value)} />
              {offsiteBackend === "s3" ? (
                <div className="stg-form__pair stg-form__pair--tight">
                  <Input id="offsite-bucket" label="Bucket" value={offsiteBucket} onChange={(event) => setOffsiteBucket(event.target.value)} />
                  <Input id="offsite-region" label="Region" value={offsiteRegion} onChange={(event) => setOffsiteRegion(event.target.value)} />
                </div>
              ) : <small className="ui-muted">No bucket or region for an HTTPS upload target.</small>}
              <Input id="offsite-prefix" label="Path prefix (optional)" value={offsitePrefix} onChange={(event) => setOffsitePrefix(event.target.value)} />
              <Input
                id="offsite-credentials"
                label={offsiteBackend === "s3" ? "Access key and secret (JSON)" : "Bearer token (optional)"}
                autoComplete="off"
                type="password"
                placeholder={offsiteBackend === "s3" ? '{"access_key_id":"…","secret_access_key":"…"}' : "leave blank to keep the stored token"}
                value={offsiteCredentials}
                onChange={(event) => setOffsiteCredentials(event.target.value)}
              />
            </>
          )}
          <span>
            <Button busy={busy === "offsite"} disabled={Boolean(busy) && busy !== "offsite"} onClick={() => void saveOffsite()} variant="primary">
              {busy === "offsite" ? "Saving…" : "Save off-site target"}
            </Button>
          </span>
        </section>
      </div>

      <section aria-labelledby="backup-archives" className="stg-backup__list">
        <header className="stg-backup__list-head"><h3 id="backup-archives">Existing backups</h3><span className="ui-muted ui-small">Newest first · {ROWS_PER_PAGE} a page</span></header>
        {status === null && !failed && <LoadingLines label="Loading backups…" />}
        {status && status.archives.length === 0 && (
          <EmptyState icon={<Icon name="download" size={18} />} text="Back up now, or turn on the automatic schedule." title="No backups yet" />
        )}
        {archivePages.visible.map((item) => (
          <div className="stg-row stg-row--backup" key={item.name}>
            <span aria-hidden="true" className="ui-row__icon"><Icon name="database" size={18} /></span>
            <div className="stg-row__text">
              <strong className="record-mono">{item.name}</strong>
              <small>{formatBytes(item.size_bytes)} · {shortStamp(item.modified_at)}</small>
            </div>
            <div className="stg-row__controls record-card-button">
              <StatusPill label={offsiteLabel(item.offsite_status)} tone={offsiteTone(item.offsite_status)} />
              {offsiteOn && (
                <Button className="record-ghost" disabled={busy === `push-${item.name}`} onClick={() => void pushOffsite(item.name)} variant="quiet">
                  {busy === `push-${item.name}` ? "Sending…" : "Send off-site"}
                </Button>
              )}
              <Button aria-label={`Restore ${item.name}`} className="record-danger-outline" disabled={Boolean(busy)} onClick={() => beginRestore(item.name)}>Restore</Button>
            </div>
          </div>
        ))}
        {archivePages.totalPages > 1 && <div className="stg-backup__pager"><RecordPager label="Backup archives" page={archivePages.page} setPage={archivePages.setPage} totalPages={archivePages.totalPages} /></div>}
      </section>

      {status && status.runs.length > 0 && (
        <section aria-labelledby="backup-runs" className="stg-backup__list">
          <header className="stg-backup__list-head"><h3 id="backup-runs">Recent runs</h3><span className="ui-muted ui-small">{ROWS_PER_PAGE} a page</span></header>
          {runPages.visible.map((run) => (
            <div className="stg-row stg-row--backup" key={run.id}>
              <span aria-hidden="true" className="ui-row__icon"><Icon name={run.status === "ok" ? "shield" : "alert"} size={18} /></span>
              <div className="stg-row__text">
                <strong>{shortStamp(run.created_at)} · {run.trigger}</strong>
                {run.status === "ok"
                  ? <small>{formatBytes(run.size_bytes)} · {runOffsiteWords(run.offsite_status)}</small>
                  : <small className="stg-row__error">{run.error || "The run did not record why it failed."}</small>}
              </div>
              <StatusPill label={run.status === "ok" ? "Succeeded" : "Failed"} tone={run.status === "ok" ? "success" : "danger"} />
            </div>
          ))}
          {runPages.totalPages > 1 && <div className="stg-backup__pager"><RecordPager label="Backup run history" page={runPages.page} setPage={runPages.setPage} totalPages={runPages.totalPages} /></div>}
        </section>
      )}

      {restoreTarget && (
        <RestoreDialog
          busy={busy}
          confirmation={restoreConfirmation}
          error={restoreError}
          name={restoreTarget}
          onApply={() => void applyRestore()}
          onCancel={cancelRestore}
          onConfirmation={setRestoreConfirmation}
          onPassphrase={setRestorePassphrase}
          onStage={() => void stageRestore()}
          passphrase={restorePassphrase}
          staged={restoreStaged}
        />
      )}
    </Card>
  );
}

/**
 * Restore from a backup, in two steps (the DialogsRecords board): the backup
 * password verifies the archive, then the typed phrase replaces this
 * appliance's state. Nothing is replaced until step two is approved.
 */
function RestoreDialog({ busy, confirmation, error, name, onApply, onCancel, onConfirmation, onPassphrase, onStage, passphrase, staged }: {
  busy: string;
  confirmation: string;
  error: string;
  name: string;
  onApply: () => void;
  onCancel: () => void;
  onConfirmation: (value: string) => void;
  onPassphrase: (value: string) => void;
  onStage: () => void;
  passphrase: string;
  staged: PortableStagedStatus | null;
}) {
  const cancel = useRef<HTMLButtonElement>(null);
  const verified = Boolean(staged?.staged);
  return (
    <RecordDialog
      actions={<>
        <Button disabled={Boolean(busy)} onClick={onCancel} ref={cancel} type="button">Cancel</Button>
        {verified ? (
          <Button
            busy={busy === "restore-apply"}
            /* S-Y2's shape: the typed phrase gates this even while something else is busy. */
            disabled={Boolean(busy) || confirmation !== staged?.confirmation}
            disabledReason={busy ? undefined : confirmation !== staged?.confirmation ? "Type the phrase above to enable this." : undefined}
            onClick={onApply}
            type="button"
            variant="danger"
          >
            {busy === "restore-apply" ? "Starting restore…" : "Replace state from this backup"}
          </Button>
        ) : (
          <Button
            busy={busy === "restore-stage"}
            disabled={Boolean(busy) || passphrase.length < 16}
            disabledReason={busy ? undefined : passphrase.length < 16 ? "Enter the backup password of at least 16 characters." : undefined}
            onClick={onStage}
            type="button"
            variant="primary"
          >
            {busy === "restore-stage" ? "Verifying…" : "Verify backup"}
          </Button>
        )}
      </>}
      alert={verified}
      error={error}
      eyebrow={verified ? "Restore from · step 2 of 2" : "Restore from · step 1 of 2"}
      eyebrowTone={verified ? "danger" : "plain"}
      initialFocusRef={cancel}
      onClose={() => { if (!busy) onCancel(); }}
      title={<span className="record-mono stg-restore-name">{name}</span>}
      wide
    >
      {verified ? (
        <>
          <Notice severity="info">Backup verified. Confirm to replace this appliance's state.</Notice>
          <Input autoComplete="off" id="restore-confirmation" label={<span>Type <strong>{staged?.confirmation}</strong> to confirm</span>} onChange={(event) => onConfirmation(event.target.value)} value={confirmation} />
        </>
      ) : (
        <>
          <p>Restoring replaces this appliance's state and ends active sessions.</p>
          <Input
            autoComplete="off"
            hint="The same password you set when this backup was made."
            id="restore-passphrase"
            label="Backup password"
            minLength={16}
            onChange={(event) => onPassphrase(event.target.value)}
            type="password"
            value={passphrase}
          />
        </>
      )}
    </RecordDialog>
  );
}
