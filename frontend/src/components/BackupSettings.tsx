import { useCallback, useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import type { Session } from "../types";
import { Icon } from "./Icon";
import { PaginatedItems } from "./PaginatedItems";
import { StatusPill } from "./StatusPill";
import { Button, Checkbox, Input, Notice, Select } from "./ui";
import type { StatusTone } from "./ui";

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

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function offsiteTone(status: string): StatusTone {
  if (status === "ok") return "success";
  if (status === "failed") return "danger";
  if (status === "pending") return "warning";
  return "neutral";
}

export function BackupSettings({ session }: { session: Session }) {
  const [status, setStatus] = useState<BackupStatus | null>(null);
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

  const refresh = useCallback(async () => {
    try {
      const next = await apiRequest<BackupStatus>("/admin/backups");
      setStatus(next);
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
    setMessage("");
  };

  const stageRestore = async () => {
    if (!restoreTarget) return;
    setBusy("restore-stage"); setMessage("");
    try {
      const staged = await apiRequest<PortableStagedStatus>(
        `/admin/backups/${encodeURIComponent(restoreTarget)}/restore`,
        { method: "POST", body: JSON.stringify({ passphrase: restorePassphrase }) },
        session.csrf_token,
      );
      setRestoreStaged(staged);
      setMessage("Backup verified. Confirm to replace this appliance's state.");
    } catch (error) {
      reportFailure(error, "The backup could not be verified. Check the passphrase.");
    } finally { setBusy(""); }
  };

  const applyRestore = async () => {
    setBusy("restore-apply"); setMessage("");
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
      reportFailure(error, "The restore was not accepted.");
    } finally { setBusy(""); }
  };

  const cancelRestore = () => {
    setRestoreTarget(null);
    setRestoreStaged(null);
    setRestoreConfirmation("");
  };

  const config = status?.config;

  return (
    <section className="data-panel admin-wide" aria-labelledby="backup-title">
      <div className="panel-heading">
        <div>
          <span className="page-eyebrow">Scheduled and off-site backups</span>
          <h2 id="backup-title">Back up this Vaelor</h2>
          <p>Save an encrypted copy of everything on this Vaelor - accounts, agents, settings, and data - so you can put it all back if something goes wrong. Do it by hand, on a schedule, and (optionally) keep a copy off this machine.</p>
        </div>
        <StatusPill
          label={config?.enabled ? "Scheduled" : "Manual only"}
          tone={config?.enabled ? "success" : "neutral"}
        />
      </div>

      {message && <Notice severity={failed ? "danger" : "info"}>{message}</Notice>}

      <div className="portable-state__actions">
        <article>
          <div className="portable-state__action-heading">
            <Icon name="lock" />
            <div><h3>1. Choose a backup password</h3><p>Your backups are locked with this. You'll need the SAME password to restore, so save it somewhere safe - it is never shown again.</p></div>
          </div>
          <Input
            autoComplete="new-password"
            hint="At least 16 characters. Write it down before you store it - it cannot be recovered."
            id="backup-passphrase"
            label="Backup password"
            minLength={16}
            onChange={(event) => setPassphrase(event.target.value)}
            type="password"
            value={passphrase}
          />
          <Button
            variant="primary"
            disabled={Boolean(busy) || passphrase.length < 16}
            disabledReason={passphrase.length > 0 && passphrase.length < 16 ? "Use at least 16 characters." : undefined}
            onClick={() => void savePassphrase()}
          >
            {busy === "passphrase" ? "Storing…" : "Store password"}
          </Button>
          <p className="portable-state__hint">
            {config?.passphrase_configured ? "A backup password is set." : "No backup password set yet."}
          </p>
        </article>

        <article>
          <div className="portable-state__action-heading">
            <Icon name="settings" />
            <div><h3>2. Back up now, or on a schedule</h3><p>Make a backup this instant, or let Vaelor make one on its own on a timer and keep only the newest.</p></div>
          </div>
          <Select hint="How often Vaelor makes a backup on its own (when the schedule is turned on below)." id="backup-interval" label="How often" value={String(interval)} onChange={(event) => setInterval(Number(event.target.value))}>
            {INTERVAL_CHOICES.map(([label, value]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </Select>
          <Input hint="Older backups beyond this count are deleted automatically." id="backup-keep" label="How many backups to keep" type="number" min={1} max={365} value={String(retentionKeep)} onChange={(event) => setRetentionKeep(Number(event.target.value))} />
          <Input hint="Also delete backups past this age. 0 means no age limit." id="backup-max-age" label="Delete backups older than (days)" type="number" min={0} max={365} value={String(retentionMaxAgeDays)} onChange={(event) => setRetentionMaxAgeDays(Number(event.target.value))} />
          <Checkbox id="backup-enabled" checked={enabled} label="Run backups automatically" onChange={(event) => setEnabled(event.target.checked)} />
          <Button
            variant="primary"
            disabled={Boolean(busy) || (enabled && !config?.passphrase_configured)}
            disabledReason={enabled && !config?.passphrase_configured ? "Set a backup password before enabling the schedule." : undefined}
            onClick={() => void saveSchedule()}
          >
            {busy === "schedule" ? "Saving…" : "Save schedule"}
          </Button>
          <Button variant="quiet" disabled={Boolean(busy)} onClick={() => void runNow()}>
            {busy === "run" ? "Backing up…" : "Back up now"}
          </Button>
        </article>

        <article>
          <div className="portable-state__action-heading">
            <Icon name="upload" />
            <div><h3>3. Keep a copy off this machine (optional)</h3><p>Also send each backup to cloud storage or another server, so a copy survives even if this machine is lost or fails. Leave this off if local backups are enough.</p></div>
          </div>
          <Select hint="Cloud storage (Amazon S3, MinIO, Backblaze B2, ...) or a plain HTTPS server that accepts an upload." id="offsite-backend" label="Where to send a copy" value={offsiteBackend} onChange={(event) => setOffsiteBackend(event.target.value)}>
            <option value="">Off - keep backups on this machine only</option>
            <option value="s3">Cloud storage (S3-compatible)</option>
            <option value="webhook">Another server (HTTPS upload)</option>
          </Select>
          {offsiteBackend !== "" && (
            <>
              <Input id="offsite-endpoint" label="HTTPS endpoint" placeholder="https://s3.example.com" value={offsiteEndpoint} onChange={(event) => setOffsiteEndpoint(event.target.value)} />
              {offsiteBackend === "s3" && (
                <>
                  <Input id="offsite-bucket" label="Bucket" value={offsiteBucket} onChange={(event) => setOffsiteBucket(event.target.value)} />
                  <Input id="offsite-region" label="Region" value={offsiteRegion} onChange={(event) => setOffsiteRegion(event.target.value)} />
                </>
              )}
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
          <Button variant="primary" disabled={Boolean(busy)} onClick={() => void saveOffsite()}>
            {busy === "offsite" ? "Saving…" : "Save off-site target"}
          </Button>
        </article>
      </div>

      <div className="credential-list">
        <h3>Existing backups</h3>
        {status && status.archives.length > 0 ? (
          <PaginatedItems
            items={status.archives}
            label="Backup archives"
            pageSize={6}
            render={(item) => (
              <div className="credential-row" key={item.name}>
                <span className="credential-row__icon"><Icon name="download" /></span>
                <span>
                  <strong>{item.name}</strong>
                  <small>{formatBytes(item.size_bytes)} · {new Date(item.modified_at * 1000).toLocaleString()}</small>
                </span>
                <StatusPill label={`Off-site: ${item.offsite_status}`} tone={offsiteTone(item.offsite_status)} />
                {config?.offsite?.backend && (
                  <Button variant="quiet" disabled={busy === `push-${item.name}`} onClick={() => void pushOffsite(item.name)}>
                    {busy === `push-${item.name}` ? "Sending…" : "Send off-site"}
                  </Button>
                )}
                <Button variant="danger" disabled={Boolean(busy)} onClick={() => beginRestore(item.name)}>Restore</Button>
              </div>
            )}
          />
        ) : (
          <div className="empty-state"><Icon name="download" /><strong>No backups yet</strong><span>Back up now, or turn on the automatic schedule.</span></div>
        )}
      </div>

      {restoreTarget && (
        <div className="portable-state__approval" role="group" aria-label="Restore from backup">
          <div>
            <small>Restore from</small>
            <strong>{restoreTarget}</strong>
            <span>Restoring replaces this appliance's state and ends active sessions.</span>
          </div>
          {!restoreStaged?.staged ? (
            <>
              <Input
                autoComplete="off"
                hint="The same password you set when this backup was made."
                id="restore-passphrase"
                label="Backup password"
                minLength={16}
                type="password"
                value={restorePassphrase}
                onChange={(event) => setRestorePassphrase(event.target.value)}
              />
              <div>
                <Button variant="quiet" disabled={Boolean(busy)} onClick={cancelRestore}>Cancel</Button>
                <Button variant="primary" disabled={Boolean(busy) || restorePassphrase.length < 16} onClick={() => void stageRestore()}>
                  {busy === "restore-stage" ? "Verifying…" : "Verify backup"}
                </Button>
              </div>
            </>
          ) : (
            <>
              <Input
                autoComplete="off"
                id="restore-confirmation"
                label={<span>Type <strong>{restoreStaged.confirmation}</strong> to confirm</span>}
                value={restoreConfirmation}
                onChange={(event) => setRestoreConfirmation(event.target.value)}
              />
              <div>
                <Button variant="quiet" disabled={Boolean(busy)} onClick={cancelRestore}>Cancel</Button>
                <Button
                  variant="danger"
                  disabled={Boolean(busy) || restoreConfirmation !== restoreStaged.confirmation}
                  onClick={() => void applyRestore()}
                >
                  {busy === "restore-apply" ? "Starting restore…" : "Replace state from this backup"}
                </Button>
              </div>
            </>
          )}
        </div>
      )}

      {status && status.runs.length > 0 && (
        <div className="credential-list">
          <h3>Recent runs</h3>
          <PaginatedItems
            items={status.runs}
            label="Backup run history"
            pageSize={6}
            render={(run) => (
              <div className="credential-row" key={run.id}>
                <span className="credential-row__icon"><Icon name={run.status === "ok" ? "shield" : "alert"} /></span>
                <span>
                  <strong>{new Date(run.created_at * 1000).toLocaleString()} · {run.trigger}</strong>
                  <small>{run.status === "ok" ? `${formatBytes(run.size_bytes)} · off-site ${run.offsite_status}` : run.error}</small>
                </span>
                <StatusPill label={run.status === "ok" ? "Succeeded" : "Failed"} tone={run.status === "ok" ? "success" : "danger"} />
              </div>
            )}
          />
        </div>
      )}
    </section>
  );
}
