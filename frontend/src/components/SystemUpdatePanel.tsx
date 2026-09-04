import { useCallback, useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import type { Session } from "../types";
import { formatBytes } from "../lib/format";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button, Notice } from "./ui";

/** One offered release, exactly as `manifest.public()` serves it. */
export interface UpgradeManifest {
  version: string;
  wheel_name: string;
  sha256: string;
  bytes: number;
  min_from_version: string;
  signature: string | null;
  published_at: string;
}

/** Whether the offered `manifest.version` may be applied to this appliance. */
export interface UpgradeEligibility {
  eligible: boolean;
  reason: string;
}

/**
 * The outcome of the last apply. Every field is optional and read
 * defensively: the broker records this after a restart and older shapes may
 * omit fields, so the panel optional-chains all of them.
 */
export interface UpgradeLastResult {
  ok?: boolean;
  completed?: boolean;
  from_version?: string;
  to_version?: string;
  running_version?: string;
  rolled_back?: boolean;
  completed_at?: number | string;
}

/** `GET /api/v2/upgrade`. `manifest === null` means no release is offered. */
export interface UpgradeStatus {
  running_version: string;
  source: string;
  manifest: UpgradeManifest | null;
  eligibility: UpgradeEligibility | null;
  confirmation: string;
  last_result: UpgradeLastResult | null;
}

/** ISO published date, shown as a local date; the raw value if it will not parse. */
function publishedOn(published_at: string): string {
  const when = new Date(published_at);
  return Number.isNaN(when.getTime()) ? published_at : when.toLocaleDateString();
}

/**
 * Pull and apply the latest pinned Vaelor release from GitHub.
 *
 * The routine recovery action, above the last-resort Remove Vaelor panel. It
 * reads `GET /upgrade` on mount, shows the running version and what is offered,
 * and — for an administrator — applies the offered release through the same
 * gated, audited job path a factory reset uses. A viewer sees the same status
 * without an action. The apply restarts the control plane, so once the job is
 * accepted the panel stops and says the page may briefly disconnect rather than
 * polling through a restart it cannot see across.
 */
export function SystemUpdatePanel({ session }: { session: Session }) {
  const isAdministrator = session.user.role === "administrator";
  const [status, setStatus] = useState<UpgradeStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [started, setStarted] = useState(false);

  const check = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const next = await apiRequest<UpgradeStatus>("/upgrade", { cache: "no-store" });
      setStatus(next);
      setConfirming(false);
    } catch (caught) {
      setError(caught instanceof Error && caught.message ? caught.message : "The update status could not be read.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void check(); }, [check]);

  const apply = async () => {
    if (!status?.manifest) return;
    setBusy(true);
    setError("");
    try {
      await apiRequest(
        "/upgrade",
        {
          method: "POST",
          body: JSON.stringify({
            payload: { confirmation: status.confirmation, to_version: status.manifest.version },
          }),
        },
        session.csrf_token,
      );
      setStarted(true);
      setConfirming(false);
    } catch (caught) {
      setError(caught instanceof Error && caught.message ? caught.message : "The update could not be started.");
    } finally {
      setBusy(false);
    }
  };

  const manifest = status?.manifest ?? null;
  const eligibility = status?.eligibility ?? null;
  const isReinstall = Boolean(manifest && status && manifest.version === status.running_version);
  // A genuinely newer release is an update; the same version offered back is a
  // reinstall, not an update. The pill and the primary action follow this so an
  // appliance already on the latest release is never told an update is waiting.
  const upgradeAvailable = Boolean(manifest && eligibility?.eligible && !isReinstall);
  const reinstallOnly = Boolean(manifest && eligibility?.eligible && isReinstall);
  const offerLabel = manifest
    ? isReinstall ? `Reinstall ${manifest.version}` : `Update to ${manifest.version}`
    : "";
  const lastResult = status?.last_result ?? null;

  return (
    <section aria-labelledby="system-update-title" className="data-panel admin-wide system-update">
      <div className="panel-heading">
        <div>
          <span className="page-eyebrow">Keep Vaelor current</span>
          <h2 id="system-update-title">Update Vaelor</h2>
          <p>Pull and apply the latest Vaelor release. Vaelor verifies the download, installs it, and restarts; a release that fails its health check is rolled back automatically.</p>
        </div>
        <StatusPill
          label={loading ? "Checking" : upgradeAvailable ? "Update available" : "Up to date"}
          tone={loading ? "neutral" : upgradeAvailable ? "info" : "success"}
        />
      </div>

      <div className="system-update__running">
        <small>Currently running</small>
        <strong>{status ? `Vaelor ${status.running_version}` : "Reading the running version…"}</strong>
        {status && <small>Release source: {status.source}</small>}
      </div>

      {error && <Notice severity="danger"><Icon name="alert" />{error}</Notice>}

      {started && (
        <Notice severity="info" heading="Update in progress">
          Vaelor will restart to apply the release; this page may briefly disconnect. Reload it in a moment to see the result.
        </Notice>
      )}

      {!started && status && !loading && (
        <>
          {!manifest && (
            <Notice severity={eligibility ? "info" : "success"}>
              {eligibility
                ? `No update is available right now from ${status.source}. ${eligibility.reason}`
                : "Vaelor is up to date. You are running the latest offered release."}
            </Notice>
          )}

          {reinstallOnly && (
            <Notice severity="success">
              Vaelor is up to date — running the latest release ({status.running_version}). You can reinstall it below to re-apply the same version.
            </Notice>
          )}

          {manifest && eligibility?.eligible && (
            <div className="system-update__offer">
              <div className="system-update__offer-heading">
                <Icon name="download" />
                <div>
                  <strong>{offerLabel}</strong>
                  <small>
                    Published {publishedOn(manifest.published_at)} · {formatBytes(manifest.bytes)} · sha256 {manifest.sha256.slice(0, 12)}
                  </small>
                </div>
              </div>
              {isAdministrator ? (
                confirming ? (
                  <div className="system-update__confirm">
                    <Notice severity="warning" heading={`${offerLabel}?`}>
                      This downloads the release from GitHub, verifies its checksum, installs it, and restarts Vaelor. If the new version fails its health check it is rolled back to {status.running_version} automatically.
                    </Notice>
                    <div className="system-update__actions">
                      <Button disabled={busy} onClick={() => setConfirming(false)} type="button" variant="quiet">Cancel</Button>
                      <Button busy={busy} onClick={() => void apply()} type="button" variant="primary">
                        {busy
                          ? isReinstall ? "Starting reinstall…" : "Starting update…"
                          : isReinstall ? "Confirm reinstall" : "Confirm update"}
                      </Button>
                    </div>
                  </div>
                ) : (
                  <Button onClick={() => setConfirming(true)} type="button" variant={upgradeAvailable ? "primary" : "quiet"}>
                    {isReinstall ? "Reinstall Vaelor" : "Update Vaelor"}
                  </Button>
                )
              ) : (
                <small>An administrator can apply this update.</small>
              )}
            </div>
          )}

          {manifest && !eligibility?.eligible && (
            <Notice severity="info">
              {eligibility?.reason ?? "This appliance is not eligible for the offered release."}
            </Notice>
          )}
        </>
      )}

      {lastResult && (lastResult.rolled_back || lastResult.completed) && (
        <p className="system-update__last-result">
          <Icon name={lastResult.rolled_back ? "alert" : "shield"} />
          {lastResult.rolled_back
            ? `A previous update to ${lastResult.to_version ?? "the offered release"} failed its health check and was rolled back to ${lastResult.from_version ?? "the previous version"}.`
            : `Updated to ${lastResult.to_version ?? status?.running_version ?? "the latest release"}.`}
        </p>
      )}

      <div className="system-update__actions">
        <Button busy={loading} disabled={busy} onClick={() => void check()} type="button" variant="quiet">
          {loading ? "Checking…" : "Check for updates"}
        </Button>
      </div>
    </section>
  );
}
