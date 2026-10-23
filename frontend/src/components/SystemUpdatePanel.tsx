import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import type { Session } from "../types";
import { formatBytes } from "../lib/format";
import { updatePill } from "../lib/upgradeState";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { KvGrid, SectionCard, SystemDialog, useVisiblePoll } from "./systemUi";
import { Button, LoadingLines, Notice } from "./ui";
import { ServedModels, UpdatePhases, type UpgradeProgress } from "./UpdateProgress";

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

/**
 * What the offered release is relative to the installed build
 * (`release_source.OFFER_KINDS`). An equal version string is not an equal
 * build (W4d-D8), so "same-build" and "replace-build" are told apart.
 */
/** `release_source.OFFER_KINDS`, copied: the Python one is right if they differ. */
export const OFFER_KINDS = [ // vocabulary: offer-kind
  "update", "same-build", "replace-build", "local-build-newer", "build-not-comparable",
  "build-unknown", "older", "too-old-to-upgrade", "unsigned", "unparseable",
] as const;
export type OfferKind = (typeof OFFER_KINDS)[number];

/** `build_provenance.INSTALLED_BUILD_STATES`, copied. */
export const INSTALLED_BUILD_STATES = ["recorded", "stale", "unknown"] as const; // vocabulary: installed-build-state
/** `build_provenance.BUILT_AT_BASES`, copied; `built_at_basis` is typed by it. */
export const BUILT_AT_BASES = ["build-stamp", "newest-file-in-wheel"] as const; // vocabulary: built-at-basis

/** Whether the offered `manifest.version` may be applied to this appliance. */
export interface UpgradeEligibility {
  eligible: boolean;
  reason: string;
  kind?: OfferKind;
}

/** The installed wheel, as `build_provenance.installed_build` proves it. */
export interface InstalledBuild {
  state: (typeof INSTALLED_BUILD_STATES)[number];
  reason: string;
  build: {
    version?: string;
    sha256?: string;
    bytes?: number;
    commit?: string | null;
    built_at?: string | null;
    built_at_basis?: (typeof BUILT_AT_BASES)[number] | null;
  } | null;
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
  installed_build?: InstalledBuild | null;
  signatures_required?: boolean;
  source: string;
  manifest: UpgradeManifest | null;
  eligibility: UpgradeEligibility | null;
  confirmation: string;
  last_result: UpgradeLastResult | null;
  /** The latest update phase by phase (`upgrade_progress`, R104); absent from older servers. */
  progress?: UpgradeProgress | null;
}

/** How often the panel reads an update it follows. */
export const FOLLOW_POLL_MS = 3000;
/** How long it follows one before saying it is taking longer than expected. */
export const FOLLOW_LIMIT_MS = 20 * 60 * 1000;
/** How long a finished update stays on the panel after a reload. */
const RECENT_MS = 60 * 60 * 1000;

/** ISO published date, shown as a local date; the raw value if it will not parse. */
function publishedOn(published_at: string): string {
  const when = new Date(published_at);
  return Number.isNaN(when.getTime()) ? published_at : when.toLocaleDateString();
}

/** The first twelve hex digits of a digest, the way the panel names a wheel. */
function shortSha(sha: string | undefined | null): string {
  return sha ? sha.slice(0, 12) : "unknown";
}

/** One line naming the running build: wheel, build date and commit when known. */
function describeBuild(installed: InstalledBuild | null | undefined): string {
  const build = installed?.state === "recorded" ? installed.build : null;
  if (!build) {
    return `Build not identified: ${installed?.reason || "no install record was found."}`;
  }
  const parts = [`Wheel ${shortSha(build.sha256)}`];
  if (build.built_at) {
    parts.push(
      build.built_at_basis === "build-stamp"
        ? `built ${publishedOn(build.built_at)}`
        : `files dated up to ${publishedOn(build.built_at)}`,
    );
  }
  parts.push(build.commit ? `commit ${build.commit.slice(0, 12)}` : "commit not recorded");
  return parts.join(" · ");
}

/**
 * Pull and apply the latest pinned Vaelor release from GitHub.
 *
 * System › Hardware and services, beside the operating system's own updates
 * (VD-200 decision 3, the SystemHardwarePi board); the confirm is the
 * DialogsSystem board's. It reads `GET /upgrade` on mount, shows the running version and what is offered,
 * and — for an administrator — applies the offered release through the same
 * gated, audited job path a factory reset uses. A viewer sees the same status
 * without an action.
 *
 * Once an update is accepted the panel follows it (R104): it reads
 * `GET /upgrade` every few seconds, phase by phase as `upgrade_progress`
 * derives them, through the restart (a read that fails then is the restart,
 * retried quietly), and says "Updated to X" by itself when the running
 * version reaches X. It stops when every phase has ended, or after
 * FOLLOW_LIMIT_MS with a plain "taking longer than expected" and a way to
 * look again. A page reloaded mid-update picks the same update up.
 */
export function SystemUpdatePanel({ session }: { session: Session }) {
  const isAdministrator = session.user.role === "administrator";
  const [status, setStatus] = useState<UpgradeStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [started, setStarted] = useState(false);
  /** Why the confirmed apply could not start: said inside the confirm dialog (DialogsSystem). */
  const [applyError, setApplyError] = useState("");
  /** Following an update: reading it until every phase has ended (R104). */
  const [following, setFollowing] = useState(false);
  /** When following began: a ref, so a read started in the same tick sees it. */
  const followSince = useRef(0);
  /** The last read failed while following: the appliance is restarting. */
  const [reconnecting, setReconnecting] = useState(false);
  const [slow, setSlow] = useState(false);
  const cancelRef = useRef<HTMLButtonElement>(null);

  const follow = useCallback(() => {
    setFollowing(true);
    followSince.current = Date.now();
    setSlow(false);
  }, []);

  const check = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const next = await apiRequest<UpgradeStatus>("/upgrade", { cache: "no-store" });
      setStatus(next);
      setConfirming(false);
      // R107: an unfinished update is followed only inside the follow window;
      // an older one is shown as taking too long, with the Update button back.
      if (next.progress && !next.progress.finished) {
        if (Date.now() - next.progress.started_at * 1000 < FOLLOW_LIMIT_MS) follow();
        else setSlow(true);
      }
    } catch (caught) {
      setError(caught instanceof Error && caught.message ? caught.message : "The update status could not be read.");
    } finally {
      setLoading(false);
    }
  }, [follow]);

  useEffect(() => { void check(); }, [check]);

  const poll = useCallback(async () => {
    if (Date.now() - followSince.current > FOLLOW_LIMIT_MS) {
      setSlow(true);
      return;
    }
    try {
      const next = await apiRequest<UpgradeStatus>("/upgrade", { cache: "no-store" });
      setStatus(next);
      setReconnecting(false);
      if (next.progress?.finished) {
        setFollowing(false);
        setStarted(false);
      }
    } catch {
      // The restart: the server that would answer is the one restarting.
      setReconnecting(true);
    }
  }, []);

  useVisiblePoll(() => { void poll(); }, FOLLOW_POLL_MS, following && !slow);

  const apply = async () => {
    if (!status?.manifest) return;
    setBusy(true);
    setApplyError("");
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
      follow();
      void poll();
    } catch (caught) {
      setApplyError(caught instanceof Error && caught.message ? caught.message : "The update could not be started.");
    } finally {
      setBusy(false);
    }
  };

  const manifest = status?.manifest ?? null;
  const eligibility = status?.eligibility ?? null;
  const installed = status?.installed_build ?? null;
  const isReinstall = Boolean(manifest && status && manifest.version === status.running_version);
  // W4d-D8: "up to date" is a claim about the BUILD. Only the same wheel is up
  // to date; a same-version offer of a different wheel is a replacement, and
  // the backend refuses one that is older or that it cannot compare.
  const kind: OfferKind = eligibility?.kind ?? (isReinstall ? "build-unknown" : "update");
  const upgradeAvailable = Boolean(manifest && eligibility?.eligible && !isReinstall);
  const sameBuild = Boolean(manifest && eligibility?.eligible && kind === "same-build");
  const replaceBuild = Boolean(manifest && eligibility?.eligible && kind === "replace-build");
  const offerLabel = manifest
    ? isReinstall
      ? replaceBuild ? `Replace with the published ${manifest.version}` : `Reinstall ${manifest.version}`
      : `Update to ${manifest.version}`
    : "";
  const lastResult = status?.last_result ?? null;
  const run = status?.progress ?? null;
  const showRun = Boolean(run && (following || started || !run.finished
    || Date.now() - run.started_at * 1000 < RECENT_MS));
  /**
   * The Update button hides while an update is followed (R107: it comes back
   * after the follow window) and whenever the backend says something is still
   * working on one (R114: never a second update over a live apply).
   */
  const busyUpdating = Boolean(run?.active) || ((started || following) && !slow);
  // One owner for these words, read by Home › Machines too (VD-133 1.5.5).
  const pill = loading ? { label: "Checking", tone: "neutral" as const } : updatePill(status);
  const runningBuild = describeBuild(installed);
  const source = status?.source ?? "the release source";
  const whatInstalls = !manifest
    ? ""
    : kind === "same-build"
      ? `This reinstalls the same build you run now (wheel ${shortSha(manifest.sha256)}): Vaelor downloads it from ${source}, checks it against the release's SHA-256, installs it, and restarts.`
      : kind === "replace-build"
        ? `This replaces the build you run now (${runningBuild}) with the published ${manifest.version} (wheel ${shortSha(manifest.sha256)}, released ${publishedOn(manifest.published_at)}), a different build of the same version. Vaelor downloads it from ${source}, checks it against the release's SHA-256, installs it, and restarts.`
        : `This installs Vaelor ${manifest.version} (wheel ${shortSha(manifest.sha256)}, released ${publishedOn(manifest.published_at)}) in place of ${status?.running_version ?? "the running version"}. Vaelor downloads it from ${source}, checks it against the release's SHA-256, installs it, and restarts.`;

  return (
    <>
    <SectionCard
      actions={<StatusPill label={pill.label} reading={loading ? "unread" : undefined} tone={pill.tone} />}
      className="sys-vaelor-update"
      eyebrow="Keep Vaelor current"
      footer={(
        <div className="sys-actions-end">
          <Button busy={loading} disabled={busy} onClick={() => void check()} type="button" variant="quiet">
            {loading ? "Checking…" : "Check for updates"}
          </Button>
          {!busyUpdating && status && !loading && manifest && eligibility?.eligible && (isAdministrator ? (
            <Button
              onClick={() => { setApplyError(""); setConfirming(true); }}
              type="button"
              variant={upgradeAvailable ? "primary" : "secondary"}
            >
              {isReinstall ? "Reinstall Vaelor" : "Update Vaelor"}
            </Button>
          ) : <span className="sys-muted">An administrator can apply this update.</span>)}
        </div>
      )}
      title="Update Vaelor"
      titleId="system-update-title"
    >
      <div className="sys-stack">
        <p className="sys-lead">Pull and apply the latest Vaelor release. Vaelor verifies the download, installs it, and restarts; a release that fails its health check is rolled back automatically.</p>
        {status ? (
          <KvGrid
            items={[
              { label: "Currently running", value: `Vaelor ${status.running_version}`, detail: runningBuild, mono: true },
              { label: "Release source", value: status.source, mono: true },
            ]}
            label="Running build"
          />
        ) : loading ? <LoadingLines label="Reading the running version…" /> : null}

        {error && <Notice severity="danger">{error}</Notice>}

        {(started || (showRun && run)) && (
          <UpdateRun reconnecting={reconnecting} run={showRun ? run : null} />
        )}

        {slow && (
          <div className="sys-run__slow">
            <Notice heading="This update is taking longer than expected" severity="warning">
              The page stopped following it after {Math.round(FOLLOW_LIMIT_MS / 60000)} minutes. Vaelor keeps working on it. Check again to read where it stands.
            </Notice>
            <Button onClick={() => { follow(); void poll(); }} type="button" variant="secondary">Check again</Button>
          </div>
        )}

        {!busyUpdating && status && !loading && (
          <>
            {!manifest && (
              <Notice severity="info">
                {eligibility
                  ? `No update is available right now from ${status.source}. ${eligibility.reason}`
                  : `No release could be read from ${status.source} right now (none is published, or it could not be reached), so Vaelor cannot say whether a newer one exists.`}
              </Notice>
            )}

            {sameBuild && (
              <Notice severity="success">
                Vaelor is up to date: this appliance runs the published {status.running_version} build itself. You can reinstall it below to re-apply the same bytes.
              </Notice>
            )}

            {manifest && eligibility?.eligible && (
              <div className="sys-offer">
                <Icon name="download" size={18} />
                <div>
                  <strong>{offerLabel}</strong>
                  <small>Published {publishedOn(manifest.published_at)} · {formatBytes(manifest.bytes)} · sha256 {manifest.sha256.slice(0, 12)}</small>
                  <small>
                    {manifest.signature
                      ? "Signed release."
                      : "Not signed. The SHA-256 check proves the download is the release's file, not who published it."}
                  </small>
                </div>
              </div>
            )}

            {manifest && !eligibility?.eligible && (
              <Notice severity={kind === "build-unknown" ? "warning" : "info"}>
                {eligibility?.reason || "This appliance is not eligible for the offered release."}
              </Notice>
            )}
          </>
        )}

        {lastResult && (lastResult.rolled_back || lastResult.completed) && (
          <p className={lastResult.rolled_back ? "sys-last sys-last--rolled-back" : "sys-last"}>
            <Icon name={lastResult.rolled_back ? "alert" : "shield"} size={16} />
            {lastResult.rolled_back
              ? `A previous update to ${lastResult.to_version ?? "the offered release"} failed its health check and was rolled back to ${lastResult.from_version ?? "the previous version"}.`
              : `Updated to ${lastResult.to_version ?? status?.running_version ?? "the latest release"}.`}
          </p>
        )}
      </div>
    </SectionCard>

    {/* The DialogsSystem board's confirm: Cancel first, a refusal said inside it. */}
    {confirming && manifest && (
      <SystemDialog
        error={applyError}
        eyebrow="Vaelor restarts"
        eyebrowTone="warning"
        footer={(
          <>
            <Button disabled={busy} onClick={() => setConfirming(false)} ref={cancelRef} type="button" variant="secondary">Cancel</Button>
            <Button busy={busy} onClick={() => void apply()} type="button" variant="primary">
              {busy
                ? isReinstall ? "Starting reinstall…" : "Starting update…"
                : isReinstall ? "Confirm reinstall" : "Confirm update"}
            </Button>
          </>
        )}
        initialFocusRef={cancelRef}
        onClose={() => { if (!busy) setConfirming(false); }}
        role="alertdialog"
        showClose={false}
        title={`${offerLabel}?`}
        titleId="system-update-confirm-title"
      >
        <p>{whatInstalls} If it fails its health check, Vaelor puts back the build running now.</p>
      </SystemDialog>
    )}
    </>
  );
}

/** The update being followed, or the last one: its headline, phases and model result (R104). */
function UpdateRun({ run, reconnecting }: { run: UpgradeProgress | null; reconnecting: boolean }) {
  if (!run) {
    return (
      <Notice heading="Update in progress" severity="info">
        Vaelor is starting the update. This page follows it and says when it is done.
      </Notice>
    );
  }
  const heading = run.failed
    ? `The update to ${run.to_version} did not complete`
    : run.reached ? `Updated to ${run.to_version}` : "Update in progress";
  const lead = run.failed
    ? "Each step below says what happened."
    : run.reached
      ? run.finished
        ? `Vaelor ${run.to_version} is running and every step has ended.`
        : `Vaelor ${run.to_version} is running. The rest of the update finishes below; this page follows it.`
      : `Vaelor is installing ${run.to_version} and restarts by itself. This page follows it and needs no reload.`;
  return (
    <div className="sys-run">
      <Notice heading={heading} severity={run.failed ? "danger" : run.reached ? "success" : "info"}>
        {lead}
      </Notice>
      <UpdatePhases phases={run.phases} restarting={reconnecting} />
      {run.model && run.model.models.length > 0 && <ServedModels models={run.model.models} />}
    </div>
  );
}
