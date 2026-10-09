import { useCallback, useEffect, useRef, useState } from "react";
import { apiRequest } from "../lib/api";
import type { Session } from "../types";
import { formatBytes } from "../lib/format";
import { Icon } from "./Icon";
import { StatusPill } from "./StatusPill";
import { KvGrid, SectionCard, SystemDialog } from "./systemUi";
import { Button, LoadingLines, Notice } from "./ui";

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
}

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
  /** Why the confirmed apply could not start: said inside the confirm dialog (DialogsSystem). */
  const [applyError, setApplyError] = useState("");
  const cancelRef = useRef<HTMLButtonElement>(null);

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
  const pill = loading
    ? { label: "Checking", tone: "neutral" as const }
    : upgradeAvailable || replaceBuild
      ? { label: "Update available", tone: "info" as const }
      : sameBuild
        ? { label: "Up to date", tone: "success" as const }
        : manifest && kind === "local-build-newer"
          ? { label: "Newer local build", tone: "info" as const }
          : { label: "Not compared", tone: "neutral" as const };
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
          {!started && status && !loading && manifest && eligibility?.eligible && (isAdministrator ? (
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

        {started && (
          <Notice heading="Update in progress" severity="info">
            Vaelor will restart to apply the release; this page may briefly disconnect. Reload it in a moment to see the result.
          </Notice>
        )}

        {!started && status && !loading && (
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
