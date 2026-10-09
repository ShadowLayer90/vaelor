import { useRef, useState } from "react";
import { formatQuantity } from "../lib/format";
import { PaginatedItems } from "./PaginatedItems";
import { StatusPill } from "./StatusPill";
import { KvGrid, SectionCard, SystemDialog } from "./systemUi";
import { UpdateJobStatus, type UpdateJob } from "./UpdateJobStatus";
import { Button, Notice } from "./ui";
import type { StatusTone } from "./ui/status";

/** The operating system's package updates, as `/system/inventory` serves them. */
export interface PackageUpdates {
  /** False when the update check did not run; `count` is then null (review B2). */
  collected?: boolean;
  reason?: string;
  available: boolean | null;
  count: number | null;
  packages: string[];
  details: Array<{
    name: string;
    installed: string;
    candidate: string;
    source: string;
    classification: string;
  }>;
  staged: boolean;
  staged_archive_count: number;
  staged_size_bytes: number;
  estimated_install_minutes: number;
  reboot_likelihood: "unlikely" | "possible" | "likely";
  reboot_required: boolean;
  /** What the last install left waiting, each with apt's reason (W4d-D31). */
  held_back?: Array<{ name: string; reason: string }>;
  last_action?: "stage" | "apply";
  last_completed_at?: number;
}

function updatesPill(updates: PackageUpdates | undefined, loading: boolean): { label: string; tone: StatusTone; reading?: "unread" } {
  if (loading) return { label: "Checking…", tone: "neutral", reading: "unread" };
  if (updates?.reboot_required) return { label: "Restart required", tone: "warning" };
  // A check that did not run is not "Up to date" (review B2).
  if (updates?.collected === false) return { label: "Not checked", tone: "neutral", reading: "unread" };
  if (updates?.count) return { label: `${updates.count} available`, tone: "warning" };
  return { label: "Up to date", tone: "success" };
}

/**
 * System › Hardware and services › Software updates (VD-200, the System board):
 * what is waiting, what installing it would cost, and the two steps - download,
 * then install - with the install confirmed in a dialog (DialogsSystem).
 */
export function SoftwareUpdatesCard({
  busy,
  canStage,
  isAdministrator,
  job,
  loading,
  onDismissJob,
  onRun,
  outcome,
  updates,
}: {
  busy: "" | "stage" | "apply";
  canStage: boolean;
  isAdministrator: boolean;
  job: UpdateJob | null;
  loading: boolean;
  onDismissJob: () => void;
  onRun: (action: "stage" | "apply") => void;
  /** The last run's result, in this card. */
  outcome: { text: string; refused: boolean } | null;
  updates: PackageUpdates | undefined;
}) {
  const [reviewing, setReviewing] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const notNowRef = useRef<HTMLButtonElement>(null);
  const pill = updatesPill(updates, loading);
  const count = updates?.count ?? 0;
  const held = updates?.held_back ?? [];
  const summary = loading
    ? "Checking operating-system updates…"
    : updates?.collected === false
      ? updates.reason || "The update check did not run, so whether updates are waiting is not known."
      : count
        ? `${count} package${count === 1 ? "" : "s"} · ${held.length ? `${held.length} held back` : "none held back"} · a restart is ${updates?.reboot_likelihood ?? "not estimated"}`
        : "No pending operating-system packages were reported.";

  return (
    <>
    <SectionCard
      actions={<StatusPill label={pill.label} reading={pill.reading} tone={pill.tone} />}
      className="sys-updates"
      description={summary}
      footer={(
        <div className="sys-actions-end">
          {updates?.packages?.length ? (
            <Button aria-expanded={reviewing} onClick={() => setReviewing((open) => !open)} variant="secondary">
              {reviewing ? "Hide package updates" : `Review ${count} package update${count === 1 ? "" : "s"}`}
            </Button>
          ) : null}
          <Button
            disabled={!canStage || !updates?.available || Boolean(busy)}
            disabledReason={!canStage && updates?.available ? "Operator access is required to download updates." : undefined}
            onClick={() => onRun("stage")}
            variant={updates?.staged ? "secondary" : "primary"}
          >
            {busy === "stage" ? "Downloading…" : updates?.staged ? "Download again" : "Download updates"}
          </Button>
          {updates?.staged && (
            <Button
              disabled={!isAdministrator || Boolean(busy)}
              disabledReason={!isAdministrator ? "Administrator access is required to install operating-system updates." : undefined}
              onClick={() => setConfirming(true)}
              variant="primary"
            >
              Install staged updates
            </Button>
          )}
        </div>
      )}
      title="Software updates"
    >
      {!loading && count > 0 && (
        <KvGrid
          items={[
            { label: "Downloaded", value: updates?.staged ? formatQuantity(updates.staged_size_bytes, "capacity") : "Not staged", mono: true },
            { label: "Estimated install", value: `About ${updates?.estimated_install_minutes ?? 0} min`, mono: true },
            { label: "Restart likelihood", value: updates?.reboot_likelihood ?? "Not estimated", mono: true },
          ]}
          label="What installing would take"
        />
      )}
      {reviewing && updates?.packages?.length ? (
        <div className="sys-packages">
          <p className="sys-muted">Package scripts may restart affected desktop or background services. Vaelor keeps the device online unless it explicitly reports that a full restart is required.</p>
          <PaginatedItems items={updates.details ?? []} label="Package updates" pageSize={10} render={(item) => (
            <span className="sys-package" key={item.name}>
              <strong>{item.name}</strong>
              <small>{item.installed} → {item.candidate}</small>
              <em>{item.classification}</em>
            </span>
          )} />
        </div>
      ) : null}
      {!loading && held.length > 0 && (
        <Notice heading={`${held.length} held back by the last install`} severity="info">
          <ul className="sys-held">
            {held.map((item) => <li key={item.name}><strong>{item.name}</strong> {item.reason}</li>)}
          </ul>
        </Notice>
      )}
      {job && <UpdateJobStatus job={job} onDismiss={onDismissJob} />}
      {updates?.staged && !busy && (
        <Notice severity="info">Downloaded and staged — not yet installed. Use Install staged updates to apply them.</Notice>
      )}
      {outcome && <Notice severity={outcome.refused ? "danger" : "success"}>{outcome.text}</Notice>}
    </SectionCard>

      {/* Outside the card: a dialog is never a child of a clipped surface. */}
      {confirming && (
        <SystemDialog
          eyebrow="Services may restart"
          eyebrowTone="warning"
          footer={(
            <>
              <Button onClick={() => setConfirming(false)} ref={notNowRef} variant="secondary">Not now</Button>
              <Button onClick={() => { setConfirming(false); onRun("apply"); }} variant="primary">Yes, install updates</Button>
            </>
          )}
          initialFocusRef={notNowRef}
          onClose={() => setConfirming(false)}
          role="alertdialog"
          showClose={false}
          title="Install the downloaded updates now?"
          titleId="install-updates-title"
        >
          <p>Services may restart. Vaelor will tell you if the device itself needs a restart.</p>
        </SystemDialog>
      )}
    </>
  );
}
