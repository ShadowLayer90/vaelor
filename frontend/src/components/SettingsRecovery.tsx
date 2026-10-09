import { useRef, useState } from "react";
import { apiRequest, downloadApiRequest } from "../lib/api";
import type { Session } from "../types";
import { BackupSettings } from "./BackupSettings";
import { RecordDialog } from "./RecordKit";
import { ReadPill } from "./SettingsAccounts";
import { Button, Card, Input, Notice } from "./ui";
import "../styles/settings.css";

export interface RecoveryStatus {
  confirmation: string;
  erases: string[];
  retains: string[];
}

// Mirrors RecoveryStatus for the full REMOVE VAELOR uninstall; the API returns
// `removes`/`retains` scope arrays instead of factory-reset's `erases`/`retains`.
export interface RemoveVaelorStatus {
  confirmation: string;
  removes: string[];
  retains: string[];
  // The OS-stack scope sentence, authored once server-side
  // (appliance_recovery.UNINSTALL_OS_SCOPE_SENTENCE) and rendered here rather
  // than retyped, so the card cannot drift from what the button removes.
  scope_summary: string;
}

export interface PortableStateStatus {
  staged: boolean;
  confirmation: string;
  plan: {
    id: string;
    source_version?: string;
    created_at: number;
    expires_at: number;
    approved_at?: number;
  } | null;
  scope: { includes: string[]; excludes: string[] };
  last_result?: {
    ok: boolean;
    completed_at: number;
    imported?: number;
    error?: string;
  } | null;
}

type Report = (text: string, failed?: boolean) => void;

/** The page's long-running actions, named for the sentence that locks the others while one runs (S-Y8). */
const BUSY_NAMES: Record<string, string> = {
  "portable-export": "the encrypted export",
  reset: "the reset",
  remove: "the removal",
};

/** Why a control is locked while another of the page's actions runs; undefined when none (or this one) is running. */
function lockedBy(busy: string, own: string): string | undefined {
  if (!busy || busy === own) return undefined;
  return `Wait for ${BUSY_NAMES[busy] ?? "the running action"} to finish.`;
}

/** "the OS stack Vaelor installed - Docker ..." as a list line that starts with a capital. */
function sentenceCase(text: string) {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}

/**
 * The import half of Move this Vaelor, as a dialog: choose the archive and its
 * passphrase and verify it, then type the phrase to replace this machine's
 * state. Nothing changes until the second step is approved.
 */
function ImportDialog({ onClose, onReport, portable, refresh, session, setPortable }: {
  onClose: () => void;
  onReport: Report;
  portable: PortableStateStatus | null;
  refresh: () => Promise<void>;
  session: Session;
  setPortable: (status: PortableStateStatus) => void;
}) {
  const [archive, setArchive] = useState<File | null>(null);
  const [passphrase, setPassphrase] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const cancel = useRef<HTMLButtonElement>(null);
  const staged = Boolean(portable?.staged);
  const fail = (caught: unknown, fallback: string) => setError(caught instanceof Error && caught.message ? caught.message : fallback);

  const stage = async () => {
    if (!archive) return;
    setBusy("stage"); setError("");
    const form = new FormData();
    form.set("archive", archive);
    form.set("passphrase", passphrase);
    try {
      setPortable(await apiRequest<PortableStateStatus>("/admin/portable-state/import/stage", { method: "POST", body: form }, session.csrf_token));
    } catch (caught) {
      fail(caught, "The transfer archive could not be verified.");
    } finally { setBusy(""); }
  };

  const cancelStaged = async () => {
    setBusy("cancel"); setError("");
    try {
      setPortable(await apiRequest<PortableStateStatus>("/admin/portable-state/import/stage", { method: "DELETE" }, session.csrf_token));
      onReport("Staged transfer removed. No appliance data changed.");
      onClose();
    } catch (caught) {
      fail(caught, "The staged transfer could not be removed.");
    } finally { setBusy(""); }
  };

  const apply = async () => {
    setBusy("apply"); setError("");
    try {
      await apiRequest("/admin/portable-state/import", { method: "POST", body: JSON.stringify({ confirmation }) }, session.csrf_token);
      onReport("Import accepted. Vaelor will replace portable state, restart services, and end this sign-in.");
      onClose();
      await refresh();
    } catch (caught) {
      fail(caught, "The portable import was not accepted.");
    } finally { setBusy(""); }
  };

  const approved = Boolean(portable?.plan?.approved_at);
  return (
    <RecordDialog
      actions={staged ? <>
        <Button disabled={Boolean(busy) || approved} onClick={() => void cancelStaged()} ref={cancel} type="button">Cancel transfer</Button>
        {/*
          * S-Y2: the typed phrase gates this whatever else is true. It once
          * dropped the gate while busy or once approved, because the reason was
          * the only thing disabling it.
          */}
        <Button
          busy={busy === "apply"}
          disabled={Boolean(busy) || approved || confirmation !== portable?.confirmation}
          disabledReason={busy
            ? undefined
            : approved
              ? "This transfer is already approved and being applied."
              : confirmation !== portable?.confirmation ? "Type the phrase above to enable this." : undefined}
          onClick={() => void apply()}
          type="button"
          variant="danger"
        >
          Replace portable state
        </Button>
      </> : <>
        <Button disabled={Boolean(busy)} onClick={onClose} ref={cancel} type="button">Cancel</Button>
        <Button
          busy={busy === "stage"}
          disabledReason={!archive ? "Choose the transfer archive first." : passphrase.length < 16 ? "Enter its passphrase of at least 16 characters." : undefined}
          onClick={() => void stage()}
          type="button"
          variant="primary"
        >
          Review transfer
        </Button>
      </>}
      error={error}
      alert={staged}
      eyebrow={staged ? "Import on this machine · step 2 of 2" : "Import on this machine · step 1 of 2"}
      eyebrowTone={staged ? "danger" : "plain"}
      initialFocusRef={cancel}
      onClose={() => { if (!busy) onClose(); }}
      title={staged ? `Vaelor ${portable?.plan?.source_version ?? "compatible archive"}` : "Choose a Vaelor transfer archive"}
    >
      {staged ? (
        <>
          <Notice severity="info">Transfer verified. Review the replacement scope before approving it.</Notice>
          <p>Approval expires {portable?.plan ? new Date(portable.plan.expires_at * 1000).toLocaleTimeString() : "soon"}. Import ends active sessions.</p>
          <Input autoComplete="off" id="import-confirmation" label={<span>Type <strong>{portable?.confirmation}</strong> to confirm</span>} onChange={(event) => setConfirmation(event.target.value)} value={confirmation} />
        </>
      ) : (
        <>
          <p>Verify first; replacement starts only after exact approval.</p>
          <label className="ui-field">
            <span className="ui-field__label">Vaelor transfer archive</span>
            <input accept=".vaelor,application/octet-stream" className="ui-control ui-control--file-picker" onChange={(event) => setArchive(event.target.files?.[0] ?? null)} type="file" />
          </label>
          <Input autoComplete="current-password" id="import-passphrase" label="Transfer passphrase" minLength={16} onChange={(event) => setPassphrase(event.target.value)} type="password" value={passphrase} />
        </>
      )}
    </RecordDialog>
  );
}

/** Move this Vaelor: what travels in the encrypted archive, what stays, and the export and import. */
function MoveThisVaelor({ busy, onReport, portable, refresh, session, setBusy, setPortable }: {
  busy: string;
  onReport: Report;
  portable: PortableStateStatus | null;
  refresh: () => Promise<void>;
  session: Session;
  setBusy: (value: string) => void;
  setPortable: (status: PortableStateStatus) => void;
}) {
  const [passphrase, setPassphrase] = useState("");
  const [again, setAgain] = useState("");
  const [importing, setImporting] = useState(false);
  const mismatch = Boolean(again && passphrase !== again);
  const exportState = async () => {
    setBusy("portable-export"); onReport("");
    try {
      await downloadApiRequest("/admin/portable-state/export", "vaelor-state.vaelor", { method: "POST", body: JSON.stringify({ passphrase }) }, session.csrf_token);
      setPassphrase(""); setAgain("");
      onReport("Encrypted Vaelor state downloaded. Store the file and passphrase separately.");
    } catch (error) {
      onReport(error instanceof Error && error.message ? error.message : "Portable state could not be exported.", true);
    } finally { setBusy(""); }
  };
  const locked = lockedBy(busy, "portable-export");
  const exportBlocked = passphrase.length < 16
    ? "Use a transfer passphrase of at least 16 characters."
    : passphrase !== again ? "Type the same passphrase twice." : undefined;
  return (
    <Card
      actions={<ReadPill notLabel="Transfer staged" ok={Boolean(portable) && !portable?.staged} okLabel="Ready" read={Boolean(portable)} />}
      as="section"
      className="stg-move"
      heading={<><span className="record-eyebrow">Encrypted state transfer</span>Move this Vaelor</>}
    >
      <div className="stg-move__grid">
        <div>
          <span className="stg-list-label">Moves with the archive</span>
          {portable ? <ul className="stg-list stg-list--plus">{portable.scope.includes.map((item) => <li key={item}>{sentenceCase(item)}</li>)}</ul> : <p className="ui-muted ui-small">Not read yet.</p>}
        </div>
        <div>
          <span className="stg-list-label">Stays on this hardware</span>
          {portable ? <ul className="stg-list stg-list--dot">{portable.scope.excludes.map((item) => <li key={item}>{sentenceCase(item)}</li>)}</ul> : <p className="ui-muted ui-small">Not read yet.</p>}
        </div>
        <div className="stg-move__form record-card-button">
          <Input autoComplete="new-password" id="export-passphrase" label="Transfer passphrase" minLength={16} onChange={(event) => setPassphrase(event.target.value)} type="password" value={passphrase} />
          <Input autoComplete="new-password" error={mismatch ? "The transfer passphrases do not match." : undefined} id="export-passphrase-again" label="Confirm passphrase" minLength={16} onChange={(event) => setAgain(event.target.value)} type="password" value={again} />
          <div className="stg-actions">
            <Button busy={busy === "portable-export"} disabledReason={busy === "portable-export" ? undefined : locked ?? exportBlocked} onClick={() => void exportState()} type="button" variant="primary">
              {busy === "portable-export" ? "Encrypting…" : "Export encrypted state"}
            </Button>
            <Button disabled={Boolean(busy)} disabledReason={busy === "portable-export" ? undefined : locked} onClick={() => setImporting(true)} type="button">Import on this machine</Button>
          </div>
        </div>
      </div>
      <p className="ui-muted ui-small stg-move__note">Import ends active sessions. Provider keys, SSH credentials, models, container volumes, TLS keys, and hardware identity are never transferred.</p>
      {importing && <ImportDialog onClose={() => setImporting(false)} onReport={onReport} portable={portable} refresh={refresh} session={session} setPortable={setPortable} />}
    </Card>
  );
}

/** Reset or Remove: what goes, what stays, and the typed phrase that unlocks the final action. */
function LastResort({ actionLabel, busyLabel, confirmation, description, eraseLabel, erases, id, locked, onApply, retains, title, working }: {
  actionLabel: string;
  busyLabel: string;
  confirmation: string | null;
  /** What this does, in a sentence, under the title (the explanation the old Administration page carried). */
  description: string;
  eraseLabel: string;
  erases: string[] | null;
  id: string;
  /** Why it is locked while another of the page's actions runs. */
  locked?: string;
  onApply: (typed: string) => void;
  retains: string[] | null;
  title: string;
  working: boolean;
}) {
  const [typed, setTyped] = useState("");
  const blocked = locked ?? (confirmation === null
    ? "Vaelor has not read what this removes yet."
    : typed !== confirmation ? `Type ${confirmation} above to enable this.` : undefined);
  return (
    <Card as="section" className="stg-last-resort" description={description} heading={<><span className="record-eyebrow record-eyebrow--danger">Can't be undone</span>{title}</>}>
      <div className="stg-scope">
        <div>
          <span className="stg-list-label">{eraseLabel}</span>
          {erases ? <ul className="stg-list">{erases.map((item) => <li key={item}>{item}</li>)}</ul> : <p className="ui-muted ui-small">Not read yet.</p>}
        </div>
        <div>
          <span className="stg-list-label">Will remain</span>
          {retains ? <ul className="stg-list stg-list--muted">{retains.map((item) => <li key={item}>{item}</li>)}</ul> : <p className="ui-muted ui-small">Not read yet.</p>}
        </div>
      </div>
      <Input autoComplete="off" id={id} label={`Type ${confirmation ?? "the phrase"} to confirm`} onChange={(event) => setTyped(event.target.value)} placeholder={confirmation ?? undefined} value={typed} />
      <div className="stg-actions stg-actions--end record-card-button">
        <Button busy={working} disabledReason={working ? undefined : blocked} onClick={() => onApply(typed)} type="button" variant="danger">
          {working ? busyLabel : actionLabel}
        </Button>
      </div>
    </Card>
  );
}

/**
 * Settings › Recovery (VD-200, the Recovery and SettingsBackup boards), in the
 * board's order: Move this Vaelor, Back up this Vaelor, then Reset and Remove
 * side by side. Update Vaelor moved to System › Hardware and services.
 */
export function SettingsRecovery({ onReport, portable, recovery, refresh, removeVaelor, session, setPortable }: {
  onReport: Report;
  portable: PortableStateStatus | null;
  recovery: RecoveryStatus | null;
  refresh: () => Promise<void>;
  removeVaelor: RemoveVaelorStatus | null;
  session: Session;
  setPortable: (status: PortableStateStatus) => void;
}) {
  const [busy, setBusy] = useState("");
  const apply = async (kind: "reset" | "remove", typed: string) => {
    setBusy(kind); onReport("");
    try {
      await apiRequest(
        kind === "reset" ? "/admin/recovery/factory-reset" : "/admin/recovery/remove-vaelor",
        { method: "POST", body: JSON.stringify({ confirmation: typed }) },
        session.csrf_token,
      );
      onReport(kind === "reset"
        ? "Factory reset accepted. This session will end and first-run setup will return."
        : "Removal started. Every Vaelor service, its data, and the OS stack it installed are being removed; this session will end and the appliance will be gone.");
    } catch (error) {
      onReport(error instanceof Error && error.message ? error.message : kind === "reset" ? "Factory reset was not accepted." : "Removal was not accepted.", true);
    } finally { setBusy(""); }
  };
  const removes = removeVaelor
    ? [...removeVaelor.removes, ...(removeVaelor.scope_summary ? [sentenceCase(removeVaelor.scope_summary)] : [])]
    : null;
  return (
    <div className="stg-stack">
      <MoveThisVaelor busy={busy} onReport={onReport} portable={portable} refresh={refresh} session={session} setBusy={setBusy} setPortable={setPortable} />
      <BackupSettings session={session} />
      <div className="stg-two-col stg-two-col--even">
        <LastResort
          actionLabel="Approve and reset"
          busyLabel="Starting reset…"
          confirmation={recovery?.confirmation ?? null}
          description="Use this only when the control plane cannot be repaired. Resetting returns the app to first-run setup."
          eraseLabel="Will be permanently erased"
          erases={recovery?.erases ?? null}
          id="reset-confirmation"
          locked={lockedBy(busy, "reset")}
          onApply={(typed) => void apply("reset", typed)}
          retains={recovery?.retains ?? null}
          title="Reset Vaelor"
          working={busy === "reset"}
        />
        <LastResort
          actionLabel="Approve and remove Vaelor"
          busyLabel="Starting removal…"
          confirmation={removeVaelor?.confirmation ?? null}
          description="This does not return Vaelor to first-run setup; it removes Vaelor entirely, with its data and the OS stack it installed. It cannot be undone."
          eraseLabel="Will be permanently removed"
          erases={removes}
          id="remove-confirmation"
          locked={lockedBy(busy, "remove")}
          onApply={(typed) => void apply("remove", typed)}
          retains={removeVaelor?.retains ?? null}
          title="Remove Vaelor"
          working={busy === "remove"}
        />
      </div>
    </div>
  );
}
