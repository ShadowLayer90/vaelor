import { Fragment, useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import { RecordFacts } from "./RecordKit";
import { StatusPill } from "./StatusPill";
import { Button, Card, Notice } from "./ui";
import type { StatusTone } from "./ui/status";
import type { TransportStatus } from "./ConnectionsSettings";
import "../styles/trust.css";

/*
 * Settings › Connections › Trust this Vaelor (VD-212): the household root's
 * fingerprints, one command per computer that installs it only when the
 * fingerprint matches, and a QR code for phones. Every command and the QR come
 * from the controller's one renderer (`vaelor/tls_trust_commands.py`, the same
 * text the installer prints); nothing here builds a command or an address.
 */

/** The public route the root downloads from without signing in (`vaelor/tls_paths.py` ROOT_ROUTE). */
export const ROOT_CERTIFICATE_PATH = "/api/v2/security/trust/root.crt";
const QR_PATH = "/api/v2/security/trust/qr.svg";

export interface TrustCommand { os: string; label: string; where: string; command: string }
export interface TrustCommands {
  address: string; root_url: string;
  fingerprint_sha256: string; fingerprint_sha1: string;
  commands: TrustCommand[]; firefox: string;
}

/** What the console serves while it has not switched yet (`MIGRATING_KINDS` in tls_authority_pki.py). */
const STILL_SERVING: Record<string, string> = {
  "legacy-self-signed": "The console still serves its earlier self-signed certificate",
  "previous-household": "The console still serves a certificate from this Vaelor's previous authority",
};

/** The owner's override when the switch is held (stream A's CLI, run as root on the controller). */
export const PROMOTE_COMMAND = "sudo /opt/vaelor/venv/bin/python -m vaelor.tls_authority promote";

const sentence = (text: string) => text.replace(/\.$/, "");

/**
 * The switch to the household certificate, said only as the authority states
 * it: waiting and why, held and why (with the override), or done (the reason
 * carries the date the replaced certificate is kept until). "none" says
 * nothing - promising an automatic switch that is held was the defect (R21).
 */
function MigrationNote({ migration, kind }: { migration: TransportStatus["migration"]; kind: string }) {
  if (!migration || migration.state === "none") return null;
  const serving = STILL_SERVING[kind] ?? "The console has not switched to this authority yet";
  const reason = migration.reason ? sentence(migration.reason) : "";
  if (migration.state === "waiting") {
    return <p className="trust-note">{`${serving}. It switches over by itself${reason ? `: ${reason}` : ""}.`}</p>;
  }
  if (migration.state === "blocked") {
    return (
      <Notice severity="warning">
        {`${serving}, and the switch is held${reason ? `: ${reason}` : ""}. To switch now - a machine that does not hold this authority yet will need it again - run `}
        <code className="record-mono">{PROMOTE_COMMAND}</code> on this controller.
      </Notice>
    );
  }
  return <p className="trust-note">{`The switch to this authority is done${reason ? `: ${reason}` : ""}.`}</p>;
}

const STATE_PILL: Record<string, { label: string; tone: StatusTone }> = {
  ready: { label: "Ready to trust", tone: "success" },
  "not-set-up": { label: "Not set up yet", tone: "neutral" },
  unreadable: { label: "Could not be read", tone: "warning" },
};

/**
 * A fingerprint that wraps only after a colon, so every pair stays whole and
 * the owner can compare it with a phone's or a terminal's pair by pair.
 */
export function Fingerprint({ value }: { value: string }) {
  const pairs = value.split(":");
  return <span className="trust-fingerprint">{pairs.map((pair, index) => (
    <Fragment key={index}>{pair}{index < pairs.length - 1 && <>:<wbr /></>}</Fragment>
  ))}</span>;
}

function CopyCommand({ entry }: { entry: TrustCommand }) {
  const [copied, setCopied] = useState<"" | "done" | "failed">("");
  const copy = () => {
    void navigator.clipboard.writeText(entry.command)
      .then(() => setCopied("done"), () => setCopied("failed"));
  };
  return (
    <div className="trust-command">
      <div className="trust-command__head">
        <div className="stg-row__text">
          <strong>{entry.label}</strong>
          <small>{entry.where}</small>
        </div>
        <span className="record-card-button">
          <Button aria-label={`Copy the ${entry.label} command`} onClick={copy}>{copied === "done" ? "Copied" : "Copy"}</Button>
        </span>
      </div>
      <pre className="trust-command__code" tabIndex={0}><code>{entry.command}</code></pre>
      {copied === "failed" && <small className="trust-note" role="status">This browser did not allow copying. Select the command and copy it by hand.</small>}
    </div>
  );
}

type QrState = { svg: string } | { error: string } | null;

/**
 * The QR code is fetched, not pointed at by an <img>: a controller that cannot
 * draw it answers with a JSON reason (503 without segno), which an <img> would
 * show as a broken picture. The reason is said in words instead.
 */
function useQrCode(rootUrl: string): QrState {
  const [qr, setQr] = useState<QrState>(null);
  useEffect(() => {
    let live = true;
    setQr(null);
    Promise.resolve()
      .then(() => fetch(QR_PATH, { credentials: "same-origin", cache: "no-store" }))
      .then(async (response) => {
        if (response.ok) {
          const svg = await response.text();
          if (live) setQr({ svg });
          return;
        }
        let reason = `the controller answered ${response.status}`;
        try {
          const envelope = (await response.json()) as { error?: { message?: string } };
          reason = envelope.error?.message ?? reason;
        } catch {
          // Not JSON (a proxy's page): keep the status as the reason.
        }
        throw new Error(reason);
      })
      .catch((error: unknown) => { if (live) setQr({ error: error instanceof Error ? error.message : String(error) }); });
    return () => { live = false; };
  }, [rootUrl]);
  return qr;
}

function PhoneSteps({ rootUrl }: { rootUrl: string }) {
  const qr = useQrCode(rootUrl);
  return (
    <div className="trust-phones">
      {qr && "svg" in qr
        ? <img alt={`QR code that opens ${rootUrl}`} className="trust-phones__qr" height={176} src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(qr.svg)}`} width={176} />
        : <div className="trust-phones__qr trust-phones__qr--empty" role={qr ? "status" : undefined}>
            <small className="trust-note">{qr ? `The QR code could not be drawn: ${qr.error.replace(/\.$/, "")}. Use the link beside it instead.` : "Drawing the QR code."}</small>
          </div>}
      <div className="trust-phones__steps">
        <p className="trust-note">Scan with the phone's camera, or open this link on the phone: <a className="record-mono" href={rootUrl}>{rootUrl}</a>. If your phone reaches this console by another address, open that address with <code className="record-mono">{ROOT_CERTIFICATE_PATH}</code> instead.</p>
        <div>
          <strong>iPhone and iPad</strong>
          <ol>
            <li>Open the link and allow the download.</li>
            <li>Settings › General › VPN &amp; Device Management: choose the Vaelor household authority and Install. Its details show a SHA-256 fingerprint: check it matches the one above.</li>
            <li>Settings › General › About › Certificate Trust Settings: turn on full trust for it. Without this step Safari still warns.</li>
          </ol>
        </div>
        <div>
          <strong>Android</strong>
          <ol>
            <li>Open the link and download the certificate.</li>
            <li>Settings › Security &amp; privacy › More security settings › Encryption &amp; credentials › Install a certificate › CA certificate, then choose the downloaded file.</li>
          </ol>
        </div>
      </div>
    </div>
  );
}

/**
 * The panel. `transport` is the one `/security/transport` read Settings makes;
 * the commands are fetched here only once that read says the authority exists.
 */
export function TrustThisVaelor({ transport }: { transport: TransportStatus | null }) {
  const authority = transport?.authority;
  const ready = authority?.state === "ready";
  const [commands, setCommands] = useState<TrustCommands | null>(null);
  const [commandsFailed, setCommandsFailed] = useState("");

  useEffect(() => {
    if (!ready) return;
    let live = true;
    apiRequest<TrustCommands>("/security/trust/commands")
      .then((value) => { if (live) { setCommands(value); setCommandsFailed(""); } })
      .catch((error: unknown) => { if (live) setCommandsFailed(error instanceof Error ? error.message : "The commands could not be read."); });
    return () => { live = false; };
  }, [ready, authority?.fingerprint_sha256]);

  // Not read is not "not set up": an old controller or a failed read has no authority field at all.
  const pill = !transport ? { label: "Reading", tone: "neutral" as StatusTone }
    : authority ? (STATE_PILL[authority.state] ?? STATE_PILL.unreadable)
    : { label: "Not read", tone: "warning" as StatusTone };

  return (
    <Card
      actions={<StatusPill label={pill.label} tone={pill.tone} />}
      as="section"
      className="trust-panel"
      description="Install this Vaelor's own certificate authority on your computers and phones, so browsers stop warning"
      heading="Trust this Vaelor"
    >
      {transport && !authority && <p className="trust-note">Not read: this controller did not report its certificate authority.</p>}
      {authority?.state === "not-set-up" && (
        <p className="trust-note">This Vaelor's household authority has not been set up yet. The vaelor-tls-authority service creates it; this panel fills in once it has.</p>
      )}
      {authority?.state === "unreadable" && <Notice severity="warning">{authority.message || "The household authority's certificate could not be read."}</Notice>}
      {ready && authority && (
        <>
          <div className="stg-fingerprint">
            <RecordFacts className="trust-fingerprints" facts={[
              { key: "sha256", label: "SHA-256 fingerprint", value: <Fingerprint value={authority.fingerprint_sha256} />, mono: true },
              { key: "sha1", label: "SHA-1 fingerprint", value: <Fingerprint value={authority.fingerprint_sha1} />, mono: true },
            ]} />
            <span className="record-card-button"><a className="ui-button ui-button--secondary" download href={ROOT_CERTIFICATE_PATH}>Download certificate</a></span>
          </div>
          <MigrationNote kind={authority.kind} migration={transport?.migration} />
          {(transport?.names_left_out ?? []).map((left) => (
            <p className="trust-note" key={left.name}>{`${left.name} is not covered by this Vaelor's authority: ${sentence(left.reason)}. Reach the console by its address or a .local name instead.`}</p>
          ))}
          {authority.kind === "custom" && (
            <p className="trust-note">The console serves your own certificate, which Vaelor leaves alone. Trusting this authority helps with Vaelor's own machines; browsers reaching the console follow your certificate.</p>
          )}
          <h3 className="trust-subhead">Computers</h3>
          <p className="trust-note">Each command downloads the authority and installs it only if its SHA-256 fingerprint matches the one above. A mismatch installs nothing.</p>
          {!commands && !commandsFailed && <p className="trust-note">Reading the commands for this address.</p>}
          {commandsFailed && <Notice severity="warning">{`The commands could not be read: ${commandsFailed}`}</Notice>}
          {commands?.commands.map((entry) => <CopyCommand entry={entry} key={entry.os} />)}
          {commands && <p className="trust-note">{commands.firefox}</p>}
          {commands && (
            <>
              <h3 className="trust-subhead">Phones</h3>
              <PhoneSteps rootUrl={commands.root_url} />
            </>
          )}
          <p className="trust-note">Moving this authority to a new controller is a command-line step for now: <code className="record-mono">python -m vaelor.tls_authority export --output FILE</code> on this one, then install the new controller with <code className="record-mono">--import-authority FILE</code>. Without it, a new controller makes a new authority and every device has to trust that one.</p>
        </>
      )}
    </Card>
  );
}
