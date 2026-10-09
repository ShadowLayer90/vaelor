import { useState } from "react";
import { apiRequest } from "../lib/api";
import type { Session } from "../types";
import { StatusPill } from "./StatusPill";
import { Button, Card, Input, Notice, type NoticeSeverity } from "./ui";
import "../styles/settings.css";

/**
 * Settings › Accounts › Your two-factor sign-in (VD-200, the DialogsRecords
 * board draws it in four states): optional, set up with one code, enabled,
 * and the failures. Its outcomes are said inside the card, beside the code.
 */
export function SettingsTwoFactor({ enabled, onChanged, onSecret, secret, session }: {
  /** Whether two-factor is on for this account, as `/auth/totp` last said; null before it was read. */
  enabled: boolean | null;
  onChanged: (enabled: boolean) => void;
  /**
   * The setup secret lives with Settings, not this card (S-Y7): the card
   * unmounts on every tab change, and a secret held here was lost mid-setup
   * while the authenticator app already had it.
   */
  onSecret: (secret: string) => void;
  secret: string;
  session: Session;
}) {
  const setSecret = onSecret;
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [outcome, setOutcome] = useState<{ text: string; severity: NoticeSeverity } | null>(null);

  const run = async (action: () => Promise<void>, failure: string) => {
    setBusy(true);
    setOutcome(null);
    try {
      await action();
    } catch (error) {
      setOutcome({ text: error instanceof Error && error.message ? error.message : failure, severity: "danger" });
    } finally {
      setBusy(false);
    }
  };

  const start = () => run(async () => {
    const setup = await apiRequest<{ secret: string }>("/auth/totp/setup", { method: "POST", body: "{}" }, session.csrf_token);
    setSecret(setup.secret);
    setCode("");
    setOutcome({ text: "Add this Vaelor node to your authenticator app, then enter its code.", severity: "info" });
  }, "Two-factor setup could not start.");

  const confirm = () => run(async () => {
    await apiRequest("/auth/totp/confirm", { method: "POST", body: JSON.stringify({ code }) }, session.csrf_token);
    setSecret("");
    setCode("");
    setOutcome({ text: "Two-factor authentication is now required for your account.", severity: "success" });
    onChanged(true);
  }, "The authenticator code was not accepted.");

  const disable = () => run(async () => {
    await apiRequest("/auth/totp", { method: "DELETE", body: JSON.stringify({ code }) }, session.csrf_token);
    setCode("");
    setOutcome({ text: "Two-factor authentication disabled.", severity: "success" });
    onChanged(false);
  }, "The authenticator code was not accepted.");

  const codeField = (id: string, label: string) => (
    <Input
      autoComplete="one-time-code"
      className="stg-code"
      id={id}
      inputMode="numeric"
      label={label}
      maxLength={6}
      onChange={(event) => setCode(event.target.value.replace(/\D/g, ""))}
      value={code}
    />
  );

  return (
    <Card
      as="section"
      className="stg-two-factor"
      description={secret ? "Step 2: confirm one code" : "Add an authenticator code after your password."}
      heading="Your two-factor sign-in"
    >
      <div className="stg-two-factor__body record-card-button">
        {!secret && (enabled === null
          ? <StatusPill label="Reading" tone="neutral" />
          : <StatusPill label={enabled ? "Enabled" : "Optional"} tone={enabled ? "success" : "neutral"} />)}
        {outcome && <Notice severity={outcome.severity}>{outcome.text}</Notice>}
        {enabled === false && !secret && (
          <span><Button busy={busy} onClick={() => void start()} variant="primary">Set up authenticator</Button></span>
        )}
        {secret && (
          <>
            <div className="stg-secret">
              <small className="ui-muted">Manual setup key</small>
              <code className="record-mono">{secret}</code>
              <span><Button className="record-ghost" onClick={() => void navigator.clipboard.writeText(secret)} variant="quiet">Copy</Button></span>
            </div>
            {codeField("totp-code", "Six-digit verification code")}
            <span>
              <Button
                busy={busy}
                disabledReason={code.length === 6 ? undefined : "Enter the six-digit code from your authenticator app to continue."}
                onClick={() => void confirm()}
                variant="primary"
              >
                Verify and enable
              </Button>
            </span>
          </>
        )}
        {enabled === true && !secret && (
          <>
            {codeField("totp-code-current", "Current authenticator code")}
            <span>
              <Button
                busy={busy}
                className="record-danger-outline"
                disabledReason={code.length === 6 ? undefined : "Enter a current code to turn two-factor off."}
                onClick={() => void disable()}
              >
                Verify and disable
              </Button>
            </span>
          </>
        )}
      </div>
    </Card>
  );
}
