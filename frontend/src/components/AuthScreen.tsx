import { useState, type FormEvent } from "react";
import { Icon, type IconName } from "./Icon";
import { ProductMark } from "./ProductMark";
import { brand } from "../lib/brand";
import { Button, Input, Notice } from "./ui";
import "../styles/auth.css";

/** The shortest administrator passphrase first-run setup accepts (the server holds the same rule). */
const MIN_PASSPHRASE = 12;

/** The four things Vaelor reaches, as the sign-in page draws them. */
const DOMAINS: ReadonlyArray<{ label: string; icon: IconName }> = [
  { label: "Hardware", icon: "system" },
  { label: "Workloads", icon: "apps" },
  { label: "Intelligence", icon: "settings" },
  { label: "Fleet", icon: "cluster" },
];

/**
 * Sign in and first-run Commission (VD-200: the SignIn, GlobalTwoFactor and
 * GlobalCommission boards). The two-factor code box appears only after the
 * password was accepted for an account with two-factor on, and the username
 * and password stay filled, so a wrong code costs one field, not three.
 */
export function AuthScreen({
  mode,
  error,
  errorCode = "",
  errorStatus = 0,
  busy,
  totpRequired,
  onSubmit,
}: {
  mode: "bootstrap" | "login";
  error: string;
  /** The server's error code for `error` ("totp_required", "already_configured", ...), when it gave one. */
  errorCode?: string;
  /** The HTTP status of the refusal: 429 is the sign-in limit. */
  errorStatus?: number;
  busy: boolean;
  totpRequired: boolean;
  onSubmit: (username: string, password: string, totpCode: string) => Promise<void>;
}) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [fieldError, setFieldError] = useState<{ field: "username" | "password" | "confirmation" | "totp"; text: string } | null>(null);
  const [totpCode, setTotpCode] = useState("");
  // The code sent with the last attempt: the first `totp_required` answer is
  // the server asking for a code, not a wrong one.
  const [sentCode, setSentCode] = useState("");
  const commissioning = mode === "bootstrap";

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setFieldError(null);
    /*
     * S-Y5: the form is `noValidate` so these sentences replace the browser's
     * bubbles, which means the form itself must refuse what the browser would
     * have: an empty attempt reaches the server and counts toward its
     * sign-in limit.
     */
    if (!username.trim()) {
      setFieldError({ field: "username", text: "Enter a username." });
      return;
    }
    if (!commissioning && !password) {
      setFieldError({ field: "password", text: "Enter your password." });
      return;
    }
    if (!commissioning && totpRequired && !/^\d{6}$/.test(totpCode)) {
      setFieldError({ field: "totp", text: "Enter all six digits of the code showing now." });
      return;
    }
    if (commissioning && password.length < MIN_PASSPHRASE) {
      setFieldError({ field: "password", text: `Use at least ${MIN_PASSPHRASE} characters.` });
      return;
    }
    if (commissioning && password !== confirmation) {
      setFieldError({ field: "confirmation", text: "The passphrases do not match." });
      return;
    }
    setSentCode(totpCode);
    await onSubmit(username, password, totpCode);
  };

  const askingForCode = totpRequired && errorCode === "totp_required" && sentCode === "";
  const codeRefused = totpRequired && errorCode === "totp_required" && sentCode !== "";
  const limited = errorStatus === 429;
  const shownError = askingForCode ? "" : error;
  const errorFollowUp = codeRefused
    ? "The code changes every 30 seconds. Use the one showing now."
    : limited && totpRequired
      ? "Failed codes count toward the same sign-in limit as failed passwords."
      : errorCode === "already_configured"
        ? "Someone finished setup first. Reload the page to reach Sign in."
        : "";

  return (
    <main className="sgn-shell">
      <section className="sgn-intro" aria-label={brand.controlPlane}>
        <header className="sgn-brand">
          <span className="sgn-mark" aria-hidden="true"><ProductMark /></span>
          <span><strong>{brand.name}</strong><small>Infrastructure command</small></span>
        </header>
        <div className="sgn-intro__content">
          <p className="sgn-overline">The system behind your systems</p>
          <h1>One command surface.<br />Every machine in reach.</h1>
          <p className="sgn-copy">Operate hardware, applications, private AI and other machines from one local authority.</p>
          <div className="sgn-domains" aria-label="Vaelor control domains" role="group">
            <ul>
              {DOMAINS.map((domain) => (
                <li key={domain.label}><span aria-hidden="true" className="sgn-domain__icon"><Icon name={domain.icon} size={18} /></span>{domain.label}</li>
              ))}
            </ul>
            <span aria-hidden="true" className="sgn-domains__link" />
            <div className="sgn-core"><strong>Vaelor core</strong><small>Observe · act · recover</small></div>
          </div>
        </div>
        <footer className="sgn-foot">
          <span>Runs where your systems live</span>
          <span>No cloud dependency</span>
        </footer>
      </section>

      <section className="sgn-access" aria-labelledby="auth-access-title">
        <div className="sgn-frame">
          <div className="sgn-status">
            <span aria-hidden="true" className={commissioning ? "sgn-dot sgn-dot--info" : "sgn-dot"} />
            <strong>{commissioning ? "Ready to commission" : "Node online"}</strong>
            <small>· Secure local endpoint</small>
          </div>
          <div className="sgn-card">
            <div className="sgn-card__heading">
              <span className={commissioning ? "sgn-eyebrow sgn-eyebrow--accent" : "sgn-eyebrow"}>{commissioning ? "First-run setup" : "Authorized operators"}</span>
              <h2 id="auth-access-title">{commissioning ? "Commission this node" : "Sign in to Vaelor"}</h2>
              <p>{commissioning ? "Create the administrator account that will own this control plane." : "Use the account stored on this machine."}</p>
            </div>
            <form noValidate onSubmit={submit}>
              <Input autoComplete="username" error={fieldError?.field === "username" ? fieldError.text : undefined} id="username" label="Username" maxLength={64} onChange={(event) => setUsername(event.target.value)} value={username} />
              <Input
                autoComplete={commissioning ? "new-password" : "current-password"}
                error={fieldError?.field === "password" ? fieldError.text : undefined}
                id="password"
                label={commissioning ? "Administrator passphrase" : "Password"}
                onChange={(event) => setPassword(event.target.value)}
                type="password"
                value={password}
              />
              {commissioning && (
                <Input
                  autoComplete="new-password"
                  error={fieldError?.field === "confirmation" ? fieldError.text : undefined}
                  hint={fieldError ? undefined : `Use at least ${MIN_PASSPHRASE} characters.`}
                  id="confirmation"
                  label="Confirm passphrase"
                  onChange={(event) => setConfirmation(event.target.value)}
                  type="password"
                  value={confirmation}
                />
              )}
              {!commissioning && totpRequired && (
                <div className="sgn-code-step">
                  <div className="sgn-code-step__head">
                    <span aria-hidden="true" className="ui-row__icon ui-row__icon--accent"><Icon name="shield" size={18} /></span>
                    <span>
                      <strong>Two-factor sign-in is on for this account</strong>
                      <small>Enter the six-digit code from your authenticator app.</small>
                    </span>
                  </div>
                  <Input
                    aria-invalid={codeRefused || undefined}
                    autoComplete="one-time-code"
                    autoFocus
                    className="sgn-code"
                    disabled={busy}
                    error={fieldError?.field === "totp" ? fieldError.text : undefined}
                    hint={fieldError?.field === "totp" ? undefined : "Enter the current six-digit code."}
                    id="totp-code"
                    inputMode="numeric"
                    label="Authenticator code"
                    maxLength={6}
                    onChange={(event) => setTotpCode(event.target.value.replace(/\D/g, ""))}
                    pattern="[0-9]{6}"
                    value={totpCode}
                  />
                </div>
              )}
              {shownError && (
                <div className="sgn-error">
                  <Notice severity="danger">{shownError}</Notice>
                  {errorFollowUp && <small>{errorFollowUp}</small>}
                </div>
              )}
              <Button className="ui-button--wide sgn-submit" disabled={busy} type="submit" variant="primary">
                {busy ? "Authenticating…" : commissioning ? "Commission Vaelor" : "Open control plane"}
              </Button>
            </form>
          </div>
          {commissioning && (
            <div className="sgn-after">
              <span className="sgn-eyebrow">After you commission</span>
              <ol>
                <li>This account becomes the administrator. Nobody else can sign in until you add them in Settings.</li>
                <li>You are signed in with it straight away.</li>
              </ol>
            </div>
          )}
          <p className="sgn-security">
            <Icon name="key" size={16} />
            Credentials stay on this machine. No cloud account is needed.
          </p>
        </div>
      </section>
    </main>
  );
}
