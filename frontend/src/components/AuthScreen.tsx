import { useState, type FormEvent } from "react";
import { Icon } from "./Icon";
import { ProductMark } from "./ProductMark";
import { brand } from "../lib/brand";
import { Button, Input } from "./ui";

export function AuthScreen({
  mode,
  error,
  busy,
  totpRequired,
  onSubmit,
}: {
  mode: "bootstrap" | "login";
  error: string;
  busy: boolean;
  totpRequired: boolean;
  onSubmit: (username: string, password: string, totpCode: string) => Promise<void>;
}) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [localError, setLocalError] = useState("");
  const [totpCode, setTotpCode] = useState("");

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setLocalError("");
    if (mode === "bootstrap" && password !== confirmation) {
      setLocalError("The passphrases do not match.");
      return;
    }
    await onSubmit(username, password, totpCode);
  };

  return (
    <main className="auth-shell">
      <section className="auth-intro" aria-label={brand.controlPlane}>
        <header className="auth-intro__brand">
          <div className="auth-intro__mark"><ProductMark /></div>
          <div>
            <strong>{brand.name}</strong>
            <span>Infrastructure command</span>
          </div>
        </header>

        <div className="auth-intro__content">
          <div className="auth-intro__statement">
            <p className="auth-intro__overline">The system behind your systems</p>
            <h1>One command surface.<br />Every machine in reach.</h1>
          </div>
          <p className="auth-intro__copy">
            Operate hardware, applications, private AI, and remote nodes from
            one local authority.
          </p>

          <div className="auth-convergence" aria-label="Vaelor control domains">
            <div className="auth-convergence__header">
              <span>Control domains</span>
              <span>Local authority</span>
            </div>
            <div className="auth-convergence__map">
              <div className="auth-convergence__sources">
                <span>Hardware</span>
                <span>Workloads</span>
                <span>Intelligence</span>
                <span>Fleet</span>
              </div>
              <div className="auth-convergence__rail" aria-hidden="true">
                <i /><i /><i /><i />
              </div>
              <div className="auth-convergence__core">
                <ProductMark />
                <span>Vaelor core</span>
                <small>Observe · act · recover</small>
              </div>
            </div>
          </div>
        </div>

        <footer className="auth-intro__footer">
          <span>Runs where your systems live</span>
          <span>No cloud dependency</span>
        </footer>
      </section>

      <section className="auth-access" aria-labelledby="auth-access-title">
        <div className="auth-access__frame">
          <div className="auth-access__status">
            <span aria-hidden="true" />
            <strong>{mode === "bootstrap" ? "Ready to commission" : "Node online"}</strong>
            <small>Secure local endpoint</small>
          </div>
          <div className="auth-card">
            <div className="auth-card__heading">
              <span>{mode === "bootstrap" ? "First-run setup" : "Authorized operators"}</span>
              <h2 id="auth-access-title">
                {mode === "bootstrap" ? "Commission this node" : "Sign in to Vaelor"}
              </h2>
              <p>
                {mode === "bootstrap"
                  ? "Create the administrator account that will own this control plane."
                  : "Use the account stored locally on this Vaelor node."}
              </p>
            </div>
            <form onSubmit={submit}>
              <Input autoComplete="username" id="username" label="Username" maxLength={64} onChange={(event) => setUsername(event.target.value)} required value={username} />
              <Input autoComplete={mode === "bootstrap" ? "new-password" : "current-password"} id="password" label={mode === "bootstrap" ? "Administrator passphrase" : "Password"} minLength={mode === "bootstrap" ? 12 : undefined} onChange={(event) => setPassword(event.target.value)} required type="password" value={password} />
              {mode === "bootstrap" && (
                <>
                  <Input autoComplete="new-password" hint="Use at least 12 characters." id="confirmation" label="Confirm passphrase" minLength={12} onChange={(event) => setConfirmation(event.target.value)} required type="password" value={confirmation} />
                </>
              )}
              {mode === "login" && totpRequired && (
                <>
                  <Input autoComplete="one-time-code" hint="Enter the current six-digit code." id="totp-code" inputMode="numeric" label="Authenticator code" maxLength={6} minLength={6} onChange={(event) => setTotpCode(event.target.value.replace(/\D/g, ""))} pattern="[0-9]{6}" required value={totpCode} />
                </>
              )}
              {(localError || error) && (
                <p className="form-error" role="alert">
                  {localError || error}
                </p>
              )}
              <Button className="ui-button--wide" disabled={busy} type="submit" variant="primary">{busy ? "Authenticating…" : mode === "bootstrap" ? "Commission Vaelor" : "Open control plane"}</Button>
            </form>
          </div>
          <div className="auth-card__security">
            <Icon name="lock" size={14} />
            <span>
              <strong>Local authentication</strong>
              Credentials stay on this node. No cloud account is required.
            </span>
          </div>
        </div>
      </section>
    </main>
  );
}
