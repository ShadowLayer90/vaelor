import { useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import { brand } from "../lib/brand";
import type { TransportStatus } from "./ConnectionsSettings";
import { ProductMark } from "./ProductMark";
import { TrustThisVaelor } from "./TrustThisVaelor";
import { Button, Notice } from "./ui";
import "../styles/auth.css";
import "../styles/trust.css";

/*
 * First-run setup's last step (VD-133 1.5.5 #2, owner 2026-10-20): right after
 * the administrator account is created, before Home, the owner is shown how to
 * trust this Vaelor - the same Trust this Vaelor panel Settings › Connections
 * carries, not a copy of it. The first visit already clicked past the
 * browser's warning once to create the account, so this is the one moment
 * every owner is guaranteed to pass through with the console open.
 */

/** Where the same steps stay reachable after a skip (Administration.tsx). */
export const TRUST_STEPS_LOCATION = "Settings › Connections › Trust this Vaelor";

type TransportRead = { state: "reading" } | { state: "read"; transport: TransportStatus } | { state: "failed"; message: string };

export function FirstRunTrust({ onContinue }: { onContinue: () => void }) {
  const [read, setRead] = useState<TransportRead>({ state: "reading" });

  useEffect(() => {
    let live = true;
    apiRequest<TransportStatus>("/security/transport")
      .then((transport) => { if (live) setRead({ state: "read", transport }); })
      .catch((error: unknown) => {
        if (live) setRead({ state: "failed", message: error instanceof Error ? error.message : "The certificate authority could not be read." });
      });
    return () => { live = false; };
  }, []);

  return (
    <main className="frt-shell">
      <div className="frt-frame">
        <header className="sgn-brand">
          <span aria-hidden="true" className="sgn-mark"><ProductMark /></span>
          <span><strong>{brand.name}</strong><small>Infrastructure command</small></span>
        </header>
        <div className="frt-heading">
          <span className="sgn-eyebrow sgn-eyebrow--accent">First-run setup · Last step</span>
          <h1 id="first-run-trust-title">Trust this Vaelor on your devices</h1>
          <p>
            Your browser warned you before this page because it does not know this Vaelor's own certificate authority yet.
            Until a device trusts it, the browser warns on every visit, and when Vaelor renews its certificate, pages you
            have open can lose their connection. Trust it once on each computer and phone you use.
          </p>
          <p>
            Check the SHA-256 fingerprint below against the one on this Vaelor's own screen (its text login screen) or
            the installer's last screen: a page reached past a browser warning cannot vouch for itself.
          </p>
        </div>
        {read.state === "reading" && <p className="trust-note" role="status">Reading this Vaelor's certificate authority.</p>}
        {read.state === "failed" && (
          <Notice severity="warning">{`The certificate authority could not be read: ${read.message.replace(/\.$/, "")}. You can continue and find the steps later under ${TRUST_STEPS_LOCATION}.`}</Notice>
        )}
        {read.state === "read" && (
          <TrustThisVaelor
            description="Check the fingerprint, then run the command for this computer or scan the code with your phone"
            heading="Household certificate authority"
            transport={read.transport}
          />
        )}
        <div className="frt-actions">
          <Button onClick={onContinue} variant="primary">Done, continue</Button>
          <Button onClick={onContinue}>Skip for now</Button>
        </div>
        <p className="trust-note">{`You can come back to these steps at any time under ${TRUST_STEPS_LOCATION}.`}</p>
      </div>
    </main>
  );
}
