import { Button, Notice } from "./ui";
import { clockTime, useCertificateRenewal, type RenewalView } from "../lib/certificateRenewal";
import "../styles/notices.css";

/** Reload the page. The default for `reloadPage`, which a test replaces. */
const reloadThisPage = () => globalThis.location.reload();

/** Where the trust steps live; the same place the command search sends "certificate". */
export const TRUST_PANEL_PATH = "Settings › Connections › Trust this Vaelor";

/**
 * The console's certificate is about to be renewed, or was renewed under this
 * page (VD-133 1.5.5 #1). Rendered once by the shell above every screen, like
 * the newer-build notice beside it. It offers a reload and never reloads by
 * itself. `view` is for tests; the shell lets it poll.
 */
export function CertificateRenewalNotice({
  reloadPage = reloadThisPage,
  view,
}: { reloadPage?: () => void; view?: RenewalView | null }) {
  const polled = useCertificateRenewal();
  const shown = view === undefined ? polled : view;
  if (!shown) return null;
  if (shown.kind === "announced") {
    return (
      <Notice severity="warning" standing className="app-reload-notice gn-reload"
        heading={`This console's certificate is renewed at about ${clockTime(shown.at)}`}>
        <span className="app-reload-notice__row">
          <span className="gn-reload__text">
            Vaelor renews it because {shown.reason}. If this browser trusts the Vaelor household
            authority, nothing changes. If you opened the console by accepting a certificate warning
            instead, this browser will refuse the renewed certificate: this page stops updating, and
            reloading shows the warning again. To avoid that, trust the household authority before
            then under {TRUST_PANEL_PATH}.
          </span>
        </span>
      </Notice>
    );
  }
  return (
    <Notice severity="danger" className="app-reload-notice gn-reload"
      heading="This page lost its connection when the certificate was renewed">
      <span className="app-reload-notice__row">
        <span className="gn-reload__text">
          The console&apos;s certificate was renewed at about {clockTime(shown.at)}, and this page has not
          reached Vaelor since. This usually means this browser accepted the old certificate&apos;s warning
          and refuses the renewed one. Reload the page and accept the warning again; to stop renewals
          interrupting you, trust the Vaelor household authority after the reload, under {TRUST_PANEL_PATH}.
        </span>
        <Button variant="secondary" type="button" onClick={reloadPage}>Reload page</Button>
      </span>
    </Notice>
  );
}
