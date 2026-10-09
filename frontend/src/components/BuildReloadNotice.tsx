import { Icon } from "./Icon";
import { Button, Notice } from "./ui";
import { useBuildFreshness } from "../lib/buildVersion";
import "../styles/notices.css";

/** Reload the page. The default for `reloadPage`, which a test replaces. */
const reloadThisPage = () => globalThis.location.reload();

/**
 * "This page is older than the installed version" (ACC-167), for every screen
 * of the console. The shell renders it once above whatever screen is open, so
 * a page left open across an update says so wherever the owner is reading.
 * It offers a reload and never reloads by itself.
 */
export function BuildReloadNotice({ reloadPage = reloadThisPage }: { reloadPage?: () => void }) {
  const { stale } = useBuildFreshness();
  if (!stale) return null;
  return (
    <Notice severity="info" className="app-reload-notice gn-reload">
      <span className="app-reload-notice__row">
        <Icon className="gn-reload__icon" name="download" size={16} />
        <span className="gn-reload__text">
          Vaelor was updated on this machine after this page was opened, so this page is still running the
          older version. Reload it to use the new one.
        </span>
        <Button variant="secondary" type="button" onClick={reloadPage}>Reload page</Button>
      </span>
    </Notice>
  );
}
