import type { ReactNode } from "react";
import "../styles/apps-setup.css";
import { joinClassNames } from "./ui/field";

/*
 * Small pieces the catalogs, the Assistant setup and the approvals share
 * (VD-200, the AppsCatalog, AppsModelCatalog, AppsAssistantSetup and
 * AppsWebResearch boards). They live beside those components rather than in
 * the area kit, which another stream owns; Claude consolidates after the merge.
 */

/**
 * The boards' tinted banner with its action at the right ("The blueprint list
 * could not be read ... Try again", "Port 8080 is in use ... Use port 8082").
 * A standing state is a status, never an alert: it is announced politely.
 */
export function AppsBanner({
  action,
  children,
  className,
  tone,
}: {
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  tone: "warning" | "info" | "danger";
}) {
  return (
    <div
      aria-atomic="true"
      aria-live="polite"
      className={joinClassNames("apps-banner", "apps-banner--" + tone, className)}
      data-severity={tone}
      role="status"
    >
      <span className="apps-banner__text">{children}</span>
      {action && <span className="apps-banner__actions">{action}</span>}
    </div>
  );
}

