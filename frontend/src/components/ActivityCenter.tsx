import { useEffect, useState } from "react";
import { destinations } from "../lib/destinations";
import { TopbarPageActions, usePagePlace } from "../lib/topbarSlot";
import type { Session } from "../types";
import { ActivityAudit } from "./ActivityAudit";
import { ActivityOperations } from "./ActivityOperations";
import { Icon } from "./Icon";
import { RecoveryPointList } from "./RecoveryPointList";
import { StatusPill } from "./StatusPill";
import { Button, Card, TabSet } from "./ui";
import "../styles/activity.css";

export { readOperationSummary } from "./ActivityOperations";

/** Activity's three tabs (VD-200 decision 2), each with its own address. */
export type ActivitySection = "operations" | "audit" | "restore-points";

const SECTIONS: ReadonlyArray<{ id: ActivitySection; label: string }> = [
  { id: "operations", label: "Operations" },
  { id: "audit", label: "Security audit" },
  { id: "restore-points", label: "Restore points" },
];

/** The top bar breadcrumb's second part (the Activity boards). */
const ACTIVITY_PLACE = ["Everything that changed, and who changed it"] as const;

/** Which tab an address opens: `#/activity/audit`, `#/activity/restore-points`, else Operations. */
export function activitySectionFromHash(hash: string): ActivitySection {
  const section = hash.match(/^#\/activity\/(audit|restore-points)(?:[/?].*)?$/)?.[1];
  return section === "audit" || section === "restore-points" ? section : "operations";
}

export function activityHashForSection(section: ActivitySection): string {
  return section === "operations" ? "#/activity" : `#/activity/${section}`;
}

/**
 * Activity › Restore points: the managed apps' configuration checkpoints. The
 * list itself (verify, restore, delete and their dialogs) is the app
 * manager's, placed here as it is; this card only frames it and reloads it.
 */
function ActivityRestorePoints({ session }: { session: Session }) {
  const [reload, setReload] = useState(0);
  return (
    <Card
      actions={<Button onClick={() => setReload((value) => value + 1)} type="button"><Icon name="refresh" size={16} />Reload</Button>}
      as="section"
      className="acty-restore-points"
      description="Inspect, verify, restore, or remove configuration restore points for managed apps."
      heading="Recovery checkpoints"
    >
      <RecoveryPointList refreshSignal={reload} session={session} showReload={false} />
    </Card>
  );
}

/**
 * Activity (VD-200, the Activity, ActivityOperation, ActivityAudit and
 * ActivityCheckpoints boards): who changed what, and where each operation
 * ran, in three tabs - Operations, Security audit and Restore points.
 */
export function ActivityCenter({ session }: { session: Session }) {
  const [section, setSection] = useState<ActivitySection>(() => activitySectionFromHash(window.location.hash));
  useEffect(() => {
    const follow = () => setSection(activitySectionFromHash(window.location.hash));
    window.addEventListener("hashchange", follow);
    window.addEventListener("popstate", follow);
    return () => {
      window.removeEventListener("hashchange", follow);
      window.removeEventListener("popstate", follow);
    };
  }, []);
  // The Activity boards' breadcrumb is the same on all three tabs.
  usePagePlace(ACTIVITY_PLACE);
  const choose = (next: string) => {
    const target = next as ActivitySection;
    setSection(target);
    if (activitySectionFromHash(window.location.hash) !== target) {
      window.history.pushState(null, "", activityHashForSection(target));
    }
  };
  return (
    <div className="acty-page">
      {/* The page header as the board draws it; the name alone is the level-one heading, so a
          screen reader's page heading is "Activity", not the sentence under it. */}
      <div className="ui-page-header">
        <div className="record-page-title">
          <h1 className="ui-page-header__title">{destinations.activity.name}</h1>
          <p className="ui-page-header__title record-page-title__line">See who changed what, and jump to where each operation ran.</p>
        </div>
        <TopbarPageActions><StatusPill label="Auditing active" status="healthy" /></TopbarPageActions>
      </div>
      <TabSet items={SECTIONS} label="Activity sections" onSelect={choose} selectedId={section}>
        {section === "operations" && <ActivityOperations session={session} />}
        {section === "audit" && <ActivityAudit />}
        {section === "restore-points" && <ActivityRestorePoints session={session} />}
      </TabSet>
    </div>
  );
}
