import type { ReactNode } from "react";
import { AppsIconTile, AppsTags } from "./appsKit";
import type { IconName } from "./Icon";
import { StatusPill } from "./StatusPill";
import { Button, Card, LoadingLines, Notice } from "./ui";
import type { ManagedInventory } from "./WorkloadManager";
import { appRowDetail, appStatePill } from "./manageModel";
import type { Session } from "../types";
import type { ApplicationFeatures } from "./workloads-types";
import type { DockerReadiness } from "./WorkloadsReadiness";
import "../styles/apps-install.css";

/*
 * The Install tab's five doors and what follows them (VD-200, the Apps and
 * AppsInstallStates boards): each card keeps its button visible and says why
 * it cannot be used beside it; then the messages; then Running now.
 */

type Role = Session["user"]["role"];

const OPERATOR_REQUIRED = "Operator access is required to install an app.";
const DOCKER_NOT_READY = "Docker is not ready on this node yet. Finish the setup check above first.";

export type InstallDoor = "catalog" | "model" | "copilot" | "custom" | "researched";

interface DoorCard {
  door: InstallDoor;
  eyebrow: string;
  title: string;
  text: string;
  tags: string[];
  button: string;
  icon: IconName;
  primary?: boolean;
  reason?: string;
}

/** Why each door is closed, or undefined when it is open. Exported for tests. */
export function installDoorReasons(role: Role, readiness: DockerReadiness, features: ApplicationFeatures | null, featuresLoading: boolean) {
  const viewer = role === "viewer";
  const dockerReason = viewer ? OPERATOR_REQUIRED : readiness.composeReady ? undefined : DOCKER_NOT_READY;
  return {
    catalog: dockerReason,
    // It reuses Browse blueprints' sentence: the same two gates close it.
    model: dockerReason,
    copilot: viewer ? "Operator access is required to set up the Assistant." : undefined,
    custom: role !== "administrator"
      ? "Administrator access is required to import a Docker stack."
      : readiness.composeReady ? undefined : DOCKER_NOT_READY,
    researched: viewer
      ? "Describing a custom application needs an operator or administrator."
      : !features
        ? (featuresLoading ? "Checking whether assisted research is available." : "Whether assisted research is available could not be read. Reload to check again.")
        : features.research
          ? undefined
          : "Assisted research needs a working Assistant model. Install the Assistant model and it turns on automatically - no setting to flip.",
  } satisfies Record<InstallDoor, string | undefined>;
}

/**
 * VD-049 / VD-201: the Assistant's model is a fixed Vaelor install on this
 * machine and never a hosted one; an added connection is AI Chat's. The board
 * drew "a local or hosted model for the Assistant"; the ledger outranks it.
 */
export const ASSISTANT_DOOR_TEXT = "Install the model Vaelor runs the Assistant on, or connect a model for AI Chat. Nothing changes without your approval.";

export function WorkloadsInstallCards({
  reasons,
  onOpen,
}: {
  reasons: Record<InstallDoor, string | undefined>;
  onOpen: (door: InstallDoor) => void;
}) {
  const cards: DoorCard[] = [
    { door: "catalog", eyebrow: "Reviewed blueprints", title: "Blueprint apps", text: "Apps with a known, reviewed configuration that Vaelor installs and looks after.", tags: ["Known configuration", "Reviewed template", "Managed lifecycle"], button: "Browse blueprints", icon: "apps", primary: true },
    { door: "model", eyebrow: "Run local AI", title: "Install an AI model", text: "Pick a model that fits this machine. Memory is checked before anything downloads.", tags: ["Recommendations", "Memory check", "Private by default"], button: "Choose a model", icon: "chat" },
    { door: "copilot", eyebrow: "Set up Assistant", title: "Improve the Assistant", text: ASSISTANT_DOOR_TEXT, tags: ["Hardware-aware", "Runs on this machine", "Approval required"], button: "Set up Assistant", icon: "assistant" },
    { door: "custom", eyebrow: "Advanced, guarded", title: "Import a Docker stack", text: "Paste a compose file. Vaelor scans it against policy and backs up before it applies.", tags: ["Policy scan", "Automatic backup", "Recoverable removal"], button: "Open composer", icon: "server" },
    { door: "researched", eyebrow: "Assisted research", title: "Custom application", text: "Describe an app. Vaelor researches it, cites its sources and asks before installing.", tags: ["Guided request", "Cited research", "Approval first"], button: "Describe an app", icon: "activity" },
  ];
  return (
    <div className="apps-doors">
      {cards.map((card) => (
        <section aria-labelledby={`apps-door-${card.door}`} className="apps-door" key={card.door}>
          <div className="apps-door__top">
            <AppsIconTile accent={card.primary} name={card.icon} />
            <span className="apps-door__eyebrow">{card.eyebrow}</span>
          </div>
          <div className="apps-door__text">
            <h2 id={`apps-door-${card.door}`}>{card.title}</h2>
            <p>{card.text}</p>
          </div>
          <AppsTags items={card.tags} label={`${card.title} features`} />
          <div className="apps-door__action">
            <Button disabledReason={reasons[card.door]} onClick={() => onOpen(card.door)} variant={card.primary ? "primary" : "secondary"}>{card.button}</Button>
          </div>
        </section>
      ))}
    </div>
  );
}

export function WorkloadsMessages({
  busy,
  loading,
  loadError,
  notice,
  noticeRefused,
  savedResearch,
  onDiscardResearch,
  onResumeResearch,
  onRetry,
}: {
  busy: boolean;
  loading: boolean;
  loadError: string;
  notice: string;
  noticeRefused: boolean;
  savedResearch?: { requestSummary?: string };
  onDiscardResearch: () => void;
  onResumeResearch: () => void;
  onRetry: () => void;
}) {
  if (!savedResearch && !notice && !loadError) return null;
  return (
    <div className="apps-messages">
      {savedResearch && (
        <Notice className="apps-message apps-message--research" severity="info">
          <span className="apps-message__text"><strong>Saved application research</strong> {savedResearch.requestSummary || "A durable application request is waiting to be resumed."}</span>
          <span className="apps-message__actions">
            <Button disabled={busy} onClick={onResumeResearch}>Resume application research</Button>
            <Button disabled={busy} onClick={onDiscardResearch} variant="danger">Discard saved research</Button>
          </span>
        </Notice>
      )}
      {notice && <Notice className="apps-message" severity={noticeRefused ? "danger" : "info"}>{notice}</Notice>}
      {loadError && (
        <Notice className="apps-message" severity="warning">
          <span className="apps-message__text">{loadError} Existing tools remain available.</span>
          <span className="apps-message__actions">
            <Button busy={loading} onClick={onRetry}>{loading ? "Checking…" : "Try again"}</Button>
          </span>
        </Notice>
      )}
    </div>
  );
}

/** The running apps from the inventory, with their state pills; Manage all switches tabs. */
export function WorkloadsRunningNow({
  inventory,
  inventoryState,
  manageCount,
  onManage,
  onRetry,
}: {
  inventory: ManagedInventory | null;
  inventoryState: "loading" | "ok" | "error";
  manageCount: number | null;
  onManage: () => void;
  onRetry: () => void;
}) {
  const running = (inventory?.apps ?? []).filter((app) => app.running);
  let body: ReactNode;
  if (!inventory) {
    body = inventoryState === "error"
      ? (
        <div className="apps-running__unread">
          <span>The installed apps could not be read, so what is running is not known.</span>
          <Button onClick={onRetry}>Try again</Button>
        </div>
      )
      : <div className="apps-running__unread"><LoadingLines label="Reading the installed apps" /></div>;
  } else if (!running.length) {
    body = <div className="apps-running__unread"><span>No app is running on this machine.</span></div>;
  } else {
    body = (
      <ul className="apps-running__list">
        {running.map((app) => {
          // The same pill the Manage list draws for this app (one state, one colour).
          const state = appStatePill(app);
          return (
            <li className="apps-running__row" key={app.id}>
              <div className="apps-running__text">
                <strong>{app.name}</strong>
                <span>{appRowDetail(app)}</span>
              </div>
              <StatusPill description={state.description} label={state.label} tone={state.tone} />
            </li>
          );
        })}
      </ul>
    );
  }
  return (
    <Card
      actions={<Button className="apps-link" onClick={onManage} variant="quiet">{manageCount === null ? "Manage all" : `Manage all ${manageCount}`}</Button>}
      as="section"
      className="apps-running"
      flush
      heading="Running now"
    >
      {body}
    </Card>
  );
}
