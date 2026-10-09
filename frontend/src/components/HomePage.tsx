import type { Device, Health, Metrics, Session } from "../types";
import type { FleetSummary } from "./fleetTypes";
import type { MachineProfile } from "../lib/machine";
import { hashForPage, type NavigationPage } from "../lib/navigation";
import { useHomeSummary, type Read } from "../hooks/useHomeSummary";
import { HomeAttention, HomeMachines, HomeServing, HomeStatTiles } from "./HomePanels";
import { PageHeader } from "./ui";

/** "Good morning." / "Good afternoon." / "Good evening." by this browser's clock. */
function greeting(now: Date): string {
  const hour = now.getHours();
  if (hour < 5 || hour >= 18) return "Good evening.";
  return hour < 12 ? "Good morning." : "Good afternoon.";
}

/**
 * Home is the redesign's Main board and nothing else (VD-200, the design
 * guide's placement map): the greeting and the page's two doors, four stat
 * tiles, What's serving, Needs your attention and Machines. What Home used to
 * show below that lives elsewhere now: the live readings, This machine and the
 * enclosure panels on System › Compute, the power actions behind System's
 * Power button, storage on System › Hardware and services, recent activity on
 * Activity. The task shortcuts were retired for the header doors, Search and
 * the panels' own links.
 */
export function HomePage({
  cluster,
  device,
  health,
  live,
  machine,
  metrics,
  noun,
  pageAllowed,
  session,
}: {
  /** The shell's one `/cluster` read, shared rather than read again. */
  cluster: Read<FleetSummary>;
  device: Device | null;
  health: Health;
  /** False when there is no fresh sample: the tiles then say "Not read". */
  live: boolean;
  machine: MachineProfile;
  metrics: Metrics;
  noun: string;
  pageAllowed: (page: NavigationPage) => boolean;
  session: Session;
}) {
  const summary = useHomeSummary(session.user.role, true, cluster);
  return (
    <div className="home-page">
      <PageHeader
        actions={<>
          {pageAllowed("activity") && <a className="ui-button ui-button--secondary" href={hashForPage("activity")}>Open Activity</a>}
          {pageAllowed("workloads") && <a className="ui-button ui-button--primary" href={hashForPage("workloads")}>Deploy an app</a>}
        </>}
        as="p"
        subtitle={`Everything this ${noun} runs, at a glance.`}
        title={greeting(new Date())}
      />
      <HomeStatTiles device={device} live={live} machine={machine} metrics={metrics} summary={summary} />
      <div className="home-pair">
        <HomeServing summary={summary} />
        <HomeAttention health={health} summary={summary} />
      </div>
      <HomeMachines device={device} live={live} machine={machine} metrics={metrics} summary={summary} />
    </div>
  );
}
