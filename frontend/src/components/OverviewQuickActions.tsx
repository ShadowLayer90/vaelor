import { Icon, type IconName } from "./Icon";
import { destinations } from "../lib/destinations";
import type { MachineClass } from "../lib/machine";
import { hashForPage, type NavigationPage } from "../lib/navigation";

/**
 * Home used to report status and stop there: nothing on it led to a task, so a
 * new operator had to guess which sidebar word owned the job they came to do.
 * These are the four jobs people actually arrive with, each pointing at the
 * destination that owns it and each named with that destination's canonical
 * name so the landing page confirms the click.
 *
 * They are real anchors rather than scripted buttons: the hash router already
 * listens for `hashchange`, so they work with middle-click, "open in new tab",
 * and keyboard activation without any extra handler.
 */
interface EntryPoint {
  page: NavigationPage;
  job: string;
  detail: string;
  icon: IconName;
}

/**
 * The first job named a fan policy. On a machine with no readable fan it sent
 * the reader to a tab whose cooling controls do not exist, under a fan glyph,
 * so the one job Home offered first was the one it could not do.
 */
const entryPointsFor = (machineClass: MachineClass): EntryPoint[] => [
  machineClass === "pi-appliance"
    ? {
      page: "system",
      job: "Check cooling",
      detail: "Temperatures, fan policy, and the hardware services behind them.",
      icon: "fan",
    }
    : {
      page: "system",
      job: "Check hardware",
      detail: "Temperatures, storage, network, and the services behind them.",
      icon: "nvme",
    },
  {
    page: "workloads",
    job: "Install an app",
    detail: "Pick a reviewed app or a local AI model and let Vaelor set it up.",
    icon: "database",
  },
  {
    page: "assistant",
    job: "Ask the assistant",
    detail: "Describe a problem in your own words and get an evidence-backed answer.",
    icon: "memory",
  },
  {
    page: "activity",
    job: "View activity",
    detail: "Every authenticated change, with the operation record behind it.",
    icon: "activity",
  },
];

export function OverviewQuickActions({
  machineClass,
  pageAllowed,
}: {
  machineClass: MachineClass;
  pageAllowed: (page: NavigationPage) => boolean;
}) {
  const available = entryPointsFor(machineClass).filter((entry) => pageAllowed(entry.page));
  if (!available.length) return null;

  return (
    <section className="home-launcher" aria-labelledby="home-launcher-heading">
      <div className="section-heading">
        <div>
          <h2 id="home-launcher-heading">Start a task</h2>
          <p>The common jobs, and where each one lives.</p>
        </div>
      </div>
      <ul className="home-launcher__grid">
        {available.map((entry) => (
          <li key={entry.page}>
            <a className="home-launcher__link" href={hashForPage(entry.page)}>
              <span className="home-launcher__icon" aria-hidden="true">
                <Icon name={entry.icon} size={19} />
              </span>
              <strong>{entry.job}</strong>
              <small>{entry.detail}</small>
              <span className="home-launcher__target">
                {destinations[entry.page].name}
                <Icon name="chevron" size={13} />
              </span>
            </a>
          </li>
        ))}
      </ul>
    </section>
  );
}
