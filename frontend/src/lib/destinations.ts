import { brand } from "./brand";
import type { MachineClass } from "./machine";
import { navigationPages, type NavigationPage } from "./navigation";

/**
 * One canonical name per destination.
 *
 * Alpha 11 gave seven of the nine destinations two or three competing names:
 * the sidebar said "Assistant", the page heading said "Ask Vaelor", and the
 * eyebrow said "Assistant / Live operator". A beginner could not tell whether
 * those were the same place. Every surface that names a destination — the
 * sidebar, the page heading, the browser title, and any cross-reference — now
 * reads `name` from this one table.
 *
 * `descriptor` is deliberately a sentence-shaped support line, never a noun
 * phrase that could be mistaken for a second name for the same page. It carries
 * no closing full stop: several callers extend it with a page-specific clause,
 * and a descriptor that punctuated itself produced a visible ".." on those.
 */
export interface Destination {
  page: NavigationPage;
  name: string;
  descriptor: string;
}

export const destinations: Record<NavigationPage, Destination> = {
  overview: {
    page: "overview",
    name: "Home",
    descriptor: "Live status of this appliance and the quickest way into a task",
  },
  kvm: {
    page: "kvm",
    name: "Remote console",
    descriptor: "Reach the screen and keyboard of this machine from here",
  },
  system: {
    page: "system",
    name: "System",
    descriptor: "Cooling, case lighting, and appliance services",
  },
  workloads: {
    page: "workloads",
    name: "Apps and AI",
    descriptor: "Install and manage container apps and local AI models",
  },
  fleet: {
    page: "fleet",
    name: "Cluster",
    /*
     * A worker is a separate computer from the one reading this, and the
     * enrollment path is fingerprint-pinned SSH to a Linux host — not to a
     * particular board. Naming the board excluded every worker a Pi head
     * controller can legitimately enroll.
     */
    descriptor: "Enroll Linux workers and place work across them",
  },
  /*
   * Vaelor deliberately ships two AI destinations, and "which one do I use?"
   * had no answer on screen. Each descriptor now states exactly one capability
   * and exactly one limit, and the two limits are each other's capability, so
   * the split explains itself without a mode switch or a routing heuristic.
   */
  assistant: {
    page: "assistant",
    name: "Assistant",
    descriptor: "Ask about this machine. Answers come from live readings — it cannot read your files",
  },
  "ai-chat": {
    page: "ai-chat",
    name: "AI Chat",
    descriptor: "Ask about your documents, or anything else. It reads files you add — it cannot see this machine's readings",
  },
  activity: {
    page: "activity",
    name: "Activity",
    descriptor: "Who changed what, and how every operation finished",
  },
  admin: {
    page: "admin",
    name: "Settings",
    descriptor: "Accounts, access, sessions, and encrypted connections",
  },
};

export const destinationList: Destination[] = navigationPages.map(
  (page) => destinations[page],
);

export function destinationName(page: NavigationPage): string {
  return destinations[page].name;
}

/**
 * Two descriptors name hardware rather than a job, and are wrong on anything
 * that is not a Pi in a Pironman case: Home calls the computer an "appliance",
 * System advertises "Cooling, case lighting, and appliance services" — two of
 * which do not exist on a workstation — and Cluster says "Raspberry Pi
 * workers" when the enrollment path is an SSH connection to a Linux host.
 *
 * The *names* are untouched: "System" and "Cluster" are right on every
 * machine, and one canonical name per destination is a contract the whole shell
 * is built on. Only the support line varies, and only where the shipped
 * wording states a fact that is false on the machine reading it.
 */
const machineDescriptors: Partial<Record<NavigationPage, Partial<Record<MachineClass, string>>>> = {
  overview: {
    workstation: "Live status of this machine and the quickest way into a task",
    generic: "Live status of this machine and the quickest way into a task",
  },
  system: {
    workstation: "Processor, storage, network, and the services behind them",
    generic: "Processor, storage, network, and the services behind them",
  },
};

export function destinationDescriptorFor(
  page: NavigationPage,
  machineClass: MachineClass,
): string {
  return machineDescriptors[page]?.[machineClass] ?? destinations[page].descriptor;
}

/**
 * WCAG 2.4.2 Page Titled: the canonical name comes first so a tab strip, a
 * history entry, and a screen-reader page announcement all identify the page
 * before they identify the product.
 */
export function documentTitleForPage(page: NavigationPage): string {
  return `${destinationName(page)} · ${brand.name}`;
}
