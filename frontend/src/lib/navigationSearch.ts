import { DEPLOYMENTS_MODELS_HREF } from "./clusterSections";
import { destinations } from "./destinations";
import type { MachineProfile } from "./machine";
import { hashForPage, memoryRailItem, type NavigationPage } from "./navigation";
import { SYSTEM_SECTION_LABELS } from "./systemSections";

/**
 * The places Ctrl+K can take the reader: the rail's destinations (Memory
 * included, for an administrator), the tabs and views an address can open,
 * and the sections inside Settings (VD-200, the redesign's Search). Front end
 * only - it searches names, never data, and asks no route.
 *
 * Only tabs an address really opens are listed: System's, Settings',
 * Activity's tabs and Remote console's views follow the hash
 * (`#/system/lighting`, `#/admin/recovery`, `#/activity/audit`,
 * `#/kvm/remote-login`); Cluster's and Apps and AI's follow a query
 * (`?cluster=fleet`). The Assistant's tabs are not addressable, so its
 * sections are found through the Assistant itself.
 */
export interface SearchEntry {
  id: string;
  label: string;
  /** Where it lives, e.g. "Settings" for a Settings section. */
  context: string;
  /** Words it also answers to. */
  keywords: string[];
  /** The address: a hash, or a query followed by a hash. */
  href: string;
  page: NavigationPage;
}

const settingsSections: Array<[string, string, string[]]> = [
  ["accounts", "Accounts", ["Add an account", "Local accounts", "Your two-factor sign-in", "users", "authenticator"]],
  ["sessions", "Sessions", ["Signed-in devices", "sign out"]],
  ["connections", "Connections", ["Secure remote access", "Encrypted connections", "External API access", "provider", "API keys"]],
  // Update Vaelor moved to System > Hardware and services (VD-200); its words live there.
  ["recovery", "Recovery", ["Move this Vaelor", "Back up this Vaelor", "Reset Vaelor", "Remove Vaelor", "backup", "restore"]],
];

const clusterTabs: Array<[string, string, string[]]> = [
  ["setup", "Setup", ["GPU memory and cluster link", "add a machine", "controller"]],
  ["fleet", "Fleet", ["machines", "workers"]],
  ["deployments", "Deployments", ["apps", "agents", "deployed"]],
  ["performance", "Performance", ["dashboard", "charts"]],
  ["activity", "Activity", ["alerts", "alert rules"]],
  ["agents", "Agents & tools", ["MCP", "skills"]],
];

const activityTabs: Array<[string, string, string[]]> = [
  ["audit", "Security audit", ["audit trail", "sign-ins", "who changed"]],
  ["restore-points", "Restore points", ["checkpoints", "recovery points", "app backups"]],
];

const consoleViews: Array<[string, string, string[]]> = [
  ["remote-login", "Remote Login", ["RDP", "remote desktop", "browser desktop"]],
  ["hardware", "Physical KVM", ["KVM", "keyboard video mouse", "HDMI capture"]],
];

/**
 * `memoryAllowed`: Memory is a standalone route, not a `NavigationPage`, so
 * the page gate cannot answer for it; every memory endpoint is
 * administrator-only, and so is its entry.
 */
export function searchEntries(
  machine: MachineProfile,
  allowed: (page: NavigationPage) => boolean,
  memoryAllowed = false,
): SearchEntry[] {
  const entries: SearchEntry[] = [];
  const add = (entry: SearchEntry) => { if (allowed(entry.page)) entries.push(entry); };
  for (const destination of Object.values(destinations)) {
    add({
      id: `page-${destination.page}`,
      label: destination.name,
      context: "Page",
      keywords: [destination.descriptor],
      href: hashForPage(destination.page),
      page: destination.page,
    });
    // Memory sits where the rail draws it, after the page it follows, not last among the pages.
    if (memoryAllowed && destination.page === memoryRailItem.after) {
      entries.push({ id: "page-memory", label: memoryRailItem.label, context: "Page", keywords: [memoryRailItem.descriptor, "remembered", "facts"], href: memoryRailItem.hash, page: memoryRailItem.after });
    }
  }
  const appliance = machine.machine_class === "pi-appliance";
  // Compute is every machine's section now (VD-200); Cooling is the Pi's alone.
  const systemTabs: Array<[string, string, string[], boolean]> = [
    ["cooling", SYSTEM_SECTION_LABELS.cooling, ["fans", "fan policy", "temperature"], appliance],
    ["compute", SYSTEM_SECTION_LABELS.compute, ["processor", "graphics", "neural processor", "memory", "live readings"], true],
    ["lighting", SYSTEM_SECTION_LABELS.lighting, ["LEDs", "colour"], machine.capabilities.case_lighting.available],
    ["hardware", SYSTEM_SECTION_LABELS.hardware, ["storage", "network", "services", "Update Vaelor", "update", "software updates", "reinstall"], true],
  ];
  for (const [id, label, keywords, shown] of systemTabs) {
    if (shown) add({ id: `system-${id}`, label, context: destinations.system.name, keywords, href: `#/system/${id}`, page: "system" });
  }
  // Two controls with places of their own: the Power menu is in System's
  // header, and Tune memory is on Compute's Memory in use reading.
  add({ id: "system-power", label: "Power", context: destinations.system.name, keywords: ["restart service", "reboot", "shut down", "power off"], href: "#/system/compute", page: "system" });
  add({ id: "system-tune-memory", label: "Tune memory", context: `${destinations.system.name} › Compute`, keywords: ["memory optimizer", "free memory"], href: "#/system/compute", page: "system" });
  for (const [id, label, keywords] of settingsSections) {
    add({ id: `admin-${id}`, label, context: destinations.admin.name, keywords, href: `#/admin/${id}`, page: "admin" });
  }
  for (const [id, label, keywords] of activityTabs) {
    add({ id: `activity-${id}`, label, context: destinations.activity.name, keywords, href: `#/activity/${id}`, page: "activity" });
  }
  for (const [id, label, keywords] of consoleViews) {
    add({ id: `kvm-${id}`, label, context: destinations.kvm.name, keywords, href: `#/kvm/${id}`, page: "kvm" });
  }
  for (const [id, label, keywords] of clusterTabs) {
    add({ id: `fleet-${id}`, label, context: destinations.fleet.name, keywords, href: `?cluster=${id}#/fleet`, page: "fleet" });
  }
  // The LLM Server and its API keys are drawn only under Deployments' Models filter (S-D1).
  add({ id: "fleet-llm-server", label: "LLM Server", context: `${destinations.fleet.name} › Deployments`, keywords: ["API keys", "endpoints", "models", "serving"], href: DEPLOYMENTS_MODELS_HREF, page: "fleet" });
  add({ id: "workloads-install", label: "Install", context: destinations.workloads.name, keywords: ["blueprints", "AI model", "deploy an app"], href: "?workloads=install#/workloads", page: "workloads" });
  add({ id: "workloads-manage", label: "Manage", context: destinations.workloads.name, keywords: ["installed apps", "logs", "files"], href: "?workloads=manage#/workloads", page: "workloads" });
  return entries;
}

/** How many results a typed query lists; an empty query lists every page, Settings included. */
export const SEARCH_RESULT_LIMIT = 8;

/** Entries whose name, place or keywords contain every word typed, best first. */
export function searchMatches(entries: SearchEntry[], query: string): SearchEntry[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return entries.filter((entry) => entry.context === "Page");
  return entries
    .map((entry) => {
      const label = entry.label.toLowerCase();
      const haystack = [label, entry.context.toLowerCase(), ...entry.keywords.map((word) => word.toLowerCase())].join(" ");
      if (!words.every((word) => haystack.includes(word))) return null;
      const score = label.startsWith(words[0]) ? 0 : label.includes(words[0]) ? 1 : 2;
      return { entry, score };
    })
    .filter((match): match is { entry: SearchEntry; score: number } => match !== null)
    .sort((a, b) => a.score - b.score)
    .map((match) => match.entry);
}

/**
 * Go to an entry. A hash alone is an ordinary hash change. An address with a
 * query (a Cluster or Apps and AI tab) is pushed and announced as a history
 * step, which is how those pages already read their tab, so it opens without
 * reloading the console.
 */
export function goToEntry(entry: SearchEntry): void {
  // A hash alone keeps whatever query the address already has, so a jump from
  // Cluster's `?cluster=fleet` to System carried it along. Replace both.
  if (entry.href.startsWith("#") && !window.location.search) {
    window.location.hash = entry.href;
    return;
  }
  window.history.pushState(null, "", entry.href.startsWith("#") ? `${window.location.pathname}${entry.href}` : entry.href);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

/** Whether a key press landed in something the reader types into. */
export function isEditable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || target.getAttribute("contenteditable") === "true"
    || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement
    || (target instanceof HTMLInputElement && !["button", "checkbox", "radio", "submit", "reset", "range", "color", "file"].includes(target.type));
}
