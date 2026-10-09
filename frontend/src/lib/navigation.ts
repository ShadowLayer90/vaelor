export const navigationPages = [
  "overview",
  "kvm",
  "system",
  "workloads",
  "fleet",
  "assistant",
  "ai-chat",
  "activity",
  "admin",
] as const;

export type NavigationPage = (typeof navigationPages)[number];

const navigationPageSet = new Set<string>(navigationPages);

/**
 * Names a user can reasonably type or keep in a bookmark that are not the
 * route segment. `#/settings` in particular is the name shown in the sidebar,
 * so it has to resolve to the Settings workspace instead of silently dropping
 * the reader on Home.
 */
const navigationAliases: Record<string, NavigationPage> = {
  settings: "admin",
  access: "admin",
  home: "overview",
  dashboard: "overview",
  console: "kvm",
  "remote-console": "kvm",
  cooling: "system",
  hardware: "system",
  lighting: "system",
  apps: "workloads",
  models: "workloads",
  cluster: "fleet",
  workers: "fleet",
  chat: "ai-chat",
  audit: "activity",
  operations: "activity",
};

/**
 * Routes reachable by URL that the shell renders outside `NavigationPage`.
 *
 * `#/memory` is appliance-wide: `_list_memories` has no actor filter, so the
 * same store answers the Assistant and AI Chat. The redesign gives it its own
 * rail item in the AI group, after the Assistant (`memoryRailItem`), rather
 * than nesting it under either surface; the chips on those surfaces still
 * open it. Every memory endpoint is administrator-only, so the item is too.
 */
export const standaloneRoutes = ["memory"] as const;

export type StandaloneRoute = (typeof standaloneRoutes)[number];

const standaloneRouteSet = new Set<string>(standaloneRoutes);

/** The rail's Memory item: AI group, after the Assistant; administrators only. */
export const memoryRailItem = {
  route: "memory" as StandaloneRoute,
  label: "Memory",
  descriptor: "What Vaelor remembers, for the Assistant and AI Chat alike",
  hash: "#/memory",
  after: "assistant" as NavigationPage,
} as const;

export function standaloneRouteFromHash(hash: string): StandaloneRoute | null {
  const candidate = hash.replace(/^#\/?/, "").split(/[/?]/, 1)[0].toLowerCase();
  return standaloneRouteSet.has(candidate) ? candidate as StandaloneRoute : null;
}

export type HashResolutionReason = "exact" | "alias" | "unknown";

export interface HashResolution {
  page: NavigationPage;
  reason: HashResolutionReason;
  /** True when the address bar disagrees with the page actually shown. */
  redirect: boolean;
}

/**
 * Resolve a hash to a real destination and say how it got there.
 *
 * Reporting the reason is the point: the caller rewrites the address bar so an
 * alias or an unknown path never leaves the URL pointing somewhere the user is
 * not. The previous silent `?? "overview"` fallback is what made `#/settings`
 * look like a broken Settings link.
 */
export function resolveHash(hash: string): HashResolution {
  const candidate = hash.replace(/^#\/?/, "").split(/[/?]/, 1)[0].toLowerCase();
  if (candidate === "") return { page: "overview", reason: "exact", redirect: false };
  if (navigationPageSet.has(candidate)) {
    return { page: candidate as NavigationPage, reason: "exact", redirect: false };
  }
  const alias = navigationAliases[candidate];
  if (alias) return { page: alias, reason: "alias", redirect: true };
  return { page: "overview", reason: "unknown", redirect: true };
}

export function pageFromHash(hash: string): NavigationPage {
  return resolveHash(hash).page;
}

export function hashForPage(page: NavigationPage) {
  return page === "overview" ? "#/" : `#/${page}`;
}

export function hashTargetsPage(hash: string, page: NavigationPage): boolean {
  const path = hash.replace(/^#\/?/, "").split("?", 1)[0];
  if (page === "overview") return path === "" || path === "overview";
  return path === page || path.startsWith(`${page}/`);
}

/**
 * A backend route ("/workloads/models", the `owner_route` an operation names)
 * as this app's link. The app is hash-routed: a path href reloads the page at
 * that path, and the router - reading only the hash - shows Home (W4d-D24).
 * An address that is already a hash or carries a scheme is left alone.
 */
export function routeHref(route: string): string {
  const trimmed = route.trim();
  if (trimmed.startsWith("#") || /^[a-z][a-z0-9+.-]*:/i.test(trimmed)) return trimmed;
  return "#/" + trimmed.replace(/^\/+/, "");
}
