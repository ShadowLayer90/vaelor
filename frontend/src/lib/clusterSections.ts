/**
 * The Cluster page's six tabs (VD-200, the Cluster boards), and where the
 * open one is kept: the `cluster` query parameter, so a tab is linkable and
 * survives Back and Forward. Setup is the default and is kept as no
 * parameter at all.
 *
 * The labels are the tab strip's words and the top bar breadcrumb's second
 * part ("Cluster / Fleet"); one table serves both so the two cannot drift.
 */

export type ClusterSection = "setup" | "fleet" | "deployments" | "performance" | "activity" | "agents";

export const CLUSTER_SECTIONS: ReadonlyArray<{ id: ClusterSection; label: string }> = [
  { id: "setup", label: "Setup" },
  { id: "fleet", label: "Fleet" },
  { id: "deployments", label: "Deployments" },
  { id: "performance", label: "Performance" },
  { id: "activity", label: "Activity" },
  { id: "agents", label: "Agents & tools" },
];

export const CLUSTER_SECTION_LABELS: Readonly<Record<ClusterSection, string>> = Object.fromEntries(
  CLUSTER_SECTIONS.map((section) => [section.id, section.label]),
) as Record<ClusterSection, string>;

/** The tab a `?cluster=` value names; anything else (or nothing) is Setup. */
export function clusterSectionFrom(search: string): ClusterSection {
  const value = new URLSearchParams(search).get("cluster");
  return CLUSTER_SECTIONS.some((section) => section.id === value) ? value as ClusterSection : "setup";
}

/** The open tab, from the address bar. */
export function clusterSectionFromLocation(): ClusterSection {
  return clusterSectionFrom(window.location.search);
}

/**
 * The event the Cluster page sends whenever its tab changes, so the top bar
 * breadcrumb ("Cluster / Fleet") follows a tab chosen on the page: a
 * `pushState` fires no `popstate` of its own.
 */
export const CLUSTER_SECTION_EVENT = "vaelor:cluster-section";

/**
 * The Deployments tab's type filter, kept in the address like the tab
 * (`?cluster=deployments&deployments=models`), so a link can land on the
 * filter that draws what it names (VD-200 review S-D1).
 */
export const DEPLOYMENTS_FILTER_PARAM = "deployments";

/**
 * Where the LLM Server, its endpoints and their API keys are: the Models
 * filter of Cluster > Deployments. Under "All" they are not drawn at all, so
 * every link that names them comes here.
 */
export const DEPLOYMENTS_MODELS_HREF = `?cluster=deployments&${DEPLOYMENTS_FILTER_PARAM}=models#/fleet`;

/**
 * Cluster > Deployments with no filter ("All"): where a link that names
 * everything serving lands, such as Home's "What's serving" (AI Chat, the
 * Assistant and the LLM Server together). Only a link that names the LLM
 * Server or its endpoints takes DEPLOYMENTS_MODELS_HREF (VD-200 central N2).
 */
export const DEPLOYMENTS_HREF = "?cluster=deployments#/fleet";

