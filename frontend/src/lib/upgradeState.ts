import type { OfferKind, UpgradeStatus } from "../components/SystemUpdatePanel";

export type UpdateTone = "success" | "info" | "neutral";

/**
 * Whether this appliance runs the published build, as one pill: the words
 * System › Update Vaelor shows, and Home › Machines shows for the controller
 * (VD-133 1.5.5: Home read a grey "Vaelor 1.5.4" beside a worker's green "Up
 * to date", two meanings in one column). W4d-D8: "up to date" is a claim
 * about the BUILD - only the same wheel is up to date.
 */
export function updatePill(status: UpgradeStatus | null): { label: string; tone: UpdateTone } {
  const manifest = status?.manifest ?? null;
  const eligibility = status?.eligibility ?? null;
  const isReinstall = Boolean(manifest && status && manifest.version === status.running_version);
  const kind: OfferKind = eligibility?.kind ?? (isReinstall ? "build-unknown" : "update");
  const upgradeAvailable = Boolean(manifest && eligibility?.eligible && !isReinstall);
  const sameBuild = Boolean(manifest && eligibility?.eligible && kind === "same-build");
  const replaceBuild = Boolean(manifest && eligibility?.eligible && kind === "replace-build");
  if (upgradeAvailable || replaceBuild) return { label: "Update available", tone: "info" };
  if (sameBuild) return { label: "Up to date", tone: "success" };
  if (manifest && kind === "local-build-newer") return { label: "Newer local build", tone: "info" };
  return { label: "Not compared", tone: "neutral" };
}
