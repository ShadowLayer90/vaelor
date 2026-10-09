import { useId } from "react";
import { Button } from "./ui";
import { OptionCard } from "./ClusterPrimitives";
import type { GpuLinkRecommendation, GpuSplitLink, GpuSplitMode } from "./fleetTypes";

/**
 * A split's own choices in the GPU serve form (VD-167, owner 2026-10-01): how
 * a model split across machines is split - a pipeline unless the owner
 * chooses tensor-parallel - and the link it rides, Ethernet unless the owner
 * chooses the cluster link, for either mode.
 *
 * Presentational: the form owns the choices and the fit. Every sentence about
 * the split is the backend's - the chosen split's `parallelism_reason` and a
 * tensor fit's `link_recommendation.reason` - printed as it came. The
 * recommendation is a note with a button the owner may press; the link is
 * never switched for them.
 */

/** The two ways a split runs: a pipeline unless the owner chooses tensor. */
const SPLIT_MODES: ReadonlyArray<readonly [GpuSplitMode, string, string]> = [
  ["pipeline", "Pipeline", "Default · each machine runs whole layers in turn"],
  ["tensor", "Tensor-parallel", "Every layer shared across machines · far more link traffic"],
];

export interface GpuSplitChoiceProps {
  mode: GpuSplitMode;
  onMode: (mode: GpuSplitMode) => void;
  link: GpuSplitLink;
  onLink: (link: GpuSplitLink) => void;
  /** The fit's own sentence for the chosen split; "" while none is known. */
  reason: string;
  /** A tensor fit's recommendation, or null (pipeline, or not answered yet). */
  recommendation: GpuLinkRecommendation | null;
  /** The cluster link Setup holds, described; null when none can be used. */
  clusterLink: string | null;
  /** Why the cluster link cannot be chosen, when it cannot. */
  clusterLinkAbsent: string;
  /** "Link between machines: ..." for the chosen link; "" when unknown. */
  summary: string;
}

export function GpuSplitChoice({
  mode, onMode, link, onLink, reason, recommendation, clusterLink, clusterLinkAbsent, summary,
}: GpuSplitChoiceProps) {
  const ids = useId().replaceAll(":", "");
  const recommended = recommendation?.recommended ?? null;
  const offerRecommended = recommended !== null && recommended !== link
    && (recommended === "ethernet" || clusterLink !== null);
  // The button names the link in the backend's own label for it.
  const recommendedLabel = recommendation?.links.find((entry) => entry.key === recommended)?.label;
  return (
    <>
      <fieldset className="cd-group"
        aria-describedby={reason ? `${ids}-reason` : undefined}>
        <legend className="cd-label">How the model is split across machines</legend>
        <div className="cd-options">
          {SPLIT_MODES.map(([value, title, detail]) => (
            <OptionCard key={value} checked={mode === value} detail={detail} name="gpu-split-mode"
              onChange={() => onMode(value)} title={title} type="radio" value={value} />
          ))}
        </div>
        {reason && <p className="cd-note" id={`${ids}-reason`}>{reason}</p>}
      </fieldset>
      <fieldset className="cd-group"
        aria-describedby={recommendation ? `${ids}-recommendation` : undefined}>
        <legend className="cd-label">Network for the split</legend>
        <div className="cd-options">
          <OptionCard checked={link === "ethernet"} name="gpu-split-link" onChange={() => onLink("ethernet")}
            title={`Ethernet${recommended === "ethernet" ? " (recommended)" : ""}`}
            detail="Default · the connection each machine joined on" type="radio" value="ethernet" />
          <OptionCard checked={link === "cluster-link"} disabled={!clusterLink} name="gpu-split-link"
            onChange={() => onLink("cluster-link")}
            title={`Cluster link${recommended === "cluster-link" ? " (recommended)" : ""}`}
            detail={clusterLink ?? clusterLinkAbsent} type="radio" value="cluster-link" />
        </div>
        {summary && <p className="cd-note" role="status">{summary}</p>}
        {recommendation && (
          <p className="cd-info-text" id={`${ids}-recommendation`} role="status">{recommendation.reason}</p>
        )}
        {offerRecommended && (
          <div className="cl-actions">
            <Button variant="secondary" onClick={() => onLink(recommended)}>
              {recommendedLabel ? `Use ${recommendedLabel}` : "Use the recommended link"}
            </Button>
          </div>
        )}
      </fieldset>
    </>
  );
}
