import { SegmentedControl } from "./ui";
import type { ClusterMode } from "../lib/clusterMode";

/**
 * The Cluster page's detail level (the ClusterHeader board): "Detail" and an
 * Easy / Advanced choice at the right end of the tab strip, so it reads as
 * applying to the tab below. Each option is a pressed/unpressed button, so the
 * choice is announced. Advanced only ever adds detail, so neither option is a
 * destructive or gated action; the choice is remembered for each account
 * (lib/clusterMode).
 */

const MODES: Array<{ value: ClusterMode; label: string }> = [
  { value: "easy", label: "Easy" },
  { value: "advanced", label: "Advanced" },
];

export function ClusterModeToggle({
  mode,
  onChange,
}: {
  mode: ClusterMode;
  onChange: (mode: ClusterMode) => void;
}) {
  return (
    <div className="cl-detail">
      <span aria-hidden="true">Detail</span>
      <SegmentedControl label="Cluster detail level" onChange={onChange} options={MODES} value={mode} />
    </div>
  );
}
