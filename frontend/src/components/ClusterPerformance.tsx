import { useEffect, useState } from "react";
import { Notice } from "./ui";
import { exactTime, timeAgo } from "../lib/format";
import type { Session } from "../types";
import type { ClusterMode } from "../lib/clusterMode";
import { useClusterPerformance, type PerformanceSnapshot } from "../lib/clusterPerformance";
import { readStoredRange, storeRange } from "../lib/performanceDashboard";
import { ModelUsageCard } from "./ModelUsageCard";
import {
  FleetUseSummary,
  HealthOverview,
  PercentilesCard,
  RequestCountsCard,
  ServingOverview,
  TracesOverview,
  UncollectedPanel,
  UtilisationTable,
  WHY_TONE,
} from "./PerformanceDiagnostics";
import { PerformanceDashboard, type ResolvedRange } from "./PerformanceDashboard";
import { ProfilePanel } from "./PerformanceProfile";
import { RequestHealthCard } from "./RequestHealthCard";

/** The Why on the status card's right: the snapshot's own sentences, in its tone. */
function WhyLine({ snapshot }: { snapshot: PerformanceSnapshot }) {
  return (
    <section className={`perf-why perf-why--${WHY_TONE[snapshot.why.signal] ?? "idle"}`} aria-label="Diagnosis">
      <span className="perf-why__eyebrow">Why</span>
      <p className="perf-why__text">{snapshot.why.headline}</p>
      {snapshot.why.detail && <p className="perf-why__detail">{snapshot.why.detail}</p>}
    </section>
  );
}

/**
 * The Performance tab (§6b, VD-147; VD-200, the four ClusterPerformance
 * boards): the dashboard on top, then Diagnostics, then - in Advanced - the
 * per-machine table, the percentiles, the profile and the counts. Everything
 * sits on one four-column grid. The shell owns the range (remembered per
 * browser), auto-refresh and reload. Diagnostics reads the range the
 * dashboard's backend answered with, so both describe the same span (the
 * snapshot caps it at a day).
 */
export function ClusterPerformance({ mode, session, clusterServing = false }: { mode: ClusterMode; session?: Session; clusterServing?: boolean }) {
  const [range, setRange] = useState<string | null>(readStoredRange);
  const [resolved, setResolved] = useState<ResolvedRange | null>(null);
  const [auto, setAuto] = useState(true);
  const [reloadKey, setReloadKey] = useState(0);
  // Diagnostics waits for the dashboard to say which range it shows; "" asks for the backend's default.
  const window = range ?? resolved?.range ?? null;
  const { snapshot, loading, error, reload, receivedAt, failedAt } = useClusterPerformance(window);
  const isAdmin = session?.user.role === "administrator";
  const advanced = mode === "advanced";
  // The chosen range is remembered per browser (spec 4.4).
  useEffect(() => storeRange(range), [range]);

  return (
    <div className="cluster-perf">
      <PerformanceDashboard
        auto={auto}
        decodeFloor={snapshot?.requests.generation?.decode.floor_tokens_per_second ?? null}
        mode={mode}
        onReload={() => { setReloadKey((value) => value + 1); reload(); }}
        range={range}
        reloadKey={reloadKey}
        setAuto={setAuto}
        setRange={setRange}
        setResolved={setResolved}
        why={snapshot ? <WhyLine snapshot={snapshot} /> : undefined}
      />

      <section className="cluster-perf__diagnostics" aria-labelledby="cluster-perf-diagnostics">
        <header className="cluster-perf__diagnostics-head">
          <h2 id="cluster-perf-diagnostics">Diagnostics</h2>
          {resolved?.label && <p>{resolved.label}</p>}
        </header>
        {error && (
          <Notice severity="warning">
            {snapshot && receivedAt && failedAt
              ? `${error} Still showing the reading from ${timeAgo(receivedAt, failedAt)} (${exactTime(receivedAt)}).`
              : error}
          </Notice>
        )}
        {loading && !snapshot && <p className="perf-empty">Reading performance signals…</p>}

        {snapshot && (
          <div className="perf-grid">
            <RequestHealthCard className="perf-span-4" health={snapshot.requests} />
            <ServingOverview className="perf-span-2 perf-rows-2" clusterServing={clusterServing} serving={snapshot.serving} />
            <TracesOverview className="perf-span-2" traces={snapshot.traces} />
            <FleetUseSummary className="perf-span-1" nodes={snapshot.nodes} />
            <HealthOverview className="perf-span-1" health={snapshot.health} />
            {!advanced && (
              <>
                <ModelUsageCard className="perf-span-2" deployments={snapshot.deployments ?? []} readable={snapshot.deployments_readable !== false} />
                <UncollectedPanel className="perf-span-2" uncollected={snapshot.uncollected} />
              </>
            )}
          </div>
        )}
      </section>

      {snapshot && advanced && (
        <section aria-label="Advanced diagnostics" className="perf-grid cluster-perf__advanced">
          <UtilisationTable className="perf-span-4" nodes={snapshot.nodes} />
          <ModelUsageCard className="perf-span-2" deployments={snapshot.deployments ?? []} readable={snapshot.deployments_readable !== false} />
          <PercentilesCard className="perf-span-2" health={snapshot.requests} />
          {isAdmin && session && <ProfilePanel className="perf-span-2 perf-rows-2" session={session} />}
          <RequestCountsCard className="perf-span-2" health={snapshot.requests} />
          <UncollectedPanel className="perf-span-2" uncollected={snapshot.uncollected} />
        </section>
      )}
    </div>
  );
}
