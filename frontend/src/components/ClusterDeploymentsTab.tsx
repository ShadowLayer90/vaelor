import { useCallback, useEffect, useState } from "react";
import { Button } from "./ui";
import { ClusterDeployments, type DeploymentFilter, type PooledDeployment } from "./ClusterDeployments";
import { ClusterRetainedVolumes } from "./ClusterRetainedVolumes";
import { ClusterModels } from "./ClusterModels";
import { EndpointsPanel } from "./EndpointsPanel";
import { FilterChips } from "./ClusterPrimitives";
import { Icon } from "./Icon";
import type { ClusterServiceSummary } from "./ClusterServiceManager";
import type { Session } from "../types";
import type { AppGroup, ClusterCapacityLedger, FleetNode } from "./fleetTypes";
import type { AgentDeployment } from "../lib/clusterAgents";
import { DEPLOYMENTS_FILTER_PARAM } from "../lib/clusterSections";
import "../styles/cluster-deployments.css";

/**
 * The Deployments tab (VD-200, the ClusterDeployments and
 * ClusterDeploymentsModels boards): every App, Model and Agent across the
 * cluster under a type filter, with "Deploy an agent" beside the filter.
 * Serve a model, Deploy app and Add machine live in the page's title row on
 * every tab, so they are not repeated here.
 *
 * - All, Apps and Agents: the Deployments table ({@link ClusterDeployments},
 *   which delegates every lifecycle action to the same server-reviewed flows),
 *   and under All and Apps the data that removed apps kept on their machines.
 * - Models: what a served model is reached through and what it is made of -
 *   the LLM Server and every other served endpoint with its keys
 *   ({@link EndpointsPanel}), and the model library of cached weights
 *   ({@link ClusterModels}) beside the LLM Server card.
 */

const FILTERS: Array<{ value: DeploymentFilter; label: string }> = [
  { value: "all", label: "All" },
  { value: "models", label: "Models" },
  { value: "apps", label: "Apps" },
  { value: "agents", label: "Agents" },
];

/** The filter the address names (`?deployments=`), or All. */
function filterFromLocation(): DeploymentFilter {
  const value = new URLSearchParams(window.location.search).get(DEPLOYMENTS_FILTER_PARAM);
  return FILTERS.find((option) => option.value === value)?.value ?? "all";
}

interface Props {
  services: ClusterServiceSummary[];
  pooled: PooledDeployment[];
  /** D4d: the researched multi-service apps, each as one grouped row. */
  appGroups?: AppGroup[];
  /** F6c-1: the deployed cluster agents, shown under the Agents filter. */
  agents?: AgentDeployment[];
  /** Why the agents could not be read, or "" (ACC-080). */
  agentsError?: string;
  ledger: ClusterCapacityLedger;
  session: Session;
  onServiceReview: (action: string, payload: Record<string, unknown>) => Promise<void>;
  onNotice: (message: string, refused?: boolean) => void;
  onDeployAgent: () => void;
  /** SSH workers a model pull may target (the Models library, under that filter). */
  pullTargets: FleetNode[];
  /** Bumped when a pull/remove job finishes, to reload the library. */
  modelsReloadKey: number;
  onModelPull: (payload: Record<string, unknown>) => void;
  onModelRemove: (payload: Record<string, unknown>) => void;
}

export function ClusterDeploymentsTab({
  services,
  pooled,
  appGroups = [],
  agents = [],
  agentsError = "",
  ledger,
  session,
  onServiceReview,
  onNotice,
  onDeployAgent,
  pullTargets,
  modelsReloadKey,
  onModelPull,
  onModelRemove,
}: Props) {
  // VD-200 review S-D1: the filter lives in the address, so the links that
  // name the LLM Server and its API keys land on Models, where they are
  // drawn, and Back and Forward step through filters as through tabs.
  const [filter, setFilterState] = useState<DeploymentFilter>(filterFromLocation);
  const setFilter = useCallback((next: DeploymentFilter) => {
    const url = new URL(window.location.href);
    if (next === "all") url.searchParams.delete(DEPLOYMENTS_FILTER_PARAM);
    else url.searchParams.set(DEPLOYMENTS_FILTER_PARAM, next);
    window.history.pushState({}, "", url);
    setFilterState(next);
  }, []);
  useEffect(() => {
    const restore = () => setFilterState(filterFromLocation());
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);
  // The endpoint cards are read again whenever the page reloads or a model
  // deployment changes state, so an Unload or a Load shows on them at once
  // rather than after a browser reload (ACC-202).
  const endpointsKey = `${modelsReloadKey}|${pooled.map((entry) => `${entry.name}:${entry.state}`).join(",")}`;

  return (
    <div className="cl-stack">
      <div className="cl-dep-bar">
        <FilterChips label="Filter deployments by type" onChange={setFilter} options={FILTERS} value={filter} />
        <Button onClick={onDeployAgent}>
          <Icon name="assistant" size={16} />
          Deploy an agent
        </Button>
      </div>

      {filter === "models" ? (
        <EndpointsPanel
          library={(
            // The model library is the cached weights beneath serving, so it
            // sits beside the LLM Server card under the Models filter.
            <ClusterModels
              session={session}
              pullTargets={pullTargets}
              reloadKey={modelsReloadKey}
              onPull={onModelPull}
              onRemove={onModelRemove}
            />
          )}
          reloadKey={endpointsKey}
          session={session}
        />
      ) : (
        <>
          <ClusterDeployments
            services={services}
            pooled={pooled}
            appGroups={appGroups}
            agents={agents}
            agentsError={agentsError}
            ledger={ledger}
            session={session}
            onServiceReview={onServiceReview}
            onNotice={onNotice}
            filter={filter}
          />
          {(filter === "all" || filter === "apps") && (
            <ClusterRetainedVolumes session={session} reloadKey={services.map((service) => service.name).join(",")} />
          )}
        </>
      )}
    </div>
  );
}
