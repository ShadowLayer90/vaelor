import type { LlmForm } from "../components/fleetTypes";

/** The names the serve form suggests: one for a GPU (vLLM) deployment, one for the others. */
export const GPU_DEPLOYMENT_NAME = "gpu-model";
export const MODEL_DEPLOYMENT_NAME = "private-model";

/**
 * The Swarm service prefix of a single-node or replicated model deployment:
 * `cluster_driver.deploy_llm` names its service `vaelor-llm-<name>`.
 */
const MODEL_SERVICE_PREFIX = "vaelor-llm-";

/**
 * Every deployment name already in use (review 2 of ACC-211): the pooled rows
 * (GPU and CPU alike) and the model services a single-node or replicated
 * deployment runs as.
 */
export function takenDeploymentNames(summary: {
  pooled_deployments?: ReadonlyArray<{ name: string }>;
  runtime?: { services?: ReadonlyArray<{ name?: string }> };
} | null | undefined): string[] {
  const pooled = (summary?.pooled_deployments ?? []).map((entry) => entry.name);
  const services = (summary?.runtime?.services ?? [])
    .map((service) => String(service.name ?? ""))
    .filter((name) => name.startsWith(MODEL_SERVICE_PREFIX))
    .map((name) => name.slice(MODEL_SERVICE_PREFIX.length));
  return [...pooled, ...services];
}

/** `base`, or `base-2`, `base-3`... - the first an existing deployment does not already have. */
export function freshDeploymentName(base: string, taken: readonly string[]): string {
  const used = new Set(taken);
  if (!used.has(base)) return base;
  let index = 2;
  while (used.has(`${base}-${index}`)) index += 1;
  return `${base}-${index}`;
}

/**
 * The serve form as "Serve a model" opens it.
 *
 * ACC-203: on a cluster with a GPU-capable worker it opens on GPU serving,
 * whose defaults are the recommended model; elsewhere on Single node, as
 * before. ACC-211: it opens with a fresh name rather than the one typed the
 * last time it was open. The name still lives in the page's form state, so
 * switching mode inside one opening keeps what was typed.
 */
export function openedServeForm(
  current: LlmForm,
  { gpuWorkers, taken }: { gpuWorkers: number; taken: readonly string[] },
): LlmForm {
  return {
    ...current,
    deploymentMode: gpuWorkers > 0 ? "gpu" : "single",
    gpuName: freshDeploymentName(GPU_DEPLOYMENT_NAME, taken),
    name: freshDeploymentName(MODEL_DEPLOYMENT_NAME, taken),
  };
}
