import { useEffect, useMemo, useState } from "react";
import { Button, Input, Notice, Select } from "./ui";
import { OptionCard } from "./ClusterPrimitives";
import { ClusterDialog } from "./ClusterDialog";
import "../styles/cluster-dialogs.css";
import { apiRequest } from "../lib/api";
import { summarizeHealth, useMcpCatalog, type McpServer } from "../lib/mcpCatalog";
import { useSkillsLibrary, type Skill } from "../lib/skillsLibrary";
import { UNLOAD_CAUSE_IDLE, isGpuClusterDeployment } from "../lib/gpuServingMode";
import type { AgentProfile } from "./agentTypes";
import type { FleetSummary } from "./fleetTypes";
import { AgentSkillsReview, skillsAttachCopy, useSkillsPreview } from "./AgentSkillsReview";

/**
 * The Advanced agent-deploy flow (F6c-2a): configure a READ-ONLY cluster agent
 * backed by a serving cluster model deployment, attach MCP tool grants and
 * skills, review exactly what the agent will be allowed, then queue the deploy.
 *
 * Structured as the GPU serve path is (`GpuServeForm` -> `FleetLlmModal` ->
 * `FleetCenter`): this modal collects a payload and hands it upward through
 * `onDeploy`; `FleetCenter` queues the `cluster.agent.deploy` job
 * (`queueClusterJob({ action: "deploy-agent", payload })`) and renders progress
 * in the one operation modal every cluster change uses. The deploy PAYLOAD is
 * built by the pure {@link buildAgentDeployPayload}, and the modal accepts an
 * `initial` prefill, so the F6c-2b Easy gallery can prefill the same form or
 * build a payload directly without this component.
 *
 * **One honest backing.** The backend resolves the agent's real backing from
 * the single active `cluster-inference` lease regardless of the name sent
 * (`agent_pool_operations._backing_profile`). There is effectively ONE backing,
 * so this modal never presents a multi-choice selector as authoritative: with a
 * single healthy cluster model it confirms that model; with more than one it
 * offers a selector whose copy states plainly that the agent binds the active
 * cluster inference model whichever name is chosen; with none it blocks.
 */

type PooledDeployment = NonNullable<FleetSummary["pooled_deployments"]>[number];

export interface AgentMcpGrant {
  server_id: string;
  allowed_tools: string[];
}

export interface AgentSkillPin {
  skill_id: string;
  version: number;
}

/** The values the deploy payload is built from, shared with the F6c-2b gallery. */
export interface AgentDeployValues {
  name: string;
  customAgentId: string;
  customAgentVersion: number;
  modelDeploymentName: string;
  mcpGrants: AgentMcpGrant[];
  skills: AgentSkillPin[];
}

/**
 * The version pinned onto every attached skill.
 *
 * The skills library row carries no version field, and the deployment store's
 * skill validator only requires a positive integer
 * (`agent_deployments._skills` -> `_version`, version >= 1). The pin is
 * informational: the rendered runtime config
 * (`agent_pool_operations._build_config`) never reads a skill version back, and
 * there is no versioned lookup against the unversioned library. So every
 * attached skill pins version 1, the minimum the validator accepts. Named so
 * the F6c-2b gallery pins identically.
 */
export const ATTACHED_SKILL_VERSION = 1;

/**
 * The acting permissions a deployed read-only agent may NOT assume. Mirrors
 * vaelor/custom_agents.py ALLOWED_PERMISSIONS; the backend refuses a skill that
 * carries any of these for a read-only agent. Kept tiny and documented so drift
 * from the backend is obvious.
 */
const ACTING_PERMISSIONS = ["knowledge:read", "knowledge:write", "workloads:propose"];

/** The catalog kind of Vaelor's own MCP server, which has no network address. */
const BUILTIN_SERVER_KIND = "builtin";
const BUILTIN_SERVER_REASON =
  "A deployed cluster agent cannot call Vaelor's built-in server: it has no network address.";

/**
 * Why a catalog server cannot be granted to a deployed agent, or "". A deployed
 * agent calls tool servers over the network, so Vaelor's built-in server -
 * which has no address - is shown but never selectable (ACC-076); its own
 * health sentence says why.
 */
function serverGrantBlock(server: McpServer): string {
  if (server.kind !== BUILTIN_SERVER_KIND) return "";
  return summarizeHealth(server.health).detail || BUILTIN_SERVER_REASON;
}

/**
 * Build the `cluster.agent.deploy` payload FIELDS from collected values. The
 * `confirm` token is stamped by the `deploy-agent` arm in `lib/clusterJobs.ts`
 * (mirroring `deploy-gpu`), never here, so the two cannot drift.
 * `advertise_address` and `actor` are injected server-side and are deliberately
 * absent. `node_ids` is always empty: a deployed agent is controller-placed
 * behind its inbound gate.
 */
export function buildAgentDeployPayload(values: AgentDeployValues): Record<string, unknown> {
  return {
    name: values.name.trim(),
    custom_agent_id: values.customAgentId,
    custom_agent_version: values.customAgentVersion,
    model_deployment_name: values.modelDeploymentName,
    node_ids: [],
    mcp_grants: values.mcpGrants,
    skills: values.skills,
  };
}

interface AgentDeployModalProps {
  /** The fleet's pooled deployments; the healthy vLLM ones are the candidate backings. */
  pooledDeployments: PooledDeployment[];
  busy: boolean;
  /** The session's CSRF token, for the review's skills check (an operator POST). */
  csrfToken?: string;
  /** Prefilled values for the F6c-2b Easy gallery. Omitted for the Advanced flow. */
  initial?: Partial<AgentDeployValues>;
  onClose: () => void;
  onDeploy: (payload: Record<string, unknown>) => void;
}

type Step = "configure" | "review";

export function AgentDeployModal({
  pooledDeployments,
  busy,
  csrfToken,
  initial,
  onClose,
  onDeploy,
}: AgentDeployModalProps) {
  const [profiles, setProfiles] = useState<AgentProfile[]>([]);
  const [profilesLoading, setProfilesLoading] = useState(true);
  const [profilesError, setProfilesError] = useState("");
  const mcp = useMcpCatalog();
  const skillsLib = useSkillsLibrary();

  const [step, setStep] = useState<Step>("configure");
  const [name, setName] = useState(initial?.name ?? "cluster-agent");
  const [chosenAgentId, setChosenAgentId] = useState(initial?.customAgentId ?? "");
  const [chosenBacking, setChosenBacking] = useState(
    initial?.modelDeploymentName ?? "",
  );
  const [serverTools, setServerTools] = useState<Record<string, string[]>>(() => {
    const seed: Record<string, string[]> = {};
    for (const grant of initial?.mcpGrants ?? []) seed[grant.server_id] = [...grant.allowed_tools];
    return seed;
  });
  const [attachedSkills, setAttachedSkills] = useState<Set<string>>(
    () => new Set((initial?.skills ?? []).map((entry) => entry.skill_id)),
  );

  // The inference agents the fleet can deploy. Read the way the gallery reads
  // them (`/assistant/profiles?surface=inference`), so only inference agents are
  // ever offered as a cluster backing.
  useEffect(() => {
    const controller = new AbortController();
    setProfilesLoading(true);
    apiRequest<AgentProfile[]>("/assistant/profiles?surface=inference", { signal: controller.signal })
      .then((rows) => {
        if (controller.signal.aborted) return;
        setProfiles(Array.isArray(rows) ? rows : []);
        setProfilesError("");
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        setProfilesError(reason instanceof Error ? reason.message : "The custom agents could not be read.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setProfilesLoading(false);
      });
    return () => controller.abort();
  }, []);

  const deployableProfiles = useMemo(
    () => profiles.filter((profile) => profile.custom === true && profile.enabled === true),
    [profiles],
  );
  // The candidate backings: the serving vLLM deployments, and (VD-135) one
  // scaled to zero after sitting idle, which wakes on the agent's first
  // request. One unloaded by hand does not wake on a request, so it is not one.
  const backingOptions = useMemo(
    () => pooledDeployments.filter((deployment) => isGpuClusterDeployment(deployment) || isAsleep(deployment)),
    [pooledDeployments],
  );
  // The agent defaults to the first deployable one, and the backing to the
  // first healthy cluster model, until the owner chooses. Derived in the same
  // render the options arrive in, so the select never shows an agent while
  // Review permissions still says none is chosen.
  const customAgentId = chosenAgentId || deployableProfiles[0]?.id || "";
  const modelDeploymentName = chosenBacking || backingOptions[0]?.name || "";
  const chosenBackingAsleep = backingOptions.some(
    (deployment) => deployment.name === modelDeploymentName && isAsleep(deployment),
  );

  const selectedProfile = deployableProfiles.find((profile) => profile.id === customAgentId);
  const customAgentVersion = selectedProfile?.version ?? 1;
  // assert_read_only refuses ANY non-empty permission set, so a chosen agent
  // that holds one will be rejected at deploy. Surface it here and block Deploy.
  const actingPermissions = selectedProfile?.permissions ?? [];
  const hasActingPermission = actingPermissions.length > 0;

  // A prefill can name a server that cannot be granted (the built-in one); it
  // is dropped here so the deploy never carries a grant the review hides.
  const blockedServerIds = useMemo(
    () => new Set(mcp.catalog.servers.filter((server) => serverGrantBlock(server)).map((server) => server.id)),
    [mcp.catalog.servers],
  );
  const mcpGrants = useMemo<AgentMcpGrant[]>(
    () => Object.entries(serverTools)
      .filter(([server_id, tools]) => tools.length > 0 && !blockedServerIds.has(server_id))
      .map(([server_id, allowed_tools]) => ({ server_id, allowed_tools })),
    [blockedServerIds, serverTools],
  );
  const attachedSkillPins = useMemo<AgentSkillPin[]>(
    () => [...attachedSkills].map((skill_id) => ({ skill_id, version: ATTACHED_SKILL_VERSION })),
    [attachedSkills],
  );

  const values: AgentDeployValues = {
    name,
    customAgentId,
    customAgentVersion,
    modelDeploymentName,
    mcpGrants,
    skills: attachedSkillPins,
  };

  const toggleServer = (server: McpServer) => {
    setServerTools((current) => {
      const next = { ...current };
      if (server.id in next) delete next[server.id];
      else next[server.id] = [...server.approved_tools];
      return next;
    });
  };
  const toggleTool = (serverId: string, tool: string) => {
    setServerTools((current) => {
      const tools = current[serverId] ?? [];
      const nextTools = tools.includes(tool)
        ? tools.filter((entry) => entry !== tool)
        : [...tools, tool];
      return { ...current, [serverId]: nextTools };
    });
  };
  const toggleSkill = (id: string) => {
    setAttachedSkills((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const attachableServers = mcp.catalog.servers.filter(
    (server) => server.enabled && (server.approved_tools.length > 0 || server.kind === BUILTIN_SERVER_KIND),
  );
  const attachedSkillRows = skillsLib.library.skills.filter((skill) => attachedSkills.has(skill.id));
  const attachedSkillNames = Object.fromEntries(attachedSkillRows.map((skill) => [skill.id, skill.name || "Unnamed skill"]));
  const skillsPreview = useSkillsPreview(attachedSkillPins.map((pin) => pin.skill_id), csrfToken);
  const skillRefusal = skillsPreview.preview?.refusals[0];
  const grantedServerRows = mcpGrants.map((grant) => ({
    grant,
    server: mcp.catalog.servers.find((server) => server.id === grant.server_id),
  }));

  const reviewReason = !name.trim()
    ? "Name the agent."
    : !customAgentId
      ? "Choose a custom agent to deploy."
      : !modelDeploymentName
        ? "No cluster model is available to back the agent."
        : undefined;
  const canReview = !reviewReason;

  const deployReason = hasActingPermission
    ? "This agent holds an acting permission, so a read-only cluster agent cannot be deployed from it."
    : skillRefusal
      ? "An attached skill cannot be attached to a read-only agent. Go back and remove it."
      : reviewReason;
  const canDeploy = !busy && !deployReason;

  const noBacking = backingOptions.length === 0;

  const footer = noBacking ? (
    <Button variant="secondary" onClick={onClose}>Close</Button>
  ) : step === "configure" ? (
    <>
      <Button variant="secondary" onClick={onClose}>Cancel</Button>
      <Button
        variant="primary"
        disabled={!canReview}
        disabledReason={reviewReason}
        onClick={() => setStep("review")}
      >
        Review permissions
      </Button>
    </>
  ) : (
    <>
      <Button variant="secondary" disabled={busy} onClick={() => setStep("configure")}>Back</Button>
      <Button
        variant="primary"
        disabled={!canDeploy}
        disabledReason={deployReason}
        onClick={() => onDeploy(buildAgentDeployPayload(values))}
      >
        {busy ? "Deploying..." : "Deploy agent"}
      </Button>
    </>
  );

  return (
    <ClusterDialog
      className="cd-dialog"
      eyebrow="Cluster agents"
      footer={footer}
      onClose={onClose}
      size={noBacking ? "standard" : "wide"}
      title="Deploy a cluster agent"
      titleId="cluster-agent-title"
    >
      {noBacking ? (
        <Notice severity="warning" heading="Serve a cluster model first">
          A cluster agent is backed by a serving cluster model, and none is
          healthy right now. Serve a cluster model on the GPU cluster before
          deploying an agent.
        </Notice>
      ) : step === "configure" ? (
        <>
          <p>
            A cluster agent is a read-only assistant backed by one cluster model
            deployment, reachable behind its own inbound gate. Attach the MCP
            tools and skills it may use, then review exactly what it will be
            allowed before it is deployed.
          </p>
          <div className="cd-grid">
            <Input
              label="Agent name"
              hint="Slugged on the controller; this is the name the Deployments list shows."
              value={name}
              onChange={(event) => setName(event.target.value)}
            />

            {profilesError ? (
              <Notice severity="danger" heading="Custom agents unavailable">{profilesError}</Notice>
            ) : (
              <Select
                label="Custom agent"
                hint={
                  selectedProfile?.description
                    ? `Version ${customAgentVersion}. ${selectedProfile.description}`
                    : `Deployed at version ${customAgentVersion}.`
                }
                value={customAgentId}
                disabled={profilesLoading || deployableProfiles.length === 0}
                onChange={(event) => setChosenAgentId(event.target.value)}
              >
                {profilesLoading ? (
                  <option value="">Reading custom agents...</option>
                ) : deployableProfiles.length === 0 ? (
                  <option value="">No inference agent is available</option>
                ) : (
                  deployableProfiles.map((profile) => (
                    <option key={profile.id} value={profile.id}>{profile.name}</option>
                  ))
                )}
              </Select>
            )}
          </div>

          {hasActingPermission && (
            <Notice severity="warning" heading="This agent is not read-only">
              {`"${selectedProfile?.name ?? customAgentId}" holds acting permission(s): ${actingPermissions.join(", ")}. A deployed cluster agent must be read-only, so this deploy would be rejected. Choose a read-only agent, or remove these permissions from its definition.`}
            </Notice>
          )}

          {backingOptions.length === 1 ? (
            <div className="cl-panel">
              <strong>Backing model</strong>
              <p>
                {`The agent uses this model deployment, and keeps using it even if another becomes active: ${backingOptions[0].model_id || backingOptions[0].name} (${backingOptions[0].name}).`}
              </p>
            </div>
          ) : (
            <>
              <Select
                label="Backing cluster model"
                hint="Names the cluster model you expect to be serving."
                value={modelDeploymentName}
                onChange={(event) => setChosenBacking(event.target.value)}
              >
                {backingOptions.map((deployment) => (
                  <option key={deployment.name} value={deployment.name}>
                    {`${deployment.model_id || deployment.name} - ${deployment.name}`}
                  </option>
                ))}
              </Select>
              <Notice severity="info" heading="The agent keeps this model">
                The agent uses the deployment chosen here, and keeps using it
                even if another becomes active.
              </Notice>
            </>
          )}
          {chosenBackingAsleep && (
            <Notice severity="info" heading="This model is asleep">
              It was unloaded after sitting idle. The agent deploys now; its
              first request wakes the model and is answered with a retry hint
              while the model loads, which can take a few minutes.
            </Notice>
          )}

          <fieldset className="cd-group">
            <legend className="cd-label cd-label--strong">MCP tool grants</legend>
            {mcp.loading ? (
              <p className="cd-note" role="status">Reading the MCP catalog...</p>
            ) : mcp.error ? (
              <p className="cd-bad-text" role="alert">{mcp.error}</p>
            ) : attachableServers.length === 0 ? (
              <p className="cd-note">No MCP server with approved tools is registered, so none can be attached.</p>
            ) : (
              <>
                <p className="cd-note">Attach a server, then choose which of its approved tools the agent may call.</p>
                <div className="cd-box">
                  {attachableServers.map((server) => {
                    const block = serverGrantBlock(server);
                    const attached = !block && server.id in serverTools;
                    const selected = serverTools[server.id] ?? [];
                    return (
                      <div className="cl-stack" key={server.id}>
                        <OptionCard
                          checked={attached}
                          detail={block || `${server.approved_tools.length} approved tool(s)`}
                          disabled={Boolean(block)}
                          onChange={() => toggleServer(server)}
                          title={server.name || "Unnamed server"}
                          type="checkbox"
                        />
                        {attached && (
                          <div className="cd-options cd-options--chips" role="group" aria-label={`${server.name || "Unnamed server"} tools`}>
                            {server.approved_tools.map((tool) => (
                              <OptionCard
                                checked={selected.includes(tool)}
                                key={tool}
                                onChange={() => toggleTool(server.id, tool)}
                                title={tool}
                                type="checkbox"
                              />
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                  <output aria-live="polite" className="cd-note">{mcpGrants.length} server(s) granted</output>
                </div>
              </>
            )}
          </fieldset>

          <fieldset className="cd-group">
            <legend className="cd-label cd-label--strong">Skills</legend>
            {skillsLib.loading ? (
              <p className="cd-note" role="status">Reading the skills library...</p>
            ) : skillsLib.error ? (
              <p className="cd-bad-text" role="alert">{skillsLib.error}</p>
            ) : skillsLib.library.skills.length === 0 ? (
              <p className="cd-note">No skill is registered in the library, so none can be attached.</p>
            ) : (
              <div className="cd-box">
                <p className="cd-note">
                  {`${skillsAttachCopy(skillsPreview.preview?.limits)} Only an enabled skill with no credentials and no acting permission can be attached to a read-only agent.`}
                </p>
                <div className="cd-options">
                  {skillsLib.library.skills.map((skill) => {
                    const block = skillAttachBlock(skill);
                    return (
                      <OptionCard
                        checked={attachedSkills.has(skill.id)}
                        detail={block || describeSkillScopes(skill)}
                        disabled={Boolean(block)}
                        key={skill.id}
                        onChange={() => toggleSkill(skill.id)}
                        title={skill.name || skill.id}
                        type="checkbox"
                      />
                    );
                  })}
                </div>
                <output aria-live="polite" className="cd-note">{attachedSkills.size} skill(s) attached</output>
              </div>
            )}
          </fieldset>
        </>
      ) : (
        <>
          {/* A chosen agent that holds an acting permission is refused, so the
              review says that instead of promising a read-only agent. */}
          {hasActingPermission ? (
            <Notice severity="danger" heading="This agent cannot be deployed">
              {`"${selectedProfile?.name ?? customAgentId}" holds acting permission(s): ${actingPermissions.join(", ")}. The deploy would be rejected by the read-only gate. Go back and choose a read-only agent.`}
            </Notice>
          ) : (
            <Notice severity="info" heading="Read-only agent">
              The deployed agent holds no acting permission. The runtime enforces
              this: it can read and call the granted tools below, but cannot
              write, deploy, or otherwise change anything.
            </Notice>
          )}

          <div className="cl-panel">
            <strong>Agent</strong>
            <p>{`${selectedProfile?.name ?? customAgentId} - version ${customAgentVersion}, named "${name.trim()}".`}</p>
          </div>

          <div className="cl-panel">
            <strong>Backing model</strong>
            <p>{`The agent uses this model deployment, and keeps using it even if another becomes active: ${backingModelLabel(backingOptions, modelDeploymentName)}.`}</p>
          </div>

          <div className="cl-panel">
            <strong>MCP tools the agent may call</strong>
            {grantedServerRows.length === 0 ? (
              <p>No MCP tools attached.</p>
            ) : (
              <ul className="cd-list">
                {grantedServerRows.map(({ grant, server }) => (
                  <li key={grant.server_id}>
                    {`${server?.name || grant.server_id}: ${grant.allowed_tools.join(", ")}`}
                  </li>
                ))}
              </ul>
            )}
          </div>

          <AgentSkillsReview names={attachedSkillNames} state={skillsPreview} />

          <p className="cd-note">
            After deploy, the agent appears in Deployments under the Agents
            filter. Its first API key is shown once, right after the deploy is
            queued; manage its keys under Deployments › Models.
          </p>
        </>
      )}
    </ClusterDialog>
  );
}

/**
 * One line naming a skill's read scopes - what attaching it lets the agent
 * read. An attachable skill carries no credentials and no acting permission, so
 * its grants are the read scopes that widen the agent's tool surface.
 */
function describeSkillScopes(skill: Skill): string {
  return skill.grants.length
    ? `read scopes: ${skill.grants.join(", ")}`
    : "no added read scopes";
}

/**
 * Why a library skill cannot be attached to a read-only agent, or "" if it can.
 * Mirrors the backend's accept rule (commit 41c9e59): a skill is applied only
 * when enabled, carrying no credentials, and holding no acting permission;
 * otherwise the whole deploy is refused. We surface the reason rather than
 * silently hide the skill so the operator understands why it is not attachable.
 */
function skillAttachBlock(skill: Skill): string {
  if (!skill.enabled) return "disabled";
  if (skill.credentials.length) return "carries credentials - not yet supported";
  if (skill.grants.some((grant) => ACTING_PERMISSIONS.includes(grant)))
    return "carries an acting permission - not allowed for a read-only agent";
  return "";
}

/** The human label for the chosen backing, falling back to its slug. */
/** A vLLM deployment scaled to zero after sitting idle (it wakes on a request). */
function isAsleep(deployment: PooledDeployment): boolean {
  return deployment.engine === "vllm" && deployment.state === "unloaded"
    && deployment.unload_cause === UNLOAD_CAUSE_IDLE;
}

function backingModelLabel(options: PooledDeployment[], name: string): string {
  const match = options.find((deployment) => deployment.name === name);
  if (!match) return name;
  const human = match.model_id || match.name;
  return `${human} (${match.name})`;
}
