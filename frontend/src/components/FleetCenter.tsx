import { Button, Notice } from "./ui";
import { clusterState } from "../lib/clusterState";
import { useCallback, useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import { jobCanCancel, jobIsReady, jobIsRetryable, jobIsTerminal, jobStateLabel } from "../lib/jobPresentation";
import type { Session } from "../types";
import { Icon } from "./Icon";
import { ModalShell } from "./ModalShell";
import { StatusPill } from "./StatusPill";
import { ClusterServiceManager } from "./ClusterServiceManager";
import { FleetLlmModal } from "./FleetLlmModal";
import {
  type ActionPlan,
  type AppTemplate,
  type FleetJob,
  type FleetSummary,
  type InferenceRuntimes,
  type LlmForm,

  formatMemory,
  isPrivateIpv4,
} from "./fleetTypes";
import { destinations } from "../lib/destinations";

type ProjectedFleetJob = FleetJob & {
  operation_state?: string;
  attention?: boolean;
  retryable?: boolean;
  readiness?: string;
  liveness?: string;
};
function validSshHost(value: string) {
  const host = value.trim();
  if (!host || host.length > 253) return false;
  return /^(?:[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?|[A-Fa-f0-9:]+)$/.test(host)
    && !host.includes("..");
}

function validSshPort(value: string) {
  return /^\d+$/.test(value) && Number(value) >= 1 && Number(value) <= 65535;
}

function validSshUsername(value: string) {
  return /^[A-Za-z0-9._-]{1,64}$/.test(value);
}

export function FleetCenter({ session, onBack }: { session: Session; onBack?: () => void }) {
  const fleetOperationKey = `vaelor.fleet.operation.${session.user.username}`;
  const [summary, setSummary] = useState<FleetSummary | null>(null);
  const cluster = clusterState(summary?.runtime, summary?.enrollment);
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState("");
  const [showEnroll, setShowEnroll] = useState(false);
  const [showInitialize, setShowInitialize] = useState(false);
  const [showLlm, setShowLlm] = useState(false);
  const [showApp, setShowApp] = useState(false);
  const [appTemplates, setAppTemplates] = useState<AppTemplate[]>([]);
  const [inferenceRuntimes, setInferenceRuntimes] = useState<InferenceRuntimes>({});
  const [plan, setPlan] = useState<ActionPlan | null>(null);
  const [planRequest, setPlanRequest] = useState<{
    action: string;
    nodeId?: string;
    payload?: Record<string, unknown>;
  } | null>(null);
  const [form, setForm] = useState({
    name: "",
    host: "",
    port: "22",
    username: "",
    password: "",
  });
  const [fingerprint, setFingerprint] = useState("");
  const [fingerprintAlgorithm, setFingerprintAlgorithm] = useState("");
  const [controllerAddress, setControllerAddress] = useState("");
  const [llmForm, setLlmForm] = useState<LlmForm>({
    deploymentMode: "single",
    nodeId: "",
    nodeIds: [] as string[],
    pooledModel: "",
    name: "private-model",
    repository: "",
    file: "",
    sizeGb: "",
    port: "8100",
    useForAssistant: false,
  });
  const [appForm, setAppForm] = useState({
    nodeId: "",
    templateId: "",
    name: "",
    port: "",
  });
  const [busy, setBusy] = useState(false);
  const [activeJob, setActiveJob] = useState<ProjectedFleetJob | null>(() => {
    try {
      const value = JSON.parse(window.localStorage.getItem(fleetOperationKey) || "null");
      return value && typeof value.id === "string" && typeof value.type === "string" ? value : null;
    } catch {
      return null;
    }
  });
  const [showActiveJob, setShowActiveJob] = useState(Boolean(activeJob));
  const enrollmentHostValid = validSshHost(form.host);
  const enrollmentPortValid = validSshPort(form.port);
  const enrollmentUsernameValid = validSshUsername(form.username);
  const eligibleLlmWorkers = summary?.enrolled_nodes.filter(
    (node) =>
      node.labels?.swarm_node_id
      && node.runtime?.status?.toLowerCase() === "ready"
      && node.runtime?.availability?.toLowerCase() === "active",
  ) ?? [];
  const controllerPlacement = summary?.controller.placement?.eligible
    ? summary.controller.placement
    : null;
  const eligibleLlmTargets = controllerPlacement
    ? [controllerPlacement, ...eligibleLlmWorkers]
    : eligibleLlmWorkers;
  const selectableLlmTargets = llmForm.deploymentMode === "pooled"
    ? eligibleLlmWorkers
    : eligibleLlmTargets;
  const toggleLlmWorker = (nodeId: string) => {
    setLlmForm((current) => ({
      ...current,
      nodeIds: current.nodeIds.includes(nodeId)
        ? current.nodeIds.filter((id) => id !== nodeId)
        : [...current.nodeIds, nodeId],
    }));
  };

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setSummary(await apiRequest<FleetSummary>("/cluster"));
      setNotice("");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Fleet status is unavailable.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    if (activeJob) window.localStorage.setItem(fleetOperationKey, JSON.stringify(activeJob));
    else window.localStorage.removeItem(fleetOperationKey);
  }, [activeJob, fleetOperationKey]);
  useEffect(() => {
    if (!activeJob || jobIsTerminal(activeJob)) return;
    const timer = window.setTimeout(() => {
      void apiRequest<ProjectedFleetJob>(`/jobs/${activeJob.id}`)
        .then(setActiveJob)
        .catch((error) => setNotice(error instanceof Error ? error.message : "Cluster progress could not be refreshed."));
    }, 2000);
    return () => window.clearTimeout(timer);
  }, [activeJob]);
  useEffect(() => {
    if (!controllerAddress && summary?.controller.candidate_address && isPrivateIpv4(summary.controller.candidate_address)) {
      setControllerAddress(summary.controller.candidate_address);
    }
  }, [controllerAddress, summary?.controller.candidate_address]);
  useEffect(() => {
    void apiRequest<AppTemplate[]>("/apps/catalog")
      .then(setAppTemplates)
      .catch(() => setAppTemplates([]));
  }, []);
  useEffect(() => {
    void apiRequest<InferenceRuntimes>("/cluster/inference/runtimes")
      .then((result) => {
        setInferenceRuntimes(result);
        const firstModel = result.pooled?.models[0]?.id;
        if (firstModel) {
          setLlmForm((current) => (
            current.pooledModel ? current : { ...current, pooledModel: firstModel }
          ));
        }
      })
      .catch(() => setInferenceRuntimes({}));
  }, []);

  const inspect = async () => {
    setBusy(true);
    setNotice("");
    try {
      const result = await apiRequest<{
        fingerprint: string;
        algorithm: string;
      }>("/cluster/ssh-fingerprint", {
        method: "POST",
        body: JSON.stringify({ host: form.host, port: Number(form.port) }),
      }, session.csrf_token);
      setFingerprint(result.fingerprint);
      setFingerprintAlgorithm(result.algorithm);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "SSH inspection failed.");
    } finally {
      setBusy(false);
    }
  };

  const enroll = async () => {
    setBusy(true);
    setNotice("");
    try {
      await apiRequest("/cluster/nodes", {
        method: "POST",
        body: JSON.stringify({
          ...form,
          port: Number(form.port),
          host_key_fingerprint: fingerprint,
          sudo_uses_login_password: true,
        }),
      }, session.csrf_token);
      setShowEnroll(false);
      setFingerprint("");
      setForm({ name: "", host: "", port: "22", username: "", password: "" });
      setNotice("Worker enrolled and hardware inventory verified. Review the join plan next.");
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Worker enrollment failed.");
    } finally {
      setBusy(false);
    }
  };

  const reviewPlan = async (
    action: string,
    nodeId?: string,
    actionPayload: Record<string, unknown> = {},
  ) => {
    try {
      setPlan(await apiRequest<ActionPlan>("/cluster/plan", {
        method: "POST",
        body: JSON.stringify({ action, node_id: nodeId, ...actionPayload }),
      }, session.csrf_token));
      setPlanRequest({ action, nodeId, payload: actionPayload });
      if (action === "deploy-llm") setShowLlm(false);
      if (action === "deploy-app") setShowApp(false);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The action plan is unavailable.");
    }
  };

  const approvePlan = async () => {
    if (!planRequest || !summary) return;
    if (planRequest.action === "evict-mismatched") {
      // Not a queued job: the controller acts on its own fleet record, and the
      // audit entry is written from the outcome rather than from the request,
      // so the trail says what was destroyed instead of what was asked for.
      setBusy(true);
      try {
        const result = await apiRequest<{
          evicted: Array<{ name: string }>;
          failed: Array<{ name: string; error: string }>;
        }>("/cluster/architecture/evictions", {
          method: "POST",
          body: JSON.stringify({
            confirm: "evict-mismatched-nodes",
            ...(planRequest.nodeId ? { node_id: planRequest.nodeId } : {}),
          }),
        }, session.csrf_token);
        setPlan(null);
        setPlanRequest(null);
        // Reloaded first: `refresh` clears the notice, and the report of what
        // was destroyed is the part of this that must not be swallowed.
        await refresh();
        setNotice(
          result.failed.length
            ? `Removed ${result.evicted.length} mismatched worker(s); ${
              result.failed.map((item) => `${item.name}: ${item.error}`).join("; ")
            }`
            : `Drained and removed ${result.evicted.length} mismatched worker(s): ${
              result.evicted.map((item) => item.name).join(", ")
            }.`,
        );
      } catch (error) {
        setNotice(error instanceof Error ? error.message : "The mismatched worker could not be removed.");
      } finally {
        setBusy(false);
      }
      return;
    }
    let job: { type: string; payload: Record<string, unknown> } | null = null;
    if (planRequest.action === "initialize") {
      job = {
          type: "cluster.initialize",
          payload: {
            confirm: "initialize-head-controller",
            advertise_address: planRequest.payload?.advertise_address,
          },
        };
    } else if (planRequest.action === "join-node") {
      job = {
            type: "cluster.node.join",
            payload: { confirm: "join-worker-node", node_id: planRequest.nodeId },
          };
    } else if (["drain-node", "resume-node"].includes(planRequest.action)) {
      const availability = planRequest.action === "drain-node" ? "drain" : "active";
      job = {
        type: "cluster.node.availability",
        payload: {
          confirm: `set-worker-${availability}`,
          availability,
          node_id: planRequest.nodeId,
        },
      };
    } else if (planRequest.action === "remove-node") {
      job = {
        type: "cluster.node.remove",
        payload: {
          confirm: "remove-worker-node",
          force: false,
          node_id: planRequest.nodeId,
        },
      };
    } else if (planRequest.action === "deploy-llm") {
      job = {
        type: "cluster.llm.deploy",
        payload: {
          ...planRequest.payload,
          confirm: "deploy-cluster-llm",
          ...(planRequest.nodeId ? { node_id: planRequest.nodeId } : {}),
        },
      };
    } else if (planRequest.action === "deploy-app") {
      job = {
        type: "cluster.app.deploy",
        payload: {
          ...planRequest.payload,
          confirm: "deploy-cluster-app",
          node_id: planRequest.nodeId,
        },
      };
    } else if (planRequest.action === "remove-service") {
      job = {
        type: "cluster.service.remove",
        payload: {
          confirm: "remove-cluster-service",
          service_name: planRequest.payload?.service_name,
        },
      };
    } else if ([
      "restart-service", "refresh-service", "rollback-service",
    ].includes(planRequest.action)) {
      const action = planRequest.action.replace("-service", "");
      job = {
        type: "cluster.service.manage",
        payload: {
          action,
          confirm: `${action}-cluster-service`,
          service_name: planRequest.payload?.service_name,
        },
      };
    } else if (planRequest.action === "configure-service") {
      job = {
        type: "cluster.service.configure",
        payload: {
          ...planRequest.payload,
          confirm: "configure-cluster-service",
        },
      };
    } else if (planRequest.action === "backup-service") {
      job = {
        type: "cluster.service.backup",
        payload: {
          confirm: "backup-cluster-service",
          service_name: planRequest.payload?.service_name,
        },
      };
    } else if (planRequest.action === "restore-service") {
      job = {
        type: "cluster.service.restore",
        payload: {
          confirm: "restore-cluster-service",
          service_name: planRequest.payload?.service_name,
          backup_id: planRequest.payload?.backup_id,
        },
      };
    } else if (planRequest.action === "remove-pooled") {
      job = {
        type: "cluster.pooled.remove",
        payload: {
          confirm: "remove-pooled-inference",
          name: planRequest.payload?.name,
        },
      };
    }
    if (!job) return;
    setBusy(true);
    try {
      const result = await apiRequest<Partial<ProjectedFleetJob>>("/jobs", {
        method: "POST",
        body: JSON.stringify(job),
      }, session.csrf_token);
      setActiveJob({
        id: String(result.id),
        type: result.type ?? job.type,
        state: result.state ?? "queued",
        progress: result.progress ?? 0,
        message: result.message ?? "Waiting for the cluster worker",
      });
      setShowActiveJob(true);
      setPlan(null);
      setPlanRequest(null);
      setNotice("Cluster change approved. Progress and verification remain in this workflow.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The cluster change could not be queued.");
    } finally {
      setBusy(false);
    }
  };

  if (loading && !summary) {
    return <section className="fleet-page"><p>Reading cluster state…</p></section>;
  }
  const joinedWorkers = summary?.enrolled_nodes.filter(
    (node) => node.labels?.swarm_node_id && node.runtime?.availability !== "drain",
  ) ?? [];
  // Split rather than filtered on "does not match": the conflict list also
  // holds nodes whose architecture could not be read, and those are reported
  // and left alone. Unknown is not mismatch (VD-033).
  const conflicts = summary?.architecture?.conflicts ?? [];
  const removableConflicts = conflicts.filter((conflict) => conflict.removable);
  const unreadableConflicts = conflicts.filter((conflict) => !conflict.removable);
  const removals = summary?.architecture?.removals ?? [];

  return (
    <section className="fleet-page">
      <header className="fleet-hero">
        <div>
          <h1>{destinations.fleet.name}</h1>
          <p>Enroll Linux workers over fingerprint-pinned SSH, then place apps and AI services across them. This node is the head controller.</p>
        </div>
        <div className="fleet-hero__actions">
          {onBack && <Button variant="quiet" onClick={onBack} type="button">Back to overview</Button>}
          <Button variant="quiet" disabled={busy} onClick={() => void refresh()} type="button">Reload</Button>
          <StatusPill label={cluster.label} status={cluster.tone} />
          {/*
            * Task #75. This was an enabled primary button on a screen whose
            * own text read "Workers cannot be enrolled." — and it opened a
            * full SSH enrollment form including a sudo password field. A dead
            * control that asks for the most sensitive input in the product is
            * worse than a dead control: the credential is collected for an
            * operation known in advance to fail.
            *
            * `cluster.enrollment.actionable` is derived inside `clusterState`
            * from the state it reported one line earlier (VD-009a, task #54),
            * so this call site cannot offer a control the state does not
            * support. It is the same gate "Configure LLM server" and "Deploy
            * app" already carry, with the same reason text.
            */}
          {session.user.role === "administrator" && (
            <Button
              variant="primary"
              disabled={!cluster.enrollment.actionable}
              disabledReason={cluster.enrollment.actionable ? undefined : cluster.enrollment.reason}
              onClick={() => setShowEnroll(true)}
            >
              Add worker
            </Button>
          )}
        </div>
      </header>

      {notice && <Notice severity="info">{notice}</Notice>}
      {activeJob && !showActiveJob && <Notice severity="info"><span>Cluster operation {jobStateLabel(activeJob)}: {activeJob.message}</span><Button variant="quiet" onClick={() => setShowActiveJob(true)}>View current operation</Button></Notice>}

      <section className="fleet-controller" aria-labelledby="fleet-controller-title">
        <div className="fleet-controller__mark"><Icon name="network" size={26} /></div>
        <div>
          <span className="eyebrow">THIS VAELOR NODE</span>
          <h2 id="fleet-controller-title">
            {cluster.operatingAsController ? "Head controller" : "Cluster controller not active"}
          </h2>
          {/* One statement of what is actually true, instead of describing a
              working Swarm manager beside a badge that says otherwise. */}
          <p>{cluster.summary}</p>
          {cluster.unavailable && <p className="fleet-controller__limit">{cluster.unavailable}</p>}
        </div>
        <dl>
          <div><dt>Engine</dt><dd>{summary?.runtime.available ? "Docker Swarm" : "Not available"}</dd></div>
          <div><dt>Workers</dt><dd>{summary?.enrolled_nodes.length ?? 0} enrolled</dd></div>
          <div><dt>Services</dt><dd>{summary?.runtime.services?.length ?? 0} managed</dd></div>
        </dl>
        {cluster.nextStep && (
          session.user.role === "administrator" && cluster.id === "not-initialized"
            ? <Button onClick={() => setShowInitialize(true)}>Review controller setup</Button>
            : <p className="fleet-controller__next">{cluster.nextStep}</p>
        )}
      </section>

      {!!removableConflicts.length && (
        <Notice severity="warning">
          <span>
            {removableConflicts.length === 1
              ? `${removableConflicts[0].name || removableConflicts[0].host} is ${removableConflicts[0].label}; this controller is ${summary?.architecture?.label}. Work placed on it cannot start, so it is drained and removed from the fleet.`
              : `${removableConflicts.length} enrolled workers do not share this controller's processor architecture (${summary?.architecture?.label}). Work placed on them cannot start, so they are drained and removed from the fleet.`}
          </span>
          {session.user.role === "administrator" && (
            <Button variant="danger" onClick={() => void reviewPlan("evict-mismatched")}>
              Review drain and removal
            </Button>
          )}
        </Notice>
      )}
      {!!unreadableConflicts.length && (
        <Notice severity="info">
          {/* Reported, never removed. A probe that failed is not evidence of a
              mismatch, and treating it as one would destroy an enrollment on
              the strength of a reading nobody took. */}
          <span>
            {unreadableConflicts.length} enrolled worker(s) report an architecture this
            controller does not recognise. They are left enrolled and unchanged: an
            architecture that could not be read is not a mismatch. Recheck the worker,
            or remove it yourself if it does not belong.
          </span>
        </Notice>
      )}

      {!!removals.length && (
        <section className="fleet-section" aria-labelledby="fleet-removals-title">
          <div className="fleet-section__header">
            <div>
              <span className="eyebrow">ARCHITECTURE ENFORCEMENT</span>
              <h2 id="fleet-removals-title">Workers this controller removed</h2>
            </div>
          </div>
          {/* A removed node is gone from the list above, so without this the
              fleet view would simply stop showing it. Silent removal is its
              own kind of lie (VD-033). */}
          <ul className="fleet-removals">
            {removals.map((removal) => (
              <li key={`${removal.node_id}-${removal.at}`}>
                <strong>{removal.name || removal.host}</strong>
                <span>{removal.reason}</span>
                <span>
                  {removal.drained ? "Drained, then unenrolled" : "Unenrolled"}
                  {removal.forced ? " (forced: the worker did not leave cleanly)" : ""}
                  {" · its stored SSH credential was deleted"}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="fleet-section">
        <div className="fleet-section__header">
          <div>
            <span className="eyebrow">COMPUTE NODES</span>
            <h2>Enrolled workers</h2>
          </div>
          <Button variant="quiet" onClick={() => void refresh()}>Reload</Button>
        </div>
        {summary?.enrolled_nodes.length ? (
          <div className="fleet-node-grid">
            {summary.enrolled_nodes.map((node) => (
              <article className="fleet-node" key={node.id}>
                <div className="fleet-node__heading">
                  <div className="fleet-node__icon"><Icon name="cpu" size={20} /></div>
                  <div><h3>{node.name}</h3><p>{node.host}:{node.port}</p></div>
                  <StatusPill
                    label={node.runtime?.availability || node.runtime_state || node.state}
                    status={node.runtime_state === "ready" || (node.state === "enrolled" && node.inventory.reachable) ? "healthy" : "degraded"}
                  />
                </div>
                {node.architecture && !node.architecture.matches_controller && (
                  <p className="fleet-node__mismatch" role="status">
                    {/* A confirmed mismatch says it is going; an architecture
                        this appliance could not read says only that, because
                        unknown is not mismatch and nothing is removed on it. */}
                    {node.architecture.removable
                      ? node.architecture.removal_reason
                      : node.architecture.reason}
                  </p>
                )}
                <dl className="fleet-node__facts">
                  <div><dt>OS</dt><dd>{node.inventory.os || "Not reported"}</dd></div>
                  <div><dt>CPU</dt><dd>{node.inventory.cpu_count ?? "?"} cores · {node.inventory.architecture || "unknown"}</dd></div>
                  <div><dt>Memory</dt><dd>{formatMemory(node.inventory.memory_bytes)}</dd></div>
                  <div><dt>Docker</dt><dd>{node.inventory.docker ? "Ready" : "Install during join"}</dd></div>
                  <div><dt>Cluster</dt><dd>{node.runtime ? `${node.runtime.status} · ${node.runtime.availability}` : node.state}</dd></div>
                </dl>
                <div className="fleet-node__actions">
                  {!node.runtime && !node.labels?.swarm_node_id && (
                    <Button onClick={() => void reviewPlan("join-node", node.id)}>
                      Review join plan
                    </Button>
                  )}
                  {node.labels?.swarm_node_id && node.runtime?.availability !== "drain" && (
                    <Button onClick={() => void reviewPlan("drain-node", node.id)}>
                      Drain for maintenance
                    </Button>
                  )}
                  {node.labels?.swarm_node_id && node.runtime?.availability === "drain" && (
                    <Button onClick={() => void reviewPlan("resume-node", node.id)}>
                      Return to service
                    </Button>
                  )}
                  <Button variant="quiet" onClick={async () => {
                    try {
                      await apiRequest(`/cluster/nodes/${node.id}/refresh`, { method: "POST", body: "{}" }, session.csrf_token);
                      await refresh();
                    } catch (error) {
                      setNotice(error instanceof Error ? error.message : "The worker could not be reached.");
                    }
                  }}>Recheck</Button>
                  {node.architecture?.removable && session.user.role === "administrator" && (
                    <Button variant="danger" onClick={() => void reviewPlan("evict-mismatched", node.id)}>
                      Review drain and removal
                    </Button>
                  )}
                  <Button variant="danger" onClick={() => void reviewPlan("remove-node", node.id)}>
                    Remove worker
                  </Button>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <div className="fleet-empty">
            <Icon name="network" size={30} />
            <h3>No workers enrolled</h3>
            <p>Add another node with its SSH address and credentials. Vaelor verifies the host key before sending a password.</p>
          </div>
        )}
      </section>

      <section className="fleet-deploy">
        <div>
          <span className="eyebrow">CLUSTER WORKLOAD</span>
          <h2>OpenAI-compatible LLM server</h2>
          <p>Place a reviewed llama.cpp server and GGUF model on this controller or an active worker with enough RAM and storage.</p>
        </div>
        <Button
          disabled={!eligibleLlmTargets.length}
          disabledReason={!eligibleLlmTargets.length
            ? cluster.operatingAsController
              ? "No node currently has enough free memory and storage for a model server."
              : cluster.unavailable
            : undefined}
          onClick={() => setShowLlm(true)}
        >
          Configure LLM server
        </Button>
      </section>

      <section className="fleet-deploy fleet-deploy--app">
        <div>
          <span className="eyebrow">CLUSTER APPLICATION</span>
          <h2>Deploy a reviewed app</h2>
          <p>Choose a catalog app and a worker. Vaelor validates matching CPU architecture, memory headroom, port policy, and one healthy replica before reporting success.</p>
        </div>
        <Button
          disabled={!joinedWorkers.length}
          disabledReason={!joinedWorkers.length
            ? cluster.operatingAsController
              ? "Enrol at least one worker before deploying a cluster app."
              : cluster.unavailable
            : undefined}
          onClick={() => setShowApp(true)}
        >
          Deploy app
        </Button>
      </section>

      {!!summary?.runtime.services?.length && (
        <section className="fleet-section" aria-labelledby="cluster-services-title">
          <div className="fleet-section__header">
            <div>
              <span className="eyebrow">MANAGED SERVICES</span>
              <h2 id="cluster-services-title">Cluster services</h2>
            </div>
          </div>
          <div className="fleet-service-list">
            {summary.runtime.services.map((service) => (
              <ClusterServiceManager
                key={service.id || service.name}
                service={service}
                session={session}
                onNotice={setNotice}
                onReview={(action, payload) => reviewPlan(action, undefined, payload)}
              />
            ))}
          </div>
        </section>
      )}

      {!!summary?.pooled_deployments?.length && (
        <section className="fleet-section" aria-labelledby="pooled-services-title">
          <div className="fleet-section__header">
            <div>
              <span className="eyebrow">POOLED INFERENCE</span>
              <h2 id="pooled-services-title">Distributed model deployments</h2>
            </div>
          </div>
          <div className="fleet-service-list">
            {summary.pooled_deployments.map((deployment) => (
              <article className="fleet-service" key={deployment.name}>
                <div>
                  <strong>{deployment.name}</strong>
                  <span>
                    {deployment.model_id} · {deployment.node_ids.length} nodes ·{" "}
                    {deployment.endpoint || "No active endpoint"}
                  </span>
                </div>
                <StatusPill
                  label={deployment.state}
                  status={deployment.state === "healthy" ? "healthy" : "degraded"}
                />
                <Button variant="danger"

                  onClick={() => void reviewPlan("remove-pooled", undefined, {
                    name: deployment.name,
                  })}
                >
                  Review removal
                </Button>
              </article>
            ))}
          </div>
        </section>
      )}

      <aside className="fleet-requirements">
        <strong>Worker requirements</strong>
        <span>{summary?.requirements.network}</span>
        <span>{summary?.requirements.ports.join(" · ")}</span>
        {summary?.architecture && (
          <span>Processor architecture: {summary.architecture.label}. {summary.architecture.note}</span>
        )}
        {summary?.requirements.container_runtime && (
          <span>Container runtime: {summary.requirements.container_runtime}</span>
        )}
      </aside>

      {showInitialize && (
        <ModalShell labelledBy="initialize-controller-title" onClose={() => !busy && setShowInitialize(false)}>
          <section className="fleet-enroll">
            <header>
              <span className="eyebrow">HEAD CONTROLLER SETUP</span>
              <h2 id="initialize-controller-title">Choose the controller LAN address</h2>
              <p>Workers use this fixed private IPv4 address to join the cluster. Confirm that it belongs to this Vaelor node.</p>
            </header>
            <div className="fleet-enroll__grid">
              <label>
                Controller LAN address
                <input
                  className="ui-control" inputMode="decimal"
                  placeholder="192.168.0.173"
                  value={controllerAddress}
                  onChange={(event) => setControllerAddress(event.target.value.trim())}
                />
              </label>
              <aside className="fleet-controller-address-help">
                <strong>Use a stable address</strong>
                <p>Reserve this address in your router or configure a static LAN address before enrolling workers.</p>
              </aside>
            </div>
            <div className="fleet-enroll__footer">
              <Button variant="quiet" onClick={() => setShowInitialize(false)}>Cancel</Button>
              <Button variant="primary"

                disabled={!isPrivateIpv4(controllerAddress)}
                onClick={() => {
                  setShowInitialize(false);
                  void reviewPlan("initialize", undefined, {
                    advertise_address: controllerAddress,
                  });
                }}
              >
                Review controller plan
              </Button>
            </div>
          </section>
        </ModalShell>
      )}

      {showEnroll && (
        <ModalShell labelledBy="enroll-worker-title" onClose={() => !busy && setShowEnroll(false)}>
          <section className="fleet-enroll">
            <header>
              <span className="eyebrow">SECURE NODE ENROLLMENT</span>
              <h2 id="enroll-worker-title">Add a Linux worker</h2>
              <p>The password is encrypted by the credential broker. It never appears in cluster jobs or inventory.</p>
            </header>
            <div className="fleet-enroll__grid">
              <label>Friendly name<input className="ui-control" value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} /></label>
              <label>Private IP or hostname<input className="ui-control" aria-invalid={Boolean(form.host) && !enrollmentHostValid} value={form.host} onChange={(event) => {
                setFingerprint("");
                setForm({ ...form, host: event.target.value });
              }} placeholder="192.168.0.181" /></label>
              {form.host && !enrollmentHostValid && <small role="alert">Enter an IP address or hostname without spaces or command characters.</small>}
              <label>SSH port<input className="ui-control" aria-invalid={!enrollmentPortValid} inputMode="numeric" value={form.port} onChange={(event) => setForm({ ...form, port: event.target.value })} /></label>
              {!enrollmentPortValid && <small role="alert">Use an SSH port from 1 to 65535.</small>}
              <label>SSH user<input className="ui-control" aria-invalid={Boolean(form.username) && !enrollmentUsernameValid} autoComplete="username" value={form.username} onChange={(event) => setForm({ ...form, username: event.target.value })} /></label>
              {form.username && !enrollmentUsernameValid && <small role="alert">Use up to 64 letters, numbers, dots, dashes, or underscores.</small>}
              <label className="fleet-enroll__password">SSH / sudo password<input className="ui-control" autoComplete="new-password" type="password" value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} /></label>
            </div>
            {!fingerprint ? (
              <div className="fleet-enroll__footer">
                <Button variant="quiet" onClick={() => setShowEnroll(false)}>Cancel</Button>
                <Button variant="primary" disabled={busy || !enrollmentHostValid || !enrollmentPortValid} onClick={() => void inspect()}>
                  {busy ? "Inspecting…" : "Inspect SSH identity"}
                </Button>
              </div>
            ) : (
              <div className="fleet-fingerprint">
                <div><Icon name="shield" size={22} /></div>
                <div>
                  <strong>Confirm this host key</strong>
                  <span>{fingerprintAlgorithm}</span>
                  <code>{fingerprint}</code>
                  <p>Compare this with the fingerprint shown on the worker or from <code>ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub</code>.</p>
                </div>
                <Button variant="primary" disabled={busy || !enrollmentUsernameValid || !form.password} onClick={() => void enroll()}>
                  {busy ? "Verifying…" : "Fingerprint matches — enroll"}
                </Button>
              </div>
            )}
          </section>
        </ModalShell>
      )}

      {showLlm && (
        <FleetLlmModal
          busy={busy}
          eligibleWorkers={eligibleLlmWorkers}
          form={llmForm}
          inferenceRuntimes={inferenceRuntimes}
          onClose={() => !busy && setShowLlm(false)}
          onReview={(nodeId, payload) => void reviewPlan("deploy-llm", nodeId, payload)}
          onToggleWorker={toggleLlmWorker}
          selectableTargets={selectableLlmTargets}
          setForm={setLlmForm}
        />
      )}

      {showApp && (
        <ModalShell labelledBy="cluster-app-title" onClose={() => !busy && setShowApp(false)}>
          <section className="fleet-enroll">
            <header>
              <span className="eyebrow">RESOURCE-AWARE PLACEMENT</span>
              <h2 id="cluster-app-title">Deploy an app to a worker</h2>
              <p>Catalog images use bounded memory and reviewed ports. Persistent data remains on the selected worker.</p>
            </header>
            <div className="fleet-enroll__grid">
              <label>
                Worker
                <select className="ui-control" value={appForm.nodeId} onChange={(event) => setAppForm({ ...appForm, nodeId: event.target.value })}>
                  <option value="">Choose an active worker</option>
                  {joinedWorkers.map((node) => (
                    <option key={node.id} value={node.id}>
                      {node.name} · {formatMemory(node.inventory.memory_bytes)}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Application
                <select className="ui-control" value={appForm.templateId} onChange={(event) => {
                  const template = appTemplates.find((item) => item.id === event.target.value);
                  setAppForm({
                    ...appForm,
                    templateId: event.target.value,
                    name: event.target.value,
                    port: template ? String(template.default_port) : "",
                  });
                }}>
                  <option value="">Choose a catalog app</option>
                  {appTemplates.map((template) => (
                    <option key={template.id} value={template.id}>
                      {template.name} · {template.memory}
                    </option>
                  ))}
                </select>
              </label>
              <label>Deployment name<input className="ui-control" value={appForm.name} onChange={(event) => setAppForm({ ...appForm, name: event.target.value })} /></label>
              <label>Published port<input className="ui-control" inputMode="numeric" value={appForm.port} onChange={(event) => setAppForm({ ...appForm, port: event.target.value })} /></label>
            </div>
            <div className="fleet-enroll__footer">
              <Button variant="quiet" onClick={() => setShowApp(false)}>Cancel</Button>
              <Button variant="primary"

                disabled={busy || !appForm.nodeId || !appForm.templateId || !appForm.name || !(Number(appForm.port) >= 1024)}
                onClick={() => void reviewPlan("deploy-app", appForm.nodeId, {
                  template_id: appForm.templateId,
                  name: appForm.name,
                  port: Number(appForm.port),
                })}
              >
                Review placement
              </Button>
            </div>
          </section>
        </ModalShell>
      )}

      {plan && (
        <ModalShell labelledBy="cluster-plan-title" onClose={() => {
          setPlan(null);
          setPlanRequest(null);
        }}>
          <section className="fleet-plan">
            <header><span className="eyebrow">CHANGE PLAN</span><h2 id="cluster-plan-title">{plan.title}</h2></header>
            <ol>{plan.steps.map((step) => <li key={step}>{step}</li>)}</ol>
            <div className="fleet-plan__impact"><strong>Expected impact</strong><p>{plan.impact}</p></div>
            {plan.terminology && <p className="fleet-plan__note">{plan.terminology}</p>}
            <footer>
              <Button variant="quiet" onClick={() => {
                setPlan(null);
                setPlanRequest(null);
              }}>Close</Button>
              <Button variant="primary"

                disabled={busy}
                onClick={() => void approvePlan()}
              >
                {busy ? "Queuing…" : "Approve and queue"}
              </Button>
            </footer>
          </section>
        </ModalShell>
      )}

      {activeJob && showActiveJob && (
        <ModalShell labelledBy="cluster-operation-title" onClose={() => setShowActiveJob(false)}>
          <section className="fleet-plan fleet-operation">
            <header><span className="eyebrow">CURRENT CLUSTER OPERATION</span><h2 id="cluster-operation-title">{activeJob.type.replaceAll(".", " ")}</h2></header>
            <p>{activeJob.message || jobStateLabel(activeJob)}</p>
            <progress max="100" value={activeJob.progress}>{activeJob.progress}%</progress>
            <div className="fleet-plan__impact"><strong>Current state</strong><p>{jobStateLabel(activeJob)}</p></div>
            <footer>
              <Button variant="quiet" onClick={() => setShowActiveJob(false)}>Continue in background</Button>
              {jobCanCancel(activeJob) && <Button disabled={busy} onClick={() => {
                setBusy(true);
                void apiRequest<ProjectedFleetJob>(`/jobs/${activeJob.id}/cancel`, { method: "POST", body: "{}" }, session.csrf_token)
                  .then(setActiveJob)
                  .catch((error) => setNotice(error instanceof Error ? error.message : "Cancellation could not be requested."))
                  .finally(() => setBusy(false));
              }}>Cancel operation</Button>}
              {(jobIsRetryable(activeJob)) && <Button variant="primary" disabled={busy} onClick={() => {
                setBusy(true);
                void apiRequest<ProjectedFleetJob>(`/jobs/${activeJob.id}/retry`, { method: "POST", body: "{}" }, session.csrf_token)
                  .then(setActiveJob)
                  .catch((error) => setNotice(error instanceof Error ? error.message : "Retry could not be queued."))
                  .finally(() => setBusy(false));
              }}>Retry safely</Button>}
              {(jobIsReady(activeJob) || (jobIsTerminal(activeJob) && !jobIsRetryable(activeJob))) && <Button variant="primary" onClick={() => { setActiveJob(null); setShowActiveJob(false); void refresh(); }}>Done</Button>}
            </footer>
          </section>
        </ModalShell>
      )}
    </section>
  );
}
