/**
 * The pure mapping from an approved cluster plan to the job it queues.
 *
 * This was the long `if (planRequest.action === …)` chain inside
 * `FleetCenter.approvePlan`, lifted out whole so the component stays well under
 * the 1,000-line module limit and so the exact request→job translation — the
 * part that must not drift, because every job is approval-gated and audited — is
 * testable on its own. The behaviour is unchanged: the same `type` and the same
 * `confirm` token and payload keys for each action, and `null` for an action
 * this map does not queue (notably `evict-mismatched`, which the controller acts
 * on directly rather than through the job queue — see `FleetCenter`).
 */

export interface ClusterPlanRequest {
  action: string;
  nodeId?: string;
  payload?: Record<string, unknown>;
}

export interface ClusterJob {
  type: string;
  payload: Record<string, unknown>;
}

export function buildClusterJob(request: ClusterPlanRequest): ClusterJob | null {
  const { action, nodeId, payload } = request;
  if (action === "initialize") {
    return {
      type: "cluster.initialize",
      payload: {
        confirm: "initialize-head-controller",
        advertise_address: payload?.advertise_address,
      },
    };
  }
  if (action === "join-node") {
    return {
      type: "cluster.node.join",
      payload: { confirm: "join-worker-node", node_id: nodeId },
    };
  }
  if (action === "drain-node" || action === "resume-node") {
    const availability = action === "drain-node" ? "drain" : "active";
    return {
      type: "cluster.node.availability",
      payload: {
        confirm: `set-worker-${availability}`,
        availability,
        node_id: nodeId,
        // Only a stateful drain with no backup carries this; the plan collects
        // the typed token and the backend gate reads it back. Absent for a
        // stateless drain or any resume, so nothing extra rides on those.
        ...(payload?.data_loss_ack ? { data_loss_ack: payload.data_loss_ack } : {}),
      },
    };
  }
  if (action === "remove-node") {
    return {
      type: "cluster.node.remove",
      payload: { confirm: "remove-worker-node", force: false, node_id: nodeId },
    };
  }
  if (action === "force-remove-node") {
    // B1: remove a worker the controller cannot reach, taking its share of any
    // deployment with it. Offered only after a plain removal was refused, and
    // only through its reviewed plan, whose typed acknowledgement rides here -
    // the backend refuses a forced removal without it.
    return {
      type: "cluster.node.remove",
      payload: {
        confirm: "force-remove-worker-node",
        force: true,
        node_id: nodeId,
        data_loss_ack: payload?.data_loss_ack,
      },
    };
  }
  if (action === "deploy-llm") {
    return {
      type: "cluster.llm.deploy",
      payload: {
        ...payload,
        confirm: "deploy-cluster-llm",
        ...(nodeId ? { node_id: nodeId } : {}),
      },
    };
  }
  if (action === "deploy-gpu") {
    // GPU serving shares the `cluster.llm.deploy` job type and its
    // `deploy-cluster-llm` confirm gate; `deployment_mode: "gpu"` is what routes
    // `cluster_operations.deploy_llm` to the vLLM path. The modal supplies the
    // whole GPU payload (node_ids, name, model_source, optional model_spec, port,
    // interface, gpu_memory_utilization, max_model_len, link, and the `intent`
    // its fit preview was taken with); this arm only stamps the mode and the
    // confirm token so the two cannot drift. `use_for_assistant` is deliberately
    // never among them: the GPU path refuses it up front (VD-127 D4), because a
    // teardown's credential cascade would strand the on-device Assistant.
    return {
      type: "cluster.llm.deploy",
      payload: {
        ...payload,
        deployment_mode: "gpu",
        confirm: "deploy-cluster-llm",
      },
    };
  }
  if (action === "remove-gpu") {
    return {
      type: "cluster.gpu.remove",
      payload: { confirm: "remove-gpu-inference", name: payload?.name },
    };
  }
  if (action === "unload-gpu") {
    // G3a: stop a healthy vLLM deployment's serving units to reclaim the
    // GPU, keeping the deployment, its cached weights and its credential so
    // `load-gpu` brings it back warm. Mirrors `remove-gpu`: only the confirm
    // token and the name, so this arm cannot drift from the backend gate.
    return {
      type: "cluster.gpu.unload",
      payload: { confirm: "unload-gpu-inference", name: payload?.name },
    };
  }
  if (action === "load-gpu") {
    // G3a: re-serve an unloaded vLLM deployment from its stored record.
    return {
      type: "cluster.gpu.load",
      payload: { confirm: "load-gpu-inference", name: payload?.name },
    };
  }
  if (action === "model-pull") {
    return {
      type: "cluster.model.pull",
      payload: {
        confirm: "pull-cluster-model",
        model_source: payload?.model_source,
        node_ids: payload?.node_ids,
      },
    };
  }
  if (action === "model-remove") {
    // `gpu_model_library.remove` resolves the target through
    // `resolve_model_source(payload["model_source"])`, exactly like the pull
    // path — a bare `org/name` resolves fine. Sending `repo` instead left it
    // with no source and it raised every time (VD-B3b-2).
    return {
      type: "cluster.model.remove",
      payload: {
        confirm: "remove-cluster-model",
        model_source: payload?.model_source,
        ...(payload?.node_ids ? { node_ids: payload.node_ids } : {}),
      },
    };
  }
  if (action === "deploy-app") {
    return {
      type: "cluster.app.deploy",
      payload: { ...payload, confirm: "deploy-cluster-app", node_id: nodeId },
    };
  }
  if (action === "deploy-researched-app") {
    // D4d: an APPROVED researched draft deployed across the cluster as N Swarm
    // services. The backend consumes the draft ONLY via `resolve_import` (state
    // must be approved, digests intact) and re-runs `validate_normalized`, so
    // the job carries only the draft identity + the per-service placements the
    // preview was taken with — never a raw manifest. This arm stamps the confirm
    // token `cluster_job_confirmations` gates on and cannot drift from it.
    return {
      type: "cluster.app.deploy-researched",
      payload: {
        draft_id: payload?.draft_id,
        manifest_digest: payload?.manifest_digest,
        placements: payload?.placements,
        confirm: "deploy-researched-app",
      },
    };
  }
  if (action === "remove-researched-app") {
    // D4d: remove a researched multi-service app as a unit (every service, then
    // the overlay network, retaining node-local volumes). The app is found by
    // its `vaelor.app-group` label. `data_loss_ack` is carried ONLY when the
    // plan required it (a stateful member with no backup) and the operator typed
    // the exact token; a stateless-only app sends nothing extra.
    return {
      type: "cluster.app.remove-researched",
      payload: {
        confirm: "remove-researched-app",
        app_group: payload?.app_group,
        ...(payload?.data_loss_ack ? { data_loss_ack: payload.data_loss_ack } : {}),
      },
    };
  }
  if (action === "remove-service") {
    return {
      type: "cluster.service.remove",
      payload: {
        confirm: "remove-cluster-service",
        service_name: payload?.service_name,
        // Set only for a stateful remove with no backup, where the plan carried
        // the required token and the modal collected the operator's typed value.
        ...(payload?.data_loss_ack ? { data_loss_ack: payload.data_loss_ack } : {}),
      },
    };
  }
  if (["restart-service", "refresh-service", "rollback-service"].includes(action)) {
    const verb = action.replace("-service", "");
    return {
      type: "cluster.service.manage",
      payload: {
        action: verb,
        confirm: `${verb}-cluster-service`,
        service_name: payload?.service_name,
      },
    };
  }
  if (action === "configure-service") {
    return {
      type: "cluster.service.configure",
      payload: { ...payload, confirm: "configure-cluster-service" },
    };
  }
  if (action === "backup-service") {
    return {
      type: "cluster.service.backup",
      payload: {
        confirm: "backup-cluster-service",
        service_name: payload?.service_name,
      },
    };
  }
  if (action === "restore-service") {
    return {
      type: "cluster.service.restore",
      payload: {
        confirm: "restore-cluster-service",
        service_name: payload?.service_name,
        backup_id: payload?.backup_id,
      },
    };
  }
  if (action === "remove-pooled") {
    return {
      type: "cluster.pooled.remove",
      payload: { confirm: "remove-pooled-inference", name: payload?.name },
    };
  }
  if (action === "deploy-agent") {
    // F6c-2a: the Advanced agent-deploy flow. Mirrors `deploy-gpu`: the modal
    // (`AgentDeployModal`) supplies the whole payload - name, custom_agent_id,
    // custom_agent_version, model_deployment_name, node_ids ([]), mcp_grants and
    // skills - and this arm only stamps the `deploy-cluster-agent` confirm token
    // `cluster_job_confirmations` gates on, so the two cannot drift.
    // `advertise_address` and `actor` are injected server-side and never sent.
    return {
      type: "cluster.agent.deploy",
      payload: {
        ...payload,
        confirm: "deploy-cluster-agent",
      },
    };
  }
  if (action === "remove-agent") {
    // F6c. Stop a deployed cluster agent and revoke its inbound key, the exact
    // shape `cluster_job_confirmations` and `agent_pool_operations.remove` gate:
    // the `remove-cluster-agent` confirm token and the agent's `name`. The
    // backing model deployment is untouched; only this agent and its key go.
    return {
      type: "cluster.agent.remove",
      payload: { confirm: "remove-cluster-agent", name: payload?.name },
    };
  }
  return null;
}
