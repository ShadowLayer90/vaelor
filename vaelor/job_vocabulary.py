"""The job types an appliance operation may claim, and the states it moves through."""

from __future__ import annotations


#: The one cluster LLM deploy job, whatever engine the payload picks (pooled
#: CPU, GPU vLLM, single/replicated). Named because the GPU serving mode watch
#: asks the executor "is one of these running here right now" to tell a deploy
#: in flight from one that died (VD-127, B3); the dispatch in `cluster_jobs`
#: and the confirm-token map read the same name rather than re-spelling it.
CLUSTER_LLM_DEPLOY_JOB = "cluster.llm.deploy"

#: The two cluster AGENT jobs (F4b-ii-C2): deploy turns a pinned custom
#: agent into a LAN-reachable OpenAI runtime behind an inbound gate, remove
#: tears it down. Named here so the dispatch in `cluster_jobs`, the
#: confirm-token map and the projection read one name rather than respelling
#: the string at each site.
CLUSTER_AGENT_DEPLOY_JOB = "cluster.agent.deploy"
CLUSTER_AGENT_REMOVE_JOB = "cluster.agent.remove"

#: The two GPU LOAD/UNLOAD jobs (G3a): unload stops a healthy vLLM cluster
#: deployment's serving units to reclaim the GPU WITHOUT deleting the record,
#: credential or cached weights; load re-serves an unloaded one warm. Named
#: here so the dispatch, the confirm-token map and the mode watch read one
#: name. The mode reconcile counts ``cluster.gpu.load`` as an active deploy
#: (`executor_service.deploy_job_active`), so a load's ``deploying`` window is
#: left alone exactly as a deploy's is.
CLUSTER_GPU_UNLOAD_JOB = "cluster.gpu.unload"
CLUSTER_GPU_LOAD_JOB = "cluster.gpu.load"

#: The post-upgrade refresh of one serving GPU deployment (W4-D1): an Unload
#: and a Load in one job, queued by the installer for a deployment whose units
#: this release renders differently. Counted as a deploy in flight for the same
#: reason a Load is: its Load half writes the row ``deploying``.
CLUSTER_GPU_REFRESH_JOB = "cluster.gpu.refresh"

#: Lay a worker's slim profile down, or take it off (VD-194 P2). Queued by
#: this controller at join, at Recheck and by the 15-minute check, and
#: retryable: it re-reads the worker first, changes only what differs, and
#: changes no machine setting.
CLUSTER_NODE_PROFILE_JOB = "cluster.node.profile"

ALLOWED_JOB_TYPES = {
    "application.research",
    "compose.draft",
    "compose.install",
    "compose.import",
    "compose.backup",
    "compose.validate",
    "compose.pull",
    "compose.deploy",
    "compose.stop",
    "compose.start",
    "compose.restart",
    "compose.update",
    "managed.remove",
    "model.inspect",
    "model.download",
    "model.deploy",
    "model.install_release",
    "llm_server.apply",
    "phoenix.apply",
    "agent.deploy",
    "checkpoint.restore",
    "system.update.stage",
    "system.update.apply",
    "appliance.factory-reset",
    "appliance.remove-vaelor",
    "appliance.portable-import",
    "appliance.upgrade",
    "host.vnc.enable",
    "host.docker.install",
    "host.docker.repair",
    "host.memory.optimize",
    "host.gpu-memory.apply",
    "host.web-research.manage",
    "cluster.initialize",
    "cluster.node.join",
    "cluster.node.availability",
    "cluster.node.remove",
    "cluster.node.gpu-memory",
    "cluster.node.reboot",
    CLUSTER_NODE_PROFILE_JOB,
    "cluster.app.deploy",
    "cluster.app.deploy-researched",
    "cluster.app.remove-researched",
    "cluster.service.backup",
    "cluster.service.configure",
    "cluster.service.manage",
    "cluster.service.remove",
    "cluster.service.restore",
    "cluster.pooled.remove",
    "cluster.gpu.remove",
    "cluster.gpu.rotate-key",
    CLUSTER_GPU_UNLOAD_JOB,
    CLUSTER_GPU_LOAD_JOB,
    CLUSTER_GPU_REFRESH_JOB,
    CLUSTER_LLM_DEPLOY_JOB,
    CLUSTER_AGENT_DEPLOY_JOB,
    CLUSTER_AGENT_REMOVE_JOB,
    "cluster.model.pull",
    "cluster.model.remove",
}
JOB_STATES = {
    "queued",
    "validating",
    "downloading",
    "running",
    "waiting",
    "needs_approval",
    "paused",
    "rejected",
    "superseded",
    "starting",
    "healthy",
    "failed",
    "cancelling",
    "cancelled",
    "completed",
}
ACTIVE_JOB_STATES = {
    "validating",
    "downloading",
    "starting",
    "running",
    "cancelling",
}
#: CR1: the job types that are never RETRIED. Each runs a one-use plan the
#: owner staged and approved with a typed confirmation, and a retry copies the
#: original payload - plan id and confirmation included - into a new job with no
#: fresh review. While that plan is still staged (a first attempt that failed
#: before the recovery broker claimed it) the copy would run it again. The owner
#: re-plans instead; the retry route refuses with this sentence and the
#: projection reports these jobs as not retryable, so no screen offers Retry.
REPLAN_REQUIRED_JOB_TYPES = frozenset({
    "appliance.factory-reset",
    "appliance.remove-vaelor",
    "appliance.portable-import",
})
REPLAN_REQUIRED_REASON = (
    "This action ran on a plan you approved with a typed confirmation, so it "
    "is not retried. Start it again from Settings > Recovery, where you review "
    "and confirm it afresh."
)
#: The generic ``POST /jobs`` refuses these types for every account: only the
#: Settings > Recovery routes create them, after staging or approving the plan.
RECOVERY_ROUTE_REQUIRED = (
    "Start this from Settings > Recovery, where the plan is staged and "
    "confirmed; it cannot be queued directly."
)
#: What a restart-interrupted recovery job says (never "retry").
REPLAN_INTERRUPTED_MESSAGE = (
    "Interrupted when the workload service restarted. Check the appliance, "
    "then start it again from Settings > Recovery."
)

#: VD-161 (review B1): the jobs that change or restart a MACHINE on a
#: confirmation given for that one time. Like the recovery set above they are
#: never retried - a retry copies the stored payload, confirmation included,
#: into a new job with no fresh review, so one click would restart or
#: reconfigure a machine again. Unlike the recovery set they ARE created by
#: the generic job route, which is why this is a second set and not three more
#: entries in that one.
RECONFIRM_REQUIRED_JOB_TYPES = frozenset({
    "host.gpu-memory.apply",
    "cluster.node.gpu-memory",
    "cluster.node.reboot",
})
RECONFIRM_REQUIRED_REASON = (
    "This action changed or restarted a machine on a confirmation you gave "
    "for that one time, so it is not retried. Start it again from Cluster > "
    "Setup > Machine settings, where you review and confirm it afresh."
)
RECONFIRM_INTERRUPTED_MESSAGE = (
    "Interrupted when the workload service restarted. Look at the machine "
    "under Cluster > Setup > Machine settings to see what it holds now, then "
    "start it again there if it is still wanted."
)

#: Every type the store refuses to retry and the projection offers no Retry
#: for. One set, so the two cannot disagree about a type.
NEVER_RETRIED_JOB_TYPES = REPLAN_REQUIRED_JOB_TYPES | RECONFIRM_REQUIRED_JOB_TYPES


def retry_refusal(job_type: object) -> str:
    """Why ``job_type`` is never retried, or ``""`` when it may be."""
    if job_type in REPLAN_REQUIRED_JOB_TYPES:
        return REPLAN_REQUIRED_REASON
    if job_type in RECONFIRM_REQUIRED_JOB_TYPES:
        return RECONFIRM_REQUIRED_REASON
    return ""


def interrupted_message(job_type: object) -> str:
    """What a never-retried job says when a service restart cut it short."""
    if job_type in REPLAN_REQUIRED_JOB_TYPES:
        return REPLAN_INTERRUPTED_MESSAGE
    if job_type in RECONFIRM_REQUIRED_JOB_TYPES:
        return RECONFIRM_INTERRUPTED_MESSAGE
    return ""


#: The job types only an administrator may create - and so only an
#: administrator may retry (review B1: the retry route used to re-check
#: nothing). Every cluster mutation is administrator-only by prefix, so a new
#: cluster job type is gated by default, which is the safe direction.
ADMINISTRATOR_ONLY_JOB_TYPES = frozenset({
    "compose.remove",
    "managed.remove",
    "compose.import",
    "compose.draft",
    "application.research",
    "checkpoint.restore",
    "system.update.apply",
    "appliance.factory-reset",
    "appliance.remove-vaelor",
    "host.vnc.enable",
    "host.docker.install",
    "host.docker.repair",
    "host.memory.optimize",
    "host.gpu-memory.apply",
    "host.web-research.manage",
})


def administrator_only(job_type: object) -> bool:
    """Whether creating OR retrying ``job_type`` needs an administrator."""
    text = str(job_type or "")
    return text.startswith("cluster.") or text in ADMINISTRATOR_ONLY_JOB_TYPES


TERMINAL_JOB_STATES = {
    "completed", "healthy", "failed", "rejected", "cancelled", "superseded",
}
