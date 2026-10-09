"""Administrator-only fleet enrollment and cluster planning routes."""

from __future__ import annotations

from flask import g, request

from .api_common import ApiContext, payload
from .api_cluster_retained_routes import register_cluster_retained_routes
from .api_worker_profile_routes import recheck_refusal, register_worker_profile_routes
from .credential_broker import CredentialError
from .cluster_driver import ClusterDriverError
from .ssh_transport import SshTransportError
from .cluster_gpu_sizing import INTENT_CAPACITY, build_model_spec, fit_decision
from .gpu_memory_pool import pool_fit_hint
from .hf_cached_config import source_repo, with_cached_config
from .vllm_serve_options import serving_choices
from .cluster_placement import add_controller_placement
from .cluster_plan_contract import ClusterPlanContext, ClusterPlanError
from .cluster_plan_inference import inference_runtimes
from .cluster_plans import build_cluster_plan
from .cluster_node_removal import WorkerInUse, WorkerNotReached
from .cluster_backups import BACKUP_IN_USE, restore_in_progress
from .cluster_fit_guards import apply_fit_guards, fit_pins, refresh_scope
from .cluster_link_recommendation import SPLIT_LINK_FIELD, parse_split_link, recommend_for_split
from .cluster_split_mode import (
    SPLIT_MODE_FIELD, SPLIT_TENSOR, annotate_fit, parse_split_mode, placed_devices,
)
from .cluster_store import NODE_NOT_FOUND
from .cluster_worker_telemetry import (
    TELEMETRY_IS_PROFILE, TELEMETRY_STAYS_WITH_WORKER, audit_reconcile, reconcile_sentence,
)
from .gpu_model_catalog import served_repo
from .gpu_pool_replicas import cluster_serving_view
from .model_thinking import thinking_switch
from .gpu_serving_target import deployment_unload_cause


def _unload_cause(broker, name: str) -> str:
    """An unloaded row's cause, read as the LLM Server surface reads it (VD-136)."""
    from .gpu_cluster_mode_state import ClusterModeStore

    try:
        mode_state = ClusterModeStore().read()
    except Exception:  # noqa: BLE001 - an unreadable mode file is "unknown"
        return ""
    return deployment_unload_cause(broker, mode_state, name)


def register_cluster_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    require_auth = context.require_auth
    security = context.security
    register_cluster_retained_routes(context)
    register_worker_profile_routes(context)

    @blueprint.get("/cluster")
    @require_auth("operator")
    def cluster_summary():
        manager = callbacks.get("cluster_manager")
        if manager is None:
            return payload(
                error={"code": "cluster_unavailable", "message": "Fleet management is unavailable."},
                status=503,
            )
        result = manager.summary(context.appliance_address())
        probe = callbacks.get("hardware_inventory")
        hardware = probe() if probe is not None else {}
        add_controller_placement(result, hardware)
        return payload(result)

    @blueprint.get("/cluster/capacity")
    @require_auth("operator")
    def cluster_capacity():
        """Read-only capacity ledger: per-node capacity/reserved/free + total.

        Operator-gated like the other cluster reads. The shared data layer app
        placement and distributed-GPU LLM sizing consume; it computes nothing
        destructive and takes no body.
        """
        manager = callbacks.get("cluster_manager")
        if manager is None:
            return payload(
                error={
                    "code": "cluster_unavailable",
                    "message": "Fleet management is unavailable.",
                },
                status=503,
            )
        try:
            return payload(manager.capacity_ledger())
        except (AttributeError, ClusterDriverError, SshTransportError, ValueError) as error:
            return payload(
                error={
                    "code": "cluster_capacity_unavailable",
                    "message": str(error),
                },
                status=400,
            )

    @blueprint.post("/cluster/fit")
    @require_auth("operator", csrf=True)
    def cluster_fit():
        """Will this model fit on the cluster's GPUs, and how should it run.

        The Phase-2a decision engine (`cluster_gpu_sizing`) as a v2 endpoint:
        operator-gated like the sibling reads, but a POST because it carries a
        model spec. It computes nothing destructive - it reads the same
        capacity ledger `GET /cluster/capacity` returns and answers over it -
        so it is CSRF-protected but audits nothing. The body is validated
        defensively; a malformed spec is a 400 with the reason, never a trace.
        """
        manager = callbacks.get("cluster_manager")
        if manager is None:
            return payload(
                error={
                    "code": "cluster_unavailable",
                    "message": "Fleet management is unavailable.",
                },
                status=503,
            )
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            body = {}
        try:
            nested = body.get("model")
            spec = nested if isinstance(nested, dict) else body
            link = str(body.get("link", "cross-node") or "cross-node")
            # Scope the fit to the selected workers so the preview matches the
            # deploy (VD-B3b-1). Only well-formed string ids are kept; anything
            # else is ignored, and an empty result means whole-fleet, as before.
            raw_ids = body.get("node_ids")
            node_ids = (
                [str(node_id) for node_id in raw_ids if isinstance(node_id, str) and node_id.strip()]
                if isinstance(raw_ids, list) else None
            )
            # What clustering is being asked FOR (VD-125, D6). The engine
            # answers a throughput ask honestly rather than sizing it as a
            # capacity question, and refuses a capacity ask that fits one
            # machine - both in its own words, so this preview and the deploy
            # say the same sentence.
            intent = str(body.get("intent", "") or INTENT_CAPACITY)
            # Defect C: live-refresh the operator-selected workers before
            # sizing, so a node whose GPU was freed is sized on its current
            # free memory rather than the join-time snapshot. Best-effort
            # inside the manager - a probe failure falls back to the stored
            # snapshot and never turns this preview into a 500.
            # ACC-190: nothing selected re-probes every enrolled worker, not none.
            ledger = manager.capacity_ledger(
                refresh_node_ids=refresh_scope(getattr(manager, "store", None), node_ids))
            # The launch fraction and the launch context the form will deploy
            # with (VD-129): a replica's fit is decided on the fraction and
            # every fit's KV cache is sized at the context, so the preview
            # reads the same two fields the deploy reads, and the engine's
            # defaults apply when the body carries neither.
            # The model's own config.json, from this controller's model
            # library, sizes a hybrid model's KV cache and any multi-token
            # prediction exactly as the deploy will (`hf_cached_config`).
            cached_repo, cached_revision = source_repo(body.get("model_source"))
            # Read and built ONCE: the fit sizes this spec, and the form's
            # serving choices come from the same config.json facts.
            sized = build_model_spec(with_cached_config(spec, cached_repo, cached_revision))
            def decide(over):
                return fit_decision(
                    spec, over, built=sized,
                    link=link, node_ids=node_ids, intent=intent,
                    gpu_memory_utilization=body.get("gpu_memory_utilization"),
                    max_model_len=body.get("max_model_len"),
                    mtp_tokens=body.get("mtp_tokens"),
                    vllm_image=body.get("vllm_image"),
                )

            decision = decide(ledger)
            # VD-161: a model that will not fit says so when a larger GPU
            # memory pool would change that - the same engine, asked again
            # over the largest pools these machines allow.
            pool_hint = pool_fit_hint(decision, ledger, decide)
            if pool_hint is not None:
                decision = {**decision, "gpu_memory_pool": pool_hint}
            # VD-167: pipeline unless the owner chose tensor; a tensor split
            # carries the measured note and the fastest shared link.
            store = getattr(manager, "store", None)
            split_mode = parse_split_mode(body.get(SPLIT_MODE_FIELD))
            decision = annotate_fit(decision, split_mode, sized, recommend_for_split(
                decision, store) if split_mode == SPLIT_TENSOR else None,
                device_counts=placed_devices(decision, ledger))
            # ACC-188/189: what the deploy would refuse before starting -
            # the held mode switch, a chosen link that is gone - rides
            # beside the verdict as `refusal`, in the deploy's own words.
            # The owner's link choice for a split rides the same guard.
            decision = apply_fit_guards(decision, store, deployment=body.get("name"),
                                        split_link=parse_split_link(body.get(SPLIT_LINK_FIELD)),
                                        pins=fit_pins(body), node_ids=node_ids)
            # Whether this model has a thinking switch the deploy form may
            # offer (`model_thinking`, the one owner), decided about the repo
            # the deploy would serve. Absent when the body names no source.
            # A source the deploy would refuse (a mistyped link) says nothing
            # about thinking and must not turn the whole preview into a 400:
            # the fit answers as it always did, and the deploy refuses it.
            # What the form may offer this model - the images, multi-token
            # prediction, text-only - from the profile the deploy decides with.
            decision = {**decision, "serving": serving_choices(
                cached_repo, sized.facts, vllm_image=body.get("vllm_image"),
                gpu_nodes=decision["cluster"]["nodes"],
            )}
            if body.get("model_source"):
                try:
                    repo, _revision = served_repo(body.get("model_source"))
                except ValueError:
                    repo = None
                if repo is not None:
                    decision = {**decision, "thinking": thinking_switch(repo)}
            return payload(decision)
        except (
            AttributeError, ClusterDriverError, SshTransportError,
            TypeError, ValueError,
        ) as error:
            return payload(
                error={
                    "code": "cluster_fit_unavailable",
                    "message": str(error),
                },
                status=400,
            )

    @blueprint.get("/cluster/models")
    @require_auth("operator")
    def cluster_models():
        """The cached-weights inventory joined with per-node disk accounting.

        Operator-gated like the sibling cluster reads. Lists every enrolled
        node's cached models with the bytes each holds and the node's free root
        space, so the console can show what is already on the box before a serve.
        It reads the ``model_cache`` store and node inventory only - no SSH, no
        mutation - and maps a known failure to a 400 with the reason.
        """
        operations = callbacks.get("cluster_operations")
        if operations is None:
            return payload(
                error={
                    "code": "cluster_unavailable",
                    "message": "Fleet management is unavailable.",
                },
                status=503,
            )
        try:
            return payload(operations.model_library.inventory())
        except (AttributeError, OSError, ValueError) as error:
            return payload(
                error={
                    "code": "cluster_models_unavailable",
                    "message": str(error),
                },
                status=400,
            )

    @blueprint.get("/cluster/agents")
    @require_auth("operator")
    def cluster_agents():
        """The deployed cluster agents, key-free and placement-honest (F4b-ii).

        Operator-gated like the sibling cluster reads. Lists each agent's
        endpoint, state, backing cluster model and inbound-key registry id -
        and NEVER a key value; the store holds only the id. Reads the agent
        deployment store only, with no bridge call and no mutation.
        """
        operations = callbacks.get("cluster_operations")
        if operations is None:
            return payload(
                error={
                    "code": "cluster_unavailable",
                    "message": "Fleet management is unavailable.",
                },
                status=503,
            )
        try:
            return payload({"agents": operations.list_agent_deployments()})
        except (AttributeError, OSError, ValueError) as error:
            return payload(
                error={
                    "code": "cluster_agents_unavailable",
                    "message": str(error),
                },
                status=400,
            )

    @blueprint.get("/cluster/agents/<name>")
    @require_auth("operator")
    def cluster_agent_details(name):
        operations = callbacks.get("cluster_operations")
        if operations is None:
            return payload(
                error={
                    "code": "cluster_unavailable",
                    "message": "Fleet management is unavailable.",
                },
                status=503,
            )
        try:
            view = operations.get_agent_deployment(name)
        except (AttributeError, OSError, ValueError) as error:
            return payload(
                error={
                    "code": "cluster_agents_unavailable",
                    "message": str(error),
                },
                status=400,
            )
        if view is None:
            return payload(
                error={
                    "code": "cluster_agent_not_found",
                    "message": "No cluster agent deployment goes by that name.",
                },
                status=404,
            )
        return payload(view)

    @blueprint.get("/cluster/services/<service_name>")
    @require_auth("operator")
    def cluster_service_details(service_name):
        manager = callbacks.get("cluster_manager")
        try:
            return payload(manager.driver.service_details(service_name))
        except (AttributeError, ClusterDriverError, ValueError) as error:
            return payload(
                error={
                    "code": "cluster_service_unavailable",
                    "message": str(error),
                },
                status=400,
            )

    @blueprint.get("/cluster/services/<service_name>/logs")
    @require_auth("operator")
    def cluster_service_logs(service_name):
        manager = callbacks.get("cluster_manager")
        try:
            lines = int(request.args.get("lines", 200))
            return payload(
                manager.driver.service_logs(service_name, lines=lines)
            )
        except (AttributeError, ClusterDriverError, TypeError, ValueError) as error:
            return payload(
                error={
                    "code": "cluster_service_logs_unavailable",
                    "message": str(error),
                },
                status=400,
            )

    @blueprint.post("/cluster/services/<service_name>/diagnostics")
    @require_auth("operator", csrf=True)
    def cluster_service_diagnostics(service_name):
        operations = callbacks.get("cluster_operations")
        body = request.get_json(silent=True) or {}
        try:
            # Read through the manager's brokered driver (W4d-D18): this
            # process has no docker group, so the operations' direct driver
            # answers "permission denied" here.
            details = callbacks.get("cluster_manager").driver.service_details(
                service_name
            )
            result = operations.service_diagnostics(
                service_name,
                str(body.get("tool", "stats")),
                details=details,
            )
        except (
            AttributeError, CredentialError, ClusterDriverError,
            SshTransportError, ValueError,
        ) as error:
            return payload(
                error={
                    "code": "cluster_service_diagnostic_failed",
                    "message": str(error),
                },
                status=400,
            )
        security.audit(
            g.auth_session.username,
            "cluster.service.diagnostic",
            "success",
            target=service_name,
            remote_addr=request.remote_addr or "",
            details={"tool": result["tool"], "node_id": result["node_id"]},
        )
        return payload(result)

    @blueprint.get("/cluster/backups")
    @require_auth("operator")
    def cluster_backups():
        store = callbacks.get("cluster_backups")
        if store is None:
            return payload(
                error={
                    "code": "cluster_backups_unavailable",
                    "message": "Cluster backup inventory is unavailable.",
                },
                status=503,
            )
        return payload(store.list(
            service_name=str(request.args.get("service_name", ""))[:80],
            limit=request.args.get("limit", 100),
        ))

    @blueprint.get("/cluster/backups/<backup_id>")
    @require_auth("operator")
    def cluster_backup_details(backup_id):
        store = callbacks.get("cluster_backups")
        try:
            result = store.get(
                backup_id,
                verify=str(request.args.get("verify", "")).lower()
                in {"1", "true", "yes"},
            )
            result.pop("archive_path", None)
            return payload(result)
        except (AttributeError, OSError, ValueError) as error:
            return payload(
                error={
                    "code": "cluster_backup_unavailable",
                    "message": str(error),
                },
                status=404,
            )

    @blueprint.delete("/cluster/backups/<backup_id>")
    @require_auth("administrator", csrf=True)
    def cluster_backup_delete(backup_id):
        store = callbacks.get("cluster_backups")
        body = request.get_json(silent=True) or {}
        # Review follow-up 8 (LESSONS 22): not while a restore of the
        # backup's service may still need it or its safety backup.
        jobs = callbacks.get("job_store")
        service = ""
        try:
            service = str(store.get(backup_id).get("service_name", ""))
            holder = restore_in_progress(jobs, service) if jobs is not None else ""
        except (AttributeError, OSError, ValueError):
            holder = ""  # the delete below refuses an unknown backup in its own words
        if holder:
            return payload(
                error={
                    "code": "cluster_backup_in_use",
                    "message": BACKUP_IN_USE.format(service, holder),
                },
                status=409,
            )
        try:
            result = store.delete(
                backup_id,
                confirmation=str(body.get("confirmation", "")),
            )
        except (AttributeError, OSError, ValueError) as error:
            return payload(
                error={
                    "code": "cluster_backup_delete_failed",
                    "message": str(error),
                },
                status=400,
            )
        security.audit(
            g.auth_session.username,
            "cluster.backup.delete",
            "success",
            target=backup_id,
            remote_addr=request.remote_addr or "",
            # W7-D1: the backup is gone after this row; its service names it.
            details={"service": service[:80]},
        )
        return payload(result)

    @blueprint.get("/cluster/inference/runtimes")
    @require_auth("operator")
    def cluster_inference_runtimes():
        return payload(inference_runtimes())

    @blueprint.get("/cluster/serving")
    @require_auth("administrator")
    def cluster_serving_status():
        """The internal cluster serving key's read model (F3e, design section 5).

        Fingerprint-only and internal: this key gates the balancer->worker hop
        and is never presented by a client, so nothing here reveals it. Absent
        when there is no cluster deployment; otherwise it carries the row's
        state and the watch's measured ``serving`` reading (ACC-055).
        """
        operations = callbacks.get("cluster_operations")
        broker = callbacks.get("credential_broker")
        if operations is None or broker is None:
            return payload({"present": False, "endpoint": None})
        try:
            deployments = operations.store.list_pooled_deployments()
            credentials = broker.list()
        except (AttributeError, CredentialError, OSError, ValueError) as error:
            return payload(
                error={
                    "code": "cluster_serving_unavailable",
                    "message": str(error),
                },
                status=503,
            )
        view = cluster_serving_view(
            deployments, credentials,
            unload_cause_of=lambda name: _unload_cause(broker, name),
        )
        return payload({"present": view is not None, "endpoint": view})

    @blueprint.post("/cluster/serving/rotate-key")
    @require_auth("administrator", csrf=True)
    def cluster_serving_rotate_key():
        """Enqueue an overlap-tolerant rotation of the internal cluster key (F3e).

        The rotation touches the live Mode-B serving path (worker gates and the
        balancer over SSH/the bridge), so only the executor can drive it: this
        enqueues one `cluster.gpu.rotate-key` job. Admin + CSRF; the audit carries
        no key, and no plaintext is ever returned (the key is internal). Refused
        when there is no keyed (replicated) deployment whose key can be rotated.
        """
        operations = callbacks.get("cluster_operations")
        broker = callbacks.get("credential_broker")
        job_store = callbacks.get("job_store")
        if operations is None or broker is None or job_store is None:
            return payload(
                error={
                    "code": "cluster_unavailable",
                    "message": "Fleet management is unavailable.",
                },
                status=503,
            )
        try:
            deployments = operations.store.list_pooled_deployments()
            credentials = broker.list()
        except (AttributeError, CredentialError, OSError, ValueError) as error:
            return payload(
                error={
                    "code": "cluster_serving_unavailable",
                    "message": str(error),
                },
                status=503,
            )
        view = cluster_serving_view(deployments, credentials)
        # The card now shows a row that is not serving too (ACC-055); only a
        # serving one's key can be rotated.
        if view is None or not view.get("key_present") or view.get("state") != "healthy":
            return payload(
                error={
                    "code": "cluster_serving_not_rotatable",
                    "message": (
                        "There is no keyed (replicated) cluster deployment whose "
                        "internal key can be rotated."
                    ),
                },
                status=409,
            )
        try:
            job = job_store.create(
                "cluster.gpu.rotate-key", g.auth_session.username,
                {"name": view["name"], "confirm": "rotate-cluster-key"},
            )
        except ValueError as error:
            return payload(
                error={"code": "invalid_job_request", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "cluster.serving.rotate-key", "success",
            target=view["name"], remote_addr=request.remote_addr or "",
        )
        return payload(
            {"job_id": job.get("id"), "apply": "queued", "name": view["name"]},
            status=202,
        )

    @blueprint.post("/cluster/ssh-fingerprint")
    @require_auth("administrator", csrf=True)
    def cluster_fingerprint():
        body = request.get_json(silent=True) or {}
        manager = callbacks.get("cluster_manager")
        try:
            result = manager.inspect_host(body.get("host", ""), body.get("port", 22))
        except (AttributeError, OSError, SshTransportError, ValueError) as error:
            return payload(
                error={"code": "ssh_inspection_failed", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "cluster.ssh.inspect", "success",
            target=result["host"], remote_addr=request.remote_addr or "",
        )
        return payload(result)

    @blueprint.post("/cluster/nodes")
    @require_auth("administrator", csrf=True)
    def cluster_enroll():
        body = request.get_json(silent=True) or {}
        manager = callbacks.get("cluster_manager")
        try:
            node = manager.enroll(body)
        except (
            AttributeError, CredentialError, OSError, SshTransportError, ValueError
        ) as error:
            security.audit(
                g.auth_session.username, "cluster.node.enroll", "failure",
                target=str(body.get("host", ""))[:80],
                remote_addr=request.remote_addr or "",
            )
            return payload(
                error={"code": "cluster_enrollment_failed", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "cluster.node.enroll", "success",
            target=node["id"], remote_addr=request.remote_addr or "",
        )
        return payload(node, status=201)

    @blueprint.post("/cluster/nodes/<node_id>/refresh")
    @require_auth("administrator", csrf=True)
    def cluster_refresh(node_id):
        manager = callbacks.get("cluster_manager")
        try:
            node = manager.refresh_node(node_id)
        except Exception as error:  # noqa: BLE001 - PH-R6: each kind answered in its own words
            return recheck_refusal(node_id, error)
        # ACC-121: a Recheck can restart, re-key or reinstall the worker's
        # telemetry agent; that outcome is recorded like any other change
        # rather than discarded (`audit_reconcile` decides what is recorded).
        telemetry = node.get("telemetry") or {}
        audit_reconcile(
            security.audit, g.auth_session.username, node_id,
            result=telemetry if telemetry.get("action") != "failed" else None,
            error=(
                RuntimeError(telemetry.get("message", ""))
                if telemetry.get("action") == "failed" else None
            ),
            remote_addr=request.remote_addr or "",
        )
        return payload(node)

    @blueprint.delete("/cluster/nodes/<node_id>")
    @require_auth("administrator", csrf=True)
    def cluster_remove(node_id):
        body = request.get_json(silent=True) or {}
        if body.get("confirm") != "remove-enrollment":
            return payload(
                error={
                    "code": "confirmation_required",
                    "message": "Confirm removal of this unjoined enrollment.",
                },
                status=400,
            )
        manager = callbacks.get("cluster_manager")
        try:
            removed = manager.remove_node(node_id)
        except WorkerInUse as error:
            return payload(
                error={"code": "cluster_node_in_use", "message": str(error)},
                status=409,
            )
        except WorkerNotReached as error:
            return payload(
                error={"code": "cluster_node_unreachable", "message": str(error)},
                status=409,
            )
        except ValueError as error:
            return payload(
                error={"code": "cluster_node_joined", "message": str(error)},
                status=400,
            )
        if not removed:
            return payload(
                error={"code": "cluster_node_not_found", "message": NODE_NOT_FOUND},
                status=404,
            )
        security.audit(
            g.auth_session.username, "cluster.node.remove", "success",
            target=node_id, remote_addr=request.remote_addr or "",
        )
        return payload({"removed": True})

    @blueprint.post("/cluster/nodes/<node_id>/telemetry")
    @require_auth("administrator", csrf=True)
    def cluster_worker_telemetry_install(node_id):
        # Install the lean Telegraf telemetry agent onto one enrolled worker
        # (E2b). Synchronous SSH work, exactly like enroll/refresh above: it
        # ships the pinned binary and emitter, writes the 0600 keyed config, and
        # enables the unit. The controller must have the tarballs staged
        # (install-vaelor.sh stage_worker_telegraf); a missing stage surfaces here
        # as a 400 the operator can act on, not a silent no-op.
        manager = callbacks.get("cluster_manager")
        try:
            node = manager.store.get_node(node_id) if manager is not None else None
        except ValueError:
            node = None
        if node is not None and manager.telemetry_owned_by_profile(node):
            # Review S10: one owner of a joined worker's telegraf.conf.
            return payload(error={"code": "worker_telemetry_in_profile", "message": TELEMETRY_IS_PROFILE},
                           status=409)
        try:
            written = manager.install_worker_telemetry(node_id)
        except (
            AttributeError, CredentialError, OSError, SshTransportError, ValueError
        ) as error:
            security.audit(
                g.auth_session.username, "cluster.telemetry.install", "failure",
                target=node_id, remote_addr=request.remote_addr or "",
            )
            return payload(
                error={"code": "worker_telemetry_install_failed", "message": str(error)},
                status=400,
            )
        security.audit(
            g.auth_session.username, "cluster.telemetry.install", "success",
            target=node_id, remote_addr=request.remote_addr or "",
        )
        return payload({"node": node_id, "installed": True, "paths": written})

    @blueprint.delete("/cluster/nodes/<node_id>/telemetry")
    @require_auth("administrator", csrf=True)
    def cluster_worker_telemetry_remove(node_id):
        # Owner, 2026-10-05: a worker's telemetry is part of its worker
        # software and comes off only when the machine leaves the cluster;
        # removing it alone was undone by the next 15-minute check. The refusal
        # is audited, in fixed words.
        security.audit(
            g.auth_session.username, "cluster.telemetry.remove", "failure",
            target=node_id, remote_addr=request.remote_addr or "",
        )
        return payload(
            error={"code": "worker_telemetry_stays", "message": TELEMETRY_STAYS_WITH_WORKER},
            status=409,
        )

    @blueprint.post("/cluster/nodes/<node_id>/telemetry/reconcile")
    @require_auth("administrator", csrf=True)
    def cluster_worker_telemetry_reconcile(node_id):
        # Bring a down or reporting-stale agent back without evicting an
        # unreachable worker. This is no longer a pure read: reconcile can drive
        # the reprovision path, which re-mints and revokes the node's ingest key
        # and re-ships the whole stack, so a state-changing outcome is audited.
        # A "healthy"/"absent"/"warming" tick made no change and is left
        # unaudited to avoid noise; the returned action says what it did.
        manager = callbacks.get("cluster_manager")
        try:
            result = manager.reconcile_worker_telemetry(node_id)
        except (
            AttributeError, CredentialError, OSError, SshTransportError, ValueError
        ) as error:
            # A reconcile can fail mid-reprovision, after the key was re-minted,
            # so a failure on this credential-rotating path is audited too.
            audit_reconcile(
                security.audit, g.auth_session.username, node_id, error=error,
                remote_addr=request.remote_addr or "",
            )
            return payload(
                error={"code": "worker_telemetry_reconcile_failed", "message": str(error)},
                status=400,
            )
        # The one list of outcomes that changed something (ACC-121), shared
        # with Recheck and the 15-minute self-heal.
        audit_reconcile(
            security.audit, g.auth_session.username, node_id, result=result,
            remote_addr=request.remote_addr or "",
        )
        return payload({
            "node": node_id, **result, "message": reconcile_sentence(result),
        })

    @blueprint.post("/cluster/architecture/evictions")
    @require_auth("administrator", csrf=True)
    def cluster_evict_mismatched():
        """Drain and unenrol nodes of the wrong architecture (VD-033).

        Audited per node and after the fact, so the trail records what was
        actually destroyed rather than what was asked for. A node that could
        not be classified is never a target, and a target that could not be
        removed is audited as a failure rather than quietly dropped.
        """
        operations = callbacks.get("cluster_operations")
        if operations is None:
            return payload(
                error={
                    "code": "cluster_unavailable",
                    "message": "Fleet management is unavailable.",
                },
                status=503,
            )
        body = request.get_json(silent=True) or {}
        try:
            result = operations.evict_mismatched_nodes(body)
        except (
            AttributeError, CredentialError, ClusterDriverError,
            SshTransportError, ValueError,
        ) as error:
            return payload(
                error={
                    "code": "cluster_eviction_failed",
                    "message": str(error),
                },
                status=400,
            )
        for record in result["evicted"] + result["failed"]:
            security.audit(
                g.auth_session.username,
                "cluster.node.architecture-eviction",
                "success" if record.get("removed") else "failure",
                target=str(record.get("node_id", "")),
                remote_addr=request.remote_addr or "",
                details={
                    "host": record.get("host", ""),
                    "node_architecture": record.get("architecture", ""),
                    "controller_architecture": record.get(
                        "controller_architecture", ""
                    ),
                    "reason": record.get("reason", ""),
                    "drained": bool(record.get("drained")),
                    "forced": bool(record.get("forced")),
                    "error": record.get("error", ""),
                },
            )
        return payload(result)

    @blueprint.post("/cluster/plan")
    @require_auth("administrator", csrf=True)
    def cluster_plan():
        body = request.get_json(silent=True) or {}
        manager = callbacks.get("cluster_manager")
        summary = manager.summary()
        summary["controller"]["candidate_address"] = context.appliance_address()
        probe = callbacks.get("hardware_inventory")
        hardware = probe() if probe is not None else {}
        add_controller_placement(summary, hardware)
        try:
            return payload(build_cluster_plan(
                str(body.get("action", "")).strip(),
                body,
                ClusterPlanContext(
                    manager=manager, summary=summary, callbacks=callbacks,
                    actor=g.auth_session.username,
                ),
            ))
        except ClusterPlanError as error:
            return payload(
                error={"code": error.code, "message": error.message},
                status=error.status,
            )
