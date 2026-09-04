"""Authenticated staged application-research and deployment routes.

The API never accepts Compose text or Compose objects. Research and Compose
generation are server-owned callbacks, while this module enforces actor
ownership, feature gates, immutable state transitions, and mutation auditing.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from flask import g, request

from .api_common import ApiContext, payload
from .application_deployments import (
    LIVE_DEPLOYMENT_DRAFT_DISCARD_REFUSED,
    ApplicationDeploymentError,
    classify_application_intent,
)
from .application_features import application_features
from .application_research_capability import (
    application_research_capability,
    manifest_research_risks,
)
from .application_intent_refinement import (
    application_change_signal,
    validate_refinement,
)
from .application_learning import ApplicationLearningError


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_RAW_COMPOSE_KEYS = frozenset({
    "compose", "compose_json", "compose_yaml", "raw_compose", "yaml",
})
MAX_SOURCES = 8
MAX_ERROR = 500
MAX_REQUEST_BYTES = 64 * 1024


def register_application_routes(context: ApiContext) -> None:
    """Register the staged application workflow on an existing v2 context.

    Callback contracts:
    ``application_deployment_store`` is an ApplicationDeploymentStore-like
    object. ``application_research_manifest(draft, source_urls)`` returns
    ``{"broker_results": [...], "manifest": {...}}``. Each broker result must
    be an ApplicationResearchResult.to_dict-compatible mapping. The optional
    ``application_compose_validator(compose)`` returns a mapping whose policy
    is ``validated``. ``application_features`` may be a mapping or callable.
    Production research requests require ``job_store`` and return immediately;
    the inline research callback exists only for isolated contract tests.
    """
    blueprint = context.blueprint
    callbacks = context.callbacks
    require_auth = context.require_auth
    security = context.security

    def features() -> dict[str, bool]:
        configured = callbacks.get("application_features")
        if callable(configured):
            configured = configured()
        if configured is None:
            # Nothing wired the presence signal, so treat the model as absent
            # and keep the staged workflow off until it is configured.
            return application_features(False)
        if not isinstance(configured, Mapping):
            return {"research": False, "drafts": False, "deploy": False}
        research = bool(configured.get("research", False))
        drafts = research and bool(configured.get("drafts", False))
        deploy = drafts and bool(configured.get("deploy", False))
        return {"research": research, "drafts": drafts, "deploy": deploy}

    def store():
        value = callbacks.get("application_deployment_store")
        if value is None:
            raise _RouteFailure(
                "application_workflow_unavailable",
                "Application deployment storage is unavailable.",
                503,
            )
        return value

    def learning_store():
        return callbacks.get("application_learning_store")

    def dismiss_research_jobs(draft_id: str, actor: str) -> list[dict[str, Any]]:
        job_store = callbacks.get("job_store")
        dismiss = getattr(job_store, "dismiss_application_research", None)
        return dismiss(draft_id, actor) if callable(dismiss) else []

    def live_project_names() -> frozenset[str]:
        """Compose project names of currently-running managed applications.

        This is the liveness signal that keeps discard from removing the saved
        record of an app that is actually running. It never blocks on a runtime
        hiccup: a missing inventory or any inspection failure yields an empty
        set, because deleting a draft row cannot stop a container and refusing to
        clear the banner is the trap this change removes.
        """
        inventory = callbacks.get("workload_inventory")
        lister = getattr(inventory, "list_all", None)
        if not callable(lister):
            return frozenset()
        try:
            apps = lister().get("apps", [])
        except Exception:
            return frozenset()
        return frozenset(
            str(app["project"])
            for app in apps
            if isinstance(app, dict) and app.get("running") and app.get("project")
        )

    def present(draft: dict[str, Any]) -> dict[str, Any]:
        """Attach current model guidance without persisting provider details."""
        intent = draft.get("intent") if isinstance(draft, dict) else {}
        query = str((intent or {}).get("application_query", ""))
        manifest = draft.get("manifest") if isinstance(draft, dict) else None
        evidence_count = len((manifest or {}).get("sources", [])) if isinstance(manifest, dict) else 0
        risks = manifest_research_risks(manifest)
        resolver = callbacks.get("application_research_capability")
        guidance = (
            resolver(query, evidence_count=evidence_count, risk_flags=risks)
            if callable(resolver) else
            application_research_capability(
                query, evidence_count=evidence_count, risk_flags=risks,
            )
        )
        return {**draft, "research_capability": guidance}

    def refined_intent(message: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        resolver = callbacks.get("application_intent_refiner")
        candidate = None
        if callable(resolver):
            try:
                candidate = resolver(message)
            except Exception:
                # The one on-device model is shared with the Assistant, so a
                # busy/slow/failed resolver is the common case, not the
                # exception: LocalModelBusy, a connection reset, or a timeout
                # can surface even before the refiner's own degrade guard
                # (its connection lookup runs outside that guard). Treat ANY
                # resolver failure as "no model candidate" and let the
                # deterministic built-in parser mint the intent below. Without
                # this the draft handler's outer `except Exception` would turn a
                # transient model-busy into a workflow failure and make the whole
                # Custom Application intake unusable whenever the model is in use.
                # The intake still degrades honestly - the built-in parser reads
                # deploy intent from the phrasing and research asks the reader
                # clarifying questions when unsure, rather than erroring on
                # infrastructure the reader cannot act on.
                candidate = None
        builtin = classify_application_intent(message)
        if application_change_signal(message) and isinstance(candidate, dict):
            proposed = candidate.get("intent")
            normalized = validate_refinement(message, {
                "is_deployment_request": isinstance(proposed, dict),
                "application_name": (proposed or {}).get("application_query", "") if isinstance(proposed, dict) else "",
                "delivery_preference": (proposed or {}).get("delivery_method", "auto") if isinstance(proposed, dict) else "auto",
                "interpretation": candidate.get("interpretation", ""),
                "follow_up_questions": candidate.get("follow_up_questions", []),
            }, source=str(candidate.get("source", "selected-model"))[:80], prefer_model=True)
        elif builtin is not None:
            normalized = validate_refinement(message, None, source="built-in-parser")
        else:
            normalized = validate_refinement(message, None, source="built-in-parser")
        intent = normalized.get("intent")
        if isinstance(intent, dict):
            intent = {**intent, "refinement": {
                "source": normalized.get("source", "built-in-parser"),
                "interpretation": normalized.get("interpretation", ""),
                "follow_up_questions": normalized.get("follow_up_questions", []),
                "model_used": bool(normalized.get("model_used", False)),
            }}
        return intent, normalized

    def audit(action: str, result: str, target: str = "", **details: Any) -> None:
        security.audit(
            g.auth_session.username,
            action,
            result,
            target=str(target)[:256],
            remote_addr=request.remote_addr or "",
            details={key: value for key, value in details.items()},
        )

    def require_feature(name: str) -> None:
        if not features().get(name, False):
            raise _RouteFailure(
                "application_feature_disabled",
                "This staged application capability is not enabled.",
                409,
            )

    def body(allowed: set[str]) -> dict[str, Any]:
        if request.content_length is not None and request.content_length > MAX_REQUEST_BYTES:
            raise _RouteFailure(
                "invalid_request", "The application request is too large.", 413,
            )
        value = request.get_json(silent=True)
        if not isinstance(value, dict):
            raise _RouteFailure("invalid_request", "Send a JSON request object.", 400)
        forbidden = set(value) & _RAW_COMPOSE_KEYS
        if forbidden:
            raise _RouteFailure(
                "raw_compose_not_accepted",
                "Raw Compose is not accepted by the application workflow.",
                400,
            )
        if set(value) - allowed:
            raise _RouteFailure(
                "invalid_request",
                "The application request contains unsupported fields.",
                400,
            )
        return value

    def failure(action: str, error: BaseException, target: str = ""):
        if isinstance(error, _RouteFailure):
            route_error = error
        elif isinstance(error, ApplicationDeploymentError):
            route_error = _RouteFailure(
                "application_draft_invalid", str(error)[:MAX_ERROR], 400,
            )
        elif isinstance(error, (TypeError, ValueError)):
            route_error = _RouteFailure(
                "application_request_invalid",
                "The application workflow returned invalid data.",
                400,
            )
        else:
            route_error = _RouteFailure(
                "application_workflow_failed",
                "The application workflow could not complete safely.",
                502,
            )
        audit(action, "failure", target, code=route_error.code)
        return payload(
            error={"code": route_error.code, "message": route_error.message},
            status=route_error.status,
        )

    @blueprint.get("/applications/features")
    @require_auth("viewer")
    def application_feature_status():
        return payload(features())

    @blueprint.get("/applications/lessons")
    @require_auth("viewer")
    def application_lessons():
        ledger = learning_store()
        if ledger is None:
            return payload(
                error={
                    "code": "application_learning_unavailable",
                    "message": "Deployment lessons are unavailable.",
                },
                status=503,
            )
        try:
            limit = int(request.args.get("limit", "25"))
            app_id = str(request.args.get("app_id", ""))
            lessons = ledger.list(
                g.auth_session.username, app_id=app_id, limit=limit,
            )
        except (ApplicationLearningError, TypeError, ValueError) as error:
            return payload(
                error={
                    "code": "application_lessons_invalid",
                    "message": str(error)[:MAX_ERROR],
                },
                status=400,
            )
        return payload({"lessons": lessons, "count": len(lessons)})

    @blueprint.post("/applications/drafts")
    @require_auth("administrator", csrf=True)
    def application_draft_create():
        action = "application.draft.create"
        try:
            require_feature("research")
            request_body = body({"message"})
            message = request_body.get("message")
            if not isinstance(message, str):
                raise _RouteFailure(
                    "application_intent_invalid",
                    "Describe the application you want to deploy.",
                    400,
                )
            intent, refinement = refined_intent(message)
            if intent is None:
                questions = refinement.get("follow_up_questions", [])
                raise _RouteFailure(
                    "application_intent_not_detected",
                    str(questions[0]) if questions else
                    "Ask to deploy or install an application, service, or server.",
                    422,
                )
            result = present(store().create(g.auth_session.username, intent))
        except Exception as error:
            return failure(action, error)
        audit(action, "success", result["id"], state=result["state"])
        return payload(result, status=201)

    @blueprint.get("/applications/drafts/<draft_id>")
    @require_auth("viewer")
    def application_draft_get(draft_id: str):
        try:
            result = store().get(draft_id, g.auth_session.username)
        except (ApplicationDeploymentError, TypeError, ValueError):
            result = None
        if result is None:
            return payload(
                error={
                    "code": "application_draft_not_found",
                    "message": "The application draft was not found.",
                },
                status=404,
            )
        return payload(present(result))

    @blueprint.route(
        "/applications/drafts/<draft_id>/discard",
        methods=["POST", "DELETE"],
    )
    @require_auth("administrator", csrf=True)
    def application_draft_discard(draft_id: str):
        """Discard saved draft state; this is not active operation cancel."""
        action = "application.draft.discard"
        try:
            require_feature("drafts")
            current = store().get(draft_id, g.auth_session.username)
            if current is None:
                result = {
                    "draft_id": str(draft_id),
                    "actor": g.auth_session.username,
                    "discarded": False,
                    "already_absent": True,
                    "cleanup": {
                        "state": "completed",
                        "saved_draft_removed": False,
                    },
                }
            else:
                # Liveness, not draft state, decides protection: a wedged
                # approved-but-failed or imported-but-gone draft has no live
                # container, so its banner must be clearable. Only a draft
                # backing a running app is refused.
                try:
                    result = store().discard(
                        draft_id, g.auth_session.username,
                        live_projects=live_project_names(),
                    )
                except ApplicationDeploymentError as blocked:
                    if str(blocked) == LIVE_DEPLOYMENT_DRAFT_DISCARD_REFUSED:
                        raise _RouteFailure(
                            "application_draft_discard_prohibited",
                            str(blocked),
                            409,
                        ) from blocked
                    raise
            dismissed_jobs = dismiss_research_jobs(
                draft_id, g.auth_session.username,
            )
            result.setdefault("cleanup", {})["research_jobs_superseded"] = len(
                dismissed_jobs
            )
        except Exception as error:
            return failure(action, error, draft_id)
        audit(
            action,
            "success",
            draft_id,
            cleanup=result.get("cleanup", {}),
            active_cancel=False,
        )
        return payload(result)

    @blueprint.post("/applications/drafts/<draft_id>/research")
    @require_auth("administrator", csrf=True)
    def application_draft_research(draft_id: str):
        action = "application.draft.research"
        try:
            require_feature("research")
            request_body = body({"source_urls"})
            source_urls = _source_urls(request_body.get("source_urls"))
            current = store().get(draft_id, g.auth_session.username)
            if current is None:
                raise ApplicationDeploymentError("The application draft was not found.")
            job_store = callbacks.get("job_store")
            if job_store is not None:
                job = job_store.create(
                    "application.research",
                    g.auth_session.username,
                    {"draft_id": draft_id, "source_urls": source_urls},
                )
                audit(
                    action, "queued", draft_id,
                    job_id=job["id"],
                    deduplicated=bool(job.get("deduplicated")),
                )
                return payload({"draft": present(current), "job": job}, status=202)
            if not callbacks.get("application_inline_research_for_tests", False):
                raise _RouteFailure(
                    "application_research_queue_unavailable",
                    "Durable application research is temporarily unavailable.",
                    503,
                )
            researcher = callbacks.get("application_research_manifest")
            if not callable(researcher):
                raise _RouteFailure(
                    "application_research_unavailable",
                    "Application research is unavailable.",
                    503,
                )
            broker_contract = researcher(current, source_urls)
            manifest = _broker_manifest(broker_contract, source_urls)
            result = present(store().attach_manifest(
                draft_id, g.auth_session.username, manifest,
            ))
            ledger = learning_store()
            if ledger is not None:
                ledger.record_research(result)
        except Exception as error:
            return failure(action, error, draft_id)
        audit(
            action, "success", draft_id,
            manifest_digest=result["manifest_digest"],
            sources=len(result["manifest"].get("sources", [])),
        )
        return payload(result)

    @blueprint.post("/applications/drafts/<draft_id>/configure")
    @require_auth("administrator", csrf=True)
    def application_draft_configure(draft_id: str):
        action = "application.draft.configure"
        try:
            require_feature("drafts")
            request_body = body({"configuration"})
            configuration = request_body.get("configuration")
            if not isinstance(configuration, dict):
                raise _RouteFailure(
                    "application_configuration_invalid",
                    "Application configuration must be a JSON object.",
                    400,
                )
            # `add_ports`/`host_mounts` are operator-added capabilities (VD-117):
            # publish a port an app declared none of, or grant an allowlisted,
            # consent-gated host mount. `build_compose` validates them fully
            # (range, allowlist membership, explicit consent) - this route only
            # gates the KEY SET, so it must admit them or the guided-config UI's
            # draft POST is rejected before the backend that handles them runs.
            if set(configuration) - {
                "ports", "variables", "resources", "replace_existing",
                "add_ports", "host_mounts",
            }:
                raise _RouteFailure(
                    "application_configuration_invalid",
                    "Only reviewed ports and variables can be configured.",
                    400,
                )
            result = store().configure(
                draft_id, g.auth_session.username, configuration,
            )
            # Generating a Compose draft ends the research stage: the saved
            # research has been consumed into a configured draft and is no longer
            # resumable *as research*. Retire its research job now (as approve and
            # discard already do) so the "Saved application research" resume banner
            # -- which reads the research job, not the draft state -- stops
            # dangling a draft the reader can no longer usefully re-research.
            # Best-effort: this is cleanup, never part of the configuration. It
            # must not turn a completed configure (compose generated) into a
            # reported failure; the banner briefly re-dangling is harmless because
            # discard clears it, so any error is swallowed.
            job_store = callbacks.get("job_store")
            dismiss = getattr(job_store, "dismiss_application_research", None)
            if callable(dismiss):
                try:
                    dismiss(
                        draft_id, g.auth_session.username,
                        reason="Saved application research was configured into a deployment draft.",
                    )
                except Exception:  # noqa: BLE001 - cleanup must not fail configure
                    pass
        except Exception as error:
            return failure(action, error, draft_id)
        audit(
            action, "success", draft_id,
            compose_digest=result["compose_digest"],
        )
        return payload(result)

    @blueprint.post("/applications/drafts/<draft_id>/validate")
    @require_auth("administrator", csrf=True)
    def application_draft_validate(draft_id: str):
        action = "application.draft.validate"
        try:
            require_feature("drafts")
            body(set())
            current = store().get(draft_id, g.auth_session.username)
            if current is None:
                raise ApplicationDeploymentError("The application draft was not found.")
            if current.get("compose") is None:
                raise ApplicationDeploymentError(
                    "Generate the server-owned Compose draft before validation."
                )
            validator = callbacks.get("application_compose_validator")
            if not callable(validator):
                raise _RouteFailure(
                    "application_validation_unavailable",
                    "Deterministic Compose validation is unavailable.",
                    503,
                )
            try:
                validation = validator(current["compose"])
            except ValueError as reason:
                # The deterministic validator raises USER-ACTIONABLE reasons - a
                # published port already in use, an over-budget memory/storage
                # limit, an unpinned image. `failure()` flattens a bare ValueError
                # to a generic "returned invalid data" that hides WHY, so an
                # operator who (say) publishes a port the box already uses is told
                # nothing they can act on. Surface the specific reason instead.
                raise _RouteFailure(
                    "application_validation_failed", str(reason)[:MAX_ERROR], 400,
                ) from reason
            if not isinstance(validation, dict) or validation.get("policy") != "validated":
                raise ApplicationDeploymentError(
                    "Deterministic Compose policy validation did not pass."
                )
            result = store().mark_validated(
                draft_id, g.auth_session.username, validation,
            )
        except Exception as error:
            return failure(action, error, draft_id)
        audit(
            action, "success", draft_id,
            compose_digest=result["compose_digest"],
        )
        return payload(result)

    @blueprint.post("/applications/drafts/<draft_id>/approve")
    @require_auth("administrator", csrf=True)
    def application_draft_approve(draft_id: str):
        action = "application.draft.approve"
        try:
            require_feature("deploy")
            request_body = body({"manifest_digest"})
            digest = str(request_body.get("manifest_digest", "")).strip().lower()
            if not _MANIFEST_DIGEST.fullmatch(digest):
                raise _RouteFailure(
                    "application_digest_invalid",
                    "Provide the reviewed application manifest digest.",
                    400,
                )
            result = store().approve(draft_id, g.auth_session.username, digest)
            ledger = learning_store()
            if ledger is not None:
                ledger.record_approval(result["draft"])
            # The saved research has been consumed into a deployment, so it is no
            # longer resumable. Retire its research job now (rather than only on
            # discard) so an approved -- and possibly later failed -- draft is
            # never dangled in the resume banner as a record that can neither be
            # resumed usefully nor discarded.
            # Best-effort: retiring the research job is cleanup, not part of the
            # approval. It must never turn a completed approval (record written,
            # deploy job proposed) into a reported failure -- that would confuse
            # the operator and could prompt a duplicate approve/deploy on retry.
            # If it fails the banner may briefly re-dangle, but discard now clears
            # it, so swallowing the error is the safe degrade.
            job_store = callbacks.get("job_store")
            dismiss = getattr(job_store, "dismiss_application_research", None)
            if callable(dismiss):
                try:
                    dismiss(
                        draft_id, g.auth_session.username,
                        reason="Saved application research was approved for deployment.",
                    )
                except Exception:  # noqa: BLE001 - cleanup must not fail approve
                    pass
        except Exception as error:
            return failure(action, error, draft_id)
        audit(
            action, "success", draft_id,
            manifest_digest=digest,
            job_type=result["proposed_job"]["type"],
        )
        return payload(result)


class _RouteFailure(ValueError):
    def __init__(self, code: str, message: str, status: int) -> None:
        super().__init__(message)
        self.code = str(code)[:100]
        self.message = str(message)[:MAX_ERROR]
        self.status = max(400, min(int(status), 599))


def _source_urls(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_SOURCES:
        raise _RouteFailure(
            "application_sources_invalid",
            "Choose up to eight public application source URLs.",
            400,
        )
    result = []
    for item in value:
        if not isinstance(item, str) or not item.startswith("https://") or len(item) > 2000:
            raise _RouteFailure(
                "application_sources_invalid",
                "Application sources must be bounded HTTPS URLs.",
                400,
            )
        result.append(item)
    return result


def _broker_manifest(value: Any, requested_urls: list[str]) -> dict[str, Any]:
    """Verify the explicit broker-result contract and bind it to the manifest."""
    if not isinstance(value, dict) or set(value) != {"broker_results", "manifest"}:
        raise _RouteFailure(
            "application_research_invalid",
            "Application research returned an invalid broker result.",
            502,
        )
    broker_results = value.get("broker_results")
    manifest = value.get("manifest")
    if not isinstance(broker_results, list) or not isinstance(manifest, dict):
        raise _RouteFailure(
            "application_research_invalid",
            "Application research returned an invalid broker result.",
            502,
        )
    if not 1 <= len(broker_results) <= MAX_SOURCES:
        raise _RouteFailure(
            "application_research_invalid",
            "Application research did not return bounded source evidence.",
            502,
        )
    evidence: set[tuple[str, str]] = set()
    requested = set(requested_urls)
    for item in broker_results:
        if not isinstance(item, dict):
            raise _research_contract_error()
        if item.get("kind") not in {"document_metadata", "registry_metadata"}:
            raise _research_contract_error()
        if item.get("trust") != "untrusted" or not isinstance(item.get("metadata"), dict):
            raise _research_contract_error()
        source = item.get("source")
        if not isinstance(source, dict):
            raise _research_contract_error()
        requested_url = source.get("requested_url")
        final_url = source.get("final_url")
        digest = str(source.get("sha256", "")).lower()
        media_type = source.get("media_type")
        size = source.get("size_bytes")
        if (
            (requested and requested_url not in requested)
            or not isinstance(requested_url, str)
            or not requested_url.startswith("https://")
            or not isinstance(final_url, str)
            or not final_url.startswith("https://")
            or not _SHA256.fullmatch(digest)
            or not isinstance(media_type, str)
            or len(media_type) > 200
            or not isinstance(size, int)
            or isinstance(size, bool)
            or not 0 <= size <= 4 * 1024 * 1024
        ):
            raise _research_contract_error()
        evidence.add((final_url, "sha256:" + digest))
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise _research_contract_error()
    for source in sources:
        if (
            not isinstance(source, dict)
            or not source.get("verified")
            or (source.get("url"), str(source.get("sha256", "")).lower()) not in evidence
        ):
            raise _research_contract_error()
    return manifest


def _research_contract_error() -> _RouteFailure:
    return _RouteFailure(
        "application_research_invalid",
        "Application research evidence could not be verified.",
        502,
    )
