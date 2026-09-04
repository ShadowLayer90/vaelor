"""Executor helpers for digest-bound researched application imports."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .application_deployments import (
    ApplicationDeploymentError,
    ApplicationDeploymentStore,
    compose_project_name,
)
from .application_features import application_features
from .application_research_server import ApplicationResearchClient
from .application_research_workflow import ResearchWorkflowError, validate_source_urls
from .application_learning import (
    ApplicationLearningError, ApplicationLearningStore,
    deployment_failure_category,
)
from .application_validation import validate_application_compose
from .copilot_setup import hardware_inventory
from .credential_broker import CredentialBrokerClient, CredentialError
from .model_connection import assistant_model_configured


def condense_docker_error(raw: Any, lead_in: str) -> str:
    """Turn multi-line docker daemon output into one owner-facing line.

    Malformed YAML, a bad image reference and a port collision all arrive as a
    wall of progress lines and stack noise with the actual cause buried in the
    middle or at the end (#205 finding 4). Shown verbatim on the appliance that
    is the *primary* message the owner reads. This keeps the single most
    relevant line - the last one that mentions an error, else the last non-empty
    line - collapses its internal whitespace, and leads with a clean sentence.
    The full detail is not the primary message; the salient line is bounded so
    it cannot become one.
    """
    text = str(raw or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "{}.".format(lead_in)
    salient = next(
        (line for line in reversed(lines) if "error" in line.lower()),
        lines[-1],
    )
    salient = re.sub(r"\s+", " ", salient)[:300]
    return "{}: {}".format(lead_in, salient) if salient else "{}.".format(lead_in)


def application_learning_store(workloads_root: Path) -> ApplicationLearningStore:
    return ApplicationLearningStore(str(
        workloads_root.parent / "applications" / "deployment-lessons.sqlite3"
    ))


def record_application_failure(
    learning: ApplicationLearningStore, draft: Optional[Dict[str, Any]],
    error: BaseException, *, rollback_completed: bool,
) -> None:
    """Persist categories, never failure text, without masking the real error."""
    if draft is None:
        return
    category = deployment_failure_category(
        error, rollback_completed=rollback_completed,
    )
    try:
        learning.record_deployment(
            draft, succeeded=False, recovery_category=category,
        )
        learning.record_recovery(
            draft, completed=rollback_completed,
            recovery_category=(
                "rollback_completed" if rollback_completed
                else "rollback_incomplete"
            ),
        )
    except (ApplicationLearningError, OSError):
        return


def resolve_application_import(
    payload: Dict[str, Any],
    store: ApplicationDeploymentStore,
    actor: str,
    broker: Optional[CredentialBrokerClient] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Resolve an approved draft without accepting browser/model Compose."""
    if not (payload.get("draft_id") or payload.get("manifest_digest")):
        return None, payload
    broker = broker or CredentialBrokerClient()
    if not application_features(assistant_model_configured(broker))["deploy"]:
        raise ValueError(
            "Researched application deployment needs a working Assistant model."
        )
    try:
        draft = store.resolve_import(
            str(payload.get("draft_id", "")),
            str(payload.get("manifest_digest", "")),
            actor,
        )
    except ApplicationDeploymentError as error:
        raise ValueError(str(error)) from error
    return draft, {
        # The manifest id only passed the looser research validator; slugify it
        # here so the executor's stricter project-name gate can never reject a
        # legitimately-researched app at the final deploy step.
        "project": compose_project_name(draft["manifest"]),
        "content": json.dumps(
            draft["compose"], sort_keys=True, separators=(",", ":")
        ),
    }


def complete_application_import(
    store: ApplicationDeploymentStore,
    draft: Optional[Dict[str, Any]],
    result: Dict[str, Any],
    actor: str,
    learning: ApplicationLearningStore | None = None,
) -> Dict[str, Any]:
    if draft is None:
        return result
    imported = store.mark_imported(draft["id"], draft["manifest_digest"], actor)
    if learning is not None:
        learning.record_deployment(imported, succeeded=True)
    return {
        **result,
        "draft_id": draft["id"],
        "manifest_digest": draft["manifest_digest"],
        "compose_digest": draft["compose_digest"],
    }


def restore_previous_compose(
    compose_file: Path,
    previous_content: Optional[str],
    start_previous: Callable[[], Any],
) -> None:
    """Restore the last known configuration after pull/start failure."""
    if previous_content is None:
        compose_file.unlink(missing_ok=True)
        return
    temporary = compose_file.parent / ".compose.restore.tmp"
    temporary.write_text(previous_content, encoding="utf-8")
    temporary.chmod(0o660)
    temporary.replace(compose_file)
    start_previous()


def wait_for_application_health(
    inspect: Callable[[], Dict[str, Any]],
    expected_services: list[str],
    timeout_seconds: int = 90,
) -> Dict[str, Any]:
    """Require every researched service to be running and health-check clean."""
    expected = set(expected_services)
    deadline = time.monotonic() + timeout_seconds
    last_detail = "No service status was returned."
    while True:
        raw = str(inspect().get("output", "")).strip()
        records: list[Dict[str, Any]] = []
        try:
            decoded = json.loads(raw)
            records = decoded if isinstance(decoded, list) else [decoded]
        except json.JSONDecodeError:
            try:
                records = [json.loads(line) for line in raw.splitlines() if line]
            except json.JSONDecodeError:
                records = []
        observed: Dict[str, Dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            service = str(record.get("Service") or record.get("service") or "")
            if service:
                observed[service] = record
        if expected and expected.issubset(observed):
            unhealthy = []
            for service in sorted(expected):
                record = observed[service]
                state = str(record.get("State") or record.get("state") or "").lower()
                health = str(record.get("Health") or record.get("health") or "").lower()
                if state != "running" or health in {"starting", "unhealthy"}:
                    unhealthy.append(f"{service}: {state or 'unknown'} {health}".strip())
            if not unhealthy:
                return {"services": sorted(expected), "status": "healthy"}
            last_detail = "; ".join(unhealthy)
        else:
            missing = sorted(expected - set(observed))
            last_detail = "Missing service status: " + ", ".join(missing)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Application services did not become healthy: " + last_detail)
        time.sleep(min(2, remaining))


def rollback_application_compose(
    compose: Callable[..., Any],
    project: Path,
    compose_file: Path,
    previous_content: Optional[str],
) -> None:
    """Remove a failed candidate before restoring its last known configuration."""
    compose(project, "down", "--remove-orphans", timeout=180)
    restore_previous_compose(
        compose_file,
        previous_content,
        lambda: compose(project, "up", "-d", "--remove-orphans", timeout=180),
    )


def execute_application_job(
    job_type: str,
    payload: Dict[str, Any],
    actor: str,
    store: ApplicationDeploymentStore,
    workloads_root: Path,
    progress: Callable[[int, str, str], None] | None = None,
    learning: ApplicationLearningStore | None = None,
    researcher: Any = None,
    broker: Optional[CredentialBrokerClient] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Run the non-deploy research/draft stages through the audited queue."""
    draft_id = str(payload.get("draft_id", ""))
    current = store.get(draft_id, actor)
    if current is None:
        raise ResearchWorkflowError("The application draft was not found.", "policy")
    features = application_features(
        assistant_model_configured(broker or CredentialBrokerClient())
    )
    if job_type == "application.research":
        if not features["research"]:
            raise ResearchWorkflowError(
                "Application research needs a working Assistant model.", "policy"
            )
        try:
            urls = validate_source_urls(payload.get("source_urls", []))
        except ValueError as error:
            raise ResearchWorkflowError(str(error), "policy") from error
        client = researcher or ApplicationResearchClient()
        progressive = getattr(client, "research_manifest_with_progress", None)
        if callable(progressive):
            contract = progressive(
                current, urls,
                lambda phase, percent, message: (
                    progress(percent, message, phase) if progress else None
                ),
            )
        elif researcher is not None:
            # Explicitly injected adapters are retained for tests and legacy
            # callers; production JobExecutor supplies the progressive owner.
            if progress:
                progress(20, "Understanding the deployment request", "interpreting")
                progress(55, "Retrieving bounded public evidence", "acquiring")
            try:
                contract = client.research_manifest(current, urls)
            except ResearchWorkflowError:
                raise
            except (OSError, RuntimeError, ValueError) as error:
                raise ResearchWorkflowError(
                    "The injected research adapter could not retrieve evidence safely.",
                    "acquisition",
                ) from error
        else:
            raise ResearchWorkflowError(
                "A progressive application-research coordinator is required.",
                "execution",
            )
        if progress:
            progress(90, "Checking device fit and immutable image identity", "validating")
        try:
            updated = store.attach_manifest(draft_id, actor, contract["manifest"])
        except (ApplicationDeploymentError, KeyError, ValueError) as error:
            raise ResearchWorkflowError(
                "Vaelor could not validate this application against the current node: {}".format(
                    str(error)[:300]
                ), "compatibility", phase="needs_input",
            ) from error
        if learning is not None:
            learning.record_research(updated)
        if progress:
            progress(98, "Research is ready for your review", "ready_for_review")
        return "completed", "Application research completed", {
            "draft_id": draft_id,
            "state": updated["state"],
            "manifest_digest": updated["manifest_digest"],
            "compatibility": updated["manifest"]["compatibility"],
            "sources": updated["manifest"]["sources"],
        }
    if job_type == "compose.draft":
        if not features["drafts"] or current.get("compose") is None:
            raise ValueError("Generate a server-owned application draft first.")
        validation = validate_application_compose(
            current["compose"], str(workloads_root), hardware_inventory()
        )
        updated = store.mark_validated(draft_id, actor, validation)
        return "completed", "Application Compose draft validated", {
            "draft_id": draft_id,
            "state": updated["state"],
            "manifest_digest": updated["manifest_digest"],
            "compose_digest": updated["compose_digest"],
            "validation": validation,
        }
    raise ValueError("The application job is not supported.")


def application_secret_environment(
    compose: Mapping[str, Any],
    broker: CredentialBrokerClient | None = None,
) -> Optional[Dict[str, str]]:
    """Resolve only purpose-bound references into a child-process env."""
    references = compose.get("x-vaelor-secret-references", {})
    if not references:
        return None
    if not isinstance(references, dict) or len(references) > 64:
        raise ValueError("Application credential references are invalid.")
    client = broker or CredentialBrokerClient(timeout_seconds=10)
    environment = dict(os.environ)
    for name, credential_id in references.items():
        if not isinstance(name, str) or not isinstance(credential_id, str):
            raise ValueError("Application credential references are invalid.")
        try:
            lease = client.resolve(credential_id, "application-deploy")
        except CredentialError as error:
            raise ValueError("An application credential could not be leased.") from error
        token = str(lease.get("token", ""))
        if not token:
            raise ValueError("An application credential is empty.")
        environment["VAELOR_CREDENTIAL_" + name] = token
    return environment


def stored_application_environment(compose_file: Path) -> Optional[Dict[str, str]]:
    try:
        compose = json.loads(compose_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return application_secret_environment(compose) if isinstance(compose, dict) else None


def application_draft_environment(
    draft: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, str]]:
    return application_secret_environment(draft["compose"]) if draft else None
