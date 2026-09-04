"""Validation and deterministic fallback plans for the deployment agent."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .application_deployments import classify_application_intent
from .job_secrets import contains_secret

AGENT_NAME = "Vaelor Deployment Copilot"
CATALOG_REQUESTS = {
    "grafana": ("grafana", 3000),
    "uptime kuma": ("uptime-kuma", 3001),
    "nginx": ("nginx-welcome", 8080),
}
CATALOG_TEMPLATES = {template for template, _port in CATALOG_REQUESTS.values()}
MODEL_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
MODEL_FILE_QUERY = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")


def _catalog_request(message: str) -> Optional[tuple[str, int]]:
    lower = message.lower()
    return next(
        (selection for name, selection in CATALOG_REQUESTS.items() if name in lower),
        None,
    )


def _model_inspection_payload(message: str) -> Dict[str, Any]:
    """Preserve an explicit Hugging Face repo/file request without guessing."""
    clean = " ".join(str(message).strip().split())[:500]
    tokens = clean.split()
    payload: Dict[str, Any] = {"query": clean, "runtime": "llama.cpp"}
    if not tokens or not MODEL_REPOSITORY.fullmatch(tokens[0]):
        return payload
    payload["repo"] = tokens[0]
    if len(tokens) > 1 and MODEL_FILE_QUERY.fullmatch(tokens[1]):
        if tokens[1].lower().endswith(".gguf"):
            payload["file"] = tokens[1]
        else:
            payload["file_query"] = tokens[1]
    payload["selection_required"] = True
    return payload


def validate_plan(plan: Dict[str, Any], source: str) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("The deployment agent returned an invalid plan.")
    expected_fields = {
        "summary", "rationale", "warnings", "checklist", "proposed_job"
    }
    if not expected_fields.intersection(plan):
        raise ValueError(
            "The deployment agent returned an unsupported response schema."
        )
    proposed = plan.get("proposed_job")
    if proposed is not None:
        if not isinstance(proposed, dict):
            raise ValueError("The proposed job is invalid.")
        if proposed.get("type") not in {
            "compose.install",
            "model.inspect",
            "agent.deploy",
        }:
            raise ValueError("The agent proposed an unsupported job.")
        if not isinstance(proposed.get("payload", {}), dict):
            raise ValueError("The proposed job payload is invalid.")
        if contains_secret(proposed.get("payload", {})):
            raise ValueError("The agent attempted to include a raw secret.")
        if proposed.get("type") == "model.inspect":
            payload = proposed.get("payload", {})
            if payload.get("repo") and not MODEL_REPOSITORY.fullmatch(str(payload["repo"])):
                raise ValueError("The agent proposed an invalid model repository.")
            for key in ("file", "file_query"):
                if payload.get(key) and not MODEL_FILE_QUERY.fullmatch(str(payload[key])):
                    raise ValueError("The agent proposed an invalid model file selection.")
        if (
            proposed.get("type") == "compose.install"
            and proposed.get("payload", {}).get("template") not in CATALOG_TEMPLATES
        ):
            raise ValueError("The agent proposed an app outside the reviewed catalog.")

    def text_items(name: str) -> list[str]:
        value = plan.get(name, [])
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"The deployment agent returned invalid {name}.")
        return [str(item).strip()[:300] for item in value[:8] if str(item).strip()]

    result = {
        "agent": AGENT_NAME,
        "source": source,
        "summary": str(plan.get("summary", "Deployment plan prepared."))[:800],
        "rationale": str(plan.get("rationale", ""))[:1200],
        "warnings": text_items("warnings"),
        "checklist": text_items("checklist"),
        "proposed_job": proposed,
        "approval_required": proposed is not None,
    }
    intent = plan.get("application_intent")
    if intent is not None:
        if not isinstance(intent, dict) or intent.get("type") != "application.deploy":
            raise ValueError("The deployment agent returned an invalid application intent.")
        # Reclassify from the bounded subject so a model cannot widen policy.
        classified = classify_application_intent(
            "Deploy an application server "
            + str(intent.get("application_query", ""))
        )
        if classified is None:
            raise ValueError("The deployment agent returned an invalid application intent.")
        classified.update({
            "delivery_method": (
                intent.get("delivery_method")
                if intent.get("delivery_method") in {"auto", "container"}
                else "auto"
            ),
            "confidence": (
                intent.get("confidence")
                if intent.get("confidence") in {"low", "medium", "high"}
                else "low"
            ),
        })
        result["application_intent"] = classified
    else:
        result["application_intent"] = None
    return result


def fallback_plan(message: str) -> Dict[str, Any]:
    lower = message.lower()
    application_intent = classify_application_intent(message)
    if any(
        term in lower
        for term in ("llm", "model", "gguf", "hugging face", "huggingface")
    ) or re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", message):
        inspection = _model_inspection_payload(message)
        exact = inspection.get("repo")
        plan = {
            "summary": (
                f"Check the exact files in {exact} before downloading."
                if exact else
                "Find AI models that are a good fit for this Vaelor node."
            ),
            "rationale": "Vaelor will verify repository metadata and file sizes, then compare memory and storage fit before offering an exact download.",
            "warnings": ["Some models require a Hugging Face account. The assistant will tell you before one is needed."],
            "checklist": [
                "Find models made for local use",
                "Remove choices that are too large for this device",
                "Check the model license and access requirements",
                "Show the best choices in plain language",
            ],
            "proposed_job": {
                "type": "model.inspect",
                "payload": inspection,
            },
        }
    elif "agent" in lower or "copilot" in lower or "assistant" in lower:
        plan = {
            "summary": "Set up a private local AI model for Vaelor Assistant.",
            "rationale": "The assistant already works with its built-in guide. A local model adds more detailed, conversational setup help.",
            "warnings": ["Larger AI models give better answers but respond more slowly on lower-powered hosts."],
            "checklist": [
                "Check available memory and storage",
                "Recommend a model that fits",
                "Install it privately on this device",
                "Connect the assistant and test it",
            ],
            "proposed_job": {
                "type": "agent.deploy",
                "payload": {
                    "profile": "deployment-copilot",
                    "runtime": "llama.cpp",
                },
            },
        }
    elif application_intent is not None or any(
        term in lower
        for term in (
            "app", "container", "dashboard", "docker", "compose", "grafana",
            "home assistant", "open webui", "plex", "portainer", "service",
            "server", "game server",
        )
    ):
        catalog = _catalog_request(message)
        if catalog:
            template, port = catalog
            plan = {
                "summary": "Install this supported app with appliance-safe defaults.",
                "rationale": "Vaelor will verify the default port, storage, and memory policy before it downloads or starts the app.",
                "warnings": ["Approval starts the real installation. Nothing runs before approval."],
                "checklist": [
                    "Check that the app's default port is available",
                    "Create its managed Compose project",
                    "Start it with CPU and memory limits",
                    "Verify the service becomes healthy",
                ],
                "proposed_job": {
                    "type": "compose.install",
                    "payload": {"template": template, "port": port},
                },
            }
        else:
            subject = (application_intent or {}).get("application_query") or " ".join(
                word for word in re.findall(r"[A-Za-z0-9_.-]+", message)
                if word.lower() not in {
                    "a", "an", "deploy", "i", "install", "launch", "please",
                    "run", "server", "set", "the", "to", "up", "want",
                }
            )[:80] or "this app"
            plan = {
                "summary": f"Review compatibility before deploying {subject}.",
                "rationale": "This app is not in Vaelor's reviewed one-click catalog, so the assistant will not invent an image or claim it supports this node.",
                "warnings": [
                    "A compatible image or native package must exist for this node's detected architecture.",
                    "Use the guarded Compose importer after the image, ports, storage, and license are confirmed.",
                ],
                "checklist": [
                    "Confirm support for this node's detected architecture",
                    "Confirm required ports, memory, and persistent storage",
                    "Review the source image and license",
                    "Provide a Compose file for policy validation",
                ],
                "proposed_job": None,
                "application_intent": application_intent,
            }
    else:
        plan = {
            "summary": "More deployment detail is needed.",
            "rationale": "Specify whether this is a Docker service, a Hugging Face model, or the local deployment agent.",
            "warnings": [],
            "checklist": [
                "Name the application or model",
                "Describe required ports and storage",
                "State whether it should start automatically",
            ],
            "proposed_job": None,
        }
    return validate_plan(plan, source="built-in-planner")
