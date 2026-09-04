"""Model-assisted application intake with deterministic intent minting."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional

from .application_deployments import (
    APPLICATION_ACTION_CUES,
    classify_application_intent,
    looks_like_pure_question,
)
from .application_intent_skill import APPLICATION_INTENT_SKILL
from .inference_client import (
    allowed_inference_endpoint,
    chat_completion,
    inference_timeout,
    parse_model_object,
)
from .local_inference_gate import LocalModelBusy
from .model_profiles import with_structured_response_format
from .provider_runtime import (
    generation_parameters,
    managed_local_connection,
    provider_system_prompt,
    provider_user_content,
    structured_task_budget,
)


# Intake refinement is a best-effort recall widener: whatever it returns, the
# deterministic built-in parser has already minted a correct intent for an
# explicit deploy request, so waiting a long time on the shared on-device model
# buys little. When that single NPU model is busy with another request - the
# Assistant, or a second intake in another control-plane worker process, which
# the in-process LocalModelBusy gate cannot see across - the completion call
# otherwise blocks for the full structured-task budget (15 s on a 1.7B,
# 24 s on a 4B) before it degrades. This ceiling fails that busy call over in
# low single-digit seconds instead. It applies to managed-local endpoints ONLY;
# remote/connected models run their own concurrency and are not the shared
# single slot this guards, so their intent budget is left untouched.
#
# 6 seconds is chosen from measured behaviour, not guessed. A responsive managed
# local model answers short requests in 0.7-4.7 s (measured live on myhost,
# 2026-08-17, recorded in local_inference_gate; a direct curl to it answered in
# 1.5 s), and this intake task is a tiny temperature-0 classification capped at
# 72-144 tokens - smaller than those chat turns. 6 s clears the 4.7 s responsive
# ceiling with margin, so a model that WOULD answer in normal time is not cut
# off, while a busy model that will not answer quickly is detected in ~6 s
# instead of 15-24 s. The degrade path and its correct built-in-parsed result
# are unchanged - only the wait before the existing fallback shortens.
LOCAL_REFINEMENT_TIMEOUT_SECONDS = 6


# The thin one-line prompt is replaced by APPLICATION_INTENT_SKILL (#247y): a
# structured description that teaches the model what the deployment tool is for
# and how to recognise a request for it across varied phrasing, rather than
# leaning on the cue lists alone. The skill widens recall; the deterministic
# policy below still mints and guards the final intent.

# Multi-word action cues a single-word vocabulary cannot express. The verb
# vocabulary itself is APPLICATION_ACTION_CUES in application_deployments - the
# single source both gates share, so "run" and every other cue stay in step
# (LESSONS 6 / #247f).
_ACTION_PHRASE_CUES = (
    re.compile(r"\bset\s*up\b", re.I),
    re.compile(r"\bget\b.{1,100}\b(?:going|running|online)\b", re.I),
    re.compile(r"\bput\b.{1,100}\b(?:on|onto)\b.{1,80}\b(?:pi|server|machine|node)\b", re.I),
    re.compile(r"\bmake\b.{1,100}\bavailable\b", re.I),
)
# A plain description of an app to stand up carries no action verb (#247z):
# "an nginx web server that serves a static page on my LAN" names the app but
# says neither "deploy" nor "run". This matches an indefinite noun phrase headed
# by an application-noun - "a/an <up to a few words> server|app|service|..." -
# which reads as "a NEW X I want". The indefinite article is the load-bearing
# discriminator: a report about existing state ("the server is down") uses a
# definite article and does not match, and a question ("is the server up?") is
# excluded separately by looks_like_pure_question before this is consulted. The
# noun set is generic infrastructure vocabulary (per-class, no app literals, per
# the no-hardcoded-app-data rule); the model still judges deploy-vs-not and the
# deterministic policy still mints and bounds any resulting intent.
_DESCRIPTIVE_APP_PHRASE = re.compile(
    r"\b(?:a|an)\s+(?:[\w.+-]+\s+){0,6}?"
    r"(?:server|service|website|dashboard|app|application|container|"
    r"database|proxy|daemon|wiki|bot)\b",
    re.I,
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+()-]{0,119}$")
_UNHELPFUL_NAMES = frozenset({
    "unknown", "unknown application", "app", "application", "server", "service",
})
_IDENTITY_NOISE = frozenset({
    "app", "application", "container", "deploy", "deployment", "docker", "host",
    "install", "lan", "official", "private", "run", "server", "service", "setup",
    "use", "web", "website",
})


def _identity_tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(value).lower())
        if len(token) > 1 and token not in _IDENTITY_NOISE
    }


def application_change_signal(message: str) -> bool:
    """True when a message asks Vaelor to deploy or run an application.

    A single clear action cue is enough (#247d): "stream my movies to my TV"
    is a valid Jellyfin request that names no server, app, or container, so
    requiring both an action cue and an app-noun cue wrongly rejected it. The
    near-miss questions ("is the dashboard app available?") stay out because
    they carry no action cue at all.

    A verb-free but plainly descriptive request is also accepted (#247z): "an
    nginx web server that serves a static page on my LAN" describes an app to
    deploy without any action verb. It matches the indefinite app-noun phrase
    below, while a question or a report about existing state does not (see
    _DESCRIPTIVE_APP_PHRASE). This only widens which messages reach the model;
    the model still judges deploy-vs-not and policy still mints any intent.
    """
    text = " ".join(str(message or "").strip().split())
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    if words & APPLICATION_ACTION_CUES:
        return True
    if any(pattern.search(text) for pattern in _ACTION_PHRASE_CUES):
        return True
    return bool(
        not looks_like_pure_question(text)
        and _DESCRIPTIVE_APP_PHRASE.search(text)
    )


def _questions(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:3]:
        text = " ".join(str(item).strip().split())[:240]
        if text and not re.search(
            r"(?:password|token|secret|api.?key|platform|container technology|"
            r"dependenc|\bports?\b|configuration|image|architecture|"
            r"exact name|official project url)",
            text,
            re.I,
        ):
            result.append(text)
    return result


def validate_refinement(
    message: str, value: Any, *, source: str, prefer_model: bool = False,
) -> Dict[str, Any]:
    """Validate model output and let deterministic policy mint the intent."""
    builtin = classify_application_intent(message)
    allowed_fields = {
        "is_deployment_request", "application_name", "delivery_preference",
        "confidence", "interpretation", "follow_up_questions",
        "deploy", "name", "mode", "summary", "questions",
    }
    if isinstance(value, dict) and set(value) - allowed_fields:
        value = None
    if builtin is not None and not prefer_model:
        return {
            "intent": builtin, "source": "built-in-parser",
            "interpretation": "Vaelor recognized an explicit application deployment request.",
            "follow_up_questions": [], "model_used": False,
        }
    # The cue lists are a fast-path and safety net, not the sole gate (#247y).
    # On the deterministic pass they still decline a message that carries no
    # action cue. The skill-guided model path (prefer_model) is trusted to judge
    # the recall the cues miss - a request phrased so the cues do not match but a
    # person still reads it as "deploy X" - subject to the same name and
    # deterministic re-mint guards below, so recognition widens while the intent
    # is still minted and bounded by policy.
    if not prefer_model and not application_change_signal(message):
        return {
            "intent": None, "source": "built-in-parser",
            "interpretation": "No explicit application change was requested.",
            "follow_up_questions": [], "model_used": False,
        }
    requested = (
        value.get("deploy", value.get("is_deployment_request"))
        if isinstance(value, dict) else False
    )
    if not isinstance(value, dict) or requested is not True:
        if builtin is not None:
            return {
                "intent": builtin, "source": "built-in-fallback",
                "interpretation": "Vaelor recognized an explicit application deployment request.",
                "follow_up_questions": [], "model_used": False,
            }
        return {
            "intent": None, "source": source,
            "interpretation": "The request may describe an application, but deployment intent is unclear.",
            "follow_up_questions": _questions(
                value.get("questions", value.get("follow_up_questions", []))
                if isinstance(value, dict) else []
            ),
            "model_used": source == "selected-model",
        }
    name = " ".join(
        str(value.get("name", value.get("application_name", ""))).strip().split()
    )
    if builtin is not None and name.lower() in _UNHELPFUL_NAMES:
        return {
            "intent": builtin, "source": "built-in-fallback",
            "interpretation": "Vaelor kept the application identity from the explicit deployment request.",
            "follow_up_questions": [], "model_used": False,
        }
    if not _SAFE_NAME.fullmatch(name):
        if builtin is not None:
            return {
                "intent": builtin, "source": "built-in-fallback",
                "interpretation": "Vaelor kept the application identity from the explicit deployment request.",
                "follow_up_questions": [], "model_used": False,
            }
        return {
            "intent": None, "source": source,
            "interpretation": "Vaelor needs the application's exact name before research.",
            "follow_up_questions": ["What is the application's exact name or official project URL?"],
            "model_used": source == "selected-model",
        }
    if builtin is not None:
        explicit_tokens = _identity_tokens(str(builtin.get("application_query", "")))
        model_tokens = _identity_tokens(name)
        if explicit_tokens and model_tokens and explicit_tokens.isdisjoint(model_tokens):
            return {
                "intent": builtin, "source": "built-in-fallback",
                "interpretation": (
                    "Vaelor kept the application identity from the explicit request because "
                    "the selected model named a different application."
                ),
                "follow_up_questions": [], "model_used": False,
            }
    delivery = value.get("mode", value.get("delivery_preference"))
    phrase = "Deploy a Docker application server " if delivery == "container" else "Deploy an application server "
    intent = classify_application_intent(phrase + name)
    if intent is None:
        raise ValueError("Deterministic application policy rejected the refined request.")
    interpretation = " ".join(
        str(value.get("summary", value.get("interpretation", ""))).strip().split()
    )[:400]
    return {
        "intent": intent,
        "source": source,
        "interpretation": interpretation or f"Vaelor interpreted this as a request to deploy {name}.",
        "follow_up_questions": _questions(
            value.get("questions", value.get("follow_up_questions"))
        ),
        "model_used": source == "selected-model",
    }


class ApplicationIntentRefiner:
    """Use selected intelligence only when deterministic parsing is ambiguous."""

    def __init__(
        self, connection_resolver: Callable[[], Optional[Dict[str, str]]],
        timeout_seconds: int = 20,
    ):
        self.connection_resolver = connection_resolver
        self.timeout_seconds = max(5, min(int(timeout_seconds), 60))

    def refine(self, message: str) -> Dict[str, Any]:
        clean = " ".join(str(message or "").strip().split())
        if not clean or len(clean) > 4000:
            raise ValueError("Describe the application in 4,000 characters or fewer.")
        deterministic = validate_refinement(clean, None, source="built-in-parser")
        cue_signal = application_change_signal(clean)
        # A message with no action cue that also reads as a plain question is
        # declined without a model call (#247y): "what is X?" / "can this run X?"
        # is the common non-deploy, and handing obvious chatter to the flaky
        # local model is the mis-mint risk this intake guards against. Everything
        # else - cue-matched, or an unusual statement the cues miss (the #247d
        # false-rejection class) - is shown to the skill-guided model to judge,
        # rather than hard-rejected here as it was before.
        if not cue_signal and looks_like_pure_question(clean):
            return deterministic
        connection = self.connection_resolver()
        if not connection or not allowed_inference_endpoint(connection):
            # Safety net (#247v / #247w): with no model to consult, fall back to
            # the deterministic cue decision and never fabricate an intent. A
            # cue-matched request already carries its minted intent; anything the
            # cues did not match declines with a concrete next step.
            if deterministic.get("intent") is None:
                deterministic["follow_up_questions"] = [
                    "What is the application's exact name or official project URL?"
                ]
            return deterministic
        try:
            raw = self._request(clean, connection)
            return validate_refinement(
                clean, raw, source="selected-model", prefer_model=True,
            )
        except LocalModelBusy as error:
            # The single local model is answering another request. Degrade to the
            # deterministic result and, when it has nothing to offer, tell the
            # reader to retry - never crash into a generic workflow 502
            # (#247c / LESSONS 1). This is a distinct branch because "wait and
            # try again" is the right next step, not "give the exact name".
            deterministic["source"] = "built-in-fallback"
            deterministic["model_used"] = False
            if deterministic.get("intent") is None:
                deterministic["interpretation"] = str(error)
                deterministic["follow_up_questions"] = [str(error)]
            return deterministic
        except (OSError, KeyError, ValueError, urllib.error.URLError):
            # A model 502/503 reaches here as urllib HTTPError (both an OSError
            # and a URLError); a malformed /models list as the ValueError
            # _request raises for it. These are inference-endpoint failures, so
            # they decline cleanly to the built-in parser with a helpful next
            # step. A real IndexError/TypeError bug in refinement is deliberately
            # NOT caught - it must surface, not silently degrade (LESSONS 1).
            deterministic["source"] = "built-in-fallback"
            deterministic["model_used"] = False
            if deterministic.get("intent") is None:
                deterministic["follow_up_questions"] = [
                    "What is the application's exact name or official project URL?"
                ]
            return deterministic

    def _request(self, message: str, connection: Dict[str, str]) -> Dict[str, Any]:
        model = str(connection.get("model", ""))
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if connection.get("api_key"):
            headers["Authorization"] = "Bearer {}".format(connection["api_key"])
        if not model:
            request = urllib.request.Request(
                "{}/models".format(connection["base_url"]), headers=headers,
            )
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                models = json.loads(response.read(1024 * 1024).decode("utf-8"))
            try:
                model = str(models["data"][0]["id"])
            except (IndexError, KeyError, TypeError) as error:
                # An empty or malformed /models list must decline cleanly, not
                # crash (#247c / LESSONS 1). Narrowing the conversion here - to a
                # ValueError the caller's own OSError/KeyError/ValueError/URLError
                # clause degrades - keeps the outer catch from swallowing a real
                # Index/Type bug elsewhere in refinement.
                raise ValueError(
                    "The connected AI endpoint returned no usable model in its /models list."
                ) from error
        policy = structured_task_budget(connection, "application-intent")
        completion_timeout = min(
            inference_timeout(connection, self.timeout_seconds),
            policy["timeout_seconds"],
        )
        if managed_local_connection(connection):
            # Fail a busy single on-device model over to the built-in parser in
            # low single-digit seconds rather than blocking the whole
            # structured-task budget. See LOCAL_REFINEMENT_TIMEOUT_SECONDS.
            completion_timeout = min(
                completion_timeout, LOCAL_REFINEMENT_TIMEOUT_SECONDS
            )
        body = chat_completion(
            connection,
            # No schema is declared here, so no `response_format` is sent unless
            # this endpoint was measured to want one. Hard-coding
            # `{"type":"json_object"}` guaranteed a rejected round trip against
            # every server that asks for `json_schema` instead.
            with_structured_response_format({
                "model": model,
                "messages": [
                    {"role": "system", "content": provider_system_prompt(APPLICATION_INTENT_SKILL, connection)},
                    {"role": "user", "content": provider_user_content({"request": message}, connection)},
                ],
                **generation_parameters(
                    connection, max_tokens=policy["max_tokens"], temperature=0.0,
                ),
            }, connection),
            headers,
            completion_timeout,
        )
        return parse_model_object(body["choices"][0]["message"].get("content", ""))
