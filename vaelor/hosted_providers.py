"""The hosted AI services AI Chat may connect to, and what each one is (VD-206).

**One table, one owner** (LESSONS 6). The broker's provider list, its purpose
table, the Connections listing, AI Chat's "where does this run" sentence and
the residency read all ask this module, so a fifth hosted service is one row
here rather than five edits that can disagree.

Each hosted service is its own provider kind rather than a preset stored as an
``openai-compatible`` profile, for two reasons the design note records:

- the broker lists a credential's kind and never its address, so only the kind
  can say, without decrypting anything, that prompts leave this machine and
  for whom (VD-206: every hosted connection is labelled);
- ``openai-compatible`` ("a server on your network") keeps its private-network
  rule, which every agent path also relies on; only these kinds reach a public
  address, and only through `hosted_transport`.

**AI Chat only** (VD-049, VD-201, VD-206 item 3): these kinds are added to the
``ai-chat`` purpose and to nothing else, and `model_credential_roles` refuses
them for the Assistant as well.

Leaf module: imports nothing that imports the broker.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
from typing import Any, Callable, Dict, List, Mapping, Optional

from .anthropic_messages import (
    ANTHROPIC_BASE_URL,
    AnthropicError,
    list_models as anthropic_models,
    probe as anthropic_probe,
)
from .credential_broker_client import CredentialError
from .hosted_transport import (
    HostedAddressError,
    UNNAMED_SERVICE,
    bearer_headers,
    checked_address,
    error_type_of,
    parse_hosted_url,
    redact,
    refusal_for,
    request,
)

ANTHROPIC = "anthropic"
GEMINI = "gemini"
OPENROUTER = "openrouter"
HOSTED_COMPATIBLE = "hosted-compatible"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Each hosted kind: the name the console shows, the company that receives
#: prompts (empty for a custom service, which is named by its host), the fixed
#: address (empty when the owner supplies one) and the protocol it speaks.
HOSTED_PROVIDERS: Dict[str, Dict[str, str]] = {
    ANTHROPIC: {
        "name": "Anthropic", "company": "Anthropic",
        "base_url": ANTHROPIC_BASE_URL, "engine": "anthropic-messages",
    },
    GEMINI: {
        "name": "Google Gemini", "company": "Google",
        "base_url": GEMINI_BASE_URL, "engine": "chat-completions",
    },
    OPENROUTER: {
        "name": "OpenRouter", "company": "OpenRouter",
        "base_url": OPENROUTER_BASE_URL, "engine": "chat-completions",
    },
    HOSTED_COMPATIBLE: {
        "name": "Hosted OpenAI-compatible service", "company": "",
        "base_url": "", "engine": "chat-completions",
    },
}

#: The hosted kinds VD-206 adds. AI Chat's purpose takes them; nothing else does.
HOSTED_KINDS = frozenset(HOSTED_PROVIDERS)

#: Every kind whose prompts leave this machine for a service on the internet:
#: the hosted kinds and the OpenAI connection that predates them.
OFF_MACHINE_KINDS = HOSTED_KINDS | {"openai"}

#: The answer ceiling a hosted model gets in AI Chat. Hosted frontier models
#: often reason before they answer and count that against the same limit, so
#: the 1,024 / 1,600 a connected server gets would leave them empty-handed.
#: It is a ceiling, not a spend: a short answer stops when it is done.
HOSTED_ANSWER_TOKENS = 8192

MAX_HOSTED_KEY_BYTES = 8192

Send = Callable[..., Any]


def provider_name(provider: str) -> str:
    """The console's name for a hosted kind, ``""`` for any other kind."""
    return HOSTED_PROVIDERS.get(str(provider), {}).get("name", "")


def destination_sentence(provider: str, base_url: str = "") -> str:
    """Where prompts and files sent to this connection go, in one sentence."""
    if provider == "openai":
        return "Prompts sent to this connection go to OpenAI."
    company = HOSTED_PROVIDERS.get(provider, {}).get("company", "")
    if company:
        return "Prompts and files sent to this connection go to {}.".format(company)
    host = urllib.parse.urlsplit(str(base_url or "")).hostname or "a hosted service"
    return "Prompts and files sent to this connection go to {}, a service on the internet.".format(host)


def validate_hosted_profile(
    secret_value: str, *, resolve: bool = True, resolver: Optional[Callable[..., Any]] = None,
) -> Dict[str, str]:
    """Validate a custom hosted service's profile ``{base_url, model, api_key}``.

    The address must pass `hosted_transport`'s rule: HTTPS, and every resolved
    address public. ``resolve=False`` checks the shape only - a lease is read on
    every answer, and the address is checked again, and pinned, when the
    request is actually made.
    """
    try:
        profile = json.loads(secret_value)
    except (TypeError, json.JSONDecodeError):
        profile = None
    if not isinstance(profile, dict):
        raise CredentialError("The hosted service profile is invalid.")
    base_url = str(profile.get("base_url", "")).strip().rstrip("/")
    model = str(profile.get("model", "")).strip()
    api_key = str(profile.get("api_key", "")).strip()
    if len(model) > 200 or len(api_key.encode("utf-8")) > MAX_HOSTED_KEY_BYTES:
        raise CredentialError("The hosted service profile is too large.")
    if not api_key:
        raise CredentialError("A hosted service needs its API key.")
    parsed = parse_hosted_url(base_url)
    if resolve:
        checked_address(parsed.hostname, parsed.port or 443, resolver)
    return {"base_url": base_url, "model": model, "api_key": api_key}


def lease_profile(provider: str, plaintext: str) -> Dict[str, str]:
    """A hosted credential as AI Chat uses it: ``base_url``, ``api_key``, ``model``."""
    if provider == HOSTED_COMPATIBLE:
        return validate_hosted_profile(plaintext, resolve=False)
    return {
        "base_url": HOSTED_PROVIDERS[provider]["base_url"],
        "api_key": str(plaintext).strip(),
        "model": "",
    }


def provider_detail(provider: str, detail: str, api_key: str) -> str:
    """The provider's own error text to show, redacted - or none at all.

    A named service's error text is worth showing once redacted. A custom
    service is unknown, and what it echoes is not ours to vouch for, so only
    its status is reported.
    """
    if provider == HOSTED_COMPATIBLE:
        return ""
    return redact(detail, api_key)


def company_of(provider: str, base_url: str) -> str:
    return (
        HOSTED_PROVIDERS.get(provider, {}).get("company")
        or urllib.parse.urlsplit(base_url).hostname
        or UNNAMED_SERVICE
    )


def _compatible_models(provider: str, profile: Mapping[str, str], company: str, send: Send) -> List[str]:
    try:
        with send(
            "GET", profile["base_url"] + "/models",
            headers=bearer_headers(profile["api_key"]), timeout=20,
        ) as response:
            payload = response.json()
    except urllib.error.HTTPError as error:
        kind, detail = error_type_of(error)
        _code, sentence = refusal_for(
            company, error.code, kind, provider_detail(provider, detail, profile["api_key"]),
        )
        raise CredentialError(sentence) from None
    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise CredentialError("{}'s model list was not in the shape Vaelor reads.".format(company))
    models: List[str] = []
    for item in items:
        model_id = str(item.get("id") or "").strip() if isinstance(item, dict) else ""
        if model_id and model_id not in models:
            models.append(model_id)
    return models


def list_models(provider: str, plaintext: str, *, send: Optional[Send] = None) -> List[str]:
    """The models a hosted connection offers, in the order the service lists them."""
    profile = lease_profile(provider, plaintext)
    if provider == ANTHROPIC:
        return anthropic_models(profile["api_key"], send=send)
    return _compatible_models(provider, profile, company_of(provider, profile["base_url"]), send or request)


def probe(provider: str, plaintext: str, *, send: Optional[Send] = None) -> Dict[str, Any]:
    """The connection test for a hosted kind: the model list, which needs the key.

    The model list is the smallest request each of these services answers
    only to a valid key, and it costs nothing - no chat is sent. A custom
    service whose list needs no key is reported as answering, not as having
    accepted the key, because that is all the test shows.
    """
    if provider == ANTHROPIC:
        return anthropic_probe(lease_profile(provider, plaintext)["api_key"], send=send)
    try:
        profile = lease_profile(provider, plaintext)
        company = company_of(provider, profile["base_url"])
        models = list_models(provider, plaintext, send=send)
    except (HostedAddressError, AnthropicError, CredentialError) as error:
        return {"ok": False, "message": str(error)}
    except (urllib.error.URLError, OSError):
        return {
            "ok": False,
            "message": "{} could not be reached from this appliance. Check its "
                       "internet connection.".format(company_of(provider, "")),
        }
    if not models:
        return {"ok": False, "message": "{} answered but listed no models.".format(company)}
    if profile["model"] and profile["model"] not in models:
        return {"ok": False, "message": "{} does not list the model {}.".format(company, profile["model"])}
    count = "{} model{}".format(len(models), "" if len(models) == 1 else "s")
    if provider == HOSTED_COMPATIBLE:
        return {"ok": True, "message": "{} answered and lists {}.".format(company, count)}
    return {"ok": True, "message": "{} accepted the API key and lists {}.".format(company, count)}
