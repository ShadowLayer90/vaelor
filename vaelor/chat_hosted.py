"""How one AI Chat answer reaches a hosted service (VD-206): the adapter seam.

`ChatInference.answer` builds the turn once - system prompt, history, the
question and the retrieved passages - and, for a hosted kind, hands it here
instead of posting it with ``urllib``. This picks the engine from the kind's
row in `hosted_providers` and always goes out through `hosted_transport`, so
no hosted request can skip the address check or the pinning:

- ``chat-completions`` (Gemini, OpenRouter, a custom HTTPS service): the same
  OpenAI-shaped body every other connection gets, POSTed to the kind's address;
- ``anthropic-messages``: the native connector in `anthropic_messages`.

Either way the answer comes back as an OpenAI-shaped body, so the rest of
`answer` reads it exactly as it reads any other connection's. Every failure is
a `HostedChatError` whose sentence names the service and never the key.

Runs in the control plane, on a key leased from the broker for this one answer.
"""

from __future__ import annotations

import json
import socket
import ssl
import urllib.error
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .anthropic_messages import AnthropicError, complete as anthropic_complete
from .chat_thinking import (
    CAPABILITIES,
    ThinkingCapabilities,
    ThinkingCapability,
    ThinkingPlan,
    ThinkingSettingError,
    refusal_sentence,
    refused_setting,
)
from .credential_broker_client import CredentialError
from .hosted_providers import GEMINI, HOSTED_PROVIDERS, company_of, provider_detail
from .hosted_transport import (
    HostedAddressError,
    bearer_headers,
    error_type_of,
    refusal_for,
    request,
)

#: The status AI Chat answers with when a hosted service failed: the request
#: was fine, the service upstream of this appliance was not.
HOSTED_FAILURE_STATUS = 502


#: The prefix Google's OpenAI-compatible model list puts on every id
#: ("models/gemini-..."). Its chat examples send the bare id, so the bare id is
#: what a chat request carries (VD-209 follow-up, owner's Gemini test).
GEMINI_ID_PREFIX = "models/"


def chat_model_id(provider: str, model: str) -> str:
    """The model id as the service's chat endpoint takes it."""
    if provider == GEMINI and model.startswith(GEMINI_ID_PREFIX):
        return model[len(GEMINI_ID_PREFIX):]
    return model


def read_thinking(connection: Mapping[str, Any], model: str,
                  capabilities: Optional[ThinkingCapabilities] = None,
                  ) -> Tuple[Optional[ThinkingCapability], str]:
    """Whether ``model`` on this connection can think (VD-209), read through
    the same checked, pinned transport an answer uses."""
    return (capabilities or CAPABILITIES).read(connection, model, request)


class HostedChatError(Exception):
    """A hosted answer that failed, with the owner's sentence and a code."""

    def __init__(self, message: str, *, code: str, status: int = HOSTED_FAILURE_STATUS):
        super().__init__(message)
        self.code = code
        self.status = status


def _post_compatible(connection: Mapping[str, Any], payload: Mapping[str, Any],
                     timeout: float, send: Any) -> Tuple[int, Dict[str, Any]]:
    with send(
        "POST", str(connection["base_url"]).rstrip("/") + "/chat/completions",
        headers=bearer_headers(str(connection.get("api_key") or ""), json_body=True),
        body=json.dumps(payload).encode("utf-8"), timeout=timeout,
    ) as response:
        body = response.json()
    if not isinstance(body, dict):
        raise HostedChatError(
            "{} returned a response Vaelor could not read.".format(
                company_of(str(connection.get("provider")), str(connection.get("base_url")))),
            code="chat_model_invalid_response",
        )
    return len(json.dumps(body)), body


def answer_hosted(
    connection: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    system: str,
    history: Iterable[Mapping[str, Any]],
    question: str,
    sources: Sequence[Mapping[str, str]],
    max_tokens: int,
    timeout: float,
    send: Any = None,
    thinking: Optional[ThinkingPlan] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Send one turn to a hosted service; ``(response bytes, OpenAI-shaped body)``.

    ``thinking`` is the plan for the owner's thinking step (VD-209). For a
    chat-completions service its fields are already in ``payload``; for
    Anthropic they are handed to the native connector. A 400 that names the
    setting is reported as a refusal of that step, never retried without it.
    """
    provider = str(connection.get("provider") or "")
    company = company_of(provider, str(connection.get("base_url") or ""))
    api_key = str(connection.get("api_key") or "")
    plan = thinking or ThinkingPlan()
    model = chat_model_id(provider, str(payload.get("model") or ""))
    payload = {**payload, "model": model}
    transport = send or request
    too_slow = (
        "{} did not finish the answer within {} seconds. Retry, or choose a "
        "faster model.".format(company, int(timeout))
    )
    try:
        if HOSTED_PROVIDERS[provider]["engine"] == "anthropic-messages":
            body = anthropic_complete(
                api_key, model=model, system=system,
                history=history, question=question, sources=sources,
                max_tokens=max_tokens, timeout=timeout, send=transport,
                base_url=str(connection.get("base_url") or ""), thinking=plan.body,
            )
            return len(json.dumps(body)), body
        return _post_compatible(connection, payload, timeout, transport)
    except AnthropicError as error:
        if refused_setting(plan, error.status, error.detail):
            raise HostedChatError(
                refusal_sentence(company, model, plan, error.detail),
                code=ThinkingSettingError.code,
            ) from None
        raise HostedChatError(str(error), code=error.code) from None
    except HostedAddressError as error:
        raise HostedChatError(str(error), code="chat_hosted_address_refused") from None
    except urllib.error.HTTPError as error:
        kind, detail = error_type_of(error)
        detail = provider_detail(provider, detail, api_key)
        if refused_setting(plan, error.code, detail):
            raise HostedChatError(
                refusal_sentence(company, model, plan, detail),
                code=ThinkingSettingError.code,
            ) from None
        code, sentence = refusal_for(company, error.code, kind, detail, model)
        raise HostedChatError(sentence, code=code) from None
    except (TimeoutError, socket.timeout):
        raise HostedChatError(too_slow, code="chat_model_timeout", status=504) from None
    except urllib.error.URLError as error:
        if isinstance(error.reason, (TimeoutError, socket.timeout)):
            raise HostedChatError(too_slow, code="chat_model_timeout", status=504) from None
        raise HostedChatError(
            "Vaelor could not reach {}. Check this appliance's internet "
            "connection, then retry.".format(company),
            code="chat_connection_unreachable",
        ) from None
    except CredentialError as error:
        raise HostedChatError(str(error), code="chat_model_invalid_response") from None
    except ConnectionRefusedError:
        raise HostedChatError(
            "Vaelor could not connect to {}: the connection was refused. "
            "Retry in a moment.".format(company),
            code="chat_connection_unreachable",
        ) from None
    except ssl.SSLError:
        raise HostedChatError(
            "{} did not present a valid HTTPS certificate for its address, so "
            "Vaelor sent nothing to it.".format(company),
            code="chat_hosted_certificate",
        ) from None
    except OSError:
        raise HostedChatError(
            "Vaelor lost the connection to {} before the answer was complete. "
            "Retry.".format(company),
            code="chat_connection_unreachable",
        ) from None
