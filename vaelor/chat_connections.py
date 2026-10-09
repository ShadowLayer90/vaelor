"""Present every AI Chat connection with where it runs.

**A local model is a connection.** ``model.deploy`` stands up an
OpenAI-compatible llama.cpp server on ``127.0.0.1`` and registers a credential
for it, so the appliance's own model already travels the same path as LM
Studio on the LAN or a hosted endpoint. The privacy distinction the interface
needs to draw is therefore a ``local`` flag on a list, not a second code path,
and grouping by it is presentation.

**Only what is established is stated.** This module used to decorate each
connection with per-model load state as well (``model_states``,
``loaded_models``, ``selected_model_loaded``) and the setup route shipped a
per-engine tier map beside it. The records it was handed come from the
credential broker's listing, which carries neither the endpoint address nor
the offered models, so the load probe never ran and every list was empty - and
no screen read any of it (ACC-113). What cannot be read is no longer sent, and
a locality the record cannot establish is reported as unknown rather than as
"leaves this appliance".
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .hosted_providers import OFF_MACHINE_KINDS, destination_sentence, provider_name
from .managed_local_credentials import PREFIX as MANAGED_LOCAL_PREFIX
from .provider_runtime import managed_local_connection


#: How a connection's ``local`` flag was decided, so a caller can tell a
#: measured answer from a structural one.
LOCAL_BY_PREFIX = "vaelor-managed"
LOCAL_BY_ADDRESS = "loopback-address"
NOT_LOCAL = "remote"
#: The record carries no endpoint address to judge by (the broker's listing
#: withholds it), so where prompts go is not stated either way.
LOCALITY_UNKNOWN = "unknown"


def connection_locality(connection: Mapping[str, Any]) -> Dict[str, Any]:
    """Whether this connection is served by this appliance, and how we know.

    Two independent signals, because either alone has a gap: a credential
    Vaelor generated for its own deploy carries the managed prefix, and any
    connection whose base URL is a loopback address is on this machine
    regardless of who created it. A hosted provider matches neither. A
    compatible endpoint whose address this record does not carry matches
    nothing, and that is reported as not known rather than as remote.
    """
    identifier = str(
        connection.get("id") or connection.get("credential_id") or ""
    )
    if identifier.startswith(MANAGED_LOCAL_PREFIX):
        return {
            "local": True,
            "local_source": LOCAL_BY_PREFIX,
            "local_reason": (
                "Vaelor deployed this model on this appliance, so prompts sent "
                "to it never leave the machine."
            ),
        }
    if connection.get("base_url") and managed_local_connection(dict(connection)):
        return {
            "local": True,
            "local_source": LOCAL_BY_ADDRESS,
            "local_reason": (
                "This endpoint is a loopback address on this appliance, so "
                "prompts sent to it never leave the machine."
            ),
        }
    if connection.get("provider") in OFF_MACHINE_KINDS:
        # VD-206: a hosted kind's destination is in its kind, so it is stated
        # even though the listing carries no address.
        return {
            "local": False,
            "local_source": NOT_LOCAL,
            "local_reason": destination_sentence(
                str(connection.get("provider")), str(connection.get("base_url") or ""),
            ),
        }
    if not connection.get("base_url"):
        return {
            "local": None,
            "local_source": LOCALITY_UNKNOWN,
            "local_reason": (
                "Vaelor did not read this endpoint's address here, so whether "
                "prompts sent to it stay on this appliance is not stated."
            ),
        }
    return {
        "local": False,
        "local_source": NOT_LOCAL,
        "local_reason": (
            "This endpoint is on another machine, so prompts sent to it leave "
            "this appliance."
        ),
    }


def describe_connections(
    connections: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Every connection the picker may offer, each with where it runs."""
    described: List[Dict[str, Any]] = []
    for connection in connections:
        entry = dict(connection)
        entry.update(connection_locality(entry))
        # The console's name for a hosted kind ("Anthropic"), not its wire id.
        entry["provider_label"] = provider_name(str(entry.get("provider") or "")) or str(
            entry.get("provider") or ""
        )
        described.append(entry)
    return described
