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

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .credential_broker import PROVIDERS
from .gpu_serving_target import MODE_CLUSTER
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
#: Vaelor's own GPU cluster (VD-133 1.5.5 #11): Vaelor minted this credential
#: for the cluster it deployed, so its prompts go to the enrolled machines on
#: the owner's network. Not "this appliance" alone - the controller's balancer
#: forwards to replicas on the workers - so ``local`` stays ``None`` (never the
#: "never leaves this machine" claim), but where prompts go is stated.
LOCAL_BY_CLUSTER = "vaelor-cluster"

#: What the cluster's connection says about where prompts go.
CLUSTER_LOCALITY_REASON = (
    "Vaelor deployed this model on its GPU cluster, so prompts sent to it go "
    "to the cluster's machines on your network, this one included, and no "
    "further."
)


def active_cluster_credential_id(mode_state: Any = None) -> str:
    """The connection the GPU cluster serves through while it is serving, or "".

    The mode file is the fact (VD-127): its ``cluster_credential_id`` counts
    only while it says the cluster holds the GPU (Mode B). After a teardown or
    a failed re-serve reverts to Mode A, a credential Vaelor made for the
    cluster may linger, and its endpoint no longer serves, so it is not called
    the cluster's (VD-133 1.5.5 #11 review). ``mode_state`` is a state a caller
    already read (a route with its own store); ``None`` reads the file, which
    fails safe to Mode A.
    """
    if mode_state is None:
        from .gpu_cluster_mode_state import ClusterModeStore  # the store imports this layer's peers

        try:
            mode_state = ClusterModeStore().read()
        except Exception:  # noqa: BLE001 - absence-ok: an unreadable mode is no cluster
            return ""
    if str(getattr(mode_state, "mode", "") or "") != MODE_CLUSTER:
        return ""
    return str(getattr(mode_state, "cluster_credential_id", "") or "")


def vaelor_cluster_connection(
    connection: Mapping[str, Any], cluster_credential_id: str,
) -> bool:
    """Whether this is the connection Vaelor's serving GPU cluster answers on.

    Asked of what Vaelor itself recorded, never of a name, a label or an
    address: the mode file's ``cluster_credential_id`` while the cluster
    serves. A listed row carries ``id``, a resolved lease ``credential_id``;
    both are matched, so the picker and every lease reader agree (LESSONS 6).
    """
    identifier = str(connection.get("id") or connection.get("credential_id") or "")
    return bool(cluster_credential_id) and identifier == str(cluster_credential_id)


def connection_locality(
    connection: Mapping[str, Any], cluster_credential_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Whether this connection is served by this appliance, and how we know.

    Two independent signals, because either alone has a gap: a credential
    Vaelor generated for its own deploy carries the managed prefix, and any
    connection whose base URL is a loopback address is on this machine
    regardless of who created it. A hosted provider matches neither. A
    compatible endpoint whose address this record does not carry matches
    nothing, and that is reported as not known rather than as remote - unless
    it is the GPU cluster Vaelor deployed and is serving: Vaelor made that
    endpoint, so where its prompts go is known even though the listing
    withholds the address (VD-133 1.5.5 #11). ``cluster_credential_id`` is
    :func:`active_cluster_credential_id`'s answer when the caller already has
    it ("" = no cluster serving); ``None`` reads the mode file here, so no
    caller can forget to ask.
    """
    if cluster_credential_id is None:
        cluster_credential_id = active_cluster_credential_id()
    identifier = str(
        connection.get("id") or connection.get("credential_id") or ""
    )
    if vaelor_cluster_connection(connection, cluster_credential_id):
        # Before the loopback test: the balancer's address is loopback, but
        # it forwards to replicas on other machines (w2-creds review).
        return {
            "local": None,
            "local_source": LOCAL_BY_CLUSTER,
            "local_reason": CLUSTER_LOCALITY_REASON,
        }
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
    connections: Sequence[Mapping[str, Any]], cluster_credential_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Every connection the picker may offer, each with where it runs.

    ``cluster_credential_id`` as :func:`connection_locality` takes it; the
    mode file is read once here when the caller passes ``None``.
    """
    if cluster_credential_id is None:
        cluster_credential_id = active_cluster_credential_id()
    described: List[Dict[str, Any]] = []
    for connection in connections:
        entry = dict(connection)
        entry.update(connection_locality(entry, cluster_credential_id))
        # The console's name for the kind ("Anthropic", "OpenAI-compatible
        # server"), never its wire id (VD-133 1.5.5 cosmetic: the picker read
        # "openai-compatible").
        kind = str(entry.get("provider") or "")
        entry["provider_label"] = (
            provider_name(kind) or str((PROVIDERS.get(kind) or {}).get("name") or "") or kind
        )
        described.append(entry)
    return described
