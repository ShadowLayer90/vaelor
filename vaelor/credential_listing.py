"""What the owner is shown about the secrets Vaelor holds, and what they may do.

The broker's ``list`` answers "which rows exist"; the console needs a different
answer: which of those are CONNECTIONS the owner made (and so may disconnect),
which are secrets something on this appliance still uses (a cluster node's
sign-in, the credential in front of a model Vaelor runs, an app's secret), and
which are not outbound connections at all. This module is the one owner of that
answer, so the Settings list, the Assistant's saved connections and the delete
route cannot classify one row three ways (LESSONS 6).

**In use is read from the references, never assumed from the kind.** A cluster
sign-in is Vaelor's only while an enrolled node names it; an app secret only
while an alert, backup, app draft or custom-agent connector names it. A secret
nothing names is the owner's to remove. When a reference store cannot be read,
the answer is "could not check", and nothing is offered for removal on a guess
(fail closed).

**Inbound endpoint keys are not connections (ACC-133, ACC-069, ACC-111).** A
``served-endpoint`` key is one Vaelor minted for a LAN client to present to
the LLM Server or a deployed agent. It is left out of the listing - counted, so
the page can say where such keys are managed - and a delete of one is refused
by the vault itself (`vaelor.served_endpoint_keys.DELETE_REFUSED`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional

from .credential_broker import ALERT_PURPOSE_PREFIX, ASSISTANT_PURPOSE, PROVIDERS, CredentialError
from .gpu_serving_target import CLUSTER_INFERENCE_PURPOSE, MODE_CLUSTER
from .hosted_providers import HOSTED_KINDS, OFF_MACHINE_KINDS
from .managed_local_credentials import PREFIX as MANAGED_LOCAL_PREFIX
from .model_credential_roles import AI_CHAT_PURPOSE_NAME, ai_chat_may_use, assistant_may_use
from .served_endpoint_keys import DELETE_REFUSED, SERVED_ENDPOINT_PROVIDER

#: What each stored kind is, in the owner's words (never the provider id):
#: the broker's own provider names, read rather than re-spelled.
KIND_LABELS = {provider: policy["name"] for provider, policy in PROVIDERS.items()}

#: The kinds that are AI connections an owner adds from the Assistant setup.
AI_CONNECTION_KINDS = frozenset({"openai", "huggingface", "openai-compatible", *HOSTED_KINDS})

#: The kinds the broker has a real connection test for. An app secret's "test"
#: only proves it decrypts, so it is not offered as one.
TESTABLE_KINDS = frozenset({"openai", "huggingface", "openai-compatible", "ssh", *HOSTED_KINDS})

#: The manager that means "a reference store could not be read".
UNVERIFIED = "unverified"

#: Who uses a credential, and the sentence that says where it is changed. A
#: credential with no manager is the owner's own to disconnect or remove.
MANAGED_NOTES = {
    "cluster-node": (
        "An enrolled cluster node signs in with this. It is removed when the "
        "node is removed from Cluster."
    ),
    "local-model": (
        "Vaelor created this for a model it runs on this appliance. It is "
        "removed with that model."
    ),
    "cluster-model": (
        "A cluster model deployment uses this. It is removed with that "
        "deployment."
    ),
    "application": (
        "An app, alert channel, backup or custom agent uses this. Change or "
        "remove it where it is used."
    ),
    UNVERIFIED: (
        "Vaelor could not check what uses this right now, so it is not "
        "offered for removal. Try again in a moment."
    ),
}

#: The same answer for a model-server connection Vaelor did not tag with an
#: owner, naming the one store it could not read: the cluster's deployment
#: records are what say whether a cluster model uses it (V-R2-5).
UNVERIFIED_MODEL_SERVER_NOTE = (
    "Vaelor could not read the cluster's deployment records right now, so it "
    "cannot tell whether a cluster model uses this, and it is not offered for "
    "removal. Try again in a moment."
)


def manage_note(item: Mapping[str, Any], manager: Optional[str]) -> str:
    """The sentence saying who manages ``item``, naming what could not be read."""
    if manager == UNVERIFIED and str(item.get("provider", "")) == "openai-compatible":
        return UNVERIFIED_MODEL_SERVER_NOTE
    return MANAGED_NOTES.get(manager or "", "")


#: What each purpose assignment is, in the owner's words: the things that stop
#: working with a credential when it is deleted (S-A).
PURPOSE_NAMES = {
    "deployment-agent": "the Assistant",
    "ai-chat": "AI Chat conversations",
    "model-download": "model downloads",
    "hosted-agent": "hosted agents",
    "cluster-node": "a cluster node",
    "cluster-inference": "the cluster model",
    "application-deploy": "an app deployment",
    "custom-agent-connector": "a custom agent connector",
    "backup-passphrase": "scheduled backups",
    "backup-offsite": "off-site backups",
}


def purpose_name(purpose: str) -> str:
    """One purpose in plain words (alert channels are a prefix family)."""
    if purpose.startswith(ALERT_PURPOSE_PREFIX):
        return "an alert channel"
    return PURPOSE_NAMES.get(purpose, "another Vaelor feature")


#: What `/copilot/setup` says when the Assistant's lease names a connection the
#: owner added - a box set up before VD-201 item 3. It is reported, never moved
#: behind the owner's back, and the sentence names the three real ways back and
#: who each reaches (VD-202): a Vaelor model install re-points the lease for
#: everyone (`activate_managed_local`); basic mode changes only the account that
#: picks it; disconnecting the connection clears the lease for everyone.
ASSISTANT_ON_ADDED_CONNECTION = (
    "Vaelor Assistant is still using {label}, a connection you added. The "
    "Assistant now runs only on Vaelor's own model, so this is no longer "
    "supported. Installing Vaelor's model moves the Assistant back for every "
    "account. Built-in basic mode changes only your own account's answers. "
    "Disconnecting {label} takes it off the Assistant for every account, and "
    "off AI Chat too."
)

#: The same report when the lease names a model Vaelor runs for something else
#: - a cluster or pooled deployment, or AI Chat's own model (VD-202 item 1).
ASSISTANT_ON_ANOTHER_MODEL = (
    "Vaelor Assistant is using {label}, a model Vaelor runs for AI Chat or the "
    "cluster, not the Assistant's own. Installing Vaelor's model moves the "
    "Assistant back for every account. Built-in basic mode changes only your "
    "own account's answers."
)

#: The report when the broker's listing could not be read (LESSONS 1 / 8):
#: unknown, with the reason, never "none".
ASSISTANT_CONNECTION_UNREAD = "Vaelor couldn't read which model the Assistant uses: {reason}"

#: What replacing the key of a connection the Assistant still uses answers. The
#: owner was replacing a key, not assigning anything, so the sentence says what
#: is blocked and the ways out (LESSONS 10, VD-202): the copy would be a new
#: added connection, which the Assistant's lease can no longer take.
ASSISTANT_KEY_IN_USE = (
    "This key can't be replaced while Vaelor Assistant still uses this "
    "connection. Move the Assistant to Vaelor's own model or to built-in basic "
    "mode, or disconnect the connection and add it again with the new key."
)

#: What choosing a model answers for a model Vaelor runs on this appliance: its
#: model is the deploy's (the NPU's pinned FLM tag marks it as the Assistant's,
#: VD-202), so it is changed by installing a different model, not from here.
MODEL_SET_BY_VAELOR = (
    "Vaelor chose this model when it installed it. Install a different model "
    "to change it."
)


#: The roles that see a connection's name and which connection the Assistant
#: or AI Chat uses (VD-204 item 1): operators and administrators, because AI
#: Chat's picker needs them. Viewers see neither, so for a viewer the name is
#: left out of the JSON, not only off the screen (LESSONS 24). This one rule
#: decides it for `/copilot/setup` and `/agent/status`; `/ai-chat/setup` and
#: the Assistant's tools are operator-level already. Unknown roles see nothing.
NAME_SEEING_ROLES = frozenset({"operator", "administrator"})

#: The Assistant reports above without the name, for a viewer, followed by the
#: ways back - none of which a viewer has, so the sentence says who does.
ASSISTANT_ON_ADDED_CONNECTION_UNNAMED = (
    "Vaelor Assistant is still using a connection added to this appliance, not "
    "Vaelor's own model. The Assistant now runs only on Vaelor's own model, so "
    "this is no longer supported."
)
ASSISTANT_ON_ANOTHER_MODEL_UNNAMED = (
    "Vaelor Assistant is using a model Vaelor runs for AI Chat or the cluster, "
    "not the Assistant's own."
)
VIEWER_WAYS_BACK = (
    "An operator or administrator can move it back for every account by "
    "installing Vaelor's model, or choose built-in basic mode for their own "
    "account. Only an operator or administrator can see which connection it is."
)

#: AI Chat's lease still names the Assistant's NPU model - left by a release
#: before VD-202, which the vault now refuses to assign. AI Chat refuses to
#: send to it (`ChatInference._refuse_a_lease_ai_chat_may_not_use`, VD-210
#: owner rule), so AI Chat has no working model. The next NPU install
#: supersedes and deletes that credential
#: (`managed_local_credentials.activate_managed_local`), and the lease goes
#: with it. Reported, never migrated (V-R2-gap2).
AI_CHAT_ON_THE_ASSISTANT_MODEL = (
    "AI Chat is still set to the Assistant's on-device model, a setting left "
    "from an earlier release. AI Chat never uses that model, so it cannot "
    "answer until another model is chosen. The next time Vaelor installs the "
    "on-device model, this setting is removed."
)
#: Nothing holds AI Chat's lease outside cluster mode. Every way today's code
#: clears it: a new box; a connection disconnected or a model removed
#: (`workload_removal`); the cluster model taken down with no earlier model to
#: return to (`gpu_cluster_mode._restore_ai_chat`); the NPU sweep above.
AI_CHAT_HAS_NO_MODEL = (
    "AI Chat has no model connected, so it cannot answer. This is how a new "
    "appliance starts. It also happens when the connection or model AI Chat "
    "used is disconnected or removed, when the cluster model stops serving AI "
    "Chat and there was no earlier model to go back to, or when AI Chat was "
    "left on the Assistant's on-device model and that model was installed again."
)
#: Cluster mode is being entered: `_degrade_ai_chat` cleared the lease and the
#: deploy points AI Chat at the cluster once it answers (`repoint`), or gives
#: the earlier model back if it fails (`leave`). Nothing for anyone to install.
AI_CHAT_SWITCHING_TO_THE_CLUSTER = (
    "AI Chat is switching to the cluster model, which is still starting. It has "
    "no model until the cluster model answers, and then moves onto it on its "
    "own. If the cluster model does not start, AI Chat goes back to the model "
    "it had before, if it had one."
)
#: Cluster mode holds AI Chat with no deploy running and nothing assigned.
AI_CHAT_HELD_FOR_THE_CLUSTER = (
    "AI Chat is set aside for the cluster model, and nothing is assigned to it "
    "right now. Cluster > Deployments shows the cluster model's state."
)
#: What each role can do about AI Chat's model, after the sentence above it.
#: Connecting is an administrator's (`POST /credentials`); installing and
#: choosing in AI Chat are an operator's; a viewer is told who can.
AI_CHAT_NEXT_STEP = {
    "assistant-model": {
        "administrator": (
            "Choose another model in AI Chat, or connect one for AI Chat, so "
            "it can answer again."
        ),
        "operator": (
            "Choose another model in AI Chat, or have an administrator connect "
            "one, so it can answer again."
        ),
        "viewer": "An operator or administrator can choose another model for AI Chat.",
    },
    "none": {
        "administrator": (
            "Install a model for AI Chat or connect one, then choose it in AI Chat."
        ),
        "operator": (
            "Install a model for AI Chat, or have an administrator connect one, "
            "then choose it in AI Chat."
        ),
        "viewer": "An operator or administrator can give AI Chat a model.",
    },
}
#: What either unread report says when the failure carried no sentence.
NO_REASON_GIVEN = "no reason was given."

#: AI Chat's report when the broker's listing could not be read.
AI_CHAT_CONNECTION_UNREAD = "Vaelor couldn't read which model AI Chat uses: {reason}"


def sees_connection_names(role: str) -> bool:
    """Whether ``role`` sees connection names and which one each lease uses (VD-204)."""
    return str(role or "") in NAME_SEEING_ROLES


def assistant_connection(
    credentials: Iterable[Mapping[str, Any]], role: str = "",
) -> Dict[str, str]:
    """Whose model the Assistant's lease names, from the broker's listing.

    ``state`` is ``vaelor`` (the Assistant's own install), ``added`` (a
    connection the owner added), ``other`` (a model Vaelor runs for something
    else), or ``none``. Decided by `model_credential_roles.assistant_may_use`,
    the predicate the vault refuses with, so the report and the refusal cannot
    disagree about one credential (LESSONS 6, VD-202).

    ``role`` is the reader's. A :func:`sees_connection_names` role gets the
    label and the sentence naming it; a viewer gets the state, an empty label
    and a sentence without it (VD-204). The default is the narrowest reading.
    """
    named = sees_connection_names(role)
    for item in credentials:
        if ASSISTANT_PURPOSE not in (item.get("active_for") or []):
            continue
        label = str(item.get("label") or "a connection") if named else ""
        if assistant_may_use(item):
            return {"state": "vaelor", "label": label, "message": ""}
        state = (
            "added"
            if str(item.get("owner") or "").strip() or item.get("provider") in OFF_MACHINE_KINDS
            else "other"
        )
        if named:
            template = (
                ASSISTANT_ON_ADDED_CONNECTION if state == "added"
                else ASSISTANT_ON_ANOTHER_MODEL
            )
            return {"state": state, "label": label, "message": template.format(label=label)}
        lead = (
            ASSISTANT_ON_ADDED_CONNECTION_UNNAMED if state == "added"
            else ASSISTANT_ON_ANOTHER_MODEL_UNNAMED
        )
        return {"state": state, "label": "", "message": "{} {}".format(lead, VIEWER_WAYS_BACK)}
    return {"state": "none", "label": "", "message": ""}


def _with_next_step(state: str, lead: str, role: str) -> Dict[str, str]:
    """``lead`` followed by the step ``role`` can take; unknown roles get the viewer's."""
    steps = AI_CHAT_NEXT_STEP[state]
    step = steps.get(str(role or ""), steps["viewer"])
    return {"state": state, "message": "{} {}".format(lead, step)}


def ai_chat_connection(
    credentials: Iterable[Mapping[str, Any]], role: str = "", cluster: Any = None,
) -> Dict[str, str]:
    """Whether AI Chat's lease names a model, and whether it may keep it.

    ``state`` is ``connected``, ``assistant-model`` (the Assistant's NPU model,
    which the next NPU install sweeps - `model_credential_roles.ai_chat_may_use`,
    the vault's own predicate), ``switching`` (cluster mode is being entered),
    ``cluster`` (cluster mode holds AI Chat with nothing assigned) or ``none``.
    ``cluster`` is the `ClusterModeState` read; ``None`` reads as single-node.
    No label in it; the closing step is the one ``role`` can take (LESSONS 10).
    """
    for item in credentials:
        if AI_CHAT_PURPOSE_NAME not in (item.get("active_for") or []):
            continue
        if not ai_chat_may_use(item):
            return _with_next_step("assistant-model", AI_CHAT_ON_THE_ASSISTANT_MODEL, role)
        return {"state": "connected", "message": ""}
    if getattr(cluster, "mode", "") == MODE_CLUSTER:
        if getattr(cluster, "deploy_in_flight", False):
            return {"state": "switching", "message": AI_CHAT_SWITCHING_TO_THE_CLUSTER}
        return {"state": "cluster", "message": AI_CHAT_HELD_FOR_THE_CLUSTER}
    return _with_next_step("none", AI_CHAT_HAS_NO_MODEL, role)


def unread_ai_chat_connection(reason: str) -> Dict[str, str]:
    """AI Chat's report when the listing could not be read: ``unknown``, with why."""
    return {
        "state": "unknown",
        "message": AI_CHAT_CONNECTION_UNREAD.format(reason=reason or NO_REASON_GIVEN),
    }


def unread_assistant_connection(reason: str) -> Dict[str, str]:
    """The report when the listing could not be read: ``unknown``, with why."""
    return {
        "state": "unknown", "label": "",
        "message": ASSISTANT_CONNECTION_UNREAD.format(reason=reason or NO_REASON_GIVEN),
    }


def setup_connection_reports(
    broker: Any, role: str, unavailable: str, cluster: Any = None,
) -> Dict[str, Any]:
    """``assistant_connection`` and ``ai_chat_connection`` for `/copilot/setup`.

    One read of the broker's listing answers both, so the two reports describe
    the same moment. ``unavailable`` is the sentence for a missing broker; a
    listing that raises `CredentialError` reports its own sentence. Either way
    both are ``unknown``, never ``none`` (LESSONS 8).
    """
    reason = unavailable
    if broker is not None:
        try:
            listed = list(broker.list())
        except CredentialError as error:
            reason = str(error)
        else:
            return {
                "assistant_connection": assistant_connection(listed, role),
                "ai_chat_connection": ai_chat_connection(listed, role, cluster),
            }
    return {
        "assistant_connection": unread_assistant_connection(reason),
        "ai_chat_connection": unread_ai_chat_connection(reason),
    }


#: What a viewer reads instead of a recorded failure sentence (VD-204). The
#: sentence a failed run records can name the server's address (an agent run's
#: "For operators: ... could not reach <address>") or the connection's label
#: (AI Chat's "lost the connection to <label>"), and both are cached against
#: the connection the Assistant and the engines are probed through.
ASSISTANT_NOT_ANSWERING_FOR_VIEWERS = (
    "The Assistant's model is not answering right now. An operator or "
    "administrator can see why."
)
ENGINE_NOT_ANSWERING_FOR_VIEWERS = (
    "This engine's model server is not answering right now. An operator or "
    "administrator can see why."
)


def redact_agent_status_for_role(status: Dict[str, Any], role: str) -> Dict[str, Any]:
    """`/agent/status` as ``role`` may read it (VD-204).

    A viewer keeps every state and the model's name, and loses the connection's
    label (``provider`` becomes the model's name, which is the truth), the
    server's address (``endpoint``, ``model_profile.endpoint``) and the
    recorded failure sentence, which can name either
    (:data:`ASSISTANT_NOT_ANSWERING_FOR_VIEWERS` replaces it). Other roles get
    it as it is.
    """
    if sees_connection_names(role):
        return status
    redacted = dict(status)
    if redacted.get("configured"):
        redacted["provider"] = str(redacted.get("model") or "")
    if redacted.get("unreachable_reason"):
        redacted["unreachable_reason"] = ASSISTANT_NOT_ANSWERING_FOR_VIEWERS
    redacted.pop("endpoint", None)
    profile = redacted.get("model_profile")
    if isinstance(profile, Mapping):
        redacted["model_profile"] = {
            key: value for key, value in profile.items() if key != "endpoint"
        }
    return redacted


def redact_inference_status_for_role(payload: Dict[str, Any], role: str) -> Dict[str, Any]:
    """`/inference/status` as ``role`` may read it (VD-204).

    Each engine's ``health.detail`` for an engine that did not answer is the
    probe's sentence, and a cached failure sentence there can name a
    connection's label or its server's address; ``connection_id`` says which
    connection the engine runs. A viewer gets
    :data:`ENGINE_NOT_ANSWERING_FOR_VIEWERS` and no id; the state, the model's
    name and every fixed sentence stay.
    """
    if sees_connection_names(role):
        return payload
    engines = []
    for engine in payload.get("engines") or []:
        engine = dict(engine)
        engine["connection_id"] = None
        health = engine.get("health")
        if isinstance(health, Mapping) and health.get("state") == "unreachable":
            engine["health"] = {**health, "detail": ENGINE_NOT_ANSWERING_FOR_VIEWERS}
        engines.append(engine)
    return {**payload, "engines": engines}


def used_by(item: Mapping[str, Any]) -> list:
    """The plain names of everything assigned to use this credential, in order."""
    names = []
    for purpose in item.get("active_for") or []:
        name = purpose_name(str(purpose))
        if name not in names:
            names.append(name)
    return names


#: What a Test answers for the cluster model's credential while the model is
#: resting on purpose: nothing is serving, so a test could only fail, and that
#: failure would be recorded against a credential that is fine.
RESTING_TEST_SENTENCES = {
    "asleep": (
        "Not tested: the cluster model is asleep after sitting idle, so "
        "nothing is serving to test against. The next AI Chat message or "
        "inference-gateway request wakes it; test it once it is serving."
    ),
    "unloaded": (
        "Not tested: the cluster model was unloaded by hand, so nothing is "
        "serving to test against. Load it from Cluster > Deployments first."
    ),
    "unloaded-unknown": (
        "Not tested: the cluster model is unloaded and why could not be read, "
        "so nothing is serving to test against. Load it from Cluster > "
        "Deployments first."
    ),
    "starting": (
        "Not tested: the cluster model is still loading. Test it once it is "
        "serving."
    ),
}


def resting_test_answer(credential_id: str, rest: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The answer to Test while the cluster model rests, or ``None`` to test.

    ``rest`` is `inference_gateway.cluster_model_rest_state`'s reading.
    """
    if rest.get("state") == "unloaded":
        state = {"unloaded-idle": "asleep", "unloaded-manual": "unloaded"}.get(
            str(rest.get("reason") or ""), "unloaded-unknown"
        )
    elif rest.get("state") == "deploying":
        state = "starting"
    else:
        return None
    return {
        "id": credential_id, "ok": False, "tested": False, "state": state,
        "message": RESTING_TEST_SENTENCES[state],
    }


@dataclass(frozen=True)
class References:
    """The credential ids each kind of user names; ``None`` = could not be read."""

    nodes: Optional[FrozenSet[str]] = frozenset()
    deployments: Optional[FrozenSet[str]] = frozenset()
    app_secrets: Optional[FrozenSet[str]] = frozenset()


def managed_by(item: Mapping[str, Any], refs: References = References()) -> Optional[str]:
    """What uses this credential, :data:`UNVERIFIED`, or ``None`` when nothing does.

    Decided from facts, never from the label: a node record, a pooled
    deployment record, the cluster-inference assignment, a purpose assignment,
    an app draft or connector reference, the id prefix Vaelor gives the
    credentials it creates for its own local models, and - for a model server -
    the owner an administrator's add records, which no Vaelor deploy writes.
    """
    provider = str(item.get("provider", ""))
    credential_id = str(item.get("id", ""))
    if provider == "ssh":
        if refs.nodes is None:
            return UNVERIFIED
        return "cluster-node" if credential_id in refs.nodes else None
    if provider == "application-secret":
        if item.get("active_for"):
            return "application"
        if refs.app_secrets is None:
            return UNVERIFIED
        return "application" if credential_id in refs.app_secrets else None
    if credential_id.startswith(MANAGED_LOCAL_PREFIX):
        return "local-model"
    if provider == "openai-compatible":
        # The facts first, whoever owns the row: a key replaced through
        # `POST /credentials` carries the cluster's assignment onto a row with
        # an owner, and a deployment record naming it still means it is in use.
        if CLUSTER_INFERENCE_PURPOSE in (item.get("active_for") or []):
            return "cluster-model"
        if refs.deployments is not None:
            return "cluster-model" if credential_id in refs.deployments else None
        if str(item.get("owner") or "").strip():
            # V-R2-5: only when the cluster store could not be read. A
            # connection an administrator added carries their name, and no
            # Vaelor deploy writes one, so its actions are not withheld.
            return None
        return UNVERIFIED
    return None


def describe(item: Mapping[str, Any], refs: References = References()) -> Dict[str, Any]:
    """The owner-facing fields for one stored credential, beside its metadata."""
    manager = managed_by(item, refs)
    provider = str(item.get("provider", ""))
    return {
        "kind_label": KIND_LABELS.get(provider, "Stored secret"),
        "ai_connection": provider in AI_CONNECTION_KINDS,
        # VD-206: every hosted connection is labelled as sending prompts and
        # files off this machine; the kind says so without decrypting it.
        "sends_off_machine": provider in OFF_MACHINE_KINDS,
        "testable": provider in TESTABLE_KINDS,
        "managed_by": manager,
        "manage_note": manage_note(item, manager),
        "used_by": used_by(item),
        "can_disconnect": manager is None,
    }


def connection_listing(
    credentials: Iterable[Mapping[str, Any]], refs: References = References(),
) -> Dict[str, Any]:
    """The Connections listing: outbound credentials, and a count of inbound keys.

    ``credentials`` is the broker's ``list`` output. Served-endpoint keys are
    counted into ``endpoint_keys`` (active and revoked separately) and left out
    of ``credentials``; every other row keeps all of its metadata and gains the
    fields :func:`describe` adds.
    """
    listed = []
    endpoint_keys = {"active": 0, "revoked": 0}
    for item in credentials:
        if str(item.get("provider", "")) == SERVED_ENDPOINT_PROVIDER:
            endpoint_keys["revoked" if item.get("revoked") else "active"] += 1
            continue
        listed.append({**item, **describe(item, refs)})
    return {"credentials": listed, "endpoint_keys": endpoint_keys}


def delete_refusal(item: Mapping[str, Any], refs: References = References()) -> tuple:
    """``(http_status, sentence)`` refusing removal from a connections list, or ``(0, "")``.

    409 when something uses the credential (the sentence says what); 503 when a
    reference store could not be read, so whether anything uses it is unknown.
    """
    if str(item.get("provider", "")) == SERVED_ENDPOINT_PROVIDER:
        return 409, DELETE_REFUSED
    manager = managed_by(item, refs)
    if manager is None:
        return 0, ""
    if manager == UNVERIFIED:
        return 503, manage_note(item, manager)
    return 409, "This is in use, so it is not removed from here. " + MANAGED_NOTES[manager]


def _ids(rows: Iterable[Mapping[str, Any]], field: str) -> FrozenSet[str]:
    return frozenset(str(row.get(field) or "") for row in rows or [] if row.get(field))


def read_references(callbacks: Mapping[str, Any], actor: str) -> References:
    """Collect every store's references. An absent store names nothing; a store
    that raises is ``None`` (could not check), so its kind fails closed."""

    def read(name: str, collect):
        source = callbacks.get(name)
        if source is None:
            return frozenset()
        try:
            return frozenset(collect(source))
        except Exception:  # noqa: BLE001 - "could not check" is the recorded answer
            return None

    nodes = read("cluster_manager", lambda manager: manager.store.node_credential_ids())
    deployments = read(
        "cluster_manager",
        lambda manager: _ids(manager.store.list_pooled_deployments(), "credential_id"),
    )
    drafts = read("application_deployment_store", lambda store: store.secret_references())
    connectors = read(
        "custom_agents",
        lambda store: {
            str(connector.get("credential_ref") or "")
            for agent in store.list(actor)
            for connector in agent.get("connectors") or []
            if connector.get("credential_ref")
        },
    )
    app_secrets = None if drafts is None or connectors is None else drafts | connectors
    return References(nodes=nodes, deployments=deployments, app_secrets=app_secrets)
