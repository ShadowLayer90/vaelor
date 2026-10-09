"""Routes for the outbound LLM Server (M1): expose the GPU model on the LAN.

The LLM Server turns Vaelor's own single-node GPU AI-Chat model into an
OpenAI-compatible API on the LAN, protected by a generated API key, so external
OpenAI clients can use it. The model itself stays loopback-only; the LAN gate is a
Vaelor-controlled nginx auth proxy (:mod:`vaelor.llm_server_proxy`) that enforces
the key, so the exposure is trustworthy and engine-agnostic. These routes surface
the exposure (the proxy's base URL, the model, the key) and drive its lifecycle
(enable, disable, rotate the key).

**The control plane owns the persisted state; the executor applies it.** A toggle
writes the desired ``{enabled, api_key}`` record here and enqueues one
``llm_server.apply`` job. Only the workload executor can drive the root hardware
bridge that starts/stops the auth proxy, so it is the account that brings the LAN
gate up or down to match the state. That split is why the key is generated here (so
the surface can show it immediately) and read there (so the proxy config matches).

Every verb is administrator-only: the response carries the plaintext API key, and
exposing a model to the network is a privileged decision.

**The surface reports what the LAN door is DOING, not only what it was told.**
``enabled`` is the flag; ``runtime`` is the reading (:func:`llm_server_runtime`):
the proxy container's own status over the root bridge - the verb the executor's
reconcile reads, through the same controller - and a loopback connect to the
proxy port. The console showed "Serving" from the flag alone for eight days
while port 11434 refused every connection (2026-09-28, LESSONS pattern 1).
"""

from __future__ import annotations

import http.client
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from flask import g, request

from .api_common import ApiContext, payload as _payload
from .gpu_cluster_mode import (
    DEPLOYING_STATE, UNLOADED_STATE, ClusterModeStore,
)
from .gpu_serving_target import (
    KIND_CLUSTER, UNLOAD_CAUSE_IDLE, UNLOAD_CAUSE_MANUAL, UNLOAD_CAUSE_UNKNOWN,
    gpu_cluster_mode_active, resolve_gpu_serving_target, unload_cause,
)
from .credential_broker import CredentialError
from .llm_gate_usage import (
    USAGE_NOT_READING, USAGE_UNREADABLE, cached_proxy_status, gate_logging_state,
    key_usage_for_rows,
)
from .llm_server_proxy import (
    LLM_SERVER_PROXY_PORT, UPSTREAM_MODEL, UPSTREAM_UNLOADED_NOTICE,
    UPSTREAM_WAKE_DOOR, LlmServerProxyController,
)
from .llm_server_state import (
    LLM_SERVER_ENDPOINT_ID,
    LlmServerStore,
    binding_marker,
    disable as disable_llm_server,
    enable as enable_llm_server,
    enqueue_apply,
    external_base_url,
    llm_server_rows,
    migrate_legacy_key,
    row_fingerprints,
)


def _resolve_gpu_chat(broker):
    """Facts about the GPU serving target the LLM Server would expose.

    A thin adapter over :func:`vaelor.gpu_serving_target.resolve_gpu_serving_target`,
    which is the ONE gate the executor's failure-watch and its ``apply_llm_server``
    read too (VD-125). This module used to spell the VD-085 independence test for
    itself, in its own vocabulary, and could not see GPU clustering at all - so a
    Mode B appliance reported "no GPU model to expose" while vLLM was serving on
    it. Returns the gate's :class:`~vaelor.gpu_serving_target.ServingTarget`
    whole - the surface publishes its ``kind`` as ``target_kind``, and a tuple
    that dropped it left the panel's documented primary mode source unpublished;
    the reason tokens are the gate's own, and the panel's ``UNAVAILABLE_COPY``
    table keys on exactly them.

    **The mode file is read here, in the control plane.** It is written by the
    executor into its own directory and made group-readable to the shared jobs
    group, which this account is a member of; an unreadable record fails safe to
    Mode A, so the surface degrades to the pre-cluster answer rather than
    inventing one.
    """
    return resolve_gpu_serving_target(broker, ClusterModeStore().read())


#: Every word ``runtime.state`` may carry - the wire vocabulary the console's
#: badge switches on (`frontend/src/lib/llmServerStatus.ts` holds the one
#: copy, marked, and `tests/test_wire_vocabularies.py` pins the two together).
RUNTIME_STATES = frozenset({
    "serving", "not-running", "model-not-answering", "applying-keys",
    "no-keys", "paused", "starting", "unknown", "off", "still-open",
})

#: SERVING is the only state the console may call "Serving".
RUNTIME_SERVING = "serving"
#: Enabled and something to front, but the gate itself is not answering.
RUNTIME_NOT_RUNNING = "not-running"
#: Enabled, the gate answers, but the model behind it does not: nginx takes the
#: connection and answers every request with a 502 (or times out).
RUNTIME_MODEL_DOWN = "model-not-answering"
#: Enabled, the gate and the model answer, but the gate runs a key set that is
#: not the broker's current one: a key minted since is refused and one revoked
#: since is still admitted, and ``/health`` - unauthenticated - cannot show it.
RUNTIME_APPLYING_KEYS = "applying-keys"
#: Enabled with an EMPTY key set: no gate runs by design (the no-keyless-door
#: invariant, `llm_server_state.serve_binding`), so the fix is a key, not a wait.
RUNTIME_NO_KEYS = "no-keys"
#: Enabled in Mode B while the cluster model is unloaded (G3a/G3b). The door
#: is shut on purpose - or, after an IDLE unload and while the control plane's
#: wake responder answers, kept open in front of it (ACC-058) - and fronts the
#: model again once it is loaded.
RUNTIME_PAUSED = "paused"
#: Enabled in Mode B while a deploy or a load is bringing the model up.
RUNTIME_STARTING = "starting"
#: Enabled, but the bridge could not be asked: neither up nor down is known.
RUNTIME_UNKNOWN = "unknown"
#: Disabled, and the gate is not answering: the settled off state.
RUNTIME_OFF = "off"
#: Disabled, yet the gate still answers - a disable not yet applied to it.
RUNTIME_STILL_OPEN = "still-open"

#: What one ``GET /health`` through the gate found (:func:`_gate_health`).
GATE_DOWN = "gate-down"
GATE_MODEL_SILENT = "model-silent"
GATE_SERVING = "gate-serving"

#: How long the health request through the gate may take. The gate is on this
#: machine and a serving model answers ``/health`` at once; two seconds bounds
#: a GET the console polls, and a model slower than that is not serving.
GATE_PROBE_SECONDS = 2.0

#: The sentence for each state that has exactly one cause. Each says what IS
#: and what the owner can do - never a recovery Vaelor may not make: which
#: loop, if any, re-applies the gate depends on the mode and on the model's
#: mechanism, and the control plane cannot see all of that from here.
RUNTIME_DETAILS = {
    RUNTIME_STARTING: (
        "The cluster model is loading. The LLM Server opens its port once the "
        "model is serving."
    ),
    RUNTIME_UNKNOWN: (
        "Vaelor could not ask the hardware bridge whether the LLM Server proxy "
        "is running, so it is not known to be serving."
    ),
    RUNTIME_STILL_OPEN: (
        "Disabled, but the LLM Server port still answers: the disable has not "
        "reached the proxy yet. This card checks it again every few seconds; "
        "if the port is still open after two minutes, enable and disable the "
        "endpoint again."
    ),
    RUNTIME_NO_KEYS: (
        "Enabled, but the LLM Server has no API keys, so Vaelor keeps its port "
        "closed rather than open it without one. Mint a key to open it."
    ),
    RUNTIME_MODEL_DOWN: (
        "The LLM Server proxy is up, but the model behind it did not answer "
        "its health check, so requests through it fail. Check the model that "
        "serves AI Chat."
    ),
    RUNTIME_APPLYING_KEYS: (
        "The LLM Server is answering, but its proxy still carries an older set "
        "of API keys: a key minted or rotated since may be refused, and one "
        "revoked since may still be accepted, until the proxy is re-keyed. If "
        "this does not clear, disable and enable the endpoint."
    ),
}

#: Why an enabled gate reads not-running, one sentence per cause.
NOT_RUNNING_DETAILS = {
    "no-target": (
        "Enabled, but there is no GPU model serving for it to expose ({}), so "
        "its port is closed."
    ),
    "proxy-stopped": (
        "Enabled, but the LLM Server proxy is not running, so clients on your "
        "LAN get connection refused. Disable and enable the endpoint to start "
        "it; the apply job's result says why if it cannot start."
    ),
    "port-closed": (
        "Enabled, and the LLM Server proxy container is up, but port {} does "
        "not accept connections. Disable and enable the endpoint to restart it."
    ),
}

#: ``runtime.reason`` for a paused door whose unload cause could not be read
#: (``gpu_serving_target.unload_cause`` answered ``UNLOAD_CAUSE_UNKNOWN``). The
#: state stays ``paused`` - the door is shut and the record says unloaded - so
#: the badge vocabulary is unchanged; only the sentence admits what is unknown.
PAUSED_CAUSE_UNKNOWN = "unload-cause-unknown"

#: The paused sentence WHILE THE GATE IS NOT RUNNING, by who unloaded the
#: model - because that decides who can wake it. An idle unload (G3b) holds AI
#: Chat on the cluster, so an AI Chat message loads it again; a manual unload
#: (G3a) parks AI Chat on the Assistant's model, so AI Chat cannot. The
#: inference gateway wakes either (its lease stays on the cluster), and
#: NOTHING wakes it through the LLM Server's own port, which is closed. The
#: cause is ``gpu_serving_target.unload_cause``'s, the one rule the fleet
#: summary reads. Since VD-159 the gate normally IS running while unloaded, and
#: the sentence is then :data:`PAUSED_OPEN_DOOR_DETAILS` or
#: :data:`PAUSED_WAKE_DOOR_DETAIL`; these remain for a gate that could not be
#: brought up, so "closed" is only ever said of a port that is closed.
PAUSED_DETAILS = {
    UNLOAD_CAUSE_IDLE: (
        "The cluster model was unloaded after sitting idle, so the LLM Server "
        "is paused and its port is closed. A client of the LLM Server cannot "
        "wake it; an AI Chat message or a request to the inference gateway "
        "loads it again, and the port reopens once it is serving."
    ),
    UNLOAD_CAUSE_MANUAL: (
        "The cluster model was unloaded by hand, so the LLM Server is paused "
        "and its port is closed. Neither a client of the LLM Server nor AI "
        "Chat can wake it: load it from Cluster > Deployments, or send a "
        "request to the inference gateway, and the port reopens once it is "
        "serving."
    ),
    PAUSED_CAUSE_UNKNOWN: (
        "The cluster model is unloaded, so the LLM Server is paused and its "
        "port is closed. Vaelor could not read whether it was unloaded after "
        "sitting idle or by hand, so it cannot say whether AI Chat would wake "
        "it. Loading it from Cluster > Deployments, or a request to the "
        "inference gateway, brings it back, and the port reopens once it is "
        "serving."
    ),
}

#: The paused sentence while the UNLOADED DOOR is actually up (VD-159): the
#: gate runs, fronts nothing, and answers a keyed request "not loaded" itself.
#: Read off the bridge's status like the wake door's, never assumed. Who can
#: wake the model is still the cause's answer.
PAUSED_OPEN_DOOR_DETAILS = {
    UNLOAD_CAUSE_IDLE: (
        "The cluster model was unloaded after sitting idle, so the LLM Server "
        "is paused. Its port stays open and tells a client with an LLM Server "
        "key that the model is not loaded, but that client cannot wake it "
        "right now; an AI Chat message or a request to the inference gateway "
        "loads it again, and the server answers normally once it is serving."
    ),
    UNLOAD_CAUSE_MANUAL: (
        "The cluster model was unloaded by hand, so the LLM Server is paused. "
        "Its port stays open, says the model is unloaded and answers every "
        "request with an error: a client with an LLM Server key is told the "
        "model is not loaded, and neither that client nor AI Chat can wake "
        "it. Load it from Cluster > Deployments, or send a request to the "
        "inference gateway, and the server answers normally once it is serving."
    ),
    PAUSED_CAUSE_UNKNOWN: (
        "The cluster model is unloaded, so the LLM Server is paused. Its port "
        "stays open and tells a client with an LLM Server key that the model "
        "is not loaded. Vaelor could not read whether the model was unloaded "
        "after sitting idle or by hand, so it cannot say whether AI Chat "
        "would wake it; loading it from Cluster > Deployments, or a request "
        "to the inference gateway, brings it back."
    ),
}

#: The paused sentence while the gate still stands in front of the STOPPED
#: MODEL: the moments before Vaelor moves it (an unload does that in its own
#: step, so this is rare), or a gate it could not move. The port is open and
#: failing, which is neither "closed" nor "tells a client it is not loaded".
PAUSED_STALE_DOOR_DETAIL = (
    "The cluster model is unloaded, so the LLM Server is paused. Its port is "
    "still open in front of the stopped model, so requests through it fail "
    "until Vaelor moves it, which it retries every half minute. Loading the "
    "model from Cluster > Deployments brings the server back."
)

#: The idle sentence while the wake door is ACTUALLY up (ACC-058): the gate is
#: running in front of the control plane's wake responder, not a model port.
#: Read off the bridge's status, never assumed from the unload cause - the
#: switch opens that door only while the responder answers (review S2/S5).
PAUSED_WAKE_DOOR_DETAIL = (
    "The cluster model was unloaded after sitting idle, so the LLM Server is "
    "paused, but its port stays open: a request with an LLM Server key loads "
    "the model again and is told to retry in a moment, and the server answers "
    "normally once the model is serving. An AI Chat message or a request to "
    "the inference gateway loads it too."
)
#: Loading after a wake: the door still fronts the responder until the model serves.
STARTING_WAKE_DOOR_DETAIL = (
    "The cluster model is loading. Until it is serving, the LLM Server answers "
    "a request with an LLM Server key by asking the client to retry in a moment."
)


#: Loading with the unloaded door still up (a Load after a manual unload,
#: VD-159): the port is open and says "not loaded" until the model serves, so
#: "opens its port once serving" would be false of it.
STARTING_OPEN_DOOR_DETAIL = (
    "The cluster model is loading. Until it is serving, the LLM Server's port "
    "stays open and tells a client with an LLM Server key that the model is "
    "not loaded yet."
)

#: Mode A with the loading door up (W4-D8): leaving cluster serving moved the
#: gate onto it while the GPU watch relaunches AI Chat's model here.
STARTING_MODE_A_DETAIL = (
    "AI Chat's model on this machine is starting. Until it is serving, the LLM "
    "Server's port stays open and tells a client with an LLM Server key that "
    "the model is loading."
)

#: The starting sentence by the door that is actually running; any other
#: reading - no gate, which is a first deploy's state - keeps the default.
STARTING_DOOR_DETAILS = {
    UPSTREAM_WAKE_DOOR: STARTING_WAKE_DOOR_DETAIL,
    UPSTREAM_UNLOADED_NOTICE: STARTING_OPEN_DOOR_DETAIL,
}


def _running_door(bridge: Any) -> str:
    """What the RUNNING gate fronts, as its status names it, or ``""``.

    ``""`` for a gate that is not running, a bridge that is not there, one
    that did not answer, and a status too old to say - none of which is a
    door a sentence may describe as open.
    """
    if bridge is None or not getattr(bridge, "available", False):
        return ""
    try:
        status = LlmServerProxyController(bridge).status() or {}
    except Exception:  # noqa: BLE001 - an unanswered bridge is "not up"
        return ""
    # A status older than the field fronts the model: that was all there was.
    return str(status.get("upstream") or UPSTREAM_MODEL) if status.get("running") else ""


def _gate_health(port: int) -> str:
    """``GET /health`` THROUGH the gate on loopback: which half is answering.

    The gate proxies an unauthenticated ``/health`` to the model
    (`llm_server_proxy.render_proxy_config`), so a 2xx means nginx AND the
    model behind it answered. A bare TCP connect cannot tell that from a dead
    upstream: nginx accepts the connection either way and answers 502. A
    connection that cannot be made is the gate down; a failure after it was
    made, or a non-2xx answer, is the gate up with the model not serving.
    """
    connection = http.client.HTTPConnection(
        "127.0.0.1", int(port), timeout=GATE_PROBE_SECONDS
    )
    try:
        try:
            connection.connect()
        except OSError:
            return GATE_DOWN
        try:
            connection.request("GET", "/health")
            status = connection.getresponse().status
        except (OSError, http.client.HTTPException):
            return GATE_MODEL_SILENT
    finally:
        connection.close()
    return GATE_SERVING if 200 <= int(status) < 300 else GATE_MODEL_SILENT


def _proxy_reading(bridge: Any, gate_health: Callable[[int], str]):
    """``(container_running, gate_health, live_key_set)``, or ``None`` if unreadable.

    The container reading is `LlmServerProxyController.status` over the root
    bridge - the same verb and controller the executor's reconcile reads, not
    a new privileged path - attempted only when the bridge socket exists, so a
    down bridge never hangs the GET. ``live_key_set`` is the SET-hash the
    bridge rendered into the running gate, or ``None`` from a bridge that does
    not report one. The health request goes through the gate only when the
    container runs; a stopped one is :data:`GATE_DOWN` unasked.
    """
    if bridge is None or not getattr(bridge, "available", False):
        return None
    try:
        status = LlmServerProxyController(bridge).status() or {}
    except Exception:  # noqa: BLE001 - an unanswered bridge is "not known", reported
        return None
    running = bool(status.get("running"))
    live = str(status.get("key_set") or "") if "key_set" in status else None
    health = gate_health(LLM_SERVER_PROXY_PORT) if running else GATE_DOWN
    return running, health, live


def llm_server_runtime(
    *,
    enabled: bool,
    target: Any,
    mode_state: Any,
    record_state: str,
    bridge: Any,
    fingerprints: Optional[Sequence[str]] = None,
    gate_health: Optional[Callable[[int], str]] = None,
) -> Dict[str, str]:
    """What the LLM Server's LAN door is actually doing: ``{state, reason, detail}``.

    ``state`` is one of :data:`RUNTIME_STATES`; ``detail`` is the sentence the
    console shows when it is not :data:`RUNTIME_SERVING`. Serving means every
    link answered AND carries the current keys: the container runs (bridge
    status), a ``/health`` request through it reached the model
    (:func:`_gate_health`), and the key set the bridge rendered into it is the
    broker's current one - ``/health`` is unauthenticated, so without that
    last comparison a gate still admitting a revoked key would read
    "Serving". In Mode B a paused (``unloaded``) or starting (``deploying``)
    deployment is named as such before any live read, because a shut door is
    then the intended state, not a fault; so is an enabled server with no
    keys, which runs no gate by design. ``fingerprints`` is the broker's
    per-key fingerprint list, or ``None`` when the broker could not say - and
    is then read as neither "no keys" nor "different keys". ``gate_health``
    defaults to :func:`_gate_health`, resolved at call time.
    """
    gate_health = gate_health or _gate_health
    clustered = gpu_cluster_mode_active(mode_state)
    if enabled and clustered and record_state == UNLOADED_STATE:
        # Who unloaded it is the one rule's answer (idle keeps ai-chat on the
        # cluster; manual parked it elsewhere; an unreadable lease is unknown).
        cause = unload_cause(target)
        if cause == UNLOAD_CAUSE_UNKNOWN:
            cause = PAUSED_CAUSE_UNKNOWN
        # The sentence follows the door that is actually running (one status
        # read), never the plan: "closed" only of a port that is closed.
        door = _running_door(bridge)
        detail = PAUSED_DETAILS[cause]
        if cause == UNLOAD_CAUSE_IDLE and door == UPSTREAM_WAKE_DOOR:
            detail = PAUSED_WAKE_DOOR_DETAIL
        elif door == UPSTREAM_UNLOADED_NOTICE:
            detail = PAUSED_OPEN_DOOR_DETAILS[cause]
        elif door == UPSTREAM_MODEL:
            detail = PAUSED_STALE_DOOR_DETAIL
        return _runtime(RUNTIME_PAUSED, cause, detail=detail)
    if enabled and clustered and (
        record_state == DEPLOYING_STATE
        or bool(getattr(mode_state, "deploy_in_flight", False))
    ):
        return _runtime(
            RUNTIME_STARTING, "deployment-loading",
            detail=STARTING_DOOR_DETAILS.get(_running_door(bridge)),
        )
    if enabled and not clustered and target.available and (
        _running_door(bridge) == UPSTREAM_UNLOADED_NOTICE
    ):
        # W4-D8: the way back from cluster serving - the door answers
        # "loading" until the relaunched model is fronted again.
        return _runtime(RUNTIME_STARTING, "model-loading", detail=STARTING_MODE_A_DETAIL)
    reading = _proxy_reading(bridge, gate_health)
    if not enabled:
        if reading is not None and reading[1] != GATE_DOWN:
            return _runtime(RUNTIME_STILL_OPEN, "apply-pending")
        return _runtime(RUNTIME_OFF, "disabled", detail="")
    if not target.available:
        return _runtime(
            RUNTIME_NOT_RUNNING, "no-target",
            detail=NOT_RUNNING_DETAILS["no-target"].format(
                target.reason or "unavailable"
            ),
        )
    if fingerprints is not None and not fingerprints:
        return _runtime(RUNTIME_NO_KEYS, "no-keys")
    if reading is None:
        return _runtime(RUNTIME_UNKNOWN, "bridge-unavailable")
    running, health, live = reading
    if health == GATE_SERVING:
        if fingerprints is not None and live is not None and live != (
            binding_marker(True, fingerprints).key_fingerprint
        ):
            return _runtime(RUNTIME_APPLYING_KEYS, "key-set-drift")
        return _runtime(RUNTIME_SERVING, "", detail="")
    if health == GATE_MODEL_SILENT:
        return _runtime(RUNTIME_MODEL_DOWN, "model-not-answering")
    if running:
        return _runtime(
            RUNTIME_NOT_RUNNING, "port-closed",
            detail=NOT_RUNNING_DETAILS["port-closed"].format(LLM_SERVER_PROXY_PORT),
        )
    return _runtime(
        RUNTIME_NOT_RUNNING, "proxy-stopped",
        detail=NOT_RUNNING_DETAILS["proxy-stopped"],
    )


def _runtime(state: str, reason: str, *, detail: Optional[str] = None) -> Dict[str, str]:
    return {
        "state": state,
        "reason": reason,
        "detail": RUNTIME_DETAILS.get(state, "") if detail is None else detail,
    }


def _record_state(manager: Any, mode_state: Any) -> str:
    """The Mode B deployment's record state, or ``""`` when there is none to read.

    Read from the cluster store the control plane already holds (the one
    `_lan_host` reads the advertise address from). Unreadable is ``""``: the
    runtime then falls through to the live reading, which is still the truth
    about the door.
    """
    name = str(getattr(mode_state, "deployment_name", "") or "")
    if not name or not gpu_cluster_mode_active(mode_state):
        return ""
    try:
        record = manager.store.get_pooled_deployment(name)
    except Exception:  # noqa: BLE001 - absence-ok: the live reading still decides
        return ""
    return str((record or {}).get("state", "") or "")


def _fronted_model(manager: Any, mode_state: Any, target: Any) -> Tuple[str, bool]:
    """``(model, known)``: the model the LLM Server's door fronts, and whether that is known.

    An available target is what the door fronts, in either mode. With none:
    in Mode A there is nothing to front - AI Chat on the NPU Assistant or an
    external endpoint is not this door's - so no model, and that is known. In
    Mode B the door is the cluster's: a manual unload parks AI Chat's lease on
    the Assistant's connection, so the lease's model is the NPU Assistant's and
    the paused card named it (ACC-209). The cluster record's model is the one
    that comes back on Load; when the record cannot be read, which model the
    door serves is NOT KNOWN, which is not the same as none loaded (review 2).
    """
    if target.available:
        return str(target.model or ""), True
    if not gpu_cluster_mode_active(mode_state):
        return "", True
    name = str(getattr(mode_state, "deployment_name", "") or "")
    try:
        record = manager.store.get_pooled_deployment(name) if name else None
    except Exception:  # noqa: BLE001 - absence-ok: an unread record is said as not known
        record = None
    model = str((record or {}).get("model_id") or "")
    return model, bool(model)


#: B3: the card's words are the backend's, so a rename is one backend change and
#: the browser holds none. The line under "LLM Server" when no model id can
#: stand there: an unread record is NOT KNOWN, never "none loaded" (ACC-209).
MODEL_LINE_NOT_KNOWN = "Which model it serves is not known"
MODEL_LINE_NONE_LOADED = "No model is loaded"
MODEL_LINE_NO_MODEL = "OpenAI-compatible endpoint"
#: The lead with nothing to expose. Sent only when the runtime has no sentence
#: of its own: a paused cluster model is not "nothing to expose".
NOTHING_TO_EXPOSE_NOTE = (
    "There is no independent GPU AI-Chat model to expose yet. Deploy one, and "
    "this endpoint can be enabled and given keys."
)


#: What the revoke confirmation says when the key is the LLM Server's last
#: active one. No keys means no door (VD-159, `llm_server_state.serve_binding`),
#: and the owner revoked every key without being told the port would close
#: (2026-10-03). Sent on every read; the card shows it only for the last key.
LAST_KEY_REVOKE_NOTE = (
    "This is the LLM Server's last API key. Revoking it closes port {port}, so "
    "clients on your LAN get connection refused, and it stays closed until you "
    "mint a new key."
)
#: The same warning while the server is disabled (W6-5): its port is already
#: closed, so the revoke closes nothing; what it costs is the way back.
LAST_KEY_REVOKE_NOTE_DISABLED = (
    "This is the LLM Server's last API key. The server is disabled, so port "
    "{port} is already closed; with no key left, enabling it opens nothing until "
    "you mint a new key."
)


def llm_server_card_words(
    *, model: str, model_known: bool, runtime: Dict[str, str], available: bool,
) -> Dict[str, str]:
    """``{model_line, unavailable_note}``: what the LLM Server card says (B3)."""
    line = model or (MODEL_LINE_NOT_KNOWN if not model_known else MODEL_LINE_NONE_LOADED
                     if runtime.get("state") == RUNTIME_PAUSED else MODEL_LINE_NO_MODEL)
    note = NOTHING_TO_EXPOSE_NOTE if not available and not runtime.get("detail") else ""
    return {"model_line": line, "unavailable_note": note}


def active_key_rows(broker):
    """The ACTIVE llm-server key rows, or ``None`` when the broker cannot say.

    ``None`` and ``[]`` are different answers: the runtime reports "no API
    keys" only for a broker that answered with none, never for one that did
    not answer.
    """
    if broker is None:
        return None
    try:
        return llm_server_rows(broker)
    except CredentialError:
        return None


def llm_server_reading(*, store: Any, broker: Any, manager: Any, bridge: Any) -> Dict[str, Any]:
    """What the LLM Server is set to and doing - the one reading behind its card.

    The console card (``_surface``) and the Assistant's answer
    (`assistant_serving_answers`) both read this, so the two cannot disagree
    about whether the server is running (LESSONS 6). ``rows`` are the key rows,
    secret-adjacent metadata no caller may print; the rest is ``settings``,
    ``target``, ``model``, ``model_known`` and ``runtime``.
    """
    # First read after an upgrade imports a pre-F3b-ii key into the broker
    # (idempotent), so the card and the Assistant read the same key set.
    if broker is not None:
        migrate_legacy_key(store, broker)
    settings = store.read()
    mode_state = ClusterModeStore().read()
    target = resolve_gpu_serving_target(broker, mode_state)
    rows = active_key_rows(broker)
    model, model_known = _fronted_model(manager, mode_state, target)
    runtime = llm_server_runtime(
        enabled=settings.enabled, target=target, mode_state=mode_state,
        record_state=_record_state(manager, mode_state), bridge=bridge,
        fingerprints=None if rows is None else row_fingerprints(rows),
    )
    return {"settings": settings, "target": target, "rows": rows, "model": model,
            "model_known": model_known, "runtime": runtime}


def register_llm_server_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    security = context.security
    require_auth = context.require_auth

    def _store() -> LlmServerStore:
        # A pre-wired store (a test's, over a tmp file) wins; production reads
        # the persisted record from the state root.
        existing = callbacks.get("llm_server_store")
        return existing if existing is not None else LlmServerStore()

    def _keys_for_surface(rows) -> list:
        """The non-secret profile of each ACTIVE llm-server key - never plaintext.

        Built from the broker's fingerprint-only listing (F3b-ii, item 7): label,
        fingerprint, last4 and id per key. The served-endpoint / this-endpoint /
        not-revoked filter is :func:`~vaelor.llm_server_state.llm_server_rows`, the
        ONE definition the executor's convergence poll reads too, so the surface and
        the gate can never disagree about which keys count. The plaintext appears
        ONLY in a mint/rotate response, so a GET can be read by anyone without
        leaking a key. A broker that cannot answer degrades to an empty list.

        ``last_used_at`` is the vault's one column for it, which the gate's usage
        log now writes (ACC-043); ``requests`` is how many requests the gate
        admitted with the key, from the same log, or None when that log has not
        been read (see :func:`_key_usage_state`).
        """
        counts = key_usage_for_rows(callbacks.get("llm_gate_usage"), rows or [])
        return [
            {
                "credential_id": row.get("id"),
                "label": row.get("label"),
                "key_fingerprint": row.get("fingerprint"),
                "last4": row.get("last4"),
                "created_at": row.get("created_at"),
                "last_used_at": row.get("last_used_at"),
                "requests": (counts.get(str(row.get("id") or "")) or {}).get("requests"),
            }
            for row in rows or []
        ]

    def _key_usage_state(enabled: bool) -> dict:
        """Whether the key table's use figures are current, and what they missed.

        An enabled server whose running gate is not recording use says so first
        (the bridge's own reason, `gate_logging_state`): a key must not read
        "Never used" off a gate that was not writing it down. Then the log
        reader's own state. ``last_gap_at`` is when requests were last lost to
        a rotation the reader missed; ``refused_24h`` how many requests the gate
        refused for a wrong or missing key in the last day.
        """
        store = callbacks.get("llm_gate_usage")
        if store is None:
            return {
                "state": USAGE_NOT_READING,
                "detail": "Key use is not collected by this control plane.",
                "last_gap_at": None, "refused_24h": None,
            }
        try:
            state = store.reader_state()
            refused = store.window_totals(86400)["refused"]
        except Exception:  # noqa: BLE001 - an unreadable store is "not known", not "none"
            return {
                "state": USAGE_UNREADABLE,
                "detail": "Vaelor could not read its record of LLM Server key use.",
                "last_gap_at": None, "refused_24h": None,
            }
        bridge = callbacks.get("hardware_bridge_client")
        status = cached_proxy_status(bridge) if enabled else None
        logging_state = gate_logging_state(enabled, status)
        if logging_state is not None:
            state = {**state, **logging_state}
        return {
            "state": state["state"], "detail": state["detail"],
            "last_gap_at": state.get("last_gap_at"), "refused_24h": refused,
        }

    def _lan_host(target) -> str:
        """The authoritative LAN host for the advertised base URL.

        When the serving target is the cluster (Mode B), the real LAN
        endpoint is the cluster controller's ``advertise_address`` - the
        address the cluster was formed with, which can never be a stray
        bridge or loopback. For a single-node target (Mode A), or when the
        advertise address is empty or unreadable, fall back to the
        NIC-ranking appliance address.
        """
        if target.kind == KIND_CLUSTER:
            manager = callbacks.get("cluster_manager")
            try:
                advertise = str(
                    manager.store.controller().get("advertise_address", "")
                    or ""
                ).strip()
            except (AttributeError, OSError, TypeError, ValueError):
                advertise = ""
            if advertise:
                return advertise
        return context.appliance_address()

    def _surface() -> dict:
        """The full LLM Server surface: persisted state + resolved GPU facts."""
        store = _store()
        broker = callbacks.get("credential_broker")
        reading = llm_server_reading(
            store=store, broker=broker, manager=callbacks.get("cluster_manager"),
            bridge=callbacks.get("hardware_bridge_client"),
        )
        settings, target, rows = reading["settings"], reading["target"], reading["rows"]
        model, model_known, runtime = reading["model"], reading["model_known"], reading["runtime"]
        # The advertised endpoint is the auth PROXY's LAN port, not the model's
        # loopback port: external clients reach the model only THROUGH the keyed
        # proxy. The target's own port gates availability (there must be a
        # loopback model to front) but is never itself exposed.
        proxy_port = LLM_SERVER_PROXY_PORT if target.available else 0
        base_url = (
            external_base_url(
                _lan_host(target), proxy_port, enabled=settings.enabled
            )
            if target.available else ""
        )
        return {
            "enabled": settings.enabled,
            "state": "enabled" if settings.enabled else "disabled",
            # What the LAN door is DOING: the console's badge reads this, and
            # says "Serving" only when it is RUNTIME_SERVING.
            "runtime": runtime,
            "available": target.available,
            "unavailable_reason": target.reason,
            # The gate's kind, verbatim (``none`` when there is no target): the
            # panel's primary source for which serving mode it is looking at.
            "target_kind": target.kind,
            "model": model,
            # False when which model the door serves could not be read (ACC-209).
            "model_known": model_known,
            **llm_server_card_words(model=model, model_known=model_known,  # B3
                                    runtime=runtime, available=target.available),
            # VD-159: the revoke confirmation's warning for the last key.
            "last_key_note": (LAST_KEY_REVOKE_NOTE if settings.enabled
                              else LAST_KEY_REVOKE_NOTE_DISABLED).format(port=LLM_SERVER_PROXY_PORT),
            "port": proxy_port,
            "base_url": base_url,
            # The keys are NON-SECRET profiles (fingerprint + last4 + label); the
            # plaintext appears ONLY in a mint/rotate response (F3b-ii, item 7).
            "keys": _keys_for_surface(rows),
            # Whether their use figures are current (ACC-043, LESSONS 8).
            "usage": _key_usage_state(settings.enabled),
        }

    def _enqueue_apply(action: str):
        """Enqueue one ``llm_server.apply`` job so the executor relaunches the fork.

        Non-fatal if the job service is down: the state is already persisted, so
        the 30 s failure-watch reconcile (which reads the same state) still brings
        the server to the persisted binding. The response says whether a job was
        queued so the frontend can distinguish "applying now" from "will apply".
        """
        return enqueue_apply(
            callbacks.get("job_store"), g.auth_session.username, action
        )

    def _toggle_payload(job_id, minted_key=""):
        """A toggle response that does not LIE about whether it took effect.

        FINDING A (surface): the persisted state is set here, but the running fork
        is only reconverged by the executor - the enqueued ``llm_server.apply``
        job, or, if the job service was down (``_enqueue_apply`` returned
        ``None``), the 30 s failure-watch reconcile. So ``apply`` reports
        ``queued`` or ``pending`` rather than letting the bare ``state`` imply the
        LAN exposure was already revoked/armed. The reconcile is the guarantee;
        this never blocks on the job, it just tells the truth about timing.
        """
        return _payload({
            **_surface(),
            "job_id": job_id,
            "apply": "queued" if job_id else "pending",
            # The plaintext of a NEWLY minted or rotated key, revealed exactly
            # once; "" when nothing new was created (an idempotent enable).
            "minted_key": minted_key,
        })

    @blueprint.get("/llm-server")
    @require_auth("administrator")
    def llm_server_status():
        return _payload(_surface())

    @blueprint.post("/llm-server/enable")
    @require_auth("administrator", csrf=True)
    def llm_server_enable():
        # FINDING D: refuse to arm the LAN bind on a box with no independent GPU
        # AI-Chat tier to expose. Without this the surface would show enabled+key
        # while the apply job does "nothing to do" and nothing is ever bound. The
        # gate is the SAME availability the GET surface reports, and it is
        # fail-safe: nothing is persisted and no key is generated, so no exposure.
        target = _resolve_gpu_chat(callbacks.get("credential_broker"))
        if not target.available:
            security.audit(
                g.auth_session.username, "llm_server.enable", "refused",
                target="gpu-ai-chat", remote_addr=request.remote_addr or "",
            )
            return _payload(
                error={
                    "code": "llm_server_unavailable",
                    "message": (
                        "There is no independent GPU AI-Chat model to expose on "
                        "this appliance ({}). Deploy a GPU AI-Chat model first."
                    ).format(target.reason or "unavailable"),
                },
                status=409,
            )
        try:
            outcome = enable_llm_server(_store(), callbacks.get("credential_broker"))
        except CredentialError as error:
            security.audit(
                g.auth_session.username, "llm_server.enable", "refused",
                target="gpu-ai-chat", remote_addr=request.remote_addr or "",
            )
            return _payload(
                error={
                    "code": "llm_server_broker_unavailable",
                    "message": (
                        "The LLM Server key store could not be reached, so the "
                        "server was not enabled ({}).".format(str(error))
                    ),
                },
                status=503,
            )
        job_id = _enqueue_apply("enable")
        security.audit(
            g.auth_session.username, "llm_server.enable", "success",
            target="gpu-ai-chat", remote_addr=request.remote_addr or "",
            # W5-D5: the job this change queued, so its evidence finds this row.
            details={"job_id": job_id or ""},
        )
        return _toggle_payload(job_id, minted_key=outcome.minted_key)

    @blueprint.post("/llm-server/disable")
    @require_auth("administrator", csrf=True)
    def llm_server_disable():
        # Disable never revokes the keys - they stay in the broker so re-enabling
        # restores the same set an external client already holds (items 2 and 6).
        disable_llm_server(_store(), callbacks.get("credential_broker"))
        job_id = _enqueue_apply("disable")
        security.audit(
            g.auth_session.username, "llm_server.disable", "success",
            target="gpu-ai-chat", remote_addr=request.remote_addr or "",
            # W5-D5: the job this change queued, so its evidence finds this row.
            details={"job_id": job_id or ""},
        )
        return _toggle_payload(job_id)

    @blueprint.post("/llm-server/rotate-key")
    @require_auth("administrator", csrf=True)
    def llm_server_rotate_key():
        # Per-key rotate now (F3b-ii): the request NAMES which key, and the broker
        # rotates that credential in place, revealing the new plaintext ONCE. The
        # generic .../endpoints/llm-server/keys routes (F3b-i) also mint and revoke.
        broker = callbacks.get("credential_broker")
        body = request.get_json(silent=True) or {}
        credential_id = str(body.get("credential_id") or "").strip()
        if not credential_id:
            return _payload(
                error={
                    "code": "llm_server_key_required",
                    "message": "Name the key to rotate (credential_id).",
                },
                status=400,
            )
        try:
            rotated = broker.rotate(credential_id, LLM_SERVER_ENDPOINT_ID)
        except CredentialError as error:
            return _payload(
                error={"code": "llm_server_rotate_failed", "message": str(error)},
                status=409,
            )
        job_id = _enqueue_apply("rotate")
        security.audit(
            g.auth_session.username, "llm_server.rotate_key", "success",
            target="gpu-ai-chat", remote_addr=request.remote_addr or "",
            # W5-D5: the job this change queued, so its evidence finds this row.
            details={"job_id": job_id or ""},
        )
        return _toggle_payload(job_id, minted_key=str(rotated.get("key", "")))
