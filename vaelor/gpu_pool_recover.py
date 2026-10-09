"""Load a GPU deployment the mode watch failed: the recovery Unload never offered (W4-D7).

Live, 2026-10-01: a split whose Ray worker hit its start limit was torn down by
the watch and written ``failed``. Unload refused it (nothing serving to
reclaim) and Load refused it (nothing paused to load), each naming the other,
so only Remove and a fresh deploy got out.

**Load is the recovery, and Unload's refusal now says so (VD-170).** A failed
row keeps everything a Load re-serves from - its machines, link, addresses,
model, options and its credential (the watch's teardown restores Mode A and
leaves the credential to Remove) - so it is Loaded the way an unloaded row is,
with the two differences the teardown made:

* **what the record derives is stopped first.** The watch already stopped it
  when this controller held the switch; a split it did not own was only marked,
  and its parts may still run. The stop is the one every remove and unload
  makes, idempotent for a unit that is gone; a unit on a machine that keeps
  the deployment that will not stop refuses the Load with the row as it was.
* **the mode switch is taken as a deploy takes it** when this controller leads
  and the switch is not already this deployment's: `enter` stops llama.cpp and
  parks AI Chat, the unloaded-row Load re-serves, and `repoint` moves AI Chat
  back onto the unchanged credential. A switch another deployment holds is
  refused by `enter` in its own words, before anything is stopped.

A row that never served (a deploy that failed before minting its credential)
has nothing to Load and is refused with Remove. Everything the Load checks
before it writes is asked first (`load_participants`, `load_options`,
`confirm_load_bindings`). A Load that cannot start it again stops what it
started (the Load's own rollback), gives Mode A back if it took it, and leaves
the row ``failed`` with the reason - Load stays offered for the retry.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .cluster_placement import CONTROLLER_PLACEMENT_ID, departed_node_ids
from .gpu_cluster_mode import UNLOADED_STATE
from .credential_broker_client import (
    VERB_FAILED, broker_did_not_answer, credential_not_found,
)
from .gpu_serving_target import CLUSTER_INFERENCE_PURPOSE
from .gpu_pool_units import (
    RAY_PLANE_KIND, VLLM_ENGINE, describe_unstopped, remove_blockers,
)

#: The row state the watch and a failed deploy write - the one spelling; the
#: mode watch (`gpu_cluster_mode_watch`) and the reload import it from here.
FAILED_STATE = "failed"

#: Set on a failed row by the mode watch when the broker said the cluster's
#: connection was deleted (`gpu_cluster_mode.CLUSTER_CREDENTIAL_MISSING`): a
#: Load would refuse it with :data:`CONNECTION_GONE`, so :func:`load_offered`
#: does not offer one (review A1: the row and its note give one answer).
CONNECTION_DELETED_FIELD = "connection_deleted"

#: What the watch's own bookkeeping put on a failed row; a recovered row
#: starts from the record without it.
_FAILURE_FIELDS = ("failure", "unstopped", "health", "plane_cleanup")

#: The way out of a failed row Load can start again - one sentence, said by
#: Unload's refusal (`gpu_pool_reload.FAILED_TO_UNLOAD`) and by the mode
#: watch's teardown note (`gpu_cluster_mode_watch`) alike.
LOAD_IT_AGAIN = "Load it to start it again from its record, or remove it."

#: Refused when the row never served: no credential was minted, so there is
#: no record of a serving deployment to start again.
NEVER_SERVED = (
    "This GPU deployment never finished deploying, so there is no serving "
    "record to load again; remove it and deploy it again."
)

#: Refused when the cluster's connection was deleted since: AI Chat and the
#: LLM Server would have nothing to point at, so it is redeployed instead.
CONNECTION_GONE = (
    "This GPU deployment's connection was deleted from the credential broker, "
    "so it cannot be loaded again; remove it and deploy it again."
)

#: Refused, retryably, when the broker could not be asked: nothing was touched.
BROKER_DID_NOT_ANSWER = (
    "Vaelor could not ask the credential broker for this deployment's "
    "connection ({}), so nothing was stopped. Load it again once the broker "
    "answers."
)

#: Refused when the broker ANSWERED with a refusal a retry cannot change - a
#: provider unusable for this purpose, a secret that would not decrypt, an
#: explicit rejection (review A6; a verb that failed inside the broker is a
#: retry since round 1, `broker_did_not_answer`): nothing was touched, and the
#: answer and where to read why are said instead of "try again".
BROKER_REFUSED = (
    "The credential broker refused this deployment's connection ({}), so "
    "nothing was stopped. Loading it again gets the same answer; the "
    "vaelor-credential-broker journal says why. Remove it and deploy it again "
    "if the connection cannot be repaired."
)

#: Said after :data:`BROKER_DID_NOT_ANSWER` when a verb failed INSIDE the
#: broker (``VERB_FAILED``): the broker did answer, and logged why (round 2, F5).
BROKER_JOURNAL = " The vaelor-credential-broker journal says why it failed."

#: Refused when a leftover unit would not stop: the row is left as it was.
LEFTOVERS_NOT_STOPPED = (
    "Could not stop what is left of this deployment ({}), so it was not "
    "started again; stop it on that machine, then Load it again."
)

#: The row's note when the Load could not start it again.
RESTART_FAILED = (
    "Vaelor could not start this deployment again: {} Load it to retry once "
    "that is fixed, or remove it."
)


def load_offered(deployment: Dict[str, Any]) -> bool:
    """Whether the row is one a Load can serve: paused, or failed after serving.

    The one answer to "can Load start this again?": the console offers Load
    from it and the mode watch's teardown note points at Load or at Remove by
    it (`gpu_cluster_mode_watch._mark_left`).
    """
    state = str(deployment.get("state", "") or "")
    units = deployment.get("units") or {}
    if units.get("engine") != VLLM_ENGINE:
        return False
    if state == UNLOADED_STATE:
        return True
    return (
        state == FAILED_STATE
        and bool(str(deployment.get("credential_id", "") or ""))
        and not units.get(CONNECTION_DELETED_FIELD)
    )


def load_failed_deployment(
    ops: Any, deployment: Dict[str, Any],
    progress: Optional[Callable[[int, str], None]] = None,
    placement_status: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Re-serve a failed vLLM row from its record (see the module docstring)."""
    from .gpu_pool_reload import (
        LOAD_CONFIRM, _switch_owns, confirm_load_bindings, load_deployment,
        load_options, load_participants,
    )

    name = str(deployment.get("name", ""))
    credential_id = str(deployment.get("credential_id", "") or "")
    if not credential_id:
        raise ValueError(NEVER_SERVED)
    try:
        ops.broker.resolve(credential_id, CLUSTER_INFERENCE_PURPOSE)
    except Exception as error:  # noqa: BLE001 - every failure is said, as what it is
        # Review S1 (LESSONS 8): only the broker saying it holds no such
        # credential is "deleted"; a broker that did not answer produced the
        # absence itself, and its remedy is a retry, never a redeploy.
        if credential_not_found(error):
            raise ValueError(CONNECTION_GONE) from error
        if broker_did_not_answer(error):
            said = BROKER_DID_NOT_ANSWER.format(error)
            if str(error) == VERB_FAILED:
                said += BROKER_JOURNAL
            raise RuntimeError(said) from error
        raise ValueError(BROKER_REFUSED.format(error)) from error
    report = progress or (lambda _percent, _message: None)
    report(5, "Checking {} can be started again before anything is stopped".format(name))
    participants = load_participants(ops, deployment, placement_status)
    load_options(ops, deployment, participants)
    transports = {node["id"]: ops._transport(node) for node in participants}
    confirm_load_bindings(ops, deployment, participants, transports)

    switch = getattr(ops, "mode_switch", None)
    takes_switch = (
        switch is not None
        and participants[0]["id"] == CONTROLLER_PLACEMENT_ID
        and not _switch_owns(ops, name)
    )
    if takes_switch:
        # Refused in the switch's own words while another deployment holds
        # it - before a single unit of this one is touched.
        switch.enter(name, report=lambda message: report(7, message))
    try:
        report(8, "Stopping whatever is left of {}".format(name))
        left = [entry for entry in ops.stop_units_for(deployment)
                if entry.get("kind") != RAY_PLANE_KIND]
        node_ids = [str(node["id"]) for node in participants]
        blockers = remove_blockers(deployment, left, departed_node_ids(ops.store, node_ids))
        if blockers:
            raise RuntimeError(LEFTOVERS_NOT_STOPPED.format(describe_unstopped(blockers)))
    except Exception:
        if takes_switch:
            switch.leave_if_owned(name)
        raise

    units = {key: value for key, value in (deployment.get("units") or {}).items()
             if key not in _FAILURE_FIELDS}
    _write(ops, deployment, UNLOADED_STATE, units)
    loaded = None
    try:
        loaded = load_deployment(
            ops, {"name": name, "confirm": LOAD_CONFIRM}, progress, placement_status,
        )
        if takes_switch:
            switch.repoint(credential_id, int(units.get("port", 0) or 0), name)
    except Exception as error:
        if loaded is not None:
            # Serving, but the switch would not take it: stop it again rather
            # than leave a model on a GPU the mode file does not name.
            current = ops.store.get_pooled_deployment(name) or deployment
            ops.stop_units_for(current)
        if takes_switch:
            switch.leave_if_owned(name)
        _write(ops, deployment, FAILED_STATE, {
            **units, "failure": RESTART_FAILED.format(_sentence(str(error))),
        })
        raise
    return {**loaded, "recovered": True}


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".!?" else text + "."


def _write(ops: Any, deployment: Dict[str, Any], state: str, units: Dict[str, Any]) -> None:
    with ops._serving_lock():
        ops.store.put_pooled_deployment(
            name=str(deployment.get("name", "")), state=state,
            model_id=str(deployment.get("model_id", "") or ""),
            node_ids=list(deployment.get("node_ids") or []), units=units,
            endpoint=str(deployment.get("endpoint", "") or ""),
            credential_id=str(deployment.get("credential_id", "") or ""),
        )
