"""AI Chat while the cluster serves: the cluster takes it once, then it is the owner's (VD-210).

`ClusterModeSwitch.repoint` moves ``ai-chat`` onto the cluster credential when
the cluster first answers. Until VD-210 the reconcile's healthy pass then put it
back on the cluster every 30 s, so a connection the owner chose in AI Chat was
undone within one pass and the only way to use another model was to remove the
cluster. Every rule here answers one question - does AI Chat still follow the
cluster, or has the owner chosen? - and every place that moves the lease in
Mode B asks it, so the reconcile, `leave`, the unload and the Load cannot answer
it four different ways (LESSONS 6).

**The owner's choice is the broker's own ``ai-chat`` assignment.** It is
durable (the vault keeps it across a reboot), it is written by the owner's
connection picker and by nothing else while the box is in Mode B, and it needs
no second record. A second record would need a second writer: the mode file is
the executor's (`gpu_cluster_mode_state`), the picker runs in the control plane,
and two accounts writing one file is the lost update LESSONS 13 warns of. What
else can move ``ai-chat`` in Mode B, and why it is not mistaken for a choice:

* `enter` clears it, inside the window ``deploy_in_flight`` marks, which
  `repoint` closes by taking it - the first serve;
* a manual unload clears it, and only while it is on the cluster;
* the Load and the healthy pass put it back on the cluster, and only while it
  follows the cluster;
* deleting the connection it was on clears it (the broker's cascade);
* a model deploy that would assign it is refused in Mode B
  (`executor_gpu_deploy._refuse_deploy_under_cluster`).

So **AI Chat follows the cluster** while its lease is the cluster credential,
is empty, or pins the Assistant's NPU model (never AI Chat's, W4d-D26); any
other connection is the owner's choice and stands. An empty lease is healed
onto the cluster because nothing chose it: it is the first-serve window, a
manual unload's park, or a connection that no longer exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Tuple

from .credential_broker import CredentialError
from .gpu_serving_target import (
    AI_CHAT_PURPOSE, MODE_CLUSTER, UNLOAD_CAUSE_IDLE, UNLOAD_CAUSE_MANUAL,
    no_active_credential,
)
from .managed_local_credentials import PREFIX as MANAGED_LOCAL_PREFIX
from .managed_local_credentials import pins_an_flm_tag


LOGGER = logging.getLogger(__name__)

#: What `leave` reports when it leaves the owner's AI Chat connection in place
#: instead of restoring the one AI Chat had before clustering (VD-210 item 4).
OWNER_CHOICE_KEPT = "owner-choice-kept"


@dataclass(frozen=True)
class AiChatLease:
    """The ``ai-chat`` assignment as read now: ``readable`` False if it could not be."""

    readable: bool
    credential_id: str = ""
    model: str = ""


def read_ai_chat(broker: Any) -> AiChatLease:
    """Read the ``ai-chat`` lease. "Nothing assigned" is a readable empty lease."""
    try:
        lease = broker.resolve_active(AI_CHAT_PURPOSE)
    except CredentialError as error:
        return AiChatLease(readable=no_active_credential(error))
    except (AttributeError, TypeError, ValueError):
        return AiChatLease(readable=False)
    if not isinstance(lease, Mapping):
        return AiChatLease(readable=False)
    return AiChatLease(
        readable=True, credential_id=str(lease.get("credential_id", "") or ""),
        model=str(lease.get("model") or ""),
    )


def follows_cluster(lease: AiChatLease, cluster_credential_id: str) -> Optional[bool]:
    """True while AI Chat follows the cluster; False once the owner chose; None if unread.

    See the module docstring for why an empty lease and the NPU model's lease
    follow the cluster and every other connection is the owner's.
    """
    if not lease.readable:
        return None
    if not lease.credential_id or lease.credential_id == str(cluster_credential_id or ""):
        return True
    return bool(pins_an_flm_tag(lease.model))


def heal_onto_cluster(switch: Any, state: Any) -> bool:
    """The healthy pass: AI Chat back on the cluster only while it follows it.

    Answers whether the lease was moved. An unreadable lease is not moved: a
    broker that could not say what the owner chose has not said "nothing".
    """
    lease = read_ai_chat(switch.broker)
    if follows_cluster(lease, state.cluster_credential_id) is not True:
        return False
    if lease.credential_id == state.cluster_credential_id:
        return False
    switch._quietly(
        lambda: switch.broker.activate(state.cluster_credential_id, AI_CHAT_PURPOSE),
        "point AI Chat at the cluster",
    )
    return True


def leave_keeps_owner_choice(switch: Any, state: Any) -> Optional[AiChatLease]:
    """The owner's lease `leave` must keep, or ``None`` when it restores as before.

    VD-210 item 4: the connection AI Chat had before clustering comes back
    only while AI Chat is still on the cluster (or on nothing). An unreadable
    lease restores, as every leave did before VD-210 - a Mode A with AI Chat
    on its old model is the recoverable direction.
    """
    lease = read_ai_chat(switch.broker)
    if follows_cluster(lease, state.cluster_credential_id) is False:
        return lease
    return None


def keep_owner_choice(switch: Any, lease: AiChatLease, unassign_because: str) -> Tuple[bool, str]:
    """`leave`'s restore step when the owner chose: ``(restored, reason)``.

    The owner's connection stays. The one exception is the controller unit
    that would not stop (VD-127): a managed-local GPU lease would have the
    failure-watch relaunch llama.cpp beside it, so that lease is cleared, as
    the restore clears the previous one in the same case. A hosted or network
    connection starts nothing on this GPU and is left alone.
    """
    if unassign_because and lease.credential_id.startswith(MANAGED_LOCAL_PREFIX):
        switch._degrade_ai_chat()
        return False, unassign_because
    return False, OWNER_CHOICE_KEPT


def record_unload(switch: Any, deployment_name: str, *, idle: bool) -> None:
    """Record why the Mode B deployment is unloaded, and park AI Chat on a manual one.

    Called by the unload under the serving lock (a re-entrant lock, so taking
    it again here is safe). Only when the mode file names ``deployment_name``.
    A manual unload clears ``ai-chat`` only while it follows the cluster - or
    could not be read, which parks as every unload did before VD-210 - and
    never takes the owner's own connection away.
    """
    with switch.watch_lock:
        state = switch.store.read()
        if state.mode != MODE_CLUSTER or state.deployment_name != str(deployment_name):
            return
        cause = UNLOAD_CAUSE_IDLE if idle else UNLOAD_CAUSE_MANUAL
        if state.unload_cause != cause:
            switch.store.write(replace(state, unload_cause=cause))
        if idle:
            return
        lease = read_ai_chat(switch.broker)
        on_the_cluster = follows_cluster(lease, state.cluster_credential_id) is not False
        if on_the_cluster and (lease.credential_id or not lease.readable):
            switch._degrade_ai_chat()


def resume_after_load(switch: Any, deployment_name: str, credential_id: str) -> bool:
    """A Load's end: the unload cause cleared, and AI Chat back only if it follows.

    Answers whether ``ai-chat`` was pointed at the cluster. Raises what the
    broker raises, which the Load logs and the reconcile repairs.
    """
    with switch.watch_lock:
        state = switch.store.read()
        if state.mode != MODE_CLUSTER or state.deployment_name != str(deployment_name):
            return False
        if state.unload_cause:
            switch.store.write(replace(state, unload_cause=""))
        if follows_cluster(read_ai_chat(switch.broker), credential_id) is False:
            return False
        switch.broker.activate(credential_id, AI_CHAT_PURPOSE)
        return True


def settle_unload_cause(switch: Any, state: Any, *, unloaded: bool) -> Any:
    """Keep the recorded unload cause true on the reconcile's pass; the state after.

    Healthy: a cause left behind by a Load that died before clearing it is
    cleared, or the door would treat a serving cluster as manually unloaded.
    Unloaded with no cause recorded - a box unloaded before VD-210 - the cause
    is derived once from the lease as VD-136 read it (on the cluster is idle,
    anything else is manual, the direction that never wakes a model nobody
    asked for); an unreadable lease records nothing and is asked again.
    """
    if not unloaded:
        if state.unload_cause:
            state = replace(state, unload_cause="")
            switch.store.write(state)
        return state
    if state.unload_cause:
        return state
    lease = read_ai_chat(switch.broker)
    if not lease.readable:
        return state
    idle = bool(lease.credential_id) and lease.credential_id == state.cluster_credential_id
    state = replace(state, unload_cause=UNLOAD_CAUSE_IDLE if idle else UNLOAD_CAUSE_MANUAL)
    switch.store.write(state)
    return state
