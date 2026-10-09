"""Removing a worker: what refuses it, what Vaelor takes off it, what stays (ACC-117).

Removing a worker used to delete the only SSH credential that could reach it
while Vaelor's GPU/pooled model units and its telemetry agent kept running
there, and the plan and the job both called the removal clean. Afterwards a
split deployment could not be removed at all (its units live on a machine
Vaelor can no longer reach), a replicated one kept that machine's GPU, and the
agent kept posting with a key nobody could revoke on the worker.

The rules this module owns, for every door that removes a worker - the approved
plan's job (`ClusterOperations.remove_node`), the unjoined-enrolment route
(`ClusterManager.remove_node`), architecture eviction, and the plan text:

* **A plain removal is refused while any model deployment names the machine**,
  whatever its state: removing the deployment first is the one path that stops
  its units, and it needs the credential this would delete.
* **A FORCED removal (owner decision, 2026-09-28) goes ahead only for a machine
  Vaelor cannot reach**, after a typed confirmation (:data:`FORCED_REMOVAL_ACK`).
  It names every deployment that loses the machine, marks each record degraded
  with a plain reason (``units.lost_nodes`` and ``units.degraded_reason`` - the
  ``state`` word is left for the GPU mode switch, which tears a cluster down
  for any state other than ``healthy``), says what may still be running on the
  dead box, and still revokes its telemetry key and deletes its credential. A
  reachable machine in use is refused even when forced: its deployments can be
  removed properly. The lost machine no longer blocks removing those
  deployments later: every removal path asks the cluster store which of a
  record's machines have left the fleet (`cluster_placement.departed_node_ids`);
  the record's own ``lost_nodes`` marker is for display only.
* **The machine leaves the cluster first, then its telemetry agent is taken
  off** (review SC4), while the credential still works. A leave that fails
  changes nothing; an agent that cannot be removed after a successful leave
  leaves the record enrolled-but-unjoined, so a retry takes the unjoined path.
* **What stays is said**: Docker, pulled images and downloaded model files.

Runs in the process that calls it: the executor for the job, the control plane
for the route and the plan.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional

from .cluster_store import NODE_NOT_FOUND
from .gpu_memory_pool import pool_left_note
from .credential_broker_client import CredentialError
from .ssh_transport import SshTransport, SshTransportError, channel_drops

LOGGER = logging.getLogger(__name__)

#: What a removal leaves on the machine, stated wherever a removal is planned or
#: reported. Vaelor installs Docker during a join when it is missing, and GPU
#: serving keeps its model cache and runtime image on each node by design
#: (`gpu_pool_operations.remove` answers `model_cache_retained`).
LEFT_ON_MACHINE = (
    "Docker, the container images Vaelor pulled and any downloaded model files "
    "under /var/lib/vaelor/models stay on the machine; delete them there if you "
    "no longer need them."
)

#: What may still be RUNNING on a machine removed by force while unreachable.
LEFT_RUNNING_ON_DEAD_MACHINE = (
    "Because Vaelor could not reach it, anything it started there may still be "
    "running when the machine comes back: model servers (vaelor-gpu-* and "
    "vaelor-pooled-* systemd units), the telemetry agent and its GPU sampler "
    "(vaelor-telegraf and vaelor-gpu-sampler), Docker, pulled "
    "images and downloaded models. Stop and delete them on the machine itself."
)

#: Fixed words for a leave that failed (review round 3): the machine's own
#: answer is logged, never shown (LESSONS 24).
UNREACHABLE_FIXED = "Vaelor could not reach the machine over SSH"
LEAVE_REFUSED = (
    "The machine answered but could not leave the cluster (its answer is in this controller's "
    "log). Nothing was removed; run 'docker swarm leave' on it, then remove it again."
)

#: What a removal of a worker whose stored cluster entry would not decode says.
UNREAD_ENTRY_LEFT = (
    "Its cluster entry on this controller could not be read, so the machine was asked to leave "
    "the cluster itself; this controller's cluster list may show it as down until that entry is "
    "removed there (docker node rm)."
)

#: The typed confirmation a forced removal requires, in the plan and the job.
FORCED_REMOVAL_ACK = "remove-unreachable-machine"

class WorkerInUse(ValueError):
    """Refused: a model deployment still places work on this machine."""


class WorkerNotReached(ValueError):
    """Refused: Vaelor could not do what a plain removal needs on the machine."""


TELEMETRY_REMOVED = "removed"
TELEMETRY_NOT_INSTALLED = "not-installed"
TELEMETRY_LEFT_RUNNING = "left-running"


def deployments_using_node(
    deployments: Optional[List[Dict[str, Any]]], node_id: str
) -> List[str]:
    """Names of every model deployment record that places work on ``node_id``."""
    names = []
    for deployment in deployments or []:
        if not isinstance(deployment, dict):
            continue
        placed = {str(item) for item in deployment.get("node_ids") or []}
        if str(node_id) in placed:
            names.append(str(deployment.get("name", "") or "unnamed"))
    return sorted(set(names))


def in_use_reason(machine: str, names: List[str]) -> str:
    """Why this machine cannot be removed yet, in the owner's words."""
    listed = ", ".join(names)
    plural = len(names) != 1
    return (
        "{machine} is still part of the model deployment{s} {listed}. Remove "
        "{them} on the Deployments tab first: removing a deployment stops its "
        "model server on every machine it uses, and Vaelor could no longer do "
        "that for {machine} once the machine is removed. If {machine} is dead "
        "and the deployment cannot be removed, review a forced removal."
    ).format(
        machine=machine, s="s" if plural else "", listed=listed,
        them="them" if plural else "it",
    )


def refuse_if_in_use(store: Any, node: Dict[str, Any]) -> None:
    """Raise ``ValueError`` when a deployment record still names this node."""
    names = deployments_using_node(store.list_pooled_deployments(), node["id"])
    if names:
        raise WorkerInUse(in_use_reason(machine_name(node), names))


def machine_name(node: Dict[str, Any]) -> str:
    return str(node.get("name") or node.get("host") or "This machine")


def telemetry_provisioned(store: Any, node_id: str) -> bool:
    """Whether Vaelor holds a reporting key for this node's telemetry agent."""
    return any(
        stored_id == node_id and stored_hash
        for stored_id, stored_hash in store.ingest_key_hashes()
    )


def failure_words(error: BaseException) -> str:
    """What actually went wrong reaching a worker, in the owner's words.

    Review nit: every failure used to read "could not reach", including a
    stored sign-in the broker could not read and a command the machine ran and
    refused. The three are different faults with different remedies.
    """
    detail = str(error)[:200] or type(error).__name__
    if isinstance(error, CredentialError):
        return "Vaelor could not read the machine's stored sign-in ({})".format(detail)
    refused_sign_in, changed_key = _paramiko_failures()
    # PH-R6: paramiko's own errors are not OSErrors. A refused sign-in and a
    # changed host key are said as what they are, not as a command that failed.
    if isinstance(error, refused_sign_in):
        return "The machine refused Vaelor's stored sign-in ({})".format(detail)
    if isinstance(error, changed_key):
        return ("The machine's SSH host key is not the one Vaelor recorded at enrolment; remove "
                "and enrol it again if the change is expected ({})".format(detail))
    if isinstance(error, (SshTransportError,) + channel_drops()):
        return "Vaelor could not reach the machine over SSH ({})".format(detail)
    return "the command on the machine failed ({})".format(detail)


def _paramiko_failures() -> tuple:
    """paramiko's refused-sign-in and changed-host-key classes, or empty tuples."""
    try:
        import paramiko
    except ImportError:
        return (), ()
    return (paramiko.AuthenticationException,), (paramiko.BadHostKeyException,)


def take_telemetry_off(
    store: Any, node: Dict[str, Any], transport_for: Callable[[], Any],
    *, force: bool, left_cluster: bool = False,
) -> str:
    """Remove Vaelor's telemetry agent from the machine and revoke its key.

    Returns :data:`TELEMETRY_REMOVED`, :data:`TELEMETRY_NOT_INSTALLED` (Vaelor
    holds no key, so it installed nothing it still answers for), or - only when
    ``force`` - :data:`TELEMETRY_LEFT_RUNNING`, after which the key is revoked
    anyway. A plain removal that cannot take the agent off raises a plain
    ``ValueError``; ``left_cluster`` says the machine already left the cluster
    in this removal, so the sentence says that too rather than "nothing
    changed".
    """
    from .worker_telemetry_runtime import WorkerTelemetryRuntime

    if not telemetry_provisioned(store, node["id"]):
        return TELEMETRY_NOT_INSTALLED
    try:
        WorkerTelemetryRuntime().uninstall(transport_for())
    except Exception as error:  # noqa: BLE001 - every failure is reported below
        LOGGER.warning(
            "could not remove the telemetry agent from node %s: %s",
            node["id"], error,
        )
        if not force:
            raise WorkerNotReached(
                "{} to remove its telemetry agent. {} Fix that and try again, or "
                "review a forced removal, which leaves the agent installed "
                "there.".format(
                    failure_words(error),
                    "It has already left the cluster and stays enrolled, so a "
                    "retry only removes the agent and the enrolment."
                    if left_cluster else "Nothing was removed.",
                )
            ) from error
        store.set_ingest_key_hash(node["id"], "")
        return TELEMETRY_LEFT_RUNNING
    store.set_ingest_key_hash(node["id"], "")
    return TELEMETRY_REMOVED


#: What a removal says when the worker profile's pieces could not be taken off
#: (VD-194 P2). The marker left behind makes a later full install refuse until
#: it is passed --leave-worker-role, so the owner is told where it is.
PROFILE_LEFT = (
    "Vaelor could not take the worker profile off the machine ({}); its marker "
    "/etc/vaelor/worker-profile.json may remain, and a full install there then asks "
    "for --leave-worker-role."
)


def profile_applied(store: Any, node_id: str) -> bool:
    """Whether this controller laid the worker profile down on the machine.

    Read from the node's stored software record: an amd-smi record (written
    only by an apply) or a reading that found the profile's marker. A record
    that cannot be read counts as applied, so the take-off is tried rather
    than a fence left behind silently.
    """
    try:
        record = store.node_profile(node_id)
    except Exception:  # noqa: BLE001 - unknown is treated as applied
        return True
    if not isinstance(record, dict) or record.get("unreadable"):
        return True
    reading = record.get("reading") if isinstance(record.get("reading"), dict) else {}
    marker = (reading.get("items") or {}).get("marker") if isinstance(reading.get("items"), dict) else None
    return isinstance(record.get("amd_smi"), dict) or (isinstance(marker, dict) and marker.get("present") is True)


#: What the profile leaves on a machine that marks it as laid down: the
#: installer's fence, and the amd-smi preferences file an apply writes before
#: it holds the packages (a failed apply can leave that without a record).
PROFILE_MARKS = ("/etc/vaelor/worker-profile.json", "/etc/apt/preferences.d/vaelor-amd-smi")


def profile_on_machine(transport: Any) -> Any:
    """Whether the machine holds the profile's marks: True, False, or None when unread.

    Read on the machine (LESSONS 9), not from the stored record, which can
    miss a marker written after its last reading. ``stat -c %f`` prints the
    raw mode in hex, whatever the machine's language; only an answer that
    parses as a regular file is "present", and only stat's own "No such file"
    is "absent" - anything else is unread (LESSONS 8).
    """
    import stat as stat_module

    from .worker_telemetry_runtime import _absent

    found = False
    for path in PROFILE_MARKS:
        try:
            answer = str(transport.run(["stat", "-c", "%f", path], sudo=True)).strip()
        except Exception as error:  # noqa: BLE001 - absent or unread, told apart below
            if _absent(error):
                continue
            return None
        try:
            mode = int(answer, 16)
        except ValueError:
            return None
        found = found or stat_module.S_ISREG(mode)
    return found


def take_profile_off_safely(store: Any, node: Dict[str, Any], transport_for: Callable[[], Any]) -> str:
    """Take the worker profile off a reachable machine that leaves; ``""`` or what was left.

    Whether there is a profile to take off is read on the machine (its marker
    or amd-smi preferences file); the stored record decides only when that
    could not be read. A machine that never got the profile runs nothing - the
    full installer's own holds and files there are not the profile's.
    Never raises: the machine has already left the cluster by now, and a
    profile that could not be taken off is said, not allowed to undo that.
    """
    from .worker_profile_job import take_profile_off

    try:
        transport = transport_for()
        present = profile_on_machine(transport)
        if present is None:
            present = profile_applied(store, node["id"])
        if not present:
            return ""
        take_profile_off(store, node, transport)
    except Exception as error:  # noqa: BLE001 - said in the result; logged with its cause
        LOGGER.warning("could not take the worker profile off node %s: %s", node["id"], error)
        if not profile_applied(store, node["id"]):
            return ""
        return PROFILE_LEFT.format(failure_words(error))
    return ""


#: The pauses between reachability attempts before a forced removal may call a
#: machine gone, in seconds (review R3): three tries over about half a minute,
#: so a machine rebooting or on a flaky link is not declared dead by one
#: dropped packet. Module-level so a test can shorten it.
REACHABILITY_RETRY_DELAYS = (10.0, 20.0)
_sleep = time.sleep

REACHABLE = "reachable"
UNREACHABLE = "unreachable"
CANNOT_TELL = "cannot-tell"


def machine_reachability(transport_for: Callable[[], Any]) -> str:
    """Whether the machine answers over its pinned SSH, asked more than once.

    :data:`REACHABLE` on the first answer; :data:`UNREACHABLE` only after every
    attempt failed to connect; :data:`CANNOT_TELL` when the stored sign-in
    itself cannot be read - the broker failing says nothing about the machine,
    so it must never be taken as proof the machine is dead.
    """
    delays = list(REACHABILITY_RETRY_DELAYS)
    while True:
        try:
            transport_for().run(["uname", "-n"])
            return REACHABLE
        except CredentialError:
            return CANNOT_TELL
        except Exception:  # noqa: BLE001 - a failed attempt; maybe try again
            if not delays:
                return UNREACHABLE
            _sleep(delays.pop(0))


def answers_now(transport_for: Callable[[], Any]) -> bool:
    """One immediate probe, no retries: does the machine answer right now?"""
    try:
        transport_for().run(["uname", "-n"])
    except Exception:  # noqa: BLE001 - not answering, for whatever reason
        return False
    return True


def came_back(
    machine: str, in_use: List[str], *, changed: bool = False,
    left_cluster: bool = False, agent_removed: bool = False,
) -> "WorkerInUse":
    """The refusal when a machine answers part-way through a forced removal.

    ``changed=False``: it answered the probe made immediately before the
    forced leave, so nothing was touched and the ordinary path still works.
    Otherwise say what already happened - the forced leave ran (the record is
    now enrolled but not joined) and/or the telemetry agent came off - and
    give the steps that work from there: a deployment's removal resolves its
    machines as JOINED workers, so the machine must join again first
    (final-check verdict).
    """
    deployments = ", ".join(in_use)
    if not changed:
        return WorkerInUse(
            "{} answered just before it was taken out of the cluster, so it is "
            "not dead and nothing was changed. Remove {} on the Deployments "
            "tab, which stops its model servers cleanly, then remove the "
            "machine normally.".format(machine, deployments)
        )
    done = []
    if left_cluster:
        done.append("it has already left the cluster")
    if agent_removed:
        done.append(
            "its telemetry agent was removed and its reporting key revoked"
        )
    return WorkerInUse(
        "{machine} answered during the forced removal, so it is not dead. "
        "Vaelor stopped before marking {deployments} degraded or deleting the "
        "machine, but {done}; it stays enrolled. To finish: join it again "
        "under Setup > Machines awaiting join, then remove {deployments} on "
        "the Deployments tab, then remove the machine.".format(
            machine=machine, deployments=deployments,
            done=" and ".join(done) or "part of the removal had already run",
        )
    )


def mark_deployments_lost(
    operations: Any, names: List[str], node: Dict[str, Any]
) -> List[str]:
    """Record on each deployment that a forced removal took this machine away.

    ``units.lost_nodes`` gains the machine and ``units.degraded_reason`` says so
    in plain words; ``state`` is kept (see the module docstring). Written under
    the GPU serving lock when this process has one, so the mode watch's own
    record writes cannot interleave with it.
    """
    store = operations.store
    gpu = getattr(operations, "gpu_operations", None)
    lock_for = getattr(gpu, "_serving_lock", None)
    machine = machine_name(node)
    marked = []
    with (lock_for() if callable(lock_for) else nullcontext()):
        for name in names:
            record = store.get_pooled_deployment(name)
            if record is None:
                continue
            units = dict(record.get("units") or {})
            lost = [dict(entry) for entry in units.get("lost_nodes") or []]
            lost.append({
                "node_id": str(node["id"]), "name": machine,
                "at": int(time.time()),
                "reason": "removed from the fleet by force while unreachable",
            })
            units["lost_nodes"] = lost
            units["degraded_reason"] = (
                "{} was removed from the fleet by force while Vaelor could not "
                "reach it, so this deployment has lost that machine's part. "
                "Remove the deployment and deploy it again on machines that "
                "are in service.".format(
                    ", ".join(entry["name"] for entry in lost)
                )
            )
            store.put_pooled_deployment(
                name=record["name"], state=record["state"],
                model_id=record.get("model_id", ""),
                node_ids=list(record.get("node_ids") or []), units=units,
                endpoint=record.get("endpoint", ""),
                credential_id=record.get("credential_id", ""),
            )
            marked.append(name)
    return marked


def removal_message(result: Dict[str, Any]) -> str:
    """The finished job's message: what was done, and what was not."""
    parts = ["Worker removed and its SSH credential deleted."]
    lost = list(result.get("lost_deployments") or [])
    if lost:
        parts.append(
            "It was removed by force while unreachable, so {} lost {} part "
            "and {} now marked degraded.".format(
                ", ".join(lost), "its" if len(lost) == 1 else "their",
                "is" if len(lost) == 1 else "are",
            )
        )
    telemetry = result.get("telemetry_agent")
    if telemetry == TELEMETRY_REMOVED:
        parts.append("Its telemetry agent was removed.")
    elif telemetry == TELEMETRY_LEFT_RUNNING:
        parts.append(
            "Vaelor could not reach it, so its telemetry agent is still "
            "installed there; its reporting key was revoked."
        )
    if result.get("forced") and result.get("worker_leave_warning"):
        parts.append(
            "It did not leave the cluster cleanly, so it may still believe it "
            "is a member; run 'docker swarm leave --force' on it."
        )
    if result.get("swarm_entry_unread"):
        parts.append(UNREAD_ENTRY_LEFT)
    if result.get("profile_left"):
        parts.append(str(result["profile_left"]))
    if result.get("forced_while_unreachable"):
        parts.append(LEFT_RUNNING_ON_DEAD_MACHINE)
    else:
        parts.append(LEFT_ON_MACHINE)
    # VD-161: a GPU memory pool file Vaelor wrote there stays too, and the
    # sign-in that could have removed it is gone now - so it is said.
    return " ".join(parts) + str(result.get("gpu_memory_pool_left") or "")


def remove_worker_node(operations: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """The approved-plan removal job (``cluster.node.remove``), in order.

    Refuse a plain removal while a deployment names the node; for a forced one,
    require the typed acknowledgement and refuse a machine that is reachable
    and in use. Then leave the cluster, take the telemetry agent off, mark any
    deployment that lost the machine, and delete the record and credential.
    Housed here rather than in `cluster_operations`, which is at the ceiling.
    """
    force = bool(payload.get("force", False))
    expected = "force-remove-worker-node" if force else "remove-worker-node"
    if payload.get("confirm") != expected:
        raise ValueError("Confirm the reviewed worker removal.")
    if force and str(payload.get("data_loss_ack", "")).strip() != FORCED_REMOVAL_ACK:
        raise ValueError(
            'A forced removal needs the typed confirmation "{}".'.format(
                FORCED_REMOVAL_ACK
            )
        )
    store = operations.store
    try:
        node = store.get_node(str(payload.get("node_id", "")), include_credential=True)
    except ValueError:
        # Review round 2 (LESSONS 22): a record that will not decode is
        # removed as far as it reads; its exit is this plan.
        node = store.node_core(str(payload.get("node_id", "")))
    if node is None:
        raise ValueError(NODE_NOT_FOUND)

    def transport():
        profile = operations.broker.resolve(node["credential_id"], "cluster-node")
        return SshTransport(profile, timeout=120)

    in_use = deployments_using_node(store.list_pooled_deployments(), node["id"])
    if in_use and not force:
        raise WorkerInUse(in_use_reason(machine_name(node), in_use))
    if in_use:
        reach = machine_reachability(transport)
        if reach == CANNOT_TELL:
            raise WorkerNotReached(
                "Vaelor could not read the stored sign-in for {}, so it cannot "
                "tell whether the machine is really gone, and a forced removal "
                "is refused. Check the credential store, then try again.".format(
                    machine_name(node)
                )
            )
        if reach == REACHABLE:
            raise WorkerInUse(
                "{} can still be reached, so it is not removed by force: "
                "remove {} on the Deployments tab first, which stops its model "
                "servers cleanly, then remove the machine.".format(
                    machine_name(node), ", ".join(in_use)
                )
            )
    swarm_node_id = str(node.get("labels", {}).get("swarm_node_id", ""))
    result: Dict[str, Any] = {"removed": True, "forced": False, "swarm_node_id": ""}
    if not swarm_node_id and node.get("record_unreadable") and node.get("state") == "joined" and not in_use:
        # Its cluster entry will not decode, so the controller cannot name it to
        # drain; the machine itself is asked to leave (no deployment names it).
        try:
            transport().run(["docker", "swarm", "leave"], sudo=True, timeout=180)
        except Exception as error:  # noqa: BLE001 - classified below; the raw text is logged
            # Review round 3 (LESSONS 22 / 24): a machine already out of the
            # cluster (left by hand, or by an earlier attempt) is the state
            # wanted, so the removal continues. Any other answer is said in
            # fixed words; the machine's own text goes to the log only.
            from .cluster_driver import DockerSwarmDriver

            if not DockerSwarmDriver._already_left_swarm(str(error)):  # noqa: SLF001 - its one rule
                LOGGER.warning("node %s could not leave the cluster: %s", node["id"], error)
                if isinstance(error, channel_drops()):
                    raise WorkerNotReached("{} to ask it to leave the cluster. Nothing was removed.".format(
                        UNREACHABLE_FIXED)) from error
                raise WorkerNotReached(LEAVE_REFUSED) from error
        result["swarm_entry_unread"] = True
    if swarm_node_id:
        # Final check: one more immediate probe before a machine in use is
        # taken out of the cluster by force, so a machine that is back is, in
        # the common case, left exactly as it was.
        if in_use and answers_now(transport):
            raise came_back(machine_name(node), in_use)
        result = operations.driver.remove_worker(
            transport(), swarm_node_id, force=force,
        )
        # SC4: the machine is out of the cluster now. Say so in the record
        # before the next step, so a failure there leaves an honest
        # enrolled-but-unjoined row a retry can finish, not a stale binding.
        labels = dict(node.get("labels") or {})
        labels.pop("swarm_node_id", None)
        store.update_node(node["id"], labels=labels, state="enrolled")
        # R3: a forced leave that met no error ran `docker swarm leave` ON the
        # machine - it is alive. Stop before losing its deployments.
        if in_use and not result.get("worker_leave_warning"):
            raise came_back(
                machine_name(node), in_use, changed=True, left_cluster=True,
            )
    telemetry = take_telemetry_off(
        store, node, transport, force=force, left_cluster=bool(swarm_node_id),
    )
    if in_use and telemetry == TELEMETRY_REMOVED:
        # R3: the agent was removed over SSH, so the machine answered.
        raise came_back(
            machine_name(node), in_use, changed=True,
            left_cluster=bool(swarm_node_id), agent_removed=True,
        )
    # VD-194 P2: a reachable machine leaving keeps none of the worker profile.
    profile_left = "" if in_use else take_profile_off_safely(store, node, transport)
    lost = mark_deployments_lost(operations, in_use, node) if in_use else []
    credential_id = store.delete_node(node["id"])
    if credential_id:
        operations.broker.delete(credential_id)
    forget_ingest_status(node["id"])
    return {
        **result,
        "node_id": node["id"],
        "telemetry_agent": telemetry,
        "lost_deployments": lost,
        "forced_while_unreachable": bool(in_use),
        "left_on_machine": (
            LEFT_RUNNING_ON_DEAD_MACHINE if in_use else LEFT_ON_MACHINE
        ),
        "gpu_memory_pool_left": pool_left_note(node.get("inventory")),
        "profile_left": profile_left,
    }


def forget_ingest_status(node_id: str) -> None:
    """Drop the removed node's ingest record (review nit). Only the control
    plane holds one; in the executor this is a no-op on an empty map."""
    try:
        from .telemetry_ingest_status import forget

        forget(node_id)
    except Exception:  # noqa: BLE001 - a stale entry for a gone node is harmless
        LOGGER.debug("no ingest status to forget for node %s", node_id)
