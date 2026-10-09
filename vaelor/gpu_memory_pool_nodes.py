"""The GPU memory pool jobs: this controller, an enrolled worker, and a worker restart.

VD-161. Three owner-confirmed jobs, each changing exactly one machine:

* ``host.gpu-memory.apply`` - this controller, through the root bridge's
  dedicated verbs (`hardware_bridge_host_settings`).
* ``cluster.node.gpu-memory`` - one enrolled worker, over the SSH channel it
  was enrolled on. There is no fleet-wide form: the owner confirms each
  machine, because each one has to be restarted for the size to count.
* ``cluster.node.reboot`` - restart one worker. Its own job with its own
  confirmation, so changing a setting can never restart a machine.

None of the three is ever retried (`job_vocabulary.RECONFIRM_REQUIRED_JOB_TYPES`):
a retry copies the stored confirmation into a new job with no fresh review.

**The worker path and the controller path share every rule** - the status, the
bounds and the file text all come from `gpu_memory_pool` - and differ only in
who carries the write. A worker is reached as a full sudoer already, so nothing
is widened there; the one new token on that channel is the boot-image tool.

**The file's text never travels on standard input.** `SshTransport.run` sends
the sign-in password down standard input for ``sudo``; on a machine whose
``sudo`` asks for none, that line would reach the command instead, and a
``tee`` would write it into a world-readable file. The text is an argument of a
small writer here, which reads nothing from standard input.

**What was written is read back** (LESSONS 1): the job reports the setting the
machine returns after the change, and refuses to call it applied if the file
does not say what was asked for.

**A failure says what the machine holds now** (review S1). A rebuild that
fails - or a connection that drops under it - is rolled back, the roll-back is
rebuilt once, and the refusal says which of three things is true: the machine
is as it was; the old file is back but the boot image is unconfirmed; or the
old file could NOT be put back and the file now holds the new size. Whatever
happened, the machine is read again afterwards and the stored reading is
replaced, so the card never keeps showing the state from before the attempt.

**A restart is followed until the machine answers again** (review S7), by its
boot id, and its pool is read once it does - so "Restart needed" clears
without anyone pressing Recheck.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Optional, Tuple

from . import gpu_memory_pool as pool
from .cluster_store import NODE_NOT_FOUND
from .gpu_memory_pool_apply import BOOT_IMAGE_TOOLS
from .ssh_transport import (
    COMMAND_ENDED_WITHOUT_REASON, SshTransport, SshTransportError,
)

HOST_GPU_MEMORY_JOB = "host.gpu-memory.apply"
NODE_GPU_MEMORY_JOB = "cluster.node.gpu-memory"
NODE_REBOOT_JOB = "cluster.node.reboot"

ACTION_SET = "set"
ACTION_REVERT = "revert"

#: The exact ``confirm`` value each controller action must carry. The two
#: worker jobs' fixed tokens live with the other cluster tokens
#: (`cluster_job_confirmations`), where the job route checks them.
HOST_CONFIRMATIONS = {
    ACTION_SET: "set-gpu-memory-pool",
    ACTION_REVERT: "revert-gpu-memory-pool",
}
#: What an unconfirmed controller request is told, by the route and by the job.
CONFIRMATION_REQUIRED = "Review and confirm the GPU memory pool change first."
NODE_GPU_MEMORY_CONFIRM = "change-worker-gpu-memory"
NODE_REBOOT_CONFIRM = "reboot-worker-node"

REBUILD_TIMEOUT_SECONDS = 600

#: How long a restarted worker is waited for, and how often it is asked. An
#: upper bound on a restart, not a measurement of one: a machine that takes
#: longer is reported as not yet back, and Recheck reads it when it is.
RESTART_WAIT_SECONDS = 480
RESTART_POLL_SECONDS = 10
#: Each look for the machine during the follow opens a connection with at
#: most this timeout, and never more than the window has left, so a machine
#: that is down cannot hold the job past the window by a whole SSH timeout.
FOLLOW_CONNECT_SECONDS = 15
_SHORTEST_CONNECT_SECONDS = 5
#: A restart the owner confirmed is carried out only while it is still that
#: decision: one that waited in the job queue longer than this is refused.
RESTART_QUEUE_LIMIT_SECONDS = 600
BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"


def _channel_drops() -> tuple:
    """Every way a connection can die under a command that was already sent."""
    drops: tuple = (OSError, EOFError)
    try:
        import paramiko
    except ImportError:
        return drops
    return drops + (paramiko.SSHException,)


_CHANNEL_DROPS = _channel_drops()
#: A command that did not finish cleanly, for whatever reason.
_FAILURES = (SshTransportError,) + _CHANNEL_DROPS

#: What both pool jobs report while the change is being made.
PROGRESS_MESSAGE = "Writing the GPU memory pool setting and rebuilding the boot image"

#: Writes argv[2] to argv[1] in one step: a new file that must not already
#: exist and is never opened through a link, flushed to disk, then renamed
#: over the old one. Reads nothing from standard input.
_NODE_WRITER = (
    "import os, sys\n"
    "path, text = sys.argv[1], sys.argv[2]\n"
    "new = path + '.new'\n"
    "try:\n"
    "    os.unlink(new)\n"
    "except FileNotFoundError:\n"
    "    pass\n"
    "fd = os.open(new, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)\n"
    "with os.fdopen(fd, 'w', encoding='utf-8') as handle:\n"
    "    handle.write(text)\n"
    "    handle.flush()\n"
    "    os.fsync(handle.fileno())\n"
    "os.chmod(new, 0o644)\n"
    "os.replace(new, path)\n"
)


def _action(payload: Dict[str, Any]) -> str:
    action = str(payload.get("action", ""))
    if action not in (ACTION_SET, ACTION_REVERT):
        raise ValueError("Choose whether to set the GPU memory pool or remove the setting.")
    return action


def controller_job_confirmed(payload: Any) -> bool:
    """Whether a ``host.gpu-memory.apply`` payload carries its action's own token.

    Read by the job route, so an unconfirmed request never becomes a job, and
    again by the job itself, which is the check that counts.
    """
    if not isinstance(payload, dict):
        return False
    expected = HOST_CONFIRMATIONS.get(str(payload.get("action", "")))
    return expected is not None and payload.get("confirm") == expected


def outcome_message(name: str, action: str, status: Dict[str, Any]) -> str:
    """What one finished change says, from the status the machine returned."""
    if status.get("restart_pending"):
        if action == ACTION_SET:
            return (
                "{}: GPU memory pool set to {} GiB. Restart the machine for the "
                "new size to count."
            ).format(name, (status["override"]["bytes"] or 0) // pool.GIB)
        return (
            "{}: GPU memory pool setting removed. Restart the machine to go "
            "back to the kernel's own size."
        ).format(name)
    if status.get("restart_pending") is None:
        return "{}: the setting was written. {}".format(
            name, status.get("restart_reason", "")
        ).strip()
    return "{}: the GPU memory pool is already that size; no restart is needed.".format(name)


# --- this controller ------------------------------------------------------

def run_controller_pool_job(
    job: Dict[str, Any], store: Any, checkpoint: Callable[..., Any],
    bridge: Any = None,
) -> Dict[str, Any]:
    """``host.gpu-memory.apply``: set or remove this controller's pool setting."""
    payload = job["payload"]
    action = _action(payload)
    if not controller_job_confirmed(payload):
        raise ValueError(CONFIRMATION_REQUIRED)
    if bridge is None:
        from .hardware_bridge_client import HardwareBridgeClient

        bridge = HardwareBridgeClient()
    checkpoint(20, PROGRESS_MESSAGE, "starting")
    if action == ACTION_SET:
        status = bridge.gpu_memory_pool_set(payload.get("size_gib"))
    else:
        status = bridge.gpu_memory_pool_revert()
    return store.finish(
        job["id"], state="completed",
        message=outcome_message(pool.CONTROLLER_NAME, action, status),
        result={
            "status": pool.card_view(status),
            "restart_required": bool(status.get("restart_pending")),
        },
    )


# --- an enrolled worker ---------------------------------------------------

def read_node_facts(transport: Any) -> Dict[str, Any]:
    """One worker's gathered facts, or a refusal when it returned none."""
    facts = pool.parse_remote_facts(
        transport.run(["python3", "-c", pool.REMOTE_FACTS_PROGRAM])
    )
    if facts is None:
        raise RuntimeError(
            "This machine did not return its memory settings, so nothing was "
            "changed on it."
        )
    return facts


def _node(operations: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    node = operations.store.get_node(
        str(payload.get("node_id") or ""), include_credential=True
    )
    if node is None:
        raise ValueError(NODE_NOT_FOUND)
    return node


def _transport(
    operations: Any, node: Dict[str, Any], factory: Callable[..., Any], timeout: int = 120,
) -> Any:
    profile = operations.broker.resolve(node["credential_id"], "cluster-node")
    return factory(profile, timeout=timeout)


#: Prints the worker file's exact text as JSON (``null`` when there is no
#: file). The transport strips what a command prints, so ``cat`` would lose
#: a trailing newline and a restore would not be the file the owner had.
_NODE_READER = (
    "import json, os, sys\n"
    "try:\n"
    "    fd = os.open(sys.argv[1], os.O_RDONLY | os.O_NOFOLLOW)\n"
    "except FileNotFoundError:\n"
    "    print(json.dumps(None))\n"
    "else:\n"
    "    with os.fdopen(fd, encoding='utf-8') as handle:\n"
    "        print(json.dumps(handle.read()))\n"
)


_NOT_READ_BACK = "The settings file could not be read back."


def _read_exact(transport: Any) -> Optional[str]:
    """The worker file's text byte for byte, or ``None`` when it is absent."""
    output = transport.run(["python3", "-c", _NODE_READER, pool.CONFIG_PATH])
    try:
        text = json.loads(str(output))
    except ValueError as error:
        raise SshTransportError(_NOT_READ_BACK) from error
    if text is not None and not isinstance(text, str):
        raise SshTransportError(_NOT_READ_BACK)
    return text


def _put(transport: Any, text: Optional[str]) -> None:
    """Make the worker's file say ``text``, or remove it for ``None``."""
    if text is None:
        transport.run(["rm", "-f", pool.CONFIG_PATH], sudo=True)
        return
    transport.run(
        ["python3", "-c", _NODE_WRITER, pool.CONFIG_PATH, text], sudo=True
    )


def _rebuild(transport: Any) -> bool:
    try:
        transport.run(
            ["update-initramfs", "-u"], sudo=True, timeout=REBUILD_TIMEOUT_SECONDS
        )
    except _FAILURES:
        return False
    return True


def record_reading(
    operations: Any, node_id: str, status: Dict[str, Any], now: Callable[[], float],
) -> None:
    """Store one machine's pool reading, and when it was read, on its record.

    Merged into the record AS IT IS NOW, not into the copy the job started
    with: a Recheck that ran while the job was working wrote a newer
    inventory, and the job-start copy would put the old one back over it.
    """
    fresh = operations.store.get_node(node_id)
    if fresh is None:
        return
    inventory = dict(fresh.get("inventory") or {})
    inventory["gpu_memory_pool"] = status
    inventory["gpu_memory_pool_checked_at"] = int(now())
    operations.store.update_node(node_id, inventory=inventory)


def _roll_back(transport: Any, name: str, previous: Optional[str], now_holds: str) -> str:
    """Undo a change whose rebuild failed, and say which of three things is true.

    The restore is proved by reading the file again, as the controller's is:
    a write the connection accepted but that did not land is not put back.
    """
    try:
        _put(transport, previous)
        landed = _read_exact(transport) == previous
    except _FAILURES:
        landed = False
    if not landed:
        return (
            "The boot image could not be rebuilt on {name} and its previous "
            "setting could not be put back: the settings file on {name} {holds}. "
            "Recheck {name}, then set or remove the size again."
        ).format(name=name, holds=now_holds)
    if _rebuild(transport):
        return (
            "The boot image could not be rebuilt on {} with the new setting. "
            "Its previous setting was put back and the boot image rebuilt with "
            "it, so the machine is as it was."
        ).format(name)
    return (
        "The boot image could not be rebuilt on {name}. Its previous setting "
        "file was put back, but the boot image could not be rebuilt with it "
        "either, so it could not be confirmed which size {name}'s next restart "
        "will use. Recheck {name}, then run 'sudo update-initramfs -u' on it."
    ).format(name=name)


def change_worker_pool(
    operations: Any, payload: Dict[str, Any],
    transport_factory: Callable[..., Any] = SshTransport,
    now: Callable[[], float] = time.time,
) -> Dict[str, Any]:
    """``cluster.node.gpu-memory``: set or remove one worker's pool setting."""
    action = _action(payload)
    if payload.get("confirm") != NODE_GPU_MEMORY_CONFIRM:
        raise ValueError("Review and confirm this machine's GPU memory pool change first.")
    node = _node(operations, payload)
    name = str(node.get("name") or node["id"])
    gpu = (node.get("inventory") or {}).get("gpu")
    transport = _transport(operations, node, transport_factory)
    facts = read_node_facts(transport)
    status = pool.pool_status(facts, gpu)
    if action == ACTION_SET:
        size = payload.get("size_gib")
        pages: Optional[int] = pool.require_size_gib(size, status)
        wanted: Optional[str] = pool.render_config(pages)
        now_holds = pool.file_now_holds(size)
    else:
        pool.require_revert(status)
        pages, wanted, now_holds = None, None, pool.file_now_holds(None)
    tool = BOOT_IMAGE_TOOLS[0][0]
    try:
        transport.run(["stat", "-c", "%n", tool])
    except _FAILURES as error:
        raise RuntimeError(
            "{} has no boot-image tool Vaelor knows how to run "
            "(update-initramfs), so a new size could not be made to count at "
            "its next restart. Nothing was written.".format(name)
        ) from error
    # The previous file's own text, to put back exactly. Held for this job
    # only: it is Vaelor's one file, and it is neither stored nor served.
    previous = (
        None if status["override"]["state"] == pool.OVERRIDE_ABSENT
        else _read_exact(transport)
    )
    touched = False
    try:
        touched = True
        try:
            _put(transport, wanted)
        except _FAILURES as error:
            raise RuntimeError(
                "The GPU memory pool setting could not be written on {name}, "
                "and the connection did not say whether any of it landed. "
                "Recheck {name} to see what it holds.".format(name=name)
            ) from error
        if not _rebuild(transport):
            raise RuntimeError(_roll_back(transport, name, previous, now_holds))
        after = read_node_facts(transport)
        written = pool.parse_config(after.get("config"))
        expected_state = pool.OVERRIDE_VAELOR if pages is not None else pool.OVERRIDE_ABSENT
        if written["state"] != expected_state or written["pages"] != pages:
            raise RuntimeError(
                "{} did not read back the GPU memory pool setting that was just "
                "written, so Vaelor cannot say it is applied. Recheck the "
                "machine.".format(name)
            )
        new_status = pool.pool_status(after, gpu)
        record_reading(operations, node["id"], new_status, now)
        touched = False
        return {
            "node_id": node["id"], "name": name, "action": action,
            "status": pool.card_view(new_status),
            "restart_required": bool(new_status.get("restart_pending")),
            "message": outcome_message(name, action, new_status),
        }
    finally:
        if touched:
            # Something was attempted and did not finish cleanly: read the
            # machine again so the card shows what it holds NOW - or says
            # that it could not be read - never the state from before.
            try:
                reading = pool.pool_status(read_node_facts(transport), gpu)
            except (RuntimeError,) + _FAILURES:
                reading = pool.unknown_after_change(name)
            record_reading(operations, node["id"], reading, now)


def _boot_id(transport: Any) -> str:
    """The machine's boot id, or ``""`` when it does not answer."""
    try:
        return str(transport.run(["cat", BOOT_ID_PATH])).strip()
    except _FAILURES:
        return ""


def _wait_for_return(
    connect: Callable[[int], Any], before: str, progress: Callable[[int, str], Any],
    name: str, sleep: Callable[[float], None], monotonic: Callable[[], float],
) -> bool:
    """Whether the machine answered again with a NEW boot id inside the window.

    The boot id changes on every start, so "answers, and is a different boot"
    cannot be mistaken for a machine that simply had not gone down yet.
    Each look connects with a timeout no longer than what the window has
    left, so the deadline holds around the connection as well as the sleep.
    """
    deadline = monotonic() + RESTART_WAIT_SECONDS
    while True:
        left = deadline - monotonic() - RESTART_POLL_SECONDS
        if left < _SHORTEST_CONNECT_SECONDS:
            return False
        sleep(RESTART_POLL_SECONDS)
        progress(60, "Waiting for {} to start again".format(name))
        current = _boot_id(connect(int(min(FOLLOW_CONNECT_SECONDS, left))))
        if current and current != before:
            return True


def reboot_worker(
    operations: Any, payload: Dict[str, Any],
    transport_factory: Callable[..., Any] = SshTransport,
    *, progress: Optional[Callable[[int, str], Any]] = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    now: Callable[[], float] = time.time,
    queued_at_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """``cluster.node.reboot``: restart one worker the owner named and confirmed.

    ``queued_at_ms`` is when the job was created, which is when the owner
    confirmed it; a restart that waited longer than the queue limit is refused.
    """
    if payload.get("confirm") != NODE_REBOOT_CONFIRM:
        raise ValueError("Review and confirm restarting this machine first.")
    node = _node(operations, payload)
    name = str(node.get("name") or node["id"])
    if queued_at_ms is not None and now() * 1000 - queued_at_ms > RESTART_QUEUE_LIMIT_SECONDS * 1000:
        raise RuntimeError(
            "{name} was not restarted: the restart waited more than {minutes} "
            "minutes behind other work after you confirmed it. If it is still "
            "wanted, confirm the restart again.".format(
                name=name, minutes=RESTART_QUEUE_LIMIT_SECONDS // 60,
            )
        )
    transport = _transport(operations, node, transport_factory)
    before = _boot_id(transport)
    if not before:
        raise RuntimeError(
            "{} cannot be reached right now, so Vaelor cannot restart it. "
            "Restart it by hand, or check its connection.".format(name)
        )
    confirmed = True
    try:
        transport.run(["systemctl", "reboot"], sudo=True)
    except SshTransportError as error:
        # A machine that says WHY it refused did not restart. One that gave
        # no reason - the transport's own generic sentence - closed the
        # connection, which is what a restarting machine does (review B1:
        # that was recorded as a failure, and a failure offered Retry).
        if str(error) != COMMAND_ENDED_WITHOUT_REASON:
            raise RuntimeError(
                "{} did not accept the restart: {}".format(name, str(error)[:200])
            ) from error
        confirmed = False
    except _CHANNEL_DROPS:
        confirmed = False
    returned = _wait_for_return(
        lambda seconds: _transport(operations, node, transport_factory, seconds),
        before, progress or (lambda _percent, _message: None), name, sleep, monotonic,
    )
    if returned:
        try:
            reading = pool.pool_status(
                read_node_facts(transport), (node.get("inventory") or {}).get("gpu")
            )
        except (RuntimeError,) + _FAILURES:
            reading = None
        if reading is not None:
            record_reading(operations, node["id"], reading, now)
        message = "{} restarted and is answering again.".format(name)
    elif confirmed:
        message = (
            "{name} is restarting. Models and apps on it stop until it is back. "
            "It has not answered again within {minutes} minutes; use Recheck "
            "on its card once it does."
        ).format(name=name, minutes=RESTART_WAIT_SECONDS // 60)
    else:
        message = (
            "A restart was requested on {name}. The connection closed before "
            "the machine confirmed it, which is what a restarting machine "
            "does. It has not answered again within {minutes} minutes; use "
            "Recheck on its card once it does."
        ).format(name=name, minutes=RESTART_WAIT_SECONDS // 60)
    return {
        "node_id": node["id"], "name": name, "restarting": True,
        "confirmed": confirmed, "returned": returned, "message": message,
    }
