"""The LLM Server's "loading" door on the way back from cluster serving (W4-D8, B1).

Leaving cluster serving moves AI Chat back to its previous model. When the
GPU failure-watch will relaunch that model AND front it with the LLM Server
gate, port 11434 answers "loading" (ACC-193's door) until it does, instead of
refusing for the length of the relaunch. The first cut held the door for any
managed-local lease and left it to the watch to replace - and the watch never
fronts a stock-GGUF (``model-chat`` compose) model, nor a generic lease whose
``.gguf`` it cannot find, so 11434 said "loading" for ever and neither a Disable
nor a key change reached it (review B1, LESSONS 1 and 6).

**One answer to "will something relaunch behind this door?"** It is the
watch's own: `ExecutorGpuDeployMixin.gpu_chat_relaunch` names the arm the
watch's pass dispatches on, and only :data:`FRONTED_RELAUNCHES` - the two arms
that end with the gate converged in front of the relaunched model - hold the
door. The switch asks that same method (the executor wires it); with no answer
(a switch built without it) the door is closed, as before W4-D8.

**The live gate is the authority; the record only times it (final review
B-1).** A stop that failed used to clear the record anyway, and a record that
was corrupt, missing or never written read as "not held", so 11434 said
"loading" for ever. Now every ruling reads the gate's own status first: a door
up in Mode A for an arm that will not front it is closed whatever the record
says; for one that will, a missing or unreadable record is written now so the
bound starts. The record is cleared only after the gate stopped, a record that
cannot be written never stops a leave, and a hold begun in the future (a clock
stepped back) is over.

**The door has a stated lifetime.** It is recorded when held (:class:`LoadingDoor`)
and cleared the moment the gate is converged in front of a model (every
successful apply). A watch pass that finds it older than
:data:`LOADING_DOOR_SECONDS` - a relaunch that crash-loops or never answers -
closes it and logs why: "loading" is never said of a model that is not coming.
The watch also closes it at once when its arm is one that does not front the
gate, and the LLM Server's own apply closes it when it has nothing to put in
front (compose-backed, no target), so Disable and a key change always act on it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from .gpu_cluster_lan_doors import STOPPING_THE_BALANCER
from .gpu_cluster_mode_state import MODE_DIRECTORY
from .gpu_rocm_supervisor import GPU_HEALTH_DEADLINE_SECONDS
from .llm_server_proxy import UPSTREAM_UNLOADED_NOTICE
from .llm_server_state import harden_for_jobs_group

LOGGER = logging.getLogger(__name__)

#: The watch's relaunch arms (`ExecutorGpuDeployMixin.gpu_chat_relaunch`).
RELAUNCH_FP4 = "fp4"
RELAUNCH_GENERIC = "generic"
RELAUNCH_COMPOSE = "compose"
RELAUNCH_UNRESOLVED = "unresolved"
RELAUNCH_NONE = "none"

#: The arms that end with the LLM Server gate converged in front of the model.
FRONTED_RELAUNCHES = frozenset({RELAUNCH_FP4, RELAUNCH_GENERIC})

#: How long the door may say "loading": one full health-gated relaunch plus two
#: watch passes (30 s each). Past it the relaunch is not coming in time.
LOADING_DOOR_SECONDS = int(GPU_HEALTH_DEADLINE_SECONDS) + 60

#: Logged when the hold record cannot be written; the door stands without it.
RECORD_UNWRITTEN = "The LLM Server loading-door record could not be written: %s"

#: Where the hold is recorded, beside the mode file the switch owns.
LOADING_DOOR_FILE = str(Path(MODE_DIRECTORY) / "loading-door.json")


class LoadingDoor:
    """When the loading door was put up on the way back, or 0 when it is not held."""

    def __init__(self, path: str = LOADING_DOOR_FILE):
        self._path = Path(path)

    @classmethod
    def beside(cls, mode_store: Any) -> "LoadingDoor":
        """The record beside a mode store's own file - the switch's and the
        watch's one place, wherever that store lives (a test's temp dir too)."""
        path = getattr(mode_store, "_path", None)
        return cls(str(Path(path).with_name("loading-door.json"))) if path else cls()

    def since(self) -> int:
        """When the hold began; 0 when there is no readable record."""
        try:
            return int(json.loads(self._path.read_text(encoding="utf-8")).get("since") or 0)
        except (OSError, ValueError, TypeError, AttributeError):
            return 0

    def hold(self, now: float) -> bool:
        """Record the hold. Never raises: a record that cannot be written leaves
        the door held without one, and the live-gate rule still bounds it
        (final review follow-up 2: a full disk aborted the whole leave).

        Group-hardened for the jobs group like the mode file beside it
        (`ClusterModeStore.write`), the temporary file and the record alike
        (review A7); best effort, as that hardening always is.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_name(self._path.name + ".tmp")
            temporary.write_text(json.dumps({"since": int(now)}) + "\n", encoding="utf-8")
            harden_for_jobs_group(temporary)
            os.replace(temporary, self._path)
            harden_for_jobs_group(self._path)
            return True
        except OSError as error:
            LOGGER.warning(RECORD_UNWRITTEN, error)
            return False

    def clear(self) -> None:
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            LOGGER.warning("The LLM Server loading-door record could not be cleared: %s", error)

    def expired(self, now: float) -> bool:
        """Past the bound - or begun in the future, which only a clock stepped
        back produces; the hold's age is then unknown, so it is over
        (final review follow-up 3)."""
        since = self.since()
        return bool(since) and (now - since > LOADING_DOOR_SECONDS or since > now)


def _hold(door: LoadingDoor, now: float) -> None:
    """Record a hold, never raising: the door may stand without a record (the
    live-gate rule bounds it), but a leave must never stop half-way because a
    record could not be written (final review follow-up 2)."""
    try:
        door.hold(now)
    except Exception as error:  # noqa: BLE001 - see above
        LOGGER.warning(RECORD_UNWRITTEN, error)


def gate_on_door(proxy: Any) -> Optional[bool]:
    """Whether the LIVE gate is the door that fronts nothing (the loading or
    unloaded-notice door), read off its own status; ``None`` when it cannot
    be asked. The gate, not the record, is the authority (final review B-1)."""
    try:
        status = proxy.status() or {}
    except Exception:  # noqa: BLE001 - unreadable is "not known", said by None
        return None
    return bool(status.get("running")) and status.get("upstream") == UPSTREAM_UNLOADED_NOTICE


def relaunch_is_fronted(asker: Optional[Callable[[], str]]) -> bool:
    """Whether the watch will front the restored model: its own answer, or no."""
    if asker is None:
        return False
    try:
        return asker() in FRONTED_RELAUNCHES
    except Exception as error:  # noqa: BLE001 - not known is not "it will"
        LOGGER.warning("Could not ask whether AI Chat's model is relaunched: %s", error)
        return False


def close_door(door: LoadingDoor, proxy: Any, why: str, now: float) -> bool:
    """Close the door - the live gate on it, or a hold recorded - and say why.

    The record is cleared only once the gate has stopped. A stop that failed
    keeps it (writing one when there was none), so the next watch pass sees
    the door and tries again; a cleared record over a running door was how
    11434 said "loading" for ever (final review B-1, LESSONS 1).
    """
    if not door.since() and not gate_on_door(proxy):
        return False
    LOGGER.warning("The LLM Server's loading door is closed: %s", why)
    try:
        proxy.stop()
    except Exception as error:  # noqa: BLE001 - kept for the next pass to retry
        LOGGER.warning("The LLM Server gate could not be stopped; the next pass "
                       "tries again: %s", error)
        if not door.since():
            _hold(door, now)
        return False
    door.clear()
    return True


def watch_door(door: LoadingDoor, proxy: Any, relaunch: str, now: float) -> None:
    """One Mode A watch pass's ruling on the door, before it relaunches anything.

    Read off the live gate first: a door up for an arm that will never front
    the gate is closed whatever the record says (or whether there is one);
    for an arm that will, the record times it, and a missing or unreadable
    record is written now so the bound starts and cannot stand for ever.
    """
    on_door = gate_on_door(proxy)
    if on_door is False:
        door.clear()  # nothing to hold: the gate is not on the door
        return
    if on_door is None and not door.since():
        return  # neither the gate nor a record says a door is up
    if relaunch not in FRONTED_RELAUNCHES:
        close_door(door, proxy, "nothing this machine relaunches is fronted by the "
                   "LLM Server ({})".format(relaunch), now)
    elif not door.since():
        _hold(door, now)
    elif door.expired(now):
        close_door(door, proxy, "AI Chat's model did not serve within {} seconds".format(
            LOADING_DOOR_SECONDS), now)


def leave_door(switch: Any, deployment: str, holds: bool, restore: Callable[[], tuple]) -> tuple:
    """The gate on `leave`, around the lease restore: ``(restored, reason)``.

    ``holds`` is the switch's pre-check (no unassign exception, a managed-local
    lease to go back to). The door goes up BEFORE the restore so 11434 never
    refuses; after it, the watch is asked whether it will front the restored
    model, and anything but yes closes the door again under the same lock -
    or, if the gate will not stop, leaves a record for the watch to retry.
    """
    if holds:
        switch._hold_door_while_loading()
        switch._quietly(lambda: switch.balancer.stop(deployment), STOPPING_THE_BALANCER)
    else:
        switch._stop_lan_doors(deployment)
    restored, reason = restore()
    door = switch.loading_door
    if holds and restored and relaunch_is_fronted(getattr(switch, "relaunch_fronted", None)):
        _hold(door, switch._now())
    elif holds:
        if not door.since():
            _hold(door, switch._now())  # so a stop that fails is retried
        close_door(door, switch.llm_proxy, "AI Chat's model is not one the "
                   "LLM Server fronts once it is back", switch._now())
    return restored, reason
