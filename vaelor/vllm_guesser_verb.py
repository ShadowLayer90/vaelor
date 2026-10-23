"""The root bridge's ``guesser_build`` verb: the controller's 4-bit guesser, checked or started (N94).

**Why it exists** (VD-217, N94). The refresh after an update fetches the
drafter while the model serves, and the post-install build step ran before
it had anything to build from, so the Load after the stop served the
full-size guesser and the 4-bit copy waited for a second Load. Asked before
the stop, this starts the SAME background build the bridge starts at a Load
(`vllm_guesser_build.background_build_argv`: a fixed argv, one at a time,
bounded by systemd) and answers how it stands; the refresh polls with
``start`` false (`gpu_pool_refresh.prebuild_guesser`).

**Who can call it** (LESSONS 18). The bridge socket is ``0660 root:vaelor``:
every group-``vaelor`` service, the research broker that reads untrusted web
content included. So (R110):

* only a pinned image reference from `vllm_guesser_build.GUESSERS` and a bool
  cross - no path, no argv;
* one call at a time (:data:`_VERB_LOCK`, never waited on): a call that finds
  another checking answers "building" instead of hashing again;
* a "ready" verdict is kept by the built folder's own ``lstat`` signature, so
  a ready folder is hashed once, not 1 GB per call; the folder is root's alone
  (``0755``/``0644``), so only root can change what the signature covers;
* a start that ended without a build is not repeated for
  :data:`START_COOLDOWN_SECONDS` unless the drafter it builds from changed -
  no automatic restart loop;
* the background unit is read with ``systemctl show``, which answers for
  every state, so a start that is still ``activating`` reads "building".

Nothing is built in the call itself, and no bridge lock is held across one.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from . import vllm_guesser_build as guesser_build
from .vllm_overlays import OVERLAY_ROOT

#: What :func:`bridge_build_state` answers, by the ``state`` key.
BUILD_STATES = ("ready", "building", "started", "refused")

#: How long a start that ended without a build holds off the next one when
#: the drafter has not changed. Longer than one build's bound
#: (`vllm_guesser_build.BUILD_SECONDS`).
START_COOLDOWN_SECONDS = 600

#: Said when a start is held off.
START_HELD = (
    "; a build of it ended without a result {} s ago and its input has not "
    "changed since, so it is not started again yet"
)

#: The background unit's states that mean a build is on its way or running.
_BUILDING = frozenset({"active", "activating", "reloading"})

#: How long a call that may START a build waits for another call's check to
#: end (VD-133 1.5.5 #10, adversarial M2). A read that finds the lock held
#: answers "building" at once; a start that did so started nothing, and the
#: wait polling after it read "refused" with no build ever begun. A check is
#: one hash of about 1 GB, about a second on the Z2.
START_LOCK_WAIT_SECONDS = 15.0

_VERB_LOCK = threading.Lock()
_READY: Dict[str, Tuple[Any, ...]] = {}
_STARTED: Dict[str, Tuple[float, Optional[Tuple[Any, ...]]]] = {}
#: A refused verdict, kept by the ``lstat`` signatures of the build's own
#: folder AND the key folder its record sits in (adversarial M2): asked again
#: with neither changed - no build ran, no record was written - the folder is
#: not hashed again as root at every read.
_REFUSED: Dict[str, Tuple[Any, Dict[str, Any]]] = {}


def folder_signature(folder: str) -> Optional[Tuple[Any, ...]]:
    """The ``lstat`` identity of ``folder`` and every entry in it, or ``None``."""
    try:
        own = os.lstat(folder)
        entries = []
        with os.scandir(folder) as listing:
            for entry in listing:
                found = entry.stat(follow_symlinks=False)
                entries.append((entry.name, found.st_ino, found.st_size, found.st_mtime_ns,
                                found.st_mode, found.st_uid))
    except OSError:
        return None
    return (own.st_ino, own.st_mtime_ns, own.st_mode, own.st_uid, tuple(sorted(entries)))


def _unit_state(run: Callable[[List[str]], Any]) -> str:
    try:
        return str(run(["systemctl", "show", "-p", "ActiveState", "--value",
                        guesser_build.BACKGROUND_UNIT]) or "").strip()
    except Exception:  # noqa: BLE001 - an unread unit reads as not building
        return ""


def bridge_build_state(
    image_ref: Any, start: Any, *, run: Callable[[List[str]], Any],
    root: str = OVERLAY_ROOT, owners: FrozenSet[int] = frozenset({0}),
    hub: str = guesser_build.MODEL_HUB, clock: Callable[[], float] = time.monotonic,
) -> Dict[str, str]:
    """``{"state": one of BUILD_STATES, "reason": ...}`` for ``image_ref`` (module docstring)."""
    if not isinstance(image_ref, str) or not isinstance(start, bool):
        raise ValueError("The guesser build request does not carry exactly its own values.")
    row = guesser_build.guesser_for_image(image_ref)
    if row is None:
        raise ValueError("No 4-bit guesser is built for that image.")
    acquired = (_VERB_LOCK.acquire(timeout=START_LOCK_WAIT_SECONDS) if start
                else _VERB_LOCK.acquire(blocking=False))
    if not acquired:
        return {"state": "building", "reason": ""}
    try:
        return _answer(row, start, run, root, owners, hub, clock)
    finally:
        _VERB_LOCK.release()


def _answer(row: Any, start: bool, run: Callable[[List[str]], Any], root: str,
            owners: FrozenSet[int], hub: str, clock: Callable[[], float]) -> Dict[str, str]:
    if _unit_state(run) in _BUILDING:
        return {"state": "building", "reason": ""}
    folder = guesser_build.installed_dir(row, root)
    signature = folder_signature(folder)
    if signature is not None and _READY.get(row.key) == signature:
        return {"state": "ready", "reason": ""}
    seen = (signature, folder_signature(os.path.dirname(folder)))
    held = _REFUSED.get(row.key)
    if held is not None and held[0] == seen:
        verdict = held[1]
    else:
        verdict = guesser_build.run_host(guesser_build.host_plan(row, "check", root=root,
                                                                 owners=owners))
    if verdict.get("ready"):
        _READY[row.key] = folder_signature(folder)
        _STARTED.pop(row.key, None)
        _REFUSED.pop(row.key, None)
        return {"state": "ready", "reason": ""}
    _READY.pop(row.key, None)
    _REFUSED[row.key] = (seen, dict(verdict))
    reason = guesser_build.reason_of(verdict)
    if not (start and guesser_build.worth_starting(verdict)):
        return {"state": "refused", "reason": reason}
    inputs = folder_signature(guesser_build.source_dir(row, hub))
    last = _STARTED.get(row.key)
    if last is not None and last[1] == inputs and clock() - last[0] < START_COOLDOWN_SECONDS:
        return {"state": "refused", "reason": reason + START_HELD.format(int(clock() - last[0]))}
    try:
        run(guesser_build.background_build_argv())
    except Exception:  # noqa: BLE001 - a start that raced another is that one building
        if _unit_state(run) in _BUILDING:
            return {"state": "building", "reason": ""}
        raise
    _STARTED[row.key] = (clock(), inputs)
    return {"state": "started", "reason": reason}
