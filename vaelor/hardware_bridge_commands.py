"""The root bridge's controller-command verbs: ``run_argv`` and the unit verbs.

Moved out of `hardware_bridge` (VD-143) when the structured unit verbs took
that module past the 1,000-line ceiling - the same move `hardware_bridge_client`
made for the client (VD-129). Everything here is the controller's half of the
GPU cluster, the path on which `GpuPoolRuntime` drives THIS machine as a
serving node; `_HardwareRuntime` inherits it, so the handler and the tests
reach it exactly as before.

Two kinds of verb, one per kind of act:

* ``run_argv`` - a reviewed argv, spawned as root, under
  `bridge_argv_policy`. After VD-143 no shape in that policy writes a file or
  starts a unit: what is left reads, pulls the one image, stops, resets,
  reloads and removes.
* ``managed_unit_install``, ``pull_progress_read`` and ``model_cache_remove`` -
  the acts that DO write, or that read inside the group-writable model store,
  each taking typed values and doing its own no-follow file work in
  `bridge_managed_units`. A client sends a unit's kind and values, never its
  text, and the root side renders it from the runtime's own template.
"""

from __future__ import annotations

import contextlib
from typing import Any

from .bridge_argv_policy import STDIN_REFUSAL, check_bridge_argv, mutates_units
from .hardware_bridge_client import (
    RUN_ARGV_MAX_OUTPUT_BYTES,
    RUN_ARGV_MAX_TIMEOUT_SECONDS,
    RUN_ARGV_TIMEOUT_SECONDS,
)

#: The structured verbs, dispatched by :meth:`ControllerCommandsMixin.controller_verb`.
CONTROLLER_UNIT_VERBS = frozenset({
    "managed_unit_install", "pull_progress_read", "model_cache_remove",
    # A split's Ray token, slice and firewall on this controller (ACC-163,
    # ACC-187). The fence verb is new with the slice and `shared`: a bridge
    # from before it does not know it, so it is never handed a shared fence
    # it would ignore (review 1, SC2); the old prepare verb is gone.
    "ray_plane_fence", "ray_plane_clear", "ray_container_cgroup",
})

#: How long a Ray-plane verb waits for the bridge's lock (held by, say, a unit
#: install) before it refuses as busy: with the verb's own worst case
#: (`bridge_managed_units.RAY_PLANE_STEPS` x `RAY_PLANE_STEP_SECONDS`) this
#: stays inside the client's wait, so the bridge never fences after the
#: client has given up (ACC-187 review 3).
RAY_PLANE_LOCK_WAIT_SECONDS = 60
BRIDGE_BUSY = (
    "The hardware bridge is busy with another change on this controller; "
    "try again in a moment."
)


@contextlib.contextmanager
def _held_briefly(lock: Any):
    if not lock.acquire(timeout=RAY_PLANE_LOCK_WAIT_SECONDS):
        raise RuntimeError(BRIDGE_BUSY)
    try:
        yield
    finally:
        lock.release()


#: Each Ray-plane verb's payload: exactly these keys, each of this type.
_TYPED_PAYLOADS = {
    "ray_plane_fence": {"name": str, "token": str, "link": str, "address": str,
                        "peers": list, "shared": bool},
    "ray_plane_clear": {"name": str},
    "ray_container_cgroup": {"name": str, "role": str},
}


def _typed(action: str, payload: dict) -> dict:
    """``payload`` when it carries exactly its verb's keys with their types, else a refusal."""
    shape = _TYPED_PAYLOADS[action]
    if set(payload) != set(shape) or any(
        type(payload[key]) is not kind for key, kind in shape.items()
    ):
        raise ValueError("The {} request does not carry exactly its own values.".format(action))
    return payload


class ControllerCommandsMixin:
    """``run_argv`` and the structured unit/store verbs, for `_HardwareRuntime`."""

    def run_argv(
        self, argv: Any, stdin_text: Any = "", timeout: Any = None
    ) -> dict[str, Any]:
        """Run one reviewed argv as root and return its stdout, or raise.

        The controller's own GPU is a serving node, and the runtime that drives
        a node writes units, runs ``docker`` and reads files. On a worker that
        is `SshTransport.run`; here it is this, and the two present the same
        "stdout on success, the stderr tail on failure" contract, so
        `GpuPoolRuntime` never learns which one it holds.

        **This is the boundary that matters** (LESSONS #178). The client's check
        is advisory; the argv is re-validated HERE by
        :func:`vaelor.bridge_argv_policy.check_bridge_argv`, whose docstring
        holds the shapes and the reason they are narrower than the SSH
        allowlist. There is no ``sudo`` step: this service asserted
        ``geteuid()==0`` before serving, so the child is root already - the same
        reason :meth:`gpu_start` needs none. No shell either: the argv is a LIST
        to :func:`subprocess.run`, so nothing is word-split or glob-expanded.

        **The lock is held only across a unit mutation**
        (:func:`vaelor.bridge_argv_policy.mutates_units`). Two concurrent
        deploys interleaving a unit write with a ``daemon-reload`` is the race
        VD-001's lock exists to stop, and those commands are milliseconds long.
        The rest are not: a cold ``docker pull`` holds the process for up to an
        hour, and taking the lock across it froze every NPU, GPU and proxy verb
        for the length of the pull. Ordering does not matter for the rest - a
        read is a read, and an image pull and a weights fetch are idempotent
        against each other.

        **Nothing here writes a file (VD-143).** The shapes left are reads,
        ``docker`` image and container verbs, the removal of a managed unit and
        the systemd verbs that stop, reset or reload. A unit is installed by
        :meth:`managed_unit_install`, which renders it; the model store is
        touched only by the two store verbs below, which follow no link.
        """
        import subprocess

        if not isinstance(argv, list) or not argv:
            raise ValueError(
                "A controller command must be a non-empty list of arguments."
            )
        args = [str(item) for item in argv]
        check_bridge_argv(args)
        if str(stdin_text or ""):
            raise ValueError(STDIN_REFUSAL)
        try:
            seconds = min(
                RUN_ARGV_MAX_TIMEOUT_SECONDS,
                max(1, int(timeout if timeout is not None else
                           RUN_ARGV_TIMEOUT_SECONDS)),
            )
        except (TypeError, ValueError):
            seconds = RUN_ARGV_TIMEOUT_SECONDS
        ordered = self._lock if mutates_units(args) else contextlib.nullcontext()
        with ordered:
            try:
                completed = subprocess.run(
                    args, stdin=subprocess.DEVNULL, capture_output=True,
                    text=True, timeout=seconds, check=False,
                )
            except subprocess.TimeoutExpired as error:
                # Not one of the exception types the dispatch already answers
                # with, so it is translated here rather than killing the
                # handler thread and leaving the caller waiting on a socket
                # that will never answer.
                raise RuntimeError(
                    "The controller command {!r} did not finish within {} "
                    "seconds.".format(args[0], seconds)
                ) from error
        if completed.returncode != 0:
            raise RuntimeError(
                str(completed.stderr or "").strip()[:400]
                or "The controller command failed."
            )
        return _bounded_stdout(str(completed.stdout or "").strip())

    @staticmethod
    def _managed_units():
        """The root side of the controller's unit and store verbs, built per call:
        it holds no state, and its imports stay off a bridge that never serves
        the GPU tier."""
        from .bridge_managed_units import ManagedUnits

        return ManagedUnits()

    def managed_unit_install(self, kind: Any, values: Any) -> dict[str, Any]:
        """Render, write and start one managed GPU unit, as root (VD-143).

        Held under ``self._lock`` for the whole install, as a ``tee`` and its
        ``daemon-reload`` were: two concurrent deploys must not interleave a
        unit write with a reload. The kind and the values are carried through
        unaltered; every rule lives in `bridge_managed_units` and the runtime
        template it renders with, at this root boundary (LESSONS #178).
        """
        with self._lock:
            return self._managed_units().install(kind, values)

    def pull_progress_read(self, repo: Any) -> dict[str, Any]:
        """A pull's progress record, read as root without following a link."""
        return {"text": self._managed_units().read_progress(repo)}

    def model_cache_remove(self, repo: Any) -> dict[str, Any]:
        """Remove one repo's weights from the model store, following no link.

        No lock: a removal races nothing systemd reads, and on a large model it
        takes long enough to stall every other verb if it held one.
        """
        return {"path": self._managed_units().remove_cache(repo)}

    def controller_verb(self, action: str, payload: Any) -> dict[str, Any]:
        """Route one of :data:`CONTROLLER_UNIT_VERBS` from the socket handler."""
        if not isinstance(payload, dict):
            raise ValueError("Invalid controller unit request.")
        if action == "managed_unit_install":
            return self.managed_unit_install(payload.get("kind"), payload.get("values"))
        if action == "pull_progress_read":
            return self.pull_progress_read(payload.get("repo"))
        if action == "model_cache_remove":
            return self.model_cache_remove(payload.get("repo"))
        if action == "ray_plane_fence":
            values = _typed(action, payload)
            with _held_briefly(self._lock):
                return self._managed_units().prepare_ray_plane(
                    values["name"], values["token"], values["link"],
                    values["address"], values["peers"], values["shared"],
                )
        if action == "ray_plane_clear":
            values = _typed(action, payload)
            with _held_briefly(self._lock):
                return self._managed_units().clear_ray_plane(values["name"])
        if action == "ray_container_cgroup":
            values = _typed(action, payload)
            return self._managed_units().container_cgroup(values["name"], values["role"])
        raise ValueError("That is not a controller unit verb.")


def _bounded_stdout(output: str) -> dict[str, Any]:
    """``{"stdout", "truncated"}`` for a command's output, cut on BYTES.

    The reply has to fit this socket's frame, and the cut is made on the encoded
    bytes because that is what the frame counts - slicing characters would let a
    UTF-8 answer overflow it. ``truncated`` is reported so the caller can refuse
    the answer: `BridgeTransport.run` does, because a cut group file or routing
    table parses cleanly into facts that are simply wrong.
    """
    encoded = output.encode("utf-8")
    if len(encoded) <= RUN_ARGV_MAX_OUTPUT_BYTES:
        return {"stdout": output, "truncated": False}
    return {
        "stdout": encoded[:RUN_ARGV_MAX_OUTPUT_BYTES].decode(
            "utf-8", errors="replace"
        ),
        "truncated": True,
    }
