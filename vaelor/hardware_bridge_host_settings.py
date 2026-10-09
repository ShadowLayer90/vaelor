"""The root bridge's host-setting verbs: the GPU memory pool (VD-161).

Two dedicated verbs, NOT a ``run_argv`` widening (LESSONS 18): who can open
this socket is the control plane and the workload executor
(`vaelor.bridge_peers`), neither of which holds a root primitive, so what they
may ask for here is one thing, spelled narrowly.

* ``gpu_memory_pool_set`` takes ONE value, ``size_gib``. It must be a JSON
  whole number - ``true``, ``"44"`` and ``44.0`` are refused - and it must sit
  inside the bounds THIS side works out from its own read of the machine
  (`gpu_memory_pool.require_size_gib`). The path written, the line written and
  the boot-image command are constants of `gpu_memory_pool_apply`; nothing a
  client sends reaches a path, a file body or an argument other than as the
  digits of that checked number.
* ``gpu_memory_pool_revert`` takes nothing and removes Vaelor's own file.

Neither restarts the machine. The new size counts from the next restart, and
a restart is the existing power verb, confirmed separately by the owner.

Both serialise on their own family lock: a rebuild takes a minute or more and
must not hold the lock the model and proxy verbs share (ACC-067).
"""

from __future__ import annotations

from typing import Any, Dict

#: The verbs :func:`dispatch_host_setting` answers.
HOST_SETTING_ACTIONS = frozenset({"gpu_memory_pool_set", "gpu_memory_pool_revert"})

_LOCK_FAMILY = "gpu-memory-pool"


class HostSettingVerbs:
    """Mixed into the bridge runtime, which supplies ``self._family_lock``."""

    def gpu_memory_pool_set(self, size_gib: Any) -> Dict[str, Any]:
        """Set this machine's GPU memory pool from the next restart."""
        from . import gpu_memory_pool as pool
        from . import gpu_memory_pool_apply as apply

        with self._family_lock(_LOCK_FAMILY):  # type: ignore[attr-defined]
            # Checked HERE, before anything is started: the transient unit
            # checks again, but a refusal should cost no unit at all.
            pool.require_size_gib(size_gib, apply.local_status())
            return apply.run_apply_unit(apply.ACTION_SET, size_gib)

    def gpu_memory_pool_revert(self) -> Dict[str, Any]:
        """Remove Vaelor's GPU memory pool file; the kernel's size returns on restart."""
        from . import gpu_memory_pool as pool
        from . import gpu_memory_pool_apply as apply

        with self._family_lock(_LOCK_FAMILY):  # type: ignore[attr-defined]
            pool.require_revert(apply.local_status())
            return apply.run_apply_unit(apply.ACTION_REVERT)


def dispatch_host_setting(runtime: Any, action: str, payload: Any) -> Dict[str, Any]:
    """Answer one host-setting verb; the rule lives at the root boundary (#178)."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid machine setting request.")
    if action == "gpu_memory_pool_set":
        if set(payload) != {"size_gib"}:
            raise ValueError("A GPU memory pool request carries one value: size_gib.")
        return runtime.gpu_memory_pool_set(payload["size_gib"])
    if action == "gpu_memory_pool_revert":
        if payload:
            raise ValueError("A GPU memory pool revert carries no values.")
        return runtime.gpu_memory_pool_revert()
    raise ValueError("Unsupported machine setting action.")
