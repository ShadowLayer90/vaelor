"""The persisted GPU serving mode record: which mode, and what Mode A is owed.

Housed out of `gpu_cluster_mode` (VD-129) when the replica balancer became the
switch's fifth collaborator and that module stood at the 1,000-line ceiling
`CLAUDE.md` sets; the switch re-exports every name here, so its readers import
what they always did. What lives here is the record and nothing that acts on
it: the dataclass, the fail-safe read (`_coerce`), and the store that writes it
atomically and group-hardened.

**The mode file is written by the executor and read by the control plane.**
LESSONS pattern 13: the suite runs as one user against a temporary directory,
and the appliance runs five. ``/var/lib/vaelor`` gives write to the control
plane's account alone (`vaelor.state_root_layout`) and
the executor runs as ``vaelor-workloads``, so a record written at the state root
would fail with ``EACCES`` on every appliance while passing every test. So the
record lives in its OWN directory, created by the installer as
``2770 vaelor-workloads:vaelor-jobs`` - owned by the writer, setgid to the group
the control plane is a member of - and `tests/test_gpu_cluster_mode.py` asserts
that against `deploy/install-vaelor.sh` and the two units, which is where the
fact actually lives.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from .gpu_serving_target import (
    MODE_CLUSTER, MODE_SINGLE, UNLOAD_CAUSE_IDLE, UNLOAD_CAUSE_MANUAL,
)
from .llm_server_state import harden_for_jobs_group
from .runtime_paths import state_path


#: The mode record, in its own executor-owned directory. See the module
#: docstring: the state root itself is not writable by the account that writes
#: this file, and the file has to be readable by the account that reads it.
MODE_DIRECTORY = state_path("gpu-cluster-mode")
MODE_FILE = str(Path(MODE_DIRECTORY) / "state.json")



@dataclass(frozen=True)
class ClusterModeState:
    """The persisted mode record: which mode, and what Mode A is owed back.

    ``previous_ai_chat_credential_id`` is the whole restore contract - what
    ``ai-chat`` pointed at before the cluster took over. ``cluster_credential_id``
    is empty between `enter` and `repoint`, which is exactly the window in which
    the cluster is not yet serving anything.

    ``llm_server_was_enabled`` is a recorded FACT, not a contract: whether the
    LLM Server was on when the switch was taken, kept (and still read back from
    older files) for the operator reading the record. Nothing decides on it.
    It used to decide the LAN gate at `repoint`, on the healthy pass and on
    `leave`, and a flag switched on after `enter` then never reached the gate
    (2026-09-28); the gate now follows the LLM Server record as it is on every
    pass, and `tests/test_gpu_cluster_mode_llm_server.py` refuses a production
    read of this field anywhere but here.

    ``deploy_in_flight`` is what makes that window safe rather than fatal: it is
    true for exactly that window, and the reconcile reads it as a claim on the
    switch that the running deploy job then witnesses - "wait" with the job,
    "the deploy died, leave" without it. ``entered_at`` is a recorded fact for
    the operator reading the file (when the switch was taken); nothing decides
    on it, so every field here is an identity and none is a deadline.

    ``unload_cause`` is why the deployment is unloaded (VD-210), written by the
    unload under the serving lock and cleared by the Load and by every healthy
    pass: ``""`` (not unloaded, or a record from before VD-210), or one of
    `gpu_serving_target`'s two causes. The ``ai-chat`` lease used to be the
    only witness (VD-136: on the cluster means idle, parked means manual), and
    VD-210 lets the owner move AI Chat off the cluster for good, so the lease
    can no longer tell the LLM Server's door which unload it is looking at.
    """

    mode: str = MODE_SINGLE
    deployment_name: str = ""
    cluster_credential_id: str = ""
    cluster_port: int = 0
    previous_ai_chat_credential_id: str = ""
    llm_server_was_enabled: bool = False
    entered_at: int = 0
    deploy_in_flight: bool = False
    unload_cause: str = ""

    def as_record(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "deployment_name": self.deployment_name,
            "cluster_credential_id": self.cluster_credential_id,
            "cluster_port": int(self.cluster_port),
            "previous_ai_chat_credential_id": self.previous_ai_chat_credential_id,
            "llm_server_was_enabled": bool(self.llm_server_was_enabled),
            "entered_at": int(self.entered_at),
            "deploy_in_flight": bool(self.deploy_in_flight),
            "unload_cause": self.unload_cause,
        }


def _coerce(raw: Any) -> ClusterModeState:
    """A stored record read back, failing safe to Mode A.

    An unreadable, truncated or hand-edited record reads as "not clustering",
    which is the direction that cannot strand AI Chat: the failure-watch then
    supervises the managed-local model as it always did, and a cluster that IS
    running is found by the reconcile through its deployment record instead.
    """
    if not isinstance(raw, Mapping):
        return ClusterModeState()
    mode = str(raw.get("mode") or MODE_SINGLE)
    if mode != MODE_CLUSTER:
        return ClusterModeState()
    try:
        port = int(raw.get("cluster_port") or 0)
    except (TypeError, ValueError):
        port = 0
    try:
        entered = int(raw.get("entered_at") or 0)
    except (TypeError, ValueError):
        entered = 0
    cause = str(raw.get("unload_cause") or "")
    return ClusterModeState(
        mode=MODE_CLUSTER,
        deployment_name=str(raw.get("deployment_name") or ""),
        cluster_credential_id=str(raw.get("cluster_credential_id") or ""),
        cluster_port=port,
        previous_ai_chat_credential_id=str(
            raw.get("previous_ai_chat_credential_id") or ""
        ),
        llm_server_was_enabled=bool(raw.get("llm_server_was_enabled")),
        entered_at=entered,
        deploy_in_flight=bool(raw.get("deploy_in_flight")),
        # Anything but the two causes reads as "not recorded", which the
        # reconcile's unloaded pass then derives from the lease as VD-136 did.
        unload_cause=cause if cause in (UNLOAD_CAUSE_IDLE, UNLOAD_CAUSE_MANUAL) else "",
    )


class ClusterModeStore:
    """Read and write the mode record. Injectable ``path`` for tests.

    Reads fail safe to Mode A; writes are atomic (temp file then ``os.replace``)
    and group-hardened, exactly as :class:`vaelor.llm_server_state.LlmServerStore`
    is, because this record crosses the same two accounts in the other direction.
    """

    def __init__(self, path: str = MODE_FILE):
        self._path = Path(path)

    def read(self) -> ClusterModeState:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ClusterModeState()
        return _coerce(raw)

    def write(self, state: ClusterModeState) -> ClusterModeState:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state.as_record(), separators=(",", ":"))
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(payload + "\n", encoding="utf-8")
        harden_for_jobs_group(tmp)
        os.replace(tmp, self._path)
        harden_for_jobs_group(self._path)
        return state
