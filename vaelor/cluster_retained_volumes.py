"""Data volumes a cluster app removal leaves on a worker, and how the owner deletes them.

W4d-D20: removing a stateful cluster app keeps its node-local volume (Swarm's
`service rm` never deletes one), while the console said the data would be
"irreversibly" lost - and then showed nothing, so the volume could be neither
reused nor deleted. A removal now records each volume it leaves, with the
machine it is on; the console lists them, and the owner can delete one.
Deploying the same app under the same name on that machine reattaches the data,
because a service's volume is named ``<service>-<volume>``.

Two processes, one directory. The executor (``vaelor-workloads``) runs the
removal and writes the record; the control plane (``vaelor``) lists records and
deletes them. Both are in group ``vaelor-jobs``, and the records live in the
cluster backup directory, which is setgid 2770 for exactly that reason
(`cluster_backups.BACKUP_DIRECTORY_MODE`, VD-178). Records are 0640.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

from .cluster_placement import CONTROLLER_PLACEMENT_ID, CONTROLLER_PLACEMENT_NAME

RETAINED_ID = re.compile(r"retained-[a-f0-9]{16}")
# At most 160 characters, the length RetainedVolumeStore.record() keeps: a
# longer name would be stored cut short, and that shorter name - still a valid
# volume name - is what a later delete would remove.
VOLUME_NAME = re.compile(r"vaelor-(?:app|llm)-[a-z0-9][a-z0-9_.-]{0,148}")
UNREADABLE_RECORD = "The retained volume record could not be read."
_NODE_PIN = re.compile(r"node\.labels\.vaelor\.node_id==(.+)")


def retained_id(node_id: str, volume: str) -> str:
    """One record per volume per machine: a second removal rewrites it."""
    digest = hashlib.sha256(f"{node_id}\0{volume}".encode()).hexdigest()
    return f"retained-{digest[:16]}"


def volume_mounts(details: Dict[str, Any]) -> List[str]:
    """The named volumes of a normalized ``service_details`` view."""
    return [
        str(mount.get("source", ""))
        for mount in (details or {}).get("mounts", []) or []
        if isinstance(mount, dict) and mount.get("type") == "volume"
        and VOLUME_NAME.fullmatch(str(mount.get("source", "")))
    ]


def pinned_node(constraints: Any) -> str:
    """The Vaelor node id a service's placement constraints pin it to, or ``""``."""
    node_id = ""
    for constraint in constraints or []:
        match = _NODE_PIN.fullmatch(str(constraint))
        if match:
            node_id = match.group(1)
    return node_id


#: What the owner types to forget a controller record (F3): nothing the console
#: can run sees the controller's volumes, so only the owner can say it is gone.
CONTROLLER_REMOVED_ACK = "removed on the controller"

#: A record whose machine could not be read from the service's tasks (F3).
UNKNOWN_MACHINE = "a machine Vaelor could not identify"


def task_holders(operations: Any, tasks: Any) -> List[str]:
    """The machines an unpinned service's tasks ran on, as Vaelor node ids.

    F3: an unpinned stateful service's volume is on whichever machine Swarm
    placed it, so the record names that machine from the task rows read
    before removal - each one, since a rescheduled task leaves a copy on every
    machine it ran on. A hostname that resolves to no enrolled machine, or no
    task rows at all, is ``""``: the record then says the machine is unknown.

    Review R2-5: task rows carry the hostname only (`docker service ps` has no
    node-id placeholder), so a hostname two Swarm nodes share is ambiguous and
    is recorded as unknown too - never resolved to one of them, which sent
    Delete to the wrong machine and forgot data still on the other.
    """
    hostnames = sorted({
        str(task.get("node", "")) for task in tasks or []
        if isinstance(task, dict) and task.get("node")
    })
    if not hostnames:
        return [""]
    try:
        status = operations.driver.status() or {}
        swarm_by_host: Dict[str, List[str]] = {}
        for node in status.get("nodes", []) or []:
            if isinstance(node, dict):
                swarm_by_host.setdefault(str(node.get("hostname", "")), []).append(
                    str(node.get("id", ""))
                )
        controller = str(
            operations.store.controller().get("cluster_id") or status.get("node_id") or ""
        )
        vaelor_by_swarm = {
            str((node.get("labels", {}) or {}).get("swarm_node_id", "")): str(node["id"])
            for node in operations.store.list_nodes()
        }
    except Exception:  # an unread placement is recorded as unknown, never guessed
        return [""]
    holders: List[str] = []
    for hostname in hostnames:
        candidates = swarm_by_host.get(hostname, [])
        swarm_id = candidates[0] if len(candidates) == 1 else ""
        holder = (
            CONTROLLER_PLACEMENT_ID if swarm_id and swarm_id == controller
            else vaelor_by_swarm.get(swarm_id, "") if swarm_id else ""
        )
        if holder not in holders:
            holders.append(holder)
    return holders


def holders_of(operations: Any, details: Dict[str, Any], pinned: str) -> List[str]:
    """Where a service's volumes are: its pinned machine, else where it ran."""
    return [pinned] if pinned else task_holders(operations, (details or {}).get("tasks"))


def holder_name(operations: Any, node_id: str) -> str:
    """The name a record shows for the machine holding its volume."""
    if node_id == CONTROLLER_PLACEMENT_ID:
        return CONTROLLER_PLACEMENT_NAME
    node = operations.store.get_node(node_id) if node_id else None
    if node_id and node is None:
        return "an unidentified machine"
    return str((node or {}).get("name", "") or UNKNOWN_MACHINE)


def inspect_volume_mounts(inspect: Dict[str, Any]) -> tuple[str, List[str], str]:
    """``(service, volumes, pinned node id)`` from one raw ``service inspect``."""
    spec = (inspect or {}).get("Spec", {}) or {}
    task = spec.get("TaskTemplate", {}) or {}
    volumes = [
        str(mount.get("Source", ""))
        for mount in (task.get("ContainerSpec", {}) or {}).get("Mounts", []) or []
        if isinstance(mount, dict) and str(mount.get("Type", "")) == "volume"
        and VOLUME_NAME.fullmatch(str(mount.get("Source", "")))
    ]
    node_id = pinned_node((task.get("Placement", {}) or {}).get("Constraints"))
    return str(spec.get("Name", "")), volumes, node_id


class RetainedVolumeStore:
    """Records of retained volumes, beside the cluster backups they relate to."""

    def __init__(self, backup_store: Any):
        self.backup_store = backup_store

    @property
    def root(self) -> Path:
        return Path(self.backup_store.root)

    def _path(self, record_id: str) -> Path:
        clean = str(record_id).strip()
        if not RETAINED_ID.fullmatch(clean):
            raise ValueError("Choose a retained data volume Vaelor listed.")
        return self.root / f"{clean}.json"

    def record(
        self, *, service_name: str, volume: str, node_id: str, node_name: str,
    ) -> Dict[str, Any]:
        """Write (or rewrite) the record of one retained volume. Raises OSError."""
        self.backup_store._ensure_root()
        record_id = retained_id(node_id, volume)
        value = {
            "id": record_id,
            "service_name": str(service_name)[:80],
            "volume": str(volume)[:160],
            "node_id": str(node_id)[:80],
            "node_name": str(node_name)[:120],
            "removed_at": int(time.time()),
        }
        path = self._path(record_id)
        # F9 / LESSONS 18: this directory is group-writable (2770), so a fixed
        # staging name could be a planted symlink, and write_text and chmod both
        # follow one. mkstemp creates a fresh name with O_CREAT|O_EXCL (and
        # O_NOFOLLOW where the platform has it); the mode is set on the
        # descriptor, and os.replace renames over a link, never through it.
        descriptor, staged = tempfile.mkstemp(
            dir=str(self.root), prefix=f".{record_id}.", suffix=".next",
        )
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o640)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(value, sort_keys=True))
            os.replace(staged, path)
        except BaseException:
            Path(staged).unlink(missing_ok=True)
            raise
        return value

    def get(self, record_id: str) -> Dict[str, Any]:
        path = self._path(record_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError("That retained data volume is no longer listed.") from error
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(UNREADABLE_RECORD) from error
        if not isinstance(value, dict) or value.get("id") != record_id:
            raise ValueError(UNREADABLE_RECORD)
        return value

    def list(self, *, live_services: set, backups_for: Callable[[str], int]) -> List[Dict[str, Any]]:
        """Every retained volume whose service is not running again.

        A service redeployed under the same name has taken its volume back, so
        its record is no longer "retained" and is left out (and rewritten if
        that service is removed again).
        """
        if not self.root.is_dir():
            return []
        values = []
        for path in self.root.glob("retained-*.json"):
            try:
                item = self.get(path.stem)
            except ValueError:
                continue
            if item.get("service_name") in live_services:
                continue
            values.append({**item, "backups": backups_for(str(item.get("service_name", "")))})
        values.sort(key=lambda item: item.get("removed_at", 0), reverse=True)
        return values

    def forget(self, record_id: str) -> None:
        self._path(record_id).unlink(missing_ok=True)


def record_retained(operations: Any, removed: List[tuple]) -> Dict[str, Any]:
    """Record what a removal left behind; ``removed`` is ``(service, volumes, holders)``.

    ``holders`` is the list of machines that may hold the volumes (`holders_of`);
    a bare node id is accepted as a list of one.

    Returns the records written and, if any write failed, why - the removal
    itself has already happened, so a failed record is reported, not raised
    (LESSONS 8: the absence of a record must not read as the absence of data).
    """
    written: List[Dict[str, Any]] = []
    errors: List[str] = []
    store = RetainedVolumeStore(operations.backup_store)
    for service_name, volumes, holders in removed:
        for node_id in [holders] if isinstance(holders, str) else holders or [""]:
            for volume in volumes:
                try:
                    written.append(store.record(
                        service_name=service_name, volume=volume,
                        node_id=str(node_id or ""),
                        node_name=holder_name(operations, str(node_id or "")),
                    ))
                except OSError as error:
                    errors.append(f"{volume}: {error}")
    result: Dict[str, Any] = {"retained_volumes": written}
    if errors:
        result["retained_record_error"] = (
            "The data volumes were kept on the worker, but Vaelor could not list "
            "them in the console: " + "; ".join(errors)
        )[:600]
    return result


def delete_retained_volume(
    operations: Any, record_id: str, confirmation: str, *, live_services: set,
    transport_factory: Callable[[Dict[str, Any]], Any], forget_only: bool = False,
    removed_ack: str = "",
) -> Dict[str, Any]:
    """Delete one retained volume on its worker, then its record.

    Refused unless the owner typed the volume's name, while a service of that
    name runs again (it has the volume back), or when the record names no
    machine Vaelor still knows. ``docker volume rm`` runs elevated on that
    worker over SSH; "no such volume" means someone removed it already, so the
    record goes and the answer says so.
    """
    store = RetainedVolumeStore(operations.backup_store)
    item = store.get(record_id)
    volume = str(item.get("volume", ""))
    if not VOLUME_NAME.fullmatch(volume):
        raise ValueError("The retained volume record names no Vaelor volume.")
    if str(confirmation or "").strip() != volume:
        raise ValueError(f'Type "{volume}" to delete this data.')
    if item.get("service_name") in live_services:
        raise ValueError(
            f"{item['service_name']} is running again and is using this volume. "
            "Remove that service first."
        )
    node_id = str(item.get("node_id", ""))
    if node_id in (CONTROLLER_PLACEMENT_ID, "") and not forget_only:
        # F3: no console path reaches Docker on the controller (the control
        # plane holds no Docker privilege, task #153), and an unknown holder
        # has no machine to reach. Say exactly what to run, and where.
        if node_id:
            raise ValueError(
                "Deleting data on the controller from the console is not available "
                f"yet. Run \"sudo docker volume rm {volume}\" on this controller, "
                "then forget this entry."
            )
        raise ValueError(
            f"Vaelor cannot delete this data from the console. Run \"sudo docker "
            f"volume rm {volume}\" on each machine the app may have run on (Vaelor "
            "could not tell which one held it when the app was removed), then "
            "forget this entry."
        )
    if node_id == CONTROLLER_PLACEMENT_ID and (
        str(removed_ack or "").strip().lower() != CONTROLLER_REMOVED_ACK
    ):
        # The data is still on the controller until the owner removes it, and
        # nothing here can see that it has gone (LESSONS 22): forgetting needs
        # the owner's word that the command ran.
        raise ValueError(
            f'Run "sudo docker volume rm {volume}" on this controller first, then '
            f'type "{CONTROLLER_REMOVED_ACK}" to forget this entry.'
        )
    node = operations.store.get_node(node_id, include_credential=True) if node_id else None
    if node is not None and forget_only:
        raise ValueError(
            f"{node.get('name', 'That machine')} is still enrolled, so delete the "
            "data there instead of only forgetting it."
        )
    if node is None and forget_only:
        # The one exit for a record whose machine is gone (LESSONS 22): the
        # data, if any, is on a machine Vaelor can no longer reach.
        store.forget(record_id)
        return {"id": record_id, "volume": volume, "deleted": False, "forgotten": True}
    if node is None:
        raise ValueError(
            "The machine this volume is on is no longer enrolled, so Vaelor cannot "
            "reach it to delete the data. Remove the volume on that machine "
            "yourself, or forget this entry."
        )
    profile = operations.broker.resolve(node["credential_id"], "cluster-node")
    transport = transport_factory(profile)
    already_gone = False
    try:
        transport.run(["docker", "volume", "rm", volume], sudo=True, timeout=60)
    except Exception as error:  # SshTransportError and friends: read the cause
        message = str(error)
        if "no such volume" in message.lower():
            already_gone = True
        else:
            raise ValueError(
                f"{node.get('name', 'The worker')} did not delete {volume}: {message[:400]}"
            ) from error
    store.forget(record_id)
    return {
        "id": record_id, "volume": volume, "node_name": node.get("name", ""),
        "deleted": True, "already_gone": already_gone,
    }
