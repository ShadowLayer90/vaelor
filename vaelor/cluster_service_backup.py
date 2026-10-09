"""Verified worker-volume backup and restore for a cluster service.

Extracted from `cluster_operations` (which sat at the 1,000-line ceiling) as a
cohesive cluster of functions, each taking the `ClusterOperations` instance and
reaching back through it for the driver, the credential broker, the backup
store, and the node resolution — the same shape `cluster_app_deploy` and
`cluster_app_multideploy` use. The move is behaviour-neutral: `ClusterOperations`
keeps `backup_service`/`restore_service`/`_backup_target`/`_apply_cluster_archive`
as thin methods that delegate here, so every call site and the instance seams the
tests mock (`operations.backup_service`, `operations._apply_cluster_archive`) are
unchanged.

A stateful cluster app keeps its data on the one machine it is pinned to and has
no failover, so a verified backup is the ONLY recovery path a remove or a node
drain leaves it (the data-loss gate in `cluster_service_state` refuses those
without one). Backup runs a capability-dropped, network-less alpine container on
the pinned worker to tar the named volume read-only, copies the archive back, and
records it with its digest; restore verifies the digest on the worker, makes a
safety backup first, scales the service to zero, replaces the volume contents,
and scales back — rolling the safety archive back in on any failure.

**Staging (VD-172, review round 1).** Both directions stage the archive in a
private directory (`vaelor.ssh_private_staging`: 0700, the login account's,
verified, random name), never at a bare ``/tmp`` path where root's tar left a
whole volume 0644. The directory is handed to root (``chown``) only for the
capability-less container's own write or read - it mounts that directory,
never ``/tmp`` - and is removed with ``sudo rm -rf`` afterwards, on failure too.
"""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from .ssh_transport import SshTransport

#: The pin constraint that ties a service (and therefore its named volume) to one
#: enrolled worker. Backup and restore both need that node to reach the volume.
_NODE_PIN = re.compile(r"node\.labels\.vaelor\.node_id==(.+)")

#: The largest archive a backup copies back or a restore uploads.
CLUSTER_ARCHIVE_MAX_BYTES = 10 * 1024 ** 3

#: What the helper container gets back after ``--cap-drop ALL`` (W4c-D1, live
#: 2026-10-03). Its root is uid 0 with no capabilities, so it could neither
#: read an app user's 0600 file (Heimdall's ``keys/cert.key``) nor remove an
#: entry from a volume root the app user owns (Uptime Kuma's
#: ``/data/screenshots``). Backup reads and searches everything and writes
#: nothing (the volume is mounted read-only too); restore clears the tree and
#: puts each file's owner and mode back. Busybox tar as root restores owners
#: and modes by default, and ``--numeric-owner`` keeps uid 1000 as 1000
#: whatever the helper image's /etc/passwd names it. ``--network none`` and the
#: read-only root filesystem stay on both.
BACKUP_CAPABILITIES = ("--cap-add", "DAC_READ_SEARCH")
#: FSETID (review follow-up 4): without it the kernel clears a setgid bit the
#: restore sets on a file whose group the helper's root is not in, so shared
#: group directories came back without it. It lets the helper keep setuid and
#: setgid bits on files it chmods - inside the one volume it mounts, from an
#: archive of that same volume - and nothing else: no network, a read-only
#: root, no-new-privileges, and a pinned image.
RESTORE_CAPABILITIES = (
    "--cap-add", "DAC_OVERRIDE", "--cap-add", "CHOWN", "--cap-add", "FOWNER",
    "--cap-add", "FSETID",
)

#: The helper both directions run (review follow-up 5, LESSONS 18). With
#: DAC_READ_SEARCH a process may call open_by_handle_at under Docker's default
#: seccomp, so the image is trusted with the worker's whole filesystem: it is
#: pinned by digest - a moving tag a poisoned registry could repoint is not
#: enough - and runs with no-new-privileges. The digest is Docker Hub's for
#: ``alpine:3.20`` as of 2026-10-03 (tag last updated 2026-04-17). To re-pin:
#: ``docker pull alpine:3.20`` and ``docker image inspect --format
#: '{{index .RepoDigests 0}}' alpine:3.20``.
HELPER_IMAGE = (
    "alpine:3.20"
    "@sha256:d9e853e87e55526f6b2917df91a2115c36dd7c696a35be12163d44e6e2a4b6bc"
)
HELPER_HARDENING = ("--security-opt", "no-new-privileges")

#: How much of one raw error a failed restore's sentence carries: the job
#: keeps 1,000 characters, and what to do must survive it (follow-up 3).
RAW_ERROR_CHARS = 200


def backup_target(operations, service_name: str):
    """Resolve the pinned worker, its SSH profile, and the one named volume.

    A cluster backup needs exactly one managed named volume on a service pinned
    to an enrolled worker; anything else is refused honestly here, before any
    container runs.
    """
    details = operations.driver.service_details(service_name)
    node_id = ""
    for constraint in details.get("constraints", []):
        match = _NODE_PIN.fullmatch(str(constraint))
        if match:
            node_id = match.group(1)
            break
    if not node_id:
        raise ValueError(
            "This service is not pinned to an enrolled Vaelor worker."
        )
    mounts = [
        item for item in details.get("mounts", [])
        if item.get("type") == "volume"
        and re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", item.get("source", ""))
    ]
    if len(mounts) != 1:
        raise ValueError(
            "Cluster backup currently requires exactly one managed named volume."
        )
    node = operations._joined_node(node_id, include_credential=True)
    profile = operations.broker.resolve(node["credential_id"], "cluster-node")
    return details, node, profile, mounts[0]


def backup_service(operations, payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("confirm") != "backup-cluster-service":
        raise ValueError("Confirm the reviewed cluster service backup.")
    name = str(payload.get("service_name", "")).strip()
    details, node, profile, mount = operations._backup_target(name)
    transport = SshTransport(profile)
    archive_name = f"vaelor-cluster-{uuid.uuid4().hex}.tar.gz"
    # The controller's own staging first (round 2, F4): a controller that
    # cannot stage - a full disk - then fails before the worker holds anything.
    staging_root = Path(
        operations.backup_store.root.parent / ".cluster-staging"
    )
    staging_root.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(
        prefix="cluster-", suffix=".tar.gz", dir=staging_root
    )
    os.close(descriptor)
    staged_path = Path(staged_name)
    directory = ""
    try:
        staged_path.unlink(missing_ok=True)
        directory = transport.make_private_directory()
        # The container's root has no capabilities, so it writes only where
        # uid 0 owns the directory: root's for the write, 0700 throughout.
        # -h: the directory itself, never something a link points at.
        transport.run(["chown", "-h", "root:root", directory], sudo=True)
        transport.run(
            [
                "docker", "run", "--rm",
                "--network", "none",
                "--read-only",
                "--cap-drop", "ALL",
                *HELPER_HARDENING,
                # W4c-D1: read an app user's private files; writes nothing.
                *BACKUP_CAPABILITIES,
                "-v", f"{mount['source']}:/data:ro",
                "-v", f"{directory}:/backup",
                HELPER_IMAGE,
                "tar", "-czf", f"/backup/{archive_name}",
                "-C", "/data", ".",
            ],
            sudo=True,
            timeout=1800,
        )
        # The login account's again, so the copy back reads it over SFTP.
        transport.run(
            ["chown", "-R", str(profile["username"]), directory],
            sudo=True,
        )
        transport.get_cluster_archive(f"{directory}/{archive_name}", str(staged_path))
        result = operations.backup_store.record(
            service_name=name,
            node_id=node["id"],
            volume=mount["source"],
            staged_archive=staged_path,
            reason=str(payload.get("reason", "manual")),
        )
        return {
            **result,
            "node_name": node["name"],
            "desired_replicas": details["desired_replicas"],
            "verified": True,
        }
    finally:
        staged_path.unlink(missing_ok=True)
        if directory:
            transport.remove_private_directory(directory)


def apply_cluster_archive(
    operations,
    *,
    transport: SshTransport,
    archive: Dict[str, Any],
    volume: str,
) -> None:
    # 0600 inside a private 0700 directory from its first byte (VD-172).
    remote_name = transport.stage_private_artifact(
        archive["archive_path"], f"vaelor-cluster-{uuid.uuid4().hex}.tar.gz",
        maximum_bytes=CLUSTER_ARCHIVE_MAX_BYTES,
    )
    directory = remote_name.rsplit("/", 1)[0]
    try:
        observed = transport.run(["sha256sum", remote_name]).split()[0]
        if observed != archive["sha256"]:
            raise ValueError("The worker received an invalid backup archive.")
        # The directory and the archive become root's, still 0700 and 0600,
        # so the container's root process reads them as their owner.
        transport.run(
            ["chown", "-R", "root:root", directory],
            sudo=True,
        )
        transport.run(
            [
                "docker", "run", "--rm",
                "--network", "none",
                "--read-only",
                "--cap-drop", "ALL",
                *HELPER_HARDENING,
                # W4c-D1: clear a tree the app user owns, then put each
                # file's owner and mode back.
                *RESTORE_CAPABILITIES,
                "-v", f"{volume}:/data",
                "-v", f"{directory}:/backup:ro",
                HELPER_IMAGE,
                "sh", "-ceu",
                (
                    'find /data -mindepth 1 -maxdepth 1 '
                    '-exec rm -rf -- "{}" +; '
                    'tar -xzf "$1" --numeric-owner -C /data'
                ),
                "vaelor-restore",
                f"/backup/{Path(remote_name).name}",
            ],
            sudo=True,
            timeout=1800,
        )
    finally:
        transport.remove_private_directory(directory)


#: What a failed restore says (W4c-D2): the data state it ended in and
#: whether the service is running again FIRST, then each raw error, capped -
#: the job keeps 1,000 characters, and a 40-line rm failure used to push what
#: to do off the end (review follow-up 3, LESSONS 11).
RESTORE_FAILED = "The restore failed. {} {} What failed: {}"
NOT_CHANGED = "The volume was not changed: the service did not stop."
SAFETY_PUT_BACK = "The safety backup {} was put back, so the volume holds what it held before."
SAFETY_NOT_PUT_BACK = (
    "The safety backup {} could not be put back either, so the volume may be "
    "partly cleared; restore that backup once the cause is fixed."
)
SCALED_BACK = "The service was scaled back to {} replicas."
NOT_SCALED_BACK = (
    "The service could not be scaled back to {} replicas and is stopped; "
    "scale it up by hand."
)
RESTORED_NOT_SCALED_BACK = "The backup was restored, but the service could not be scaled back to {} replicas and is stopped; scale it up by hand. What failed: {}"


def _raw(label: str, error: BaseException) -> str:
    """One raw error for the end of the sentence, capped to RAW_ERROR_CHARS."""
    text = " ".join(str(error).split())
    if len(text) > RAW_ERROR_CHARS:
        text = text[:RAW_ERROR_CHARS - 1] + "\u2026"
    return "{}: {}".format(label, text)


def restore_service(operations, payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("confirm") != "restore-cluster-service":
        raise ValueError("Confirm the reviewed cluster service restore.")
    name = str(payload.get("service_name", "")).strip()
    backup = operations.backup_store.get(
        str(payload.get("backup_id", "")), verify=True
    )
    if backup["service_name"] != name:
        raise ValueError("Choose a backup created for this service.")
    details, node, profile, mount = operations._backup_target(name)
    if (
        backup["node_id"] != node["id"]
        or backup["volume"] != mount["source"]
    ):
        raise ValueError(
            "The backup does not match this worker volume placement."
        )
    safety = operations.backup_service({
        "confirm": "backup-cluster-service",
        "service_name": name,
        "reason": "pre-restore",
    })
    safety_archive = operations.backup_store.get(safety["id"], verify=True)
    transport = SshTransport(profile)
    replicas = int(details["desired_replicas"])
    # W4c-D2 and review B1: the service is scaled back to its replicas
    # whatever happens - to the data, or to the stop itself. `docker service
    # scale name=0` sets the desired count before its wait for 0/0 can fail,
    # so the stop is inside the guarded region too; the live run left the
    # service at 0/0, "Deploying", because the scale-back sat inside the try
    # whose rollback failed the same way.
    failure: Optional[BaseException] = None
    raw = []
    data_state = NOT_CHANGED
    applying = False
    try:
        operations.driver.scale_service(name, 0)
        applying = True
        operations._apply_cluster_archive(
            transport=transport, archive=backup, volume=mount["source"],
        )
    except Exception as error:  # noqa: BLE001 - said below, then raised
        failure = error
        raw.append(_raw("restore" if applying else "stop", error))
        if applying:
            try:
                operations._apply_cluster_archive(
                    transport=transport, archive=safety_archive, volume=mount["source"],
                )
                data_state = SAFETY_PUT_BACK.format(safety["id"])
            except Exception as rollback_error:  # noqa: BLE001 - said, never hidden
                data_state = SAFETY_NOT_PUT_BACK.format(safety["id"])
                raw.append(_raw("putting the safety backup back", rollback_error))
    try:
        ready = operations.driver.scale_service(name, replicas)
        scaled = SCALED_BACK.format(replicas)
    except Exception as scale_error:  # noqa: BLE001 - said, never hidden
        ready, scaled = {}, NOT_SCALED_BACK.format(replicas)
        raw.append(_raw("scaling back", scale_error))
        if failure is None:
            raise RuntimeError(RESTORED_NOT_SCALED_BACK.format(
                replicas, "; ".join(raw))) from scale_error
    if failure is not None:
        raise RuntimeError(RESTORE_FAILED.format(data_state, scaled, "; ".join(raw))) from failure
    return {
        "restored": True,
        "service_name": name,
        "backup_id": backup["id"],
        "safety_backup_id": safety["id"],
        **ready,
    }
