"""A worker's private staging directory: where a file lands before root installs it.

Review A10 (VD-172). Artifacts used to be uploaded to a ``/tmp/vaelor-<name or
digest>`` path any account on the worker could compute, plant, or swap between
a digest check and root's ``install``/``tar``; cluster backups and restores
used a ``/tmp/vaelor-cluster-<uuid>.tar.gz`` that root's tar left 0644 and
that a restore made 0600 only after uploading it. Everything a worker install
or a cluster backup stages now goes through here.

A staging directory is ``/tmp/vaelor-stage-<128 random bits>``, created by the
SFTP session itself (``mkdir`` fails on a name that exists, so nothing can be
planted in advance), asked for ``0700`` and set to it, then checked before
anything is written into it: a directory (``lstat``, so a symlink is not
followed), owned by the login account, no group or other bits, and empty.
OpenSSH satisfies all four on its own; the checks are for a server that does
not. A staging that fails after its ``mkdir`` removes what it made.

Runs in whichever process holds an `SshTransport` (the workload executor for
installs, the control plane for backups); the `SshTransport` methods of the
same names delegate here. Removal is best effort and never raises - it runs in
a ``finally``, where raising would replace the error that matters.
"""

from __future__ import annotations

import logging
import re
import secrets
import stat
from pathlib import Path
from typing import Any

from .ssh_transport import SshTransportError

LOGGER = logging.getLogger(__name__)

#: An artifact to upload that is missing or over its size bound, and a
#: staging path outside the shapes written here.
ARTIFACT_UNAVAILABLE = "The reviewed runtime artifact is unavailable."
STAGING_PATH_REFUSED = "The runtime staging path is not permitted."
NOT_PRIVATE = "The private staging directory is not private."
_NOT_REMOVED = "Could not remove the staged %s: %s"

#: The default bound on one staged artifact; a cluster backup passes its own.
ARTIFACT_MAX_BYTES = 100 * 1024 ** 2

PRIVATE_STAGING_PREFIX = "/tmp/vaelor-stage-"
#: One plain file name inside a staging directory.
STAGED_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
#: A staging directory, and a file staged in one.
PRIVATE_DIRECTORY = re.compile(re.escape(PRIVATE_STAGING_PREFIX) + r"[0-9a-f]{32}")
PRIVATE_STAGED_PATH = re.compile(
    re.escape(PRIVATE_STAGING_PREFIX) + r"[0-9a-f]{32}/[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
)


def is_private_staged_path(value: Any) -> bool:
    """Whether ``value`` is a file in a staging directory, exactly."""
    return bool(PRIVATE_STAGED_PATH.fullmatch(str(value)))


def login_uid(transport: Any) -> int:
    """The login account's numeric uid on the worker (``id -u``)."""
    try:
        return int(str(transport.run(["id", "-u"])).strip())
    except (ValueError, TypeError) as error:
        raise SshTransportError(
            "The worker did not say which account Vaelor signs in as."
        ) from error


def refuse_unless_private(sftp: Any, directory: str, owner: int) -> None:
    """Refuse a staging directory that is not this session's own."""
    attributes = sftp.lstat(directory)
    mode = attributes.st_mode
    if (
        not stat.S_ISDIR(mode) or stat.S_IMODE(mode) & 0o077
        or getattr(attributes, "st_uid", None) != owner or sftp.listdir(directory)
    ):
        raise SshTransportError(NOT_PRIVATE)


def _remove_quietly(sftp: Any, *targets: Any) -> None:
    """Best-effort removal, file then directory, of a failed staging."""
    for remove, target in targets:
        try:
            remove(target)
        except Exception as error:  # noqa: BLE001 - the staging's own error stands
            LOGGER.info("Could not remove %s after a failed staging: %s", target, error)


def _create(sftp: Any, owner: int) -> str:
    """Make and verify one staging directory over an open SFTP session."""
    directory = PRIVATE_STAGING_PREFIX + secrets.token_hex(16)
    try:
        sftp.mkdir(directory, 0o700)
    except OSError as error:
        raise SshTransportError(
            "A private staging directory could not be created on the worker."
        ) from error
    try:
        # Set explicitly: an SFTP server may mask the mode mkdir asked for
        # with the session's umask, or ignore it.
        sftp.chmod(directory, 0o700)
        refuse_unless_private(sftp, directory, owner)
    except BaseException:
        _remove_quietly(sftp, (sftp.rmdir, directory))
        raise
    return directory


def make_private_directory(transport: Any) -> str:
    """An empty, verified staging directory; remove it with :func:`remove_private_directory`."""
    owner = transport._login_uid()
    client = transport._session()
    try:
        with client.open_sftp() as sftp:
            return _create(sftp, owner)
    finally:
        client.close()


def stage_private_artifact(
    transport: Any, source: str, filename: str, *, maximum_bytes: int = ARTIFACT_MAX_BYTES,
) -> str:
    """Upload ``source`` as ``filename`` into a fresh staging directory; answer its path."""
    path = Path(source)
    if not path.is_file() or path.stat().st_size > int(maximum_bytes):
        raise SshTransportError(ARTIFACT_UNAVAILABLE)
    if not STAGED_NAME.fullmatch(str(filename)):
        raise SshTransportError("The runtime staging file name is not permitted.")
    owner = transport._login_uid()
    client = transport._session()
    try:
        with client.open_sftp() as sftp:
            directory = _create(sftp, owner)
            remote = directory + "/" + filename
            try:
                sftp.put(str(path), remote)
                sftp.chmod(remote, 0o600)
            except BaseException:
                _remove_quietly(sftp, (sftp.remove, remote), (sftp.rmdir, directory))
                raise
        return remote
    finally:
        client.close()


def discard_private_artifact(transport: Any, remote: str) -> None:
    """Remove a staged file and its directory over SFTP; log, never raise."""
    if not is_private_staged_path(remote):
        LOGGER.warning("Not removing %r: it is not a private staging path.", remote)
        return
    try:
        client = transport._session()
    except Exception as error:  # noqa: BLE001 - logged; the caller's error stands
        LOGGER.warning(_NOT_REMOVED, remote, error)
        return
    try:
        with client.open_sftp() as sftp:
            sftp.remove(remote)
            sftp.rmdir(remote.rsplit("/", 1)[0])
    except Exception as error:  # noqa: BLE001 - logged; the caller's error stands
        LOGGER.warning(_NOT_REMOVED, remote, error)
    finally:
        client.close()


def remove_private_directory(transport: Any, directory: str) -> None:
    """Remove a staging directory root may have taken over; log, never raise.

    A backup or restore hands its directory to root (the container's root
    writes or reads inside it), so it is removed with ``sudo rm -rf`` - only
    for a path of exactly the staging-directory shape.
    """
    if not PRIVATE_DIRECTORY.fullmatch(str(directory)):
        LOGGER.warning("Not removing %r: it is not a private staging directory.", directory)
        return
    try:
        transport.run(["rm", "-rf", "--", directory], sudo=True)
    except Exception as error:  # noqa: BLE001 - logged; the caller's error stands
        LOGGER.warning("Could not remove the staging directory %s: %s", directory, error)
