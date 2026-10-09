"""Which build of Vaelor is installed, by its bytes rather than its version string.

W4d-D8: the update panel called the appliance "up to date" and offered to
"Reinstall 1.0b2" because the running version string and the published
release's version string were equal. The installed wheel had been built that
day from a newer commit; the published 1.0b2 was a month older and a different
file. Two artifacts sharing one name were treated as one (LESSONS 6), and the
version string was recorded as a fact about the build (LESSONS 9).

So the identity of an install here is the wheel's SHA-256, byte length, the
commit it was built from (when the build stamped it) and when it was built.
Three parties touch it, and each runs a different function in this module:

* the **installer** (``deploy/install-vaelor.sh``, as root) and the **upgrade
  broker** (``appliance_upgrade.record_current_wheel``, as root) call
  :func:`write_installed_record` with the wheel they just installed. The record
  is ``root:root 0644`` beside the state root, so every service can read it and
  none but root can rewrite it;
* the **control plane** and the **workload executor** call
  :func:`installed_build` to read it back.

**A record is only believed while it describes the code that is running.** A
wheel can be installed by a path that never writes the record (a hand-run
``pip install``), and a stale record would then vouch for a build that is no
longer there - and could talk the panel into offering a downgrade. So the record
carries a *content digest*: one SHA-256 over the per-file hashes the wheel's own
``RECORD`` lists for the ``vaelor`` and ``pm_dashboard`` packages. pip copies
those hashes into the installed ``RECORD`` unchanged, so the same digest is
computable from the live installation, and :func:`installed_build` reports the
record only when the two agree. Otherwise the build is ``unknown``, and the
update decision treats unknown as unknown rather than as "same".

The build stamp (``vaelor/build_stamp.json``) is written by
``tools/build_release.py`` from a clean tree and removed after the build, so a
release wheel carries the commit it came from. A wheel built any other way has
no stamp and its commit is reported as not recorded, never guessed.
"""

from __future__ import annotations

import base64
import calendar
import csv
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .runtime_paths import env_value

DISTRIBUTION = "vaelor-control-plane"
STAMP_NAME = "build_stamp.json"
RECORD_NAME = "installed-build.json"
#: The packages whose file hashes make up the content digest. Scripts under
#: ``bin/`` are rewritten by pip at install time and so are excluded, as are
#: the files pip itself adds (``INSTALLER``, ``REQUESTED``, ``direct_url.json``).
CONTENT_PACKAGES = ("vaelor/", "pm_dashboard/")
#: Where a build's time came from: the stamp the release build wrote, or the
#: newest file timestamp inside the wheel, which is a lower bound and is said so.
BASIS_STAMP = "build-stamp"
BASIS_NEWEST_FILE = "newest-file-in-wheel"
#: The owner of ``built_at_basis``, a wire vocabulary the update panel reads
#: (LESSONS 6 / VD-090 item 5, W4d-D8): its one hand copy is BUILT_AT_BASES in
#: frontend/src/components/SystemUpdatePanel.tsx.
BUILT_AT_BASES = (BASIS_STAMP, BASIS_NEWEST_FILE)

#: Whether ``installed_build`` could vouch for the running installation:
#: ``recorded`` (the record matches it), ``stale`` (it describes another build)
#: or ``unknown`` (either side unreadable). Only ``recorded`` carries ``build``.
INSTALLED_RECORDED = "recorded"
INSTALLED_STALE = "stale"
INSTALLED_UNKNOWN = "unknown"
#: The owner of ``installed_build.state``, a wire vocabulary the update panel
#: and ``release_source.upgrade_eligibility`` read (LESSONS 6 / VD-090 item 5,
#: W4d-D8): its one hand copy is INSTALLED_BUILD_STATES in SystemUpdatePanel.tsx.
INSTALLED_BUILD_STATES = (INSTALLED_RECORDED, INSTALLED_STALE, INSTALLED_UNKNOWN)


def installed_record_path() -> Path:
    """The record's path, resolved at call time so a test's state root holds."""
    root = env_value("VAELOR_STATE_ROOT", "PM_STATE_ROOT", "/var/lib/vaelor")
    return Path(root) / RECORD_NAME


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def parse_time(value: Any) -> Optional[float]:
    """An ISO-8601 time as epoch seconds, or ``None`` if it will not parse."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def content_digest(entries: Iterable[Tuple[str, str]]) -> Optional[str]:
    """One SHA-256 over ``(path, hash)`` pairs of the content packages.

    The pairs are filtered to :data:`CONTENT_PACKAGES` and to entries that carry
    a hash, sorted, and joined, so the wheel's ``RECORD`` and the installed
    ``RECORD`` produce the same digest for the same build. ``None`` when nothing
    qualifies: an empty digest would match every other empty digest.
    """
    rows = sorted(
        (path, digest)
        for path, digest in entries
        if digest and path.startswith(CONTENT_PACKAGES)
        and "__pycache__" not in path
    )
    if not rows:
        return None
    hasher = hashlib.sha256()
    for path, digest in rows:
        hasher.update("{}\t{}\n".format(path, digest).encode("utf-8"))
    return hasher.hexdigest()


def _record_entries(text: str) -> List[Tuple[str, str]]:
    return [
        (row[0], row[1])
        for row in csv.reader(io.StringIO(text))
        if len(row) >= 2
    ]


def wheel_provenance(wheel: Path) -> Dict[str, Any]:
    """Read a wheel's identity from its bytes: digest, size, version, stamp, date.

    Raises ``ValueError`` for a file that is not a Vaelor wheel, because a
    record written from the wrong file is the defect this module exists to end.
    """
    wheel = Path(wheel)
    hasher = hashlib.sha256()
    with wheel.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    try:
        archive = zipfile.ZipFile(wheel)
    except zipfile.BadZipFile as error:
        raise ValueError("{} is not a wheel archive.".format(wheel.name)) from error
    with archive:
        names = archive.namelist()
        dist_info = sorted(
            name for name in names
            if name.endswith(".dist-info/RECORD") and name.count("/") == 1
        )
        metadata = sorted(
            name for name in names
            if name.endswith(".dist-info/METADATA") and name.count("/") == 1
        )
        if not dist_info or not metadata:
            raise ValueError("{} carries no wheel metadata.".format(wheel.name))
        version = ""
        for line in archive.read(metadata[0]).decode("utf-8", "replace").splitlines():
            if line.startswith("Version:"):
                version = line.split(":", 1)[1].strip()
                break
        if not version:
            raise ValueError("{} does not state its version.".format(wheel.name))
        digest = content_digest(
            _record_entries(archive.read(dist_info[0]).decode("utf-8", "replace"))
        )
        stamp: Dict[str, Any] = {}
        stamp_member = "vaelor/" + STAMP_NAME
        if stamp_member in names:
            try:
                loaded = json.loads(archive.read(stamp_member).decode("utf-8"))
                stamp = loaded if isinstance(loaded, dict) else {}
            except ValueError:
                stamp = {}
        newest = max(
            (calendar.timegm(info.date_time + (0, 0, 0)) for info in archive.infolist()),
            default=None,
        )
    commit = stamp.get("commit") if isinstance(stamp.get("commit"), str) else None
    stamped_at = parse_time(stamp.get("built_at"))
    if stamped_at is not None:
        built_at, basis = _iso(stamped_at), BASIS_STAMP
    elif newest is not None:
        built_at, basis = _iso(newest), BASIS_NEWEST_FILE
    else:
        built_at, basis = None, None
    return {
        "version": version,
        "wheel_name": wheel.name,
        "sha256": hasher.hexdigest(),
        "bytes": wheel.stat().st_size,
        "commit": commit,
        "commit_clean": stamp.get("clean") if isinstance(stamp.get("clean"), bool) else None,
        "built_at": built_at,
        "built_at_basis": basis,
        "content_digest": digest,
    }


def write_installed_record(wheel: Path, path: Optional[Path] = None) -> Dict[str, Any]:
    """Record the wheel that was just installed, as root, readable by all.

    Written atomically (a temporary file in the same directory, then
    ``os.replace``) so a reader never sees half a record.
    """
    record = wheel_provenance(wheel)
    record["recorded_at"] = _iso(time.time())
    target = Path(path) if path is not None else installed_record_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".installed-build-", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.chmod(temporary, 0o644)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return record


def running_content_digest() -> Optional[str]:
    """The content digest of the installed ``vaelor-control-plane`` distribution.

    ``None`` when the package is not installed as a distribution (a source
    checkout) or its ``RECORD`` cannot be read - an absence the caller reports
    as "unknown", never as a match.
    """
    try:
        from importlib import metadata

        files = metadata.distribution(DISTRIBUTION).files or []
    except Exception:  # noqa: BLE001  # absence-ok: an unreadable install is reported "unknown", never a match
        return None
    entries = []
    for packaged in files:
        file_hash = getattr(packaged, "hash", None)
        if file_hash is None:
            continue
        entries.append((
            str(packaged).replace("\\", "/"),
            "{}={}".format(file_hash.mode, file_hash.value),
        ))
    return content_digest(entries)


#: Said by both the trust check and the non-blocking read (R2-7), so one home.
NOT_A_REGULAR_FILE = "The install record is not a regular file."


def record_distrust(status: Any) -> str:
    """Why an install record with this ``stat`` is not evidence, or ``""``.

    F8a (VD-184): ``/var/lib/vaelor`` is ``root:vaelor 1770``, so the control
    plane can create the record when it is absent. Only the installer and the
    root upgrade broker write it, so a record is believed only as a regular
    file owned by root that no group or other account can write.
    """
    if not stat.S_ISREG(status.st_mode):
        return NOT_A_REGULAR_FILE
    if status.st_uid != 0:
        return "The install record is not owned by root, so it is not the installer's."
    if status.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return "The install record can be written by accounts other than root."
    return ""


def _read_record(target: Path, trusted: Callable[[Any], str]) -> Any:
    """Parse the record through a descriptor opened without following a link,
    after checking the file it actually is; raise ``ValueError`` when it is
    not trusted, with the reason.

    R2-7: the open is ``O_NONBLOCK`` and the type is checked on the descriptor
    before anything reads it. A group-``vaelor`` process can create a FIFO at
    the record's path (the state root is ``1770``), and a blocking open of a
    FIFO with no writer never returns - it would have hung the update panel's
    check and the executor's re-check. Only a regular file is read, whatever
    ``trusted`` says, and blocking is restored on it before the read
    (LESSONS 8: an observer that hangs is not an observer).
    """
    nonblock = getattr(os, "O_NONBLOCK", 0)
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | nonblock
             | getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(str(target), flags)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise _DistrustedRecord(NOT_A_REGULAR_FILE)
        reason = trusted(status)
        if reason:
            raise _DistrustedRecord(reason)
        if nonblock:
            os.set_blocking(descriptor, True)
        handle = os.fdopen(descriptor, "r", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise
    with handle:
        return json.loads(handle.read())


class _DistrustedRecord(ValueError):
    """The record exists but is not one the installer could have written."""


def installed_build(
    path: Optional[Path] = None,
    running_digest: Callable[[], Optional[str]] = running_content_digest,
    trusted: Callable[[Any], str] = record_distrust,
) -> Dict[str, Any]:
    """The installed build as far as it can be proven, and how far that is.

    ``state`` is ``recorded`` when the record is root's (:func:`record_distrust`),
    parses, and its content digest equals the running installation's;
    ``stale`` when both are readable and differ (something installed Vaelor
    without recording it); ``unknown`` when either side is missing or the
    record is not trusted. Only ``recorded`` carries ``build``.

    **What the digest does not prove (F8b, parked).** The content digest is
    computed over the per-file hashes the installed distribution's ``RECORD``
    lists, not over the files on disk, so a hot-patched file that was copied
    over an installed one leaves the digest - and so ``recorded`` - unchanged.
    ``recorded`` means "the installer's wheel, as pip recorded it", not "these
    exact bytes are what runs".
    """
    target = Path(path) if path is not None else installed_record_path()
    try:
        record = _read_record(target, trusted)
    except _DistrustedRecord as error:
        return {"state": INSTALLED_UNKNOWN, "reason": str(error), "build": None}
    except FileNotFoundError:
        return {
            "state": INSTALLED_UNKNOWN,
            "reason": "No install record exists; this appliance was installed "
            "before Vaelor recorded which build it installs.",
            "build": None,
        }
    except (OSError, ValueError) as error:
        return {
            "state": INSTALLED_UNKNOWN,
            "reason": "The install record could not be read: {}.".format(
                getattr(error, "strerror", None) or type(error).__name__
            ),
            "build": None,
        }
    if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
        return {"state": INSTALLED_UNKNOWN, "reason": "The install record is malformed.", "build": None}
    live = running_digest()
    if live is None:
        return {
            "state": INSTALLED_UNKNOWN,
            "reason": "The running installation's file list could not be read, so "
            "the install record cannot be matched to it.",
            "build": None,
        }
    if record.get("content_digest") != live:
        return {
            "state": INSTALLED_STALE,
            "reason": "The install record describes a different build from the "
            "one running: Vaelor was installed since without being recorded.",
            "build": None,
        }
    public = {
        key: record.get(key)
        for key in (
            "version", "wheel_name", "sha256", "bytes", "commit",
            "commit_clean", "built_at", "built_at_basis", "recorded_at",
        )
    }
    return {"state": INSTALLED_RECORDED, "reason": "", "build": public}


def encode_record_hash(data: bytes) -> str:
    """A ``RECORD``-style hash (``sha256=<urlsafe base64, unpadded>``) of ``data``.

    The shape wheel ``RECORD`` files use; exposed so a test can build a wheel
    whose ``RECORD`` the digest reads exactly as pip would leave it.
    """
    raw = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + raw.decode("ascii")


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m vaelor.build_provenance record <wheel>`` - the installer's call."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "record":
        print("usage: python -m vaelor.build_provenance record <wheel>", file=sys.stderr)
        return 2
    try:
        record = write_installed_record(Path(args[1]))
    except (OSError, ValueError) as error:
        print("Could not record the installed build: {}".format(error), file=sys.stderr)
        return 1
    print(json.dumps({"recorded": record["sha256"], "version": record["version"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

