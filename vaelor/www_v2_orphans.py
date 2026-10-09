"""Remove files under the installed ``vaelor/www_v2/`` that the wheel did not ship.

``pip install --force-reinstall`` uninstalls exactly what the *previous*
``RECORD`` lists and nothing else, so a file put under the package by hand (a
hot-patch) or left by a build pip never recorded survives every reinstall. The
Z2 carried 24 such frontend assets: stale code served from the package
directory (W8-P1 / DEP-3). After the wheel installs, ``install-vaelor.sh`` and
the upgrade broker (``appliance_upgrade._pip_reinstall``) run
``python -I -m vaelor.www_v2_orphans``, which removes every regular file under
``vaelor/www_v2/`` that the installed distribution's ``RECORD`` does not list.
``-I`` keeps the working directory off ``sys.path``: run from a folder holding
another ``vaelor`` (a clone, an unpacked wheel), plain ``-m`` sweeps THAT tree.
The broker runs the sweep the installed package carries, and the broker itself
is the code it was started with - so the first in-product upgrade INTO a release
that has this module does not sweep. The installer does, and so does every later
in-product upgrade.

Both callers run as root, so the sweep is deliberately narrow:

* The package directory comes from the installed distribution, never a fixed
  path, and only ``vaelor/www_v2/`` is walked.
* **An untrustworthy list removes nothing** (LESSONS 8). A missing or
  unreadable ``RECORD``, one naming no ``www_v2`` file, one naming a ``www_v2``
  file in any form but the plain ``vaelor/www_v2/<name>`` pip writes, or one
  naming a ``www_v2`` file that is not on disk (it describes some other build)
  would each make shipped files look like orphans, so each refuses and says why.
* Links are never followed and never removed. ``vaelor/www_v2`` itself being a
  link is a refusal. On POSIX the walk holds a descriptor for each directory
  and opens the next one with ``O_NOFOLLOW`` and an inode check, so no path is
  re-resolved between the check and the removal. That is defence in depth: the
  services in group ``vaelor-jobs`` run ``ProtectSystem=strict``, so ``/opt``
  is read-only to them today, and the walk does not rely on it.
* It never crosses into another mount: a folder on another device is left
  alone, and on Linux a mount point at or under ``vaelor/www_v2`` (a bind mount
  keeps the device number) refuses the whole sweep.
* Only regular files are removed, plus the folders that removal empties.
  Running it twice removes nothing the second time.
"""

from __future__ import annotations

import csv
import io
import os
import posixpath
import stat
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Callable, FrozenSet, List, Optional, Set

from .build_provenance import DISTRIBUTION

MODULE = "vaelor.www_v2_orphans"
PACKAGE_DIR = "vaelor"
WWW_DIR = "www_v2"
#: How ``RECORD`` names a file under the served frontend.
WWW_PREFIX = PACKAGE_DIR + "/" + WWW_DIR + "/"

OUTCOME_REMOVED = "removed"
OUTCOME_NOTHING = "nothing-to-remove"
OUTCOME_REFUSED = "refused"

NOTHING_REMOVED = "No file was removed."
#: Why the path walk refuses a removal (the descriptor walk cannot reach one).
OUTSIDE = "resolves outside the frontend folder"
MOUNTINFO = "/proc/self/mountinfo"
#: The octal escapes the kernel writes in a mount point (space, tab, newline),
#: and the escaped backslash, which must be undone last.
MOUNT_ESCAPES = (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\"))
#: The one RECORD form a shipped frontend file is accepted in.
WWW_MARKER = "/" + WWW_PREFIX


class MalformedRecord(ValueError):
    """RECORD names a ``www_v2`` file in a form that is not plain and relative."""

#: Directory descriptors with ``*at`` calls: every POSIX box this installs on.
#: The path walk below exists so the decisions can be tested on Windows; it is
#: never what runs on an appliance.
DESCRIPTOR_WALK = (
    hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
    and {os.open, os.stat, os.unlink, os.rmdir} <= os.supports_dir_fd
    and os.listdir in os.supports_fd
)


@dataclass
class Report:
    outcome: str
    removed: List[str] = field(default_factory=list)
    removed_dirs: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    reason: str = ""


def _mentions_www(path: str) -> bool:
    forward = path.replace("\\", "/")
    return any(WWW_MARKER in "/" + form for form in (forward, posixpath.normpath(forward)))


def recorded_www_files(record_text: Optional[str]) -> Optional[FrozenSet[str]]:
    """The ``www_v2``-relative names ``RECORD`` lists, or ``None`` without a text.

    Raises :class:`MalformedRecord` for any entry that names a ``www_v2`` file in
    another form (``./``, ``..``, ``//``, absolute, backslashes). Dropping such
    an entry would remove the file it names, so it refuses instead.
    """
    if record_text is None:
        return None
    names: Set[str] = set()
    for row in csv.reader(io.StringIO(record_text, newline="")):
        if not row or not _mentions_www(row[0]):
            continue
        path = row[0]
        relative = path[len(WWW_PREFIX):] if path.startswith(WWW_PREFIX) else ""
        if ("\\" in path or not relative
                or any(part in ("", ".", "..") for part in relative.split("/"))):
            raise MalformedRecord(path)
        names.add(relative)
    return frozenset(names)


def _is_link(info: os.stat_result) -> bool:
    if stat.S_ISLNK(info.st_mode):
        return True
    # A Windows junction lstat()s as a directory; its reparse bit is the tell.
    attributes = getattr(info, "st_file_attributes", 0) or 0
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _same(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


class _FdDir:
    """A directory held open by descriptor; every child op is relative to it."""

    FLAGS = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))

    def __init__(self, fd: int):
        self.fd = fd
        self.device: Optional[int] = None

    @classmethod
    def open_root(cls, site_root: Path) -> "_FdDir":
        # The site root is the venv's own folder, resolved by the interpreter.
        # Only it is opened by path; every step below it is O_NOFOLLOW.
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
        return cls(os.open(str(site_root), flags))

    def names(self) -> List[str]:
        return os.listdir(self.fd)

    def lstat(self, name: str) -> os.stat_result:
        return os.stat(name, dir_fd=self.fd, follow_symlinks=False)

    def child(self, name: str, seen: os.stat_result) -> Optional["_FdDir"]:
        try:
            fd = os.open(name, self.FLAGS, dir_fd=self.fd)
        except OSError:
            return None  # replaced by a link or removed since it was listed
        if not _same(os.fstat(fd), seen):
            os.close(fd)
            return None
        return _FdDir(fd)

    def confine(self) -> None:
        """Nothing to do: a descriptor walk cannot leave the folder it holds."""

    def unlink(self, name: str) -> None:
        os.unlink(name, dir_fd=self.fd)

    def rmdir(self, name: str) -> None:
        os.rmdir(name, dir_fd=self.fd)

    def close(self) -> None:
        os.close(self.fd)


class _PathDir:
    """The same operations by path, for a platform without descriptor calls."""

    def __init__(self, path: Path, root_real: str):
        self.path = path
        self.root_real = root_real
        self.device: Optional[int] = None

    @classmethod
    def open_root(cls, site_root: Path) -> "_PathDir":
        return cls(site_root, "")

    def names(self) -> List[str]:
        return os.listdir(self.path)

    def lstat(self, name: str) -> os.stat_result:
        return os.lstat(self.path / name)

    def child(self, name: str, seen: os.stat_result) -> Optional["_PathDir"]:
        target = self.path / name
        try:
            now = os.lstat(target)
        except OSError:
            return None
        if _is_link(now) or not stat.S_ISDIR(now.st_mode) or not _same(now, seen):
            return None
        if self.root_real and not self._inside(name):
            return None
        return _PathDir(target, self.root_real)

    def confine(self) -> None:
        """Make this folder the boundary every path below must resolve inside."""
        self.root_real = os.path.realpath(self.path)

    def _inside(self, name: str) -> bool:
        real = os.path.realpath(self.path / name)
        return bool(self.root_real) and real.startswith(self.root_real + os.sep)

    def unlink(self, name: str) -> None:
        if not self._inside(name):
            raise OSError(OUTSIDE)
        os.unlink(self.path / name)

    def rmdir(self, name: str) -> None:
        if not self._inside(name):
            raise OSError(OUTSIDE)
        os.rmdir(self.path / name)

    def close(self) -> None:
        pass


def _open_www(site_root: Path):
    """Descend site-packages -> vaelor -> www_v2 without following a link.

    Returns ``(handle, None)`` or ``(None, reason)``; ``handle.device`` is the
    device ``www_v2`` lives on, which the walk never leaves.
    """
    opener = _FdDir if DESCRIPTOR_WALK else _PathDir
    try:
        handle = opener.open_root(site_root)
    except OSError:
        return None, "The installed package folder could not be opened."
    for name in (PACKAGE_DIR, WWW_DIR):
        try:
            seen = handle.lstat(name)
        except OSError:
            handle.close()
            return None, "The installed package has no {} folder.".format(WWW_PREFIX)
        if _is_link(seen) or not stat.S_ISDIR(seen.st_mode):
            handle.close()
            return None, "{} is a link or not a folder.".format(PACKAGE_DIR + "/" + name)
        child = handle.child(name, seen)
        handle.close()
        if child is None:
            return None, "{} changed while it was being opened.".format(name)
        handle = child
        handle.device = seen.st_dev
    handle.confine()
    return handle, None


def _mounts_under(folder: str) -> Optional[List[str]]:
    """Mount points at or under ``folder`` from ``/proc/self/mountinfo``.

    ``[]`` where there is no such file (not Linux); ``None`` when it exists but
    cannot be read, which the caller treats as unknown, not as none.
    """
    if not os.path.exists(MOUNTINFO):
        return []
    try:
        with open(MOUNTINFO, encoding="utf-8", errors="surrogateescape") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    found = []
    for line in lines:
        fields = line.split(" ")
        if len(fields) < 5:
            continue
        # Field 5 is the mount point, with space, tab, newline and backslash
        # written as three-digit octal escapes.
        point = fields[4]
        for escaped, plain in MOUNT_ESCAPES:
            point = point.replace(escaped, plain)
        if point == folder or point.startswith(folder + os.sep):
            found.append(point)
    return found


def _walk(handle, prefix: str, recorded: FrozenSet[str], report: Report,
          act: bool, found: Set[str], log: Callable[[str], None],
          device: Optional[int] = None) -> bool:
    """Visit one folder. ``act=False`` only inventories regular files into
    ``found``; ``act=True`` removes the unrecorded ones. True when this call
    removed anything below ``handle``."""
    removed_any = False
    for name in sorted(handle.names()):
        relative = prefix + name
        shown = WWW_PREFIX + relative
        try:
            info = handle.lstat(name)
        except OSError:
            continue  # gone since it was listed
        if _is_link(info):
            if act:
                report.skipped.append(shown)
                log("left in place (a link): {}".format(shown))
            continue
        if stat.S_ISDIR(info.st_mode):
            if device is not None and info.st_dev != device:
                if act:
                    report.skipped.append(shown)
                    log("left in place (another filesystem): {}".format(shown))
                continue
            child = handle.child(name, info)
            if child is None:
                if act:
                    report.skipped.append(shown)
                    log("left in place (changed while being read): {}".format(shown))
                continue
            try:
                emptied = _walk(child, relative + "/", recorded, report, act, found,
                                log, device)
                still_holds = bool(child.names())
            finally:
                child.close()
            removed_any = removed_any or emptied
            keeps_recorded = any(path.startswith(relative + "/") for path in recorded)
            if act and emptied and not still_holds and not keeps_recorded:
                try:
                    handle.rmdir(name)
                    report.removed_dirs.append(shown)
                    log("removed empty folder {}".format(shown))
                except OSError as error:
                    report.failed.append(shown)
                    log("could not remove folder {}: {}".format(shown, error))
            continue
        if not stat.S_ISREG(info.st_mode):
            if act:
                report.skipped.append(shown)
                log("left in place (not a regular file): {}".format(shown))
            continue
        found.add(relative)
        if not act or relative in recorded:
            continue
        try:
            handle.unlink(name)
        except OSError as error:
            report.failed.append(shown)
            log("could not remove {}: {}".format(shown, error))
            continue
        report.removed.append(shown)
        removed_any = True
        log("removed {}".format(shown))
    return removed_any


def _refuse(reason: str, log: Callable[[str], None]) -> Report:
    log("Stale frontend files were not cleared: {} {}".format(reason, NOTHING_REMOVED))
    return Report(OUTCOME_REFUSED, reason=reason)


def prune_www_v2(site_root: Path, record_text: Optional[str],
                 log: Callable[[str], None] = print) -> Report:
    """Remove the files under ``site_root/vaelor/www_v2`` that ``record_text``
    (the installed ``RECORD``) does not list. See the module docstring."""
    try:
        recorded = recorded_www_files(record_text)
    except MalformedRecord as entry:
        return _refuse("RECORD names a frontend file in a form other than {}<name>: "
                       "{!r}.".format(WWW_PREFIX, str(entry)), log)
    if recorded is None:
        return _refuse("The installed package's file list (RECORD) could not be read.", log)
    if not recorded:
        return _refuse("RECORD lists no file under {}.".format(WWW_PREFIX), log)
    handle, why = _open_www(site_root)
    if handle is None:
        return _refuse(why, log)
    mounts = _mounts_under(os.path.join(os.path.realpath(site_root), PACKAGE_DIR, WWW_DIR))
    if mounts is None or mounts:
        handle.close()
        return _refuse(
            "Could not read the mount table." if mounts is None else
            "A filesystem is mounted at or under {}: {}.".format(WWW_PREFIX, mounts[0]),
            log,
        )
    report = Report(OUTCOME_NOTHING)
    try:
        found: Set[str] = set()
        _walk(handle, "", recorded, report, False, found, log, handle.device)
        missing = sorted(recorded - found)
        if missing:
            return _refuse(
                "RECORD lists {} file(s) under {} that are not on disk, so it does "
                "not describe this install (first: {}).".format(
                    len(missing), WWW_PREFIX, missing[0]),
                log,
            )
        _walk(handle, "", recorded, report, True, set(), log, handle.device)
    except OSError as error:
        # A folder that could not be listed: a machine fault, said as one, with
        # what was already removed still counted below.
        report.failed.append(WWW_PREFIX)
        log("Clearing stale frontend files stopped early: a folder under {} could "
            "not be read ({}).".format(WWW_PREFIX, error))
    finally:
        handle.close()
    if report.removed or report.removed_dirs:
        report.outcome = OUTCOME_REMOVED
    log("Removed {} file(s) and {} empty folder(s) under {} that this release did "
        "not ship.".format(len(report.removed), len(report.removed_dirs), WWW_PREFIX))
    return report


def prune_installed(distribution=None, log: Callable[[str], None] = print) -> Report:
    """Sweep the installed ``vaelor-control-plane`` distribution's ``www_v2``."""
    if distribution is None:
        try:
            distribution = metadata.distribution(DISTRIBUTION)
        except metadata.PackageNotFoundError:
            return _refuse("The {} distribution is not installed.".format(DISTRIBUTION), log)
    site_root = Path(str(distribution.locate_file("")))
    return prune_www_v2(site_root, distribution.read_text("RECORD"), log)


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m vaelor.www_v2_orphans`` - run by the installer and the
    upgrade broker after the wheel installs. Exit 1 when it refused or a
    removal failed; the callers carry on either way."""
    del argv
    report = prune_installed(log=lambda line: print(line, flush=True))
    return 1 if report.outcome == OUTCOME_REFUSED or report.failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
