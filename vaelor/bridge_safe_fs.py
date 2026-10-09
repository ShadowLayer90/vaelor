"""File operations the ROOT bridge makes without following a single link.

VD-143. The root hardware bridge writes on the controller in two places another
account can reach: the model store (``/var/lib/vaelor/models``, group-writable
by every ``vaelor-jobs`` account so the workload executor can cache models) and
the units it installs. Spawning ``rm -rf``, ``install -d``, ``tee`` or ``cat``
as root there - what the bridge used to do - follows every symbolic link on the
path, so an account that could plant ``hub -> /etc/systemd/system`` in the
store turned a root ``rm -rf <store>/hub/x`` into the removal of a unit, and a
link planted as a progress file turned a root ``cat`` into a read of any file.

So every operation here walks from ``/`` one component at a time with
``O_NOFOLLOW | O_DIRECTORY``, and then works RELATIVE to the directory it
opened (``dir_fd``), never through a path string a second time. A link anywhere
on the way - an intermediate directory or the final name - is refused
(``ELOOP``/``ENOTDIR``), not followed; a removal unlinks a link rather than
descending through it; a write lands in a fresh ``O_EXCL`` temporary and is
renamed over the target, which replaces a link planted there instead of
writing through it. Nothing can resolve outside the directory that was opened.

POSIX only: the bridge is a Linux root service. On a platform without
``O_NOFOLLOW`` or ``dir_fd`` support every function raises, which is the honest
answer for a machine that cannot make the guarantee.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import stat
from typing import FrozenSet, Iterator, Optional, Tuple

#: The flags every directory on a walk is opened with: a directory, never a
#: link, never inherited by a child the bridge spawns.
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_READ_FLAGS = (
    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)
_FILE_CREATE_FLAGS = (
    os.O_WRONLY | os.O_CREAT | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)

#: Mode bits that let an account other than the owner change a directory's
#: entries or a file's bytes.
_GROUP_OR_OTHER_WRITE = stat.S_IWGRP | stat.S_IWOTH


def supported() -> bool:
    """Whether this platform can open relative to a directory without following links."""
    return (
        hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
        and os.open in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
    )


def _require_support() -> None:
    if not supported():
        raise OSError(
            "This machine cannot open files without following links, so the "
            "hardware bridge will not touch the model store here."
        )


def _component(name: str) -> str:
    """One path component, or a refusal: never empty, ``.``, ``..`` or a path."""
    text = str(name)
    if not text or text in {".", ".."} or "/" in text or "\x00" in text:
        raise ValueError("A path component may not be empty, '.', '..' or contain '/'.")
    return text


def open_child_directory(parent: int, name: str) -> int:
    """A descriptor for directory ``name`` inside ``parent``, refused if it is a link."""
    _require_support()
    return os.open(_component(name), _DIRECTORY_FLAGS, dir_fd=parent)


@contextlib.contextmanager
def directory(path: str) -> Iterator[int]:
    """Open the absolute directory ``path`` component by component, following no link.

    Yields the descriptor and closes it (and every descriptor on the walk).
    """
    _require_support()
    text = str(path)
    if not text.startswith("/"):
        raise ValueError("The bridge opens absolute directories only.")
    current = os.open("/", _DIRECTORY_FLAGS)
    try:
        for name in [part for part in text.split("/") if part]:
            child = open_child_directory(current, name)
            os.close(current)
            current = child
        yield current
    finally:
        os.close(current)


def ensure_root_directory(
    parent: int, name: str, *, mode: int = 0o755, owner: Tuple[int, int] = (0, 0),
) -> int:
    """A descriptor for directory ``name`` in ``parent``, root-owned at ``mode``.

    Made if absent. An existing one that is a link is refused (the open does
    not follow it); one owned by another account, or writable by one, is taken
    back - ``fchown``/``fchmod`` on the descriptor just opened, so it is the
    directory that was checked and not whatever a rename put at the name since.
    That is what the ``install -d -m 0755`` this replaces did to the mode.

    **Only the folder itself is converged, never what is inside it.** Taking a
    folder back stops new entries from other accounts; it does nothing about
    entries they already made - a compiled kernel, a cache file, a weight -
    which a root container would still read and run. A caller that hands the
    folder to a root container must ALSO prove its contents are root's, with
    :func:`first_foreign_entry`, and refuse when they are not. ``owner`` is
    ``(uid, gid)``, root's unless a test that cannot run as root says otherwise.
    """
    try:
        handle = open_child_directory(parent, name)
    except FileNotFoundError:
        try:
            os.mkdir(_component(name), mode, dir_fd=parent)
        except FileExistsError:
            pass
        handle = open_child_directory(parent, name)
    try:
        status = os.fstat(handle)
        if (status.st_uid, status.st_gid) != tuple(owner):
            os.fchown(handle, owner[0], owner[1])
        if stat.S_IMODE(status.st_mode) != mode:
            os.fchmod(handle, mode)
    except BaseException:
        os.close(handle)
        raise
    return handle


def write_file(parent: int, name: str, text: str, *, mode: int = 0o644) -> None:
    """Replace file ``name`` in ``parent`` with ``text``, atomically, following no link.

    The bytes go to a fresh ``O_EXCL | O_NOFOLLOW`` temporary beside the target
    and are renamed over it. A rename replaces whatever entry holds the name -
    a planted link included - rather than writing through it, and a reader
    never sees half a file.
    """
    target = _component(name)
    temporary = ".{}.{}".format(target, secrets.token_hex(6))
    handle = os.open(temporary, _FILE_CREATE_FLAGS, mode, dir_fd=parent)
    try:
        try:
            os.fchmod(handle, mode)
            data = text.encode("utf-8")
            written = 0
            while written < len(data):
                written += os.write(handle, data[written:])
            os.fsync(handle)
        finally:
            os.close(handle)
        os.replace(temporary, target, src_dir_fd=parent, dst_dir_fd=parent)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary, dir_fd=parent)
        raise


def read_file(
    parent: int, name: str, *, limit: int,
    owners: Optional[FrozenSet[int]] = None,
) -> str:
    """The text of regular file ``name`` in ``parent``, at most ``limit`` bytes.

    A link is refused by the open itself; anything that opens but is not a
    regular file (a FIFO would block a root reader forever, a device is not a
    record) is refused before a byte is read. With ``owners``, the file must
    also belong to one of them and have exactly one name: a hard link another
    account made to a root file elsewhere shares that file's owner, and its
    second name is what gives it away.
    """
    handle = os.open(
        _component(name), _FILE_READ_FLAGS | getattr(os, "O_NONBLOCK", 0),
        dir_fd=parent,
    )
    try:
        status = os.fstat(handle)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("That record is not a regular file.")
        if owners is not None and (
            status.st_uid not in owners or status.st_nlink != 1
        ):
            raise ValueError("That record was not written by this machine's root account.")
        data = b""
        while len(data) <= limit:
            chunk = os.read(handle, limit + 1 - len(data))
            if not chunk:
                break
            data += chunk
    finally:
        os.close(handle)
    if len(data) > limit:
        raise ValueError("That record is larger than the bridge will read.")
    return data.decode("utf-8", errors="replace")


def remove_file(parent: int, name: str) -> bool:
    """Unlink entry ``name`` in ``parent`` if present; a link is unlinked, never followed.

    A directory at the name is refused rather than removed: this removes one
    file, and a caller that means a tree says so with :func:`remove_tree`.
    """
    target = _component(name)
    try:
        status = os.stat(target, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if stat.S_ISDIR(status.st_mode):
        raise IsADirectoryError("Refusing to remove a directory as a file.")
    os.unlink(target, dir_fd=parent)
    return True


def remove_tree(parent: int, name: str) -> bool:
    """Remove ``name`` in ``parent`` and everything beneath it, following no link.

    Each directory is opened ``O_NOFOLLOW`` relative to the one above it and
    emptied through its own descriptor, so a link met anywhere in the tree is
    unlinked as an entry and its target is never visited. ``False`` when the
    name was already absent.
    """
    target = _component(name)
    try:
        status = os.stat(target, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(status.st_mode):
        os.unlink(target, dir_fd=parent)
        return True
    handle = open_child_directory(parent, target)
    try:
        for entry in list(os.scandir(handle)):
            remove_tree(handle, entry.name)
    finally:
        os.close(handle)
    os.rmdir(target, dir_fd=parent)
    return True


def is_sealed_directory(handle: int, *, owners: FrozenSet[int] = frozenset({0})) -> bool:
    """Whether the open directory ``handle`` is one only root can change.

    Owned by root and writable by neither group nor other: the shape the pull,
    compile and hub folders are given, so nothing but root adds, renames or
    removes an entry in them.
    """
    status = os.fstat(handle)
    return (
        stat.S_ISDIR(status.st_mode)
        and status.st_uid in owners
        and not status.st_mode & _GROUP_OR_OTHER_WRITE
    )


def first_foreign_entry(
    parent: int, name: str, *, owners: FrozenSet[int] = frozenset({0}),
    _prefix: str = "",
) -> Optional[str]:
    """The first entry at or below ``name`` another account made or can change.

    ``None`` when ``name`` is absent, or when it and everything beneath it are
    root's: owned by root (links included - a link another account made is
    its handiwork whatever it points at), writable by neither group nor other
    (links aside, whose mode means nothing), a regular file with exactly one
    name (a hard link is a file that also lives elsewhere), and a plain
    directory, file or link (a FIFO, socket or device has no place in a model
    or compile cache).

    One of two no-follow walks of these folders: `state_root_layout`
    converges them to root at install and upgrade through its own ``Host``
    seam (review S6 - the two were compared and follow the same discipline;
    they differ in purpose, converge there and refuse here, so each keeps its
    own walk rather than a third shared one).
    Otherwise the entry's path below ``parent``, for the refusal to name.

    Walked like :func:`remove_tree`: every directory opened ``O_NOFOLLOW``
    relative to the one above it, so a link is judged as an entry and never
    followed. This is the check a root container's mount needs, because
    :func:`ensure_root_directory` converges only the folder and an account
    that wrote into it before it was taken back still owns what it wrote.
    """
    target = _component(name)
    relative = _prefix + target
    try:
        status = os.stat(target, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if status.st_uid not in owners:
        return relative
    if stat.S_ISLNK(status.st_mode):
        return None
    if status.st_mode & _GROUP_OR_OTHER_WRITE:
        return relative
    if stat.S_ISREG(status.st_mode):
        # A second name is a hard link, and a hard link is a file that lives
        # somewhere else too - whoever can write it there writes it here.
        return relative if status.st_nlink > 1 else None
    if not stat.S_ISDIR(status.st_mode):
        return relative
    handle = open_child_directory(parent, target)
    try:
        for entry in sorted(os.scandir(handle), key=lambda item: item.name):
            found = first_foreign_entry(
                handle, entry.name, owners=owners, _prefix=relative + "/",
            )
            if found is not None:
                return found
    finally:
        os.close(handle)
    return None


def is_hard_link_at(parent: int, relative: str) -> bool:
    """Whether ``relative`` below ``parent`` is a regular file with a second name.

    Walked a folder at a time without following, so the answer is about the
    entry :func:`first_foreign_entry` named; ``False`` if it went away.
    """
    *folders, leaf = relative.split("/")
    handles = []
    try:
        current = parent
        for name in folders:
            current = open_child_directory(current, name)
            handles.append(current)
        status = os.stat(_component(leaf), dir_fd=current, follow_symlinks=False)
    except (OSError, ValueError):
        return False
    finally:
        for handle in handles:
            os.close(handle)
    return stat.S_ISREG(status.st_mode) and status.st_nlink > 1


def is_root_controlled(path: str, *, owners: FrozenSet[int] = frozenset({0})) -> bool:
    """Whether no account but root can change ``path`` or swap anything above it.

    Walked with no link followed, so the answer is about the file a root
    process would actually open at this moment. Every directory on the way is
    owned by root and either writable by nobody else, or sticky with the next
    entry root's own (``/tmp``'s shape: others may add entries but cannot
    rename or remove root's); the file itself is a regular, root-owned file
    nobody else can write. The bridge asks this of the pull program before it
    mounts it into a root container (VD-143). ``owners`` exists for the tests,
    which cannot run as root; the bridge never passes it.
    """
    text = str(path)
    parts = [part for part in text.split("/") if part]
    if not text.startswith("/") or not parts:
        return False
    *folders, leaf = parts

    def sealed(folder: os.stat_result, entry_owner: int) -> bool:
        if folder.st_uid not in owners:
            return False
        if not folder.st_mode & _GROUP_OR_OTHER_WRITE:
            return True
        return bool(folder.st_mode & stat.S_ISVTX) and entry_owner in owners

    try:
        _require_support()
        current = os.open("/", _DIRECTORY_FLAGS)
        try:
            current_status = os.fstat(current)
            for name in folders:
                child = open_child_directory(current, name)
                child_status = os.fstat(child)
                os.close(current)
                current = child
                if not sealed(current_status, child_status.st_uid):
                    return False
                current_status = child_status
            handle = os.open(_component(leaf), _FILE_READ_FLAGS, dir_fd=current)
            try:
                status = os.fstat(handle)
            finally:
                os.close(handle)
        finally:
            os.close(current)
    except (OSError, ValueError):
        return False
    return (
        sealed(current_status, status.st_uid)
        and stat.S_ISREG(status.st_mode)
        and status.st_uid in owners
        and not status.st_mode & _GROUP_OR_OTHER_WRITE
    )
