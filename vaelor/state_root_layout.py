"""Lay the Vaelor state root out so no service account can re-point a mount source.

ACC-063, and its reviews (B1, S4, S6, R1). The root hardware bridge mounts the
model cache into a root container with the GPU devices, and the vLLM containers
mount the cache, its ``compile`` folder and ``serving-profiles`` read-write.
Docker resolves a ``-v`` source when it runs, so every folder from ``/`` down to
each source must be one only root can rename or replace
(:func:`vaelor.gpu_model_confinement.anchor_refusal`). The layout that gives
that:

* ``/var/lib/vaelor``: ``root:vaelor``, sticky, with an ACL that gives the
  ``vaelor`` account (the control plane) write. Its own files keep working, the
  sticky bit keeps it from renaming anything it does not own, and the vaelor
  GROUP keeps read-only. ``ls -ld`` shows the ACL's mask in the group bits
  (``drwxrwx--T+``, mode ``1770``);
* ``/var/lib/vaelor/models``: ``root:vaelor-jobs 3770``. The executor still
  downloads into it (group write, setgid); nothing but root renames its entries;
* ``models/hub``, ``models/pull``, ``models/compile`` and ``serving-profiles``:
  real ``root:root 0755`` folders - and for the three a root container reads
  weights and kernels from, EVERY entry below them root-owned too, since a
  subtree another account made is consumed by root containers.

**Every step is fd-relative and never follows a link (review R1).** This runs
as root over folders a service account owned until now, so a path-based
``chown``/``chmod`` could be pointed at ``/etc/shadow`` by renaming a folder
and planting a link in its place between the check and the change. So each
folder is opened ``O_DIRECTORY|O_NOFOLLOW`` relative to its parent's open
descriptor, walking down from ``/``; the owner and mode are set on that
descriptor (``fchown``/``fchmod``); the ACL is set on the verified descriptor
(``/proc/self/fd/N``, never a path); and an entry below is re-owned with
``AT_SYMLINK_NOFOLLOW`` only after its folder is root's and not writable by
anyone else, so nothing can be swapped under the walk. A link where a folder
belongs is REFUSED - the layout stops there and the GPU launch keeps refusing -
except a link where one of the four root folders belongs, which is removed
(removing a link never removes what it pointed at) and replaced by a new
folder; if something takes the name again first, that is refused too.

**Who applies it, and when.** The installer runs this module on every install.
An in-product upgrade (`vaelor.appliance_upgrade`) runs the INSTALLED package's
copy right after every ``pip`` reinstall, forward or rollback
(:func:`settle_after_install`), and the hardware bridge asks systemd to run it
once at every start (:func:`start_layout_unit`) - a hot-patched box restarts
the bridge - so no box is left on the old ``vaelor:vaelor 0750`` layout the GPU
launch refuses. It is idempotent: a box already laid out is read and left alone.

**Without POSIX ACLs** (a filesystem that refuses them, no ``setfacl``, or no
``/proc`` to name the descriptor by) the state root is left as it is -
root-owning it would lock the control plane out of its own files - and the
report says so in plain words; everything but the GPU model launch keeps
working, and that launch refuses with :data:`FIX_SENTENCE`.

**Rolling back** to an older package: when the reinstalled package has no copy
of this module (it predates the layout), the state root goes back to the
``vaelor:vaelor 0750`` shape that package expects, ACL removed
(:func:`revert_state_root_layout`); the model folders stay root's, which older
code only reads. A later upgrade, or the next bridge start of a newer package,
lays it out again.

Runs as ROOT only: from the installer, the upgrade broker, and the transient
unit the bridge starts (``python -m vaelor.state_root_layout``).
"""

from __future__ import annotations

import logging
import os
import posixpath
import stat
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional

from .gpu_model_confinement import MODEL_CACHE_DIR, anchor_refusal
from .serving_profiler import SERVING_PROFILES_DIR

LOGGER = logging.getLogger(__name__)

#: The model store and its pull folder. Spelled from the confinement module's
#: store path: the root bridge's argv policy no longer names either, because no
#: argv touches the store any more (VD-143).
MODELS_ROOT = MODEL_CACHE_DIR
PULL_ROOT = MODELS_ROOT + "/pull"
HUB_ROOT = MODELS_ROOT + "/hub"
STATE_ROOT = posixpath.dirname(MODELS_ROOT)
COMPILE_ROOT = MODELS_ROOT + "/compile"
#: The folders a root container writes through, all root's and never links.
ROOT_FOLDERS = (HUB_ROOT, PULL_ROOT, COMPILE_ROOT, SERVING_PROFILES_DIR)
#: The ones whose whole subtree a root container reads: every entry root's.
ROOT_SUBTREES = (HUB_ROOT, PULL_ROOT, COMPILE_ROOT)
#: What a controller vLLM or pull container mounts (`gpu_pool_runtime`'s ``-v``).
MOUNT_SOURCES = (MODELS_ROOT, COMPILE_ROOT, SERVING_PROFILES_DIR)
STATE_ROOT_MODE = 0o1770
MODELS_MODE = 0o3770
ROOT_FOLDER_MODE = 0o755
SETFACL = "/usr/bin/setfacl"
STATE_ROOT_ACL = "u:vaelor:rwx,g::r-x"
VENV_PYTHON = "/opt/vaelor/venv/bin/python"
MODULE = "vaelor.state_root_layout"
#: The one sentence every refusal points at: what fixes the layout, exactly.
FIX_SENTENCE = (
    "To fix it, run 'sudo " + VENV_PYTHON + " -m " + MODULE + "' on this "
    "machine (restarting the vaelor-hardware-bridge service runs it too), then "
    "try again."
)
#: The transient unit the bridge starts; ``--collect`` lets it run again.
LAYOUT_UNIT = "vaelor-state-root-layout"
LAYOUT_COMMAND = (
    "/usr/bin/systemd-run", "--unit=" + LAYOUT_UNIT, "--collect", "--wait",
    "--quiet", VENV_PYTHON, "-m", MODULE,
)

ANCHORED = "anchored"
ACL_UNAVAILABLE = "acl-unavailable"
CACHE_NOT_A_FOLDER = "cache-not-a-folder"
LINK_REFUSED = "link-refused"
NO_STATE_ROOT = "no-state-root"
NO_ACCOUNTS = "no-service-accounts"
#: What :func:`main` prints under each outcome's word.
OUTCOME_SENTENCES = {
    ANCHORED: "The Vaelor state root is laid out: only root can re-point the model folders.",
    ACL_UNAVAILABLE: (
        "The folder " + STATE_ROOT + " does not take POSIX ACLs (or setfacl is "
        "missing), so it was left as it is. Vaelor works, but GPU models are not "
        "launched from it until it sits on a filesystem that takes ACLs (ext4, "
        "xfs or btrfs) with the acl package installed."
    ),
    CACHE_NOT_A_FOLDER: (
        "The model cache " + MODELS_ROOT + " is a link or a file, not a folder, "
        "so it was left as it is; GPU models are not launched from it until it "
        "is a plain folder."
    ),
    LINK_REFUSED: (
        "A link was found (or planted during the run) where a Vaelor folder "
        "belongs, so the layout stopped there and changed nothing through it; "
        "GPU models are not launched until it runs cleanly. Remove the link and "
        "run it again."
    ),
    NO_STATE_ROOT: "There is no " + STATE_ROOT + " on this machine; nothing was changed.",
    NO_ACCOUNTS: "The Vaelor service accounts do not exist on this machine; nothing was changed.",
}


class LinkRefused(Exception):
    """A link (or a non-folder) where a folder must be: the layout stops."""


class Host(NamedTuple):
    """Every read and change the layout makes, fd-relative; the real one is :data:`HOST`.

    ``open_at(dir_fd, name)`` opens a FOLDER without following a link (``dir_fd``
    ``None`` opens ``/``) and raises ``OSError`` for a link or a non-folder;
    ``stat_at`` is an ``lstat`` of one entry of an open folder; ``chown_at``
    re-owns one entry without following it; ``acl(fd, spec)`` sets (``spec``
    given) or removes (``""``) the ACL of the OPEN folder, answering whether it
    could.
    """

    open_at: Callable[[Optional[int], str], int]
    stat_at: Callable[[int, str], Any]
    fstat: Callable[[int], Any]
    fchown: Callable[[int, int, int], None]
    fchmod: Callable[[int, int], None]
    chown_at: Callable[[int, str, int, int], None]
    mkdir_at: Callable[[int, str, int], None]
    unlink_at: Callable[[int, str], None]
    listdir: Callable[[int], List[str]]
    close: Callable[[int], None]
    acl: Callable[[int, str], bool]
    uid: Callable[[str], Optional[int]]
    gid: Callable[[str], Optional[int]]
    #: ``chmod_file_at(dir_fd, name, mode)``: the mode of one REGULAR file, set
    #: through a descriptor opened without following (a link is refused).
    chmod_file_at: Callable[[int, str, int], None] = None  # type: ignore[assignment]


_DIR_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def _open_at(dir_fd: Optional[int], name: str) -> int:
    if dir_fd is None:
        return os.open("/", _DIR_FLAGS)
    return os.open(name, _DIR_FLAGS, dir_fd=dir_fd)


def _acl(fd: int, spec: str) -> bool:
    """``setfacl`` on the open folder itself, named by its descriptor, never a path."""
    if not os.path.isdir("/proc/self/fd"):
        return False
    argv = [SETFACL, "-m", spec] if spec else [SETFACL, "-b"]
    try:
        return subprocess.run(
            argv + ["/proc/self/fd/{}".format(fd)], pass_fds=(fd,),
            capture_output=True, timeout=30, check=False,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _chmod_file_at(dir_fd: int, name: str, mode: int) -> None:
    """``mode`` on regular file ``name`` in ``dir_fd``, through a no-follow descriptor.

    ``fchmodat`` cannot refuse a link on Linux, so the file is OPENED with
    ``O_NOFOLLOW`` (a link fails the open) and re-checked as a regular file
    with one name before ``fchmod`` touches it.
    """
    flags = (
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(name, flags, dir_fd=dir_fd)
    try:
        info = os.fstat(fd)
        if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            os.fchmod(fd, mode)
    finally:
        os.close(fd)


def _run(argv: List[str]) -> int:
    try:
        return subprocess.run(argv, capture_output=True, timeout=120, check=False).returncode
    except (OSError, subprocess.SubprocessError):
        return 127


def _uid(name: str) -> Optional[int]:
    try:
        import pwd
        return pwd.getpwnam(name).pw_uid
    except (ImportError, KeyError):
        return None


def _gid(name: str) -> Optional[int]:
    try:
        import grp
        return grp.getgrnam(name).gr_gid
    except (ImportError, KeyError):
        return None


HOST = Host(
    open_at=_open_at,
    stat_at=lambda fd, name: os.stat(name, dir_fd=fd, follow_symlinks=False),
    fstat=os.fstat,
    fchown=getattr(os, "fchown", None),
    fchmod=getattr(os, "fchmod", None),
    chown_at=lambda fd, name, uid, gid: os.chown(
        name, uid, gid, dir_fd=fd, follow_symlinks=False),
    mkdir_at=lambda fd, name, mode: os.mkdir(name, mode, dir_fd=fd),
    unlink_at=lambda fd, name: os.unlink(name, dir_fd=fd),
    chmod_file_at=_chmod_file_at,
    listdir=os.listdir, close=os.close, acl=_acl, uid=_uid, gid=_gid,
)


class _Walk:
    """The descriptors one run opened, closed together; and what it changed."""

    def __init__(self, host: Host):
        self.host = host
        self.fds: List[int] = []
        self.changed: List[str] = []

    def open(self, dir_fd: Optional[int], name: str) -> int:
        try:
            fd = self.host.open_at(dir_fd, name)
        except FileNotFoundError:
            raise
        except OSError as error:
            raise LinkRefused(name) from error
        self.fds.append(fd)
        return fd

    def open_path(self, path: str) -> int:
        """``path``'s folder, opened one part at a time from ``/``, never through a link."""
        fd = self.open(None, "/")
        for part in [piece for piece in path.split("/") if piece]:
            fd = self.open(fd, part)
        return fd

    def own(self, fd: int, uid: int, gid: int, mode: int, label: str) -> None:
        """Owner and mode on the open folder itself."""
        info = self.host.fstat(fd)
        if (info.st_uid, info.st_gid) != (uid, gid):
            self.host.fchown(fd, uid, gid)
            self.changed.append("owned " + label)
        if stat.S_IMODE(info.st_mode) != mode:
            self.host.fchmod(fd, mode)
            self.changed.append("moded " + label)

    def close(self) -> None:
        for fd in reversed(self.fds):
            try:
                self.host.close(fd)
            except OSError:
                pass
        self.fds = []


def _root_folder(walk: _Walk, parent: int, name: str, label: str) -> int:
    """A real ``root:root 0755`` folder ``name`` in ``parent``, open; or :class:`LinkRefused`."""
    host = walk.host
    try:
        info = host.stat_at(parent, name)
    except FileNotFoundError:
        info = None
    if info is not None and not stat.S_ISDIR(info.st_mode):
        try:
            host.unlink_at(parent, name)  # the link itself; never what it names
        except IsADirectoryError as error:
            # A folder was put there between the look and the unlink: refuse
            # rather than trace back, and let the next run look again.
            raise LinkRefused(label) from error
        walk.changed.append("replaced " + label)
        info = None
    if info is None:
        try:
            host.mkdir_at(parent, name, ROOT_FOLDER_MODE)
        except FileExistsError as error:
            raise LinkRefused(label) from error  # taken again in the gap
        walk.changed.append("created " + label)
    fd = walk.open(parent, name)
    walk.own(fd, 0, 0, ROOT_FOLDER_MODE, label)
    return fd


def _own_subtree(walk: _Walk, fd: int, label: str) -> None:
    """Every entry below the open, root-owned, non-writable folder ``fd``: root's.

    Top-down, so each folder is root's and writable by nobody else BEFORE its
    entries are read: nothing can be renamed or planted under the walk. A
    sub-folder is opened without following; any other entry (a file, or a
    link the model cache legitimately holds) is re-owned itself, never what
    it points at. An entry that already belongs to root is left alone.

    **A regular file with more than one name is never re-owned.** Where the
    kernel allows it (``fs.protected_hardlinks=0``), an account can hard-link
    a file it does not own into the store; the link IS that file, so a chown
    here would re-own it wherever else it lives - ``/etc/shadow`` turned
    ``root:root`` and its setgid bits lost. Such an entry is left exactly as
    it is and reported, and the root bridge refuses to start a container on
    a tree holding one (`bridge_safe_fs.first_foreign_entry`).

    Every other regular file also loses group and other write: the bridge
    refuses to start a container on a file another account can change, so
    leaving one writable would block every Mode B start with a refusal the
    installer claims to fix.
    """
    host = walk.host
    for name in sorted(host.listdir(fd)):
        info = host.stat_at(fd, name)
        path = label + "/" + name
        if stat.S_ISDIR(info.st_mode):
            child = walk.open(fd, name)
            mode = stat.S_IMODE(info.st_mode) & ~0o022
            walk.own(child, 0, 0, mode, path)
            _own_subtree(walk, child, path)
            walk.fds.remove(child)
            host.close(child)
        elif stat.S_ISREG(info.st_mode) and getattr(info, "st_nlink", 1) > 1:
            LOGGER.warning(
                "The state-root layout left %s alone: it is a hard link to a "
                "file elsewhere, and re-owning it would re-own that file.", path,
            )
            walk.changed.append("left hard-linked " + path)
        else:
            if info.st_uid != 0:
                host.chown_at(fd, name, 0, 0)
                walk.changed.append("owned " + path)
            if stat.S_ISREG(info.st_mode) and info.st_mode & 0o022:
                # The bridge refuses a root container on a file others can
                # write, so the layout takes that write away too - on the file
                # itself, opened without following, never a path.
                host.chmod_file_at(fd, name, stat.S_IMODE(info.st_mode) & ~0o022)
                walk.changed.append("moded " + path)


def apply_state_root_layout(host: Host = HOST, root: str = STATE_ROOT) -> Dict[str, Any]:
    """Lay the state root out as the module docstring says; report what changed."""
    vaelor_gid, jobs_gid = host.gid("vaelor"), host.gid("vaelor-jobs")
    if None in (host.uid("vaelor"), vaelor_gid, jobs_gid):
        return {"state": NO_ACCOUNTS, "changed": []}
    walk = _Walk(host)
    try:
        try:
            root_fd = walk.open_path(root)
        except FileNotFoundError:
            return {"state": NO_STATE_ROOT, "changed": walk.changed}
        # The ACL FIRST, while the control plane may still own the folder, so
        # it never loses write; then owner and mode. chmod on a folder with an
        # ACL sets the MASK and keeps the group entry (r-x) the ACL just wrote.
        if not host.acl(root_fd, STATE_ROOT_ACL):
            LOGGER.warning("%s", OUTCOME_SENTENCES[ACL_UNAVAILABLE])
            return {"state": ACL_UNAVAILABLE, "changed": walk.changed}
        walk.own(root_fd, 0, vaelor_gid, STATE_ROOT_MODE, root)
        models = posixpath.basename(MODELS_ROOT)
        try:
            info = host.stat_at(root_fd, models)
        except FileNotFoundError:
            try:
                host.mkdir_at(root_fd, models, MODELS_MODE)
            except FileExistsError as error:
                raise LinkRefused(root + "/" + models) from error
            walk.changed.append("created " + root + "/" + models)
        else:
            if not stat.S_ISDIR(info.st_mode):
                return {"state": CACHE_NOT_A_FOLDER, "changed": walk.changed}
        models_fd = walk.open(root_fd, models)
        walk.own(models_fd, 0, jobs_gid, MODELS_MODE, root + "/" + models)
        for folder in ROOT_FOLDERS:
            parent, name = posixpath.split(folder)
            parent_fd = models_fd if parent == MODELS_ROOT else root_fd
            label = root + folder[len(STATE_ROOT):]
            fd = _root_folder(walk, parent_fd, name, label)
            if folder in ROOT_SUBTREES:
                _own_subtree(walk, fd, label)
        return {"state": ANCHORED, "changed": walk.changed}
    except LinkRefused as refused:
        LOGGER.warning("The state-root layout refused %s: %s", refused, OUTCOME_SENTENCES[LINK_REFUSED])
        return {"state": LINK_REFUSED, "changed": walk.changed, "refused": str(refused)}
    finally:
        walk.close()


def revert_state_root_layout(host: Host = HOST, root: str = STATE_ROOT) -> Dict[str, Any]:
    """The state root back to ``vaelor:vaelor 0750``, ACL removed, for an older package."""
    uid, gid = host.uid("vaelor"), host.gid("vaelor")
    if uid is None or gid is None:
        return {"state": NO_ACCOUNTS}
    walk = _Walk(host)
    try:
        fd = walk.open_path(root)
        host.acl(fd, "")
        walk.own(fd, uid, gid, 0o750, root)
        return {"state": "reverted"}
    except (LinkRefused, FileNotFoundError):
        return {"state": LINK_REFUSED}
    finally:
        walk.close()


def settle_after_install(
    run: Callable[[List[str]], Any], package_dir: Path, python: str = VENV_PYTHON,
    host: Host = HOST,
) -> str:
    """After a ``pip`` reinstall: the INSTALLED package's layout, whichever it is.

    The broker doing the install still runs the code it started with, so the
    rule that applies is read from the package now on disk: its own copy of
    this module is run, and a package without one (older than the layout) gets
    the state root it expects back.
    """
    if (Path(package_dir) / "state_root_layout.py").is_file():
        run([python, "-m", MODULE])
        return "applied"
    revert_state_root_layout(host)
    return "reverted"


def mount_refusal(lstat: Callable[[str], Any] = os.lstat) -> Optional[str]:
    """Why the bridge must not start a managed unit now, or ``None`` (review S6).

    Asked by the root bridge's unit install (`bridge_managed_units`) before it
    writes or starts anything - the only place a controller vLLM or pull unit
    can be started since the argv policy lost ``systemctl start``/``enable``.
    Every unit mounts :data:`MOUNT_SOURCES`; a source that is missing counts as
    refused, because docker would make it after whoever can write its folder
    had the chance to plant a link there. The hub, where the weights a root
    container loads live, must be anchored too when it exists (a first pull
    makes it, root-owned, inside an anchored store).
    """
    sources = list(MOUNT_SOURCES)
    try:
        lstat(HUB_ROOT)
        sources.append(HUB_ROOT)
    except FileNotFoundError:
        pass
    except OSError:
        sources.append(HUB_ROOT)
    for source in sources:
        reason = anchor_refusal(source, lstat)
        if reason is not None:
            return (
                "{} The model server is not started until only root can change "
                "the folders it mounts. {}".format(reason, FIX_SENTENCE)
            )
    return None


def start_layout_unit(run: Callable[[List[str]], int] = _run) -> int:
    """Ask systemd to apply the layout outside the caller's sandbox (the bridge)."""
    code = run(list(LAYOUT_COMMAND))
    if code != 0:
        LOGGER.warning(
            "The state-root layout unit did not finish cleanly (exit %s); run "
            "'journalctl -u %s' to see why.", code, LAYOUT_UNIT,
        )
    return code


def main() -> None:
    report = apply_state_root_layout()
    print(report["state"])
    print(OUTCOME_SENTENCES[report["state"]])
    for change in report["changed"]:
        print(change)


if __name__ == "__main__":
    main()
