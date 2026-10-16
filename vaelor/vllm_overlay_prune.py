"""Remove the vLLM attention overlays and expert-kernel plugins a release has retired (VD-216).

The 4-bit guesser's build folders (VD-216 item 2, `vllm_guesser_build`) are
pruned by the same rules: a key in ``RETIRED_GUESSERS``, holding only the
names its build writes, never one a unit, a mount or a record names.

Every release that changes the patched attention file or the expert-kernel
plugin makes a NEW key (`vllm_overlays.SPLIT_KV_VERIFY`,
`vllm_moe_kernels.EXPERT_KERNEL`) and maps the old ones to it
(`vllm_overlays.RETIRED_OVERLAYS`, `vllm_moe_kernels.RETIRED_KEYS`), but each
old key's folder under `vllm_overlays.OVERLAY_ROOT` stays on disk, and a
worker keeps every per-hash attention copy it was ever sent
(``vllm-overlay-<sha256[:16]>.py`` beside the entry program,
`vllm_overlays.worker_path`). This removes them, as root, on the controller
(the installer: ``python -m vaelor.vllm_overlay_prune apply``) and on each
worker (the controller, over SSH, after it installs that worker's serving
unit: :func:`prune_worker`).

**What it may remove - nothing else, ever:**

* a folder under the overlay root named by a RETIRED key in those two tables
  (never the current key, never a name the tables do not hold), and only when
  everything in it is a name Vaelor writes there for that kind of folder
  (:func:`shapes`) - one stray entry keeps the whole folder;
* a ``vllm-overlay-<16 hex>.py`` in the model store's ``pull`` folder whose
  hex is no current overlay's built hash AND whose content's sha256 begins
  with that hex - so it is a copy Vaelor wrote under its content's name, not
  any file that happens to look like one.

**What keeps a candidate** (LESSONS 22: a state is left only by an owner
action that keeps what is in use): its name in the text of any systemd unit
in :data:`UNIT_FOLDERS` (one level of drop-in and ``.wants`` folders included,
a folder that is a link listed like one; a unit that is a link is read where
it points), in the mount table of any running process's mount namespace (a
container started from an older unit still holds its mount after the unit is
rewritten), or in any deployment record (the controller's cluster store); and
for a candidate FOLDER, any mount that shows it through a folder above it
(the overlay root, ``/var/lib/vaelor``, a filesystem's top) at another path,
or at its own path inside a namespace whose ``/`` is not the host's (a
container's ``-v X:X``). What that does NOT see, by design: a mount at or
below snap's :data:`SNAP_HOSTFS` (every snap namespace shows the host's whole
``/`` there; counting it would keep every folder for ever), and a same-path
mount in a namespace whose ``/`` is the host's (the machine's own mounts, a
service sandbox). A ``docker -v /:/host`` container DOES keep every folder
while it runs. A per-hash copy is kept only by its own name: the serving
container mounts the whole model store, and a copy is read once, at a
container's start, through its own mount. If any of those
cannot be read, nothing is removed and the reason is said (LESSONS 8: an
unread reference is not an absent one).

**The offline fallback** (`vllm_overlays.fetched_diff`): the build falls back
to a retired folder's ``upstream.diff`` when the current folder has none (an
upgrade that could not fetch). Before any retired overlay folder is removed,
the current folder must hold a diff whose attention-file hunks are the pinned
ones; if it does not, the newest retired diff that is pinned is copied there
first, and if that cannot be done, that retired folder is kept and says why.

**The root boundary** (#178, LESSONS 18): every folder on each path is opened
from ``/`` with ``O_NOFOLLOW | O_DIRECTORY`` and must be root's and writable
by nobody else (or sticky with the next entry root's), and everything after
is done RELATIVE to descriptors so opened - so no non-root account can swap
an entry between the check and the removal, and a link anywhere is refused or
unlinked as an entry, never followed. The remaining window is root's own
writers: this release never renders a retired key or copy (both tables map
straight to the current row), so no new reference to a candidate can appear
while this runs. It is bounded (:data:`PRUNE_SECONDS`) and holds no lock: it
runs in the installer or the controller's executor, never inside the root
bridge, so the bridge lock is never held across it.

The work itself is ONE program text (:data:`PRUNE_PROGRAM`), run in-process
on the controller and as ``python3 -c`` on a worker, which has no Vaelor
package - the same arrangement as `vllm_moe_kernels.TREE_LISTER`.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence

from .vllm_moe_kernels import DIST_INFO, PLUGIN_FILES, RETIRED_KEYS, MOE_KERNELS
from .vllm_overlays import (
    DIFF_NAME, BUILT_NAME, OVERLAY_ROOT, OVERLAYS, RETIRED_OVERLAYS, root_controlled_bytes,
)

LOG = logging.getLogger(__name__)

#: The longest one prune may take on one machine, start to finish.
PRUNE_SECONDS = 30

#: How the installer's lines name the machine it runs on.
CONTROLLER = "the controller here"

#: The model store's folder a worker keeps its per-hash attention copies in.
PULL_FOLDER = "/var/lib/vaelor/models/pull"

#: Every folder whose unit files are searched for a candidate's name: systemd's
#: unit search path for the system manager, generated and transient units
#: included (``systemd.unit(5)``). A missing folder is skipped.
UNIT_FOLDERS = (
    "/etc/systemd/system.control", "/run/systemd/system.control", "/run/systemd/transient",
    "/run/systemd/generator.early", "/etc/systemd/system", "/run/systemd/system",
    "/run/systemd/generator", "/usr/local/lib/systemd/system", "/usr/lib/systemd/system",
    "/run/systemd/generator.late",
)

#: The program both sides run. Reads its plan as JSON (argv[1]) and prints
#: its result as JSON; in-process, :func:`run_program` calls ``prune``.
PRUNE_PROGRAM = r'''
import hashlib, json, os, re, stat, sys, time

DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
OTHERS = stat.S_IWGRP | stat.S_IWOTH
COPY = re.compile(r"vllm-overlay-([0-9a-f]{16})\.py")


class Stop(Exception):
    """Nothing is removed: the reason is said."""


class Keep(Exception):
    """This candidate stays: the reason is said."""


def sealed_folder(path, owners):
    """A descriptor for ``path`` only root can change, walked from / with no link followed; None if absent."""
    fd = os.open("/", DIR_FLAGS)
    try:
        status = os.fstat(fd)
        for name in [part for part in path.split("/") if part]:
            try:
                child = os.open(name, DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                return None
            below = os.fstat(child)
            os.close(fd)
            fd = child
            if status.st_uid not in owners or (status.st_mode & OTHERS and not (
                    status.st_mode & stat.S_ISVTX and below.st_uid in owners)):
                raise Stop("a folder above {} can be changed by an account other than root".format(path))
            status = below
        if status.st_uid not in owners or status.st_mode & OTHERS:
            raise Stop("{} can be changed by an account other than root".format(path))
        result, fd = fd, None
        return result
    finally:
        if fd is not None:
            os.close(fd)


def read_at(parent, name, limit):
    handle = os.open(name, FILE_FLAGS, dir_fd=parent)
    try:
        data = os.read(handle, limit + 1)
    finally:
        os.close(handle)
    if len(data) > limit:
        raise Stop("{} is larger than the prune reads".format(name))
    return data.decode("utf-8", "replace")


def open_linked(parent, name, label):
    """A descriptor for what the unit link ``name`` points at; None when it points nowhere."""
    try:
        return os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=parent)
    except FileNotFoundError:
        return None  # a dangling link names nothing
    except OSError as error:
        raise Stop("the unit link {} could not be read ({})".format(label, error.strerror or error))


def read_handle(handle, label, limit):
    data = os.read(handle, limit + 1)
    if len(data) > limit:
        raise Stop("{} is larger than the prune reads".format(label))
    return data.decode("utf-8", "replace")


def read_linked(parent, name, limit, label):
    """The text a unit LINK points at, read (never trusted for anything else); None if not a file."""
    handle = open_linked(parent, name, label)
    if handle is None:
        return None
    try:
        if not stat.S_ISREG(os.fstat(handle).st_mode):
            return None
        return read_handle(handle, label, limit)
    finally:
        os.close(handle)


def unit_text(parent, name, status, label):
    if stat.S_ISREG(status.st_mode):
        return read_at(parent, name, 1 << 20)
    if stat.S_ISLNK(status.st_mode):
        return read_linked(parent, name, 1 << 20, label)
    return None


def drop_ins(handle, label, clock):
    """One level of a drop-in or ``.wants`` folder: each file, and each link read where it points."""
    for leaf in sorted(os.listdir(handle)):
        clock()
        where = label + "/" + leaf
        text = unit_text(handle, leaf, os.stat(leaf, dir_fd=handle, follow_symlinks=False), where)
        if text is not None:
            yield where, text


def unit_texts(folders, clock):
    """Every unit file's text in ``folders`` and one level of drop-in folders.

    A unit that is a link - linked in from anywhere - is read where it points,
    and a drop-in folder that is a link to a folder is listed like one, as
    systemd does.
    """
    for folder in folders:
        try:
            top = os.open(folder, DIR_FLAGS)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise Stop("the unit folder {} could not be read ({})".format(folder, error.strerror))
        try:
            for name in sorted(os.listdir(top)):
                clock()
                label = folder + "/" + name
                status = os.stat(name, dir_fd=top, follow_symlinks=False)
                if stat.S_ISDIR(status.st_mode):
                    inner = os.open(name, DIR_FLAGS, dir_fd=top)
                    try:
                        yield from drop_ins(inner, label, clock)
                    finally:
                        os.close(inner)
                elif stat.S_ISLNK(status.st_mode):
                    handle = open_linked(top, name, label)
                    if handle is None:
                        continue
                    try:
                        kind = os.fstat(handle).st_mode
                        if stat.S_ISDIR(kind):
                            yield from drop_ins(handle, label, clock)
                        elif stat.S_ISREG(kind):
                            yield label, read_handle(handle, label, 1 << 20)
                    finally:
                        os.close(handle)
                elif stat.S_ISREG(status.st_mode):
                    yield label, read_at(top, name, 1 << 20)
        finally:
            os.close(top)


def mount_tables(clock, result):
    """Each running mount namespace's mount table, once per namespace.

    A process whose namespace this account may not read (only when it is not
    root) stops a removal; a dry run says how many it could not see.
    """
    seen, read, denied = set(), 0, 0
    for pid in sorted(name for name in os.listdir("/proc") if name.isdigit()):
        clock()
        try:
            space = os.readlink("/proc/{}/ns/mnt".format(pid))
            if space in seen:
                continue
            with open("/proc/{}/mountinfo".format(pid), "rb") as handle:
                text = handle.read(8 << 20).decode("utf-8", "replace")
        except PermissionError:
            denied += 1
            continue
        except OSError:
            continue  # the process ended, or is a kernel thread
        seen.add(space)
        read += 1
        yield "the mounts of process {}".format(pid), text
    if not read:
        raise Stop("no running process's mounts could be read")
    if denied and result["apply"]:
        raise Stop("the mounts of {} processes could not be read (not root)".format(denied))
    if denied:
        result["note"] = ("not root: the mounts of {} processes were not read, so a container "
                          "of another account could still use a listed path".format(denied))


def unescape(field):
    return re.sub(r"\\([0-7]{3})", lambda found: chr(int(found.group(1), 8)), field)


def visible_at(source, target, inner):
    """Where a mount of filesystem path ``source`` at ``target`` shows ``inner``; None if not below it."""
    base = source.rstrip("/")
    if inner != source and not inner.startswith(base + "/"):
        return None
    return target.rstrip("/") + inner[len(base):]


def inside_path(path, device, table):
    """``path`` as its own filesystem spells it (mountinfo's root field), from THIS namespace's table."""
    best = None
    for line in table.splitlines():
        fields = line.split()
        if len(fields) < 5 or fields[2] != device:
            continue
        source, target = unescape(fields[3]), unescape(fields[4])
        if target == path or path.startswith(target.rstrip("/") + "/"):
            if best is None or len(target) > len(best[1]):
                best = (source, target)
    if best is None:
        return path
    return best[0].rstrip("/") + path[len(best[1].rstrip("/")):]


#: Where snap shows the host's whole ``/`` in every snap's namespace.
SNAP_HOSTFS = "/var/lib/snapd/hostfs"


def root_of(table):
    """(device, filesystem path) of what a mount table shows at ``/``; the top one when stacked."""
    found = None
    for line in table.splitlines():
        fields = line.split()
        if len(fields) >= 5 and unescape(fields[4]) == "/":
            found = (fields[2], unescape(fields[3]))
    return found


def exposed(line, device, folders, host_root=True):
    """The candidate folders a mountinfo ``line`` shows somewhere OTHER than their own path.

    A bind of a folder above a key (the overlay root, ``/var/lib``...) into a
    container names every key below it: the container can use them through it
    without its mount table ever spelling a key. A mount that shows a folder
    at its own path exposes nothing new only in a namespace whose ``/`` is the
    host's (``host_root``: the machine's own mounts, a service's sandbox
    re-mount); in a container, ``-v X:X`` is a use like any other. Snap's
    namespaces show the host's whole ``/`` at :data:`SNAP_HOSTFS` (seen on the
    Z2, in every snap), which would keep every folder for ever, so a mount at
    or below that target is not counted. ``folders`` maps a token to (its
    path, its path as its filesystem spells it).
    """
    fields = line.split()
    if len(fields) < 5 or fields[2] != device:
        return []
    source, target = unescape(fields[3]), unescape(fields[4])
    if target == SNAP_HOSTFS or target.startswith(SNAP_HOSTFS + "/"):
        return []
    found = []
    for token, (path, inner) in folders.items():
        shown = visible_at(source, target, inner)
        if shown is not None and (shown != path or not host_root):
            found.append(token)
    return found


def references(plan, tokens, clock, result, folders=None, device="", host_root=None):
    """``{token: where it is named}`` for every candidate token in use.

    ``folders`` maps each candidate FOLDER's token to its path, ``device`` is
    its filesystem as mountinfo spells it, so a mount of an ancestor keeps it.
    """
    named = {token: "a deployment record" for token in plan["named"] if token in tokens}
    if not tokens:
        return named
    pattern = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(
        re.escape(token) for token in sorted(tokens, key=len, reverse=True)) + r")(?![A-Za-z0-9])")
    sources = [unit_texts(plan["units"], clock), mount_tables(clock, result)]
    for index, source in enumerate(sources):
        for where, text in source:
            for token in set(pattern.findall(text)):
                named.setdefault(token, where)
            if index == 1 and folders:
                same_root = root_of(text) == host_root
                for line in text.splitlines():
                    for token in exposed(line, device, folders, same_root):
                        named.setdefault(token, where + " (a mount of a folder above it)")
    return named


def survey(parent, name, relative, shape, owners, order):
    """Check ``name`` and all below it against ``shape``; its removal order into ``order``."""
    if relative not in shape:
        raise Keep("it holds {}, which Vaelor does not put there".format(relative))
    status = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if status.st_uid not in owners:
        raise Keep("{} belongs to an account other than root".format(relative))
    if stat.S_ISDIR(status.st_mode):
        if status.st_mode & OTHERS:
            raise Keep("{} can be changed by an account other than root".format(relative))
        handle = os.open(name, DIR_FLAGS, dir_fd=parent)
        try:
            for child in sorted(os.listdir(handle)):
                survey(handle, child, child if relative == "." else relative + "/" + child,
                       shape, owners, order)
        finally:
            os.close(handle)
    elif not stat.S_ISREG(status.st_mode) or status.st_nlink != 1 or status.st_mode & OTHERS:
        raise Keep("{} is a link, a special file, a file with a second name, or writable "
                   "by another account".format(relative))
    order.append(relative)


def remove(parent, name):
    """Remove ``name`` and all below it relative to ``parent``; a link is unlinked, never followed."""
    status = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if stat.S_ISDIR(status.st_mode):
        handle = os.open(name, DIR_FLAGS, dir_fd=parent)
        try:
            for child in os.listdir(handle):
                remove(handle, child)
        finally:
            os.close(handle)
        os.rmdir(name, dir_fd=parent)
    else:
        os.unlink(name, dir_fd=parent)


def copy_check(pull, name, digest, owners):
    status = os.stat(name, dir_fd=pull, follow_symlinks=False)
    if not stat.S_ISREG(status.st_mode) or status.st_uid not in owners or status.st_nlink != 1 \
            or status.st_mode & OTHERS:
        raise Keep("it is not a single-name file only root can change")
    handle = os.open(name, FILE_FLAGS, dir_fd=pull)
    try:
        hasher, total = hashlib.sha256(), 0
        while True:
            chunk = os.read(handle, 1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > 16 << 20:
                raise Keep("it is larger than any attention copy Vaelor writes")
            hasher.update(chunk)
    finally:
        os.close(handle)
    if not hasher.hexdigest().startswith(digest):
        raise Keep("its content is not the copy its name says it is")


def prune(plan):
    deadline = time.monotonic() + plan["seconds"]
    owners = frozenset(plan["owners"])
    result = {"removed": [], "would_remove": [], "kept": [], "stopped": "",
              "apply": bool(plan["apply"])}

    def clock():
        if time.monotonic() > deadline:
            raise Stop("it ran past its {} s limit".format(plan["seconds"]))

    try:
        root = sealed_folder(plan["root"], owners)
        pull = sealed_folder(plan["pull"], owners)
        try:
            folders = {}
            for key, kind in sorted(plan["folders"].items()):
                if root is not None and key in os.listdir(root):
                    folders[key] = kind
            copies = {}
            for name in (sorted(os.listdir(pull)) if pull is not None else []):
                found = COPY.fullmatch(name)
                if found and found.group(1) not in plan["current_hashes"]:
                    copies[name] = found.group(1)
            device, inside, host_root = "", {}, None
            if root is not None and folders:
                status = os.fstat(root)
                device = "{}:{}".format(os.major(status.st_dev), os.minor(status.st_dev))
                with open("/proc/self/mountinfo", "rb") as handle:
                    table = handle.read(8 << 20).decode("utf-8", "replace")
                base = inside_path(plan["root"], device, table)
                host_root = root_of(table)
                inside = {key: (plan["root"] + "/" + key, base.rstrip("/") + "/" + key)
                          for key in folders}
            used = references(plan, set(folders) | set(copies), clock, result, inside, device,
                              host_root)
            for key, kind in sorted(folders.items()):
                clock()
                path = plan["root"] + "/" + key
                try:
                    if key in used:
                        raise Keep("it is named by " + used[key])
                    if key in plan["keep"]:
                        raise Keep(plan["keep"][key])
                    survey(root, key, ".", set(plan["shapes"][kind]), owners, [])
                except (Keep, OSError) as reason:
                    result["kept"].append([path, str(reason)])
                    continue
                if plan["apply"]:
                    remove(root, key)
                    result["removed"].append(path)
                else:
                    result["would_remove"].append(path)
            for name, digest in sorted(copies.items()):
                clock()
                path = plan["pull"] + "/" + name
                try:
                    if name in used:
                        raise Keep("it is named by " + used[name])
                    copy_check(pull, name, digest, owners)
                except (Keep, OSError) as reason:
                    result["kept"].append([path, str(reason)])
                    continue
                if plan["apply"]:
                    os.unlink(name, dir_fd=pull)
                    result["removed"].append(path)
                else:
                    result["would_remove"].append(path)
        finally:
            for handle in (root, pull):
                if handle is not None:
                    os.close(handle)
    except Stop as reason:
        result["stopped"] = str(reason)
    except OSError as error:
        result["stopped"] = "a folder could not be read ({})".format(error.strerror or error)
    return result


if __name__ == "__main__":
    print(json.dumps(prune(json.loads(sys.argv[1]))))
'''


def shapes() -> Dict[str, List[str]]:
    """Every relative path Vaelor writes in each kind of folder: anything else keeps it."""
    from . import vllm_guesser_build as guesser
    from .vllm_moe_hip_build import BUILD_LOCK, CODE_OBJECT, RECORD
    from .vllm_overlay_build import STATUS_FILE

    plugin = ["plugin/" + name for name in PLUGIN_FILES]
    # VD-216 item 2: a build folder, its lock and record, and the built files
    # every row writes (one recipe's files are another's: the union is the shape).
    built = sorted({"{}/{}".format(guesser.GUESSER_DIR, name)
                    for row in guesser.GUESSERS.values() for name, _sha in row.outputs})
    return {
        "overlay": [".", DIFF_NAME, BUILT_NAME, STATUS_FILE],
        "plugin": [".", "plugin", "plugin/" + DIST_INFO, BUILD_LOCK, *plugin,
                   "plugin/" + CODE_OBJECT, "plugin/" + RECORD],
        "guesser": [".", guesser.GUESSER_DIR, guesser.BUILD_LOCK, guesser.RECORD, *built],
    }


def retired_folders() -> Dict[str, str]:
    """The retired keys whose folders may go, by kind; never a key some row still is."""
    from .vllm_guesser_build import GUESSERS, RETIRED_GUESSERS

    current = set(OVERLAYS) | set(MOE_KERNELS) | set(GUESSERS)
    found = {key: "overlay" for key in RETIRED_OVERLAYS}
    found.update({key: "plugin" for key in RETIRED_KEYS})
    found.update({key: "guesser" for key in RETIRED_GUESSERS})
    return {key: kind for key, kind in found.items() if key not in current}


def current_hashes() -> List[str]:
    """The per-hash copy names this release writes (`vllm_overlays.worker_path`)."""
    return sorted(overlay.built_sha256[:16] for overlay in OVERLAYS.values())


def named_in(texts: Iterable[str]) -> List[str]:
    """The candidate names (retired keys, per-hash copies) that ``texts`` mention.

    A name counts whole: a record naming key ``-10`` does not name key ``-1``.
    """
    keys = set(retired_folders())
    token_pattern = re.compile(r"vllm-overlay-[0-9a-f]{16}\.py|[A-Za-z0-9][A-Za-z0-9.-]*")
    found = set()
    for text in texts:
        for token in token_pattern.findall(str(text)):
            if token in keys or token.startswith("vllm-overlay-"):
                found.add(token)
    return sorted(found)


def record_texts(database: Any = None) -> List[str]:
    """Every pooled deployment row's text, read only; ``[]`` with no store. Raises when unreadable."""
    import sqlite3
    from contextlib import closing

    from .cluster_store import POOLED_ROWS_QUERY
    from .deployment_refresh import StoreUnread, cluster_database, read_only_store

    path = database or cluster_database()
    if not path.is_file():
        return []
    try:
        with closing(read_only_store(path)) as connection:
            rows = connection.execute(POOLED_ROWS_QUERY).fetchall()
    except sqlite3.Error as error:
        raise StoreUnread("the deployment records could not be read, so nothing was removed "
                          "({})".format(error)) from error
    return [" ".join(str(value) for value in tuple(row)) for row in rows]


def _pinned_diff(data: Optional[bytes], overlay: Any) -> bool:
    from . import vllm_overlay_build as builder

    if data is None:
        return False
    try:
        hunks = builder.file_hunks(data.decode("utf-8"), builder.PATCHED_FILE)
    except Exception:  # noqa: BLE001 - any unreadable diff is simply not the pinned one
        return False
    return builder.hunks_digest(hunks) == overlay.hunks_sha256


def offline_fallback(apply: bool, *, root: str = OVERLAY_ROOT,
                     owners: FrozenSet[int] = frozenset({0}),
                     owner: Sequence[int] = (0, 0)) -> Dict[str, str]:
    """Make sure removing retired overlay folders cannot remove the only pinned diff.

    For each overlay whose own folder holds no pinned diff, the newest retired
    folder's pinned diff is copied into it (``apply`` only); a folder whose
    diff could not be copied is returned as ``{key: why it stays}``.
    """
    from . import vllm_overlay_build as builder

    keep: Dict[str, str] = {}
    for overlay in OVERLAYS.values():
        current = "{}/{}/{}".format(root, overlay.key, DIFF_NAME)
        if _pinned_diff(root_controlled_bytes(current, owners=owners), overlay):
            continue
        retired = sorted((old for old, new in RETIRED_OVERLAYS.items() if new == overlay.key),
                         reverse=True)
        for key in retired:
            data = root_controlled_bytes("{}/{}/{}".format(root, key, DIFF_NAME), owners=owners)
            if not _pinned_diff(data, overlay):
                continue
            if apply:
                try:
                    builder._write_folder(overlay, root, tuple(owner),
                                          [(DIFF_NAME, data.decode("utf-8"))])
                except (OSError, ValueError):
                    pass
                if _pinned_diff(root_controlled_bytes(current, owners=owners), overlay):
                    break
            keep[key] = ("it holds the only downloaded copy of the pinned vLLM change, which "
                         "an offline rebuild of {} needs".format(overlay.key))
            break
    return keep


def plan(apply: bool, *, named: Sequence[str], keep: Mapping[str, str],
         root: str = OVERLAY_ROOT, pull: str = PULL_FOLDER,
         units: Sequence[str] = UNIT_FOLDERS, owners: Iterable[int] = (0,),
         seconds: int = PRUNE_SECONDS) -> Dict[str, Any]:
    """The program's input: every candidate and every fact it may not discover itself."""
    return {
        "apply": bool(apply), "root": root, "pull": pull, "units": list(units),
        "folders": retired_folders(), "shapes": shapes(), "current_hashes": current_hashes(),
        "named": list(named), "keep": dict(keep), "owners": sorted(set(owners)),
        "seconds": int(seconds),
    }


def run_program(program_plan: Mapping[str, Any]) -> Dict[str, Any]:
    """:data:`PRUNE_PROGRAM` run here, on this machine."""
    namespace: Dict[str, Any] = {"__name__": "vaelor_overlay_prune"}
    exec(compile(PRUNE_PROGRAM, "vaelor-overlay-prune", "exec"), namespace)  # noqa: S102 - our own text
    return namespace["prune"](dict(program_plan))


def prune_controller(apply: bool, **seams: Any) -> Dict[str, Any]:
    """The controller's prune: records read here, the offline fallback made safe first."""
    try:
        named = named_in(record_texts(seams.pop("database", None)))
    except Exception as error:  # noqa: BLE001 - an unread store means nothing is removed
        return {"removed": [], "would_remove": [], "kept": [], "stopped": str(error)[:300]}
    keep = offline_fallback(apply, root=seams.get("root", OVERLAY_ROOT),
                            owners=frozenset(seams.get("owners", (0,))),
                            owner=seams.pop("owner", (0, 0)))
    return run_program(plan(apply, named=named, keep=keep, **seams))


def prune_worker(transport: Any, records: Iterable[Any], *, apply: bool = True) -> Dict[str, Any]:
    """A worker's prune over SSH, as root; never raises - the result says what happened.

    ``records`` are the controller's deployment records, searched here for
    the candidates' names. Retired folders on a worker hold no diff (a
    worker's attention copy is built on the controller), so nothing there is
    an offline fallback.
    """
    try:
        named = named_in(json.dumps(record, default=str) for record in records)
        argv = ["python3", "-c", PRUNE_PROGRAM, json.dumps(plan(apply, named=named, keep={}))]
        return json.loads(str(transport.run(argv, sudo=True, timeout=PRUNE_SECONDS + 30)))
    except Exception as error:  # noqa: BLE001 - housekeeping never fails a Load
        return {"removed": [], "would_remove": [], "kept": [],
                "stopped": "the prune could not run there ({})".format(str(error)[:200])}


def report_lines(result: Mapping[str, Any], machine: str = CONTROLLER) -> List[str]:
    """One plain line per removal, per kept candidate, and for a stop (LESSONS 1)."""
    lines = ["Vaelor removed a superseded kernel version on {}: {}".format(machine, path)
             for path in result.get("removed") or []]
    lines += ["Vaelor would remove a superseded kernel version on {}: {}".format(machine, path)
              for path in result.get("would_remove") or []]
    lines += ["Vaelor kept a superseded kernel version on {}: {} ({})".format(machine, path, why)
              for path, why in result.get("kept") or []]
    if result.get("note"):
        lines.append("Vaelor's prune on {}: {}".format(machine, result["note"]))
    if result.get("stopped"):
        lines.append("Vaelor removed no superseded kernel versions on {}: {}".format(
            machine, result["stopped"]))
    return lines


def worker_name(transport: Any) -> str:
    """The worker a transport reaches, as its lines name it: the SSH profile's host."""
    profile = getattr(transport, "profile", None)
    host = profile.get("host") if isinstance(profile, Mapping) else None
    return "the worker {}".format(host) if host else "a worker"


def prune_worker_after_install(transport: Any, machine: Any) -> Dict[str, Any]:
    """A worker's prune once its serving unit is installed (`gpu_pool_runtime`).

    The controller's records are read here, read only; a store that cannot be
    read means nothing is removed there. Each line goes to the log of the
    process that installed the unit. Never raises.
    """
    try:
        texts = record_texts()
    except Exception as error:  # noqa: BLE001 - an unread store removes nothing
        result: Dict[str, Any] = {"removed": [], "would_remove": [], "kept": [],
                                  "stopped": str(error)[:300]}
    else:
        result = prune_worker(transport, texts)
    for line in report_lines(result, str(machine or "a worker")):
        LOG.info("%s", line)
    return result


def main(argv: Sequence[str]) -> int:
    """``apply`` or ``dry-run``, as root, from the installer. Never fails the install."""
    verb = argv[1] if len(argv) > 1 else ""
    if verb not in ("apply", "dry-run"):
        sys.stderr.write("usage: python -m vaelor.vllm_overlay_prune apply|dry-run\n")
        return 2
    for line in report_lines(prune_controller(verb == "apply")) or [
            "Vaelor found no superseded kernel versions on {}.".format(CONTROLLER)]:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
