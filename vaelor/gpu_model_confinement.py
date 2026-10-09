"""What the root bridge trusts as a GPU model path, and the folder it mounts.

Extracted from :mod:`vaelor.gpu_rocmfpx_service`, which re-exports every name
here, so the launch, its tests and the workload executor's download check read
ONE rule. Runs in two processes: the root hardware bridge (the launch, where the
rule is the boundary) and the workload executor (the download, where the same
characters are refused before a file is fetched that the launch would later
refuse - ACC-064).

Three layers, in the order a launch meets them:

1. **Shape** (:func:`_validate_model_path`): absolute, ``.gguf``, no ``:`` or
   control characters.
2. **Confinement** (:func:`_confined_model_path`): the text starts with the
   model cache, the part below it is plain characters in normal form
   (:func:`servable_cache_relative`), no symlinked component, a regular file.
3. **Anchor** (:func:`anchor_refusal`, ACC-063): the cache is the ``-v`` source
   of a root container with the GPU devices, and docker resolves that source
   when it runs. If any account but root could rename the cache, or any folder
   above it, a link could be swapped in between the last check and the mount.
   So every folder from ``/`` down to the cache must be one no other account
   can re-point: owned by root, and either not writable by group or other, or
   sticky with the entry below it owned by root too (the kernel then lets only
   root rename that entry). The installer lays the tree out that way
   (``/var/lib/vaelor`` root-owned and sticky, the Vaelor account's write coming
   from an ACL; the cache root-owned and sticky); this check proves it at the
   privileged side on every launch rather than trusting the install.
"""

from __future__ import annotations

import os
import posixpath
import re
import stat
from typing import Any, Callable, NamedTuple, Optional, Tuple


#: The model cache the GPU launch is confined to (the executor's
#: ``data_path("models")`` on a default install).
MODEL_CACHE_DIR = "/var/lib/vaelor/models"


class GpuModelPathError(ValueError):
    """A model path was refused before it could reach the GPU launch.

    The path is the one caller-side value that reaches the command line, so it is
    checked for the shape a real ``.gguf`` on disk has and rejected here rather
    than discovered as a failed launch.
    """


def _validate_model_path(model_path: Any) -> str:
    """Return an absolute ``.gguf`` path, or raise :class:`GpuModelPathError`.

    A correctness gate on SHAPE only: an empty path, a relative one, or a
    non-``.gguf`` is a misconfiguration cheaper to refuse at planning time than
    to discover as a failed model open. The newline/NUL check is defence in
    depth even behind a list argv. **Shape is not confinement:** the container
    launch the root bridge runs goes through :func:`_confined_model_path`, which
    adds the model-cache rule on top of this one.
    """
    text = str(model_path or "").strip()
    if not text:
        raise GpuModelPathError("A model path is required to launch the GPU server.")
    if any(character in text for character in "\n\r\x00"):
        raise GpuModelPathError(
            "The model path carries a control character and is refused."
        )
    if ":" in text:
        # A ':' is the docker -v field separator (host:container[:opts]). A model
        # path carrying one would split the mount into the wrong fields and produce
        # a corrupt bind, so it is refused at planning time (the launch mounts the
        # cache root the model resolves under) rather than launched as a broken
        # container.
        raise GpuModelPathError(
            "The model path contains ':', which would corrupt the docker -v "
            "mount (it is the host:container separator); '{}' is refused.".format(text)
        )
    # Any case, as the executor's own model check accepts it
    # (:func:`vaelor.local_model_files.model_file`).
    if not text.lower().endswith(".gguf"):
        raise GpuModelPathError(
            "The GPU server serves a .gguf artifact; '{}' is not one.".format(text)
        )
    if not text.startswith("/"):
        raise GpuModelPathError(
            "The model path must be absolute so the launch does not depend on a "
            "working directory; '{}' is relative.".format(text)
        )
    return text


class ModelFilesystem(NamedTuple):
    """How the launch reads the host filesystem to confine a model path.

    ``realpath`` resolves every symlink and ``..`` the way the kernel will
    (:func:`os.path.realpath`); ``isfile`` answers whether a path is an existing
    regular file (:func:`os.path.isfile`); ``is_real_dir`` answers whether a path
    is a directory itself, NOT a symlink to one (an ``lstat``); ``lstat`` is the
    raw :func:`os.lstat` the anchor rule reads owners and modes from. The root
    bridge always runs :data:`HOST_FILESYSTEM`. The seam exists so the rules are
    provable on a test host that cannot create symlinks, cannot chown, or does not
    use POSIX paths - a security rule whose test skips there would be no test at
    all. ``lstat`` has no default on purpose: a filesystem that forgot it must not
    quietly skip the anchor check.
    """

    realpath: Callable[[str], str]
    isfile: Callable[[str], bool]
    is_real_dir: Callable[[str], bool]
    lstat: Callable[[str], Any]


def _is_real_directory(path: str) -> bool:
    """True when ``path`` itself is a directory; a symlink to one is not."""
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


#: The real filesystem, as root sees it at launch time.
HOST_FILESYSTEM = ModelFilesystem(
    realpath=os.path.realpath, isfile=os.path.isfile,
    is_real_dir=_is_real_directory, lstat=os.lstat,
)

#: What may follow ``<model cache>/``: plain parts of letters, digits, ``.``,
#: ``_`` and ``-`` joined by single ``/``, ending in a GGUF suffix of any case.
#: The same characters the executor's repository and download names use, and
#: none that a docker ``-v`` field, a shell or a log line could read as syntax.
_CACHE_RELATIVE_PATH = re.compile(
    r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*\.gguf", re.IGNORECASE
)


def servable_cache_relative(relative: str) -> bool:
    """Whether a path BELOW the model cache is one the GPU launch will serve.

    The one owner of the rule for both ends of a model's life (ACC-064): the
    launch refuses anything else (:func:`_confined_model_path`), so the
    download refuses the same names before fetching bytes the launch would then
    refuse. Plain characters (:data:`_CACHE_RELATIVE_PATH`) in normal form - no
    ``.``/``..`` part, no doubled or leading ``/``.
    """
    text = str(relative or "")
    return bool(
        _CACHE_RELATIVE_PATH.fullmatch(text)
        and posixpath.normpath(text) == text
    )


def anchor_refusal(
    path: str, lstat: Callable[[str], Any] = os.lstat,
) -> Optional[str]:
    """``None`` when no account but root can re-point ``path``; else why not.

    Walks every folder from ``/`` down to ``path`` itself. Each one must be a
    real directory (an ``lstat``, so a symlink is refused, not followed), and
    each entry on the way must be one only root can rename or replace: the
    folder holding it is owned by root and either grants no group or other
    write, or carries the sticky bit with the entry itself owned by root.
    An ACL shows up in the group bits as its mask, so a folder that grants a
    named account write lands in the sticky branch - the conservative reading.
    The folders' own contents are not walked: inside the mount a swapped link
    resolves in the container, not on the host. Plain-English reasons only,
    naming the configured path and never a resolved one.
    """
    text = str(path or "").rstrip("/")
    if not text.startswith("/"):
        return "The folder '{}' is not an absolute path.".format(path)
    current = "/"
    try:
        parent = lstat(current)
    except OSError:
        return "The root folder could not be read."
    for part in [piece for piece in text.split("/") if piece]:
        child_path = posixpath.join(current, part)
        try:
            child = lstat(child_path)
        except OSError:
            return "The folder '{}' could not be read.".format(child_path)
        if not stat.S_ISDIR(child.st_mode):
            return (
                "'{}' is not a plain folder (a link or a file), so nothing "
                "is mounted through it.".format(child_path)
            )
        if parent.st_uid != 0:
            return (
                "The folder '{}' is owned by an account other than root, which "
                "could swap '{}' for a link.".format(current, child_path)
            )
        if parent.st_mode & (stat.S_IWGRP | stat.S_IWOTH) and not (
            parent.st_mode & stat.S_ISVTX and child.st_uid == 0
        ):
            return (
                "The folder '{}' lets another account rename '{}', so it could "
                "be swapped for a link before the launch.".format(current, child_path)
            )
        current, parent = child_path, child
    return None


def _confined_model_path(
    model_path: Any,
    model_cache_dir: str = MODEL_CACHE_DIR,
    filesystem: ModelFilesystem = HOST_FILESYSTEM,
) -> Tuple[str, str]:
    """``(model, cache)`` for a servable model, both canonical, or refuse it.

    **The root-side confinement rule for the GPU launch.** The hardware bridge
    socket is reachable by every group-``vaelor`` service, and the path arrives
    from whichever of them asked, so the rule is enforced here, where root builds
    the ``docker run``, not trusted from the caller (LESSONS 18). In order:

    1. the shape gate (:func:`_validate_model_path`);
    2. the caller's TEXT must start with the configured cache path - checked on
       the string, before anything touches the disk, so root never resolves an
       arbitrary host path on a caller's behalf (no path oracle);
    3. the part below the cache must be plain ``/``-separated parts of
       ``[A-Za-z0-9._-]`` ending in ``.gguf`` (:data:`_CACHE_RELATIVE_PATH`), in
       normal form (no ``..``, ``.`` or doubled ``/``);
    4. the cache itself must be pinned: resolving it changes nothing, it is a
       real directory, not a symlink, and no account but root can re-point it or
       any folder above it (:func:`anchor_refusal`, ACC-063);
    5. the model must name the file itself (resolving it changes nothing - no
       symlinked component) and be an existing regular file now.

    Refusals quote only the caller's own text and the configured cache path,
    never a resolved one. The producer, the workload executor, already hands
    over resolved in-cache paths (:func:`vaelor.local_model_files.model_file`),
    so a legitimate launch passes unchanged. Because both values are proven
    canonical, the returned strings are exactly what the kernel resolves.
    """
    text = _validate_model_path(model_path)
    cache = str(model_cache_dir).rstrip("/")
    # An empty ``cache`` (the root directory) would admit every path, so it
    # admits none.
    if not cache or not text.startswith(cache + "/"):
        raise GpuModelPathError(
            "The GPU server only serves models from the model cache ({}); '{}' "
            "is outside it.".format(model_cache_dir, text)
        )
    if not _CACHE_RELATIVE_PATH.fullmatch(text[len(cache) + 1:]):
        raise GpuModelPathError(
            "Below the model cache the path may hold only letters, digits, '.', "
            "'_' and '-' in '/'-separated parts ending in .gguf; '{}' is "
            "refused.".format(text)
        )
    if posixpath.normpath(text) != text:
        raise GpuModelPathError(
            "The model path must be written in normal form, without '..', '.' "
            "or doubled '/' parts; '{}' is refused.".format(text)
        )
    if filesystem.realpath(cache) != cache or not filesystem.is_real_dir(cache):
        raise GpuModelPathError(
            "The model cache ({}) must be a real directory at exactly that path, "
            "not a symbolic link or under one, so no GPU model is launched "
            "from it.".format(model_cache_dir)
        )
    anchor = anchor_refusal(cache, filesystem.lstat)
    if anchor is not None:
        # Imported here: `state_root_layout` imports this module's rule.
        from .state_root_layout import FIX_SENTENCE

        raise GpuModelPathError(
            "{} The GPU server is not launched from the model cache ({}) until "
            "only root can change the folders above it. {}".format(
                anchor, model_cache_dir, FIX_SENTENCE)
        )
    if filesystem.realpath(text) != text:
        raise GpuModelPathError(
            "The model path '{}' goes through a symbolic link, and the GPU "
            "server only launches a path that names the file itself.".format(text)
        )
    if not filesystem.isfile(text):
        raise GpuModelPathError(
            "The GPU server only launches a model file that is present; '{}' is "
            "not a regular file in the model cache.".format(text)
        )
    return text, cache
