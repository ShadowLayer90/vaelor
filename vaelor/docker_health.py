"""Detect whether the Docker daemon can actually create containers.

A daemon that answers ``docker info`` is not the same as a daemon that can
start a container. When ``/var/lib/docker`` is wiped out from under a running
dockerd (a reset data-root, a vanished mount), the socket still answers and
``docker version`` still succeeds, but every ``docker run`` fails with
``mkdir /var/lib/docker/containers/<id>: no such file or directory``. Vaelor
then reports the node as "ready to run apps" and blames whichever service
tried to start, stranding a non-technical operator (see MEMORY: a wiped Docker
data-root breaks all containers).

The probe is deliberately cheap and *structural*: learn the storage location,
then one stat of that directory. It never pulls or runs a container, so it is
safe to run on every capability scan.

Two hard rules, both learned from a shipped regression:

1. **Never depend on the Docker socket.** This probe runs inside capability
   discovery, which executes as the least-privilege control-plane user - a user
   deliberately NOT in the ``docker`` group. ``docker info`` there fails with
   permission denied, indistinguishable at the exit-code level from a dead
   daemon. So a failed ``docker info`` is treated as "could not learn the
   path", never as "daemon down": the data-root is resolved socket-free from
   ``/etc/docker/daemon.json`` (or the distro default) when ``docker info``
   cannot answer.
2. **Fail safe.** Only a data-root the filesystem *confirms* is missing is
   called broken. A stat that cannot reach a verdict (a permission error, an
   unknown OSError) reads as healthy, because crying wolf on a fine box - and
   worse, disabling every app-install behind a false "Docker needs repair" - is
   the more damaging error.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from typing import Any, Callable, Optional


HEALTHY = "ok"
STORAGE_MISSING = "storage_missing"
UNKNOWN = "unknown"

# The Debian/Ubuntu default, used when neither ``docker info`` nor
# ``daemon.json`` names a data-root. Both shipped appliances use it.
DEFAULT_DATA_ROOT = "/var/lib/docker"
DAEMON_CONFIG = "/etc/docker/daemon.json"


def _directory_present(path: str) -> Optional[bool]:
    """Whether ``path`` is a directory that exists.

    Returns ``True`` when it is present, ``False`` only when the filesystem
    confirms it is gone (``FileNotFoundError``), and ``None`` when the answer
    cannot be established (a permission error, any other ``OSError``). The
    three states are kept apart on purpose: a ``None`` must never be read as a
    ``False``, or a box whose data-root Vaelor merely cannot stat would be
    reported as broken.

    Uses ``os.stat`` deliberately, NOT ``os.path.isdir``: the latter swallows
    every ``OSError`` (permission included) to ``False`` internally, which would
    make the ``None`` state unreachable and turn an un-stattable data-root into a
    false "broken" verdict.
    """
    try:
        mode = os.stat(path).st_mode
    except FileNotFoundError:
        return False
    except OSError:
        return None
    return stat.S_ISDIR(mode)


def _reported_data_root(
    docker_path: str,
    runner: Callable[..., subprocess.CompletedProcess],
) -> Optional[str]:
    """The data-root as ``docker info`` reports it, or ``None``.

    Needs Docker-socket access, so it succeeds for a caller in the ``docker``
    group (the executor, root) and returns ``None`` for one without it (the
    control-plane user). A ``None`` here is not evidence of anything - the
    caller falls back to the socket-free sources.
    """
    try:
        result = runner(
            [docker_path, "info", "--format", "{{.DockerRootDir}}"],
            capture_output=True,
            check=False,
            text=True,
            timeout=4,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    root = (result.stdout or "").strip()
    return root or None


def _configured_data_root(config_path: str = DAEMON_CONFIG) -> Optional[str]:
    """A custom data-root declared in ``daemon.json``, read without the socket.

    ``daemon.json`` is world-readable, so the control-plane user can see a
    custom ``data-root`` (or the legacy ``graph`` key) here even though it
    cannot reach the daemon. This is what keeps the default-path fallback from
    false-flagging a box whose storage lives somewhere other than
    ``/var/lib/docker``.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    root = data.get("data-root") or data.get("graph")
    if isinstance(root, str) and root.strip():
        return root.strip()
    return None


def runtime_health(
    docker_path: Optional[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    directory_present: Optional[Callable[[str], Optional[bool]]] = None,
    configured_data_root: Callable[[], Optional[str]] = _configured_data_root,
) -> dict[str, Any]:
    """Report whether Docker's storage area is present on this host.

    ``docker_path`` is the resolved ``docker`` executable (or falsy when Docker
    is not installed). ``runner``, ``directory_present`` and
    ``configured_data_root`` are injected so the probe can be exercised without
    touching the real Docker, filesystem, or ``daemon.json``.

    Never reports "broken" from a failed ``docker info`` (that only means the
    caller lacks socket access); only a data-root the filesystem confirms is
    missing is ``storage_missing``.
    """
    if not docker_path:
        return {
            "state": UNKNOWN,
            "reason": "Docker is not installed on this host.",
            "data_root": None,
        }
    data_root = (
        _reported_data_root(docker_path, runner)
        or configured_data_root()
        or DEFAULT_DATA_ROOT
    )
    present = (directory_present or _directory_present)(data_root)
    # Fail SAFE: only a directory confirmed missing (``is False``) is broken.
    # ``None`` (permission or unknown error) and ``True`` both read as healthy,
    # so the probe never blames Docker on a box it merely cannot stat and never
    # depends on socket access it may not have.
    if present is False:
        return {
            "state": STORAGE_MISSING,
            "reason": (
                "Docker's storage area ({}) is missing, so no app container "
                "can start. Repair Docker to rebuild it.".format(data_root)
            ),
            "data_root": data_root,
        }
    return {
        "state": HEALTHY,
        "reason": "The Docker runtime is healthy.",
        "data_root": data_root,
    }


def container_runtime_healthy() -> bool:
    """Whether Docker's storage area is present right now, as a bare bool.

    A convenience wrapper over :func:`runtime_health` for callers that only need
    the yes/no (the web-research status reason). It resolves ``docker`` itself
    and swallows every failure to ``True``: a probe error must never falsely
    accuse Docker of being broken.
    """
    try:
        return runtime_health(shutil.which("docker"))["state"] == HEALTHY
    except Exception:
        return True
