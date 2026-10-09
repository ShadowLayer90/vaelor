"""The controller's GPU units and model store, as the ROOT bridge handles them.

VD-143, closing the residual VD-125 left open. The controller's GPU is a
serving node in a vLLM cluster, so its units are installed here, as root. They
used to arrive as TEXT: a client ``tee``'d a ``vaelor-vllm-*.service`` of its
own composition through ``run_argv``, then asked for ``daemon-reload`` and
``enable --now`` - and the argv policy that bounded those calls could not read
the body ``tee`` was handed. Any account able to reach the socket could write
an ``ExecStart`` of its choosing and have systemd run it as root: root in three
accepted calls.

**The bridge now renders every unit it installs.** A client sends the unit's
KIND and its typed VALUES (a name, a repo, ports, addresses, degrees, GIDs);
:meth:`ManagedUnits.install` hands them to
`gpu_pool_runtime.GpuPoolRuntime.render_managed_unit` - the ONE template, the
same function the worker path renders with - which refuses a value name the
kind does not take and checks every value against its `gpu_unit_params` rule.
The image, the mounts, the environment, the user and the command are the
template's, not the caller's. The rendered text is then checked for the
template's own shape (:func:`check_rendered_unit`), written with no-follow file
operations, read back and compared, and only then reloaded and started. The
``run_argv`` policy no longer has a ``tee``, an ``install``, an ``enable`` or a
``start`` at all, so there is no other way for a managed unit to reach disk
through the bridge or to be started by it.

**The model store is touched only through `bridge_safe_fs`.** The store is
group-writable by every ``vaelor-jobs`` account, and a root ``rm -rf``,
``install -d``, ``tee`` or ``cat`` there followed any link one of them planted.
The pull and compile directories are made (or taken back) root-owned through
the descriptor that was checked; a stale progress record is unlinked, never
followed; a progress record is read without following a link; a repo's weights
are removed by a walk that never leaves the directory it opened.

**The pull PROGRAM is Vaelor's own installed file.** On a worker the program is
still written into the store over SSH, where the login is already a sudoer. On
the controller the pull unit mounts `vaelor.gpu_pull_program` read-only out of
the installed package, after :func:`bridge_safe_fs.is_root_controlled` has
confirmed that no account but root can change that file or any directory
above it - so the program a root container runs is never one a store writer
could have replaced.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import bridge_safe_fs as safe_fs
from . import gpu_pool_runtime as runtime_template
from . import state_root_layout
from .gpu_pool_units import (
    DOCKER, RAY_WORKER_ROLE, SERVER_ROLE, container_name, pull_unit_text, repo_slug,
    require_pull_unit, unit_role, unit_text,
)
from .hardware_bridge_client import RUN_ARGV_TIMEOUT_SECONDS
from .vllm_entry_program import CONTAINER_ENTRY_PROGRAM
from .vllm_images import compile_environment
from .vllm_serve_options import OPTIONS_FIELD, wants_tuned_tables
from . import gpu_ray_plane as ray_plane

#: Where the controller's managed units live, and the model store the GPU tier
#: shares with the workload executor. Spelled here, like `bridge_argv_policy`'s
#: roots, and tied to the runtime's spellings by `tests/test_bridge_managed_units.py`.
UNIT_DIRECTORY = "/etc/systemd/system"
MODEL_STORE = "/var/lib/vaelor/models"

#: The store's folders the vLLM containers use: the pull program's progress
#: records, the per-node compile cache (VD-127) and the Hugging Face hub.
_PULL_FOLDER = "pull"
_COMPILE_FOLDER = "compile"
_HUB_FOLDER = "hub"

#: A progress record is one small JSON object; anything bigger is not one.
_PROGRESS_LIMIT = 16 * 1024

#: The mode every managed unit file is written with: systemd reads it, and no
#: account but root may change it.
_UNIT_FILE_MODE = 0o644

#: Stand-ins the unit skeleton is rendered with, so every line but these two is
#: compared byte for byte with what `gpu_pool_units` writes.
_DESCRIPTION_SLOT = "\x01"
_START_SLOT = "\x02"

#: The Ray-plane verbs hold the bridge's lock, so their worst case is bounded
#: inside the client's wait (`UNIT_INSTALL_SOCKET_TIMEOUT`, 390 s): at most
#: this many fixed commands, each given at most this long (review 2).
RAY_PLANE_STEPS = 5
RAY_PLANE_STEP_SECONDS = 60

#: Characters the template never writes: systemd specifiers and variable
#: expansion, and the shell words a command could be chained with.
_FORBIDDEN_UNIT_CHARACTERS = "%$`;|<>\\\x00\r"

#: The ``docker run`` options a managed container may carry, each with the
#: values it may take, spelled from the runtime's own constants. The ORDER is
#: pinned separately (:data:`_SERVING_OPTIONS`, :data:`_PULL_OPTIONS`).
_ENTRYPOINTS = frozenset({
    runtime_template.RAY_ENTRYPOINT, runtime_template.BASH_ENTRYPOINT,
    runtime_template.PYTHON_ENTRYPOINT, runtime_template.VLLM_BINARY,
})
#: Each pinned image's own variables and compile-cache folders (`vllm_images`),
#: the model cache, the MoE entry program's tables folder, and a split's Ray
#: auth mode and token PATH (ACC-163): fixed text all.
_FIXED_ENV = frozenset({
    f"HF_HOME={runtime_template.MODELS_MOUNT}",
    runtime_template.TUNED_TABLES_ENV,
    *ray_plane.RAY_CONTAINER_ENVIRONMENT,
    *(env for image in runtime_template.IMAGES.values() for env in image.environment),
    *(env for image in runtime_template.IMAGES.values()
      for env in compile_environment(image, runtime_template.COMPILE_MOUNT)),
})
#: The images a managed container may run: exactly the pinned table.
_IMAGES = frozenset(image.image for image in runtime_template.IMAGES.values())
#: Each image's own fixed variables, and all of them together.
_IMAGE_ENV = {
    image.image: frozenset(image.environment) for image in runtime_template.IMAGES.values()
}
_ALL_IMAGE_ENV = frozenset(env for envs in _IMAGE_ENV.values() for env in envs)
#: The RCCL/Gloo variables a distributed container carries, whose values are a
#: validated address or NIC name: plain characters and nothing else.
_RCCL_ENV = r"(?:VLLM_HOST_IP|NCCL_SOCKET_IFNAME|GLOO_SOCKET_IFNAME)=[A-Za-z0-9._-]{1,64}"
_STORE_MOUNT = f"{runtime_template.MODEL_CACHE_ROOT}:{runtime_template.MODELS_MOUNT}"
_SERVING_MOUNTS = [
    _STORE_MOUNT,
    f"{runtime_template.COMPILE_CACHE_ROOT}:{runtime_template.COMPILE_MOUNT}",
    f"{runtime_template.SERVING_PROFILES_DIR}:{runtime_template.SERVING_PROFILES_MOUNT}",
]
_PROGRAM_MOUNT = (
    r"/[A-Za-z0-9._/-]{1,255}\.py:"
    + re.escape(runtime_template._CONTAINER_PACKAGED_PROGRAM) + ":ro"
)
#: The one further mount an MoE model's serving container may carry: Vaelor's
#: entry program, read-only (`vllm_entry_program`).
_ENTRY_MOUNT = r"/[A-Za-z0-9._/-]{1,255}\.py:" + re.escape(CONTAINER_ENTRY_PROGRAM) + ":ro"
#: The one further mount a split's Ray container carries: its node's Ray token
#: file, read-only (ACC-163, `gpu_ray_plane`).
_RAY_TOKEN_MOUNT = (
    re.escape(ray_plane.RAY_TOKEN_ROOT) + r"/[a-z0-9][a-z0-9-]{0,38}\.token:"
    + re.escape(ray_plane.CONTAINER_TOKEN_PATH) + ":ro"
)
#: The option sequence each unit family carries, as the option names in order.
_SERVING_OPTIONS = (
    r"--network --device --device(?: --group-add){0,8} --security-opt --ipc "
    r"--shm-size(?: -e){2,} -v -v -v(?: -v --cgroup-parent)?(?: -v)?(?: --entrypoint)?"
)
_PULL_OPTIONS = r"--network -e -v(?: -v)? --entrypoint"

#: The unit roles the bridge may install. A gate is a worker's alone (VD-129):
#: the controller's replica is behind the balancer on loopback.
_INSTALLABLE_ROLES = frozenset({SERVER_ROLE, RAY_WORKER_ROLE})


def _hub_name(model: str) -> str:
    """The Hugging Face hub folder a validated ``org/name`` repo lives in."""
    return "models--" + model.replace("/", "--")


def check_rendered_unit(unit: str, text: str, *, image: str = None) -> None:
    """Refuse ``text`` unless it has the managed template's shape, exactly.

    Belt and braces behind the template itself: every value was already
    checked by its own rule, so nothing should trip this, and a refusal here
    means the template and this check disagree - which is a bug to find before
    systemd runs anything as root. In order:

    * the name is a managed serving or pull unit, never a gate;
    * no systemd specifier or shell chaining appears anywhere;
    * every line but the description and the start is byte-identical to the
      skeleton `gpu_pool_units` renders for this container - so ``ExecStop``,
      ``After``, ``Wants``, ``Requires``, ``Type``, the restart policy and
      ``WantedBy`` are pinned to their values, not merely to their keys - and
      the description is Vaelor's;
    * the start is ``docker run --rm --name <this unit's container>``, then
      exactly the option sequence its family carries (:func:`_check_options`),
      each option with a value the runtime could have chosen, then the image
      in the next slot and nowhere else before it - ``image`` when given, else
      any image of the pinned table (`vllm_images`).
    """
    pulling = True
    try:
        require_pull_unit(unit)
    except ValueError:
        pulling = False
        if unit_role(unit) not in _INSTALLABLE_ROLES:
            raise ValueError("The bridge installs no unit of that role.")
    if any(character in text for character in _FORBIDDEN_UNIT_CHARACTERS):
        raise ValueError("The rendered unit carries a character the template never writes.")
    container = unit[: -len(".service")]
    start, ray_slice = _match_skeleton(text, container, pulling)
    words = start.split(" ")
    if words[:5] != [DOCKER, "run", "--rm", "--name", container]:
        raise ValueError("The rendered unit does not run the managed container.")
    options, rest = _split_options(words[5:])
    # Every Ray container carries the auth mode (`gpu_ray_plane`): a unit that
    # runs one must load its fence and be bound to its slice (review 1).
    if not ray_slice and ("-e", ray_plane.RAY_AUTH_ENVIRONMENT[0]) in options:
        raise ValueError("A split's Ray unit must load its firewall and run in its slice.")
    _check_options(options, pulling, ray_slice)
    if not rest or rest[0] not in ({image} if image is not None else _IMAGES):
        raise ValueError("The rendered unit does not run the pinned image in its place.")
    # An image's own variables only on that image's units (review 2): the 0.27
    # idle-spin guard never on a 0.22.1 unit, nor the reverse.
    own_env = _IMAGE_ENV.get(rest[0], frozenset())
    if any(name == "-e" and value in _ALL_IMAGE_ENV and value not in own_env
           for name, value in options):
        raise ValueError("The rendered unit carries another image's variable.")


def _match_skeleton(text: str, container: str, pulling: bool) -> Tuple[str, str]:
    """``(ExecStart value, slice)`` once every other line matches the skeleton.

    ``slice`` is the split slice the unit is bound to, ``""`` for any other.
    """
    if pulling:
        skeletons = [("", pull_unit_text(
            description=_DESCRIPTION_SLOT, container_name=container,
            exec_start=_START_SLOT,
        ))]
    else:
        # A split's Ray unit (server or Ray worker) may load its own fence,
        # and exactly its own, before it starts (ACC-163), bound to its own
        # slice and no other (ACC-187).
        role = unit_role(container + ".service")
        own = container[len("vaelor-vllm-"):-len("-" + role)]
        pres = [("", "", ("", ""))]
        if role in (SERVER_ROLE, RAY_WORKER_ROLE):
            # The fence, the slice and both slice checks, all this unit's own.
            pres.append((ray_plane.boot_start_pre(own), ray_plane.slice_name(own), (
                ray_plane.slice_check_pre(), ray_plane.slice_check_post(own, role))))
        skeletons = [
            (ray_slice, unit_text(
                description=_DESCRIPTION_SLOT, container_name=container,
                exec_start=_START_SLOT, unlimited_restarts=unlimited, start_pre=pre,
                ray_slice=ray_slice, slice_checks=checks,
            ))
            for unlimited in (False, True) for pre, ray_slice, checks in pres
        ]
    lines = text.split("\n")
    for ray_slice, skeleton in skeletons:
        expected = skeleton.split("\n")
        if len(expected) != len(lines):
            continue
        start = None
        for want, have in zip(expected, lines):
            if want == "Description=" + _DESCRIPTION_SLOT:
                if not have.startswith("Description=Vaelor "):
                    break
            elif want == "ExecStart=" + _START_SLOT:
                if not have.startswith("ExecStart="):
                    break
                start = have[len("ExecStart="):]
            elif want != have:
                break
        else:
            if start is not None:
                return start, ray_slice
    raise ValueError("The rendered unit's lines are not the template's.")


def _split_options(words: List[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """``([(option, value)], rest)``: the docker options, then everything after."""
    options: List[Tuple[str, str]] = []
    index = 0
    while index < len(words) and words[index].startswith("-"):
        if index + 1 >= len(words):
            raise ValueError("The rendered unit's start ends inside an option.")
        options.append((words[index], words[index + 1]))
        index += 2
    return options, words[index:]


def _check_options(options: List[Tuple[str, str]], pulling: bool, ray_slice: str = "") -> None:
    """Refuse an option sequence or value the runtime's template never emits.

    A ``--cgroup-parent`` is a split's Ray container's alone, and names exactly
    the slice its unit is bound to (``ray_slice``, ACC-187).
    """
    names = " ".join(name for name, _value in options)
    if not re.fullmatch(_PULL_OPTIONS if pulling else _SERVING_OPTIONS, names):
        raise ValueError("The rendered unit's container options are not the template's.")
    parents = [value for name, value in options if name == "--cgroup-parent"]
    if parents != ([ray_slice] if ray_slice else []):
        raise ValueError("The rendered unit runs its container outside its own split's slice.")
    mounts = [value for name, value in options if name == "-v"]
    if pulling:
        allowed_mounts = mounts[:1] == [_STORE_MOUNT] and all(
            re.fullmatch(_PROGRAM_MOUNT, mount) for mount in mounts[1:]
        )
    else:
        extra = mounts[3:]
        if extra and re.fullmatch(_RAY_TOKEN_MOUNT, extra[0]):
            extra = extra[1:]
        allowed_mounts = mounts[:3] == _SERVING_MOUNTS and len(extra) <= 1 and all(
            re.fullmatch(_ENTRY_MOUNT, mount) for mount in extra
        )
    if not allowed_mounts:
        raise ValueError("The rendered unit mounts a folder the template never mounts.")
    fixed = dict(zip(
        runtime_template._CONTAINER_RUNTIME_FLAGS[0::2],
        runtime_template._CONTAINER_RUNTIME_FLAGS[1::2],
    ))
    fixed["--network"] = "host"
    for name, value in options:
        if name in fixed:
            ok = value == fixed[name]
        elif name == "--device":
            ok = value in runtime_template.GPU_DEVICES
        elif name == "--group-add":
            ok = value.isdigit()
        elif name == "--entrypoint":
            ok = value in _ENTRYPOINTS
        elif name == "-e":
            ok = value in _FIXED_ENV or re.fullmatch(_RCCL_ENV, value) is not None
        elif name == "--cgroup-parent":
            ok = value == ray_slice
        else:
            ok = name == "-v"
        if not ok:
            raise ValueError("The rendered unit gives a container option a value the template never gives it.")


def _read_text(path: str) -> str:
    """A small kernel file's text, ``""`` when it cannot be read."""
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read(65536)
    except OSError:
        return ""


def _local_links() -> Any:
    """This machine's own addresses and the link carrying each (`cluster_link`)."""
    from .cluster_link import local_node_links

    return local_node_links()


class ManagedUnits:
    """Install the controller's managed GPU units and tend its model store, as root."""

    def __init__(
        self, *, unit_directory: str = UNIT_DIRECTORY,
        model_store: str = MODEL_STORE,
        run: Optional[Callable[..., Any]] = None,
        pull_program: Optional[str] = None,
        owner: Tuple[int, int] = (0, 0),
        start_refusal: Optional[Callable[[], Optional[str]]] = None,
        entry_program: Optional[str] = None, token_parent: str = "/etc",
        local_links: Optional[Callable[[], Any]] = None,
        read_text: Optional[Callable[[str], str]] = None,
    ):
        # Every seam exists for the tests, which cannot run as root or write
        # /etc; the bridge builds this with no arguments.
        self._token_parent = token_parent
        # The routing table and a process's cgroup file, read as root.
        self._read_text = read_text or _read_text
        # Each fixed command's bound; a Ray-plane verb narrows it so the whole
        # verb fits inside the client's wait (review 2).
        self._step_seconds = RUN_ARGV_TIMEOUT_SECONDS
        # This machine's own addresses per link (`cluster_link`), read here.
        self._local_links = local_links or _local_links
        self._unit_directory = unit_directory
        self._model_store = model_store
        self._run = run
        self._pull_program_path = pull_program
        self._entry_program_path = entry_program
        self._owner = (int(owner[0]), int(owner[1]))
        self._start_refusal = start_refusal or state_root_layout.mount_refusal

    def install(self, kind: Any, values: Any) -> Dict[str, str]:
        """Render, check, write, read back, reload and start one managed unit.

        Held under the bridge's runtime lock by the caller: a unit write and a
        ``daemon-reload`` from two concurrent deploys must not interleave.
        Re-rendering a unit that is already installed - the upgrade path, and
        every load of an unloaded deployment - writes the same bytes over it and
        ``enable --now`` of an active unit leaves its process running.
        """
        runtime = runtime_template.GpuPoolRuntime()
        pulling = str(kind) == runtime_template.MODEL_PULL_KIND
        program = self._pull_program() if pulling else None
        # An MoE model's container starts through Vaelor's entry program, which
        # is mounted out of the installed package once shown root's alone.
        entry = (
            self._entry_program()
            if not pulling and isinstance(values, dict)
            and wants_tuned_tables(values.get(OPTIONS_FIELD)) else None
        )
        unit, text = runtime.render_managed_unit(
            kind, values, pull_program=program, entry_program=entry,
        )
        check_rendered_unit(unit, text)
        # Every folder the container mounts, and the hub its weights live in,
        # anchored from `/` so only root can re-point them (ACC-063, review
        # S6) - asked before the store is touched or a unit file written.
        refusal = self._start_refusal()
        if refusal is not None:
            raise RuntimeError(refusal)
        repo = values.get("repo") if pulling else values.get("model")
        model = runtime._model(repo) if repo is not None else None
        self._prepare_store(model, pulling=pulling)
        with safe_fs.directory(self._unit_directory) as units:
            safe_fs.write_file(units, unit, text, mode=_UNIT_FILE_MODE)
            written = safe_fs.read_file(
                units, unit, limit=len(text.encode("utf-8")) + 1,
            )
        if written != text:
            raise RuntimeError(
                "The unit file on disk is not the one the bridge rendered, so it "
                "was not started."
            )
        self._systemctl("daemon-reload")
        if not pulling:
            # A start limit hit since must not refuse this start (review 4).
            try:
                self._systemctl("reset-failed", unit)
            except RuntimeError:
                pass
            self._systemctl("enable", "--now", unit)
            return {"unit": unit}
        try:
            self._systemctl("reset-failed", unit)
        except RuntimeError:
            pass
        self._systemctl("start", "--no-block", unit)
        return {"unit": unit}

    def prepare_ray_plane(
        self, name: Any, token: Any, link: Any, address: Any, peers: Any,
        shared: Any = False,
    ) -> Dict[str, str]:
        """Guard this controller's part of a split: its Ray token and its firewall (ACC-163).

        Only typed values arrive: a deployment name, the token, and IPv4
        addresses. The ruleset is rendered here from the fixed template
        (`gpu_ray_plane.render_ruleset`); both files are written root-only
        (``0600`` in a ``0700`` folder) through the same no-follow, atomic,
        owner-converging calls every file the bridge writes goes through, and
        the rules are loaded with ``nft -f``. No nftables, or a load it
        refuses, is a failure the deploy reports: a split never starts here
        unguarded. The token is in no argv and no log line.

        Before the rules, the split's own slice unit is written from its fixed
        text and started (ACC-187): the rules name that slice's cgroup, and
        nftables refuses the load of a path that does not exist. ``shared``
        (only ``True`` counts) says the link is this machine's shared network
        card, whose fence guards the split's own sockets only.
        """
        deployment = runtime_template.GpuPoolRuntime()._name(name)
        token = ray_plane.require_ray_token(token)
        ruleset = ray_plane.render_ruleset(
            deployment, link, address, peers, shared=shared is True,
        )
        # The address must be this machine's own, on the named link; a peer
        # must not be this machine at all (the review's S-check).
        held = self._local_links() or []
        if not any(item.get("name") == link and item.get("address") == address for item in held):
            raise ValueError("That address is not this machine's own on that link.")
        mine = {str(item.get("address", "")) for item in held}
        if any(peer in mine for peer in peers):
            raise ValueError("A machine of the split cannot be this machine itself.")
        # Never a whole-interface fence on the card this machine is reached by
        # (review 1, SC2), whatever the caller said about the link.
        self._step_seconds = RAY_PLANE_STEP_SECONDS
        closing = ray_plane.dedicated_on_default_route(
            link, shared is True, self._read_text(ray_plane.ROUTE_TABLE), held)
        if closing is not None:
            raise ValueError(closing)
        # The fence names the slice; only Docker's systemd driver puts the
        # containers in it (review 1, SC1).
        refusal = ray_plane.cgroup_driver_refusal(
            self._fixed(DOCKER, *ray_plane.CGROUP_DRIVER_ARGV[1:]))
        if refusal is not None:
            raise RuntimeError(refusal)
        nft = self._nft_binary()
        with safe_fs.directory(self._unit_directory) as units:
            safe_fs.write_file(
                units, ray_plane.slice_name(deployment),
                ray_plane.slice_unit_text(deployment), mode=_UNIT_FILE_MODE,
            )
        self._systemctl("daemon-reload")
        self._systemctl("start", ray_plane.slice_name(deployment))
        folder = ray_plane.RAY_TOKEN_ROOT.rsplit("/", 1)[1]
        with safe_fs.directory(self._token_parent) as parent:
            handle = safe_fs.ensure_root_directory(
                parent, folder, mode=0o700, owner=self._owner,
            )
            try:
                safe_fs.write_file(handle, deployment + ".token", token, mode=0o600)
                safe_fs.write_file(handle, deployment + ".nft", ruleset, mode=0o600)
                # The check every Ray unit runs at each start (review 2, SC2).
                safe_fs.write_file(
                    handle, ray_plane.SLICE_CHECK_PATH.rsplit("/", 1)[1],
                    ray_plane.slice_check_program(), mode=0o600,
                )
            finally:
                os.close(handle)
        self._nft(nft, "-f", self._ray_path(deployment, ".nft"))
        return {"table": ray_plane.table_name(deployment)}

    def container_cgroup(self, name: Any, role: Any) -> Dict[str, str]:
        """Where one of a split's Ray containers on this controller runs (review 1, SC1).

        Docker names the container's process; the kernel's ``/proc/<pid>/cgroup``
        says where it runs. ``""`` while the container is not running yet.
        """
        deployment = runtime_template.GpuPoolRuntime()._name(name)
        self._step_seconds = RAY_PLANE_STEP_SECONDS
        if role not in ray_plane.RAY_ROLES:
            raise ValueError(ray_plane.NOT_A_RAY_CONTAINER)
        try:
            pid = self._fixed(DOCKER, "inspect", "--format", "{{.State.Pid}}",
                              container_name(deployment, role)).strip()
        except RuntimeError:
            return {"cgroup": ""}
        if not pid.isdigit() or pid == "0":
            return {"cgroup": ""}
        return {"cgroup": ray_plane.cgroup_of(self._read_text("/proc/{}/cgroup".format(pid)))}

    def clear_ray_plane(self, name: Any) -> Dict[str, str]:
        """Remove this controller's part of a split's Ray plane: slice, table, files.

        **The containers go first** (review 1, SC3): the slice is stopped
        before its firewall is dropped, so no running container is ever left
        unfenced. A slice that will not stop keeps the firewall and says so; one
        already gone is no obstacle. Every later step is tolerated failing - the
        table may never have been loaded, a file never written.
        """
        deployment = runtime_template.GpuPoolRuntime()._name(name)
        self._step_seconds = RAY_PLANE_STEP_SECONDS
        try:
            self._systemctl("stop", ray_plane.slice_name(deployment))
        except RuntimeError as error:
            if ray_plane.SLICE_ALREADY_GONE not in str(error):
                raise RuntimeError(
                    "The split's containers on this controller could not be stopped, "
                    "so its firewall was kept: {}".format(str(error)[:200])
                ) from error
        try:
            self._nft(self._nft_binary(), "delete", "table", "inet",
                      ray_plane.table_name(deployment))
        except RuntimeError as error:
            # A table already gone is clear; any other failure is said (SC5).
            if ray_plane.TABLE_ALREADY_GONE not in str(error):
                raise RuntimeError(
                    "The split's firewall on this controller could not be removed: "
                    "{}".format(str(error)[:200])) from error
        for step in (
            lambda: self._remove_unit_file(ray_plane.slice_name(deployment)),
            lambda: self._systemctl("daemon-reload"),
        ):
            try:
                step()
            except (RuntimeError, OSError):
                pass
        folder = ray_plane.RAY_TOKEN_ROOT.rsplit("/", 1)[1]
        try:
            with safe_fs.directory(self._token_parent.rstrip("/") + "/" + folder) as handle:
                for suffix in (".nft", ".token"):
                    safe_fs.remove_file(handle, deployment + suffix)
        except OSError:
            pass
        return {"table": ray_plane.table_name(deployment)}

    def _fixed(self, *argv: str) -> str:
        """One fixed command's stdout, bounded; a failure raises with its own words."""
        try:
            completed = (self._run or subprocess.run)(
                list(argv), capture_output=True, text=True,
                timeout=self._step_seconds, check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("{} did not finish in time.".format(argv[1])) from error
        if completed.returncode != 0:
            raise RuntimeError(str(completed.stderr or "").strip()[:400] or "The command failed.")
        return str(completed.stdout or "")

    def _remove_unit_file(self, unit: str) -> None:
        """Remove one unit file of this bridge's own, without following a link."""
        with safe_fs.directory(self._unit_directory) as units:
            safe_fs.remove_file(units, unit)

    def _ray_path(self, deployment: str, suffix: str) -> str:
        folder = ray_plane.RAY_TOKEN_ROOT.rsplit("/", 1)[1]
        return "/".join([self._token_parent.rstrip("/"), folder, deployment + suffix])

    @staticmethod
    def _nft_binary() -> str:
        """The system's ``nft``, at a fixed root path, or the plain refusal."""
        for path in ("/usr/sbin/nft", "/sbin/nft"):
            if os.path.exists(path):
                return path
        raise RuntimeError(
            "This machine has no nftables (nft), so a split's Ray ports cannot "
            "be guarded here. Install nftables and deploy again."
        )

    def _nft(self, nft: str, *arguments: str) -> None:
        """One fixed ``nft`` call, bounded; a failure raises with nft's own words."""
        try:
            completed = (self._run or subprocess.run)(
                [nft, *arguments], capture_output=True, text=True,
                timeout=self._step_seconds, check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("nft did not finish in time.") from error
        if completed.returncode != 0:
            raise RuntimeError(str(completed.stderr or "").strip()[:400] or "nft failed.")

    def read_progress(self, repo: Any) -> str:
        """The text of ``repo``'s pull progress record, read without following a link.

        Only a record root wrote is believed: the pull folder must be one only
        root can change, and the record a root-owned regular file with one
        name - a hard link a store writer made to some root file elsewhere
        would otherwise be read back through this verb.
        """
        slug = repo_slug(runtime_template.GpuPoolRuntime._model(repo))
        with safe_fs.directory(self._model_store) as store:
            pull = safe_fs.open_child_directory(store, _PULL_FOLDER)
            try:
                self._require_sealed(pull, _PULL_FOLDER)
                return safe_fs.read_file(
                    pull, f"{slug}.json", limit=_PROGRESS_LIMIT,
                    owners=self._owners,
                )
            finally:
                os.close(pull)

    def remove_cache(self, repo: Any) -> str:
        """Remove ``repo``'s hub directory without leaving the store; its path back.

        Absent is success: the weights are gone either way, which is what the
        model library asked. A hub folder another account can change is
        refused rather than walked: it is not the hub root's containers wrote.
        """
        name = _hub_name(runtime_template.GpuPoolRuntime._model(repo))
        with safe_fs.directory(self._model_store) as store:
            try:
                hub = safe_fs.open_child_directory(store, _HUB_FOLDER)
            except FileNotFoundError:
                hub = None
            if hub is not None:
                try:
                    self._require_sealed(hub, _HUB_FOLDER)
                    safe_fs.remove_tree(hub, name)
                finally:
                    os.close(hub)
        return f"{self._model_store}/{_HUB_FOLDER}/{name}"

    @property
    def _owners(self) -> frozenset:
        return frozenset({0, self._owner[0]})

    def _require_sealed(self, handle: int, folder: str) -> None:
        if not safe_fs.is_sealed_directory(handle, owners=self._owners):
            raise RuntimeError(
                "The model store's '{}' folder can be changed by an account other "
                "than root, so this controller will not use it. Re-running the "
                "Vaelor installer returns it to root.".format(folder)
            )

    def _refuse_foreign(self, parent: int, name: str, below: str = "") -> None:
        found = safe_fs.first_foreign_entry(parent, name, owners=self._owners)
        if found is None:
            return
        where = below + found
        if safe_fs.is_hard_link_at(parent, found):
            # The installer leaves a hard link alone on purpose (re-owning it
            # would re-own the file it also names elsewhere), so it cannot be
            # the fix here: the owner removes that one name by hand.
            raise RuntimeError(
                "'{}' in the model store is a second name for a file that also "
                "lives somewhere else, so this controller will not start a root "
                "container on it. Remove it by hand (sudo rm {}/{}); the file's "
                "other name is untouched.".format(where, self._model_store, where)
            )
        raise RuntimeError(
            "'{}' in the model store was written by, or can be changed by, "
            "an account other than root, so this controller will not start a "
            "root container on it. Re-running the Vaelor installer returns "
            "the store's vLLM folders to root.".format(where)
        )

    def _prepare_store(self, model: Optional[str], *, pulling: bool) -> None:
        """Make the store's vLLM folders ready for a root container, or refuse.

        The compile folder for every unit, and for a pull the pull folder with
        the last run's progress record removed: each made (or taken back) as
        root's, none followed. Then everything the container will read from
        them is proven root's (:func:`bridge_safe_fs.first_foreign_entry`):
        taking a folder back does nothing about what another account already
        wrote inside it - a compiled kernel in the compile cache, a weight or
        a half-fetched blob in the hub - and a root container would run or load
        it. The hub itself, when present, must be a folder only root can
        change, and the repo's own tree in it root's throughout.
        """
        with safe_fs.directory(self._model_store) as store:
            compiled = safe_fs.ensure_root_directory(
                store, _COMPILE_FOLDER, owner=self._owner,
            )
            os.close(compiled)
            self._refuse_foreign(store, _COMPILE_FOLDER)
            if pulling:
                pull = safe_fs.ensure_root_directory(store, _PULL_FOLDER, owner=self._owner)
                try:
                    safe_fs.remove_file(pull, f"{repo_slug(model)}.json")
                finally:
                    os.close(pull)
                self._refuse_foreign(store, _PULL_FOLDER)
            if model is None:
                return
            try:
                hub = safe_fs.open_child_directory(store, _HUB_FOLDER)
            except FileNotFoundError:
                return
            try:
                self._require_sealed(hub, _HUB_FOLDER)
                self._refuse_foreign(hub, _hub_name(model), below=_HUB_FOLDER + "/")
            finally:
                os.close(hub)

    def _entry_program(self) -> str:
        """The installed MoE entry program's real path, once shown root's alone."""
        path = self._entry_program_path
        if path is None:
            from . import vllm_entry_program

            path = os.path.realpath(vllm_entry_program.__file__)
        if not safe_fs.is_root_controlled(path, owners=frozenset({0, self._owner[0]})):
            raise RuntimeError(
                "The installed vLLM entry program can be changed by an account "
                "other than root, so this controller will not run it as root."
            )
        return path

    def _pull_program(self) -> str:
        """The installed pull program's real path, once it is shown to be root's alone."""
        path = self._pull_program_path
        if path is None:
            from . import gpu_pull_program

            path = os.path.realpath(gpu_pull_program.__file__)
        if not safe_fs.is_root_controlled(path, owners=frozenset({0, self._owner[0]})):
            raise RuntimeError(
                "The installed model pull program can be changed by an account "
                "other than root, so this controller will not run it as root."
            )
        return path

    def _systemctl(self, *arguments: str) -> None:
        """One fixed ``systemctl`` call, built here and bounded; a failure raises."""
        command = [shutil.which("systemctl") or "/usr/bin/systemctl", *arguments]
        try:
            completed = (self._run or subprocess.run)(
                command, capture_output=True, text=True,
                timeout=self._step_seconds, check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "systemctl {} did not finish within {} seconds.".format(
                    arguments[0], self._step_seconds,
                )
            ) from error
        if completed.returncode != 0:
            raise RuntimeError(
                str(completed.stderr or "").strip()[:400]
                or "systemctl {} failed.".format(arguments[0])
            )
