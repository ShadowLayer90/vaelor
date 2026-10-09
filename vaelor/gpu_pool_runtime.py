"""Supervise the vLLM + Ray GPU runtime as CONTAINERS over pinned SSH.

The GPU-tier sibling of `pooled_runtime` (which drives the CPU `distributed-llama`
mesh). Where the CPU path shards a model across nodes' RAM, this one serves a
model on a GPU node's unified memory with **vLLM**, and, when a model is too big
for one node, across several GPU nodes with **Ray** as the process-group manager
and **RCCL/NCCL** as the collective transport (VD-054). Every shell/systemd
command that knows the docker, vLLM or Ray invocation lives here, exactly as the
pooled runtime is the one place that knows the ``dllama`` invocation, so the
operations layer never assembles a command line and every invocation is
unit-testable with a mocked transport.

**MB1: the runtime is a self-contained CONTAINER, not a pre-built host tree.**
vLLM runs on Strix Halo (gfx1151) ONLY as the :data:`VLLM_IMAGE` container - the
bare-artifact design this replaces could not be built at all - so every process
here is a ``docker run`` of that one image, mirroring the single-node llama.cpp
pivot in :mod:`vaelor.gpu_rocmfpx_service`: the image is ensured once
(`GpuPoolRuntime.ensure_image`), the accelerator devices are granted with NUMERIC
per-node GIDs (`GpuPoolRuntime.resolve_group_ids`), and one private builder
(``_docker_run_prefix``) emits the run flags so the lead's merged
head-and-server container and the workers cannot drift apart.

Design constraints carried from DECISIONS.md:

* **SSH + sudo + systemd, no Swarm.** RCCL is inherently a remote-node feature,
  so it lands on the pooled path's mechanism (VD-054 tension 1), not a second
  cluster membership.
* **Every command clears BOTH transport boundaries.** `SshTransport.run` permits
  a fixed set of first-argv tokens on a worker; on the CONTROLLER (VD-125) the
  reads and ``docker`` verbs cross the root bridge, where `bridge_argv_policy`
  pins each token to the exact shape emitted here. ``docker`` clears both, which
  is what makes the container pivot possible without widening either surface.
* **One template, and the controller's root side renders it (VD-143).** A unit
  is rendered by `render_managed_unit` from its kind and typed values. A worker
  gets the text over SSH; the controller's bridge gets the kind and the values
  and renders the unit itself, so no client-composed text is ever written as
  root there - and the model store is touched on the controller only by the
  bridge's own no-follow file operations (`bridge_managed_units`).
* **Host networking everywhere.** Ray's GCS and object-manager ports, and the
  RCCL/Gloo sockets, must be reachable across nodes on the real NIC; a bridged
  network would NAT them. So every container runs ``--network host`` and nothing
  is ``-p`` published - the vLLM server binds ``0.0.0.0:<port>`` on the host.
* **Ray is told, never asked** (VD-054 tension 2). Ray is a process group inside
  one deployment, never a source of membership or health; the lead's head and
  the vLLM server share ONE container (`GpuPoolRuntime.start_distributed_server`).
* **Strix Halo memory is unified GTT, not VRAM.** The fit engine sizes against
  GTT; this module only needs the ``--gpu-memory-utilization`` fraction the
  operator (or Easy-mode default) chose.
* **Untested on two nodes** (VD-054): the container commands are the documented
  vLLM/Ray forms, emitted and unit-tested, but not claimed to have served a
  model across two GPUs. The operations layer reports health from a real probe,
  never from "the unit was written".
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from . import cluster_link, gpu_node_facts
#: Every serving unit installed is noted with what it was rendered from, so an
#: upgrade can tell whether this release renders it differently (W4-D1).
from . import gpu_render_ledger as render_ledger
#: Every value a unit is rendered from is checked by its one rule in
#: `gpu_unit_params` (VD-143): the root bridge renders the controller's units
#: from the same values, so the rules must be callable on that side too.
from . import gpu_unit_params as unit_params
#: `gpu_pool_units` owns every name this module writes into a node's systemd -
#: the serving container and unit per role, the pull oneshot per repo, the
#: gate's config file - and the derivation "which units does a stored record
#: own" that `gpu_pool_operations` stops by. The starts below spell their names
#: through the same functions, so what a deploy starts and what a record
#: derives cannot be two spellings.
from .gpu_pool_units import (  # noqa: F401 - DOCKER is re-exported for its readers
    DOCKER, GATE_CONFIG_ROOT, GATE_ROLE, RAY_WORKER_ROLE, SERVER_ROLE,
    UNIT_AFTER, container_name, deployment_name, gate_config_path,
    pull_container_name, pull_unit_name, pull_unit_text, repo_slug,
    require_pull_unit, require_serving_unit, unit_name, unit_role, unit_text,
)
#: A worker replica's gate (VD-129, the amendment) is `gpu_pool_gate`'s, and
#: `root_renders_units` says whether a transport is the controller's root
#: bridge, which renders every unit it installs itself (VD-143).
from .bridge_transport import root_renders_units
from . import gpu_ray_plane as ray_plane
from .gpu_pool_gate import GateMixin
#: The model-library pull - its unit, its start/poll pair, its paths - is
#: `gpu_pool_pull_units`'s, inherited; its names are re-exported for readers.
from .gpu_pool_pull_units import (  # noqa: F401 - re-exported for their readers
    MODEL_PULL_KIND, PYTHON_ENTRYPOINT, UNIT_DIRECTORY, PullUnitsMixin,
    _CONTAINER_PACKAGED_PROGRAM, _CONTAINER_PULL_ROOT, _CONTAINER_PULL_SCRIPT,
    _ENTRY_SCRIPT_PATH, _PROGRAM_PATH, _PULL_PROGRESS_ROOT, _PULL_SCRIPT_PATH,
    _PULL_UNIT_PROPERTIES,
)
from .gpu_pull_program import PULL_SCRIPT, PULL_SCRIPT_DIGEST, PULL_SCRIPT_NAME
#: The ``vllm serve`` words both serving forms share are built in one place
#: (`gpu_serve_arguments`): the model's serving options, the profiler, then the
#: thinking default. The options' container variable and entry program are
#: `vllm_serve_options`' and `vllm_entry_program`'s; the thinking default's
#: unit value is `model_thinking`'s.
from .gpu_serve_arguments import serve_arguments
from .model_thinking import unit_thinking_value
from .vllm_serve_options import (
    OPTIONS_FIELD, bridge_refusal, container_environment, wants_tuned_tables,
)
from .vllm_entry_program import (
    CONTAINER_ENTRY_PROGRAM, ENTRY_INTERPRETER, ENTRY_SCRIPT, ENTRY_SCRIPT_DIGEST,
    ENTRY_SCRIPT_NAME, TUNED_FOLDER_ENV, entry_command, entry_mount,
    write_program_on_worker,
)
#: The one host a served API may bind, as `gpu_serving_target` spells it.
from .gpu_serving_target import SERVER_LOOPBACK_HOST
from .ssh_transport import SshTransportError
#: The host/container paths the on-demand GPU-kernel profile (VD-128) dumps to.
from .serving_profiler import SERVING_PROFILES_DIR, SERVING_PROFILES_MOUNT  # noqa: F401

#: The container's shape - the pinned default image, the devices, the runtime
#: flags, the model and compile caches and their mounts - is `vllm_container`'s,
#: re-exported here under the names every reader has always used; the images
#: themselves, and what each needs, are `vllm_images`'.
from .vllm_container import (  # noqa: F401 - re-exported for their readers
    COMPILE_CACHE_ENV, COMPILE_CACHE_ROOT, COMPILE_MOUNT, CONTAINER_RUNTIME_FLAGS,
    GPU_DEVICES, HSA_OVERRIDE_ENV, MODEL_CACHE_ROOT, MODELS_MOUNT, VLLM_IMAGE,
    VLLM_IMAGES, docker_run_prefix, rccl_environment,
)
from .bridge_argv_policy import journal_argv
from .vllm_images import (
    DEFAULT_IMAGE_KEY, IMAGE_FIELD, IMAGES, RECORDED_IMAGE_KEY, image_profile,
)
from .vllm_serve_options import image_of

#: The 0.22.1 image's entrypoint is a thin ``vllm "$@"``; the 0.27 image has
#: none (`vllm_images`). So a Ray, shell or Python process is run by naming it,
#: and all live on both images' ``PATH``; ``vllm`` is named too because the
#: merged lead runs ``bash`` and then has to spell the server binary itself.
RAY_ENTRYPOINT = "ray"
BASH_ENTRYPOINT = "bash"
VLLM_BINARY = "vllm"

#: The Ray head's GCS port. vLLM's Ray backend connects workers to
#: ``<head-ip>:<port>``; fixed here so the head and worker units cannot disagree.
RAY_PORT = 6379

#: Both ``ray start`` commands opt out of Ray's usage telemetry. Only the WORKER
#: adds ``--block``: the lead's head runs inside the server container and is
#: followed by ``exec vllm serve``, which is what must hold the foreground there.
_RAY_USAGE_FLAG = "--disable-usage-stats"
_RAY_BLOCK_FLAG = "--block"

_CONTAINER_RUNTIME_FLAGS = CONTAINER_RUNTIME_FLAGS

#: The systemd properties `unit_state` reads of a SERVING unit in one
#: ``systemctl show`` - the only read this module makes of one - and the
#: ``LoadState`` value that means systemd has never heard of the unit.
#: `stop_unit` reads it before it acts: a stop is issued for every unit a record
#: DERIVES (`gpu_pool_units.serving_units`), not only for units known to have
#: started, so a unit that was never written - the worker a deploy died before
#: reaching, or one an earlier stop already removed - has to be a no-op rather
#: than a failure that reads to the operator as "still running". The deploy's
#: startup wait (`gpu_pool_startup`) reads the SAME properties between health
#: probes to know whether the server it is waiting on is still alive, and
#: ``Result``/``ExecMainStatus`` are what it can say about a unit that is not -
#: the allowlists carry no ``journalctl``, so the exit status is the whole of
#: the evidence the box will hand over.
_SERVING_UNIT_PROPERTIES = "LoadState,ActiveState,SubState,Result,ExecMainStatus"
_LOAD_STATE = "LoadState"
_UNIT_NOT_FOUND = "not-found"

#: The one further variable an MoE model's containers carry: vLLM's own
#: tuned-table folder, filled inside the container (`vllm_entry_program`).
TUNED_TABLES_ENV = TUNED_FOLDER_ENV



#: The name `_degree` reports when a node's GPU count is out of range, written
#: once because both Ray starts validate the same field.
_DEVICE_COUNT_FIELD = "device count"


#: The unit kinds this runtime installs, and the parameter NAMES each is rendered
#: from (VD-143). On a worker the runtime renders and writes the unit over SSH;
#: on the controller it hands the root bridge the kind and these values - never a
#: line of unit text - and the bridge renders the same unit through
#: `render_managed_unit` itself, refusing any name not listed here. So what a
#: client can choose is a value of one of these, each checked by its rule in
#: `gpu_unit_params`, and nothing else: not the image, the mounts, the
#: environment, the user or the command.
REPLICA_KIND = "replica"
DISTRIBUTED_LEAD_KIND = "distributed-lead"
RAY_WORKER_KIND = RAY_WORKER_ROLE
_SERVING_VALUES = (
    "name", "model", "port", "tensor_parallel_size", "gpu_memory_utilization",
    "max_model_len", "group_ids", "revision", "thinking_default", OPTIONS_FIELD,
)
MANAGED_UNIT_PARAMETERS = {
    REPLICA_KIND: frozenset(_SERVING_VALUES),
    DISTRIBUTED_LEAD_KIND: frozenset(_SERVING_VALUES + (
        "pipeline_parallel_size", "bind_ip", "interface", "device_count",
        "api_host",
    )),
    RAY_WORKER_KIND: frozenset({
        "name", "head_ip", "bind_ip", "interface", "group_ids", "device_count",
        OPTIONS_FIELD,
    }),
    # The image a pull runs in, named only when it is not the one a pull
    # that names none has always run in (`start_model_pull`).
    MODEL_PULL_KIND: frozenset({"repo", "revision", IMAGE_FIELD}),
}



#: Kept under its historical name: `gpu_unit_params` is the rule's one home.
_shell_safe = unit_params.shell_safe


class GpuPoolRuntime(GateMixin, PullUnitsMixin, ray_plane.RayPlaneMixin):
    """Emit the reviewed vLLM/Ray container commands over a pinned SSH transport."""

    #: Each value's one rule, bound under the names the runtime always used.
    _name = staticmethod(deployment_name)
    _address = staticmethod(unit_params.cluster_address)
    _interface = staticmethod(unit_params.interface)
    _model = staticmethod(unit_params.model_repo)
    _revision = staticmethod(unit_params.revision)
    _degree = staticmethod(unit_params.degree)
    _context = staticmethod(unit_params.context_length)
    #: The serving-port band and the API bind-host allowlist live in
    #: `gpu_serving_target`, and the launch fraction's band in the fit engine
    #: (VD-129): one rule each, reached through `gpu_unit_params`.
    _port = staticmethod(unit_params.port)
    _api_host = staticmethod(unit_params.api_host)
    _fraction = staticmethod(unit_params.memory_fraction)

    def ensure_image(self, transport, image: str = None, on_download=None) -> str:
        """Make sure ``image`` (a `vllm_images` key; the default for ``None``)
        is on the node, pulling it once if not.

        The container replacement for the old artifact ``install``, in the shape
        `gpu_rocmfpx_service.GpuServerProcess.ensure_image` uses on the
        single-node tier: ``docker image inspect`` is the cheap presence probe and
        only a miss triggers a ``docker pull`` (generous timeout - the image is
        multi-GB). A failed pull names the image, not an obscure later failure.
        ``on_download`` is called with the image's profile just before a
        download starts - and not at all when the image is already there - so
        a job can say what it is waiting on (`vllm_images.download_line`).
        """
        profile = image_profile(image)
        pinned = profile.image
        try:
            transport.run(["docker", "image", "inspect", pinned], sudo=True)
            return pinned
        except SshTransportError:
            pass
        if on_download is not None:
            on_download(profile)
        try:
            transport.run(["docker", "pull", pinned], sudo=True, timeout=3600)
        except SshTransportError as error:
            raise RuntimeError(
                "The vLLM serving image {} is not present on this node and could "
                "not be pulled: {}".format(pinned, error)
            ) from error
        return pinned

    def pull_image_key(self, transport) -> str:
        """The image a model-library pull on this node runs in.

        A library pull serves no deployment, so it has no image of its own:
        it runs in a pinned image the node ALREADY has (the default first),
        and only a node with none of them is sent the default. A machine that
        serves on the earlier image is not made to download another one to
        fetch weights.
        """
        ordered = [DEFAULT_IMAGE_KEY] + [key for key in IMAGES if key != DEFAULT_IMAGE_KEY]
        for key in ordered:
            try:
                transport.run(["docker", "image", "inspect", IMAGES[key].image], sudo=True)
                return key
            except SshTransportError:
                continue
        return DEFAULT_IMAGE_KEY

    def resolve_group_ids(self, transport) -> List[int]:
        """The node's numeric render/video GIDs (`gpu_node_facts`)."""
        return gpu_node_facts.resolve_group_ids(transport)

    def resolve_interface(self, transport, address: str) -> str:
        """The NIC carrying ``address`` on this node (`gpu_node_facts`)."""
        return gpu_node_facts.resolve_interface(transport, address)

    def read_link_tables(self, transport) -> Any:
        """The addresses this node holds and the link carrying each (`cluster_link`).

        VD-162: with a cluster link chosen, a split binds the address a node
        holds on THAT link's network; this is the read that finds it.
        """
        return cluster_link.read_node_tables(transport)

    def read_local_link_tables(self) -> Any:
        """This controller's own addresses and links, read where this runs."""
        return cluster_link.local_node_links()

    #: The ``docker run`` shape and the RCCL env (`vllm_container`), bound
    #: under the names the renderers have always called.
    _docker_run_prefix = staticmethod(docker_run_prefix)
    _rccl_environment = staticmethod(rccl_environment)

    # --- One template, two writers (VD-143) ---------------------------------

    def render_managed_unit(
        self, kind: Any, values: Any, *, pull_program: Optional[str] = None,
        entry_program: Optional[str] = None,
    ) -> Tuple[str, str]:
        """``(unit name, unit text)`` for one managed unit, from its kind and values.

        THE template: the worker path writes what this returns over SSH, and the
        controller's root bridge calls it on the values a client sent and writes
        what it returns. Every value passes its `gpu_unit_params` rule inside
        the kind's renderer, and a name the kind does not take is refused here,
        before any renderer runs - which is what stops a client adding a mount,
        an environment variable, a user or an ``ExecStart`` of its own.

        ``pull_program`` is the one input that is not a client value: the host
        path of the pull program the controller's bridge mounts out of its own
        installed package. It is a keyword the bridge passes itself, and no
        value a client sends can reach it; ``entry_program`` is the same, for
        the program an MoE model's serving container starts through.
        """
        allowed = MANAGED_UNIT_PARAMETERS.get(str(kind))
        if allowed is None:
            raise ValueError("That is not a unit kind the GPU tier installs.")
        if not isinstance(values, Mapping):
            raise ValueError("A managed unit is rendered from named values.")
        unknown = sorted(str(key) for key in values if key not in allowed)
        if unknown:
            raise ValueError(
                "A {} unit does not take {}.".format(kind, ", ".join(unknown))
            )
        arguments = {str(key): value for key, value in values.items()}
        if kind == MODEL_PULL_KIND:
            return self.render_model_pull(pull_program=pull_program, **arguments)
        renderer = {
            REPLICA_KIND: self.render_single_server,
            DISTRIBUTED_LEAD_KIND: self.render_distributed_server,
            RAY_WORKER_KIND: self.render_ray_worker,
        }[str(kind)]
        return renderer(entry_program=entry_program, **arguments)

    def _install_unit(self, transport, kind: str, values: Dict[str, Any]) -> str:
        """Render one managed unit and put it on the node the transport reaches.

        Rendered here on BOTH paths, so a bad value is refused before anything
        crosses a wire. On the controller the kind and the values - not the
        text - go to the root bridge, which renders the same unit itself; on a
        worker the text is written over SSH, where the login is already a
        sudoer and the write is hygiene rather than a boundary. A caller with
        no serving options sends none, so its values are what they always were.
        """
        if values.get(OPTIONS_FIELD, False) is None:
            values = {key: value for key, value in values.items() if key != OPTIONS_FIELD}
        unit, text = self.render_managed_unit(kind, values)
        if root_renders_units(transport):
            try:
                transport.install_managed_unit(kind, values)
            except SshTransportError as error:
                # A bridge still running the previous version refuses a value
                # it does not know by name: say what to do, not the name.
                too_old = bridge_refusal(error, values)
                if too_old is None:
                    raise
                raise RuntimeError(too_old) from error
            render_ledger.note(transport, kind, values)
            return unit
        if wants_tuned_tables(values.get(OPTIONS_FIELD)):
            transport.run(["install", "-d", "-m", "0755", _PULL_PROGRESS_ROOT], sudo=True)
            write_program_on_worker(
                transport, _ENTRY_SCRIPT_PATH, ENTRY_SCRIPT, ENTRY_SCRIPT_DIGEST,
            )
        self._write_unit_text(
            transport, unit, text, oneshot=(kind == MODEL_PULL_KIND),
        )
        # Noted only once installed, for the upgrade's render check (W4-D1).
        render_ledger.note(transport, kind, values)
        return unit

    @staticmethod
    def _write_unit_text(transport, unit: str, text: str, *, oneshot: bool) -> None:
        """A WORKER's unit write: the text over SSH, a reload, then the start.

        A pull is a oneshot, started without blocking and after clearing any
        earlier failed state under its name (a failed oneshot otherwise refuses
        to start until reset); a serving unit is enabled and started.
        """
        transport.run(
            ["tee", f"{UNIT_DIRECTORY}/{unit}"], sudo=True, stdin_text=text,
        )
        transport.run(["systemctl", "daemon-reload"], sudo=True)
        if not oneshot:
            # A unit that hit its start limit (a split's Ray unit gives up
            # after five failed starts) refuses `enable --now` until reset, so
            # a Load soon after would fail for no current reason (review 4).
            try:
                transport.run(["systemctl", "reset-failed", unit], sudo=True)
            except (SshTransportError, RuntimeError):
                pass
            transport.run(["systemctl", "enable", "--now", unit], sudo=True)
            return
        try:
            transport.run(["systemctl", "reset-failed", unit], sudo=True)
        except SshTransportError:
            pass
        transport.run(["systemctl", "start", "--no-block", unit], sudo=True)

    @staticmethod
    def _entry(vllm_options: Any, entry_program: Optional[str]) -> List[str]:
        """The read-only mount an MoE model's container starts its entry
        program from (`vllm_entry_program`), or none for any other model."""
        if not wants_tuned_tables(vllm_options):
            return []
        host = str(entry_program or _ENTRY_SCRIPT_PATH)
        if not re.fullmatch(_PROGRAM_PATH, host) or ".." in host:
            raise ValueError("The vLLM entry program path is invalid.")
        return entry_mount(host)

    def render_ray_worker(
        self, *, name: str, head_ip: str, bind_ip: str, interface: str,
        group_ids: Iterable[int], device_count: int = 1, vllm_options: Any = None,
        entry_program: Optional[str] = None,
    ) -> Tuple[str, str]:
        """``(unit, text)`` joining a worker GPU node to the deployment's Ray cluster.

        ``ray start --address <head-ip>:<port>`` is the join; the head - which
        lives inside the lead's server container - is started first. ``--block``
        keeps the join in the foreground so systemd, not Ray's daemonisation,
        supervises it. ``--num-gpus`` is explicit because Ray does not reliably
        detect the AMD iGPU in the container, and a node advertising zero GPUs
        takes no placement.
        """
        deployment = self._name(name)
        head = self._address(head_ip)
        host = self._address(bind_ip)
        nic = self._interface(interface)
        container = container_name(deployment, RAY_WORKER_ROLE)
        image = image_of(vllm_options)
        mount = self._entry(vllm_options, entry_program)
        command = self._docker_run_prefix(
            container_name=container, group_ids=group_ids, image=image,
            envs=self._rccl_environment(host, nic) + [*ray_plane.RAY_CONTAINER_ENVIRONMENT]
            + container_environment(vllm_options),
        ) + ray_plane.token_mount(deployment) + ray_plane.cgroup_parent(deployment) + mount + (
            ["--entrypoint", ENTRY_INTERPRETER, image.image, CONTAINER_ENTRY_PROGRAM,
             RAY_ENTRYPOINT] if mount else ["--entrypoint", RAY_ENTRYPOINT, image.image]
        ) + [
            "start",
            "--address", f"{head}:{RAY_PORT}", "--node-ip-address", host,
            "--num-gpus", str(self._degree(device_count, _DEVICE_COUNT_FIELD)),
            _RAY_USAGE_FLAG, *ray_plane.RAY_PORT_PINS, _RAY_BLOCK_FLAG,
        ]
        return unit_name(deployment, RAY_WORKER_ROLE), unit_text(
            description=f"Vaelor GPU inference Ray worker ({deployment})",
            container_name=container, exec_start=" ".join(command),
            start_pre=ray_plane.boot_start_pre(deployment),
            ray_slice=ray_plane.slice_name(deployment),
            slice_checks=(ray_plane.slice_check_pre(),
                          ray_plane.slice_check_post(deployment, RAY_WORKER_ROLE)),
        )

    def start_ray_worker(
        self, transport, *, name: str, head_ip: str, bind_ip: str,
        interface: str, group_ids: Iterable[int], device_count: int = 1,
        vllm_options: Any = None,
    ) -> str:
        """Join a worker GPU node to the deployment's Ray cluster (`render_ray_worker`)."""
        return self._install_unit(transport, RAY_WORKER_KIND, {
            "name": name, "head_ip": head_ip, "bind_ip": bind_ip,
            "interface": interface, "group_ids": list(group_ids),
            "device_count": device_count, OPTIONS_FIELD: vllm_options,
        })

    def render_single_server(
        self, *, name: str, model: str, port: int,
        tensor_parallel_size: int, gpu_memory_utilization: float,
        max_model_len: int, group_ids: Iterable[int],
        revision: Optional[str] = None, thinking_default: bool = False,
        vllm_options: Any = None, entry_program: Optional[str] = None,
    ) -> Tuple[str, str]:
        """``(unit, text)`` of one REPLICA: a single-node vLLM server container (no Ray).

        **Single-node vLLM is not offered to users**: the single-node GPU tier is
        llama.cpp (Mode A, `gpu_rocmfpx_service`). This is the throughput
        intent's per-machine replica (VD-129) - one whole copy of the model on
        each selected node, the live-verified single-node runtime, behind the
        balancer the controller runs. Tensor-parallel is *within* the node;
        ``N`` is 1 on a single-GPU Strix Halo. No collective runs, so no
        cluster interface is resolved and no RCCL environment is set.

        **A replica binds loopback, always, and carries no key** - and the
        bind is not a parameter, so no caller can choose the LAN. The
        controller's replica sits behind the balancer; a worker's sits behind
        its own gate (:meth:`start_gate`), which is the only thing on that
        machine listening on the cluster address. vLLM's own ``--api-key`` is
        not used because it was live-probed (the VD-129 amendment) to gate
        ``/v1`` only: ``POST /invocations``, ``/tokenize`` and ``/metrics``
        answered a keyed vLLM 0.22.1 with no key at all, so a keyed replica
        bound to the LAN was an unkeyed inference endpoint through a side
        door. The unit carries ``StartLimitIntervalSec=0`` so a crashed
        replica is relaunched however often it crashes - it is one of N, and
        the balancer routes around it.
        """
        deployment = self._name(name)
        container = container_name(deployment, SERVER_ROLE)
        image = image_of(vllm_options)
        prefix = self._docker_run_prefix(
            container_name=container, group_ids=group_ids, image=image,
            envs=container_environment(vllm_options),
        )
        mount = self._entry(vllm_options, entry_program)
        if mount:
            start = ["--entrypoint", ENTRY_INTERPRETER, image.image,
                     CONTAINER_ENTRY_PROGRAM, VLLM_BINARY]
        elif image.server_entrypoint:
            start = ["--entrypoint", image.server_entrypoint, image.image]
        else:
            start = [image.image]
        command = " ".join(prefix + mount + start + self._serve_arguments(
            model=model, port=port,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len, distributed=False, revision=revision,
            api_host=SERVER_LOOPBACK_HOST, thinking_default=thinking_default,
            vllm_options=vllm_options,
        ))
        return unit_name(deployment, SERVER_ROLE), unit_text(
            description=f"Vaelor GPU inference server ({deployment})",
            container_name=container, exec_start=command,
            unlimited_restarts=True,
        )

    def start_single_server(
        self, transport, *, name: str, model: str, port: int,
        tensor_parallel_size: int, gpu_memory_utilization: float,
        max_model_len: int, group_ids: Iterable[int],
        revision: Optional[str] = None, thinking_default: bool = False,
        vllm_options: Any = None,
    ) -> str:
        """Launch one REPLICA on the node (`render_single_server`)."""
        return self._install_unit(transport, REPLICA_KIND, {
            "name": name, "model": model, "port": port,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len, "group_ids": list(group_ids),
            "revision": revision, **unit_thinking_value(model, thinking_default),
            OPTIONS_FIELD: vllm_options,
        })

    def render_distributed_server(
        self, *, name: str, model: str, port: int,
        tensor_parallel_size: int, pipeline_parallel_size: int,
        gpu_memory_utilization: float, max_model_len: int,
        bind_ip: str, interface: str, device_count: int,
        group_ids: Iterable[int], revision: Optional[str] = None,
        api_host: str = SERVER_LOOPBACK_HOST, thinking_default: bool = False,
        vllm_options: Any = None, entry_program: Optional[str] = None,
    ) -> Tuple[str, str]:
        """``(unit, text)`` of the lead's ONE container: the Ray head and the vLLM server.

        The head and the server must share a container, not merely a node. A
        vLLM driver's ``ray.init(address=…)`` attaches to a **local** raylet
        through the session's unix socket under the head's ``/tmp/ray``, which a
        second container cannot see however reachable the GCS port is - so a
        separate server container could not join at all. ``vllm serve`` inside
        the head container is the form vLLM's own multi-node documentation uses.
        Its API binds loopback on every lead (``api_host`` admits nothing
        else); Ray binds ``bind_ip``, the chosen cluster link.

        The head starts WITHOUT ``--block`` and ``exec vllm serve`` then becomes
        the container's foreground process, so systemd supervises the server and
        ``--rm`` cleans up when it exits. ``--distributed-executor-backend ray``
        places its ``tensor * pipeline`` workers across the Ray nodes; the RCCL
        env pins the collective transport to the private NIC. The fit engine
        chooses the degrees - pipeline across nodes, tensor only on a fast,
        KV-head-divisible fabric.
        """
        deployment = self._name(name)
        host = self._address(bind_ip)
        nic = self._interface(interface)
        gpus = self._degree(device_count, _DEVICE_COUNT_FIELD)
        container = container_name(deployment, SERVER_ROLE)
        image = image_of(vllm_options)
        prefix = self._docker_run_prefix(
            container_name=container, group_ids=group_ids, image=image,
            envs=self._rccl_environment(host, nic) + [*ray_plane.RAY_CONTAINER_ENVIRONMENT]
            + container_environment(vllm_options),
        ) + ray_plane.token_mount(deployment) + ray_plane.cgroup_parent(deployment) + (
            self._entry(vllm_options, entry_program))
        server = [VLLM_BINARY]
        if wants_tuned_tables(vllm_options):
            server = entry_command(server)
        # The only shell syntax in the script is this literal ``&& exec``; every
        # other token is a validated value, and `_shell_safe` refuses any that
        # could add more.
        script = "{} && exec {}".format(
            _shell_safe(" ".join([
                RAY_ENTRYPOINT, "start", "--head", "--node-ip-address", host,
                "--port", str(RAY_PORT), "--num-gpus", str(gpus),
                _RAY_USAGE_FLAG, *ray_plane.HEAD_HARDENING, *ray_plane.RAY_PORT_PINS,
            ])),
            _shell_safe(" ".join(server + self._serve_arguments(
                model=model, port=port,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len, distributed=True,
                revision=revision, api_host=api_host,
                thinking_default=thinking_default, vllm_options=vllm_options,
            ))),
        )
        command = " ".join(
            prefix
            + ["--entrypoint", BASH_ENTRYPOINT, image.image, "-c", f'"{script}"']
        )
        return unit_name(deployment, SERVER_ROLE), unit_text(
            description=f"Vaelor distributed GPU inference server ({deployment})",
            container_name=container, exec_start=command,
            start_pre=ray_plane.boot_start_pre(deployment),
            ray_slice=ray_plane.slice_name(deployment),
            slice_checks=(ray_plane.slice_check_pre(),
                          ray_plane.slice_check_post(deployment, SERVER_ROLE)),
        )

    def start_distributed_server(
        self, transport, *, name: str, model: str, port: int,
        tensor_parallel_size: int, pipeline_parallel_size: int,
        gpu_memory_utilization: float, max_model_len: int,
        bind_ip: str, interface: str, device_count: int,
        group_ids: Iterable[int], revision: Optional[str] = None,
        api_host: str = SERVER_LOOPBACK_HOST, thinking_default: bool = False,
        vllm_options: Any = None,
    ) -> str:
        """Launch the lead's merged Ray head and vLLM server (`render_distributed_server`)."""
        return self._install_unit(transport, DISTRIBUTED_LEAD_KIND, {
            "name": name, "model": model, "port": port,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len, "bind_ip": bind_ip,
            "interface": interface, "device_count": device_count,
            "group_ids": list(group_ids), "revision": revision,
            "api_host": api_host, **unit_thinking_value(model, thinking_default),
            OPTIONS_FIELD: vllm_options,
        })

    @staticmethod
    def _serve_arguments(**arguments: Any) -> List[str]:
        """The ``serve <repo> ...`` words both serving forms share (`gpu_serve_arguments`).

        The single-node unit appends them to the image, whose entrypoint is a
        thin ``vllm "$@"``; the merged lead spells :data:`VLLM_BINARY` itself.
        """
        return serve_arguments(**arguments)

    def stop_unit(self, transport, unit: str) -> None:
        """Disable, stop, and remove one managed vLLM/Ray unit - idempotently.

        The name is matched against the exact managed shape so a caller can never
        disable an unrelated unit, the same defence as `pooled_runtime.stop_unit`.
        Stopping runs the unit's ``ExecStop``, draining the container with it.

        **A unit systemd does not know is a no-op, not a failure.** The callers
        stop what a record DERIVES, and a derived unit may never have been
        written (the deploy died before reaching that node) or may already be
        gone (an earlier stop removed it). ``systemctl disable --now`` on such a
        unit exits non-zero, which would reach the operator as "could not be
        stopped" about a unit that was never there. So systemd is ASKED first;
        a transport that cannot answer raises as it always did, because an
        unreachable node is the real failure.

        A gate's config file (VD-129) goes with the gate's unit: the one file
        on a node that carries the cluster key, ``rm -f``'d after the unit,
        and asked for only when the unit IS a gate (`gpu_pool_units.unit_role`)
        - a server or a Ray worker never had one, and neither does anything on
        the controller, whose bridge has no gate shape at all (VD-143).
        """
        state = self.unit_state(transport, unit)
        if state.get(_LOAD_STATE) != _UNIT_NOT_FOUND:
            transport.run(["systemctl", "disable", "--now", unit], sudo=True)
        transport.run(["rm", "-f", f"{UNIT_DIRECTORY}/{unit}"], sudo=True)
        transport.run(["systemctl", "daemon-reload"], sudo=True)
        # W4-D9: a removed unit that had failed stays listed "not-found failed"
        # until systemd is told to forget it. Tolerated: a unit systemd no
        # longer holds answers non-zero, which is the state wanted.
        try:
            transport.run(["systemctl", "reset-failed", unit], sudo=True)
        except (SshTransportError, RuntimeError):
            pass
        if unit_role(unit) == GATE_ROLE and not root_renders_units(transport):
            transport.run(["rm", "-f", gate_config_path(unit)], sudo=True)

    def unit_journal(self, transport, unit: str) -> str:
        """The last lines this run of a vLLM server unit wrote, as plain text.

        Read only when the server has died on start, for vLLM's own reason
        (`gpu_pool_startup.startup_refusal`); the one reviewed shape on both
        paths (`bridge_argv_policy.journal_argv`), which refuses any other unit.
        """
        return str(transport.run(journal_argv(unit), sudo=True))

    def unit_state(self, transport, unit: str) -> Dict[str, str]:
        """A managed serving unit's systemd state, read - never assumed.

        The ONE read this module makes of a serving unit
        (:data:`_SERVING_UNIT_PROPERTIES`): `stop_unit` asks it whether systemd
        has the unit loaded at all, and the deploy's startup wait asks it
        whether the server it is waiting on is still running (VD-127 - a wait
        that read only a clock killed a live server mid-compile). ``systemctl
        show`` exits 0 for a failed or unknown unit, so this raises only when
        the NODE cannot be asked, which is the caller's real failure.
        """
        require_serving_unit(unit)
        return self._show(transport, unit, _SERVING_UNIT_PROPERTIES)

    @staticmethod
    def health_url(endpoint: str) -> str:
        """The OpenAI model-list URL a health probe hits for ``endpoint``.

        ``endpoint`` ends in ``/v1``; a served vLLM answers ``/v1/models`` once
        ready - the signal operations waits on, never "the unit was written".
        """
        return endpoint.rstrip("/") + "/models"
