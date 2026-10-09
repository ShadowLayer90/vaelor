"""Launch primitives for the GPU model server (GPU AI-Chat), in a container.

The GPU analogue of :mod:`vaelor.flm_service`. The Z2's GPU AI-Chat model is
served by ``llama-server`` on the Strix Halo / Radeon 8060S (gfx1151), and this
module builds and supervises that launch.

**C0/C0-v2: the server is launched as a self-contained ROCm CONTAINER, not a bare
host binary, and the IMAGE + run interface are chosen by the model's catalog
``engine`` (:func:`vaelor.model_catalog.catalog_engine`), which the launch reads
from the ``runtime`` block.** The bare ``llama-server`` fork broke on the
re-imaged Strix Halo box (ROCm 7.2.3 vs the box's 7.14, a soname drift no
``LD_LIBRARY_PATH`` can bridge), so nothing runs on the host libs any more; the
container carries its own ROCm.

Two recipes, one per ``engine`` (:func:`_engine_container` selects between them):

* ``engine == ""`` (stock GGUF) -> the MAINLINE image
  ``docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0``, entrypoint overridden to
  ``llama-server``, served with auto device selection (``-m``/``-ngl``, internal
  port 8080, model dir at ``/models``). The fork-only flags (``-dev Vulkan0``,
  ``-ctv turbo4``, ``--spec-*``) are DROPPED: mainline's device is ``ROCm0`` not
  ``Vulkan0`` and it cannot even read the FP4 tensors.
* ``engine == "rocmfpx"`` (fork-only FP4 27B) -> the FORK image
  ``ghcr.io/julianmb/q38rocm:latest``, whose ``/app/run_server.sh`` entrypoint
  OWNS the recipe: it auto-detects the device and applies the cache/MTP profile
  itself, so the launch passes only the model (mounted at ``/app/models``),
  ``--host``/``--port`` (internal 8000) and ``--profile speed`` - never
  ``-dev``/``-ngl``/``-ctv``/``--spec-*``. It needs ``--ipc host`` and the
  ``GGML_HIP_ENABLE_UNIFIED_MEMORY`` env.

Both endpoints the supervisor health-checks are ``http://127.0.0.1:<published>/v1``;
the internal port differs (8080 vs 8000) but the published host port does not.

**The model container ALWAYS binds LOOPBACK, and never carries an api key.** The
LAN exposure of the model (the "LLM Server" feature) is NOT done by binding the
engine to the LAN with a key - it was proven live on the Z2 that the ROCmFP4
fork's ``llama-server`` does not enforce ``--api-key``/``--api-key-file``/
``LLAMA_API_KEY`` at all, so an engine-level key was an unauthenticated LAN
endpoint. Instead, LAN exposure is a Vaelor-controlled auth PROXY
(:mod:`vaelor.llm_server_proxy`) placed in front of this loopback endpoint. So the
container here is unconditionally published on ``127.0.0.1:<port>`` and passed no
key: there is no code path by which this model binds the network, whichever engine
runs. AI Chat reaches the model on that loopback endpoint, keyless.

The launch crosses the **root hardware bridge** (``gpu_start``/``gpu_stop``/
``gpu_status``): the workload executor's sandbox hides ``/dev/dri`` + ``/dev/kfd``,
so the privileged bridge account runs ``docker run`` with the accelerator devices
passed through. The container binds ``0.0.0.0`` INSIDE its own network namespace
(docker forwards the ``-p`` publish to the container's bridge IP, never its
loopback) and the port is published onto the HOST's ``127.0.0.1`` only.

**The host-binary provisioning (:data:`GPU_ENGINE_BINARY`, :func:`resolve_rocm_lib_dir`,
the ``/opt/rocm`` runtime discovery) is SUPERSEDED by the image for serving.** The
container needs none of it, so :mod:`vaelor.gpu_model_choice`'s capability gate no
longer gates availability on the host fork or host ROCm libs - it gates on the GPU
and a usable Docker (the container prerequisites) and reads these only as
INFORMATIONAL fields. The symbols are retained because that gate still reports
them. The installer no longer provisions the bare fork at all - it pre-pulls the
two images below instead (``prepull_serving_images`` in
``deploy/install-vaelor.sh``) - so on a box installed from this release the
informational binary field simply reads absent while the model serves.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from .flm_service import _validate_port
#: The model-path rules the launch enforces live in `gpu_model_confinement`
#: (ACC-063/064: one owner, read by the executor's download check too); they are
#: re-exported here so the launch's callers and tests keep one import.
from .gpu_model_confinement import (  # noqa: F401 - re-exported
    HOST_FILESYSTEM, MODEL_CACHE_DIR, GpuModelPathError, ModelFilesystem,
    _confined_model_path, _is_real_directory,
    _validate_model_path, anchor_refusal, servable_cache_relative,
)
from .inference_context import GPU_RECOMMENDED_CONTEXT_TOKENS
from .model_catalog import RUNTIME_Z2_GPU_ROCMFP4
from .platforms.accelerators import device_grants
from .runtime_paths import data_path


#: The ROCmFPX ``llama-server`` fork binary. **Superseded by the container image
#: for serving** (see the module docstring): the launch no longer runs this path,
#: and the installer no longer places it. Kept as a module constant because
#: :mod:`vaelor.gpu_model_choice` still reads it as an INFORMATIONAL field (it no
#: longer gates availability on it), so on a freshly installed box this path is
#: simply absent and reported as such - and on a box installed by an earlier
#: release the copy already there is left alone rather than deleted.
GPU_ENGINE_BINARY = "/var/lib/vaelor/engines/rocmfpx/bin/llama-server"

#: The legacy Lemonade-snap TheRock cache, the first candidate in
#: :data:`ROCM_RUNTIME_LIB_CANDIDATES`. Retained for the capability gate only; the
#: container carries its own ROCm and needs none of these host paths.
THEROCK_LIB_DIR = (
    "/var/snap/lemonade-server/common/cache/lemonade/bin/therock/gfx1151-7.13.0/lib"
)

#: The soname whose presence marks a directory as a real gfx1151 ROCm runtime dir.
#: Probed by :func:`resolve_rocm_lib_dir` (capability gate only).
ROCM_RUNTIME_PROBE_SONAME = "libamdhip64.so.7"

#: Ordered candidate roots for the gfx1151 ROCm/HIP runtime lib dir, most specific
#: first. Used by :mod:`vaelor.gpu_model_choice`'s capability gate; not by the
#: container launch, which bundles its own ROCm.
ROCM_RUNTIME_LIB_CANDIDATES: Sequence[str] = (
    THEROCK_LIB_DIR,
    "/opt/rocm/lib",
    "/opt/rocm/core-*/lib",
)


def _rocm_version_key(path: str) -> List[int]:
    """The integer components of a path, so ``core-7.14`` sorts above ``core-7.9``.

    A plain lexical sort would rank ``core-7.9`` above ``core-7.14`` (``'9' > '1'``)
    and pick the older runtime; comparing the numbers instead keeps "highest
    version wins" true across the two-vs-one-digit boundary.
    """
    return [int(number) for number in re.findall(r"\d+", path)]


def resolve_rocm_lib_dir(
    candidates: Sequence[str] = ROCM_RUNTIME_LIB_CANDIDATES,
    *,
    probe_soname: str = ROCM_RUNTIME_PROBE_SONAME,
) -> Optional[str]:
    """The first candidate directory that exists AND holds the ROCm runtime.

    Retained for :mod:`vaelor.gpu_model_choice`'s capability gate. Walks the
    candidates in order and returns the first directory that actually contains
    ``probe_soname``; within a globbed candidate the highest version wins. Returns
    ``None`` when nothing resolves. Pure and injectable for tests.
    """
    for candidate in candidates:
        matches = sorted(glob.glob(candidate), key=_rocm_version_key, reverse=True)
        for directory in matches:
            if os.path.isfile(os.path.join(directory, probe_soname)):
                return directory
    return None


#: The host the model is published on: LOOPBACK, always. The model is never bound
#: to the network; the LLM Server feature exposes it on the LAN through a separate
#: auth proxy (:mod:`vaelor.llm_server_proxy`), not by moving this bind.
GPU_HOST = "127.0.0.1"

#: The measured Strix-Halo launch parameters, the single source the catalog entry
#: also carries. Used as the fallback when a caller's ``runtime`` dict omits a
#: field, so the argv is always the proven recipe even from a bare ``{}``.
GPU_MEASURED_DEFAULTS: Dict[str, Any] = dict(RUNTIME_Z2_GPU_ROCMFP4)


#: The engine key of the fork-only FP4 model, matched against ``runtime["engine"]``
#: to select the FORK recipe. Mirrors the catalog entry's ``engine`` (kept in sync
#: there, in one file). The empty string takes the MAINLINE recipe (stock
#: llama.cpp reads stock GGUFs); any OTHER value is refused at the root boundary
#: (ACC-060) - an engine the bridge does not know is a claim it cannot check, so
#: it is never quietly served on the Strix Halo image.
ROCMFPX_ENGINE = "rocmfpx"
MAINLINE_ENGINE = ""

#: The MAINLINE self-contained ROCm image for ``engine == ""`` stock GGUFs (bundles
#: ROCm 10.0 for gfx1151). Its entrypoint is ``/bin/bash``, so the launch overrides
#: it with ``--entrypoint llama-server``. Captured live on the Z2.
#:
#: **Pinned by DIGEST, in the ``name:tag@sha256:`` form docker accepts for
#: ``pull``/``run``/``image inspect`` alike.** A bare moving tag means the box the
#: installer pre-pulls and the box that redeploys six months later can be running
#: different engines under one name, and a serving regression would then be
#: unattributable. The tag is kept beside the digest because it is what a reader
#: recognises and what a re-pin starts from. **Measured 1.39 GB on the appliance,
#: 2026-09-05. To re-pin:** ``docker pull <name:tag>`` on the appliance, then
#: ``docker image inspect --format '{{index .RepoDigests 0}}' <name:tag>`` for the
#: digest and ``docker image ls`` for the size - and update the figure in
#: ``deploy/install-vaelor.sh``'s pre-pull disk preflight, which is sized from it.
GPU_CONTAINER_IMAGE = (
    "docker.io/kyuz0/amd-strix-halo-toolboxes:rocm-10.0"
    "@sha256:257986b5abdabc07cfb776143f3a690a5fb6dfe28e177f3096a37f1e227e1873"
)

#: The FORK image for ``engine == "rocmfpx"`` (the FP4 27B). Public; bundles ROCm
#: 7.2.3 + the ROCmFPX engine + RADV. Its ENTRYPOINT ``/app/run_server.sh``
#: auto-detects the device and applies the cache/MTP profile itself, so the launch
#: does NOT override the entrypoint and passes only the model, host/port and
#: ``--profile``. Verified serving the FP4 27B (22 GB GTT, Vulkan0) live on the Z2.
#:
#: Digest-pinned for the reason above, and more sharply: this one's tag is
#: ``latest``, so without a digest the reference names whatever the publisher
#: pushed most recently. **Measured 3.30 GB on the appliance, 2026-09-05**; re-pin
#: exactly as for :data:`GPU_CONTAINER_IMAGE`.
GPU_FORK_IMAGE = (
    "ghcr.io/julianmb/q38rocm:latest"
    "@sha256:62884d40be1f568142639f6f40734309c275f19acedb4873b2b8c8a420ab4a2c"
)

#: The serve profile the fork's ``run_server.sh`` applies (cache + MTP tuning);
#: passed as ``--profile speed``. It is the fork's job, not the launch's, to turn
#: this into the device/ngl/ctv/spec flags the FP4 model needs.
GPU_FORK_PROFILE = "speed"

#: The fixed name of the single GPU AI-Chat serving container. One server serves
#: the tier at a time and the bridge serializes ``gpu_start`` under its lock
#: (VD-001), so a fixed name lets ``stop``/``status`` address it without tracking
#: a container id across the bridge boundary.
GPU_CONTAINER_NAME = "vaelor-gpu-rocmfpx"

#: The port the MAINLINE image's ``llama-server`` binds INSIDE the container. Fixed
#: at 8080 (that image's convention); the host-side port is chosen by the supervisor
#: and published onto this one, so two deploys never collide on a host port.
GPU_CONTAINER_PORT = 8080

#: The port the FORK image's ``run_server.sh`` binds INSIDE the container (8000, the
#: fork's convention). The published host port is the same either way; only the
#: container-internal target differs, which the ``-p`` mapping absorbs.
GPU_FORK_PORT = 8000

#: The host ``llama-server`` binds INSIDE the container: all interfaces, always.
#: Docker's port publish forwards to the container's bridge IP, NOT its loopback,
#: so a container-internal ``127.0.0.1`` bind would be unreachable through ``-p``.
#: Loopback safety comes from the ``-p`` PUBLISH address being ``127.0.0.1``
#: (:func:`gpu_container_command`), never from this internal bind.
GPU_CONTAINER_BIND = "0.0.0.0"

#: The in-container mount point for the model cache directory under the MAINLINE
#: image, and the host cache root mounted there. ``-m`` names a path under it. The
#: fork image mounts the same host dir at :data:`GPU_FORK_MODELS_MOUNT` instead.
MODELS_MOUNT = "/models"


def configured_model_cache_dir() -> str:
    """The model cache this host is configured with, as the executor reads it.

    ``data_path("models")`` - ``VAELOR_DATA_ROOT`` + ``models`` - is where the
    workload executor downloads and resolves every GGUF it deploys, so the root
    bridge confines launches to the same directory rather than to a second
    spelling of it. :data:`MODEL_CACHE_DIR` stays the parameter default.
    """
    return data_path("models")

#: The FORK image's model mount point: ``run_server.sh`` expects the model under
#: ``/app/models`` and is given ``/app/models/<file>`` as its positional argument.
GPU_FORK_MODELS_MOUNT = "/app/models"

#: The env every image needs to target the 8060S's ISA, passed with ``-e``. The
#: container carries its own ROCm, so the bare fork's ``LD_LIBRARY_PATH`` juggling
#: is gone.
HSA_OVERRIDE_ENV = "HSA_OVERRIDE_GFX_VERSION=11.5.1"

#: The unified-memory env the FORK image needs so the FP4 model resides in the
#: shared GTT aperture (measured 22 GB GTT-resident on the Z2). Passed with ``-e``
#: for ``engine == "rocmfpx"`` only; the mainline image does not set it.
GGML_UNIFIED_MEMORY_ENV = "GGML_HIP_ENABLE_UNIFIED_MEMORY=1"

#: Turn on llama-server's Prometheus ``/metrics`` endpoint (Phase E′ serving
#: metrics). llama.cpp reads the ``--metrics`` flag from this env var
#: (``LLAMA_ARG_ENDPOINT_METRICS``), so metrics are enabled without depending on
#: the fork ``run_server.sh``'s own argument parsing — set on both engine images.
#: The endpoint binds the same loopback the model does (no LAN exposure), and the
#: controller scrapes it locally into InfluxDB for the Performance tab. Harmless
#: when nothing scrapes it: an idle endpoint costs nothing.
LLAMA_METRICS_ENDPOINT_ENV = "LLAMA_ARG_ENDPOINT_METRICS=1"

#: How long to let ``docker stop`` drain the container before it is killed. Freeing
#: a large model off the GPU is not instant; past this docker sends SIGKILL.
STOP_GRACE_SECONDS = 10

#: Bounds on the docker calls. ``docker run -d`` returns as soon as the container
#: is created (the model load is health-gated by the supervisor, not awaited here);
#: a ``docker pull`` of a multi-GB image is generous.
DOCKER_RUN_TIMEOUT = 120
DOCKER_PULL_TIMEOUT = 3600
DOCKER_QUERY_TIMEOUT = 30


class GpuImageMissingError(FileNotFoundError):
    """The GPU serving image is not present and could not be pulled.

    Raised by :meth:`GpuServerProcess.ensure_image` when ``docker image inspect``
    misses and ``docker pull`` fails, so an offline box with no cached image fails
    with a sentence naming the image rather than a raw ``docker run`` error. A
    subclass of ``FileNotFoundError`` (hence ``OSError``) so the GPU deploy's
    fall-back set catches it exactly as it caught the old engine-missing error.
    """


class GpuGroupResolutionError(ValueError):
    """The render/video groups could not be resolved to numeric GIDs on this host.

    A container image has no host group database, so ``--group-add render`` fails;
    numeric GIDs are required and they differ per host. When a needed group has no
    GID the launch is refused here rather than producing a container that cannot
    open the GPU devices.
    """


#: The largest ``-ngl`` (layers offloaded to the GPU) the launch accepts. The
#: recipes emit 0..99; 999 is llama.cpp's conventional "every layer" spelling.
NGL_MAX = 999


def _validate_ngl(value: Any) -> int:
    """A whole-number GPU layer count in ``0..NGL_MAX``, or refuse it.

    Checked at the root boundary because the value arrives in the caller's
    ``runtime`` over the bridge: a JSON ``Infinity`` made ``int()`` raise an
    ``OverflowError`` the bridge did not answer, and a float or bool is not a
    layer count either.
    """
    if isinstance(value, bool) or not isinstance(value, int) or not (
        0 <= value <= NGL_MAX
    ):
        raise ValueError(
            "The GPU layer count (ngl) must be a whole number from 0 to {}; {!r} "
            "is refused.".format(NGL_MAX, value)
        )
    return value


def _runtime_mapping(runtime: Any) -> Dict[str, Any]:
    """The caller's ``runtime`` as a fresh dict; anything but a mapping is refused."""
    if runtime is None:
        return {}
    if not isinstance(runtime, Mapping):
        raise ValueError("The GPU launch runtime must be a JSON object.")
    return dict(runtime)


def _flag(runtime: Mapping[str, Any], key: str) -> Any:
    """A runtime value, falling back to the measured default for its key.

    So a caller passing ``{}`` still gets the proven recipe and a caller overriding
    one field changes only that field. ``None`` is treated as absent so an explicit
    null cannot blank out a measured flag.
    """
    value = runtime.get(key)
    return GPU_MEASURED_DEFAULTS.get(key) if value is None else value


def _serve_tuning_args(model_path: str, runtime: Mapping[str, Any]) -> List[str]:
    """The model + measured tuning flags of the ``llama-server`` argv.

    Everything the proven recipe carries between the executable and the
    ``--host``/``--port`` tail: the model, device, layer count, flash-attention,
    context, KV quantisations and speculative-decode flags. Read from ``runtime``
    with :func:`_flag` backfilling the measured defaults. Used by
    :func:`gpu_serve_command` (the pure host-binary form).
    """
    # ``-fa`` takes ``on``/``off``, not a bool; the measured recipe is ``on``.
    flash = "on" if bool(_flag(runtime, "flash_attn")) else "off"
    args = [
        "-m", model_path,
        "-dev", str(_flag(runtime, "device")),
        "-ngl", str(int(_flag(runtime, "ngl"))),
        "-fa", flash,
        "-c", str(int(_flag(runtime, "context"))),
        "-ctk", str(_flag(runtime, "kv_k")),
        "-ctv", str(_flag(runtime, "kv_v")),
    ]
    # Speculative decode is the MEASURED FP4 recipe's, not a universal default:
    # emitted only when ``spec_type`` resolves to a NON-EMPTY value. A bare ``{}``
    # backfills the measured ``draft-mtp``; a GENERIC runtime sets ``spec_type`` to
    # ``""`` EXPLICITLY, which :func:`_flag` returns verbatim (present, not absent)
    # and this gate reads as "off".
    if str(_flag(runtime, "spec_type") or ""):
        args += [
            "--spec-type", str(_flag(runtime, "spec_type")),
            "--spec-draft-n-max", str(int(_flag(runtime, "spec_draft_n_max"))),
            "--spec-draft-p-min", str(float(_flag(runtime, "spec_draft_p_min"))),
        ]
    return args


def gpu_serve_command(
    model_path: str,
    port: int,
    runtime: Optional[Mapping[str, Any]] = None,
    *,
    binary: str = GPU_ENGINE_BINARY,
) -> List[str]:
    """The bare ``llama-server`` argv for the proven recipe (host-binary form).

    Superseded for serving by :func:`gpu_container_command`, but retained: it is
    the pure, GPU-free expression of the recipe and what
    :mod:`vaelor.gpu_model_choice`'s generic-runtime tests exercise. A list, never a
    string: no element can be split or chained by a shell. The tuning is read from
    ``runtime`` (:func:`_serve_tuning_args`); the model path is validated and the
    port range-checked. **The bind is ALWAYS loopback and no api key is ever
    emitted** - the model is never LAN-exposed by the engine (see the module
    docstring); LAN exposure is the separate auth proxy.
    """
    resolved = dict(runtime or {})
    safe_model = _validate_model_path(model_path)
    safe_port = _validate_port(port)
    command = [binary]
    command += _serve_tuning_args(safe_model, resolved)
    command += ["--host", GPU_HOST, "--port", str(safe_port)]
    return command


def _container_model_mount(
    model_path: str,
    model_cache_dir: str = MODEL_CACHE_DIR,
    models_mount: str = MODELS_MOUNT,
    filesystem: ModelFilesystem = HOST_FILESYSTEM,
) -> "tuple[str, str]":
    """The ``(host_mount_source, in_container_model_path)`` for a GGUF.

    The RESOLVED model cache root is always the mount source, mounted at
    ``models_mount``, and the model keeps its path relative to it (so
    ``.../repo/file.gguf`` becomes ``<mount>/repo/file.gguf``). There is no
    other source: a model outside the cache is refused by
    :func:`_confined_model_path`, never served by mounting its own directory.
    The mount point differs by engine - ``/models`` for the mainline image,
    ``/app/models`` for the fork - so it is a parameter, not the constant.
    """
    host, cache = _confined_model_path(model_path, model_cache_dir, filesystem)
    mount = models_mount.rstrip("/")
    return cache, "{}/{}".format(mount, host[len(cache) + 1:])


class _EngineContainer(NamedTuple):
    """The per-engine facts that shape a launch: which image, and how to drive it.

    One record per ``engine`` value, returned by :func:`_engine_container`, so
    :func:`gpu_container_command` has ONE code path with the image, mount point,
    internal port, entrypoint and extra env all read from here rather than branched
    inline. ``entrypoint`` ``None`` means "use the image's own" (the fork's
    ``run_server.sh``).
    """

    image: str
    models_mount: str
    internal_port: int
    entrypoint: Optional[str]
    extra_envs: Tuple[str, ...]
    ipc_host: bool


def _engine_container(engine: Any) -> _EngineContainer:
    """The container recipe for an engine: the FORK for ``rocmfpx``, MAINLINE for ``""``.

    ``rocmfpx`` (the FP4 27B) needs the fork image, its ``run_server.sh``
    entrypoint (so no ``--entrypoint`` override, and no ``-dev``/``-ngl``/``-ctv``/
    ``--spec-*`` - it applies the profile itself), ``--ipc host`` and the
    unified-memory env. ``""`` (a stock GGUF, and an omitted engine) is the
    MAINLINE image with ``--entrypoint llama-server``. **Anything else is
    refused** (ACC-060): the value arrives in a caller's ``runtime`` over the
    bridge, and an engine name the root side does not know used to fall through
    to the Strix Halo image - a launch nobody asked for, on a claim nobody
    checked.
    """
    if engine is None:
        engine = MAINLINE_ENGINE
    if not isinstance(engine, str) or engine not in (MAINLINE_ENGINE, ROCMFPX_ENGINE):
        raise ValueError(
            "The GPU server does not know the engine {!r}, so nothing was "
            "launched; it serves only the standard llama.cpp engine and the "
            "ROCmFP4 fork.".format(engine)
        )
    if engine == ROCMFPX_ENGINE:
        return _EngineContainer(
            image=GPU_FORK_IMAGE, models_mount=GPU_FORK_MODELS_MOUNT,
            internal_port=GPU_FORK_PORT, entrypoint=None,
            extra_envs=(GGML_UNIFIED_MEMORY_ENV, LLAMA_METRICS_ENDPOINT_ENV),
            ipc_host=True,
        )
    return _EngineContainer(
        image=GPU_CONTAINER_IMAGE, models_mount=MODELS_MOUNT,
        internal_port=GPU_CONTAINER_PORT, entrypoint="llama-server",
        extra_envs=(LLAMA_METRICS_ENDPOINT_ENV,), ipc_host=False,
    )


#: The largest context window (``-c``) the MAINLINE launch accepts - 2**20
#: tokens, far above any GGUF this tier serves. A bound on the integer that
#: crosses the bridge, not a sizing: the fit decides the window.
GPU_CONTEXT_MAX = 1 << 20

#: The slot count the MAINLINE launch states (``--parallel``). The fit sizes the
#: KV cache for ONE slot holding the whole window (``gpu_offload_plan``,
#: ``parallel=1``), and llama.cpp left to itself starts several - the #109
#: multiplication. A literal, never a caller value. NOT yet measured live:
#: whether one slot costs concurrent AI Chat and LLM Server requests (they now
#: queue on the one slot) is to be measured on the Z2 before it is tuned.
GPU_MAINLINE_PARALLEL = 1


def _mainline_context(runtime: Mapping[str, Any]) -> int:
    """The context window the MAINLINE launch passes as ``-c`` (ACC-059).

    The memory plan reserves room for the runtime's ``context`` (the generic GPU
    window, 8,192), so the engine must run with exactly that window, not the
    model's own trained one - left unset, llama.cpp builds the trained context
    (32K-128K on the models this tier serves), and the KV cache the plan never
    counted is what runs out of memory. Read straight off ``runtime`` rather than
    through :func:`_flag`, because the measured default belongs to the FP4 FORK
    (131,072, which the fork's own recipe applies); a runtime naming no window
    gets the generic GPU window the fit sizes by default. Validated at the root
    boundary like ``ngl``: a whole number from 1 to :data:`GPU_CONTEXT_MAX`.
    """
    value = runtime.get("context")
    if value is None:
        return GPU_RECOMMENDED_CONTEXT_TOKENS
    if isinstance(value, bool) or not isinstance(value, int) or not (
        1 <= value <= GPU_CONTEXT_MAX
    ):
        raise ValueError(
            "The GPU context window (context) must be a whole number from 1 to "
            "{}; {!r} is refused.".format(GPU_CONTEXT_MAX, value)
        )
    return value


def _mainline_serve_args(
    container_model: str, ngl: int, *,
    context: int, bind: str, internal_port: int,
) -> List[str]:
    """The ``llama-server`` args for the MAINLINE image (``engine == ""``).

    Auto device selection: the model, the context window the memory plan was
    sized for (``-c``, ACC-059) with the one slot it was sized for
    (``--parallel``), the layer count, then the internal bind/port. The
    fork-only ``-dev Vulkan0``, ``-ctv turbo4`` and ``--spec-*`` are NOT
    emitted - live on the Z2 mainline's device is ``ROCm0`` not ``Vulkan0``,
    ``turbo4`` is unsupported, and the FP4 tensors do not load here at all; a
    stock GGUF needs none of them. No key: the model is loopback-only.
    """
    args = [
        "-m", container_model, "-c", str(context),
        "--parallel", str(GPU_MAINLINE_PARALLEL), "-ngl", str(ngl),
    ]
    args += ["--host", bind, "--port", str(internal_port)]
    return args


def _fork_serve_args(
    container_model: str, *, bind: str, internal_port: int,
) -> List[str]:
    """The ``run_server.sh`` args for the FORK image (``engine == "rocmfpx"``).

    The fork's entrypoint OWNS the recipe: it auto-detects the device and applies
    the cache/MTP profile, so the launch passes ONLY the model (positional, under
    ``/app/models``), the internal bind/port and ``--profile speed``. No
    ``-dev``/``-ngl``/``-ctv``/``--spec-*``, and no key (the model is loopback-only).
    """
    return [
        container_model, "--host", bind, "--port", str(internal_port),
        "--profile", GPU_FORK_PROFILE,
    ]


def gpu_container_command(
    model_path: str,
    port: int,
    runtime: Optional[Mapping[str, Any]] = None,
    *,
    engine: str = "",
    group_ids: Sequence[int],
    docker: str = "docker",
    image: Optional[str] = None,
    container_name: str = GPU_CONTAINER_NAME,
    model_cache_dir: str = MODEL_CACHE_DIR,
    filesystem: ModelFilesystem = HOST_FILESYSTEM,
) -> List[str]:
    """The ``docker run -d`` argv that serves the model on LOOPBACK, per the engine.

    The image + interface are chosen by ``engine`` (the explicit argument, else
    ``runtime["engine"]``; :func:`_engine_container`). Two shapes, captured live on
    the Z2 - MAINLINE for a stock GGUF (``engine == ""``)::

        docker run -d --name <name> --device /dev/kfd --device /dev/dri
            --group-add <r> --group-add <v> --security-opt seccomp=unconfined
            -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -v <model-cache>:/models:ro
            -p 127.0.0.1:<port>:8080 --entrypoint llama-server <mainline-image>
            -m /models/<file>.gguf -c <context> --parallel 1 -ngl <N>
            --host 0.0.0.0 --port 8080

    and the FORK for the FP4 27B (``engine == "rocmfpx"``), whose entrypoint owns
    the recipe::

        docker run -d --name <name> --device /dev/kfd --device /dev/dri
            --group-add <r> --group-add <v> --security-opt seccomp=unconfined
            --ipc host -e HSA_OVERRIDE_GFX_VERSION=11.5.1
            -e GGML_HIP_ENABLE_UNIFIED_MEMORY=1 -v <model-cache>:/app/models:ro
            -p 127.0.0.1:<port>:8000 <fork-image>
            /app/models/<file>.gguf --host 0.0.0.0 --port 8000 --profile speed

    **The model is ALWAYS published on ``127.0.0.1`` and carries NO api key**, for
    BOTH engines: it is never LAN-exposed by the engine. The container binds
    ``0.0.0.0`` internally (docker forwards the publish to the container's bridge
    IP, never its loopback); loopback safety comes from publishing onto
    ``127.0.0.1``. LAN exposure is the separate auth proxy
    (:mod:`vaelor.llm_server_proxy`), which is the only trustworthy, engine-agnostic
    gate. GIDs must be NUMERIC (a container has no host group db); the caller
    resolves and passes them.

    **The model mount is the resolved model cache, READ-ONLY, for both
    engines.** This argv is built by root for a container with the GPU devices
    and no seccomp profile, on a path any group-``vaelor`` service may send, so
    the path is confined to the cache (:func:`_confined_model_path`) and the
    container can read the weights but never write the host.
    """
    resolved = _runtime_mapping(runtime)
    profile = _engine_container(engine or resolved.get("engine"))
    used_image = image or profile.image
    mount_source, container_model = _container_model_mount(
        model_path, model_cache_dir, profile.models_mount, filesystem
    )
    safe_port = _validate_port(port)
    # Validated for both engines (the fork ignores it), so a malformed runtime
    # is refused whichever image it names.
    ngl = _validate_ngl(_flag(resolved, "ngl"))
    # The fork applies its own measured window (run_server.sh); only the
    # mainline image is told one, and it is validated only where it is used.
    context = _mainline_context(resolved) if profile.entrypoint else 0
    command = [
        docker, "run", "-d", "--name", container_name,
        "--device", "/dev/kfd", "--device", "/dev/dri",
    ]
    for gid in group_ids:
        command += ["--group-add", str(int(gid))]
    command += ["--security-opt", "seccomp=unconfined"]
    if profile.ipc_host:
        command += ["--ipc", "host"]
    command += ["-e", HSA_OVERRIDE_ENV]
    for env in profile.extra_envs:
        command += ["-e", env]
    # Read-only, always: a serving engine only ever reads its weights.
    command += ["-v", "{}:{}:ro".format(mount_source, profile.models_mount)]
    # Loopback publish, always: the model is never bound to the network here.
    command += ["-p", "{}:{}:{}".format(GPU_HOST, safe_port, profile.internal_port)]
    if profile.entrypoint:
        command += ["--entrypoint", profile.entrypoint]
    command += [used_image]
    # The serve-arg shape follows the entrypoint: the fork drives its own
    # ``run_server.sh`` (``entrypoint is None``) with a positional model + profile;
    # the mainline drives ``llama-server`` with ``-m``/``-ngl``.
    if profile.entrypoint is None:
        command += _fork_serve_args(
            container_model, bind=GPU_CONTAINER_BIND,
            internal_port=profile.internal_port,
        )
    else:
        command += _mainline_serve_args(
            container_model, ngl, context=context, bind=GPU_CONTAINER_BIND,
            internal_port=profile.internal_port,
        )
    return command


def _numeric_group_ids(grants: Mapping[str, Any]) -> List[int]:
    """The numeric render/video GIDs a container needs, or raise.

    ``device_grants(["gpu"])`` resolves the group names for ``/dev/kfd`` +
    ``/dev/dri`` to numeric GIDs on THIS host. A container image has no host group
    database, so a name would fail; an unresolved group means the container could
    not open the device, so the launch is refused (:class:`GpuGroupResolutionError`)
    rather than started blind.
    """
    group_ids = grants.get("group_ids") or {}
    unresolved = [name for name, gid in group_ids.items() if gid is None]
    if unresolved or not group_ids:
        raise GpuGroupResolutionError(
            "Could not resolve the {} group(s) to numeric GIDs on this host, so "
            "the GPU serving container cannot be granted the accelerator "
            "devices.".format(", ".join(unresolved) or "render/video")
        )
    return [int(gid) for gid in group_ids.values()]


def _default_run(command: Sequence[str], *, timeout: Optional[int] = None):
    """Run a docker command, capturing output, never raising on non-zero.

    stdout/stderr are captured (not DEVNULL): a ``docker run`` that fails explains
    itself on stderr, which the caller surfaces in the raised error rather than
    discarding (LESSONS pattern 8).
    """
    return subprocess.run(
        list(command),
        capture_output=True, text=True, check=False, timeout=timeout,
    )


class GpuServerProcess:
    """A supervised GPU ``llama-server`` CONTAINER, launched and stopped.

    The C0 successor to the ``subprocess.Popen`` supervisor: it manages a
    ``docker run -d`` container rather than a host process, but keeps the same
    ``start``/``stop``/``status``/``alive`` surface so the hardware bridge and the
    :class:`vaelor.gpu_rocm_supervisor.GpuRocmSupervisor` above it are unchanged.
    The docker calls go through an injected ``run`` seam so the launch argv - and
    the whole lifecycle - are verifiable without a real GPU or docker.
    """

    def __init__(
        self,
        *,
        run: Optional[Callable[..., Any]] = None,
        docker: Optional[str] = None,
        image: Optional[str] = None,
        container_name: str = GPU_CONTAINER_NAME,
        model_cache_dir: str = MODEL_CACHE_DIR,
        group_resolver: Optional[Callable[[], Mapping[str, Any]]] = None,
        filesystem: ModelFilesystem = HOST_FILESYSTEM,
    ):
        self._run = run or _default_run
        self._docker = docker
        # ``None`` (the default) means "let the engine choose the image at start":
        # the mainline image for a stock GGUF, the fork image for the FP4 27B. A
        # test may pin a fixed image to force one. :meth:`start` resolves it.
        self._image = image
        self._name = container_name
        self._model_cache_dir = model_cache_dir
        # How the model path is resolved and checked at launch. Production is the
        # real filesystem; a test injects a fake one (see :class:`ModelFilesystem`).
        self._filesystem = filesystem
        # Resolves render/video to numeric GIDs on this host; injectable for tests.
        self._group_resolver = group_resolver or (lambda: device_grants(["gpu"]))
        self.model_path = ""
        self.port = 0

    def _docker_binary(self) -> str:
        docker = self._docker or shutil.which("docker") or "/usr/bin/docker"
        return docker

    def ensure_image(
        self, docker: Optional[str] = None, *, image: Optional[str] = None
    ) -> None:
        """Make sure the ENGINE'S serving image is present, pulling it once if not.

        The image is the one the launch will use - the fork image for the FP4 27B,
        the mainline image for a stock GGUF - passed in by :meth:`start` (or pinned
        on the instance). ``docker image inspect`` is the cheap presence probe, and
        only a miss triggers a ``docker pull``. On the Z2 both images are already
        pulled, so this is a fast no-op; on a fresh box it fetches once, and a
        failed fetch raises :class:`GpuImageMissingError` naming the image rather
        than letting ``docker run`` fail obscurely.
        """
        docker = docker or self._docker_binary()
        image = image or self._image or GPU_CONTAINER_IMAGE
        inspected = self._run(
            [docker, "image", "inspect", image], timeout=DOCKER_QUERY_TIMEOUT
        )
        if getattr(inspected, "returncode", 1) == 0:
            return
        pulled = self._run(
            [docker, "pull", image], timeout=DOCKER_PULL_TIMEOUT
        )
        if getattr(pulled, "returncode", 1) != 0:
            raise GpuImageMissingError(
                "The GPU serving image {} is not present and could not be "
                "pulled: {}".format(
                    image, (getattr(pulled, "stderr", "") or "").strip()
                )
            )

    def start(
        self, model_path: str, *, port: int,
        runtime: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate, then launch the container. A running one is replaced first.

        Validation happens inside :func:`gpu_container_command`, before the previous
        container is touched, so a bad model path or port - including one outside
        the model cache - cannot take down a healthy server. The engine (read from
        ``runtime["engine"]``) chooses the image + interface. The model is always
        published on loopback and carries no key.

        **The confinement is checked twice**: once to build the argv, and again
        immediately before ``docker run`` - after the image pull and the stop of
        the previous server - so the window between the check and dockerd
        resolving ``-v`` is one call, not an image download. A model that stops
        qualifying in between is refused with the old server already stopped.
        """
        resolved = _runtime_mapping(runtime)
        engine = resolved.get("engine")
        profile = _engine_container(engine)
        image = self._image or profile.image
        docker = self._docker_binary()
        group_ids = _numeric_group_ids(self._group_resolver())
        command = gpu_container_command(
            model_path, port, resolved, engine=engine or MAINLINE_ENGINE,
            group_ids=group_ids, docker=docker, image=image,
            container_name=self._name, model_cache_dir=self._model_cache_dir,
            filesystem=self._filesystem,
        )
        self.ensure_image(docker, image=image)
        if self.alive():
            self.stop()
        else:
            # Clear any stopped-but-not-removed container of our name so the fixed
            # ``--name`` is free for ``docker run`` (a prior crash can leave one).
            self._run([docker, "rm", "-f", self._name], timeout=DOCKER_QUERY_TIMEOUT)
        try:
            model, _cache = _confined_model_path(
                model_path, self._model_cache_dir, self._filesystem
            )
        except GpuModelPathError as error:
            # The previous container is already gone by this point, so say so:
            # a refusal here leaves nothing serving on the GPU.
            raise GpuModelPathError(
                "{} The previous GPU server had already been stopped, so no "
                "model is serving on the GPU until a valid one is launched.".format(error)
            ) from error
        result = self._run(command, timeout=DOCKER_RUN_TIMEOUT)
        if getattr(result, "returncode", 1) != 0:
            raise RuntimeError(
                "The GPU serving container could not be started: {}".format(
                    (getattr(result, "stderr", "") or "").strip()
                )
            )
        self.model_path = model
        self.port = _validate_port(port)
        return self.status()

    def alive(self) -> bool:
        """Whether our container exists and is running, per ``docker inspect``."""
        docker = self._docker_binary()
        try:
            result = self._run(
                [docker, "inspect", "-f", "{{.State.Running}}", self._name],
                timeout=DOCKER_QUERY_TIMEOUT,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return (
            getattr(result, "returncode", 1) == 0
            and (getattr(result, "stdout", "") or "").strip() == "true"
        )

    def stop(self, *, grace_seconds: int = STOP_GRACE_SECONDS) -> Dict[str, Any]:
        """``docker stop`` (with a grace period), then ``docker rm``.

        A container that is already gone is fine; both calls are best-effort so a
        stop never raises. The fixed name is freed either way so the next launch
        can reuse it.
        """
        docker = self._docker_binary()
        was_running = self.alive()
        try:
            self._run(
                [docker, "stop", "--time", str(int(grace_seconds)), self._name],
                timeout=DOCKER_RUN_TIMEOUT,
            )
            self._run([docker, "rm", "-f", self._name], timeout=DOCKER_QUERY_TIMEOUT)
        except (OSError, subprocess.SubprocessError):
            pass
        return {"stopped": True, "was_running": was_running}

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.alive(),
            "model_path": self.model_path,
            "port": self.port,
            "container": self._name,
        }
