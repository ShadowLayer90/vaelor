"""The shape every GPU-pool ``docker run`` shares: devices, flags, caches, mounts, env.

Moved out of `gpu_pool_runtime` (at the 1,000-line ceiling) unchanged but for
one parameter, the image (`vllm_images`): a container's fixed variables and its
compile-cache folder are the image's, so one builder takes the image a unit is
rendered for. `gpu_pool_runtime` re-exports every name here under the name its
readers - the root bridge's shape check among them - have always used.
"""

from __future__ import annotations

from typing import Iterable, List

from . import gpu_unit_params as unit_params
from .gpu_pool_units import DOCKER
from .serving_profiler import SERVING_PROFILES_DIR, SERVING_PROFILES_MOUNT
from .vllm_images import (
    DEFAULT_IMAGE_KEY, IMAGES, RECORDED_IMAGE_KEY, VLLM_0221, VllmImage,
    compile_environment,
)

#: The default image (`vllm_images`): what a new deployment serves from unless
#: its owner picks another. A model pull runs in its deployment's own image
#: (`gpu_pool_runtime.render_model_pull`), so no machine fetches a second
#: image to fetch weights. Then every pinned image, default first: what an
#: uninstall that purges data removes.
VLLM_IMAGE = IMAGES[DEFAULT_IMAGE_KEY].image
VLLM_IMAGES = (VLLM_IMAGE,) + tuple(
    image.image for image in IMAGES.values() if image.image != VLLM_IMAGE
)

#: The image a serving unit that names NONE is rendered for: the one every
#: deployment ran before a deployment could name an image (`vllm_images`).
#: Not the default - a record from then keeps the image it was deployed on.
RECORDED_VLLM_IMAGE = IMAGES[RECORDED_IMAGE_KEY].image

#: The accelerator character devices the container must be granted. The DRI card
#: index differs per box (``card0`` on one, ``card1`` on another), so the whole
#: ``/dev/dri`` is passed, not a card node. The groups that OWN them, and the
#: per-node numbers they resolve to, belong to `gpu_node_facts`.
GPU_DEVICES = ("/dev/kfd", "/dev/dri")

#: The env that targets the 8060S's ISA in the 0.22.1 image (which carries its
#: own ROCm); the 0.27 image targets gfx1151 natively and does without it.
HSA_OVERRIDE_ENV = VLLM_0221.environment[0]

#: Docker run flags every GPU container needs: ROCm's HSA runtime trips the
#: default seccomp profile, and the shm/IPC settings let workers exchange tensors.
CONTAINER_RUNTIME_FLAGS = [
    "--security-opt", "seccomp=unconfined", "--ipc", "host", "--shm-size", "8g",
]

#: The node-local Hugging Face cache the model library pulls weights into, and
#: the path it is mounted at inside every container. ``HF_HOME`` points at the
#: mount, so a pulled model lands where ``vllm serve`` looks for it. A fixed,
#: Vaelor-owned root, so the cache is enumerable for the disk-accounting
#: inventory and removable by repo.
MODEL_CACHE_ROOT = "/var/lib/vaelor/models"
MODELS_MOUNT = "/models"

#: The per-node COMPILE cache every vLLM container shares (VD-127, the
#: bigger-than-one-box proof). A first start of an FP8 model on gfx1151 spends
#: over ten minutes in inductor/Triton compiling block-scaled kernels, and with
#: only the weights cache mounted every ``docker run --rm`` threw that work
#: away: the second attempt compiled from scratch exactly as the first did.
#: One directory under the model store - so it sits on the same disk the
#: weights do, is created by `_ensure_cache_dirs` with the same ownership, and
#: crosses the root bridge as the same ``install -d`` shape - mounted at one
#: fixed container path, and the three env names vLLM, Triton and inductor read
#: for their caches point at subdirectories of it: of the image's own folder
#: inside it (`vllm_images.compile_environment`), so two images never share one.
COMPILE_CACHE_ROOT = MODEL_CACHE_ROOT + "/compile"
COMPILE_MOUNT = "/compile"
#: The three cache variables of a unit that names no image: the cache root
#: itself, which the image recorded for such a unit has always used.
COMPILE_CACHE_ENV = compile_environment(IMAGES[RECORDED_IMAGE_KEY], COMPILE_MOUNT)


def docker_run_prefix(
    *, container_name: str, group_ids: Iterable[int], envs: Iterable[str],
    image: VllmImage,
) -> List[str]:
    """The ``docker run`` flags every GPU container in the pool shares.

    One builder so the merged lead container, the single server and the Ray
    worker cannot drift on the device grants, the GID list, the image's env,
    the model mount or the compile-cache mount (every rank compiles, the
    worker's included, so every container gets it). ``--network host`` is the
    only mode: Ray's GCS/object-manager ports and the RCCL/Gloo sockets must
    reach across nodes on the real NIC, which a bridge would NAT - so nothing
    is published. ``--rm`` for a clean restart.
    """
    prefix = [DOCKER, "run", "--rm", "--name", container_name]
    prefix += ["--network", "host"]
    for device in GPU_DEVICES:
        prefix += ["--device", device]
    for group_id in unit_params.group_ids(group_ids):
        prefix += ["--group-add", str(group_id)]
    prefix += list(CONTAINER_RUNTIME_FLAGS)
    for env in (*image.environment, f"HF_HOME={MODELS_MOUNT}",
                *compile_environment(image, COMPILE_MOUNT), *envs):
        prefix += ["-e", str(env)]
    prefix += ["-v", f"{MODEL_CACHE_ROOT}:{MODELS_MOUNT}"]
    prefix += ["-v", f"{COMPILE_CACHE_ROOT}:{COMPILE_MOUNT}"]
    prefix += ["-v", f"{SERVING_PROFILES_DIR}:{SERVING_PROFILES_MOUNT}"]
    return prefix


def rccl_environment(bind_ip: str, interface: str) -> List[str]:
    """The RCCL/Gloo socket-binding env every distributed container carries.

    ``VLLM_HOST_IP`` is the address vLLM advertises to its Ray peers;
    ``NCCL_SOCKET_IFNAME`` / ``GLOO_SOCKET_IFNAME`` pin both collective
    libraries to the private cluster NIC so a stray public or virtual
    interface is never chosen - the exact variables the RCCL playbook / PoC
    set (VD-054), as ``KEY=VALUE`` strings `docker_run_prefix` makes flags.
    """
    return [
        f"VLLM_HOST_IP={bind_ip}",
        f"NCCL_SOCKET_IFNAME={interface}",
        f"GLOO_SOCKET_IFNAME={interface}",
    ]
