"""The vLLM container images the GPU cluster may serve from, and what each needs.

vLLM on gfx1151 exists only as a container: a host build cannot be produced.
Two images are pinned here, each by DIGEST (ACC-062), so every node runs one
verified build and never whatever a tag names today:

* **vLLM 0.22.1** (``ryai-vllm``), the image every deployment served from
  between 2026-09-05 and the 0.27 benchmark. Its entrypoint is a thin
  ``vllm "$@"`` that first sources the image's ROCm profile, and it needs
  ``HSA_OVERRIDE_GFX_VERSION`` to target the 8060S. It stays pinned and
  selectable: the rollback, and the image every deployment recorded before an
  image was recorded keeps (:data:`RECORDED_IMAGE_KEY`).
* **vLLM 0.27** on ROCm 10 (AMD's official ``rocm/vllm`` build for gfx1151),
  benchmarked on the pair on 2026-09-29/30. It has NO entrypoint (its command
  is ``/bin/bash``), so the server is started with ``--entrypoint vllm``;
  gfx1151 is native in every wheel, so no ISA override; AMD's run line adds
  ``FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE``, which the image does not bake in.
  Every container of it also carries ``ROCPROFILER_REGISTER_ENABLED=0``
  (ACC-186, VD-165): its torch 2.12 exports ``rocprofiler_configure`` (Kineto's
  rocprofiler-sdk backend), so without it every engine process registers
  Kineto as a profiling tool at GPU start, the tool's signal pool exhausts the
  kernel's events, and ROCr's async-events thread polls one CPU core at 100%
  for as long as the model is served, idle or not. vLLM's own profiler flags
  are not involved; the 0.22.1 image's torch links roctracer and idles cool.
  It carries the ``RDNAHybridW4A16`` kernel (upstream PR #40977), the fast
  path for GPTQ and compressed-tensors 4-bit weights on this GPU, and the
  hybrid-model cache flags (``--mamba-cache-mode``, ``--prefix-match-unit``).
  It is the DEFAULT (owner decision, 2026-09-30).

A deployment chooses one through its serving options (`vllm_serve_options`,
``vllm_image``), and the record keeps the choice, so a Load re-serves on the
image it was deployed on. **Two names, because "no image named" means two
things.** A NEW deployment that names none gets :data:`DEFAULT_IMAGE_KEY`. A
deployment RECORDED with none - every one made before the choice existed -
ran 0.22.1, and keeps it (:data:`RECORDED_IMAGE_KEY`): moving the default
never moves a deployment onto an image its owner did not choose.

**Changing the default, adding an image, or re-pinning one is a reviewed code
change**: the root bridge admits exactly the images listed here
(`bridge_argv_policy.BRIDGE_IMAGE_REFS`, spelled there literally and tied to
this table by a test), so no value a client sends can name any other.

**Each image compiles into its own cache folder** (``compile_scope``). The
0.27 image's Python 3.14 / torch 2.12 / Triton 3.8 caches must never mix with
0.22.1's, and a rollback must find the old image's cache as it left it, so the
0.22.1 image keeps the cache root it has always used and each newer image gets
a sub-folder of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

#: The name the unit value and the deploy payload carry an image choice under.
IMAGE_FIELD = "vllm_image"


@dataclass(frozen=True)
class VllmImage:
    """One pinned vLLM image and what a container of it needs."""

    key: str
    image: str
    vllm_version: str
    #: The few words a progress line or a deployment's row calls it.
    name: str
    #: What the deploy form calls it, in plain words. Short: it is shown in a
    #: quarter-width picker, where a longer label was cut off mid-word.
    label: str
    #: ``None``: the image's own entrypoint runs ``vllm``, and ``serve ...``
    #: follows the image. Otherwise the binary ``--entrypoint`` names.
    server_entrypoint: Optional[str]
    #: Fixed ``KEY=VALUE`` variables every container of this image carries.
    environment: Tuple[str, ...]
    #: The compile-cache sub-folder, ``""`` for the cache root itself.
    compile_scope: str
    #: Whether an MoE model's container may start through Vaelor's entry
    #: program (`vllm_entry_program`). Not on 0.22.1: overriding its
    #: entrypoint would skip the ROCm profile its ``vllm`` wrapper sources.
    moe_entry: bool
    #: Whether vLLM carries a fast 4-bit kernel for gfx1151 (RDNAHybridW4A16).
    fast_w4a16: bool
    #: Whether this vLLM takes the cache flags a hybrid (Gated DeltaNet) model
    #: is started with on a GPU without FP8 hardware (`vllm_serve_options`):
    #: ``--kv-cache-dtype bfloat16 --attention-backend TRITON_ATTN`` and the
    #: aligned prefix cache. Read at tag v0.27.0 and run on this image.
    hybrid_cache: bool
    #: About how much a machine downloads when it does not have the image,
    #: in GB, for the progress line that says a download is under way.
    download_gb: int


VLLM_0221 = VllmImage(
    key="vllm-0.22.1",
    image=(
        "oci-registry.ryai.dev/ryai-vllm:latest"
        "@sha256:ff4c0d784b14bcde538f015b5551d251339fc83a223e9f2967f3d1a3b05c3973"
    ),
    vllm_version="0.22.1",
    name="vLLM 0.22",
    label="vLLM 0.22 (earlier version)",
    server_entrypoint=None,
    environment=("HSA_OVERRIDE_GFX_VERSION=11.5.1",),
    compile_scope="",
    moe_entry=False,
    fast_w4a16=False,
    hybrid_cache=False,
    download_gb=26,
)

VLLM_027_ROCM10 = VllmImage(
    key="vllm-0.27-rocm10",
    image=(
        "rocm/vllm"
        "@sha256:b8a082f346d069376d35784250e38b23a043efe979408ae3a33d7c6b62ee3276"
    ),
    vllm_version="0.27.1.dev5+gf46a9dfe2",
    name="vLLM 0.27",
    label="vLLM 0.27 (recommended)",
    server_entrypoint="vllm",
    environment=("FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE", "ROCPROFILER_REGISTER_ENABLED=0"),
    compile_scope="v0.27-rocm10",
    moe_entry=True,
    fast_w4a16=True,
    hybrid_cache=True,
    download_gb=27,
)

#: Every image the cluster may serve from, keyed by the name a deployment
#: records. Order is the order they were pinned.
IMAGES: Dict[str, VllmImage] = {
    VLLM_0221.key: VLLM_0221,
    VLLM_027_ROCM10.key: VLLM_027_ROCM10,
}

#: The image a NEW deployment that names none serves from (owner decision,
#: 2026-09-30: vLLM 0.27, after the benchmark). Moving it is the reviewed
#: switch, and it moves no deployment that already exists.
DEFAULT_IMAGE_KEY = VLLM_027_ROCM10.key

#: The image a deployment RECORDED with none was deployed on: 0.22.1, the only
#: image there was before a deployment could name one. A record, or a unit
#: rendered with no serving options at all, keeps it. This is a fact about the
#: past, so it does not move when the default does.
RECORDED_IMAGE_KEY = VLLM_0221.key

#: The leading digits of the kernel's ``gfx_target_version`` (the kfd topology
#: spells gfx1151 as ``110501``: major, minor, stepping, two digits each) for
#: the RDNA 3 family - RDNA 3 and 3.5, ``gfx11xx``. What this table knows
#: about that family is that it has no FP8 hardware: ROCm's FP8 kernels need
#: CDNA 3 or gfx12, and on gfx11 an FP8 KV cache is emulated and measured
#: slower than bf16 (the 0.27 benchmark, 2026-09-30). Only gfx1151 of the
#: family was measured.
_RDNA3_MAJOR = 11


def gfx_family_major(gfx_target_version: Any) -> int:
    """The family number of a kfd target version, or 0 when it is not one.

    Read off the number the GPU probe recorded for the machine (the kfd
    topology, `platforms.kfd_topology`), never off a product name. Only the
    kernel's own spelling counts: ASCII digits, five or more. A machine that
    reported none, a hand-typed ``gfx1151`` or ``11.5.1``, digits from another
    script - each is 0: a GPU nobody read is not assumed to be of any family.
    """
    text = str(gfx_target_version or "").strip()
    if not (text.isascii() and text.isdigit()) or len(text) < 5:
        return 0
    return int(text) // 10000


def is_rdna3_family(gfx_target_version: Any) -> bool:
    """Whether this kfd target version is an RDNA 3 family GPU (gfx11xx)."""
    return gfx_family_major(gfx_target_version) == _RDNA3_MAJOR


def all_rdna3_family(gfx_target_versions: Iterable[Any]) -> bool:
    """Whether EVERY machine of a deployment has an RDNA 3 family GPU.

    One serving-options value is rendered for all of a deployment's units, so
    the answer is about the whole set, and an empty set is ``False``.
    """
    targets = list(gfx_target_versions or ())
    return bool(targets) and all(is_rdna3_family(target) for target in targets)


def download_line(profile: VllmImage, machine: Any) -> str:
    """What a job's progress says while a machine downloads ``profile``.

    Said by name and size, because the download is tens of gigabytes and
    otherwise looks like a job that has stopped.
    """
    return "Downloading the {} runtime to {}, about {} GB".format(
        profile.name, machine, profile.download_gb,
    )


def image_profile(key: Any = None) -> VllmImage:
    """The image a key names; the default for ``None``; refused for anything else."""
    if key is None:
        return IMAGES[DEFAULT_IMAGE_KEY]
    if not isinstance(key, str) or key not in IMAGES:
        raise ValueError(
            "That is not a vLLM image this Vaelor version serves from. Choose "
            "one of: {}.".format(", ".join(IMAGES))
        )
    return IMAGES[key]


def compile_environment(profile: VllmImage, mount: str) -> Tuple[str, ...]:
    """The three cache variables vLLM, Triton and inductor read, for ``profile``."""
    base = mount + ("/" + profile.compile_scope if profile.compile_scope else "")
    return (
        f"VLLM_CACHE_ROOT={base}/vllm",
        f"TRITON_CACHE_DIR={base}/triton",
        f"TORCHINDUCTOR_CACHE_DIR={base}/inductor",
    )
