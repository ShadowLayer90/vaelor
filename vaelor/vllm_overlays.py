"""Files Vaelor mounts read-only over a pinned vLLM image's own, and the checks before each start.

**VD-215 (owner decision 2026-10-12, the overlay option).** With a draft model
every decode step verifies several tokens per request, and vLLM 0.27's Triton
``unified_attention`` refuses its split-context (3D) path whenever
``max_seqlen_q > 1``, so a verify step attends over the whole context in one
pass and long-prompt decode falls from about 65 to 19 tok/s (VD-215's
measurements). Upstream pull request vllm-project/vllm#59448 (Apache-2.0, head
``cce2aaa``, unmerged) admits short multi-token queries into the 3D path behind
``VLLM_TRITON_3D_MAX_Q``. The container mounts a patched copy of that one
file over the image's own. No image is built and no registry is involved;
dropping the mount is the rollback.

**Vaelor carries no vLLM code** (owner, 2026-10-12: "for the license, the
installer should fetch it"). The patched file is BUILT on each machine
(`vllm_overlay_build`): the installer fetches the PR's diff at its pinned
commits; the image's own file is read out of the pinned image on that machine;
the PR's hunks for that file and Vaelor's own edits are applied; and the
result is kept only if its sha256 is this table's ``built_sha256``. It lives
root-owned in :data:`OVERLAY_ROOT`. A machine that cannot build it serves
without it and says why (LESSONS 1): the unit is installed without the
overlay's mount, variable and backends, never with half of them.

**One table** (:data:`OVERLAYS`, LESSONS 6): the controller's renderer, the
root bridge's check, the builder and the worker path all read it. Each row is
bound to ONE pinned image by its digest and names the in-image path, the
sha256 of the image's own file there (the file the patch was made against),
the diff's URL and sha256, the built file's sha256, the variable and its
bounds. A different image, or a re-pinned one, has no row, so it gets no
overlay.

**When a deployment gets it** (:func:`decide_overlay`): one copy per machine
(VD-213's path - a draft model is refused on a split), a draft model on, and
the image has a row - and only while :data:`OVERLAY_SERVES_WITH_DRAFTER` says
so. Then the unit carries the read-only mount, ``-e VLLM_TRITON_3D_MAX_Q=<1 +
the schedule's largest lookahead>``, ``--attention-backend TRITON_ATTN`` (the
fix is in that backend's kernel; the 30B otherwise selects ROCM_ATTN) and the
draft model's ``--speculative-config.attention_backend=TRITON_ATTN`` (the two
share KV cache tensors and a mixed pair does not start: measured on the Z2).
Without a draft model nothing changes, byte for byte.

**Records** (VD-213's rule, `vllm_serve_options`): the row's key is stored in
the deployment's ``vllm_options`` as :data:`OVERLAY_FIELD`, written only when
set. A deployment made before this change has no such field and loads exactly
as it was deployed - no silent upgrade onto a different attention backend. A
NEW deploy (including one served again after a Remove) and a refit that
decides its options afresh (`gpu_pool_refit`) get it when they qualify. A
root bridge older than this code refuses the unknown field by name, and the
deploy says "restart the hardware bridge" (`vllm_serve_options.BRIDGE_TOO_OLD`,
LESSONS 1). A stored key that is no longer in the table is refused, never
served with some other file.

**Before every start that carries the mount** (LESSONS 18, #178: the check sits
at the privilege boundary, immediately before use):

* on the controller the root bridge (:func:`root_overlay_file`) reads the
  built file through a no-follow walk in which every directory and the file
  itself are root's alone, and hashes the bytes it read through that same
  descriptor against the table; when it is absent or wrong the bridge builds
  it there and then from the diff the installer fetched and the image's own
  file - the image is pulled at the first deploy, after the install - and
  checks it the same way before it is mounted;
* on a worker the controller builds that worker's copy from the worker's OWN
  image file (read over SSH) and the controller's fetched diff, writes it
  under a content-hashed name as the MoE entry program is written, and reads
  its sha256 back before the unit is written (:func:`ensure_overlay_on_worker`);
  the worker's login is a full sudoer, so this is hygiene there, not a
  boundary (VD-143). No vLLM byte travels in Vaelor's package either way;
* when a built file is already in place, the image's own file must still have
  the pristine sha256 (:func:`image_copy_verdict`): the patch replaces the
  whole file, so on any other file it would silently replace different code.
  Read with ``docker run --rm --pull never --network none --entrypoint
  sha256sum`` (never a download); the
  bridge remembers a match per image digest - an image named by digest
  cannot change.

The client never supplies a path or a hash: only this table's values reach
``ExecStart``.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

from .vllm_drafters import Schedule, max_lookahead
from .vllm_images import VLLM_027_ROCM10, VllmImage

#: The ``vllm_options`` field a deployment records its overlay under.
OVERLAY_FIELD = "attention_overlay"

#: The variable PR #59448 reads (in this overlay, straight from the
#: environment), and the backend whose kernel it changes.
THREE_D_MAX_Q_VARIABLE = "VLLM_TRITON_3D_MAX_Q"
TRITON_BACKEND = "TRITON_ATTN"
TRITON_BACKEND_WORDS = ("--attention-backend", TRITON_BACKEND)

#: The draft model's own backend, set to the SAME value from the same decision.
#: Measured on the Z2 (pinned 0.27 image, newer runner, DFlash drafter): the
#: target on TRITON_ATTN alone fails at engine start with ``assert
#: kv_cache.shape[1] == 2`` in ``vllm/v1/worker/gpu/attn_utils.py``
#: (``_restride_blocks_first_kv_cache_to_kv_first_storage``) - the target's
#: blocks-first cache and the auto-selected ROCM_ATTN drafter's kv-first cache
#: are shared tensors and the restride refuses the mix; with the drafter on
#: TRITON_ATTN too the server starts and serves. ``SpeculativeConfig.
#: attention_backend`` (``vllm/config/speculative.py`` at f46a9dfe2, the image's
#: vLLM commit) takes the name through its ``mode="before"`` validator
#: (``AttentionBackendEnum[value.upper()]``); the dotted flag reaches it through
#: ``FlexibleArgumentParser.parse_args`` (``vllm/utils/argparse_utils.py``),
#: which tries JSON, then a human-readable integer, and keeps the plain string -
#: the same path VD-213's other ``--speculative-config.*`` words take.
DRAFTER_BACKEND_WORD = "--speculative-config.attention_backend=" + TRITON_BACKEND

#: **The VD-215 decision hook.** Whether a one-copy-per-machine deployment that
#: serves with a draft model on an image with an overlay row gets the overlay,
#: the ``VLLM_TRITON_3D_MAX_Q`` variable and the Triton attention backend - all
#: three together, because the fix lives only in that backend's kernel and the
#: variable is what turns it on. Set by VD-215's end-to-end measurement on the
#: Z2 (long-prompt decode with and without the drafter at 1 and 8 users,
#: ROCM_ATTN unpatched against TRITON_ATTN patched). ``False`` renders every
#: deployment exactly as before VD-215; nothing else needs to change to flip it.
OVERLAY_SERVES_WITH_DRAFTER = True


@dataclass(frozen=True)
class VllmOverlay:
    """One file mounted read-only over one pinned image's own copy."""

    #: What a deployment records. A new file is a new key, never the same key
    #: with other bytes, so a record always names the file it was made with.
    key: str
    image_key: str
    #: The pinned image, by digest (`vllm_images`); a test ties the two.
    image: str
    container_path: str
    #: The sha256 of the image's own file at ``container_path``.
    pristine_sha256: str
    #: The PR's diff: a URL naming both commits. GitHub renders that diff
    #: itself (abbreviated index lines, the PR's other files), so its BYTES
    #: are not pinned; the sha256 of the parsed hunks for the one file used is
    #: (`vllm_overlay_build.hunks_digest`). The real guarantee is the pristine
    #: and built hashes either side of it.
    diff_url: str
    hunks_sha256: str
    #: The sha256 of the file built from the two (`vllm_overlay_build.build`),
    #: Vaelor's edits included - so a change to them is a change to this.
    built_sha256: str
    #: The largest ``VLLM_TRITON_3D_MAX_Q`` a unit may carry: the most verify
    #: rows per request the split-context path was MEASURED with (VD-215: 5-8
    #: query tokens in the side test, 8 in session E). A longer lookahead the
    #: Advanced section allows (up to 15) is capped here, so its larger verify
    #: steps take upstream's own 2D path - the unpatched, safe direction.
    max_q_ceiling: int
    upstream: str


SPLIT_KV_VERIFY = VllmOverlay(
    # A new built file is a new key: Vaelor's tuning values are in it. -t2
    # (VD-216) adds Vaelor's decode/verify kernel and the cheaper P rounding;
    # -t3 sizes its query slots per step and per launch mode.
    key="vllm-pr59448-cce2aaa-t3",
    image_key=VLLM_027_ROCM10.key,
    image=VLLM_027_ROCM10.image,
    container_path=(
        "/opt/python/lib/python3.14/site-packages/vllm/v1/attention/ops/"
        "triton_unified_attention.py"
    ),
    pristine_sha256="992a2bc892e2e2b43fbd3c8163816ccf7e97ced56cb542bd827adb0ddb2df9fa",
    diff_url=(
        "https://github.com/vllm-project/vllm/compare/"
        "3eb6cec22ad9bb098393021b956feaa97081fc85..."
        "cce2aaadf3207b84fe2f1c5add4cfabfe63c8cc5.diff"
    ),
    hunks_sha256="72b024fdf98ef426ed70756955a6cd6a8bd8d2af91388344c25dcb12c0f11920",
    built_sha256="2578fea1ccf6a78c34eb42e6c13e15fb5c069730ce6f2c0185dbb8063c993a96",
    max_q_ceiling=8,
    upstream="vllm-project/vllm#59448 at cce2aaadf3207b84fe2f1c5add4cfabfe63c8cc5",
)

#: Every overlay, by the key a deployment records.
OVERLAYS: Dict[str, VllmOverlay] = {SPLIT_KV_VERIFY.key: SPLIT_KV_VERIFY}

#: **Keys a release no longer builds, and the row each now serves as** (VD-216,
#: LESSONS 22; the expert kernel's -1 -> -2 rule, `vllm_moe_kernels.RETIRED_KEYS`).
#: ``-t1`` is VD-215's file (PR #59448 plus the prompt-reading tuning),
#: deployed on both machines on 2026-10-12; ``-t2`` adds Vaelor's decode and
#: verify kernel, deployed on both on 2026-10-14; ``-t3`` sizes that kernel's
#: query slots per step (VD-216). This release builds only -t3, so a record of
#: -t1 or -t2 can be neither served as recorded nor refused (a deployment
#: nothing could load until removed). Each is served as -t3 deliberately: the
#: same image, mount point, variable and backends - only the file differs -
#: and the next Load records -t3. The older folders on a machine (and a
#: worker's older copies) are removed by `vllm_overlay_prune` once nothing
#: mounts or names them; their fetched diff - the same pinned hunks - still
#: serves a -t3 build offline (:func:`fetched_diff`), so the prune copies it
#: into the current folder first, or keeps the folder that holds it.
RETIRED_OVERLAYS: Dict[str, str] = {
    "vllm-pr59448-cce2aaa-t1": SPLIT_KV_VERIFY.key,
    "vllm-pr59448-cce2aaa-t2": SPLIT_KV_VERIFY.key,
}

#: The one overlay each image takes, by image key.
_BY_IMAGE: Dict[str, VllmOverlay] = {item.image_key: item for item in OVERLAYS.values()}

#: What a stored overlay this release does not carry is refused with.
OVERLAY_UNKNOWN = (
    "This deployment was made with a patched vLLM attention file this Vaelor "
    "version does not carry. Deploy the model again to serve it with this "
    "version's settings."
)

#: Why a copy serves without the draft model when the controller's root bridge
#: is the previous version and does not know this release's key (VD-216):
#: degrade, never fail - the same rule as the expert kernel's.
BRIDGE_PREDATES_OVERLAY = (
    "the hardware bridge on this controller is still the previous Vaelor "
    "version, which does not know this release's attention file. Restart the "
    "hardware bridge (or the machine) and load the model again to use it."
)


def decide_overlay(image: VllmImage, drafter: Optional[str], replicated: bool) -> Optional[str]:
    """The overlay key a NEW deployment records, or ``None`` (module docstring)."""
    if not OVERLAY_SERVES_WITH_DRAFTER or drafter is None or not replicated:
        return None
    overlay = _BY_IMAGE.get(image.key)
    return None if overlay is None else overlay.key


def require_overlay(
    value: Any, *, image: VllmImage, drafter: Optional[str],
) -> Optional[VllmOverlay]:
    """The renderer's rule for a stored overlay key, on both sides of the bridge.

    ``None`` is no overlay. Otherwise it must be a key of :data:`OVERLAYS`, for
    this deployment's image, on a deployment that serves with a draft model -
    the only combination :func:`decide_overlay` produces.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = RETIRED_OVERLAYS.get(value, value)
    if not isinstance(value, str) or value not in OVERLAYS:
        raise ValueError(OVERLAY_UNKNOWN)
    overlay = OVERLAYS[value]
    if overlay.image_key != image.key:
        raise ValueError(
            "The patched vLLM attention file is made for the {} image only."
            .format(IMAGE_NAMES.get(overlay.image_key, overlay.image_key))
        )
    if drafter is None:
        raise ValueError(
            "The patched vLLM attention file is used only with a draft model."
        )
    return overlay


#: The images' names, for the refusal above.
IMAGE_NAMES = {VLLM_027_ROCM10.key: VLLM_027_ROCM10.name}


def three_d_max_q(schedule: Schedule, overlay: Optional["VllmOverlay"] = None) -> int:
    """``VLLM_TRITON_3D_MAX_Q`` for a draft schedule: one target token plus its
    largest lookahead, so every verify step of this schedule is admitted - but
    never above the row's measured ceiling (:attr:`VllmOverlay.max_q_ceiling`).

    The VD-215 side test ran the patched kernel at 5-8 query tokens; the
    kernel itself keeps a batch whose tokens exceed its segment scratch on the
    old path, which is why 8 users measured unchanged. The 30B's E2 table
    ``[[1,1,4],[2,8,3]]`` gives 5: every value from 5 up admits the same steps.
    """
    ceiling = (overlay or SPLIT_KV_VERIFY).max_q_ceiling
    return min(1 + max_lookahead(schedule), ceiling)


def overlay_environment(overlay: VllmOverlay, schedule: Schedule) -> List[str]:
    """The one container variable an overlaid unit carries."""
    value = three_d_max_q(schedule, overlay)
    if not 2 <= value <= overlay.max_q_ceiling:
        raise ValueError("The draft model's lookahead is outside what the patched kernel takes.")
    return ["{}={}".format(THREE_D_MAX_Q_VARIABLE, value)]


def overlay_words(already_triton: bool) -> List[str]:
    """The ``vllm serve`` words an overlaid unit adds: the target's Triton
    backend, once, and ALWAYS the draft model's, to the same value - one
    without the other does not start (:data:`DRAFTER_BACKEND_WORD`)."""
    return ([] if already_triton else list(TRITON_BACKEND_WORDS)) + [DRAFTER_BACKEND_WORD]


def overlay_mount(host_path: str, overlay: VllmOverlay) -> List[str]:
    """The ``docker run`` option mounting the overlay read-only over the image's file."""
    return ["-v", "{}:{}:ro".format(host_path, overlay.container_path)]


def mount_pattern(overlay: VllmOverlay) -> str:
    """What a rendered overlay mount must match, for the bridge's shape check
    (`bridge_managed_units`): a plain absolute host path, then exactly this
    row's in-image path, read-only."""
    return r"/[A-Za-z0-9._/-]{1,255}:" + re.escape(overlay.container_path) + ":ro"


def env_in_bounds(overlay: VllmOverlay, value: str) -> bool:
    """Whether ``value`` is the ``-e`` an overlaid unit may carry: the
    variable, from 2 to the row's ceiling, in plain digits."""
    found = re.fullmatch(re.escape(THREE_D_MAX_Q_VARIABLE) + "=([1-9][0-9]?)", value)
    return found is not None and 2 <= int(found.group(1)) <= overlay.max_q_ceiling


def check_overlay_shape(
    options: List[Tuple[str, str]], rest: List[str], overlay: Optional[VllmOverlay],
) -> None:
    """The mount, the variable and both Triton backends: all four, once each, or none.

    The root bridge's shape check (`bridge_managed_units.check_rendered_unit`)
    has already admitted each shape only for the image whose row ``overlay``
    is, and never on a split. Here they must come together: the mount without
    the variable is a dead file, the variable without the mount reaches the
    image's unpatched kernel, which ignores it, and the target's backend
    without the draft model's does not start (:data:`DRAFTER_BACKEND_WORD`).
    """
    mounts = [value for name, value in options if name == "-v"
              and overlay is not None and re.fullmatch(mount_pattern(overlay), value)]
    variables = [value for name, value in options if name == "-e"
                 and value.startswith(THREE_D_MAX_Q_VARIABLE + "=")]
    if not mounts and not variables:
        return
    backend = " ".join(TRITON_BACKEND_WORDS)
    served = " ".join(rest)
    if (len(mounts) != 1 or len(variables) != 1 or backend not in served
            or rest.count(DRAFTER_BACKEND_WORD) != 1):
        raise ValueError(
            "The rendered unit carries part of the patched vLLM attention file's "
            "settings without the rest."
        )


def overlay_for_image(image_ref: str) -> Optional[VllmOverlay]:
    """The overlay a pinned image reference takes, or ``None``."""
    for overlay in OVERLAYS.values():
        if overlay.image == image_ref:
            return overlay
    return None


# --- Where the built file lives ----------------------------------------------

#: The root-only folder each overlay is built into, one sub-folder per key:
#: ``root:root 0755``, its files ``0644`` (`state_root_layout.ROOT_FOLDERS`).
OVERLAY_ROOT = "/var/lib/vaelor/vllm-overlays"
DIFF_NAME = "upstream.diff"
BUILT_NAME = "triton_unified_attention.py"


def built_path(overlay: VllmOverlay, root: str = OVERLAY_ROOT) -> str:
    """The built file's path on the controller."""
    return "{}/{}/{}".format(root, overlay.key, BUILT_NAME)


def diff_path(overlay: VllmOverlay, root: str = OVERLAY_ROOT) -> str:
    """The fetched diff's path on the controller."""
    return "{}/{}/{}".format(root, overlay.key, DIFF_NAME)


def fetched_diff(
    overlay: VllmOverlay, root: str = OVERLAY_ROOT, *, owners: FrozenSet[int] = frozenset({0}),
) -> Optional[bytes]:
    """The diff the installer fetched for ``overlay``, root's alone; ``None`` when absent.

    Its own folder first, then the folders of retired keys that now serve as
    it (:data:`RETIRED_OVERLAYS`), newest first: each retired row pinned the same URL and
    hunks, and the build checks the hunks again, so an upgrade that could not
    fetch (offline) still builds the new file from the old download.
    """
    retired = sorted((old for old, new in RETIRED_OVERLAYS.items() if new == overlay.key),
                     reverse=True)
    for key in [overlay.key] + retired:
        data = root_controlled_bytes("{}/{}/{}".format(root, key, DIFF_NAME), owners=owners)
        if data is not None:
            return data
    return None


def worker_path(overlay: VllmOverlay) -> str:
    """Where a WORKER keeps its built copy: beside the entry program, content-hashed."""
    # Imported here: `vllm_container` reaches `vllm_serve_options`, which imports this.
    from .vllm_container import MODEL_CACHE_ROOT

    return "{}/pull/vllm-overlay-{}.py".format(MODEL_CACHE_ROOT, overlay.built_sha256[:16])


# --- The checks before a start ---------------------------------------------

class OverlayUnavailable(RuntimeError):
    """The overlay could not be fetched, built or checked here; the reason is plain words."""


#: How every "not on this machine" reason starts, so the controller can tell
#: it from any other refusal the bridge sends back and install the unit
#: without the overlay instead (`gpu_pool_runtime`).
UNAVAILABLE_MARK = (
    "The speed fix for the draft model could not be set up on this machine, "
    "so this copy serves without the draft model: "
)
NOT_FETCHED = (
    "the vLLM change has not been downloaded on the controller. Re-run the "
    "Vaelor installer with internet access to fetch it."
)
BUILT_NOT_ROOTS = (
    "the built file in {} can be changed by an account other than root. "
    "Re-running the Vaelor installer returns it to root.".format(OVERLAY_ROOT)
)
IMAGE_FILE_CHANGED = (
    "the vLLM image's own attention file is not the one Vaelor's patch was "
    "made for (its checksum differs); the image on this machine is not the "
    "pinned one."
)
IMAGE_FILE_UNREAD = "the vLLM image's own attention file could not be checked: {reason}"


def without_draft_model(options: Any) -> Dict[str, Any]:
    """The serving options a copy WITHOUT the speed fix is started with (owner, VD-215).

    The owner's ruling, 2026-10-12: a machine without the patched file serves
    without a draft model - the pre-VD-213 serve (setup A), never slower than
    before - rather than the draft model on the old path, which measured
    slower than no draft model at long prompts. So the overlay, the draft
    model and the newer runner go (A was measured on the older runner); the
    request cap and the batch budget stay. Only this machine's unit is
    rendered so; the deployment's record keeps what it was deployed with, so
    a later Load tries the fix again.
    """
    from .vllm_guesser_build import GUESSER_FIELD
    from .vllm_serve_advanced import DRAFTER_FIELD, NEW_RUNNER_FIELD

    # VD-216 item 2 (R85): the 4-bit guesser is built FROM the draft model, so
    # it goes with it - a guesser key left without its drafter is refused by
    # the renderer, which would turn this fallback into a failed start.
    dropped = {OVERLAY_FIELD, DRAFTER_FIELD, NEW_RUNNER_FIELD, GUESSER_FIELD}
    return {key: value for key, value in dict(options or {}).items() if key not in dropped}


def older_bridge_overlay_refusal(error: Any, options: Any) -> Optional[str]:
    """The set-up refusal a copy falls back on when the controller's root bridge
    is the previous version and refuses this release's key as one it does not
    carry (`OVERLAY_UNKNOWN` is its words): the copy then serves without the
    draft model and says why (VD-216 - degrade, never fail; LESSONS 1)."""
    carried = options if isinstance(options, Mapping) else {}
    if carried.get(OVERLAY_FIELD) is not None and OVERLAY_UNKNOWN in str(error):
        return UNAVAILABLE_MARK + BRIDGE_PREDATES_OVERLAY
    return None


def unavailable(reason: str) -> OverlayUnavailable:
    """An :class:`OverlayUnavailable` the controller recognises (:data:`UNAVAILABLE_MARK`)."""
    text = str(reason)
    return OverlayUnavailable(text if text.startswith(UNAVAILABLE_MARK) else UNAVAILABLE_MARK + text)


#: The verdicts already read, by (image, in-image path): the sha256 found.
_IMAGE_VERDICTS: Dict[Tuple[str, str], str] = {}

#: The most bytes the no-follow read takes: the file is about 47 KB.
_READ_LIMIT = 1 << 20

_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_DIRECTORY_FLAGS = _FILE_FLAGS | getattr(os, "O_DIRECTORY", 0)
_GROUP_OR_OTHER_WRITE = stat.S_IWGRP | stat.S_IWOTH


def root_controlled_bytes(
    path: str, *, owners: FrozenSet[int] = frozenset({0}),
) -> Optional[bytes]:
    """``path``'s bytes when no account but root can change it or anything above it.

    ``None`` otherwise, or when it is absent. One walk from ``/`` with no link
    followed: every directory owned by root and writable by nobody else (or
    sticky, with the next entry root's), the file a regular, root-owned,
    single-name file nobody else can write - and the bytes are read through
    the very descriptor those facts were taken from, so no swap between the
    check and the read is possible (#178). ``owners`` exists for the tests,
    which cannot run as root; the bridge never passes it.
    """
    from . import bridge_safe_fs as safe_fs

    text = str(path)
    parts = [part for part in text.split("/") if part]
    if not text.startswith("/") or not parts or not safe_fs.supported():
        return None
    *folders, leaf = parts

    def sealed(folder: os.stat_result, entry_owner: int) -> bool:
        if folder.st_uid not in owners:
            return False
        if not folder.st_mode & _GROUP_OR_OTHER_WRITE:
            return True
        return bool(folder.st_mode & stat.S_ISVTX) and entry_owner in owners

    try:
        current = os.open("/", _DIRECTORY_FLAGS)
        try:
            current_status = os.fstat(current)
            for name in folders:
                child = safe_fs.open_child_directory(current, name)
                child_status = os.fstat(child)
                os.close(current)
                current = child
                if not sealed(current_status, child_status.st_uid):
                    return None
                current_status = child_status
            handle = os.open(leaf, _FILE_FLAGS | getattr(os, "O_NONBLOCK", 0), dir_fd=current)
        finally:
            os.close(current)
        try:
            status = os.fstat(handle)
            if not (
                sealed(current_status, status.st_uid) and stat.S_ISREG(status.st_mode)
                and status.st_uid in owners and status.st_nlink == 1
                and not status.st_mode & _GROUP_OR_OTHER_WRITE
            ):
                return None
            data = b""
            while len(data) <= _READ_LIMIT:
                chunk = os.read(handle, 65536)
                if not chunk:
                    break
                data += chunk
        finally:
            os.close(handle)
    except (OSError, ValueError):
        return None
    return None if len(data) > _READ_LIMIT else data


def _sha(data: Optional[bytes]) -> str:
    return "" if data is None else hashlib.sha256(data).hexdigest()


def image_copy_argv(overlay: VllmOverlay) -> List[str]:
    """The fixed command that reads the image's own file's sha256: no network, no GPU."""
    return [
        "docker", "run", "--rm", "--pull", "never", "--network", "none",
        "--entrypoint", "sha256sum",
        overlay.image, overlay.container_path,
    ]


def image_copy_verdict(
    overlay: VllmOverlay, read: Callable[[List[str]], str], *, remember: bool,
) -> None:
    """Refuse unless the image's own file still has the pristine sha256.

    ``read`` runs the fixed argv and returns its stdout, raising when it
    fails. With ``remember`` - the root bridge, which only ever asks about its
    own machine - a matching verdict is kept for the image digest; a mismatch
    or a failed read never is, so the next start asks again. The worker path
    does not remember: one control-plane process serves several machines, and
    the check is each machine's own.
    """
    cache = (overlay.image, overlay.container_path)
    if remember and _IMAGE_VERDICTS.get(cache) == overlay.pristine_sha256:
        return
    try:
        found = str(read(image_copy_argv(overlay))).split()
    except Exception as error:  # the bridge's RuntimeError or a transport's own
        reason = str(error)[:200] or "the check printed nothing"
        raise unavailable(IMAGE_FILE_UNREAD.format(reason=reason)) from error
    if not found or found[0] != overlay.pristine_sha256:
        raise unavailable(IMAGE_FILE_CHANGED)
    if remember:
        _IMAGE_VERDICTS[cache] = found[0]


def root_overlay_file(
    overlay: VllmOverlay, *, fixed: Callable[..., str],
    read_image: Callable[[VllmOverlay], bytes],
    owners: FrozenSet[int] = frozenset({0}), root: str = OVERLAY_ROOT,
    owner: Tuple[int, int] = (0, 0),
) -> str:
    """The controller's built file, checked - and built first when needed; refused in plain words.

    Called by the root bridge (`bridge_managed_units`) before it renders a
    unit that carries the mount. ``fixed`` runs one fixed command as root and
    returns its stdout; ``read_image`` returns the image's own file's bytes.
    ``root`` and ``owner`` are test seams. Every refusal is an
    :class:`OverlayUnavailable` starting with :data:`UNAVAILABLE_MARK`.
    """
    from . import vllm_overlay_build as builder

    path = built_path(overlay, root)
    if _sha(root_controlled_bytes(path, owners=owners)) == overlay.built_sha256:
        image_copy_verdict(overlay, lambda argv: fixed(*argv), remember=True)
        return path
    diff = fetched_diff(overlay, root, owners=owners)
    if diff is None:
        raise unavailable(NOT_FETCHED)
    try:
        built = builder.build(read_image(overlay), diff, overlay)
        builder.install_built(overlay, built, root=root, owner=owner)
    except OverlayUnavailable as error:
        builder.record_status(overlay, str(error), root=root, owner=owner)
        raise unavailable(str(error)) from error
    except OSError as error:
        raise unavailable("the built file could not be written ({}).".format(error)) from error
    if _sha(root_controlled_bytes(path, owners=owners)) != overlay.built_sha256:
        raise unavailable(BUILT_NOT_ROOTS)
    # The build read the image's own file and checked its hash just now.
    _IMAGE_VERDICTS[(overlay.image, overlay.container_path)] = overlay.pristine_sha256
    return path


def ensure_overlay_on_worker(
    transport: Any, overlay: VllmOverlay, *,
    read_diff: Optional[Callable[[VllmOverlay], Optional[bytes]]] = None,
) -> str:
    """Build a WORKER's copy from that worker's own image file, put it there, check it.

    A copy already there with the built sha256 is kept, and the image's own
    file is checked on that machine instead. Otherwise the worker's image file
    is read over SSH, built here with the controller's fetched diff
    (``read_diff``, the installer's file by default), written the way the MoE
    entry program is (`vllm_entry_program.write_program_on_worker`) under a
    content-hashed name, and its sha256 read back. Refusals are
    :class:`OverlayUnavailable` starting with :data:`UNAVAILABLE_MARK`.
    """
    from . import vllm_overlay_build as builder
    from .ssh_transport import SshTransportError
    from .vllm_entry_program import write_program_on_worker

    target = worker_path(overlay)

    def found_sha() -> str:
        try:
            return (str(transport.run(["sha256sum", target], sudo=True)).split() or [""])[0]
        except SshTransportError:
            return ""

    if found_sha() == overlay.built_sha256:
        image_copy_verdict(
            overlay, lambda argv: transport.run(argv, sudo=True, timeout=300), remember=False,
        )
        return target
    diff = (read_diff or fetched_diff)(overlay)
    if diff is None:
        raise unavailable(NOT_FETCHED)
    try:
        # Base64 (`image_file_argv`): the SSH transport strips its output and
        # decodes it as UTF-8, so the file's own bytes would not survive it.
        pristine = builder.decode_image_file(transport.run(
            builder.image_file_argv(overlay), sudo=True, timeout=300))
        built = builder.build(pristine, diff, overlay)
    except SshTransportError as error:
        raise unavailable("the vLLM image's attention file could not be read there ({}).".format(
            str(error)[:160])) from error
    except OverlayUnavailable as error:
        raise unavailable(str(error)) from error
    transport.run(["install", "-d", "-m", "0755", target.rsplit("/", 1)[0]], sudo=True)
    write_program_on_worker(transport, target, built.decode("utf-8"), overlay.built_sha256)
    if found_sha() != overlay.built_sha256:
        raise unavailable("the built file did not arrive intact there.")
    return target
