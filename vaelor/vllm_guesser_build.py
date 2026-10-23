"""The 4-bit guesser: built on each machine from the pinned bf16 drafter, as root (VD-216 item 2).

**What it is.** WO2 (2026-10-15, kb N57/G60) measured a compressed-tensors
w4a16, group-32 copy of the 30B's DFlash drafter (`vllm_drafters`) on the Z2:
the guesser's own time 4.0/4.2 -> 2.4/2.5 ms a step, acceptance held, steps
4.5-6.5% faster, HumanEval within the known flips. The guesser cannot change
an answer - every guess is verified by the 30B (D82) - so only it may be lossy.

**No 4-bit artifact ships** (D88). Each machine BUILDS it at install from the
bf16 drafter already in its model store, with Vaelor's own pinned program
(`vllm_guesser_program`, :data:`PROGRAM_SHA256`), run in ONE locked-down
container of the pinned vLLM 0.27 image (:func:`build_argv`: ``--pull never``,
no network, no GPU, read-only root filesystem, every capability dropped,
bounded memory, pids and time), whose own compressed-tensors 0.17 packs it.
Nothing is downloaded.

**The check is a pinned digest of the output**, because the build is
deterministic: measured on the Z2, repeated builds on 4, 8 and 32 cores gave
the same bytes, and the same bytes WO2 served (kb N65). So a machine serves
the 4-bit guesser only when EVERY output file carries its pinned sha256
(:attr:`GuesserBuild.outputs`) - a different result is never "close enough".
The input is pinned too (:attr:`GuesserBuild.inputs`), checked by the
program on the bytes it quantizes.

**Where it lives** (:func:`installed_dir`): ``<overlay root>/<key>/guesser``,
root's, ``0755``/``0644`` - the serving container mounts it read-only at
:data:`CONTAINER_DIR`; nobody but root can change it. A record beside it
(``record.json``) says why a build was refused.

**Who builds it** (the HIP expert kernels' pattern, `vllm_moe_hip_build`):

* the installer, as root: ``python -m vaelor.vllm_guesser_build build``
  (about 25 s on the Z2), BEFORE it queues the upgrade's reload, so the
  reload serves it; it never fails the install;
* on a WORKER, the controller over SSH, on the worker with the worker's own
  image, when it installs that worker's serving unit (:func:`ensure_on_worker`)
  - the VD-194 path the expert kernel's files and HIP build already take;
* the controller's root bridge never builds inside a unit install (the
  build is longer than its 30 s pre-start budget): it checks
  (:func:`root_guesser_dir`), and when nothing was ever built here it starts
  the same installer step in the background (:func:`background_build_argv`),
  so the next Load uses it.

**Every refusal serves the bf16 drafter** - today's behaviour - and says so:
the refusal starts with :data:`UNAVAILABLE_MARK`, the controller installs that
machine's unit without the guesser (:func:`fallback_step`) and the copy's row
carries the sentence (`gpu_pool_replicas`). Never a refused deploy.

The host-side work - prepare, settle, check, note - is ONE stdlib program
(`vllm_guesser_host.HOST_PROGRAM`), run in-process on the controller and as ``python3 -c``
on a worker, which has no Vaelor package (the `vllm_overlay_prune` arrangement).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from .vllm_drafters import DFLASH_QWEN3_30B, drafter_profile
from .vllm_images import VLLM_027_ROCM10, VllmImage
from .vllm_moe_hip_build import BUILD_LOCK as HIP_BUILD_LOCK
from .vllm_overlays import OVERLAY_ROOT, root_controlled_bytes
from .vllm_guesser_host import run_host, run_host_on_worker  # noqa: F401 - re-exported
from .vllm_serve_advanced import FUSION_FIELD, SPEEDUPS_MEASURED, measured_gpus

#: The ``vllm_options`` field a deployment records its guesser build under.
GUESSER_FIELD = "guesser"

#: **The decision hook.** ``False`` renders every deployment as before item 2.
GUESSER_SERVES = True

#: Where the built folder is mounted in the serving container.
CONTAINER_DIR = "/opt/vaelor-guesser"

#: The program the build runs, in Vaelor's installed package.
PROGRAM_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "vllm_guesser_program.py")
#: Its name in the build folder and in the container.
PROGRAM_NAME = "quantize.py"

#: The model store's hub, where the pull put the bf16 drafter (the serving
#: units mount it; `vllm_container.MODEL_CACHE_ROOT`, tied by a test).
MODEL_HUB = "/var/lib/vaelor/models/hub"

#: The folders and files inside ``<overlay root>/<key>``.
GUESSER_DIR = "guesser"
RECORD = "record.json"
BUILD_DIR = "build"
#: The lock, taken by the HIP build's own helper (`vllm_moe_hip_build._lock`),
#: so its name is that helper's: one spelling for both builds.
BUILD_LOCK = HIP_BUILD_LOCK
#: Every build container's name starts with this; each run adds its own token.
CONTAINER_PREFIX = "vaelor-guesser-build-"

#: The bound on one build: measured 21-31 s on the Z2 (4 to 32 cores).
BUILD_SECONDS = 600
#: A worker's per-build folder older than this is a crashed build's: no
#: build runs longer than its bound, plus the settle after it.
SWEEP_AFTER_SECONDS = 2 * BUILD_SECONDS
#: The least time left in the root bridge's pre-start budget worth a check
#: (hashing about 1 GB: about a second on the Z2).
CHECK_MIN_SECONDS = 4
#: The largest output file accepted (the weights are 1,000,288,096 bytes).
OUTPUT_LIMIT = 2 << 30
#: The transient unit the bridge starts the background build as.
BACKGROUND_UNIT = "vaelor-guesser-build"


@dataclass(frozen=True)
class GuesserBuild:
    """One pinned 4-bit build of one pinned drafter, for one pinned image."""

    #: What a deployment records; a new recipe, program or output is a new key.
    key: str
    #: The `vllm_drafters` key it is built from (repo and revision are its).
    drafter: str
    image_key: str
    #: The pinned image, by digest (`vllm_images`); a test ties the two.
    image: str
    group_size: int
    #: ``(file, sha256)`` of the bf16 drafter's files at its pinned revision.
    inputs: Tuple[Tuple[str, str], ...]
    #: sha256 of `vllm_guesser_program` (a test ties it to the file).
    program_sha256: str
    #: ``(file, sha256)`` of every output file: the pinned result.
    outputs: Tuple[Tuple[str, str], ...]


W4A16_G32 = GuesserBuild(
    key="dflash-qwen3-30b-a3b-2507-w4a16g32-1",
    drafter=DFLASH_QWEN3_30B.key,
    image_key=VLLM_027_ROCM10.key,
    image=VLLM_027_ROCM10.image,
    group_size=32,
    # Read on the Z2 at revision 27694eaf (2026-10-15); the weight file's is
    # also its Hugging Face LFS object name.
    inputs=(
        ("config.json", "c98b4d07384c9eaa88fece253114113b5c4a897d0978bb8fb4d4393089512f22"),
        ("config.py", "b71d94e8ac2f7dc32c610046209d18cac058655df6092af0844f655805350a17"),
        ("model.safetensors", "22ef21bf72671069f80c4564d871f441e6b9535d8acbc82a6ef4a5ac4264a5b4"),
    ),
    program_sha256="9d4ffd96373c8c8cc5f19a2169507e30dfe7a6918d1be88e2d033238f60a658a",
    # Built twice on the Z2 (2026-10-15, kb N65); the weights are byte for
    # byte the file WO2 measured (12e325fd...).
    outputs=(
        ("config.json", "dd01d2ab011ae3e96791f9c3cb9a46a4a466ec3711c08b357664c86cde14a0cd"),
        ("config.py", "b71d94e8ac2f7dc32c610046209d18cac058655df6092af0844f655805350a17"),
        ("model.safetensors", "12e325fdf9d4029de5bfe27211131d04c191c224023fdcbb7bd39edf3ff4c883"),
    ),
)

#: Every build, by the key a deployment records.
GUESSERS: Dict[str, GuesserBuild] = {W4A16_G32.key: W4A16_G32}

#: **Keys a release no longer builds, and the row each now serves as**
#: (LESSONS 22, the `vllm_moe_kernels.RETIRED_KEYS` rule): a record of an old
#: key is served as the current row - same drafter, image and mount - and the
#: next Load records the new key. Every retired key maps straight to a current
#: row, never to another retired key. None yet: ``-1`` is the first build.
RETIRED_GUESSERS: Dict[str, str] = {}

#: What a stored key this release does not carry is refused with.
GUESSER_UNKNOWN = (
    "This deployment was made with a version of the 4-bit guesser this Vaelor "
    "release does not build. Deploy the model again to serve it with this "
    "release's settings."
)

#: How every "not on this machine" reason starts, so the controller can tell it
#: from any other refusal and install the unit with the bf16 guesser instead.
UNAVAILABLE_MARK = (
    "The 4-bit guesser could not be used on this machine, so this copy serves "
    "the full-size guesser: "
)
NOT_BUILT = "it has not been built on this machine yet"
INPUT_ABSENT = (
    "the full-size guesser is not in this machine's model store yet; the 4-bit "
    "copy is built from it after the model is fetched"
)
IMAGE_ABSENT = "the vLLM 0.27 image is not on this machine yet"
PROGRAM_NOT_PINNED = (
    "the installed Vaelor package's build program is not the pinned one (its "
    "checksum differs). Re-running the Vaelor installer restores it"
)
BUILD_FAILED = "the build in the vLLM image did not finish ({})"
OUTPUT_NOT_PINNED = (
    "its files are not the pinned 4-bit guesser (a file's checksum differs), "
    "so it was not used; the next install builds it again"
)
NOT_ROOTS = "its folder or files can be changed by an account other than root"
STRAY = "its folder holds files Vaelor did not put there ({})"
MISSING = "its folder is missing {}"
UNREADABLE = "its files could not be read ({})"
BUILD_BUSY = (
    "another build of it was running on this machine; the next start uses its result"
)
CHECK_DEFERRED = (
    "this start had no time left to check it after the other pre-start checks; "
    "the next start checks it"
)
#: Why a copy serves without item 2's speed-ups when the controller's bridge
#: has not been restarted since the upgrade and refuses their fields by name:
#: a Load adds them by itself, so this must degrade, never fail (the expert
#: kernel's rule, `vllm_moe_kernels.BRIDGE_PREDATES_KERNEL`).
BRIDGE_PREDATES_SPEEDUPS = (
    "the hardware bridge on this controller is still the previous Vaelor "
    "version, which does not know the 4-bit guesser or the fused qk-norm/RoPE "
    "step. Restart the hardware bridge (or the machine) and load the model "
    "again to use them"
)
#: How a bridge without a field refuses it (`vllm_serve_options._refuse`).
FIELD_REFUSED = re.compile(r"serving option '([^']*)' is invalid")
BUILD_STARTED = (
    "; a build was started in the background, so the next Load of the model uses it"
)


class GuesserUnavailable(RuntimeError):
    """The 4-bit guesser cannot serve on this machine; the reason is plain words."""


def unavailable(reason: str) -> GuesserUnavailable:
    """A :class:`GuesserUnavailable` the controller recognises (:data:`UNAVAILABLE_MARK`)."""
    text = str(reason)
    return GuesserUnavailable(text if text.startswith(UNAVAILABLE_MARK) else UNAVAILABLE_MARK + text)


# --- The decision and the record -------------------------------------------------

def decide_guesser(image: VllmImage, drafter: Optional[str], replicated: bool,
                   gpu_targets: Any, repo: Any = None) -> Optional[str]:
    """The build key a deployment serves with, or ``None``.

    Where WO2 measured it: the pinned drafter it is built from, on its image,
    one copy per machine, every machine gfx1151 (`vllm_serve_advanced.
    measured_gpus`), and - when ``repo`` is named - the 30B it was measured
    with (`vllm_serve_advanced.SPEEDUPS_MEASURED`).
    """
    if not GUESSER_SERVES or drafter is None or not replicated or not measured_gpus(gpu_targets):
        return None
    if repo is not None and (image.key, str(repo).strip()) not in SPEEDUPS_MEASURED:
        return None
    for row in GUESSERS.values():
        if row.drafter == drafter and row.image_key == image.key:
            return row.key
    return None


def require_guesser(value: Any, *, image: VllmImage, drafter: Optional[str]) -> Optional[GuesserBuild]:
    """The renderer's rule for a stored key, on both sides of the bridge; retired keys resolve."""
    if value is None:
        return None
    if isinstance(value, str):
        value = RETIRED_GUESSERS.get(value, value)
    if not GUESSER_SERVES:
        # The hook off (R86): every unit renders as before item 2, stored ones too.
        return None
    if not isinstance(value, str) or value not in GUESSERS:
        raise ValueError(GUESSER_UNKNOWN)
    row = GUESSERS[value]
    if row.image_key != image.key:
        raise ValueError("The 4-bit guesser is built for the vLLM 0.27 image only.")
    if drafter != row.drafter:
        raise ValueError("The 4-bit guesser is built from the {} draft model only.".format(
            drafter_profile(row.drafter).label))
    return row


def without_guesser(options: Any) -> Dict[str, Any]:
    """The serving options a copy WITHOUT the 4-bit guesser starts with: the bf16 drafter."""
    return {key: value for key, value in dict(options or {}).items() if key != GUESSER_FIELD}


def refused_fields(text: Any) -> set:
    """The field names a bridge's "serving option '...' is invalid" refusal lists, exactly."""
    found = FIELD_REFUSED.search(str(text))
    return set(found.group(1).split(", ")) if found else set()


def fallback_step(error: Any, options: Any) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """``(which, reason, options without it)`` when ``error`` is a refusal of item 2.

    The guesser's own refusal (:data:`UNAVAILABLE_MARK`) strips the guesser:
    ``"guesser"``. A bridge from before item 2 refusing either field by name
    strips both, since it knows neither: ``"guesser"`` when one was carried,
    else ``"fusion"`` (R87 round 1, item 6). Anything else is ``None``.
    """
    text = str(error)
    carried = options if isinstance(options, Mapping) else {}
    if carried.get(GUESSER_FIELD) is not None and UNAVAILABLE_MARK in text:
        return ("guesser", text[text.index(UNAVAILABLE_MARK) + len(UNAVAILABLE_MARK):][:400],
                without_guesser(carried))
    named = refused_fields(text)
    ours = {field for field in (GUESSER_FIELD, FUSION_FIELD) if carried.get(field) is not None}
    if ours and (ours & named or (GUESSER_FIELD in ours and GUESSER_UNKNOWN in text)):
        return ("guesser" if GUESSER_FIELD in ours else "fusion", BRIDGE_PREDATES_SPEEDUPS,
                {key: value for key, value in without_guesser(carried).items()
                 if key != FUSION_FIELD})
    return None


# --- The unit: mount and its shape -----------------------------------------------

def installed_dir(row: GuesserBuild, root: str = OVERLAY_ROOT) -> str:
    """The built folder, the one that is mounted: the same path on a worker."""
    return "{}/{}/{}".format(root, row.key, GUESSER_DIR)


#: The ``docker run`` option the built folder is mounted with: ``--mount``,
#: never ``-v`` - ``-v`` makes an empty folder when the source is gone between
#: the check and the start, ``--mount`` refuses the start (round 1, item 8),
#: which :func:`start_refusal` turns into the bf16 fallback.
MOUNT_OPTION = "--mount"


def guesser_mount(host_dir: str) -> List[str]:
    """The ``docker run`` option mounting the built folder read-only."""
    return [MOUNT_OPTION, "type=bind,source={},target={},readonly".format(host_dir, CONTAINER_DIR)]


def mount_pattern(row: GuesserBuild, host_dir: Optional[str] = None) -> str:
    """What a rendered guesser mount must match: exactly the built folder's path."""
    return "type=bind,source={},target={},readonly".format(
        re.escape(host_dir or installed_dir(row)), re.escape(CONTAINER_DIR))


def model_word() -> str:
    """The draft-model word a unit with the 4-bit guesser carries."""
    return "--speculative-config.model={}".format(CONTAINER_DIR)


def guesser_for_image(image_ref: str) -> Optional[GuesserBuild]:
    """The build a pinned image reference takes, or ``None``."""
    for row in GUESSERS.values():
        if row.image == image_ref:
            return row
    return None


def check_guesser_shape(options: List[Tuple[str, str]], rest: List[str],
                        row: Optional[GuesserBuild], host_dir: Optional[str] = None) -> None:
    """The mount and the model word: both, once each, or neither.

    The mount without the word is a folder nothing reads; the word without
    the mount names a folder that is not there, and vLLM would refuse to start.
    """
    mounts = [value for name, value in options if name == MOUNT_OPTION and row is not None
              and re.fullmatch(mount_pattern(row, host_dir), value)]
    words = [word for word in rest if word == model_word()]
    if not mounts and not words:
        return
    if row is None or len(mounts) != 1 or len(words) != 1:
        raise ValueError(
            "The rendered unit carries part of the 4-bit guesser's settings without the rest."
        )


#: How long a start that mounts the guesser is watched for Docker refusing the
#: mount. Measured on the Z2 through a product-shaped unit (R90): the refusal
#: (exit 125) is in systemd's state 0.26 s after the start. There is no early
#: "good" signal through the reads both transports admit - the unit is
#: active/running from the first instant either way, and the container's own
#: state needs a ``docker inspect`` the root bridge's argv policy does not
#: take - so a healthy start pays the whole window: kept short, 8x the
#: measured refusal, polled every quarter second.
START_WATCH_SECONDS = 2
START_POLL_SECONDS = 0.25
#: Docker's words for a ``--mount`` whose source is gone.
MOUNT_GONE = "bind source path does not exist"
GONE_AT_START = (
    "its folder was gone when the model started (Docker refused the mount), so "
    "this copy was started again with the full-size guesser"
)


def start_refusal(runtime: Any, transport: Any, unit: str, host_dir: str, *,
                  seconds: float = START_WATCH_SECONDS,
                  sleep: Callable[[float], None] = None,
                  clock: Callable[[], float] = None) -> Optional[str]:
    """:data:`GONE_AT_START` when Docker refused this unit's guesser mount, else ``None``.

    The check and the start are two moments (round 1, item 8): a folder that
    vanished between them makes ``--mount`` refuse the container, and systemd
    would retry that for ever while the Load waited out its cap. So a start
    that mounts it is watched briefly; a refusal that names the missing mount
    is the guesser's fallback. Docker names the HOST source, never the
    container path (R90, read on the Z2: ``invalid mount config for type
    "bind": bind source path does not exist: <host_dir>``, exit 125), so the
    match is on ``host_dir`` - the folder this unit was rendered with. A state or journal that cannot be read says
    nothing (LESSONS 8: the observer failing is not evidence).
    """
    import time

    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    deadline = clock() + seconds
    while True:
        try:
            state = runtime.unit_state(transport, unit)
        except Exception:  # noqa: BLE001 - an unread state is no reading
            return None
        if not isinstance(state, Mapping) or not state.get("ActiveState"):
            return None
        if state.get("Result") not in (None, "", "success"):
            try:
                journal = str(runtime.unit_journal(transport, unit))
            except Exception:  # noqa: BLE001 - an unread journal is no reading
                return None
            gone = "{}: {}".format(MOUNT_GONE, str(host_dir).rstrip("/"))
            return GONE_AT_START if gone in journal else None
        if clock() >= deadline:
            return None
        sleep(START_POLL_SECONDS)


# --- The host program: one text, here and on a worker ------------------------------

def host_plan(row: GuesserBuild, verb: str, *, root: str = OVERLAY_ROOT,
              owners: Sequence[int] = (0,), **extra: Any) -> Dict[str, Any]:
    """One host-program call's plan: the row's pinned outputs and the folder names."""
    plan = {"verb": verb, "root": root, "key": row.key, "folder": GUESSER_DIR,
            "outputs": dict(row.outputs), "owners": sorted(set(owners)),
            "limit": OUTPUT_LIMIT}
    plan.update(extra)
    return plan


def reason_of(verdict: Mapping[str, Any]) -> str:
    """The plain sentence for a host-program answer that is not ready.

    Never built here: the record's reason, if a build was refused. A refused
    settle records the program's own code (R87), so that is put in words too.
    """
    code = str(verdict.get("reason") or "")
    if code == "not-built":
        recorded = verdict.get("record")
        if isinstance(recorded, Mapping) and recorded.get("reason"):
            return _sentence(str(recorded["reason"])) if recorded["reason"] != "not-built"                 else NOT_BUILT
        return NOT_BUILT
    return _sentence(code)


def _sentence(code: str) -> str:
    """One host-program code in plain words; a sentence already is one."""
    fixed = {"not-built": NOT_BUILT, "not-roots": NOT_ROOTS, "not-pinned": OUTPUT_NOT_PINNED,
             "program-not-pinned": PROGRAM_NOT_PINNED}
    if code in fixed:
        return fixed[code]
    for prefix, sentence in (("stray:", STRAY), ("missing:", MISSING), ("unreadable:", UNREADABLE)):
        if code.startswith(prefix):
            return sentence.format(code[len(prefix):][:160])
    return code[:300] or NOT_BUILT


# --- The build: one fixed container ------------------------------------------------

def source_dir(row: GuesserBuild, hub: str = MODEL_HUB) -> str:
    """The bf16 drafter's hub folder (snapshots and blobs) the build reads."""
    return "{}/models--{}".format(hub, drafter_profile(row.drafter).repo.replace("/", "--"))


def container_name() -> str:
    """A build container's own name: :data:`CONTAINER_PREFIX`, then twelve hex digits.

    The first six are this process group's (`post_install_steps.group_tag`), so
    an update step that is stopped removes its own containers and nobody
    else's (VD-217, R101); the last six are fresh.
    """
    from .post_install_steps import group_tag

    return CONTAINER_PREFIX + group_tag() + secrets.token_hex(3)


def build_argv(row: GuesserBuild, source: str, build: str, name: str) -> List[str]:
    """The one fixed build command: the drafter in, read-only; the 4-bit copy out.

    ``--mount`` rather than ``-v``: a source that is missing is refused, never
    created. ``/tmp`` is a small tmpfs because torch will not import without a
    temporary folder. The plan names the snapshot and the pinned inputs.
    """
    plan = {"source": "/src/snapshots/" + drafter_profile(row.drafter).revision, "out": "/out",
            "group": row.group_size, "inputs": dict(row.inputs)}
    return [
        "docker", "run", "--rm", "--name", name, "--pull", "never", "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,size=256m", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--memory", "8g", "--pids-limit", "512",
        "--mount", "type=bind,source={},target=/src,readonly".format(source),
        "--mount", "type=bind,source={}/{},target=/prog/{},readonly".format(
            build, PROGRAM_NAME, PROGRAM_NAME),
        "--mount", "type=bind,source={}/out,target=/out".format(build),
        "--entrypoint", "python3", row.image, "-I", "/prog/" + PROGRAM_NAME,
        json.dumps(plan, sort_keys=True, separators=(",", ":")),
    ]


def remove_argv(name: str) -> List[str]:
    """Stop and remove a build container by its name (a timeout kills only the client)."""
    return ["docker", "rm", "-f", name]


def _brief(error: Any) -> str:
    from . import vllm_moe_hip_build as hip

    text = " ".join(str(error).split())
    # Docker's own words for a missing image, read by the HIP build's one reader.
    if hip._brief(error) == hip.IMAGE_ABSENT:
        return IMAGE_ABSENT
    if MOUNT_GONE in text:
        return INPUT_ABSENT
    # The program's own refusal is one JSON line on stdout: say its reason.
    found = re.search(r'"reason": "([^"]{1,300})"', text)
    return BUILD_FAILED.format(found.group(1) if found else (text[:200] or type(error).__name__))


def program_text(source: str = PROGRAM_SOURCE) -> Optional[str]:
    """The installed program, read no-follow as root's alone, or ``None``."""
    data = root_controlled_bytes(source)
    try:
        return None if data is None else data.decode("utf-8")
    except UnicodeDecodeError:
        return None


# --- The controller: the installer and the background build, as root ------------------

def _run_fixed(argv: List[str], timeout: float = BUILD_SECONDS) -> str:
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                               check=False)
    if completed.returncode != 0:
        raise RuntimeError(" ".join([str(completed.stdout or ""), str(completed.stderr or "")])
                           .strip()[:600] or "the build container ended without saying why")
    return str(completed.stdout or "")


def ensure_on_controller(
    row: GuesserBuild, *, root: str = OVERLAY_ROOT, hub: str = MODEL_HUB,
    run: Callable[[List[str]], Any] = _run_fixed, lock_wait: float = 0.0,
    owners: FrozenSet[int] = frozenset({0}), program: Optional[str] = None,
) -> Optional[str]:
    """Make this machine's build current: ``None`` when it is, else why not. Never raises.

    A current build is kept. Otherwise, under :data:`BUILD_LOCK` (one build at
    a time; a second finds it held and says so), the checked program is copied
    into a fresh root-only build folder, the container builds into it, and the
    result replaces the served folder only if every file carries its pin.
    """
    from .vllm_moe_hip_build import _lock

    if run_host(host_plan(row, "check", root=root, owners=owners))["ready"]:
        return None
    try:
        from . import bridge_safe_fs as safe_fs

        with safe_fs.directory(root) as root_fd:
            os.close(safe_fs.ensure_root_directory(root_fd, row.key, mode=0o755))
        lock = _lock("{}/{}".format(root, row.key), lock_wait)
    except (OSError, ValueError, ImportError) as error:
        return "the build folder could not be made ({})".format(str(error)[:160])
    if lock is None:
        return BUILD_BUSY
    try:
        if run_host(host_plan(row, "check", root=root, owners=owners))["ready"]:
            return None
        return _build_locked(row, root, hub, run, owners, program)
    finally:
        os.close(lock)


def _build_locked(row: GuesserBuild, root: str, hub: str, run: Callable[[List[str]], Any],
                  owners: FrozenSet[int], program: Optional[str]) -> Optional[str]:
    """The build itself, under the lock (:func:`ensure_on_controller`)."""
    name = container_name()
    reason: Optional[str] = None
    try:
        text = program_text() if program is None else program
        if text is None or hashlib.sha256(text.encode("utf-8")).hexdigest() != row.program_sha256:
            raise ValueError(PROGRAM_NOT_PINNED)
        prepared = run_host(host_plan(row, "prepare", root=root, owners=owners, build=BUILD_DIR,
                                      program=text, program_sha256=row.program_sha256,
                                      program_name=PROGRAM_NAME))
        if not prepared.get("ok"):
            raise ValueError(reason_of(prepared))
        try:
            run(build_argv(row, source_dir(row, hub), "{}/{}/{}".format(root, row.key, BUILD_DIR),
                           name))
        except Exception as error:  # the runner's own words, shortened
            try:
                run(remove_argv(name))
            except Exception:  # noqa: BLE001 - nothing left to remove is the usual answer
                pass
            raise ValueError(_brief(error)) from error
        settled = run_host(host_plan(row, "settle", root=root, owners=owners, build=BUILD_DIR))
        if not settled.get("ready"):
            reason = reason_of(settled)
    except (OSError, ValueError) as error:
        reason = str(error)[:300] or type(error).__name__
        try:
            run_host(host_plan(row, "note", root=root, owners=owners, reason=reason))
        except Exception:  # noqa: BLE001 - the reason is returned either way
            pass
    finally:
        try:
            run_host(host_plan(row, "clean", root=root, owners=owners, build=BUILD_DIR))
        except Exception:  # noqa: BLE001 - a left build folder is removed by the next build
            pass
    return reason


def background_build_argv(python: str = sys.executable) -> List[str]:
    """The fixed command the root bridge starts the installer's build step with.

    A transient unit, so it outlives the bridge's call and is bounded by
    systemd; ``--collect`` so a failed one never blocks the next start of it.
    A second start while one runs is refused by systemd, which is the point.
    """
    return ["systemd-run", "--unit", BACKGROUND_UNIT, "--collect", "--no-block", "--quiet",
            "--property=RuntimeMaxSec={}".format(BUILD_SECONDS + 60),
            python, "-I", "-m", "vaelor.vllm_guesser_build", "build"]


def worth_starting(verdict: Mapping[str, Any]) -> bool:
    """Whether the bridge should start a background build: never built here, or
    refused only for something a later moment fixes (the image or the drafter
    arrived since). A refused result is not rebuilt at every Load."""
    if verdict.get("reason") != "not-built":
        return False
    recorded = verdict.get("record")
    return not isinstance(recorded, Mapping) or recorded.get("reason") in (INPUT_ABSENT, IMAGE_ABSENT)


def root_guesser_dir(
    row: GuesserBuild, *, root: str = OVERLAY_ROOT, owners: FrozenSet[int] = frozenset({0}),
    seconds_left: float = float(CHECK_MIN_SECONDS),
    start_build: Optional[Callable[[List[str]], Any]] = None,
) -> str:
    """The controller's checked built folder, for the root bridge; else :class:`GuesserUnavailable`.

    No build here (module docstring); ``start_build`` runs
    :func:`background_build_argv` when :func:`worth_starting` says so.
    """
    if seconds_left < CHECK_MIN_SECONDS:
        raise unavailable(CHECK_DEFERRED)
    try:
        verdict = run_host(host_plan(row, "check", root=root, owners=owners))
    except Exception as error:  # noqa: BLE001 - an unreadable folder is not ready
        raise unavailable(UNREADABLE.format(str(error)[:160])) from error
    if verdict.get("ready"):
        return installed_dir(row, root)
    reason = reason_of(verdict)
    if start_build is not None and worth_starting(verdict):
        # VD-133 1.5.5 #10: a record worth a new build names what an EARLIER
        # moment lacked (the drafter, the image) - the serve asking now has
        # just fetched both, so repeating it told the owner the drafter was
        # missing while it sat in the store. What is true now is "not built".
        reason = NOT_BUILT
        try:
            start_build(background_build_argv())
            reason += BUILD_STARTED
        except Exception:  # noqa: BLE001 - the reason stands without it
            pass
    raise unavailable(reason)


# --- A worker: built there, by the controller, over SSH as root -------------------------

def ensure_on_worker(transport: Any, row: GuesserBuild, *, source: str = PROGRAM_SOURCE,
                     seconds: float = BUILD_SECONDS) -> str:
    """The worker's built folder - built there first when needed - or :class:`GuesserUnavailable`.

    Built with the worker's own image from the worker's own copy of the drafter
    (VD-194: the controller provisions what its workers run). The same host
    program checks, prepares and settles, as root, over SSH.
    """
    from .ssh_transport import SshTransportError

    def host(verb: str, **extra: Any) -> Dict[str, Any]:
        return run_host_on_worker(transport, host_plan(row, verb, **extra))

    try:
        if host("check").get("ready"):
            return installed_dir(row)
    except (SshTransportError, ValueError) as error:
        raise unavailable("it could not be checked there ({})".format(str(error)[:160])) from error
    name = container_name()
    # Its own folder per build: two deploys reaching one worker never share one.
    build = "{}-{}".format(BUILD_DIR, name[len(CONTAINER_PREFIX):])
    reason: Optional[str] = None
    try:
        transport.run(["install", "-d", "-m", "0755", "{}/{}".format(OVERLAY_ROOT, row.key)],
                      sudo=True)
        # A build that crashed (the controller restarted mid-build) left its
        # ~1 GB folder (round 1, item 7): removed once it is older than any
        # build can run, so a build in progress is never touched.
        try:
            host("sweep", older_than=SWEEP_AFTER_SECONDS)
        except (SshTransportError, ValueError):
            pass
        text = program_text(source)
        if text is None or hashlib.sha256(text.encode("utf-8")).hexdigest() != row.program_sha256:
            raise ValueError(PROGRAM_NOT_PINNED)
        prepared = host("prepare", build=build, program=text, program_sha256=row.program_sha256,
                        program_name=PROGRAM_NAME)
        if not prepared.get("ok"):
            raise ValueError(reason_of(prepared))
        try:
            transport.run(build_argv(row, source_dir(row), "{}/{}/{}".format(
                OVERLAY_ROOT, row.key, build), name), sudo=True,
                timeout=max(1, int(min(BUILD_SECONDS, seconds))))
        except SshTransportError as error:
            try:
                transport.run(remove_argv(name), sudo=True)
            except SshTransportError:
                pass
            raise ValueError(_brief(error)) from error
        settled = host("settle", build=build)
        if not settled.get("ready"):
            reason = reason_of(settled)
    except (SshTransportError, ValueError) as error:
        reason = str(error)[:300] or type(error).__name__
        try:
            host("note", reason=reason)
        except (SshTransportError, ValueError):
            pass
    finally:
        try:
            host("clean", build=build)
        except (SshTransportError, ValueError):
            pass
    if reason is not None:
        raise unavailable(reason)
    return installed_dir(row)


# --- The installer ------------------------------------------------------------------

def main(argv: Sequence[str], *, root: str = OVERLAY_ROOT,
         run: Optional[Callable[[List[str]], Any]] = None) -> int:
    """``build [--within N]``, as root: every row's build, each outcome printed; never
    fails the install. ``--within`` (VD-217, R98) keeps the lock wait and the
    build inside the update's share, so a timed-out build container is
    removed here, by name, before the update's own bound is reached."""
    from .post_install_steps import step_timeout, within_seconds

    within = within_seconds(argv, ("build",))
    if within is None:
        sys.stderr.write("usage: python -m vaelor.vllm_guesser_build build [--within SECONDS]\n")
        return 2
    deadline = time.monotonic() + min(within, 2.0 * BUILD_SECONDS)
    run = run or (lambda command: _run_fixed(command, step_timeout(command, deadline, BUILD_SECONDS)))
    for row in GUESSERS.values():
        try:
            reason = ensure_on_controller(row, root=root, run=run, lock_wait=min(
                BUILD_SECONDS, max(0.0, deadline - time.monotonic())))
        except Exception as error:  # noqa: BLE001 - an installer step never fails the install
            reason = str(error)[:300] or type(error).__name__
        print("Vaelor 4-bit guesser build ({}): {}".format(row.key, "done" if reason is None
                                                           else reason))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
