"""Vaelor's own expert kernel, delivered into a pinned vLLM image as a plugin folder (VD-216).

**What it is.** VD-216 measured decode-sized expert (MoE) kernels of
Vaelor's own (clean-room, GPL-2.0-only, `vllm_moe_plugin/`) ahead of vLLM's
retuned Triton kernel on the Z2: a per-pair kernel below 5 tokens, "d2"
(gfx11 dot instructions) from 5 to 40 tokens - vLLM's largest captured step
at 8 users - and a prompt-reading path above that, up to vLLM's 8,192-token
batch; and kernels for the 30B's dense 4-bit layers (qkv and o) from 6
tokens. They serve those calls on gfx1151 and vLLM's own kernels serve the
rest. It is a vLLM *general plugin*: a folder holding
``vaelor_moe_kernel.py``, ``vaelor_moe_triton.py``, ``vaelor_moe_d2.py``,
``vaelor_moe_prefill.py``, ``vaelor_dense_w4.py`` and a minimal
``vaelor_moe_kernel-1.dist-info`` whose entry point vLLM finds on
``PYTHONPATH``. It wraps two vLLM functions at run time; no file of vLLM's
is modified, and no vLLM code travels in Vaelor.

**Delivery mirrors the VD-215 overlay (`vllm_overlays`).** One table
(:data:`MOE_KERNELS`) read by the deploy, the renderer on both sides of the
root bridge, the bridge's shape check and the worker path. A row is bound to
ONE pinned image by digest and to ONE model repo, and pins the sha256 of the
plugin's files (:func:`files_digest`), so a record always names the files it
was made with: changing the kernel is a new key, and a test refuses a file
change that keeps the old one (LESSONS 6).

**When a deployment gets it** (:func:`decide_moe_kernel`): the 30B on the
0.27 image, one copy per machine, while :data:`MOE_KERNEL_SERVES` says so.
The unit then carries a read-only mount of the folder at
:data:`CONTAINER_DIR` and ``-e PYTHONPATH=`` that folder followed by the
image's own ``PYTHONPATH`` (read from the pinned image's config: overriding it
alone would drop the image's own entry). A split never carries it.

**Records** (VD-213's rule): the key is stored in ``vllm_options`` as
:data:`MOE_KERNEL_FIELD`, written only when set. A deployment made before has
no such field and loads exactly as deployed; a new deploy of the 30B gets it;
an older root bridge refuses the unknown field by name, which the deploy says
as "restart the hardware bridge" (`vllm_serve_options.BRIDGE_TOO_OLD`).

**Before every start that carries the mount** (LESSONS 18, #178):

* on the controller the root bridge (:func:`root_kernel_dir`) reads the
  plugin files out of the installed package through a no-follow walk proving
  only root can change them, checks their digest against the row, writes them
  into a root-only folder under `vllm_overlays.OVERLAY_ROOT` (the bridge's one
  writable root folder; everything in it is made root's at every bridge start)
  and reads each back the same way before the folder is mounted, so the
  container imports exactly the checked bytes - never a byte-code cache pip
  left beside the package's copy;
* on a worker the controller writes the same files into the SAME root-only
  path (:func:`installed_dir`; not the store's group-writable pull folder,
  because the folder is first on ``PYTHONPATH`` and a planted
  ``sitecustomize.py`` there would run in the container), as the MoE entry
  program is written (`vllm_entry_program.write_program_on_worker`), and reads
  every sha256 back before the unit is written (:func:`ensure_kernel_on_worker`);
* on both, the folder is then listed with ``lstat`` and no link followed, by
  ONE lister (:data:`TREE_LISTER`, run in-process on the controller and as
  ``python3 -c`` on a worker) and judged by ONE rule (:func:`tree_refusal`):
  exactly the plugin's two folders and files (and this machine's HIP build,
  a code object and its record, when there is one), every one of them and
  every folder above them root's and writable by nobody else, no link, no
  file with a second name. The bridge's shape check admits the mount only
  from that path (:func:`mount_pattern`).

**A machine that cannot set it up serves without it** - vLLM's own expert
kernel, the serve VD-215 shipped - and says so the way the overlay's skip is
said: the refusal starts with :data:`KERNEL_UNAVAILABLE_MARK`, the controller
installs that machine's unit without the mount and the variable, and the
deployment's row carries the sentence (`gpu_pool_replicas`). The deploy
never fails for it. Inside the container the plugin stands aside by itself
(and logs it) when vLLM is not the pinned version or the method it wraps has
changed.

**What the plugin did is read back** (:func:`journal_note`): it writes one
marker line per event to the server's log, and once the copy answers its
health check the controller reads that unit's journal through the reviewed
``journalctl`` shape both transports already admit (`unit_journal`, the
read a failed start uses) and adds a sentence to the copy's row
(`gpu_pool_replicas`) when the plugin stood aside, failed, was not loaded, or
was never recorded into a GPU graph. A start-up is about 120 log lines on the
Z2, inside the 200 the reviewed shape reads.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

from .vllm_images import VLLM_027_ROCM10, VllmImage
from .vllm_overlays import OVERLAY_ROOT, root_controlled_bytes

#: The ``vllm_options`` field a deployment records its kernel under.
MOE_KERNEL_FIELD = "moe_kernel"

#: **The VD-216 decision hook.** Whether a qualifying new deployment gets the
#: kernel. ``False`` renders every deployment exactly as before VD-216.
MOE_KERNEL_SERVES = True

#: Where the folder is mounted in the container.
CONTAINER_DIR = "/opt/vaelor-moe"

#: The plugin folder in Vaelor's installed package.
PLUGIN_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm_moe_plugin")

#: The dist-info folder vLLM's entry-point scan finds the plugin by.
DIST_INFO = "vaelor_moe_kernel-1.dist-info"

#: The plugin's files, relative to the folder, in the order they are written.
PLUGIN_FILES = (
    "vaelor_moe_kernel.py",
    "vaelor_moe_triton.py",
    "vaelor_moe_d2.py",
    "vaelor_moe_prefill.py",
    "vaelor_dense_w4.py",
    "vaelor_dense_mx.py",
    "vaelor_moe_d2_hip.py",
    "vaelor_moe_d2.hip",
    DIST_INFO + "/METADATA",
    DIST_INFO + "/entry_points.txt",
)


@dataclass(frozen=True)
class VllmMoeKernel:
    """Vaelor's expert-kernel plugin for one pinned image and one model."""

    #: What a deployment records; a new set of files is a new key.
    key: str
    image_key: str
    #: The pinned image, by digest (`vllm_images`).
    image: str
    #: The one model repo whose expert layers it was built and measured for.
    repo: str
    #: The image's own ``PYTHONPATH``, kept after the plugin folder.
    image_pythonpath: str
    #: sha256 of the plugin files (:func:`files_digest`).
    files_sha256: str


EXPERT_KERNEL = VllmMoeKernel(
    key="vaelor-moe-w4a16-7",
    image_key=VLLM_027_ROCM10.key,
    image=VLLM_027_ROCM10.image,
    repo="cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
    # The image's own ``Config.Env`` PYTHONPATH, verified with ``docker image
    # inspect`` against this exact digest on the Z2 (2026-10-12, VD-216 review
    # F7); a test pins it to the digest, so a re-pinned image forces a re-check.
    image_pythonpath="/opt/python/lib/python3.14/site-packages/_rocm_sdk_core/share/amd_smi:",
    files_sha256="653114130b17fa7c5f0825157b4bf1b407f836dfb224ce02dbd27ba6e4021779",
)

#: Every kernel row, by the key a deployment records.
MOE_KERNELS: Dict[str, VllmMoeKernel] = {EXPERT_KERNEL.key: EXPERT_KERNEL}

#: **Keys a release no longer ships, and the row each now serves as** (VD-216,
#: LESSONS 22). ``-1`` was the per-pair kernel alone, deployed on both machines
#: on 2026-10-12; ``-2`` added d2 for 5-40 tokens and the gfx1151 gate;
#: ``-3`` retunes d2's plans and adds the prompt-reading path and the dense
#: 4-bit kernels (2026-10-14); ``-4`` adds the HIP d2 kernels each machine
#: builds for itself (`vllm_moe_hip_build`) and the Gluon dense kernels; ``-5``
#: adds the dense prompt-reading unpack path and the 1-5 token dense steps;
#: ``-6`` limits that unpack path to prompts above 1,536 tokens (kb X11);
#: ``-7`` adds the 65-384 token dense prompt plans and the expert prompt
#: launch values (WO1 fixes 3-4). The upgraded package carries only -7's files,
#: so a record of -1 or -2 can neither be served as recorded (its bytes are
#: gone from the package, and the copy left in its folder is stale) nor
#: refused (that is a deployment nothing can load until it is removed). Each
#: is served as -7, deliberately: the same model, the same image, the same
#: mount and variable - only the plugin's files differ - and the next Load
#: records -7, so the row then names what runs. An old key's folder is removed
#: by `vllm_overlay_prune` once no unit, mount or record names it. Every retired key maps straight to the current row, never to
#: another retired key (no chains to follow).
RETIRED_KEYS: Dict[str, str] = {
    "vaelor-moe-w4a16-1": EXPERT_KERNEL.key,
    "vaelor-moe-w4a16-2": EXPERT_KERNEL.key,
    "vaelor-moe-w4a16-3": EXPERT_KERNEL.key,
    "vaelor-moe-w4a16-4": EXPERT_KERNEL.key,
    "vaelor-moe-w4a16-5": EXPERT_KERNEL.key,
    "vaelor-moe-w4a16-6": EXPERT_KERNEL.key,
}

#: What a stored kernel key this release does not carry is refused with.
KERNEL_UNKNOWN = (
    "This deployment was made with a version of Vaelor's expert kernel this "
    "Vaelor release does not carry. Deploy the model again to serve it with "
    "this release's settings."
)


def decide_moe_kernel(image: VllmImage, repo: Optional[str], replicated: bool) -> Optional[str]:
    """The kernel key a NEW deployment records, or ``None`` (module docstring)."""
    if not MOE_KERNEL_SERVES or not replicated:
        return None
    for kernel in MOE_KERNELS.values():
        if kernel.image_key == image.key and kernel.repo == repo:
            return kernel.key
    return None


def require_moe_kernel(value: Any, *, image: VllmImage) -> Optional[VllmMoeKernel]:
    """The renderer's rule for a stored kernel key, on both sides of the bridge."""
    if value is None:
        return None
    if isinstance(value, str):
        value = RETIRED_KEYS.get(value, value)
    if not isinstance(value, str) or value not in MOE_KERNELS:
        raise ValueError(KERNEL_UNKNOWN)
    kernel = MOE_KERNELS[value]
    if kernel.image_key != image.key:
        raise ValueError("Vaelor's expert kernel is built for the {} image only.".format(
            VLLM_027_ROCM10.name if kernel.image_key == VLLM_027_ROCM10.key else kernel.image_key))
    return kernel


def require_kernel_model(kernel: VllmMoeKernel, model: Any) -> None:
    """The renderer refuses the kernel on any other model: its shapes are the 30B's."""
    if model != kernel.repo:
        raise ValueError("Vaelor's expert kernel serves {} only.".format(kernel.repo))


def kernel_environment(kernel: VllmMoeKernel) -> List[str]:
    """The one container variable a unit with the kernel carries."""
    return ["PYTHONPATH={}:{}".format(CONTAINER_DIR, kernel.image_pythonpath)]


def kernel_mount(host_dir: str, kernel: VllmMoeKernel) -> List[str]:
    """The ``docker run`` option mounting the plugin folder read-only."""
    del kernel  # every row mounts at the same place; the row picks the folder
    return ["-v", "{}:{}:ro".format(host_dir, CONTAINER_DIR)]


def mount_pattern(kernel: VllmMoeKernel, host_dir: Optional[str] = None) -> str:
    """What a rendered plugin mount must match, for the bridge's shape check:
    exactly the checked copy's path (:func:`installed_dir`, the same on a
    worker) - never any other folder (VD-216 review F6)."""
    return re.escape(host_dir or installed_dir(kernel)) + ":" + re.escape(CONTAINER_DIR) + ":ro"


def kernel_for_image(image_ref: str) -> Optional[VllmMoeKernel]:
    """The kernel a pinned image reference takes, or ``None``."""
    for kernel in MOE_KERNELS.values():
        if kernel.image == image_ref:
            return kernel
    return None


def check_kernel_shape(options: List[Tuple[str, str]], kernel: Optional[VllmMoeKernel],
                       host_dir: Optional[str] = None) -> None:
    """The plugin mount and its ``PYTHONPATH``: both, once each, or neither.

    The mount without the variable is a folder nothing imports; the variable
    without the mount names a folder that is not there. The bridge's option
    check has already admitted each only for this row's image, never on a split.
    """
    mounts = [value for name, value in options if name == "-v" and kernel is not None
              and re.fullmatch(mount_pattern(kernel, host_dir), value)]
    variables = [value for name, value in options if name == "-e"
                 and value.startswith("PYTHONPATH=")]
    if not mounts and not variables:
        return
    if (kernel is None or len(mounts) != 1 or len(variables) != 1
            or variables[0] != kernel_environment(kernel)[0]):
        raise ValueError(
            "The rendered unit carries part of Vaelor's expert kernel settings without the rest."
        )


def without_moe_kernel(options: Any) -> Dict[str, Any]:
    """The serving options a copy WITHOUT the kernel is started with: vLLM's own kernel."""
    return {key: value for key, value in dict(options or {}).items() if key != MOE_KERNEL_FIELD}


# --- The files ---------------------------------------------------------------

def files_digest(files: Mapping[str, bytes]) -> str:
    """sha256 over each file's name and sha256, in :data:`PLUGIN_FILES` order."""
    lines = "".join(
        "{}\0{}\n".format(name, hashlib.sha256(files[name]).hexdigest()) for name in PLUGIN_FILES
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def package_files(source: str = PLUGIN_SOURCE) -> Dict[str, bytes]:
    """The plugin files as this checkout or installed package holds them (tests, the worker path)."""
    files = {}
    for name in PLUGIN_FILES:
        with open(os.path.join(source, *name.split("/")), "rb") as handle:
            files[name] = handle.read()
    return files


def installed_dir(kernel: VllmMoeKernel, root: str = OVERLAY_ROOT) -> str:
    """The controller's root-only copy of the plugin folder, the one that is mounted."""
    return "{}/{}/plugin".format(root, kernel.key)


def worker_dir(kernel: VllmMoeKernel) -> str:
    """Where a WORKER keeps its copy: the controller's path, made root's there (module docstring)."""
    return installed_dir(kernel)


class KernelUnavailable(RuntimeError):
    """The kernel could not be set up on this machine; the reason is plain words."""


#: How every "not on this machine" reason starts, so the controller can tell
#: it from any other refusal and install the unit without the kernel instead.
KERNEL_UNAVAILABLE_MARK = (
    "Vaelor's expert kernel could not be set up on this machine, so this copy "
    "serves with vLLM's own expert kernel: "
)
SOURCE_NOT_ROOTS = (
    "the installed Vaelor package's kernel files can be changed by an account "
    "other than root, or are missing. Re-running the Vaelor installer restores them."
)
SOURCE_CHANGED = (
    "the installed Vaelor package's kernel files are not the ones this version "
    "was built with (their checksum differs). Re-running the Vaelor installer "
    "restores them."
)
COPY_NOT_ROOTS = (
    "the copy in {} could not be written or checked as root's alone.".format(OVERLAY_ROOT)
)
STRAY_ENTRY = "the kernel folder holds an entry Vaelor did not put there ({})."
FOLDER_BUSY = (
    "another Vaelor step on this machine (the installer) was still setting up the kernel "
    "folder; the next start uses what it finished."
)


def unavailable(reason: str) -> KernelUnavailable:
    """A :class:`KernelUnavailable` the controller recognises (:data:`KERNEL_UNAVAILABLE_MARK`)."""
    text = str(reason)
    return KernelUnavailable(
        text if text.startswith(KERNEL_UNAVAILABLE_MARK) else KERNEL_UNAVAILABLE_MARK + text)


def _root_files(folder: str, owners: FrozenSet[int]) -> Optional[Dict[str, bytes]]:
    """Each plugin file under ``folder`` read no-follow as root's alone, or ``None``."""
    files = {}
    for name in PLUGIN_FILES:
        data = root_controlled_bytes("{}/{}".format(folder, name), owners=owners)
        if data is None:
            return None
        files[name] = data
    return files


#: The ONE lister of a plugin folder (VD-216 review F4), run in-process by the
#: root bridge and as ``python3 -c`` on a worker: every entry under the folder
#: and every folder above it, by ``lstat`` - no link is ever followed - as
#: ``{"tree": {relative path: [kind, uid, mode, links]}, "above": [[path, uid, mode]]}``.
TREE_LISTER = """
import json, os, stat, sys


def _kind(mode):
    if stat.S_ISLNK(mode):
        return "link"
    if stat.S_ISDIR(mode):
        return "dir"
    return "file" if stat.S_ISREG(mode) else "other"


def tree(top):
    found = {}

    def walk(relative):
        path = top if relative == "." else top + "/" + relative
        status = os.lstat(path)
        found[relative] = [_kind(status.st_mode), status.st_uid,
                           stat.S_IMODE(status.st_mode) | (status.st_mode & stat.S_ISVTX),
                           status.st_nlink]
        if stat.S_ISDIR(status.st_mode):
            for name in sorted(os.listdir(path)):
                walk(name if relative == "." else relative + "/" + name)

    walk(".")
    above, path = [], os.path.dirname(top)
    while True:
        status = os.lstat(path)
        above.append([path, status.st_uid, stat.S_IMODE(status.st_mode)
                      | (status.st_mode & stat.S_ISVTX), _kind(status.st_mode)])
        if path == "/":
            break
        path = os.path.dirname(path)
    return {"tree": found, "above": above}
"""

#: What the folder holds, exactly: two folders and the plugin's files ...
_TREE = {".": "dir", DIST_INFO: "dir", **{name: "file" for name in PLUGIN_FILES}}
#: ... and, once this machine has built them, the HIP code object and its
#: record (`vllm_moe_hip_build`): either may be absent, nothing else may be there.
_BUILT = {"vaelor_moe_d2_hip.co": "file", "vaelor_moe_d2_hip.json": "file"}
_OTHERS_WRITE = 0o022
_STICKY = 0o1000


def local_tree(folder: str) -> Dict[str, Any]:
    """:data:`TREE_LISTER` run here (the root bridge's side)."""
    namespace: Dict[str, Any] = {}
    exec(compile(TREE_LISTER, "vaelor-moe-tree", "exec"), namespace)  # noqa: S102 - our own text
    return namespace["tree"](folder)


def tree_argv(folder: str) -> List[str]:
    """:data:`TREE_LISTER` as one ``python3 -c`` argv for a worker."""
    return ["python3", "-c", TREE_LISTER + "\nprint(json.dumps(tree(sys.argv[1])))\n", folder]


def tree_refusal(listing: Any, owners: FrozenSet[int] = frozenset({0})) -> Optional[str]:
    """Why a listed plugin folder must not be mounted, or ``None`` (module docstring)."""
    if not isinstance(listing, Mapping) or not isinstance(listing.get("tree"), Mapping):
        return "the kernel folder could not be listed."
    entries = listing["tree"]
    stray = sorted(set(entries) - set(_TREE) - set(_BUILT))
    if stray:
        return STRAY_ENTRY.format(", ".join(stray)[:200])
    missing = sorted(set(_TREE) - set(entries))
    if missing:
        return "the kernel folder is missing {}.".format(", ".join(missing))
    for relative in sorted(entries):
        kind, uid, mode, links = entries[relative]
        wanted = _TREE.get(relative) or _BUILT[relative]
        if kind != wanted:
            return "{} is a {}, not the {} Vaelor wrote (a link is never followed).".format(
                relative, kind, wanted)
        if uid not in owners or mode & _OTHERS_WRITE:
            return "{} in the kernel folder can be changed by an account other than root.".format(
                relative)
        if kind == "file" and links != 1:
            return "{} in the kernel folder has a second name elsewhere.".format(relative)
    for path, uid, mode, kind in listing.get("above") or [["", -1, 0, "dir"]]:
        if kind != "dir" or uid not in owners or (mode & _OTHERS_WRITE and not mode & _STICKY):
            return ("the folder above the kernel ({}) can be changed by an account other "
                    "than root.".format(path or "?"))
    return None


def _write_copy(kernel: VllmMoeKernel, files: Mapping[str, bytes], root: str,
                owner: Tuple[int, int]) -> None:
    """Write the files into the root-only copy, no link followed, each atomically."""
    from . import bridge_safe_fs as safe_fs

    with safe_fs.directory(root) as root_fd:
        keyed = safe_fs.ensure_root_directory(root_fd, kernel.key, mode=0o755, owner=owner)
        try:
            plugin = safe_fs.ensure_root_directory(keyed, "plugin", mode=0o755, owner=owner)
            try:
                dist = safe_fs.ensure_root_directory(plugin, DIST_INFO, mode=0o755, owner=owner)
                try:
                    for name in PLUGIN_FILES:
                        parent, leaf = (dist, name.split("/")[1]) if "/" in name else (plugin, name)
                        safe_fs.write_file(parent, leaf, files[name].decode("utf-8"), mode=0o644)
                finally:
                    os.close(dist)
            finally:
                os.close(plugin)
        finally:
            os.close(keyed)


def root_kernel_dir(
    kernel: VllmMoeKernel, *, owners: FrozenSet[int] = frozenset({0}),
    root: str = OVERLAY_ROOT, owner: Tuple[int, int] = (0, 0), source: str = PLUGIN_SOURCE,
    build: Optional[Callable[[List[str]], Any]] = None, lock_wait: float = 0.0,
) -> str:
    """The controller's checked copy of the plugin folder - written first when needed.

    Called by the root bridge before it renders a unit that mounts it, and by
    the installer. ``owners``, ``root``, ``owner`` and ``source`` are test
    seams; the bridge passes the installed package's folder. Every refusal is
    a :class:`KernelUnavailable` starting with :data:`KERNEL_UNAVAILABLE_MARK`.
    ``build`` runs the HIP build's one fixed command as root
    (`vllm_moe_hip_build.ensure_on_controller`): a HIP build that is missing
    or fails is never a refusal here - the plugin then serves on Triton and
    its log says why.

    **One writer at a time** (R50): the installer and the root bridge both run
    this as root. The copy, the HIP build and the folder check all happen
    under the build lock (`vllm_moe_hip_build`), waited for at most
    ``lock_wait`` seconds - so the check never lists the folder while the
    other is mid-write with a temporary beside a file. A lock still held
    after the wait is :data:`FOLDER_BUSY`: this start serves without the
    kernel, and the next uses what the other writer finished.
    """
    from . import bridge_safe_fs as safe_fs
    from .vllm_moe_hip_build import _lock

    try:
        with safe_fs.directory(root) as root_fd:
            os.close(safe_fs.ensure_root_directory(root_fd, kernel.key, mode=0o755, owner=owner))
        lock = _lock("{}/{}".format(root, kernel.key), lock_wait)
    except (OSError, ValueError, ImportError) as error:  # ImportError: no fcntl, not Linux
        raise unavailable(COPY_NOT_ROOTS) from error
    if lock is None:
        raise unavailable(FOLDER_BUSY)
    try:
        return _checked_copy(kernel, owners=owners, root=root, owner=owner, source=source,
                             build=build)
    finally:
        os.close(lock)


def _checked_copy(kernel: VllmMoeKernel, *, owners: FrozenSet[int], root: str,
                  owner: Tuple[int, int], source: str,
                  build: Optional[Callable[[List[str]], Any]]) -> str:
    """:func:`root_kernel_dir`'s work, under the lock."""
    from .vllm_moe_hip_build import ensure_on_controller

    folder = installed_dir(kernel, root)
    copied = _root_files(folder, owners)
    if copied is None or files_digest(copied) != kernel.files_sha256:
        shipped = _root_files(os.path.realpath(source), owners)
        if shipped is None:
            raise unavailable(SOURCE_NOT_ROOTS)
        if files_digest(shipped) != kernel.files_sha256:
            raise unavailable(SOURCE_CHANGED)
        try:
            _write_copy(kernel, shipped, root, owner)
        except (OSError, ValueError, UnicodeDecodeError) as error:
            raise unavailable("the copy could not be written ({}).".format(error)) from error
        copied = _root_files(folder, owners)
        if copied is None or files_digest(copied) != kernel.files_sha256:
            raise unavailable(COPY_NOT_ROOTS)
    ensure_on_controller(kernel, folder, build=build, owners=owners, owner=owner, locked=True)
    try:
        refused = tree_refusal(local_tree(folder), owners)
    except OSError as error:
        raise unavailable(COPY_NOT_ROOTS) from error
    if refused:
        raise unavailable(refused)
    return folder


def ensure_kernel_on_worker(
    transport: Any, kernel: VllmMoeKernel, *, source: str = PLUGIN_SOURCE,
) -> str:
    """Put the plugin folder on a WORKER, read every sha256 back, then judge the folder.

    The files come from the controller's own installed package and must carry
    the row's digest before anything is sent. The folder is the controller's
    root-only path; once the hashes match it is listed there with the one
    lister and refused by the one rule the bridge uses (review F4). Refusals
    are :class:`KernelUnavailable` starting with :data:`KERNEL_UNAVAILABLE_MARK`.
    """
    from .ssh_transport import SshTransportError
    from .vllm_entry_program import write_program_on_worker

    target = worker_dir(kernel)
    try:
        files = package_files(source)
    except OSError as error:
        raise unavailable("the controller's kernel files could not be read ({}).".format(
            error)) from error
    if files_digest(files) != kernel.files_sha256:
        raise unavailable(SOURCE_CHANGED)

    def found(name: str) -> str:
        try:
            output = transport.run(["sha256sum", "{}/{}".format(target, name)], sudo=True)
        except SshTransportError:
            return ""
        return (str(output).split() or [""])[0]

    wanted = {name: hashlib.sha256(files[name]).hexdigest() for name in PLUGIN_FILES}
    try:
        if any(found(name) != wanted[name] for name in PLUGIN_FILES):
            transport.run(["install", "-d", "-m", "0755", target + "/" + DIST_INFO], sudo=True)
            for name in PLUGIN_FILES:
                write_program_on_worker(transport, "{}/{}".format(target, name),
                                        files[name].decode("utf-8"), wanted[name])
    except SshTransportError as error:
        raise unavailable("the files could not be written there ({}).".format(
            str(error)[:160])) from error
    if any(found(name) != wanted[name] for name in PLUGIN_FILES):
        raise unavailable("the files did not arrive intact there.")
    # The HIP kernels, built THERE with that machine's own image; a failure is
    # recorded beside them and never refuses the plugin (`vllm_moe_hip_build`).
    from .vllm_moe_hip_build import ensure_on_worker

    ensure_on_worker(transport, kernel, target)
    try:
        listing = json.loads(str(transport.run(tree_argv(target), sudo=True)))
    except (SshTransportError, ValueError) as error:
        raise unavailable("the folder could not be listed there ({}).".format(
            str(error)[:160])) from error
    refused = tree_refusal(listing)
    if refused:
        raise unavailable(refused)
    return target


#: Why a copy serves without the kernel when the controller's root bridge is the
#: previous version and refuses the field (review nit b): degrade, never fail.
BRIDGE_PREDATES_KERNEL = (
    "the hardware bridge on this controller is still the previous Vaelor version, "
    "which does not know it. Restart the hardware bridge (or the machine) and load "
    "the model again to use it."
)


def older_bridge_refusal(error: Any, options: Any) -> Optional[str]:
    """The kernel's own set-up refusal when an older bridge refused it: by the
    field's name (a bridge from before the kernel), or as a key it does not
    carry (a bridge from before this key - `KERNEL_UNKNOWN` is its words)."""
    carried = options if isinstance(options, Mapping) else {}
    text = str(error)
    if carried.get(MOE_KERNEL_FIELD) is not None and (
            "'{}'".format(MOE_KERNEL_FIELD) in text or KERNEL_UNKNOWN in text):
        return KERNEL_UNAVAILABLE_MARK + BRIDGE_PREDATES_KERNEL
    return None


def fallback_step(error: Any, options: Any) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """``(which, reason, options without it)`` when ``error`` is a set-up refusal.

    ``which`` is ``"overlay"`` (VD-215: the copy serves without the draft
    model, `vllm_overlays.without_draft_model`) or ``"kernel"`` (VD-216: the
    copy serves with vLLM's own expert kernel). Anything else is ``None``,
    and the caller raises the error as it is.
    """
    from .vllm_overlays import OVERLAY_FIELD, UNAVAILABLE_MARK, without_draft_model

    text = str(error)
    carried = options if isinstance(options, Mapping) else {}
    for which, mark, field, strip in (
        ("overlay", UNAVAILABLE_MARK, OVERLAY_FIELD, without_draft_model),
        ("kernel", KERNEL_UNAVAILABLE_MARK, MOE_KERNEL_FIELD, without_moe_kernel),
    ):
        if mark in text and carried.get(field) is not None:
            reason = text[text.index(mark) + len(mark):][:400]
            return which, reason, strip(carried)
    return None


# --- What the plugin did, read back from the server's log (review F3) ---------

#: The plugin's marker word and events (`vllm_moe_plugin/vaelor_moe_kernel.py`
#: writes them; a test ties the two spellings).
MARKER = "VAELOR_MOE_KERNEL"
REGISTERED = "registered"
SERVED_EAGER = "served-eager"
SERVED_IN_CAPTURE = "served-in-capture"
STOOD_ASIDE = "stood-aside"
STOOD_ASIDE_IN_CAPTURE = "stood-aside-in-capture"
FAILED = "failed"
#: Every event the plugin writes.
_EVENTS = frozenset({REGISTERED, SERVED_EAGER, SERVED_IN_CAPTURE, STOOD_ASIDE,
                     STOOD_ASIDE_IN_CAPTURE, FAILED})
#: The tag that opens every line about the dense 4-bit kernels, so they are
#: read apart from the expert kernel's: a dense line never stands for the experts.
DENSE_MARK = "[dense]"
#: The same for the HIP d2 kernels (`vllm_moe_hip_build`), read apart again.
HIP_MARK = "[hip]"

NOT_LOADED = (
    "vLLM did not load it on this machine (none of its lines is in the server's "
    "log), so every expert call uses vLLM's own kernel."
)
NEVER_CAPTURED = (
    "it registered but was never recorded into a GPU graph during start-up, so "
    "decode steps, which replay graphs, use vLLM's own kernel."
)
#: The same two states for the dense 4-bit kernels (qkv and o), said apart
#: from the expert kernel's (review round 1).
DENSE_NOT_LOADED = (
    "they wrote no line to the server's log, so vLLM's own kernels serve the "
    "qkv and o layers."
)
#: The HIP kernels wrote nothing: start-up always runs expert calls in their
#: range, so silence means they were never tried.
HIP_NOT_REPORTED = (
    "they wrote no line to the server's log, so the Triton kernels serve every "
    "expert call they would have taken."
)
DENSE_NEVER_CAPTURED = (
    "they registered but no GPU graph recorded them during start-up, so "
    "writing steps use vLLM's slower qkv and o kernels."
)


def journal_note(journal: Any) -> str:
    """What the plugin's own lines in a started copy's log say went wrong, or ``""``.

    In order of what matters: a failure; a stand-aside at registration (the
    method was never wrapped); no line at all (the plugin was not loaded); a
    graph captured without it; registered but never recorded into a graph.
    A copy whose log shows it recorded into a graph and nothing worse is ``""``.
    The dense kernels' lines (:data:`DENSE_MARK`) are read apart, by the same
    rules: the expert kernel's state comes first, then the dense kernels',
    then the HIP d2 kernels' (:data:`HIP_MARK`: failed, stood aside - with the
    build's own reason - or silent).
    """
    events: Dict[str, str] = {}
    dense: Dict[str, str] = {}
    hip: Dict[str, str] = {}
    for line in str(journal or "").splitlines():
        at = line.find(MARKER + " ")
        event, colon, text = line[at + len(MARKER) + 1:].partition(": ")
        found = dense if text.startswith(DENSE_MARK + " ") else (
            hip if text.startswith(HIP_MARK + " ") else events)
        if at >= 0 and colon and event in _EVENTS and event not in found:
            found[event] = text.strip()
    for event in (FAILED, STOOD_ASIDE):
        if event in events:
            return events[event]
    if not events:
        return NOT_LOADED
    if STOOD_ASIDE_IN_CAPTURE in events:
        return events[STOOD_ASIDE_IN_CAPTURE]
    if SERVED_IN_CAPTURE not in events:
        return NEVER_CAPTURED
    for event in (FAILED, STOOD_ASIDE):
        if event in dense:
            return dense[event]
    if not dense:
        return DENSE_NOT_LOADED
    if STOOD_ASIDE_IN_CAPTURE in dense:
        return dense[STOOD_ASIDE_IN_CAPTURE]
    if SERVED_IN_CAPTURE not in dense:
        return DENSE_NEVER_CAPTURED
    # The HIP d2 kernels last: standing aside costs a few per cent on 1-12 tokens.
    for event in (FAILED, STOOD_ASIDE):
        if event in hip:
            return hip[event]
    if not hip:
        return HIP_NOT_REPORTED
    return ""


#: The sentence a copy's row carries for it (`gpu_pool_replicas`).
KERNEL_NOTE = "Vaelor's expert kernel on {machine}: {note}"
#: The same for the dense 4-bit kernels' state; the journal's tag is not shown.
DENSE_KERNEL_NOTE = "Vaelor's dense-layer kernels on {machine}: {note}"
#: The same for the HIP d2 kernels.
HIP_KERNEL_NOTE = "Vaelor's HIP expert kernels on {machine}: {note}"


def copy_note(note: str, machine: Any) -> str:
    """The row's sentence for a :func:`journal_note` result, or ``""`` for none."""
    if not note:
        return ""
    hip_tag = HIP_MARK + " "
    if note.startswith(hip_tag) or note == HIP_NOT_REPORTED:
        note = note[len(hip_tag):] if note.startswith(hip_tag) else note
        return HIP_KERNEL_NOTE.format(machine=machine, note=note.rstrip(". "))
    tag = DENSE_MARK + " "
    if note.startswith(tag) or note in (DENSE_NOT_LOADED, DENSE_NEVER_CAPTURED):
        note = note[len(tag):] if note.startswith(tag) else note
        return DENSE_KERNEL_NOTE.format(machine=machine, note=note.rstrip(". "))
    return KERNEL_NOTE.format(machine=machine, note=note.rstrip(". "))


def started_copy_note(runtime: Any, transport: Any, unit: str, machine: Any,
                      options: Any) -> str:
    """:data:`KERNEL_NOTE` for one started copy that was rendered WITH the kernel, else ``""``.

    The journal read is the reviewed one both transports admit; a read that
    fails says nothing, because the observer failing is not evidence about
    the kernel (LESSONS 8).
    """
    carried = options if isinstance(options, Mapping) else {}
    if carried.get(MOE_KERNEL_FIELD) is None:
        return ""
    try:
        journal = runtime.unit_journal(transport, unit)
    except Exception:  # an unreadable journal is no reading at all
        return ""
    if not isinstance(journal, str):
        return ""
    return copy_note(journal_note(journal), machine)
