"""Build Vaelor's HIP d2 expert kernels on the machine that serves them, as root (VD-216).

**What is built.** ``vllm_moe_plugin/vaelor_moe_d2.hip`` - Vaelor's own HIP
source, pinned by sha256 (:data:`SOURCE_SHA256`) - compiled into one GPU code
object for gfx1151 (:data:`CODE_OBJECT`). No host code, no torch extension:
the plugin loads the object's bytes with ``hipModuleLoadData``
(`vllm_moe_plugin/vaelor_moe_d2_hip.py`).

**Where the compiler is.** Neither box's host ROCm has one (the Z2's is
runtime-only); the pinned vLLM 0.27 image does, at :data:`HIPCC` (verified
2026-10-14 on the Z2 against the pinned digest). So the build is ONE fixed
``docker run`` of that image (:func:`build_argv`): ``--pull never`` (it never
downloads), no network, no GPU device, a read-only root filesystem, every
capability dropped, bounded memory, pids and time; the checked source file
mounted read-only and a fresh root-only output folder (:data:`BUILD_DIR`,
``0700``) the only thing it can write. Nothing of the serving container is
shared with it. It takes about a second.

**The hash chain** (LESSONS 18, #178 - the check and the use must see the
same bytes):

1. the source is read no-follow out of the root-only plugin copy and must
   carry :data:`SOURCE_SHA256` before it is compiled;
2. the code object is read no-follow, root's alone, single-name, through the
   descriptor its checks were taken from (`vllm_overlays.root_controlled_bytes`),
   sanity-checked (:func:`code_refusal`), and THOSE bytes are hashed and written
   atomically into the plugin folder - never the build's file itself;
3. the record (:data:`RECORD`: ``ready``, ``reason``, the source's and the
   object's sha256) is written beside it, and the copy read back;
4. before every mount the root bridge re-checks object against record
   (:func:`ensure_on_controller`); inside the server the plugin hashes the
   bytes it read and loads those same bytes.

**Every failure is a named record, never a crash**: the record says
``ready: false`` and why, the code object is removed, and the plugin serves
with the Triton d2 kernels exactly as before, saying so in the server's log
(``[hip]`` lines) and so on the deployment's row. A build failure never fails
an install, a start or a deploy.

**Where it runs.** The installer runs ``build`` as root
(``python -m vaelor.vllm_moe_hip_build build``) - effective when the vLLM
image is already on the machine; the root bridge builds at the first start
that mounts the plugin when no current record is there (the image is pulled
by then). A worker is built by the controller over SSH, on the worker, with
the worker's own image (:func:`ensure_on_worker`): each machine builds its own.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from typing import Any, Callable, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from .vllm_overlays import OVERLAY_ROOT, root_controlled_bytes

#: The source, the code object and the record, in the plugin folder (the
#: plugin's `vaelor_moe_d2_hip` holds the same names; a test ties them).
HIP_SOURCE = "vaelor_moe_d2.hip"
CODE_OBJECT = "vaelor_moe_d2_hip.co"
RECORD = "vaelor_moe_d2_hip.json"

#: sha256 of :data:`HIP_SOURCE`, the one source this release compiles (a test
#: ties it to the file and to the plugin's own pin).
SOURCE_SHA256 = "032ebc7ee3dcb67419e9ed1ea7bd2e0dce924491c8586653c9d8e0ce15115bde"

#: The compiler inside the pinned 0.27 image (``_rocm_sdk_devel``), and the GPU.
HIPCC = "/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/bin/hipcc"
OFFLOAD_ARCH = "gfx1151"

#: The root-only output folder, beside the plugin folder (never inside it).
#: On a worker each build gets its own (:func:`ensure_on_worker`).
BUILD_DIR = "hip-build"
#: The lock a controller build holds, beside it: the installer and the root
#: bridge both build as root on the controller and must not share the folder.
BUILD_LOCK = "hip-build.lock"
#: Every build container's name starts with this; each run adds its own token,
#: so a cleanup removes exactly the container that run started.
CONTAINER_PREFIX = "vaelor-hip-build-"

#: The bound on one build from the installer or on a worker (measured: about a
#: second on the Z2). The root bridge gives it less (`bridge_managed_units`).
BUILD_SECONDS = 180

#: Largest code object accepted (the gfx1151 build is about 40 KB), and the
#: most the no-follow reader takes (`vllm_overlays.root_controlled_bytes`).
CODE_LIMIT = 1 << 20
#: What a code object starts with: an ELF, or clang's offload bundle around one.
CODE_MAGICS = (b"\x7fELF", b"__CLANG_OFFLOAD_BUNDLE__")

IMAGE_ABSENT = (
    "the vLLM 0.27 image is not on this machine yet; the root bridge builds the "
    "kernels at the first start of the model"
)
NOT_BUILT = "not built on this machine"
SOURCE_NOT_PINNED = "the root copy of the HIP source does not carry the pinned checksum"
BUILD_FAILED = "the compiler in the vLLM image did not build the kernels ({})"
#: Why a start skipped the build (the root bridge's shared pre-start budget ran out).
BUILD_DEFERRED = (
    "this start had no time left for the build after the other pre-start checks; "
    "the next start builds it"
)
BUILD_BUSY = (
    "another build of these kernels is running on this machine (the installer or "
    "the bridge); the next start uses its result"
)
NO_OUTPUT = "the compiler left no code object, or one not root's alone"
COPY_FAILED = "the built code object did not arrive intact in the kernel folder"
RECORD_MISMATCH = (
    "the code object in the kernel folder does not match its build record; it was "
    "removed and is rebuilt at the next start"
)


def compiler_args(source: str, output: str) -> List[str]:
    """The compiler's own arguments: one gfx1151 code object, no host code.
    The GPU test inside the image builds with exactly these."""
    return ["--genco", "--offload-arch=" + OFFLOAD_ARCH, "-O3", "-o", output, source]


def container_name() -> str:
    """A build container's own name: :data:`CONTAINER_PREFIX`, then twelve hex digits.

    The first six are this process group's (`post_install_steps.group_tag`), so
    an update step that is stopped removes its own containers and nobody
    else's (VD-217, R101); the last six are fresh.
    """
    from .post_install_steps import group_tag

    return CONTAINER_PREFIX + group_tag() + secrets.token_hex(3)


def build_argv(image: str, source_path: str, out_dir: str, name: str) -> List[str]:
    """The one fixed build command (module docstring): source in, code object out.
    ``name`` (:func:`container_name`) is what :func:`remove_argv` stops it by."""
    return [
        "docker", "run", "--rm", "--name", name, "--pull", "never", "--network", "none",
        "--read-only", "--tmpfs", "/tmp:rw,size=256m", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--memory", "4g", "--pids-limit", "256",
        "-v", "{}:/src/{}:ro".format(source_path, HIP_SOURCE),
        "-v", "{}:/out".format(out_dir),
        "--entrypoint", HIPCC, image,
    ] + compiler_args("/src/" + HIP_SOURCE, "/out/" + CODE_OBJECT)


def remove_argv(name: str) -> List[str]:
    """Stop and remove a build container by its name. Run after every failed or
    timed-out build: a runner's timeout kills the ``docker`` client, and the
    container it started would otherwise keep running."""
    return ["docker", "rm", "-f", name]


def code_refusal(data: Optional[bytes]) -> Optional[str]:
    """Why ``data`` is not a code object this accepts, or ``None``."""
    if not data or len(data) > CODE_LIMIT or not data.startswith(CODE_MAGICS):
        return NO_OUTPUT
    return None


def record_text(key: str, built_sha256: Optional[str], reason: Optional[str]) -> str:
    """The build record: ready with the object's sha256, or not ready and why."""
    body = {
        "key": key, "ready": reason is None, "reason": reason or "",
        "source_sha256": SOURCE_SHA256, "built_sha256": built_sha256 or "",
    }
    return json.dumps(body, sort_keys=True) + "\n"


def parse_record(raw: Any) -> Optional[Mapping[str, Any]]:
    """A record's fields, or ``None`` when it is absent or unreadable."""
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        value = json.loads(text) if raw is not None else None
    except (UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def is_current(record: Optional[Mapping[str, Any]], code_sha256: str) -> bool:
    """Whether a record and the code object beside it are this release's good build."""
    return bool(
        record and record.get("ready") is True
        and record.get("source_sha256") == SOURCE_SHA256
        and code_sha256 and record.get("built_sha256") == code_sha256
    )


def _sha(data: Optional[bytes]) -> str:
    return "" if data is None else hashlib.sha256(data).hexdigest()


def _brief(error: Any) -> str:
    text = " ".join(str(error).split())
    if text == BUILD_DEFERRED:
        return text
    if "No such image" in text or "Unable to find image" in text:
        return IMAGE_ABSENT
    return BUILD_FAILED.format(text[:200] or type(error).__name__)


# --- The controller (the root bridge and the installer, as root) ------------------

def _write_bytes(parent: int, name: str, data: bytes, mode: int = 0o644) -> None:
    """Replace ``name`` in ``parent`` with ``data`` atomically, following no link
    (`bridge_safe_fs.write_file`, for bytes rather than text)."""
    temporary = ".{}.{}".format(name, secrets.token_hex(6))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(
        os, "O_CLOEXEC", 0)
    handle = os.open(temporary, flags, mode, dir_fd=parent)
    try:
        try:
            os.fchmod(handle, mode)
            written = 0
            while written < len(data):
                written += os.write(handle, data[written:])
            os.fsync(handle)
        finally:
            os.close(handle)
        os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=parent)
        except OSError:
            pass
        raise


def _settle(plugin_path: str, key: str, code: Optional[bytes], reason: Optional[str],
            owner: Tuple[int, int]) -> None:
    """Write the code object (or remove it) and the record, in the plugin folder."""
    from . import bridge_safe_fs as safe_fs

    with safe_fs.directory(plugin_path) as plugin:
        if code is None:
            safe_fs.remove_file(plugin, CODE_OBJECT)
        else:
            _write_bytes(plugin, CODE_OBJECT, code)
        safe_fs.write_file(plugin, RECORD, record_text(key, _sha(code) or None, reason))


def ensure_on_controller(
    kernel: Any, plugin_path: str, *, build: Optional[Callable[[List[str]], Any]],
    owners: FrozenSet[int] = frozenset({0}), owner: Tuple[int, int] = (0, 0),
    locked: bool = False,
) -> Optional[str]:
    """Make the plugin folder's HIP build current; ``None`` when it is, else why not.

    ``plugin_path`` is the checked root-only copy (`vllm_moe_kernels.
    root_kernel_dir`). A current record and object are kept. Otherwise the
    object is removed and, when ``build`` (one fixed command run as root,
    raising with its own words) is given, the source is compiled again; any
    failure is recorded and returned. Never raises: the plugin then serves on
    Triton and says so. On the controller a build holds :data:`BUILD_LOCK`;
    a second build finding it held waits for nothing and says so
    (:data:`BUILD_BUSY`), touching nothing the first is writing. ``locked``
    says the caller already holds it (`vllm_moe_kernels.root_kernel_dir`,
    which holds it across the copy, the build and the folder check).
    """
    record = parse_record(root_controlled_bytes(plugin_path + "/" + RECORD, owners=owners))
    code = root_controlled_bytes(plugin_path + "/" + CODE_OBJECT, owners=owners)
    if is_current(record, _sha(code)):
        return None
    if build is None:
        if record is not None and record.get("ready") is not True:
            return str(record.get("reason") or NOT_BUILT)
        if record is None and code is None:
            return NOT_BUILT
        try:
            _settle(plugin_path, kernel.key, None, RECORD_MISMATCH, owner)
        except (OSError, ValueError):
            pass
        return RECORD_MISMATCH
    keyed, _ = plugin_path.rsplit("/", 1)
    if locked:
        return _build_locked(kernel, plugin_path, keyed, build, owners, owner)
    try:
        lock = _lock(keyed)
    except OSError as error:
        return "the build lock could not be taken ({})".format(str(error)[:120])
    if lock is None:
        return BUILD_BUSY
    try:
        # Another build may have finished while this one was deciding.
        record = parse_record(root_controlled_bytes(plugin_path + "/" + RECORD, owners=owners))
        code = root_controlled_bytes(plugin_path + "/" + CODE_OBJECT, owners=owners)
        if is_current(record, _sha(code)):
            return None
        return _build_locked(kernel, plugin_path, keyed, build, owners, owner)
    finally:
        os.close(lock)


def _lock(keyed: str, wait_seconds: float = 0.0) -> Optional[int]:
    """The controller's build lock, or ``None`` when another holds it for
    longer than ``wait_seconds`` (polled, never an unbounded wait). The file
    is opened no-follow in the root-only key folder and must be a regular
    file; the lock goes with the descriptor, so a crashed holder never leaves
    it held."""
    import fcntl
    import stat as stat_module
    import time

    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    handle = os.open(keyed + "/" + BUILD_LOCK, flags | getattr(os, "O_NONBLOCK", 0), 0o600)
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    try:
        if not stat_module.S_ISREG(os.fstat(handle).st_mode):
            raise OSError("the build lock is not a regular file")
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(handle)
                    return None
                time.sleep(0.05)
    except BaseException:
        try:
            os.close(handle)
        except OSError:
            pass
        raise


def _build_locked(kernel: Any, plugin_path: str, keyed: str,
                  build: Callable[[List[str]], Any], owners: FrozenSet[int],
                  owner: Tuple[int, int]) -> Optional[str]:
    """The build itself, under the lock (:func:`ensure_on_controller`)."""
    from . import bridge_safe_fs as safe_fs

    out_dir = keyed + "/" + BUILD_DIR
    name = container_name()
    built: Optional[bytes] = None
    reason: Optional[str] = None
    try:
        source = root_controlled_bytes(plugin_path + "/" + HIP_SOURCE, owners=owners)
        if _sha(source) != SOURCE_SHA256:
            raise ValueError(SOURCE_NOT_PINNED)
        with safe_fs.directory(keyed) as parent:
            safe_fs.remove_tree(parent, BUILD_DIR)
            os.close(safe_fs.ensure_root_directory(parent, BUILD_DIR, mode=0o700, owner=owner))
        try:
            build(build_argv(kernel.image, plugin_path + "/" + HIP_SOURCE, out_dir, name))
        except Exception as error:  # the runner's own words, shortened
            # A deferred build started no container, and the budget it was
            # deferred for has no time for a removal either (R49).
            if str(error) != BUILD_DEFERRED:
                try:
                    build(remove_argv(name))
                except Exception:  # nothing left to remove is the usual answer
                    pass
            raise ValueError(_brief(error)) from error
        # No-follow, root's alone, single-name: a link the compiler left is
        # refused here, whatever it points at (#178).
        built = root_controlled_bytes(out_dir + "/" + CODE_OBJECT, owners=owners)
        refused = code_refusal(built)
        if refused:
            raise ValueError(refused)
        _settle(plugin_path, kernel.key, built, None, owner)
        landed = root_controlled_bytes(plugin_path + "/" + CODE_OBJECT, owners=owners)
        if _sha(landed) != _sha(built):
            raise ValueError(COPY_FAILED)
    except (OSError, ValueError) as error:
        reason = str(error)[:300] or type(error).__name__
        try:
            _settle(plugin_path, kernel.key, None, reason, owner)
        except (OSError, ValueError):
            pass
    finally:
        try:
            with safe_fs.directory(keyed) as parent:
                safe_fs.remove_tree(parent, BUILD_DIR)
        except (OSError, ValueError):
            pass
    return reason


# --- A worker (built there, by the controller, over SSH as root) -------------------

def ensure_on_worker(transport: Any, kernel: Any, target: str) -> Optional[str]:
    """The same on a WORKER: built with ITS image, ``None`` when current, else why not.

    ``target`` is the worker's plugin folder, whose files - the source among
    them - the caller has just read back by sha256. The compiler's output is
    taken only when ``stat`` (run after the build container has exited, so
    nothing writes in the root-only folder any more) says it is a single-name
    regular file of root's; it is copied with ``install`` and read back by
    sha256 before the record names it. Never raises.
    """
    from .ssh_transport import SshTransportError

    def run(argv: Sequence[str], **options: Any) -> str:
        return str(transport.run(list(argv), sudo=True, **options))

    def sha(path: str) -> str:
        try:
            return (run(["sha256sum", path]).split() or [""])[0]
        except SshTransportError:
            return ""

    try:
        record = parse_record(run(["cat", target + "/" + RECORD]))
    except SshTransportError:
        record = None
    if is_current(record, sha(target + "/" + CODE_OBJECT)):
        return None
    keyed = target.rsplit("/", 1)[0]
    name = container_name()
    # Its own folder per build: two deploys reaching one worker never share one.
    out_dir = keyed + "/" + BUILD_DIR + "-" + name[len(CONTAINER_PREFIX):]
    built = ""
    reason: Optional[str] = None
    try:
        run(["rm", "-rf", out_dir])
        run(["install", "-d", "-m", "0700", out_dir])
        try:
            run(build_argv(kernel.image, target + "/" + HIP_SOURCE, out_dir, name),
                timeout=BUILD_SECONDS)
        except SshTransportError as error:
            try:
                run(remove_argv(name))
            except SshTransportError:
                pass
            raise ValueError(_brief(error)) from error
        shape = run(["stat", "-c", "%F|%u|%h|%s", out_dir + "/" + CODE_OBJECT]).strip()
        kind, uid, links, size = (shape.split("|") + ["", "", "", ""])[:4]
        if (kind, uid, links) != ("regular file", "0", "1") or not 0 < int(size or 0) <= CODE_LIMIT:
            raise ValueError(NO_OUTPUT)
        built = sha(out_dir + "/" + CODE_OBJECT)
        run(["install", "-m", "0644", out_dir + "/" + CODE_OBJECT, target + "/" + CODE_OBJECT])
        if not built or sha(target + "/" + CODE_OBJECT) != built:
            raise ValueError(COPY_FAILED)
    except (SshTransportError, ValueError) as error:
        reason = str(error)[:300] or type(error).__name__
        built = ""
    finally:
        try:
            run(["rm", "-rf", out_dir])
        except SshTransportError:
            pass
    try:
        if reason is not None:
            run(["rm", "-f", target + "/" + CODE_OBJECT])
        text = record_text(kernel.key, built or None, reason)
        from .vllm_entry_program import write_program_on_worker

        write_program_on_worker(transport, target + "/" + RECORD, text,
                                hashlib.sha256(text.encode("utf-8")).hexdigest())
    except SshTransportError as error:
        reason = reason or "the build record could not be written there ({})".format(
            str(error)[:160])
    return reason


# --- The installer -------------------------------------------------------------

def _run_fixed(argv: List[str], timeout: float = BUILD_SECONDS) -> str:
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                               check=False)
    if completed.returncode != 0:
        raise RuntimeError(str(completed.stderr or "").strip()[:400] or "the build failed")
    return str(completed.stdout or "")


def main(argv: Sequence[str], *, root: str = OVERLAY_ROOT,
         build: Optional[Callable[[List[str]], Any]] = None) -> int:
    """``build [--within N]``, as root, from the installer: the checked plugin copy,
    then its HIP build, for every kernel row. Every outcome is printed; never fails
    the install. ``--within`` (VD-217, R98) keeps the lock wait and the build
    inside the update's share, so a timed-out build's container is removed here."""
    from .post_install_steps import step_timeout, within_seconds
    from .vllm_moe_kernels import MOE_KERNELS, KernelUnavailable, root_kernel_dir

    within = within_seconds(argv, ("build",))
    if within is None:
        sys.stderr.write("usage: python -m vaelor.vllm_moe_hip_build build [--within SECONDS]\n")
        return 2
    deadline = time.monotonic() + min(within, 2.0 * BUILD_SECONDS)
    build = build or (lambda command: _run_fixed(
        command, step_timeout(command, deadline, BUILD_SECONDS)))
    for kernel in MOE_KERNELS.values():
        try:
            root_kernel_dir(kernel, root=root, build=build, lock_wait=min(
                BUILD_SECONDS, max(0.0, deadline - time.monotonic())))
            from .vllm_moe_kernels import installed_dir

            record = parse_record(root_controlled_bytes(installed_dir(kernel, root) + "/" + RECORD))
            outcome = "done" if record and record.get("ready") else (
                (record or {}).get("reason") or "not built")
        except KernelUnavailable as error:
            outcome = str(error)[:300]
        except Exception as error:  # an installer step never fails the install
            outcome = str(error)[:300] or type(error).__name__
        print("Vaelor HIP expert kernels build ({}): {}".format(kernel.key, outcome))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
