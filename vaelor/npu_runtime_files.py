"""The NPU Assistant's files on disk: which serving pair a box holds, and fetching one (VD-219).

FastFlowLM's folder (`flm_service.FLM_MODEL_PATH`, ``/var/lib/vaelor/flm``) holds
the portable runtime at its top (``flm``, ``flm-real``, ``lib/``, ``xclbins/``,
FastFlowLM's model list) beside ``models/<folder>/``, the package it serves, and
``.vaelor-flm-version``. The folder itself never moves: the root bridge's unit
binds it into its own namespace (``ReadWritePaths=-/var/lib/vaelor/flm``), so a
folder renamed on the host would leave the bridge serving the old inode until it
restarts. A move between pairs therefore moves the folder's *entries*
(`npu_runtime_switch`), and this module keeps its own working folders inside it
(:data:`CONTROL_ENTRIES`), on the same filesystem, so every move is a rename.

Two callers:

* the installer, on a box with no FastFlowLM yet:
  ``python -I -m vaelor.npu_runtime_files install-runtime`` places the serving
  pair's runtime (`npu_runtime_pin.serving_pair`), and ``runtime-matches`` tells
  ``install_npu_model`` whether the package it would fetch fits the runtime there;
* `npu_runtime_switch`, which stages a whole pair here before moving to it.

A box that already holds a *different* runtime is never changed here: the
runtime and the package move together or not at all (LESSONS 22: a 1.0.9
runtime beside the 1.0.2 package serves nothing, and nothing would say why).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from urllib.request import urlopen

from .flm_npu_release import ReleaseInstallError, install as install_package
from .npu_runtime_pin import (
    LEGACY_PACKAGE_ID,
    PACKAGE_MARKER,
    Package,
    Runtime,
    ServingPair,
    pair_problems,
    release_source,
    serving_pair,
)

#: The runtime's version, written by whoever placed the runtime (the installer's
#: name for it since 1.4; a 1.5.3 box holds ``1.0.2`` here).
VERSION_MARKER = ".vaelor-flm-version"

#: Where a pair is downloaded and unpacked before a move (one folder per pair).
STAGE_DIR = ".vaelor-stage"
#: Where the pair a move replaced is kept, for the return path.
PREVIOUS_DIR = ".vaelor-previous"
#: The move's own record (`npu_runtime_switch`).
JOURNAL = ".vaelor-switch.json"
#: `flm_npu_release`'s scratch folder.
RELEASE_SCRATCH = "_release_staging"

#: Entries of FastFlowLM's folder that belong to Vaelor's bookkeeping, not to a
#: serving pair; a move never carries them.
CONTROL_ENTRIES = frozenset({STAGE_DIR, PREVIOUS_DIR, JOURNAL, RELEASE_SCRATCH})

#: Inside a staged pair's folder: the tree that becomes FastFlowLM's folder, and
#: the record written last, once every byte has been checked.
STAGED_TREE = "tree"
STAGED_RECORD = "STAGED.json"

#: The runtime tarball is ~48 MB; a cap keeps a hostile stream off the disk.
_MAX_RUNTIME_BYTES = 512 * 1024 * 1024
_FETCH_TIMEOUT_SECONDS = 120

#: What the extracted runtime must contain for the launch to work at all.
RUNTIME_REQUIRED = ("flm", "flm-real", "lib")


class RuntimeFilesError(Exception):
    """A runtime or package could not be fetched or checked - stated for the owner."""


def _opener(url: str) -> Any:
    return urlopen(url, timeout=_FETCH_TIMEOUT_SECONDS)


def _package_folder(folder: Optional[str]) -> str:
    return folder or serving_pair().package.folder


def payload_entries(root: Path, folder: Optional[str] = None) -> List[str]:
    """What a move carries, as paths relative to FastFlowLM's folder.

    The runtime's top-level entries (all but :data:`CONTROL_ENTRIES` and
    ``models``) and the one model folder the pair serves, ``models/<folder>``.
    Any other model folder stays where it is (R165 R2: a move never parks or
    deletes a model it did not install).
    """
    folder = _package_folder(folder)
    try:
        names = sorted(name for name in os.listdir(root)
                       if name not in CONTROL_ENTRIES and name != "models")
    except OSError:
        return []
    if (root / "models" / folder).exists():
        names.append("models/" + folder)
    return names


def installed_runtime(root: Path) -> str:
    """The runtime version a box holds: its marker beside a wrapper, else ``""``."""
    if not (root / "flm").exists():
        return ""
    try:
        return (root / VERSION_MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def installed_package(root: Path, folder: Optional[str] = None) -> str:
    """The package in the Assistant's model folder: its own marker, the 1.5.3 package by shape, or ``""``.

    Only the folder the pair serves is read (``Qwen3.5-4B-NPU2`` for
    ``qwen3.5:4b``); another model folder beside it is not the Assistant's
    package and changes nothing here (R165 R2).
    """
    folder = _package_folder(folder)
    path = root / "models" / folder
    if not path.is_dir():
        return ""
    try:
        recorded = json.loads((path / PACKAGE_MARKER).read_text(encoding="utf-8"))
        return str(recorded.get("package_id") or "unnamed")
    except FileNotFoundError:
        legacy = installed_runtime(root) == "1.0.2" and folder == "Qwen3.5-4B-NPU2"
        return LEGACY_PACKAGE_ID if legacy else "unnamed"
    except (OSError, ValueError, AttributeError):
        return "unnamed"


def installed_key(root: Path, folder: Optional[str] = None) -> str:
    """``<runtime>+<package>`` as on disk, comparable with :attr:`ServingPair.key`."""
    return "{}+{}".format(installed_runtime(root) or "none",
                          installed_package(root, folder) or "none")


# --- The runtime tarball ------------------------------------------------------------

def fetch_runtime(runtime: Runtime, destination: Path,
                  opener: Callable[[str], Any] = _opener) -> None:
    """Download the runtime tarball to ``destination`` and check its pinned SHA-256."""
    if not runtime.url.startswith("https://"):
        raise RuntimeFilesError("The FastFlowLM runtime must come over HTTPS.")
    digest = hashlib.sha256()
    total = 0
    try:
        with opener(runtime.url) as response, open(destination, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_RUNTIME_BYTES:
                    raise RuntimeFilesError("The FastFlowLM runtime download is larger than any release.")
                digest.update(chunk)
                handle.write(chunk)
    except OSError as error:
        raise RuntimeFilesError("The FastFlowLM {} runtime could not be downloaded ({}).".format(
            runtime.version, str(error)[:160] or type(error).__name__)) from error
    if digest.hexdigest() != runtime.sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeFilesError(
            "The FastFlowLM {} runtime failed its SHA-256 pin and was not used.".format(runtime.version))


def _link_stays_inside(name: PurePosixPath, target: PurePosixPath) -> bool:
    """Whether a relative link ``name -> target`` resolves, by its words alone, inside the tree."""
    where = [part for part in name.parent.parts if part not in ("", ".")]
    for part in target.parts:
        if part == "..":
            if not where:
                return False
            where.pop()
        elif part not in ("", "."):
            where.append(part)
    return True


def _check_runtime_members(members: Iterable[tarfile.TarInfo]) -> None:
    """Files, folders, and links that stay inside the tree; nothing else.

    The portable runtime carries relative symlinks (``lib/libxrt_core.so ->
    libxrt_core.so.2`` and the ``lib/x86_64-linux-gnu`` mirror), so a symlink
    is allowed when its target, read from its own folder, stays inside the
    tree. A hard link, a device, an absolute name or a ``..`` that leaves the
    tree is refused before anything is written.
    """
    members = list(members)
    links = {PurePosixPath(m.name).as_posix().strip("/") for m in members if m.issym()}

    def through_link(parts: List[str]) -> bool:
        # A prefix of the path that is itself a link member: resolving or writing
        # through it is where a chain of individually-inside links escapes (R165).
        return any("/".join(parts[:end]) in links for end in range(1, len(parts)))

    for member in members:
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise RuntimeFilesError("The runtime archive has an unsafe path: {}.".format(member.name))
        clean = [part for part in name.parts if part not in ("", ".")]
        if through_link(clean):
            raise RuntimeFilesError("The runtime archive writes through a link: {}.".format(member.name))
        if member.issym():
            target = PurePosixPath(member.linkname)
            walked = clean[:-1]
            chained = False
            for part in target.parts:
                if part in ("", "."):
                    continue
                if walked and "/".join(walked) in links:
                    chained = True  # the target is resolved through another link
                if part == "..":
                    if walked:
                        walked.pop()
                else:
                    walked.append(part)
            if target.is_absolute() or not _link_stays_inside(name, target) or chained:
                raise RuntimeFilesError(
                    "The runtime archive links outside itself: {}.".format(member.name))
        elif not (member.isfile() or member.isdir()):
            raise RuntimeFilesError(
                "The runtime archive has a member that is not a file: {}.".format(member.name))


def unpack_runtime(tarball: Path, tree: Path, runtime: Runtime) -> None:
    """Unpack the checked tarball into ``tree`` and write its version marker."""
    tree.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tarball) as archive:
            members = archive.getmembers()
            _check_runtime_members(members)
            try:
                archive.extractall(tree, filter="data")
            except TypeError:
                archive.extractall(tree)
    except (OSError, tarfile.TarError) as error:
        raise RuntimeFilesError("The runtime archive could not be unpacked ({}).".format(
            str(error)[:160] or type(error).__name__)) from error
    missing = [name for name in RUNTIME_REQUIRED if not (tree / name).exists()]
    if missing:
        raise RuntimeFilesError("The runtime archive has no {}.".format(", ".join(missing)))
    for name in ("flm", "flm-real"):
        os.chmod(tree / name, 0o755)
    (tree / VERSION_MARKER).write_text(runtime.version + "\n", encoding="utf-8")


def install_runtime(root: Path, runtime: Optional[Runtime] = None,
                    opener: Callable[[str], Any] = _opener) -> str:
    """Place the serving pair's runtime on a box that has none; one line saying what happened.

    A box already on this runtime is left alone. A box on another runtime is
    left alone too and says so: the move to a new runtime carries the package
    with it (`npu_runtime_switch`), and replacing the runtime alone is the half
    upgrade VD-217 forbids.
    """
    runtime = runtime or serving_pair().runtime
    have = installed_runtime(root)
    if have == runtime.version:
        return "FastFlowLM {} is already installed at {}.".format(runtime.version, root)
    if have:
        return ("FastFlowLM {} is installed at {}; the move to {} is made with its model "
                "by the post-install step (vaelor-refresh-models --apply).").format(
                    have, root, runtime.version)
    if (root / "flm").exists() and installed_package(root):
        # A wrapper and a model but no version marker: not this installer's work,
        # and replacing the runtime under that model could be the half upgrade.
        return ("FastFlowLM of an unknown version is installed at {} beside a model; remove "
                "{} and run the installer again to place {}.").format(root, root, runtime.version)
    root.mkdir(parents=True, exist_ok=True)
    work = root / STAGE_DIR / "runtime-{}".format(runtime.version)
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        tarball = work / "runtime.tar.gz"
        fetch_runtime(runtime, tarball, opener)
        unpack_runtime(tarball, work / STAGED_TREE, runtime)
        # No marker means nothing here is a finished runtime: entries an
        # interrupted install left are replaced, and the marker is placed LAST,
        # so an install cut off at any point never reads as installed (R165 R4).
        names = [name for name in sorted(os.listdir(work / STAGED_TREE))
                 if name not in ("models", VERSION_MARKER)]
        (root / VERSION_MARKER).unlink(missing_ok=True)
        for name in names + [VERSION_MARKER]:
            destination = root / name
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            elif destination.exists() or destination.is_symlink():
                destination.unlink()
            os.rename(work / STAGED_TREE / name, destination)
        (root / "models").mkdir(mode=0o755, exist_ok=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        try:
            (root / STAGE_DIR).rmdir()
        except OSError:
            pass
    return "FastFlowLM {} installed at {}, verified against its pinned SHA-256.".format(
        runtime.version, root)


# --- A whole pair, staged for a move ------------------------------------------------

def staged_dir(root: Path, pair: ServingPair) -> Path:
    return root / STAGE_DIR / pair.key


def _version(text: Any) -> tuple:
    try:
        return tuple(int(part) for part in str(text).strip().split("."))
    except ValueError:
        return ()


def runtime_refuses(tree: Path, package: Package) -> str:
    """Why the runtime in ``tree`` would not serve the package as it is; ``""`` when it would.

    FastFlowLM compares the package's ``config.json`` ``flm_version`` with its own
    model list's ``flm_min_version`` for the tag. Below it, FastFlowLM 1.0.9 does
    not refuse: it deletes the files it calls outdated and downloads its stock
    model over them (VD-219 window 2), so a mis-built package would be served as
    somebody else's model. Read before a move so that never starts.
    """
    family, _, size = package.tag.partition(":")
    try:
        listed = json.loads((tree / "model_list.json").read_text(encoding="utf-8"))
        entry = listed["models"][family][size]
        config = json.loads((tree / "models" / package.folder / "config.json").read_text(
            encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError) as error:
        return "the runtime's model list or the package's config could not be read ({})".format(
            str(error)[:120] or type(error).__name__)
    needed, has = _version(entry.get("flm_min_version") or "0"), _version(config.get("flm_version"))
    if not has or has < needed:
        return ("the package says FastFlowLM {}, and this runtime needs at least {} for {}; it "
                "would replace the package with its stock model").format(
                    config.get("flm_version"), entry.get("flm_min_version"), package.tag)
    return ""


def start_refusal(root: Path, tag: str) -> str:
    """Why FastFlowLM in ``root`` must not be started on ``tag`` now; ``""`` when it may.

    Read at every start, not only at a move (R165): FastFlowLM swaps in its stock
    model at whatever start finds an installed folder below the version its
    model list needs (kb G152). Only a folder that is on disk is judged - a tag
    with no folder is a fresh pull, which replaces nothing of ours. Two small
    JSON reads; an unreadable model list says nothing, as before this check.
    """
    family, _, size = str(tag).partition(":")
    try:
        listed = json.loads((root / "model_list.json").read_text(encoding="utf-8"))
        folder = str(listed["models"][family][size]["name"])
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    config = root / "models" / folder / "config.json"
    if "/" in folder or not config.is_file():
        return ""
    if folder != serving_pair().package.folder:
        # Another model's folder (R172 item 4): only a version it states is held
        # against the list; a config that states none is FastFlowLM's business.
        try:
            stated = json.loads(config.read_text(encoding="utf-8")).get("flm_version")
        except (OSError, ValueError, AttributeError):
            stated = None
        if not stated:
            return ""
    probe = Package("", folder, str(tag), "", "", 0, "")
    return runtime_refuses(root, probe)


def staged_ready(root: Path, pair: ServingPair) -> bool:
    """Whether ``pair`` is fully staged: its record is written last, after every check."""
    try:
        record = json.loads((staged_dir(root, pair) / STAGED_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    tree = staged_dir(root, pair) / STAGED_TREE
    return (isinstance(record, dict) and record.get("key") == pair.key
            and installed_key(tree, pair.package.folder) == pair.key
            and not package_problems(tree / "models" / pair.package.folder, pair.package))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_problems(folder: Path, package: Package) -> List[str]:
    """Why ``folder`` is not exactly ``package``; empty when every file matches its manifest.

    The manifest is the ``files`` map in the folder's :data:`PACKAGE_MARKER`
    (name -> sha256 and size), written by ``tools/npu_model_package.py`` and
    covered by the package's pinned SHA-256 when it was unpacked. Read again
    before a move and after the moved pair has answered (VD-219 window 2: a
    staged tree holding another conversion's weights was "ready" by its marker
    alone, and FastFlowLM then replaced those files with its stock model).
    """
    try:
        marker = json.loads((folder / PACKAGE_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["it has no readable package marker"]
    if not isinstance(marker, dict) or marker.get("package_id") != package.package_id:
        return ["its marker does not name {}".format(package.package_id)]
    manifest = marker.get("files")
    if not isinstance(manifest, dict) or not manifest:
        return ["its marker lists no files to check"]
    problems = []
    for name, expected in sorted(manifest.items()):
        path = folder / str(name)
        if "/" in str(name) or not isinstance(expected, dict):
            problems.append("{}: not a plain manifest entry".format(name))
            continue
        try:
            size = path.stat().st_size
            if size != expected.get("size"):
                problems.append("{}: {} bytes, not {}".format(name, size, expected.get("size")))
            elif _file_sha256(path) != expected.get("sha256"):
                problems.append("{}: its SHA-256 is not the package's".format(name))
        except OSError:
            problems.append("{}: missing".format(name))
    return problems


def stage_pair(root: Path, pair: ServingPair, *, opener: Callable[[str], Any] = _opener,
               package_opener: Optional[Callable[[str], Any]] = None,
               source: Optional[str] = None) -> str:
    """Download and check ``pair`` into its stage folder; a pair already staged is kept.

    The runtime is checked against its SHA-256, the package against its own
    (`flm_npu_release`), and the package folder must name the package it was
    pinned as - so a stage that holds anything else is never moved to.
    """
    problems = pair_problems(pair)
    if problems:
        raise RuntimeFilesError("The pinned pair cannot be installed: {}.".format("; ".join(problems)))
    if staged_ready(root, pair):
        return "already staged"
    for leftover in (root / STAGE_DIR).glob("*") if (root / STAGE_DIR).is_dir() else ():
        if leftover.name != pair.key:
            shutil.rmtree(leftover, ignore_errors=True)
    folder = staged_dir(root, pair)
    shutil.rmtree(folder, ignore_errors=True)
    tree = folder / STAGED_TREE
    tree.mkdir(parents=True)
    try:
        tarball = folder / "runtime.tar.gz"
        fetch_runtime(pair.runtime, tarball, opener)
        unpack_runtime(tarball, tree, pair.runtime)
        tarball.unlink(missing_ok=True)
        try:
            installed = install_package(
                source or release_source(), tree,
                expected_sha256=pair.package.sha256, part_prefix=pair.package.part_prefix,
                **({"opener": package_opener} if package_opener is not None else {}))
        except ReleaseInstallError as error:
            raise RuntimeFilesError(str(error)) from error
        if installed.get("model") != pair.package.folder or installed_package(tree, pair.package.folder) != pair.package.package_id:
            raise RuntimeFilesError(
                "The downloaded package is not {} in {}; it was not staged.".format(
                    pair.package.package_id, pair.package.folder))
        problems = package_problems(tree / "models" / pair.package.folder, pair.package)
        if problems:
            raise RuntimeFilesError("The downloaded package does not match its own manifest ({}); "
                                    "it was not staged.".format("; ".join(problems[:3])))
        record = {"key": pair.key, "runtime": pair.runtime.version,
                  "package": pair.package.package_id, "sha256": pair.package.sha256}
        (folder / STAGED_RECORD).write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    except BaseException:
        # A stage that did not finish is not left on disk (R165 R2): it is either
        # complete and recorded, or gone.
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return "staged"


def main(argv: Sequence[str]) -> int:
    """``install-runtime`` | ``runtime-matches`` | ``installed``, on the live folder, as the installer runs them."""
    from .flm_service import FLM_MODEL_PATH

    root = Path(FLM_MODEL_PATH)
    verb = list(argv[1:])
    if verb == ["installed"]:
        print(installed_key(root))
        return 0
    if verb == ["runtime-matches"]:
        return 0 if installed_runtime(root) == serving_pair().runtime.version else 1
    if verb == ["install-runtime"]:
        try:
            print(install_runtime(root))
            return 0
        except (RuntimeFilesError, OSError) as error:
            print("FastFlowLM was not installed: {}".format(str(error)[:300]))
            return 1
    sys.stderr.write("usage: python -I -m vaelor.npu_runtime_files "
                     "install-runtime|runtime-matches|installed\n")
    return 2


def describe(root: Path) -> Dict[str, Any]:
    """The folder's state, for a status line: what is installed and what is staged."""
    stage = root / STAGE_DIR
    return {
        "installed": installed_key(root),
        "staged": sorted(p.name for p in stage.iterdir()) if stage.is_dir() else [],
        "previous": installed_key(root / PREVIOUS_DIR) if (root / PREVIOUS_DIR).is_dir() else "",
    }


if __name__ == "__main__":
    sys.exit(main(sys.argv))
