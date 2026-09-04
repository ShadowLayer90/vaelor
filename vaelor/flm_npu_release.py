"""Install the on-device NPU assistant model from its pinned GitHub release.

The stock ``lemonade-server`` snap carries only public flm models, so the
appliance's fine-tuned NPU model (Qwen3.5-4B, served on the neural processor)
cannot arrive through it. It is delivered as a **GitHub release**: the model tar
is ~3.35 GB, larger than a single release asset (capped at 2 GB), so it ships as
split ``<name>.tar.partNN`` parts. This module lists the release's assets,
downloads those parts, concatenates them in name order into one tar, verifies
that tar against a **pinned SHA-256** (the trust anchor is the catalog shipped in
the wheel, never the release JSON), and unpacks the model into the directory
FastFlowLM serves from.

This is the exact same artifact and mechanism as ``deploy/fetch-npu-model.sh``
(the installer's first-boot fetch); this module is the in-appliance "Install"
action for the same model, so the two agree byte for byte. The FLM **runtime**
(``flm``/``flm-real`` + its libraries) is provisioned separately by the
installer, so no runtime bundle is expected in this artifact — only the model.

Layout the install produces, mirroring what ``flm serve <tag>`` reads
(``$FLM_MODEL_PATH/models/``)::

    <flm_home>/models/<model_dir>/    the q4nx model + tokenizer

The release JSON is small; the parts are the large payload. Both are fetched
through the injected ``opener`` (``urllib`` by default) so a test drives the
whole path against an in-memory or ``file://`` source with no network.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any, Callable, Dict, List
from urllib.parse import urlsplit
from urllib.request import urlopen

#: The split-part naming the release publishes for the NPU model. A downloaded
#: asset is used only if its name matches this exactly, so a hostile or stray
#: release asset cannot be pulled in as a "part". ``partNN`` is captured so the
#: parts reassemble in NUMERIC order (``part10`` after ``part9``), not the lexical
#: order a plain string sort would give.
_MODEL_PART_PREFIX = "qwen35-4b-npu2.tar.part"
_SAFE_PART_NAME = re.compile(r"^qwen35-4b-npu2\.tar\.part([0-9]{2,})$")

#: A hard ceiling on the reassembled model, so a hostile release JSON pointing a
#: part URL at an unbounded stream cannot exhaust the disk before the sha check.
#: The real artifact is ~3.35 GB; 6 GiB leaves headroom without being unbounded.
_MAX_MODEL_BYTES = 6 * 1024 * 1024 * 1024

#: The release index is small JSON; cap its read so a hostile or MITM'd source
#: cannot OOM the ROOT bridge with an unbounded body before any parsing.
_MAX_INDEX_BYTES = 4 * 1024 * 1024

#: A socket-level timeout on every fetch. Without it a stalled connection would
#: hang the install indefinitely while the bridge holds the flm start/stop lock,
#: so the NPU assistant could neither start nor stop until the TCP stack gave up.
_FETCH_TIMEOUT_SECONDS = 120

#: Hosts a model part may be served from. The default source is GitHub over TLS,
#: whose release assets live on ``github.com`` and redirect to
#: ``*.githubusercontent.com``; the index's own host is also allowed so a mirror
#: (``VAELOR_NPU_RELEASE_SOURCE``) that serves both index and assets works. This
#: closes the "index body names an arbitrary internal host" SSRF at the root
#: bridge, on top of the sha256 pin that already gates the CONTENT.
_ALLOWED_ASSET_HOSTS = ("github.com",)
_ALLOWED_ASSET_HOST_SUFFIX = ".githubusercontent.com"


class ReleaseInstallError(Exception):
    """A release could not be installed - stated so the UI can show why."""


def _default_opener(url: str) -> Any:
    return urlopen(url, timeout=_FETCH_TIMEOUT_SECONDS)


def _require_https(url: str, what: str) -> None:
    if urlsplit(url).scheme != "https":
        raise ReleaseInstallError(
            "The {} must be served over HTTPS; refusing it.".format(what)
        )


def _asset_host_allowed(url: str, source_host: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    if host in _ALLOWED_ASSET_HOSTS or host.endswith(_ALLOWED_ASSET_HOST_SUFFIX):
        return True
    return bool(source_host) and host == source_host


def _read_index(opener: Callable[[str], Any], url: str) -> bytes:
    """Read the (small) release index, capped so it cannot exhaust memory."""
    try:
        with opener(url) as response:
            data = response.read(_MAX_INDEX_BYTES + 1)
    except OSError as error:
        raise ReleaseInstallError("Could not reach the release source.") from error
    if len(data) > _MAX_INDEX_BYTES:
        raise ReleaseInstallError(
            "The release index is larger than expected; refusing it."
        )
    return data


def _release_part_urls(opener: Callable[[str], Any], release_url: str) -> List[str]:
    """The model-part download URLs from a GitHub release, in numeric part order.

    Reads the release JSON (``.../releases/tags/<tag>``), keeps only assets whose
    name is an exact model part, orders them by their numeric part index so
    ``part00`` precedes ``part01`` (and ``part9`` precedes ``part10``) regardless
    of JSON order, and returns their download URLs. Refuses a release with no
    parts, an index or part URL that is not HTTPS, a part on a host outside the
    allow-list, and any name that does not match the pinned part pattern.
    """
    _require_https(release_url, "release index URL")
    source_host = (urlsplit(release_url).hostname or "").lower()
    try:
        release = json.loads(_read_index(opener, release_url))
    except (ValueError, TypeError, RecursionError) as error:
        raise ReleaseInstallError(
            "The release index is not valid JSON; the model was not installed."
        ) from error
    if not isinstance(release, dict):
        raise ReleaseInstallError("The release index is not a JSON object.")
    parts: List[tuple] = []
    for asset in release.get("assets", []) or []:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name", ""))
        if not name.startswith(_MODEL_PART_PREFIX):
            continue
        match = _SAFE_PART_NAME.fullmatch(name)
        if not match:
            raise ReleaseInstallError(
                "A release model part has an unexpected name: {}.".format(name)
            )
        url = str(asset.get("browser_download_url") or "")
        _require_https(url, "release model part")
        if not _asset_host_allowed(url, source_host):
            raise ReleaseInstallError(
                "A release model part is served from an unexpected host; refusing it."
            )
        parts.append((int(match.group(1)), url))
    if not parts:
        raise ReleaseInstallError(
            "The release has no on-device model parts. If the repository is "
            "private, the appliance cannot read it anonymously."
        )
    parts.sort(key=lambda item: item[0])
    return [url for _, url in parts]


def _stream_parts_to_file(
    opener: Callable[[str], Any], urls: List[str], destination: Path
) -> str:
    """Stream the model parts, in order, into one tar; return its sha256.

    The reassembled tar is gigabytes, so it is never held in memory: each part is
    copied in chunks onto the end of ``destination`` while a single digest runs
    across the whole stream, and the running total is capped so an oversized or
    unbounded part fails loudly instead of filling the disk.
    """
    digest = hashlib.sha256()
    total = 0
    try:
        with open(destination, "wb") as handle:
            for url in urls:
                with opener(url) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > _MAX_MODEL_BYTES:
                            raise ReleaseInstallError(
                                "The release model exceeded the {} GiB safety "
                                "ceiling and was not installed.".format(
                                    _MAX_MODEL_BYTES // (1024 * 1024 * 1024)
                                )
                            )
                        digest.update(chunk)
                        handle.write(chunk)
    except OSError as error:
        raise ReleaseInstallError("Could not download the release model.") from error
    return digest.hexdigest()


def _verify_safe_members(members: Any, root: Path) -> None:
    """Refuse any tar member that could write outside the extraction root.

    ``tarfile`` will happily honour ``../`` and absolute paths, and a symlink or
    hardlink member whose own name is safe can still point its LINK at ``/etc``
    so a later member writes through it - the classic tar-slip. The model is
    fetched over the network, so before anything is written every member is
    checked two ways: its resolved path must stay under ``root``, and it must be a
    plain file or directory. The model artifact is only files and directories, so
    a symlink, hardlink, or device/fifo member is never legitimate here and is
    refused outright (which also makes the guard independent of the tarfile
    extraction filter, whose safe default only arrived in Python 3.12+/3.14).
    """
    root = root.resolve()
    for member in members:
        if not (member.isfile() or member.isdir()):
            raise ReleaseInstallError(
                "The release artifact contains a non-file member ({}); refusing "
                "it.".format(member.name)
            )
        target = (root / member.name).resolve()
        if root != target and root not in target.parents:
            raise ReleaseInstallError(
                "The release artifact contains an unsafe path: {}.".format(member.name)
            )


def install(
    source_url: str,
    flm_home: Path,
    runtime_root: Any = None,
    *,
    expected_sha256: str = "",
    opener: Callable[[str], Any] = _default_opener,
) -> Dict[str, Any]:
    """Fetch, verify and unpack the NPU model release into the flm models dir.

    ``source_url`` is the GitHub release index (``.../releases/tags/<tag>``);
    ``expected_sha256`` is the trust anchor the caller pins **in code** (the
    model-catalog entry) — the reassembled tar is verified against it, never
    against the release JSON, because a source that could serve a bad part would
    serve a matching index too. It is required: an empty or malformed pin refuses
    the install rather than trusting whatever bytes arrive.

    ``flm_home`` is FastFlowLM's home; the model unpacks into ``flm_home/models``,
    the directory ``flm serve`` reads. ``runtime_root`` is accepted for call
    compatibility and ignored — the FLM runtime is provisioned separately by the
    installer, and this artifact carries only the model. The multi-GB tar is
    streamed to a temporary file and never held in memory.

    Returns the model directory installed and the flm home. Raises
    ``ReleaseInstallError`` on any failure.
    """
    trust = str(expected_sha256 or "").lower()
    if not re.fullmatch(r"[a-f0-9]{64}", trust):
        raise ReleaseInstallError(
            "The pinned model sha256 is missing or not a hex digest; refusing to "
            "install an unverified model."
        )

    flm_home = Path(flm_home)
    models_parent = flm_home / "models"
    models_parent.mkdir(parents=True, exist_ok=True)

    staging = flm_home / "_release_staging"
    if staging.exists():
        _rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    artifact_file = staging / "model.tar"
    try:
        urls = _release_part_urls(opener, source_url)
        got = _stream_parts_to_file(opener, urls, artifact_file)
        if got != trust:
            raise ReleaseInstallError(
                "The release model failed its sha256 pin and was not installed."
            )

        with tarfile.open(artifact_file) as archive:
            members = archive.getmembers()
            _verify_safe_members(members, staging)
            # `filter="data"` (Python 3.12+) is defence in depth on top of the
            # member check above; older supported interpreters (3.10/3.11 before
            # the backport) lack the kwarg, so fall back to a plain extract - the
            # member check has already rejected anything unsafe on every version.
            try:
                archive.extractall(staging, filter="data")
            except TypeError:
                archive.extractall(staging)
        artifact_file.unlink(missing_ok=True)

        # Install EVERY top-level entry the tar carried, the way
        # `tar -C models_dir -xf` (deploy/fetch-npu-model.sh) does, rather than
        # keeping only one directory and silently dropping siblings. Today the
        # artifact is the single `Qwen3.5-4B-NPU2/` directory; installing all
        # entries keeps the two installers identical if that ever changes.
        entries = sorted(staging.iterdir(), key=lambda child: child.name)
        installed_dirs = [child.name for child in entries if child.is_dir()]
        if not installed_dirs:
            raise ReleaseInstallError("The release model archive has no model directory.")
        for child in entries:
            destination = models_parent / child.name
            if destination.exists():
                _rmtree(destination)
            child.replace(destination)
        model_dir_name = installed_dirs[0]
        dest_model = models_parent / model_dir_name
    finally:
        _rmtree(staging)

    return {
        "flm_home": str(flm_home),
        "models_dir": str(models_parent),
        "model_dir": str(dest_model),
        "model": model_dir_name,
    }


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
