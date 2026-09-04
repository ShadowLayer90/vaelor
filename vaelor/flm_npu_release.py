"""Install a fine-tuned on-device NPU model from a pinned release artifact.

The stock ``lemonade-server`` snap carries only public flm models, so a
fine-tuned NPU model (this appliance's Qwen3.5-4B ff-4b-h) cannot arrive through
it. It is delivered instead as a **release**: a single tar the appliance
downloads, verifies against a pinned SHA-256, and unpacks into a Vaelor-managed
flm home and runtime directory. The release carries BOTH the model and the exact
``flm-real`` build it was converted for, so the two travel together and the
appliance never depends on whichever flm version the snap happens to ship
(measured on the Z2 clean install: the stock snap's flm is a different build,
and the q4nx was converted for v1.0.2).

This module is the turnkey "Install" action for such a model — no operator CLI,
and the artifact is refused unless its bytes match the manifest. It resolves a
source that today is a simulated local release and in production is a published
release URL; the caller passes the base URL, so the two are the same code.

Layout the install produces, mirroring what ``flm-real serve <tag>`` reads::

    <flm_home>/.config/flm/models/<model_dir_name>/   the q4nx model + tokenizer
    <runtime_root>/                                    flm-real + lib/ + xclbins/

The manifest is small JSON; the artifact is the large tar. Both are fetched
through the injected ``opener`` (``urllib`` by default) so a test drives the
whole path against a ``file://`` or loopback source with no network.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any, Callable, Dict
from urllib.parse import urljoin
from urllib.request import urlopen

#: Every field the appliance needs before it will fetch or trust an artifact.
#: A manifest missing any of these is rejected rather than half-read - the same
#: refuse-loudly posture as the model catalog's size+digest gate.
REQUIRED_MANIFEST_KEYS = ("tag", "artifact", "sha256", "size_bytes")

#: The model tag is the flm serving key AND a path segment, so it is constrained
#: to what is safe as both: a wrong or hostile tag must not escape the flm home.
_SAFE_TAG = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,63}$")
#: A release artifact is a plain file name, never a path - it is joined onto the
#: model directory, and ``../`` there would write outside it.
_SAFE_ARTIFACT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ReleaseInstallError(Exception):
    """A release could not be installed - stated so the UI can show why."""


def read_manifest(raw: Any) -> Dict[str, Any]:
    """Parse and validate a release manifest, or raise ``ReleaseInstallError``."""
    try:
        data = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
    except (ValueError, TypeError) as error:
        raise ReleaseInstallError("The release manifest is not valid JSON.") from error
    if not isinstance(data, dict):
        raise ReleaseInstallError("The release manifest must be a JSON object.")
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in data]
    if missing:
        raise ReleaseInstallError(
            "The release manifest is missing: {}.".format(", ".join(missing))
        )
    tag = str(data.get("tag", ""))
    if not _SAFE_TAG.fullmatch(tag):
        raise ReleaseInstallError("The release model tag is not a safe identifier.")
    artifact = str(data.get("artifact", ""))
    if not _SAFE_ARTIFACT.fullmatch(artifact):
        raise ReleaseInstallError("The release artifact name is not a safe file name.")
    sha = str(data.get("sha256", "")).lower()
    if not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ReleaseInstallError("The release manifest sha256 is not a hex digest.")
    try:
        size = int(data.get("size_bytes"))
    except (TypeError, ValueError) as error:
        raise ReleaseInstallError("The release manifest size_bytes is not a number.") from error
    if size <= 0:
        raise ReleaseInstallError("The release manifest size_bytes must be positive.")
    return {**data, "tag": tag, "artifact": artifact, "sha256": sha, "size_bytes": size}


def _fetch(opener: Callable[[str], Any], url: str) -> bytes:
    try:
        with opener(url) as response:
            return response.read()
    except OSError as error:
        raise ReleaseInstallError("Could not reach the release source.") from error


def _stream_to_file(opener: Callable[[str], Any], url: str, destination: Path) -> str:
    """Stream a (large) artifact to disk, returning its sha256.

    The artifact is gigabytes, so it is never held in memory: it is copied in
    chunks to ``destination`` while its digest is computed in the same pass, so
    the verify below costs no second read.
    """
    digest = hashlib.sha256()
    try:
        with opener(url) as response, open(destination, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                handle.write(chunk)
    except OSError as error:
        raise ReleaseInstallError("Could not download the release artifact.") from error
    return digest.hexdigest()


def _verify_no_traversal(members: Any, root: Path) -> None:
    """Refuse any tar member that would write outside the extraction root.

    ``tarfile`` will happily honour ``../`` and absolute paths; a release is
    fetched over the network, so every member is checked against the resolved
    root before anything is written (the model install must not become a way to
    drop a file into ``/etc``).
    """
    root = root.resolve()
    for member in members:
        target = (root / member.name).resolve()
        if root != target and root not in target.parents:
            raise ReleaseInstallError(
                "The release artifact contains an unsafe path: {}.".format(member.name)
            )


def install(
    source_url: str,
    flm_home: Path,
    runtime_root: Path,
    *,
    expected_sha256: str = "",
    opener: Callable[[str], Any] = urlopen,
    manifest_name: str = "manifest.json",
) -> Dict[str, Any]:
    """Fetch, verify and unpack a release into a Vaelor-managed flm home.

    ``source_url`` is the release base (a directory URL); ``manifest_name`` and
    the manifest's ``artifact`` are resolved against it. ``expected_sha256`` is
    the trust anchor the caller pins **in code** (the model-catalog entry): when
    given, the downloaded artifact is verified against it, NOT against the
    manifest fetched from the same source - a source that could serve a bad
    artifact would also serve a matching manifest, so the manifest's own digest
    is a consistency check, never the trust root. Passing it empty (tests, a
    simulated release) falls back to the manifest digest.

    Returns the paths the serving layer needs: the flm home, the runtime
    directory holding ``flm-real``, and the model tag to serve. Raises
    ``ReleaseInstallError`` on any failure. The multi-GB artifact is streamed to
    a temporary file and never held in memory.
    """
    base = source_url if source_url.endswith("/") else source_url + "/"
    manifest = read_manifest(_fetch(opener, urljoin(base, manifest_name)))
    trust = str(expected_sha256 or "").lower()
    if trust and not re.fullmatch(r"[a-f0-9]{64}", trust):
        raise ReleaseInstallError("The pinned expected_sha256 is not a hex digest.")
    if trust and trust != manifest["sha256"]:
        raise ReleaseInstallError(
            "The release manifest digest does not match the pinned catalog digest."
        )

    flm_home = Path(flm_home)
    runtime_root = Path(runtime_root)
    models_parent = flm_home / ".config" / "flm" / "models"
    models_parent.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    staging = runtime_root.parent / "_release_staging"
    if staging.exists():
        _rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    artifact_file = staging / "artifact.tar"
    got = _stream_to_file(opener, urljoin(base, manifest["artifact"]), artifact_file)
    size = artifact_file.stat().st_size
    if size != manifest["size_bytes"]:
        _rmtree(staging)
        raise ReleaseInstallError(
            "The release artifact is {} bytes; the manifest says {}.".format(
                size, manifest["size_bytes"]
            )
        )
    if got != (trust or manifest["sha256"]):
        _rmtree(staging)
        raise ReleaseInstallError(
            "The release artifact failed its sha256 pin and was not installed."
        )

    # Unpack in two stages: the outer tar (model/ + runtime/<bundle>.tar.gz), then
    # the runtime bundle. Each is traversal-checked before extraction.
    with tarfile.open(artifact_file) as outer:
        members = outer.getmembers()
        _verify_no_traversal(members, staging)
        outer.extractall(staging)
    artifact_file.unlink(missing_ok=True)

    model_src = staging / "model"
    model_dirs = [child for child in model_src.iterdir() if child.is_dir()] if model_src.is_dir() else []
    if not model_dirs:
        _rmtree(staging)
        raise ReleaseInstallError("The release artifact has no model directory.")
    model_dir_name = model_dirs[0].name
    dest_model = models_parent / model_dir_name
    if dest_model.exists():
        _rmtree(dest_model)
    model_dirs[0].replace(dest_model)

    runtime_bundles = list((staging / "runtime").glob("*.tar.gz")) if (staging / "runtime").is_dir() else []
    if not runtime_bundles:
        _rmtree(staging)
        raise ReleaseInstallError("The release artifact has no runtime bundle.")
    with tarfile.open(runtime_bundles[0]) as runtime_tar:
        rmembers = runtime_tar.getmembers()
        _verify_no_traversal(rmembers, runtime_root)
        runtime_tar.extractall(runtime_root)
    _rmtree(staging)

    flm_real = runtime_root / "flm-real"
    if not flm_real.is_file():
        raise ReleaseInstallError("The release runtime bundle has no flm-real binary.")
    flm_real.chmod(0o755)

    return {
        "tag": manifest["tag"],
        "flm_home": str(flm_home),
        "runtime_root": str(runtime_root),
        "flm_real": str(flm_real),
        "model_dir": str(dest_model),
        "name": str(manifest.get("name", manifest["tag"])),
        "version": str(manifest.get("version", "")),
    }


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
