"""Pinned Vaelor release manifests and the sources that serve them.

An appliance upgrade installs *bytes*, and every defect this project has had
around delivery came from something floating where it should have been pinned:
a version read from the installed package rather than the tree (#92/#47), a
catalog entry trusting one publisher's byte count for another's file (#104), a
"latest" tag that changed under a running deployment (#130). So a release here
is a *manifest*: a fixed version, a fixed URL, a fixed byte length and a fixed
SHA-256, and nothing about the upgrade is allowed to proceed on anything the
manifest does not pin.

`StubReleaseSource` is a real, driven source, not a test double - it reads a
pinned manifest file and copies the exact wheel that manifest names, so the
whole check -> fetch -> verify-artifact -> stage state machine runs offline
against real bytes. `GitHubReleaseSource` is the network skeleton for later;
it is deliberately inert until its verification story is built, because a
half-built network fetch that trusts a redirect is exactly the shape #130
warns about.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable
from urllib.parse import urlsplit

from packaging.version import InvalidVersion, Version

from .runtime_paths import env_value, state_path


#: The pinned fields every manifest must carry. `signature` is null in the stub
#: (there is no signing key on a developer box); it is present so a signed
#: source is a value change, not a schema change.
MANIFEST_FIELDS = (
    "version",
    "wheel_url",
    "wheel_name",
    "sha256",
    "bytes",
    "min_from_version",
    "signature",
    "published_at",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
#: A wheel filename, never a path. `..` and separators are refused so a manifest
#: can never name a destination outside the staging directory.
_WHEEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*\.whl$")


class ReleaseError(ValueError):
    """A manifest is malformed, or a source cannot honour its own pin."""


@dataclass(frozen=True)
class ReleaseManifest:
    """One immutable release, described entirely by pinned values."""

    version: str
    wheel_url: str
    wheel_name: str
    sha256: str
    bytes: int
    min_from_version: str
    signature: Optional[str]
    published_at: str

    def public(self) -> Dict[str, Any]:
        """The manifest as an API surface omits the internal fetch URL.

        The URL is where the appliance reaches for the bytes; it is not a fact
        an operator acts on, and on the stub it is a ``file://`` path on the
        box. Everything an eligibility decision needs - version, size, digest,
        the minimum it upgrades from - is here.
        """
        return {
            "version": self.version,
            "wheel_name": self.wheel_name,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "min_from_version": self.min_from_version,
            "signature": self.signature,
            "published_at": self.published_at,
        }


def parse_manifest(raw: Any) -> ReleaseManifest:
    """Validate a raw manifest into a :class:`ReleaseManifest`, or raise.

    Every field is required and typed. A missing or floating value is refused
    here rather than discovered at install time, because #130's whole lesson is
    that the pin has to be checked before the bytes are trusted, not after.
    """
    if not isinstance(raw, dict):
        raise ReleaseError("A release manifest must be a JSON object.")
    missing = [field for field in MANIFEST_FIELDS if field not in raw]
    if missing:
        raise ReleaseError(
            "The release manifest is missing pinned fields: {}.".format(
                ", ".join(missing)
            )
        )
    extra = [key for key in raw if key not in MANIFEST_FIELDS]
    if extra:
        raise ReleaseError(
            "The release manifest carries unknown fields: {}.".format(
                ", ".join(sorted(extra))
            )
        )
    version = str(raw["version"]).strip()
    min_from = str(raw["min_from_version"]).strip()
    wheel_name = str(raw["wheel_name"]).strip()
    wheel_url = str(raw["wheel_url"]).strip()
    sha256 = str(raw["sha256"]).strip().lower()
    published_at = str(raw["published_at"]).strip()
    signature = raw["signature"]
    if not version or not min_from or not published_at or not wheel_url:
        raise ReleaseError("A release manifest field is empty.")
    if not _WHEEL_NAME.fullmatch(wheel_name):
        raise ReleaseError("The manifest wheel name is not a plain .whl filename.")
    if not _SHA256.fullmatch(sha256):
        raise ReleaseError("The manifest SHA-256 is not 64 lowercase hex digits.")
    if not isinstance(raw["bytes"], int) or isinstance(raw["bytes"], bool) or raw["bytes"] <= 0:
        raise ReleaseError("The manifest byte length must be a positive integer.")
    if signature is not None and not isinstance(signature, str):
        raise ReleaseError("The manifest signature must be a string or null.")
    return ReleaseManifest(
        version=version,
        wheel_url=wheel_url,
        wheel_name=wheel_name,
        sha256=sha256,
        bytes=int(raw["bytes"]),
        min_from_version=min_from,
        signature=signature,
        published_at=published_at,
    )


def upgrade_eligibility(
    running_version: str, manifest: ReleaseManifest
) -> Dict[str, Any]:
    """Whether ``running_version`` may take this release, and why not if not.

    One decision, read by both ``GET /api/v2/upgrade`` and the executor's check
    step, so the interface and the job never disagree about eligibility - the
    LESSONS 6 shape. Note that a target *equal* to the running version is
    eligible: an upgrade to the same version number is a real reinstall through
    ``--force-reinstall`` (#134), not a no-op to be refused here.
    """
    try:
        running = Version(str(running_version))
        minimum = Version(manifest.min_from_version)
        target = Version(manifest.version)
    except InvalidVersion:
        return {"eligible": False, "reason": "A version number could not be parsed."}
    if running < minimum:
        return {
            "eligible": False,
            "reason": (
                "This release upgrades from {} or newer; this appliance runs {}."
                .format(manifest.min_from_version, running_version)
            ),
        }
    if target < running:
        return {
            "eligible": False,
            "reason": (
                "The offered release {} is older than the running {}."
                .format(manifest.version, running_version)
            ),
        }
    return {"eligible": True, "reason": ""}


@runtime_checkable
class ReleaseSource(Protocol):
    """Where an upgrade's manifest and bytes come from."""

    name: str

    def latest_manifest(self) -> Optional[ReleaseManifest]:
        """The newest offered release, or ``None`` when none is published."""

    def fetch(self, manifest: ReleaseManifest, destination: Path) -> Path:
        """Place the manifest's wheel at ``destination`` and return the path.

        A source *delivers* bytes; it does not certify them. The caller
        recomputes the digest and length against the manifest, so a source that
        happens to hand back the wrong file is caught by verify-artifact rather
        than trusted here.
        """


def default_manifest_path() -> Path:
    return Path(
        env_value(
            "VAELOR_RELEASE_MANIFEST",
            "PM_RELEASE_MANIFEST",
            state_path("upgrade/manifest.json"),
        )
    )


class StubReleaseSource:
    """A real source backed by a pinned local manifest and its wheel.

    This is not a monkeypatch. It reads an on-disk manifest and copies the exact
    wheel the manifest names, so a test - or an offline appliance - drives the
    entire upgrade state machine against real bytes. The manifest's ``wheel_url``
    is a ``file://`` URL or a bare filename resolved next to the manifest.
    """

    name = "stub"

    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifest_path = (
            Path(manifest_path) if manifest_path is not None
            else default_manifest_path()
        )

    def latest_manifest(self) -> Optional[ReleaseManifest]:
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return parse_manifest(raw)

    def _wheel_path(self, manifest: ReleaseManifest) -> Path:
        parsed = urlsplit(manifest.wheel_url)
        if parsed.scheme in ("", "file"):
            raw = parsed.path if parsed.scheme == "file" else manifest.wheel_url
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = self.manifest_path.parent / raw
            return candidate.resolve()
        raise ReleaseError(
            "The stub release source serves local wheels only; "
            "this manifest points off the box."
        )

    def fetch(self, manifest: ReleaseManifest, destination: Path) -> Path:
        source = self._wheel_path(manifest)
        if not source.is_file():
            raise ReleaseError("The pinned release wheel is not present locally.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination


class GitHubReleaseSource:
    """Network source skeleton. Inert until its verification story is built.

    Left deliberately unbuilt: a fetch that follows a redirect to arbitrary
    bytes and installs them is the #130 defect wearing a network coat. When this
    is commissioned it must pin the asset by digest before a single byte is
    trusted, exactly as verify-artifact does for the stub.
    """

    name = "github"

    def __init__(
        self,
        repository: str,
        opener: Optional[Callable[..., Any]] = None,
    ):
        self.repository = repository
        self._opener = opener

    def latest_manifest(self) -> Optional[ReleaseManifest]:  # pragma: no cover - skeleton
        raise NotImplementedError(
            "The GitHub release source is not commissioned yet. Its manifest "
            "fetch must pin the release asset by digest before any byte is "
            "trusted (#130)."
        )

    def fetch(self, manifest: ReleaseManifest, destination: Path) -> Path:  # pragma: no cover - skeleton
        raise NotImplementedError(
            "The GitHub release source is not commissioned yet."
        )
