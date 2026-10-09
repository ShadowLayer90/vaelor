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
against real bytes. `GitHubReleaseSource` is the commissioned network source:
it reads the repository's prereleases, pins the wheel's SHA-256 from the
release's own `SHA256SUMS` asset fetched over HTTPS, and hands those pinned
bytes to the same verify-artifact step the stub feeds - so the redirect a
GitHub asset URL follows to a CDN is harmless, because the caller recomputes
the digest against the manifest pin and refuses a mismatch (#130).
"""

from __future__ import annotations

import json
import re
import shutil
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable
from urllib.parse import urlsplit

from packaging.version import InvalidVersion, Version

from .build_provenance import BASIS_STAMP, BUILT_AT_BASES, INSTALLED_RECORDED, parse_time
from .published_build_stamp import (
    SUMS_NAME,
    parse_sums,
    published_stamp_name,
    verify_published_stamp,
)
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
#: The published wheel's own build identity, when the release states it: the
#: time its release build stamped (``build_provenance.BASIS_STAMP``), and the
#: commit. Optional because no release published so far carries them, and
#: ``published_at`` cannot stand in: it is the upload time, which an old build
#: uploaded again today also carries (F2, VD-184). Without them a different
#: build of the same version is never offered as a replacement.
MANIFEST_BUILD_FIELDS = ("built_at", "built_at_basis", "commit")
_HEX_DIGITS = frozenset("0123456789abcdef")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
#: A wheel filename, never a path. `..` and separators are refused so a manifest
#: can never name a destination outside the staging directory.
_WHEEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*\.whl$")

#: The repository whose prereleases the appliance updater offers. The releases
#: are curated source + wheel snapshots (see MEMORY: public-vaelor-repo), so the
#: source reads `/releases`, not `/releases/latest`, which skips prereleases.
DEFAULT_UPDATE_REPO = "ShadowLayer90/vaelor"


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
    built_at: Optional[str] = None
    built_at_basis: Optional[str] = None
    commit: Optional[str] = None
    #: Why a stamp the release published was not believed; empty when there
    #: was none or it was believed. Set by the source, never by a manifest.
    build_stamp_problem: str = ""

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
            "built_at": self.built_at,
            "built_at_basis": self.built_at_basis,
            "commit": self.commit,
            "build_stamp_problem": self.build_stamp_problem,
        }


def _manifest_build_identity(raw: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """The optional build fields, each either absent or exactly right.

    A build time that does not parse, or a basis that is not one of
    ``BUILT_AT_BASES``, is refused here: a release that states its build
    imprecisely must not reach a comparison that would believe it (LESSONS 9).
    """
    built_at = raw.get("built_at")
    basis = raw.get("built_at_basis")
    commit = raw.get("commit")
    if built_at is not None:
        if not isinstance(built_at, str) or parse_time(built_at) is None:
            raise ReleaseError("The manifest build time is not an ISO 8601 time.")
        if basis not in BUILT_AT_BASES:
            raise ReleaseError(
                "The manifest build time does not say how it was measured "
                "(built_at_basis must be one of {}).".format(", ".join(BUILT_AT_BASES)))
    elif basis is not None:
        raise ReleaseError("The manifest names a build-time basis but no build time.")
    if commit is not None and (not isinstance(commit, str) or not 7 <= len(commit) <= 40 or not set(commit) <= _HEX_DIGITS):
        raise ReleaseError("The manifest commit is not a hexadecimal commit id.")
    return {"built_at": built_at, "built_at_basis": basis, "commit": commit}


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
    extra = [key for key in raw
             if key not in MANIFEST_FIELDS and key not in MANIFEST_BUILD_FIELDS]
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
        **_manifest_build_identity(raw),
    )


#: What the offered release is relative to the installed build. ``update`` and
#: ``same-build`` and ``replace-build`` are eligible; the rest are refusals.
#: A wire vocabulary (``eligibility.kind``) the update panel branches on: its
#: one hand copy is OFFER_KINDS in SystemUpdatePanel.tsx (LESSONS 6 / VD-090
#: item 5, W4d-D8). ``unparseable`` is written by api_upgrade_routes.
OFFER_KINDS = (
    "update",
    "same-build",
    "replace-build",
    "local-build-newer",
    "build-not-comparable",
    "build-unknown",
    "older",
    "too-old-to-upgrade",
    "unsigned",
    "unparseable",
)


def _short(sha256: Any) -> str:
    return str(sha256 or "")[:12] or "unknown"


def _day(iso: Any) -> str:
    return str(iso or "")[:10] or "an unrecorded date"


def upgrade_eligibility(
    running_version: str,
    manifest: ReleaseManifest,
    installed: Optional[Dict[str, Any]] = None,
    *,
    signature_required: bool = False,
) -> Dict[str, Any]:
    """Whether this appliance may take this release, what it is, and why not.

    One decision, read by both ``GET /api/v2/upgrade`` and the executor's check
    step, so the interface and the job never disagree (LESSONS 6).

    W4d-D8: an equal version string is not an equal build. A box running a
    1.0b2 wheel built today was told it was "up to date" and offered a reinstall
    of the month-older, different published 1.0b2. So when the versions are
    equal the *build* decides, from ``installed`` (what
    ``build_provenance.installed_build`` returns): the same wheel SHA-256 is a
    reinstall of the same bytes; a different wheel is installed only when the
    published one is newer than the installed build; and an installed build
    Vaelor cannot identify is refused, because "cannot tell" must not be read as
    "same" (LESSONS 8). A newer version number is an update whatever the build.
    """
    def verdict(eligible: bool, kind: str, reason: str = "") -> Dict[str, Any]:
        return {"eligible": eligible, "kind": kind, "reason": reason}

    try:
        running = Version(str(running_version))
        minimum = Version(manifest.min_from_version)
        target = Version(manifest.version)
    except InvalidVersion:
        return verdict(False, "unparseable", "A version number could not be parsed.")
    if running < minimum:
        return verdict(
            False, "too-old-to-upgrade",
            "This release upgrades from {} or newer; this appliance runs {}."
            .format(manifest.min_from_version, running_version),
        )
    if target < running:
        return verdict(
            False, "older",
            "The offered release {} is older than the running {}."
            .format(manifest.version, running_version),
        )
    if signature_required and not manifest.signature:
        return verdict(
            False, "unsigned",
            "The offered release {} is not signed, and this release source "
            "accepts only signed releases.".format(manifest.version),
        )
    if target > running:
        return verdict(True, "update")
    installed = installed or {}
    build = installed.get("build") if installed.get("state") == INSTALLED_RECORDED else None
    if not isinstance(build, dict):
        return verdict(
            False, "build-unknown",
            "Vaelor cannot tell which build of {} this appliance runs ({}) so it "
            "cannot tell whether the published {} is the same build, a newer one "
            "or an older one, and will not replace it. Installing Vaelor with the "
            "installer records the build.".format(
                running_version,
                (installed.get("reason") or "no install record").rstrip("."),
                manifest.version,
            ),
        )
    if str(build.get("sha256") or "") == manifest.sha256:
        return verdict(True, "same-build")
    return _different_build_verdict(running_version, build, manifest, verdict)


def _different_build_verdict(running_version, build, manifest, verdict):
    """Two different wheels of one version: replace only on proof of newer.

    **F2 (adversarial review, VD-184).** This first compared the local build
    time with the manifest's ``published_at`` - the time a release was
    *uploaded*. An old build uploaded again today carries today's date, so it
    was judged ``replace-build`` and the W4d-D8 downgrade came back (LESSONS 5,
    a label that is not the measurement). Now both sides must carry a build
    time their release build *stamped* (``BASIS_STAMP``), and the published one
    must be strictly later. A ``newest-file-in-wheel`` time is a lower bound -
    it cannot show a local build is old, and a file dated in the future would
    make it show a published one is new - so it proves nothing either way, and
    equal stamped times order nothing. Commits are carried but not ordered:
    which of two commits is newer is a fact about history this appliance does
    not hold.
    """
    sha_local, sha_published = _short(build.get("sha256")), _short(manifest.sha256)
    head = "This appliance runs a different build of {} (wheel {}) from the published {} (wheel {}). ".format(
        running_version, sha_local, manifest.version, sha_published)
    if manifest.built_at is None and manifest.build_stamp_problem:
        return verdict(
            False, "build-not-comparable",
            head + "The published release carries a build stamp that could not be "
            "authenticated ({}), so its build time is not believed, it is not "
            "provably newer, and Vaelor will not replace this build with it.".format(
                manifest.build_stamp_problem))
    if manifest.built_at is None:
        return verdict(
            False, "build-not-comparable",
            head + "The published release does not say when its wheel was built - "
            "it records only its upload time ({}), which an older build uploaded "
            "again would carry too - so it is not provably newer, and Vaelor will "
            "not replace this build with it.".format(_day(manifest.published_at)))
    local_time = parse_time(build.get("built_at"))
    if local_time is None:
        return verdict(
            False, "build-not-comparable",
            head + "This build's time was never recorded, so the published one is "
            "not provably newer, and Vaelor will not replace this build with it.")
    stamped = (build.get("built_at_basis") == BASIS_STAMP,
               manifest.built_at_basis == BASIS_STAMP)
    if not all(stamped):
        side = "this appliance's build" if not stamped[0] else "the published build"
        return verdict(
            False, "build-not-comparable",
            head + "The build time of {} is only a lower bound (the newest file in "
            "its wheel), so the published one is not provably newer, and Vaelor "
            "will not replace this build with it.".format(side))
    published_time = parse_time(manifest.built_at)
    if published_time > local_time:
        return verdict(True, "replace-build")
    if published_time == local_time:
        return verdict(
            False, "build-not-comparable",
            head + "Both were stamped as built at {}, so neither is provably newer, "
            "and Vaelor will not replace this build with it.".format(manifest.built_at))
    return verdict(
        False, "local-build-newer",
        head + "This build was made {} and the published one {}: the published "
        "build is older, so Vaelor will not replace this one with it.".format(
            _day(build.get("built_at")), _day(manifest.built_at)))


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
    #: Whether a release without a signature is refused. Nothing in Vaelor signs
    #: releases yet, so no source claims to; a source that does sets this True
    #: and `upgrade_eligibility` refuses an unsigned manifest from it.
    signatures_required = False

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


#: The wheel name always carries the version in this shape; the version is read
#: from the filename so "latest" is a real version comparison, not a tag guess.
_WHEEL_VERSION = re.compile(
    r"^vaelor_control_plane-(?P<version>.+?)-py3-none-any\.whl$"
)


class GitHubReleaseSource:
    """The commissioned network source: GitHub prereleases, digest-pinned.

    The repository publishes each release as a *prerelease* carrying a wheel
    asset and a ``SHA256SUMS`` asset. This source reads the releases list (never
    ``/releases/latest``, which skips prereleases), picks the highest wheel
    version across them, and reads that release's own ``SHA256SUMS`` over HTTPS
    to pin the wheel's SHA-256 into the manifest. It never certifies the bytes
    it later fetches: the redirect a GitHub asset URL follows to a CDN is
    harmless because ``appliance_upgrade._verify_staged_wheel`` recomputes the
    digest and length against this pin and refuses a mismatch - the #130
    verification story, moved onto the network without loosening it.

    A network hiccup is *no release offered*, not an error: every reach for the
    releases list is caught and turns into ``None`` so a transient GitHub outage
    never crashes ``GET /api/v2/upgrade``. A malformed pin, by contrast, flows
    through ``parse_manifest`` and surfaces as :class:`ReleaseError`, which the
    routes already catch and report as an ineligible offer.
    """

    name = "github"
    #: GitHub releases carry a SHA256SUMS file, not a signature: the digest
    #: proves the download matches the release, not who published it.
    signatures_required = False

    def __init__(
        self,
        repository: str,
        opener: Optional[Callable[..., Any]] = None,
        *,
        min_from_version: str = "1.0b1",
    ):
        self.repository = repository
        self._opener = opener or urllib.request.urlopen
        self._min_from_version = min_from_version

    def _api_json(self, url: str) -> Any:
        """GET a GitHub API URL as JSON, or ``None`` on any network fault.

        The User-Agent header is mandatory for the GitHub API; the timeout keeps
        a stalled connection from wedging the upgrade check. Any transport or
        decode failure is swallowed to ``None`` so the caller treats it as
        "nothing offered" rather than raising into the route.
        """
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "vaelor-appliance",
                "Accept": "application/vnd.github+json",
            },
        )
        try:
            with self._opener(request, timeout=15) as response:
                return json.loads(response.read())
        except (OSError, ValueError):
            return None

    def latest_manifest(self) -> Optional[ReleaseManifest]:
        releases = self._api_json(
            "https://api.github.com/repos/{}/releases?per_page=20".format(
                self.repository
            )
        )
        if not isinstance(releases, list):
            return None

        chosen: Optional[Dict[str, Any]] = None
        chosen_version: Optional[Version] = None
        chosen_wheel: Optional[Dict[str, Any]] = None
        chosen_sums: Optional[Dict[str, Any]] = None
        chosen_stamp: Optional[Dict[str, Any]] = None
        for release in releases:
            if not isinstance(release, dict) or release.get("draft"):
                continue
            wheel_asset = None
            sums_asset = None
            assets_by_name: Dict[str, Dict[str, Any]] = {}
            for asset in release.get("assets") or []:
                if not isinstance(asset, dict):
                    continue
                name = asset.get("name")
                # A malformed API response can carry a non-string name; a bare
                # `or ""` would still hand it to fullmatch/startswith and raise
                # TypeError up into GET /api/v2/upgrade (a 500). Skip it instead.
                if not isinstance(name, str):
                    continue
                assets_by_name[name] = asset
                if name == SUMS_NAME:
                    sums_asset = asset
                elif (
                    _WHEEL_NAME.fullmatch(name)
                    and name.startswith("vaelor_control_plane-")
                ):
                    wheel_asset = asset
            if wheel_asset is None or sums_asset is None:
                continue
            match = _WHEEL_VERSION.match(wheel_asset.get("name") or "")
            if match is None:
                continue
            try:
                version = Version(match.group("version"))
            except InvalidVersion:
                continue
            if chosen_version is None or version > chosen_version:
                chosen = release
                chosen_version = version
                chosen_wheel = wheel_asset
                chosen_sums = sums_asset
                chosen_stamp = assets_by_name.get(
                    published_stamp_name(wheel_asset["name"]))

        if chosen is None:
            return None

        # A published release always carries a download URL and a publish time;
        # a missing/non-string one is a malformed offer, not something to render
        # as "None" or fetch from a bogus URL. Treat it as no offer.
        wheel_url = chosen_wheel.get("browser_download_url")
        published_at = chosen.get("published_at")
        if not isinstance(wheel_url, str) or not wheel_url:
            return None
        if not isinstance(published_at, str) or not published_at.strip():
            return None

        sums = self._sums(chosen_sums.get("browser_download_url") or "")
        if sums is None:
            return None
        sha256 = sums.get(chosen_wheel.get("name") or "")
        if sha256 is None:
            return None

        manifest = parse_manifest(
            {
                "version": _WHEEL_VERSION.match(
                    chosen_wheel["name"]
                ).group("version"),
                "wheel_url": wheel_url,
                "wheel_name": chosen_wheel.get("name"),
                "sha256": sha256,
                "bytes": chosen_wheel.get("size"),
                "min_from_version": self._min_from_version,
                "signature": None,
                "published_at": published_at,
            }
        )
        if chosen_stamp is None:
            return manifest
        return self._with_published_stamp(manifest, chosen_stamp, sums)

    def _with_published_stamp(
        self, manifest: ReleaseManifest, asset: Dict[str, Any], sums: Dict[str, str],
    ) -> ReleaseManifest:
        """The manifest with the release's build stamp, if the release vouches for it.

        W4d-D8 gap: the stamp is believed only as far as the release's own
        ``SHA256SUMS`` covers it and binds it to this wheel's digest
        (``published_build_stamp.verify_published_stamp``). Anything less leaves
        the manifest without a build time and records why, so the decision says
        "not comparable" rather than trusting it (LESSONS 9).
        """
        url = asset.get("browser_download_url")
        body: Optional[bytes] = None
        if isinstance(url, str) and url:
            try:
                with self._opener(url, timeout=30) as response:
                    body = response.read()
            except (OSError, ValueError):  # absence-ok: an unreadable stamp is recorded as not believed
                body = None
        identity, problem = verify_published_stamp(
            body, wheel_name=manifest.wheel_name, wheel_sha256=manifest.sha256, sums=sums,
        )
        if identity is None:
            return replace(manifest, build_stamp_problem=problem)
        return replace(manifest, **identity)

    def _sums(self, sums_url: str) -> Optional[Dict[str, str]]:
        """The release's ``SHA256SUMS`` as ``{filename: digest}``, or ``None``.

        The file is the ``<64hex> *<name>`` / ``<64hex>  <name>`` format
        ``sha256sum`` emits. A missing wheel line is no offer; a malformed digest
        is left for ``parse_manifest`` to refuse as a :class:`ReleaseError`.
        """
        if not sums_url:
            return None
        try:
            with self._opener(sums_url, timeout=30) as response:
                body = response.read().decode("utf-8")
        except (OSError, ValueError):
            return None
        return parse_sums(body)

    def fetch(self, manifest: ReleaseManifest, destination: Path) -> Path:
        """Stream the manifest's wheel to ``destination`` and return the path.

        The GitHub asset URL redirects to a CDN; the redirect is followed and the
        bytes are *not* verified here. The caller certifies them against the
        manifest pin, exactly as the contract on :class:`ReleaseSource` says.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._opener(manifest.wheel_url, timeout=30) as response:
            with open(destination, "wb") as handle:
                shutil.copyfileobj(response, handle)
        return destination


def default_release_source() -> ReleaseSource:
    """The release source the appliance runs against by default.

    Points at :data:`DEFAULT_UPDATE_REPO`, overridable per box through
    ``VAELOR_UPDATE_REPO`` (or the ``VAELOR_REPO`` legacy alias) so a fork or a
    staging repository can be served without a code change.
    """
    return GitHubReleaseSource(
        env_value("VAELOR_UPDATE_REPO", "VAELOR_REPO", DEFAULT_UPDATE_REPO)
    )
