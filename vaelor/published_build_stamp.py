"""The build stamp a release publishes beside its wheel, and how it is believed.

W4d-D8 gap (owner decision 2026-10-03). A same-version build may replace the
installed one only when the published build is provably newer, and the only
time a GitHub release carried was its upload time, which an old build uploaded
again today carries too (LESSONS 5). So ``tools/build_release.py`` publishes
``<wheel>.build-stamp.json`` beside the wheel - the commit and the stamped build
time the wheel itself carries in ``vaelor/build_stamp.json`` - and writes a
``SHA256SUMS`` that covers both files. ``release_source.GitHubReleaseSource``
reads it back through :func:`verify_published_stamp`.

**What authenticates it, exactly.** The same thing that authenticates the
wheel, and no more: the release's own ``SHA256SUMS``, fetched over HTTPS from
the release that offers the wheel. A stamp whose digest that file does not
list, whose bytes do not match the listed digest, or that names a different
wheel digest than the one the file pins, is not believed - the manifest then
carries no build time, and the update decision says the builds are not
comparable (LESSONS 9: a stated build time is a fact about a build only when
something binds it to that build). Nothing here is a signature; the manifest's
``signature`` stays null until Vaelor signs releases.

One module writes the format and the same module reads it, so the publisher and
the reader cannot drift (LESSONS 6).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from .build_provenance import BASIS_STAMP, parse_time

#: The checksum file a release publishes, listing the wheel and its stamp.
SUMS_NAME = "SHA256SUMS"
STAMP_SUFFIX = ".build-stamp.json"
STAMP_SCHEMA = 1
_STAMP_KEYS = ("schema", "wheel_name", "wheel_sha256", "commit", "built_at", "built_at_basis")


def published_stamp_name(wheel_name: str) -> str:
    """The asset name of the stamp published for ``wheel_name``."""
    return wheel_name + STAMP_SUFFIX


def published_stamp_document(
    wheel_name: str, wheel_sha256: str, commit: str, built_at: str,
) -> Dict[str, Any]:
    """The stamp's contents: which wheel it describes, and its build identity."""
    return {
        "schema": STAMP_SCHEMA,
        "wheel_name": wheel_name,
        "wheel_sha256": wheel_sha256,
        "commit": commit,
        "built_at": built_at,
        "built_at_basis": BASIS_STAMP,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_published_stamp(
    output: Path, wheel: Path, commit: str, built_at: str,
) -> Path:
    """Write the stamp beside ``wheel`` and a ``SHA256SUMS`` covering both.

    Release tooling only; nothing here uploads anything.
    """
    output = Path(output)
    wheel = Path(wheel)
    document = published_stamp_document(wheel.name, _sha256(wheel), commit, built_at)
    stamp = output / published_stamp_name(wheel.name)
    # Bytes, never text mode: on Windows text mode writes CRLF, and a Linux
    # `sha256sum -c` then looks for "<name>\r" (wave 6 deploy, LESSONS 18). The
    # digest listed below is of exactly these bytes.
    stamp.write_bytes((json.dumps(document, sort_keys=True) + "\n").encode("utf-8"))
    sums = output / SUMS_NAME
    temporary = sums.with_suffix(".tmp")
    temporary.write_bytes(
        "".join(
            "{} *{}\n".format(_sha256(path), path.name) for path in (wheel, stamp)
        ).encode("utf-8")
    )
    os.replace(temporary, sums)
    return stamp


def parse_sums(body: str) -> Dict[str, str]:
    """``SHA256SUMS`` as ``{filename: digest}``, in ``sha256sum``'s two formats."""
    listed: Dict[str, str] = {}
    for line in body.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            listed.setdefault(parts[1].lstrip("*").strip(), parts[0].strip().lower())
    return listed


def verify_published_stamp(
    body: Optional[bytes],
    *,
    wheel_name: str,
    wheel_sha256: str,
    sums: Mapping[str, str],
) -> Tuple[Optional[Dict[str, str]], str]:
    """The stamp's build identity if the release authenticates it, else why not.

    Returns ``(identity, "")`` with ``built_at``, ``built_at_basis`` and
    ``commit`` only when every check holds; otherwise ``(None, reason)``.
    """
    name = published_stamp_name(wheel_name)
    listed = sums.get(name)
    if listed is None:
        return None, "its SHA256SUMS does not list the stamp"
    if body is None:
        return None, "the stamp could not be downloaded"
    if hashlib.sha256(body).hexdigest() != listed:
        return None, "the stamp does not match the digest its SHA256SUMS lists"
    try:
        document = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None, "the stamp is not readable JSON"
    if not isinstance(document, dict) or sorted(document) != sorted(_STAMP_KEYS):
        return None, "the stamp does not have the published stamp's fields"
    if document["schema"] != STAMP_SCHEMA:
        return None, "the stamp is of an unknown version"
    if document["wheel_name"] != wheel_name or document["wheel_sha256"] != wheel_sha256:
        return None, "the stamp describes a different wheel"
    if document["built_at_basis"] != BASIS_STAMP or parse_time(document["built_at"]) is None:
        return None, "the stamp does not state a stamped build time"
    commit = document["commit"]
    if not isinstance(commit, str):
        return None, "the stamp's commit is not text"
    return {
        "built_at": document["built_at"],
        "built_at_basis": BASIS_STAMP,
        "commit": commit,
    }, ""
