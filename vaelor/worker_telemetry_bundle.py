"""Package the worker telemetry emitter as a self-contained zipapp (Phase E2b).

The controller ships the emitter to a lean worker as a single ``.pyz`` the
worker's system ``python3`` runs directly (``python3 emitter.pyz --node <id>``).
A zipapp is used rather than a flat file because the reader spans the
``vaelor.platforms`` package and its relative imports; the zip preserves that
package layout, so the exact canonical reader runs on the worker with no
install, no venv, and no ``sys.path`` surgery.

:data:`EMITTER_MODULES` is the module closure the bundle carries — the transitive
import closure of :data:`EMITTER_ENTRY`, every member stdlib-only so nothing but
the standard library is needed on the worker. A test asserts this list is
exactly what importing the entry point loads and that it pulls no third party, so
a future import that reached a dependency (or a new module) fails the suite rather
than shipping a broken bundle.

The archive is built **deterministically** — fixed member order and timestamps —
so identical source yields identical bytes and therefore a stable content hash.
That is what lets the runtime skip re-shipping the bundle when the worker already
holds the current one (:func:`bundle_digest`), the same content-hashed-artifact
discipline `gpu_pool_runtime` uses for its pull program.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path
from typing import Dict, FrozenSet

#: What ``python3 emitter.pyz`` runs.
EMITTER_ENTRY = "vaelor.worker_telemetry_emitter"

#: The transitive import-time closure of :data:`EMITTER_ENTRY`. Every module is
#: stdlib-only in what it imports at load time, so the bundle needs no third
#: party on the worker. Kept as an explicit manifest (not a live
#: ``sys.modules`` walk) so the bundle's contents are reviewable and a test can
#: assert the real closure has not drifted from it.
EMITTER_MODULES = frozenset({
    "vaelor",
    "vaelor.boot_forensics",
    "vaelor.cluster_placement",
    "vaelor.gpu_vendor_sample",
    "vaelor.gpu_vendor_status",
    "vaelor.linux_sensors",
    "vaelor.linux_telemetry",
    "vaelor.memory_ecc",
    "vaelor.package_power",
    "vaelor.platforms",
    "vaelor.platforms.accelerators",
    "vaelor.platforms.base",
    "vaelor.platforms.gpu_telemetry",
    "vaelor.platforms.gpu_temperature",
    "vaelor.platforms.graphics_software",
    "vaelor.platforms.raspberry_pi",
    "vaelor.platforms.workstation",
    "vaelor.runtime_paths",
    "vaelor.telemetry_bounds",
    "vaelor.telemetry_flatten",
    "vaelor.telemetry_ingest",
    "vaelor.telemetry_store",
    "vaelor.version",
    "vaelor.wmi_sensors",
    "vaelor.worker_telemetry_emitter",
})

#: What ``python3 vaelor-gpu-sampler.pyz`` runs: the non-root GPU vendor sampler
#: (VD-147). Its own entry and its own bundle, so the emitter's bundle - and
#: its guarantee that it never spawns a subprocess - are untouched by it.
SAMPLER_ENTRY = "vaelor.worker_gpu_sampler"

#: The transitive import-time closure of :data:`SAMPLER_ENTRY`, standard library
#: only, asserted by the same closure test as the emitter's.
SAMPLER_MODULES = frozenset({
    "vaelor",
    "vaelor.boot_forensics",
    "vaelor.gpu_vendor_status",
    "vaelor.platforms",
    "vaelor.platforms.accelerators",
    "vaelor.platforms.base",
    "vaelor.platforms.gpu_telemetry",
    "vaelor.platforms.gpu_temperature",
    "vaelor.platforms.graphics_software",
    "vaelor.platforms.raspberry_pi",
    "vaelor.platforms.workstation",
    "vaelor.runtime_paths",
    "vaelor.telemetry_bounds",
    "vaelor.version",
    "vaelor.wmi_sensors",
    "vaelor.worker_gpu_sampler",
})

#: The zipapp's entry point. Kept minimal — it only bridges ``python3 x.pyz`` to
#: the entry module's ``main`` — because everything real is in the bundled package.
_MAIN_TEMPLATE = (
    "import sys\n"
    "from {entry} import main\n"
    "sys.exit(main())\n"
)

#: A fixed timestamp for every archive member, so the bytes depend only on the
#: source, never on when the bundle was built. (1980-01-01 is zip's epoch.)
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)

#: The zip "made by" system: Unix, fixed rather than the build platform's.
_UNIX_CREATOR = 3

#: One spelling of the "no source" failure, shared by the two places that
#: resolve a module's file, so the message cannot drift between them.
_NO_SOURCE = "Cannot locate source for {}."


def _archive_name(module: str) -> str:
    """The path a module's source takes inside the zip.

    A package's source is its ``__init__.py``; a plain module's is
    ``<name>.py``. Derived from the spec's own origin so a package and a
    same-named module are never confused.
    """
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin or spec.origin == "built-in":
        raise ModuleNotFoundError(_NO_SOURCE.format(module))
    parts = module.split(".")
    if Path(spec.origin).name == "__init__.py":
        return "/".join(parts) + "/__init__.py"
    return "/".join(parts) + ".py"


def _sources(entry: str, modules: FrozenSet[str]) -> Dict[str, bytes]:
    """Every bundled member's archive path mapped to its source bytes."""
    members: Dict[str, bytes] = {}
    for module in modules:
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin:
            raise ModuleNotFoundError(_NO_SOURCE.format(module))
        members[_archive_name(module)] = Path(spec.origin).read_bytes()
    members["__main__.py"] = _MAIN_TEMPLATE.format(entry=entry).encode("utf-8")
    return members


def build_zipapp_bytes(
    entry: str = EMITTER_ENTRY, modules: FrozenSet[str] = EMITTER_MODULES,
) -> bytes:
    """A zipapp (the emitter's unless told otherwise) as deterministic bytes.

    Members are written in sorted order with a fixed timestamp, so two builds of
    the same source produce byte-identical archives and therefore the same
    digest. **They are stored, not deflated** (VD-194 P1 review P1-1): a
    deflated member's bytes are the compressor's, so the same source gave a
    different digest under Windows zlib-ng, CPython 3.11 zlib and the
    controller's Ubuntu build, and a zlib update on the controller would have
    read every worker as different and re-shipped every bundle. The creating
    system and the mode are fixed for the same reason (``ZipInfo`` takes the
    build platform's by default). A shebang line precedes the zip so the file is directly executable,
    matching what :func:`zipapp.create_archive` writes; it runs under the
    worker's ``python3`` either way (``python3 emitter.pyz``).
    """
    buffer = io.BytesIO()
    buffer.write(b"#!/usr/bin/env python3\n")
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(_sources(entry, modules).items()):
            info = zipfile.ZipInfo(name, date_time=_FIXED_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = _UNIX_CREATOR
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return buffer.getvalue()


#: Digests already worked out, keyed by the bundle and the size and change time
#: of every source it holds: a repair pass asked for the same digest four
#: times per worker and rebuilt the zip each time (review nit). Any change to
#: a source changes the key.
_DIGESTS: Dict[tuple, str] = {}


def _digest(entry: str, modules: FrozenSet[str]) -> str:
    stamps = []
    for module in sorted(modules):
        spec = importlib.util.find_spec(module)
        origin = Path(spec.origin) if spec is not None and spec.origin else None
        stat = origin.stat() if origin is not None and origin.exists() else None
        stamps.append((module, stat.st_size if stat else -1, stat.st_mtime_ns if stat else -1))
    key = (entry, tuple(stamps))
    if key not in _DIGESTS:
        _DIGESTS[key] = hashlib.sha256(build_zipapp_bytes(entry, modules)).hexdigest()
    return _DIGESTS[key]


def bundle_digest() -> str:
    """The sha256 hex of the current emitter bundle, for content-hashed skip."""
    return _digest(EMITTER_ENTRY, EMITTER_MODULES)


def build_sampler_bytes() -> bytes:
    """The GPU sampler zipapp as deterministic bytes."""
    return build_zipapp_bytes(SAMPLER_ENTRY, SAMPLER_MODULES)


def sampler_bundle_digest() -> str:
    """The sha256 hex of the current sampler bundle."""
    return _digest(SAMPLER_ENTRY, SAMPLER_MODULES)


def write_zipapp(destination: str) -> str:
    """Write the emitter zipapp to ``destination`` and return the path."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_zipapp_bytes())
    return str(path)
