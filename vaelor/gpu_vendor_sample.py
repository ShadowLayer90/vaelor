"""The file a worker's GPU sampler leaves for its telemetry emitter, and how it is read.

VD-147 (owner decision D1; review B1, N-S9). The sampler (`worker_gpu_sampler`)
is an unprivileged service; the emitter that reads its file runs as root inside
Telegraf, once a second. A root process reading a file an unprivileged one
wrote is a place to be careful, so the read is fenced on every side:

* the directory must be a real directory, not a symlink, and not writable by
  group or others; its parent must belong to root;
* the file is opened with ``O_NOFOLLOW`` (a symlink is refused, not followed)
  and ``O_NONBLOCK`` (a FIFO cannot hang the emitter);
* it must be a regular file **owned by the owner of the directory** - systemd
  gives the runtime directory to the sampler's dynamic user, so a file anyone
  else dropped there is refused - and no larger than :data:`MAX_SAMPLE_BYTES`;
* what it holds is parsed as data, its status must be a known code, every
  value goes through the one bounds table, and a sample older than
  :data:`~vaelor.gpu_vendor_status.VENDOR_SAMPLE_MAX_AGE_SECONDS` is not used.

Anything that fails a check reads as "the sampler has stopped or is missing"
(`gpu_vendor_status.SAMPLER_MISSING`) with no values: a refused file is never
half-trusted.

Standard library only: this module rides in the worker's emitter bundle.
"""

from __future__ import annotations

import json
import os
import re
import stat
from typing import Any, Dict, Optional

from .gpu_vendor_status import (
    AGE_FIELD, CODES_WITH_READINGS, SAMPLER_MISSING, STATUS_FIELD,
    VENDOR_SAMPLE_MAX_AGE_SECONDS, known_code,
)
from .platforms.gpu_temperature import sensor_from_label
from .platforms.graphics_software import AMD_SMI_GPU_FIELDS
from .telemetry_bounds import finite_number, plausible, plausible_temperature_limit

#: Where systemd puts the sampler's ``RuntimeDirectory=vaelor-gpu``.
SAMPLE_DIRECTORY = "/run/vaelor-gpu"
SAMPLE_FILE_NAME = "gpu-vendor.json"
SAMPLE_PATH = SAMPLE_DIRECTORY + "/" + SAMPLE_FILE_NAME

#: The largest file the emitter will read. A sample is a few hundred bytes;
#: four kibibytes is generous, and anything larger is not a sample.
MAX_SAMPLE_BYTES = 4096

#: The fields a sample's ``values`` may carry: the one extraction table's.
_VALUE_FIELDS = frozenset(field for field, _reader in AMD_SMI_GPU_FIELDS)

#: The most sensors a sample's limits may name (review nit): a GPU has a few,
#: and a file listing more is not a GPU's limits.
MAX_LIMIT_SENSORS = 4

#: What a sensor's name may be (pass-3 review nit): a driver's short label -
#: letters, digits, spaces, dots, dashes and underscores, at most 32.
_SENSOR_NAME = re.compile(r"[A-Za-z0-9 ._-]{1,32}")

#: The two limit names a sensor's entry may carry.
_LIMIT_NAMES = ("warning_c", "critical_c")

def _missing() -> Dict[str, Any]:
    """What a refused or missing file reads as."""
    return {"status": SAMPLER_MISSING, "values": {}, "age_seconds": None, "limits": {}}


def parse_limits(raw: Any) -> Dict[str, Dict[str, float]]:
    """``{sensor: {warning_c, critical_c}}`` with only plausible limits kept.

    A limit outside the bounds table's range is dropped; a sensor left with no
    limit is dropped whole. A limit is never invented or defaulted.
    """
    limits: Dict[str, Dict[str, float]] = {}
    if not isinstance(raw, dict):
        return limits
    for sensor, entry in raw.items():
        if not isinstance(sensor, str) or not isinstance(entry, dict):
            continue
        kept = {
            name: value for name in _LIMIT_NAMES
            for value in (plausible_temperature_limit(entry.get(name)),)
            if value is not None
        }
        # A warning above the critical mark is not a pair of limits (review nit).
        if "warning_c" in kept and "critical_c" in kept and kept["warning_c"] > kept["critical_c"]:
            continue
        if kept and len(limits) < MAX_LIMIT_SENSORS and _SENSOR_NAME.fullmatch(sensor):
            # Named the way the controller names its own sensor (review nit):
            # printable, lower-case, bounded - the file is data, not a label.
            limits[sensor_from_label(sensor)] = kept
    return limits


def parse_sample(text: Any, now: float) -> Dict[str, Any]:
    """A sample file's contents as ``{status, values, age_seconds, limits}``.

    ``text`` is what was read from the file (or from ``cat`` over the
    transport, for the controller's reconcile). Anything that is not a sample -
    not JSON, an unknown status code, no readable time - is
    :data:`~vaelor.gpu_vendor_status.SAMPLER_MISSING` with no values. A sample
    older than the freshness bound, or stamped in the future, keeps its limits
    (they are static) but gives no values and reads as missing too.

    Never raises (review B2): the file is written by an unprivileged process
    and read by the root emitter, so ANY failure while reading it - a number
    too large for a float, NaN, a document nested past the parser's limit - is
    the missing-sampler answer, never an exception that costs the worker its
    other telemetry.
    """
    try:
        return _parse(text, now)
    except Exception:  # noqa: BLE001 - see above: a refused file, never a crash
        return _missing()


def _parse(text: Any, now: float) -> Dict[str, Any]:
    try:
        data = json.loads(text) if isinstance(text, str) else None
    except (ValueError, RecursionError):
        data = None
    if not isinstance(data, dict):
        return _missing()
    status = known_code(data.get("status"))
    sampled_at = finite_number(data.get("sampled_at"))
    if status is None or sampled_at is None:
        return _missing()
    limits = parse_limits(data.get("limits"))
    age = float(now) - sampled_at
    if not 0 <= age <= VENDOR_SAMPLE_MAX_AGE_SECONDS:
        return {"status": SAMPLER_MISSING, "values": {}, "age_seconds": None, "limits": limits}
    values: Dict[str, float] = {}
    raw_values = data.get("values")
    if status in CODES_WITH_READINGS and isinstance(raw_values, dict):
        for field in _VALUE_FIELDS:
            value = finite_number(raw_values.get(field))
            if value is not None and plausible(field, value):
                values[field] = value
    return {"status": status, "values": values, "age_seconds": round(age, 3), "limits": limits}


def read_sample_file(
    directory: str = SAMPLE_DIRECTORY, *, now: float, fs: Any = os,
) -> Dict[str, Any]:
    """Read the sampler's file through every check, or report it missing.

    ``fs`` is the ``os`` module; a test hands in a stand-in with the same
    calls. Never raises: every refusal is the missing-sampler answer.
    """
    no_follow = getattr(fs, "O_NOFOLLOW", None)
    if no_follow is None:
        # No way to refuse a symlink on this platform: do not read at all.
        return _missing()
    try:
        directory_stat = fs.lstat(directory)
        parent_stat = fs.lstat(os.path.dirname(directory.rstrip("/")) or "/")
    except OSError:
        return _missing()
    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or directory_stat.st_mode & 0o022
        or parent_stat.st_uid != 0
    ):
        return _missing()
    flags = fs.O_RDONLY | no_follow | getattr(fs, "O_NONBLOCK", 0)
    try:
        descriptor = fs.open(directory.rstrip("/") + "/" + SAMPLE_FILE_NAME, flags)
    except OSError:
        return _missing()
    try:
        file_stat = fs.fstat(descriptor)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_uid != directory_stat.st_uid
            or file_stat.st_size > MAX_SAMPLE_BYTES
        ):
            return _missing()
        raw = fs.read(descriptor, MAX_SAMPLE_BYTES + 1)
    except OSError:
        return _missing()
    finally:
        fs.close(descriptor)
    if len(raw) > MAX_SAMPLE_BYTES:
        return _missing()
    return parse_sample(raw.decode("utf-8", "replace"), now)


def vendor_fields(sample: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """The telemetry fields an emitter stores from a read sample.

    The status code always; the readings and their age only when the sample
    carried them. Floats throughout: the store keeps one numeric type per field.
    """
    sample = sample or _missing()
    fields: Dict[str, float] = {STATUS_FIELD: float(sample["status"])}
    fields.update(sample["values"])
    if sample.get("age_seconds") is not None:
        fields[AGE_FIELD] = float(sample["age_seconds"])
    return fields
