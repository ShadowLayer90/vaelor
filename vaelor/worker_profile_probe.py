"""Read a worker's profile as it stands: one ``python3 -c`` program over SSH (VD-194 P1).

The measured side of the profile. One program runs on the worker, elevated,
through the existing `SshTransport.run` - ``python3`` is already on the SSH
allowlist, so nothing is widened (LESSONS 18) - and reports, for every
manifest path, its digest, owner, group and mode, and for every unit its
``systemctl show`` state; and, for each of the full appliance's units (named
from `vaelor/appliance_units/` in the wheel this controller runs, not from
``deploy/``, which a controller does not have), whether its file is present
and how systemd holds it.

**Nothing secret leaves the worker.** The program returns digests, modes and
names only. The Telegraf config carries the ingest key, so the program
replaces the node id and the key line with the placeholders
`worker_profile` renders, then hashes - the key is never read back, and the
frame around it is derived from the renderer (`worker_profile.key_line_frame`).

**Unread is its own answer** (LESSONS 8). A path that is not there is
``{"present": false}``; one that could not be read carries ``error``; a
``systemctl`` that could not run gives ``state_error``; and output the parser
cannot read is a reading with ``ok: false``. None of those can compare as
"absent" or "matches" (`worker_profile_compare`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import worker_profile as wp
from .platforms.accelerators import AMD_SMI_PACKAGE_PREFIX, ROCM_AMD_SMI, ROCM_CORE_AMD_SMI

#: How long the probe may take: it hashes the ~250 MB Telegraf binary.
PROBE_TIMEOUT_SECONDS = 90

#: The ``systemctl show`` properties read for every unit.
UNIT_PROPERTIES = ("Id", "LoadState", "ActiveState", "SubState", "UnitFileState",
                   "NeedDaemonReload", "FragmentPath")


def appliance_unit_names() -> List[str]:
    """The full appliance's unit names, from the wheel this controller runs.

    `appliance_upgrade` resolves the installed package's ``appliance_units``
    directory the same way for the in-box unit refresh; this reads the same
    directory, so the two cannot disagree about which units are the appliance.
    An empty answer means the list could not be read, which the reading
    reports as unread - never as "no appliance installed".
    """
    from .appliance_upgrade import _bundled_units_dir

    try:
        return sorted(path.name for path in Path(_bundled_units_dir()).glob("vaelor-*.service"))
    except OSError:
        return []


def probe_spec(node_id: str, unit_names: Iterable[str]) -> Dict[str, Any]:
    """What the program is asked to read, as one JSON argument."""
    prefix, suffix = wp.key_line_frame()
    items = []
    for component in wp.MANIFEST:
        item = {"id": component.id, "kind": component.kind, "path": component.path}
        if component.unit:
            item["unit"] = component.unit
        if component.id == "telegraf-config":
            item["redact"] = True
        items.append(item)
    return {
        "node": str(node_id), "node_placeholder": wp.PLACEHOLDER_NODE,
        "key_prefix": prefix, "key_suffix": suffix, "key_placeholder": wp.PLACEHOLDER_KEY,
        "items": items, "units": list(unit_names),
        "packages": list(wp.AMD_SMI_PACKAGES), "amd_source": wp.AMD_SOURCE_PATH,
        "amd_keyring": wp.AMD_KEYRING_PATH, "rocm_amd_smi": ROCM_AMD_SMI,
        "core_amd_smi": ROCM_CORE_AMD_SMI, "smi_prefix": AMD_SMI_PACKAGE_PREFIX,
        "wmi_guid": wp.WMI_SENSOR_GUID, "sys_root": "/sys", "os_release": "/etc/os-release",
        "systemctl": "systemctl", "dpkg_query": "dpkg-query",
    }


#: The program. Standard library only, Python 3.8+ (the existing worker runs
#: 3.14). It prints one JSON document and exits 0 whatever it found; every
#: failure is recorded per item rather than aborting the reading.
PROBE_PROGRAM = r'''
import errno, hashlib, json, os, shutil, stat, subprocess, sys
spec = json.loads(sys.argv[1])
try:
    import pwd, grp
except ImportError:
    pwd = grp = None
class NotRegular(OSError):
    pass
def why(error):
    if isinstance(error, NotRegular):
        return "not a regular file"
    return (errno.errorcode.get(getattr(error, "errno", None) or 0) or type(error).__name__)[:40]
def name_of(table, ident):
    if table is None:
        return None
    try:
        return (table.getpwuid(ident).pw_name if table is pwd else table.getgrgid(ident).gr_name)
    except (KeyError, OSError):
        return str(ident)
def acl_of(path):
    getter = getattr(os, "getxattr", None)
    if getter is None:
        return None
    try:
        getter(path, "system.posix_acl_access", follow_symlinks=False)
        return True
    except OSError as error:
        if error.errno in (errno.ENODATA, getattr(errno, "ENOTSUP", -1), getattr(errno, "EOPNOTSUPP", -1)):
            return False
        return None
def meta_of(info):
    kind = ("link" if stat.S_ISLNK(info.st_mode) else "dir" if stat.S_ISDIR(info.st_mode)
            else "file" if stat.S_ISREG(info.st_mode) else "other")
    return {"present": True, "type": kind, "mode": "%04o" % stat.S_IMODE(info.st_mode),
            "owner": name_of(pwd, info.st_uid), "group": name_of(grp, info.st_gid)}
def stat_of(path):
    try:
        return meta_of(os.lstat(path))
    except FileNotFoundError:
        return {"present": False}
    except OSError as error:
        return {"error": why(error)}
def open_nofollow(path):
    # VD-185: never follow a link swapped in after the lstat (O_NOFOLLOW), never
    # block on a pipe swapped in (O_NONBLOCK), and read only a regular file -
    # the descriptor's own fstat says so, and the metadata comes from it too.
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
             | getattr(os, "O_BINARY", 0))
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise NotRegular(errno.EINVAL, "not a regular file")
        return os.fdopen(fd, "rb"), info
    except BaseException:
        os.close(fd)
        raise
def digest_of(handle, redact):
    if not redact:
        total = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            total.update(chunk)
        return {"sha256": total.hexdigest()}
    text = handle.read(1 << 20).decode("utf-8", "replace")
    if spec["node"]:
        text = text.replace(spec["node"], spec["node_placeholder"])
    lines = []
    for line in text.split("\n"):
        if (line.startswith(spec["key_prefix"]) and line.endswith(spec["key_suffix"])
                and len(line) >= len(spec["key_prefix"]) + len(spec["key_suffix"])):
            line = spec["key_prefix"] + spec["key_placeholder"] + spec["key_suffix"]
        lines.append(line)
    return {"sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()}
def text_digest(handle):
    text = handle.read(1 << 20).decode("utf-8", "replace")
    return {"text_sha256": hashlib.sha256(text.strip().encode("utf-8")).hexdigest(),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
def file_item(path, redact=False, text=False):
    record = stat_of(path)
    if record.get("type") == "dir":
        record["acl"] = acl_of(path)
    if record.get("type") != "file":
        return record
    try:
        handle, info = open_nofollow(path)
        with handle:
            record = meta_of(info)
            record.update(text_digest(handle) if text else digest_of(handle, redact))
    except OSError as error:
        return {"present": True, "type": "file", "error": why(error)}
    return record
def run(argv):
    try:
        done = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
                              env=dict(os.environ, LC_ALL="C"))
    except (OSError, subprocess.SubprocessError) as error:
        return None, why(error)
    return done, ""
PROPS = __PROPS__
def unit_states(names):
    if not names:
        return {}, ""
    done, error = run([spec["systemctl"], "show", "--no-pager", "--property=" + ",".join(PROPS)] + list(names))
    if done is None:
        return {}, error
    if done.returncode != 0:
        return {}, "systemctl exited %d" % done.returncode
    states, current = {}, {}
    for line in done.stdout.decode("utf-8", "replace").split("\n") + [""]:
        if not line.strip():
            if current.get("Id"):
                states[current["Id"]] = current
            current = {}
            continue
        key, _, value = line.partition("=")
        if key in PROPS:
            current[key] = value.strip()[:200]
    return states, ""
def facts():
    found = {"architecture": os.uname().machine if hasattr(os, "uname") else ""}
    try:
        with open(spec["os_release"]) as handle:
            for line in handle:
                key, _, value = line.strip().partition("=")
                if key in ("ID", "VERSION_ID"):
                    found["os_id" if key == "ID" else "os_version"] = value.strip().strip("\"'").lower()
    except OSError:
        pass
    root = spec["sys_root"]
    try:
        names = os.listdir(os.path.join(root, "bus", "wmi", "devices"))
        found["wmi_guid"] = any(spec["wmi_guid"].lower() in name.lower() for name in names)
    except FileNotFoundError:
        found["wmi_guid"] = False if os.path.isdir(os.path.join(root, "bus")) else None
    except OSError:
        found["wmi_guid"] = None
    try:
        devices = os.listdir(os.path.join(root, "bus", "pci", "devices"))
    except OSError:
        found["amdgpu_bound"] = None
    else:
        found["amdgpu_bound"] = any(
            os.path.islink(os.path.join(root, "bus", "pci", "devices", device, "driver"))
            and os.path.basename(os.path.realpath(
                os.path.join(root, "bus", "pci", "devices", device, "driver"))) == "amdgpu"
            for device in devices)
    return found
def package_item(item):
    record = file_item(item["path"])
    done, error = run([spec["dpkg_query"], "-W", "-f=${Package}\t${Version}\t${db:Status-Abbrev}\n"]
                      + list(spec["packages"]))
    if done is None:
        record["package_error"] = error
    else:
        installed = {}
        for line in done.stdout.decode("utf-8", "replace").splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[2].strip().endswith("i"):
                installed[parts[0]] = {"version": parts[1], "held": parts[2].startswith("h")}
        noise = [line for line in done.stderr.decode("utf-8", "replace").splitlines()
                 if line.strip() and "no packages found" not in line.lower()]
        if done.returncode not in (0, 1) or noise:
            record["package_error"] = "dpkg-query exited %d" % done.returncode
        record["packages"] = installed
    source = file_item(spec["amd_source"], text=True)
    record["source"] = {key: source.get(key) for key in ("present", "text_sha256", "error") if key in source}
    keyring = file_item(spec["amd_keyring"])
    record["keyring"] = {key: keyring.get(key) for key in ("present", "sha256", "error") if key in keyring}
    # The resolver's own order (accelerators.resolve_amd_smi): the ROCm build,
    # then AMD's package alone at its core-series path, then PATH.
    series = sorted((name[len(spec["smi_prefix"]):] for name in record.get("packages") or {}
                     if name.startswith(spec["smi_prefix"])),
                    key=lambda value: [int(part) for part in value.split(".") if part.isdigit()])
    core = spec["core_amd_smi"].replace("{series}", series[-1]) if series else ""
    if os.access(spec["rocm_amd_smi"], os.X_OK):
        chosen = spec["rocm_amd_smi"]
    elif core and os.access(core, os.X_OK):
        chosen = core
    else:
        chosen = shutil.which("amd-smi")
    record["binary"] = chosen or ""
    return record
def marker_item(path):
    record = file_item(path)
    if record.get("type") == "file":
        record.pop("sha256", None)
        try:
            handle, _ = open_nofollow(path)
            with handle:
                value = str(json.loads(handle.read(1 << 16).decode("utf-8", "replace")).get("profile_digest", ""))
            record["profile_digest"] = value if len(value) == 64 else ""
        except (OSError, ValueError, AttributeError):
            record["profile_digest"] = ""
    return record
items = {}
unit_names = [item["unit"] for item in spec["items"] if item.get("unit")]
states, state_error = unit_states(unit_names)
for item in spec["items"]:
    kind, path = item["kind"], item["path"]
    if kind == "prerequisite":
        record = stat_of(path)
        if record.get("present"):
            record = {"present": os.access(path, os.X_OK), "type": record.get("type")}
    elif kind == "package":
        record = package_item(item)
    elif kind == "marker":
        record = marker_item(path)
    else:
        record = file_item(path, redact=bool(item.get("redact")), text=kind == "unit")
    if item.get("unit"):
        if state_error:
            record["state_error"] = state_error
        elif item["unit"] in states:
            record["state"] = states[item["unit"]]
        else:
            record["state_error"] = "not reported"
    items[item["id"]] = record
appliance, appliance_error = unit_states(spec["units"])
missing = [name for name in spec["units"] if name not in appliance]
if not appliance_error and missing:
    appliance_error = "not reported: %d units" % len(missing)
print(json.dumps({"version": 1, "facts": facts(), "items": items,
                  "appliance": {"units": appliance, "error": appliance_error}}))
'''.replace("__PROPS__", repr(UNIT_PROPERTIES))


def probe_argv(spec: Dict[str, Any]) -> List[str]:
    """The one argv the probe runs: ``python3 -c <program> <spec>``."""
    return ["python3", "-c", PROBE_PROGRAM, json.dumps(spec, separators=(",", ":"))]


def read_profile(transport, node_id: str, unit_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run the probe on a worker and return its parsed reading.

    Raises the transport's error when the worker could not be asked; the caller
    records that as an attempt, beside the last good reading.
    """
    names = appliance_unit_names() if unit_names is None else list(unit_names)
    raw = transport.run(probe_argv(probe_spec(node_id, names)), sudo=True,
                        timeout=PROBE_TIMEOUT_SECONDS)
    reading = parse_probe_output(raw)
    if not names:
        reading["appliance"] = {"units": {}, "error": (
            "this controller could not read its own list of appliance services")}
    return reading


# -- parsing: keep only known fields, bounded --------------------------------

_HEX = frozenset("0123456789abcdef")
_STRING_FIELDS = {"type": 8, "mode": 6, "owner": 64, "group": 64, "error": 64,
                  "state_error": 64, "package_error": 64, "binary": 200}
_DIGEST_FIELDS = ("sha256", "text_sha256", "profile_digest")
_FACT_FIELDS = ("architecture", "os_id", "os_version")
_TRI_FACTS = ("wmi_guid", "amdgpu_bound")


def _text(value: Any, limit: int) -> str:
    return str(value)[:limit] if isinstance(value, (str, int)) else ""


def _tri(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _states(raw: Any) -> Dict[str, Dict[str, str]]:
    if not isinstance(raw, dict):
        return {}
    return {
        _text(name, 120): {key: _text(value, 200) for key, value in state.items()
                           if key in UNIT_PROPERTIES}
        for name, state in raw.items() if isinstance(state, dict)
    }


def _item(raw: Dict[str, Any]) -> Dict[str, Any]:
    item: Dict[str, Any] = {}
    for key in ("present", "acl"):
        if key in raw:
            item[key] = _tri(raw[key])
    for key, limit in _STRING_FIELDS.items():
        if key in raw:
            item[key] = _text(raw[key], limit)
    for key in _DIGEST_FIELDS:
        if key in raw:
            value = _text(raw[key], 64)
            item[key] = value if len(value) == 64 and set(value) <= _HEX else ""
    if isinstance(raw.get("state"), dict):
        item["state"] = _states({"unit": raw["state"]}).get("unit", {})
    if isinstance(raw.get("packages"), dict):
        item["packages"] = {
            _text(name, 80): {"version": _text(entry.get("version"), 40), "held": _tri(entry.get("held"))}
            for name, entry in raw["packages"].items() if isinstance(entry, dict)
        }
    for key in ("source", "keyring"):
        if isinstance(raw.get(key), dict):
            item[key] = _item(raw[key])
    return item


def unread_reading(reason: str) -> Dict[str, Any]:
    """A reading that holds nothing but why it holds nothing."""
    return {"ok": False, "error": reason, "facts": {}, "items": {},
            "appliance": {"units": {}, "error": reason}}


def parse_probe_output(raw: Any) -> Dict[str, Any]:
    """The program's output, validated; anything else is an unread reading."""
    try:
        document = json.loads(str(raw or "").strip().splitlines()[-1]) if str(raw or "").strip() else None
    except (ValueError, IndexError):
        document = None
    items = document.get("items") if isinstance(document, dict) else None
    if not isinstance(items, dict) or any(not isinstance(value, dict) for value in items.values()):
        return unread_reading("the machine's answer could not be read")
    facts_raw = document.get("facts") if isinstance(document.get("facts"), dict) else {}
    facts = {key: _text(facts_raw.get(key, ""), 64) for key in _FACT_FIELDS}
    facts.update({key: _tri(facts_raw.get(key)) for key in _TRI_FACTS})
    appliance_raw = document.get("appliance") if isinstance(document.get("appliance"), dict) else {}
    appliance = {"units": _states(appliance_raw.get("units")),
                 "error": _text(appliance_raw.get("error", ""), 120)}
    if not isinstance(document.get("appliance"), dict):
        appliance["error"] = "the machine did not report its appliance services"
    return {"ok": True, "error": "", "facts": facts,
            "items": {_text(key, 40): _item(value) for key, value in items.items()},
            "appliance": appliance}
