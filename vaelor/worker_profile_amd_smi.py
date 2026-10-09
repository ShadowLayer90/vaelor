"""AMD's amd-smi on a worker: alone, pinned, from AMD's repository (VD-194 P2).

The profile's one package component. The owner decided on 2026-10-04 that a
worker gets AMD's own ``amd-smi`` package and nothing else of the ROCm stack,
because only AMD's build reports the GPU power and graphics temperature the
sampler reads (Ubuntu's build does not, measured on the worker). Later the
same day the owner answered what happens when AMD stops publishing the pinned
version: "update to the latest version available" - so the newest published
version is installed instead, recorded, and said on the card.

What this module lays down, every step over the enrolled SSH login with tokens
the allowlist already has (LESSONS 18):

* **AMD's signing key**, shipped inside the wheel
  (`worker_profile_keys/amdrocm-keyring.gpg`, the installer's own copy - a test
  pins the two byte-identical), staged privately (VD-172) and checked ON the
  worker before it is placed: exactly one primary key, and the pinned
  fingerprint is that key's primary fingerprint, never a subkey's. The same
  parser runs on both sides (:data:`_PARSER_SOURCE`).
* **The source and key at the installer's own paths**, so a worker the full
  installer set up has its source *adopted* (compared byte for byte), never
  duplicated - two entries for one repository with different ``Signed-By``
  values stop apt working at all. A different source there is refused.
* **The two packages by exact version, then held**, and an apt preferences
  file that keeps everything else from AMD's repository out of upgrades.
* **A record of what is installed**: the pin, the version and series actually
  installed, the binary (from ``dpkg -L``, not a constant), the tool's own
  version line, and whether it is the newest-available fallback, with why.

A failed apply removes the source and key only if this apply added them
(LESSONS 1: a cleanup must not take away what was there before). Packages it
installed stay installed, unheld, and the outcome says so. Nothing here names
a serving unit or Docker (guard G6).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from . import worker_profile as wp
from .platforms.accelerators import AMD_SMI_PACKAGE_PREFIX
from .ssh_transport import SshTransportError, channel_drops, machine_failures

#: The keyring the wheel carries; the installer's ``deploy/amdrocm-keyring.gpg``
#: is the other copy of the same bytes (guard G10).
BUNDLED_KEYRING = Path(__file__).resolve().parent / "worker_profile_keys" / "amdrocm-keyring.gpg"
KEYRING_DIRECTORY = "/etc/apt/keyrings"
SYSDEPS_PACKAGE_PREFIX = "amdrocm-sysdeps"
LOGGER = logging.getLogger(__name__)

#: How long apt may take over SSH: an update reads every source, an install
#: downloads about 96 MiB.
APT_UPDATE_SECONDS = 300
APT_INSTALL_SECONDS = 900
#: How much longer the transport waits than apt itself may run, so apt's own
#: cap always ends the install first and the job never gives up on an apt that
#: is still running on the worker (review round 2, LESSONS 6).
APT_TRANSPORT_MARGIN_SECONDS = 120
#: Every other question the apt program answers is quick.
APT_QUESTION_SECONDS = 300

#: Words the component's failure and its record say. Fixed sentences: what apt
#: or the worker printed goes to the log, never in front of an operator.
KEY_DID_NOT_VERIFY = "AMD's signing key did not verify"
SOURCE_NOT_OURS = ("AMD's package source on this machine is not the one Vaelor uses, so it was "
                   "left as it is and amd-smi was not installed")
NO_VERSION_PUBLISHED = "AMD publishes no version of amd-smi this machine can install"
INSTALL_FAILED = "apt could not install AMD's amd-smi packages"
HOLD_FAILED = "apt could not hold AMD's amd-smi packages at their version"
#: Why a version other than the pin was installed; the card says it (design §2.5).
FALLBACK_REASON = "the pinned {pin} is no longer published by AMD"
UNMET_REASON = "the pinned {pin} can no longer be installed from AMD's repository"
#: Said whenever this update installed the packages and then failed (review S3).
PACKAGES_STAY = "AMD's amd-smi packages this update installed stay installed and are not held"

#: The OpenPGP packet walk, one source for both sides: the controller checks
#: the bundled key with it before it leaves, and the worker program checks the
#: staged copy with the same text. A v4 fingerprint is SHA-1 over 0x99, the
#: two-byte body length and the public-key packet body (RFC 4880 12.2).
_PARSER_SOURCE = r'''
def openpgp_keys(data):
    import hashlib
    primaries, subkeys, index = [], [], 0
    while index < len(data):
        head = data[index]
        index += 1
        if not head & 0x80:
            raise ValueError("not an OpenPGP packet")
        if head & 0x40:
            tag, first = head & 0x3F, data[index]
            index += 1
            if first < 192:
                length = first
            elif first < 224:
                length = ((first - 192) << 8) + data[index] + 192
                index += 1
            elif first == 255:
                length = int.from_bytes(data[index:index + 4], "big")
                index += 4
            else:
                raise ValueError("partial packet lengths are not a keyring")
        else:
            tag, kind = (head >> 2) & 0x0F, head & 0x03
            if kind == 3:
                raise ValueError("indeterminate packet length")
            width = (1, 2, 4)[kind]
            length = int.from_bytes(data[index:index + width], "big")
            index += width
        body = data[index:index + length]
        if len(body) != length:
            raise ValueError("truncated packet")
        index += length
        if tag in (6, 14):
            if not body or body[0] != 4:
                raise ValueError("only version 4 keys are read")
            print_ = hashlib.sha1(b"\x99" + length.to_bytes(2, "big") + body).hexdigest().upper()
            (primaries if tag == 6 else subkeys).append(print_)
    return primaries, subkeys
'''
_NAMESPACE: Dict[str, Any] = {}
exec(compile(_PARSER_SOURCE, "<openpgp keys>", "exec"), _NAMESPACE)  # noqa: S102 - a module constant
openpgp_keys: Callable[[bytes], Tuple[List[str], List[str]]] = _NAMESPACE["openpgp_keys"]


def key_verdict(primaries: List[str], subkeys: List[str]) -> str:
    """``""`` when the keyring is exactly AMD's pinned key, else why not (design §2.5)."""
    if len(primaries) != 1:
        return "{}: the keyring holds {} primary keys, not one".format(KEY_DID_NOT_VERIFY, len(primaries))
    if primaries[0] != wp.AMD_KEY_FINGERPRINT:
        if wp.AMD_KEY_FINGERPRINT in subkeys:
            return "{}: the pinned fingerprint is only a subkey".format(KEY_DID_NOT_VERIFY)
        return "{}: its fingerprint is not the pinned one".format(KEY_DID_NOT_VERIFY)
    return ""


def bundled_key_verdict(path: Path = BUNDLED_KEYRING) -> str:
    """The controller-side check of the key the wheel carries, before it is sent."""
    try:
        return key_verdict(*openpgp_keys(path.read_bytes()))
    except (OSError, ValueError, IndexError):
        return "{}: the copy this controller carries could not be read".format(KEY_DID_NOT_VERIFY)


#: The worker-side key check: ``python3 -c <program> <keyring path>``.
KEY_CHECK_PROGRAM = _PARSER_SOURCE + r'''
import json, sys
try:
    with open(sys.argv[1], "rb") as handle:
        found = openpgp_keys(handle.read())
    print(json.dumps({"primaries": found[0], "subkeys": found[1], "error": ""}))
except (OSError, ValueError, IndexError) as error:
    print(json.dumps({"primaries": [], "subkeys": [], "error": type(error).__name__}))
'''

#: The worker-side apt questions and the hold, as one ``python3 -c`` program
#: run as root: ``apt-cache``, ``apt-mark`` and ``dpkg`` are not on the SSH
#: allowlist and are not added to it (LESSONS 18). It prints one JSON object.
APT_PROGRAM = r'''
import functools, json, re, subprocess, sys
spec = json.loads(sys.argv[1])
env = {"LC_ALL": "C", "PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "DEBIAN_FRONTEND": "noninteractive",
       "NEEDRESTART_SUSPEND": "1", "NEEDRESTART_MODE": "l"}
def run(argv, timeout=120, both=False):
    done = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=env)
    out = done.stdout.decode("utf-8", "replace")
    return done.returncode, (out + done.stderr.decode("utf-8", "replace")) if both else out
def versions(package):
    code, out = run(["apt-cache", "madison", package])
    found = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 2 and parts[0] == package and parts[1] not in found:
            found.append(parts[1])
    return found
def newer(a, b):
    if a == b:
        return 0
    return 1 if run(["dpkg", "--compare-versions", a, "gt", b])[0] == 0 else -1
def newest(candidates):
    return sorted(candidates, key=functools.cmp_to_key(newer))[-1] if candidates else ""
def pairs(series):
    smi = versions(spec["smi_prefix"] + series)
    deps = set(versions(spec["deps_prefix"] + series))
    return [version for version in smi if version in deps and version not in spec.get("exclude", [])]
def series_key(value):
    return tuple(int(part) for part in value.split("."))
def plan():
    series, pin = spec["series"], spec["pin"]
    within = pairs(series)
    if pin in within:
        return {"version": pin, "series": series, "fallback": False}
    # Only ever NEWER than the pin (review S2): an older listed version is not
    # "the latest available", and a pin that is still listed is never skipped
    # unless the caller excluded it for unmet dependencies.
    newer_within = [version for version in within if newer(version, pin) > 0]
    if newer_within:
        return {"version": newest(newer_within), "series": series, "fallback": True}
    code, out = run(["apt-cache", "pkgnames", spec["smi_prefix"]])
    shape = re.compile(re.escape(spec["smi_prefix"]) + r"(\d+\.\d+)$")
    others = sorted({m.group(1) for m in map(shape.match, out.split()) if m
                     and series_key(m.group(1)) > series_key(series)}, key=series_key, reverse=True)
    for other in others:
        found = pairs(other)
        if found:
            return {"version": newest(found), "series": other, "fallback": True}
    return {"version": "", "series": "", "fallback": True}
def install():
    names = ["{}={}".format(name, spec["version"]) for name in spec["packages"]]
    try:
        code, out = run(["apt-get", "-o", "APT::Sandbox::User=root", "--yes", "install"] + names,
                        timeout=int(spec["apt_seconds"]), both=True)
    except subprocess.TimeoutExpired:
        return {"ok": False, "unmet": False, "detail": "apt did not finish within its time"}
    return {"ok": code == 0, "unmet": "unmet dependencies" in out.lower(),
            "detail": " ".join(out.split())[-300:]}
def record():
    code, out = run(["dpkg", "-L", spec["package"]])
    binary = next((line.strip() for line in out.splitlines() if line.strip().endswith("/bin/amd-smi")), "")
    tool = ""
    if binary:
        try:
            tool = " ".join(run([binary, "version"], timeout=60)[1].split())[:200]
        except (OSError, subprocess.SubprocessError):
            tool = ""
    return {"binary": binary[:200], "tool_version": tool}
action = spec["action"]
try:
    if action == "plan":
        answer = plan()
    elif action == "install":
        answer = install()
    elif action in ("hold", "unhold"):
        code, out = run(["apt-mark", action] + list(spec["packages"]))
        answer = {"ok": code == 0}
    elif action == "record":
        answer = record()
    else:
        answer = {"error": "unknown action"}
except (OSError, subprocess.SubprocessError) as error:
    answer = {"error": type(error).__name__}
print(json.dumps(answer))
'''


def _program(transport, program: str, argument: str, timeout: int = 120) -> Dict[str, Any]:
    """Run one of this module's programs as root and parse its one JSON line."""
    raw = transport.run(["python3", "-c", program, argument], sudo=True, timeout=timeout)
    try:
        answer = json.loads(str(raw or "").strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise StepFailed("The machine's answer about amd-smi could not be read.") from error
    if not isinstance(answer, dict) or answer.get("error"):
        raise StepFailed("The machine could not answer about amd-smi.")
    return answer


def apt_question(transport, action: str, timeout: int = APT_QUESTION_SECONDS, **fields: Any) -> Dict[str, Any]:
    spec = {"action": action, "smi_prefix": AMD_SMI_PACKAGE_PREFIX,
            "deps_prefix": SYSDEPS_PACKAGE_PREFIX, **fields}
    return _program(transport, APT_PROGRAM, json.dumps(spec, separators=(",", ":")), timeout=timeout)


def worker_key_verdict(transport, path: str) -> str:
    """The key check, run on the worker against ``path``."""
    try:
        answer = _program(transport, KEY_CHECK_PROGRAM, path)
    except SshTransportError:
        return "{}: the key on the machine could not be read".format(KEY_DID_NOT_VERIFY)
    return key_verdict(list(answer.get("primaries") or []), list(answer.get("subkeys") or []))


def packages_for(series: str) -> List[str]:
    return [AMD_SMI_PACKAGE_PREFIX + series, SYSDEPS_PACKAGE_PREFIX + series]


def installed_pair(item: Mapping[str, Any]) -> Tuple[str, bool]:
    """The version both pinned packages are installed at (``""`` unless they agree), and held."""
    packages = item.get("packages") if isinstance(item.get("packages"), dict) else {}
    entries = [packages.get(name) for name in wp.AMD_SMI_PACKAGES]
    if not all(isinstance(entry, dict) and entry.get("version") for entry in entries):
        return "", False
    versions = {entry["version"] for entry in entries}
    return (versions.pop() if len(versions) == 1 else ""), all(entry.get("held") is True for entry in entries)


class StepFailed(SshTransportError):
    """A step Vaelor itself judged failed; its text is Vaelor's own sentence.

    Every other transport error may carry the worker's raw output, which goes
    to the log only, never in front of an operator (LESSONS 24, review S1).
    """


class AmdSmiFailed(RuntimeError):
    """The component could not be applied; ``str()`` is the sentence, ``undo`` what was put back."""

    def __init__(self, sentence: str, undo: List[str]):
        super().__init__(sentence)
        self.undo = undo


def apply(transport, item: Mapping[str, Any], facts: Mapping[str, Any], *,
          write_text: Callable[[str, str, str], None],
          previous_record: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Lay the amd-smi component down; return its measured record.

    ``item`` is the probe's reading of the component. ``write_text(path, text,
    mode)`` lands a root-owned text file the way the runner lands every other
    (stage, digest check, rename). Raises :class:`AmdSmiFailed` with the
    sentence and what was put back; nothing it added is left behind.
    """
    added: List[str] = []
    progress = {"installed": False}
    version_id = str(facts.get("os_version") or "")
    line = wp.amd_source_line(version_id) + "\n"
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    keyring = item.get("keyring") if isinstance(item.get("keyring"), dict) else {}
    try:
        if source.get("present") is False:
            _add_source(transport, keyring, line, write_text, added)
        elif source.get("text_sha256") != wp._text_sha(line):  # noqa: SLF001 - the manifest's own digest
            raise AmdSmiFailed(SOURCE_NOT_OURS, [])
        else:
            verdict = worker_key_verdict(transport, wp.AMD_KEYRING_PATH)
            if verdict:
                raise AmdSmiFailed(verdict, [])
        write_text(wp.AMD_PREFERENCES_PATH, wp.amd_preferences_text(), "0644")
        record = _ensure_packages(transport, item, previous_record, progress)
        held = apt_question(transport, "hold", packages=packages_for(record["series"]))
        if not held.get("ok"):
            raise AmdSmiFailed(HOLD_FAILED, [])
        record.update(apt_question(transport, "record", package=AMD_SMI_PACKAGE_PREFIX + record["series"]))
        record["added_source"] = bool(added) or bool((previous_record or {}).get("added_source"))
        return record
    except AmdSmiFailed as failure:
        failure.undo = _take_back(transport, added) + failure.undo + _stay(progress)
        raise
    except channel_drops():
        # The connection went away (review S4): the runner tells that outcome
        # apart; nothing more can be asked of the machine here.
        raise
    except machine_failures() as error:
        raise AmdSmiFailed(_why(error), _take_back(transport, added) + _stay(progress)) from error


def _stay(progress: Mapping[str, Any]) -> List[str]:
    return [PACKAGES_STAY] if progress.get("installed") else []


def _why(error: BaseException) -> str:
    """The step's own sentence, or the runner's words for the transport's error."""
    text = str(error)
    for sentence in (INSTALL_FAILED, HOLD_FAILED, NO_VERSION_PUBLISHED):
        if text.startswith(sentence):
            return sentence
    from .worker_profile_apply import failure_sentence

    return failure_sentence(error)


def _add_source(transport, keyring: Mapping[str, Any], line: str,
                write_text: Callable[[str, str, str], None], added: List[str]) -> None:
    """Add AMD's key (unless a verified one is there) and source, recording each before writing."""
    if keyring.get("present"):
        verdict = worker_key_verdict(transport, wp.AMD_KEYRING_PATH)
        if verdict:
            raise AmdSmiFailed(verdict, [])
    else:
        verdict = bundled_key_verdict()
        if verdict:
            raise AmdSmiFailed(verdict, [])
        staged = transport.stage_private_artifact(str(BUNDLED_KEYRING), "amdrocm.gpg")
        try:
            verdict = worker_key_verdict(transport, staged)
            if verdict:
                raise AmdSmiFailed(verdict, [])
            transport.run(["install", "-d", "-m", "0755", KEYRING_DIRECTORY], sudo=True)
            added.append(wp.AMD_KEYRING_PATH)
            transport.run(["install", "-m", "0644", "-o", "root", "-g", "root", staged,
                           wp.AMD_KEYRING_PATH], sudo=True)
        finally:
            transport.discard_private_artifact(staged)
    added.append(wp.AMD_SOURCE_PATH)
    write_text(wp.AMD_SOURCE_PATH, line, "0644")
    try:
        transport.run(["apt-get", "update"], sudo=True, timeout=APT_UPDATE_SECONDS)
    except channel_drops():
        raise
    except SshTransportError as error:
        LOGGER.warning("apt-get update on the worker failed: %s", error)
        raise StepFailed(INSTALL_FAILED) from error


def _ensure_packages(transport, item: Mapping[str, Any], previous: Optional[Mapping[str, Any]],
                     progress: Dict[str, Any]) -> Dict[str, Any]:
    """Install the pair when it is missing; a held installed pair is kept as it is.

    Design §2.5 (owner answer 9): only an INSTALL meets pin drift, and only a
    NEWER version replaces the pin - when AMD no longer lists it, or apt says
    its dependencies are gone (review S2). Any other failure (the dpkg lock,
    the network) is "could not install", tried again later, never a fallback.
    """
    version, _held = installed_pair(item)
    previous = dict(previous or {})
    if version == wp.AMD_SMI_PIN:
        return _record(version, wp.AMD_SMI_SERIES, reason="")
    if version and version == previous.get("installed_version") and previous.get("fallback"):
        return _record(version, str(previous.get("series") or wp.AMD_SMI_SERIES),
                       reason=str(previous.get("reason") or FALLBACK_REASON.format(pin=wp.AMD_SMI_PIN)))
    if version:
        # Installed by someone else at another version: kept, said, never downgraded.
        return _record(version, wp.AMD_SMI_SERIES, reason="", note="kept as it was found")
    plan = apt_question(transport, "plan", series=wp.AMD_SMI_SERIES, pin=wp.AMD_SMI_PIN)
    if not plan.get("version"):
        raise StepFailed(NO_VERSION_PUBLISHED)
    reason = FALLBACK_REASON.format(pin=wp.AMD_SMI_PIN) if plan.get("fallback") else ""
    answer = _install(transport, plan)
    if not answer.get("ok") and not plan.get("fallback") and answer.get("unmet"):
        plan = apt_question(transport, "plan", series=wp.AMD_SMI_SERIES, pin=wp.AMD_SMI_PIN,
                            exclude=[wp.AMD_SMI_PIN])
        if not plan.get("version"):
            raise StepFailed(NO_VERSION_PUBLISHED)
        reason = UNMET_REASON.format(pin=wp.AMD_SMI_PIN)
        answer = _install(transport, plan)
    if not answer.get("ok"):
        raise StepFailed(INSTALL_FAILED)
    progress["installed"] = True
    return _record(plan["version"], plan["series"], reason=reason)


def _install(transport, plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Install the pair by exact version, with needrestart suspended (review S5).

    Run inside :data:`APT_PROGRAM`, whose environment sets
    ``NEEDRESTART_SUSPEND`` and ``DEBIAN_FRONTEND``: an argv ``apt-get`` over
    SSH cannot set them, and needrestart could otherwise restart services the
    upgraded libraries touch - serving among them. apt's own output is logged,
    never shown (LESSONS 24).
    """
    answer = apt_question(transport, "install", version=plan["version"], packages=packages_for(plan["series"]),
                          apt_seconds=APT_INSTALL_SECONDS,
                          timeout=APT_INSTALL_SECONDS + APT_TRANSPORT_MARGIN_SECONDS)
    if not answer.get("ok"):
        LOGGER.warning("installing amd-smi %s on a worker failed: %s", plan["version"], answer.get("detail", ""))
    return answer


def _record(version: str, series: str, *, reason: str, note: str = "") -> Dict[str, Any]:
    return {"pin": wp.AMD_SMI_PIN, "installed_version": version, "series": series,
            "fallback": bool(reason), "reason": reason, "note": note, "binary": "", "tool_version": ""}


def _take_back(transport, added: List[str]) -> List[str]:
    """Remove what this apply added, newest first; say what could not be removed."""
    undone = []
    for path in reversed(added):
        try:
            transport.run(["rm", "-f", path], sudo=True)
            undone.append("removed {}".format(path))
        except machine_failures():
            undone.append("could not remove {}; delete it by hand with: sudo rm -f {}".format(path, path))
    return undone


def remove(transport, record: Optional[Mapping[str, Any]]) -> List[str]:
    """Leaving the cluster (design §5): the preferences file and the holds go;
    the packages stay; the source and key go only if the profile added them."""
    done = []
    record = dict(record or {})
    series = str(record.get("series") or wp.AMD_SMI_SERIES)
    transport.run(["rm", "-f", wp.AMD_PREFERENCES_PATH], sudo=True)
    done.append(wp.AMD_PREFERENCES_PATH)
    apt_question(transport, "unhold", packages=packages_for(series))
    if record.get("added_source"):
        for path in (wp.AMD_SOURCE_PATH, wp.AMD_KEYRING_PATH):
            transport.run(["rm", "-f", path], sudo=True)
            done.append(path)
    return done

