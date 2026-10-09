"""Compare what a worker holds with what its profile says it should (VD-194 P1).

Pure: an expectation from `worker_profile.node_expected` and a reading from
`worker_profile_probe` in, one row per component out, each ``matches``,
``differs``, ``unread`` or ``not-applicable`` with the reason in words, and the
measured digest - the expectation's own canonical digest applied to what was
read, so a worker that matches everywhere reads the controller's profile.

**Unread is never matches and never absent** (LESSONS 8). A reading that
failed, an item the probe did not report, a field it could not read: each is
``unread (why)``. Absence is only ever a positive ``present: false``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import worker_profile as wp

MATCHES, DIFFERS, UNREAD, NOT_APPLICABLE = "matches", "differs", "unread", "not-applicable"

#: The words a row's status reads as (design §6).
STATUS_WORDS = {MATCHES: "matches", DIFFERS: "differs", UNREAD: "unread",
                NOT_APPLICABLE: "does not apply"}

#: Said after an appliance-owned difference. Each difference is a sentence of
#: its own and none is wrapped in brackets, so a reason that carries brackets
#: of its own never nests (LESSONS 5; owner box check 2026-10-04, VD-194). It
#: promises no conversion: nothing converts a worker yet (found on the Z2
#: 2026-10-04, LESSONS 10).
APPLIANCE_OWNED = "The appliance set this; Vaelor does not change it yet."

#: What a unit state reads as when systemd gave none.
NO_STATE = "no state"

_TYPE_WORDS = {"dir": "a folder", "file": "a file", "link": "a link", "other": "a special file"}


def sentence(text: str) -> str:
    """One phrase as a sentence: a capital first letter and one closing full stop."""
    text = str(text or "").strip().rstrip(".")
    if not text:
        return ""
    return (text[0].upper() + text[1:] if text[0].islower() else text) + "."


def _row(entry: Mapping[str, Any], status: str, why: str = "", note: str = "",
         appliance_owned: bool = False, measured: str = "") -> Dict[str, Any]:
    """One component's row. ``words`` is the whole sentence the card draws (VD-173):
    the status, then the reason and the note, each as sentences of their own."""
    word = STATUS_WORDS[status]
    why, note = sentence(why), sentence(note)
    return {"id": entry["id"], "label": entry["label"], "status": status,
            "word": word, "why": why, "note": note, "appliance_owned": appliance_owned,
            "measured": measured, "words": " ".join(filter(None, (sentence(word), why, note)))}


def _ownership(item: Mapping[str, Any], fields: Mapping[str, Any]) -> Optional[List[str]]:
    """Owner, group and mode differences, or ``None`` when any could not be read."""
    if not item.get("owner") or not item.get("group") or not item.get("mode"):
        return None
    found = []
    if (item["owner"], item["group"]) != (fields["owner"], fields["group"]):
        found.append("Owned by {}:{}, expected {}:{}".format(
            item["owner"], item["group"], fields["owner"], fields["group"]))
    if fields.get("mode") and item["mode"] != fields["mode"]:
        found.append("Mode {}, expected {}".format(item["mode"], fields["mode"]))
    return found


def _unit_differences(item: Mapping[str, Any]) -> Optional[List[str]]:
    state = item.get("state")
    if not isinstance(state, dict) or item.get("state_error"):
        return None
    found = []
    if state.get("ActiveState") != "active":
        found.append("It is not running: {}".format(state.get("ActiveState") or NO_STATE))
    if state.get("UnitFileState") != "enabled":
        found.append("It is not enabled: {}".format(state.get("UnitFileState") or NO_STATE))
    return found


def _package_differences(item: Mapping[str, Any], fields: Mapping[str, Any]) -> Tuple[List[str], str]:
    """What differs about the packages besides their version, which is the
    row's measured value (:func:`amd_smi_version`) and said only there, and
    why the package source could not be read (``""`` when it was).

    RW-L2 (LESSONS 8): an unread source no longer drops the holds already
    found; both facts reach the row."""
    found = []
    installed = item["packages"]
    for name in fields["packages"]:
        entry = installed.get(name)
        if not entry:
            found.append("Package {} is not installed".format(name))
            continue
        if entry.get("held") is not True:
            found.append("Package {} is not held".format(name))
    source = item.get("source") if isinstance(item.get("source"), dict) else {}
    if source.get("error"):
        return found, "AMD's package source could not be read"
    if source.get("present") is False:
        found.append("AMD's package source is not configured")
    elif source.get("text_sha256") != fields["source_sha256"]:
        found.append("AMD's package source line differs")
    return found, ""


def _compare_present(entry: Mapping[str, Any], item: Mapping[str, Any],
                     appliance_present: bool, kept: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    kind, fields = entry["kind"], entry["fields"]
    if kind == wp.PREREQUISITE:
        if item.get("present") is True:
            return _row(entry, MATCHES)
        return _row(entry, DIFFERS, "It is not installed on this machine; the profile needs it")
    differences: List[str] = []
    if item.get("present") is False:
        if kind == wp.PACKAGE:
            differences.append("its apt preferences file is not on this machine")
        else:
            return _row(entry, DIFFERS, "It is not on this machine")
    else:
        wanted = "dir" if kind == wp.DIRECTORY else "file"
        if item.get("type") != wanted:
            return _row(entry, DIFFERS, "It is {}, not {}".format(
                _TYPE_WORDS.get(str(item.get("type")), "something else"), _TYPE_WORDS[wanted]))
        ownership = _ownership(item, fields)
        if ownership is None:
            return _row(entry, UNREAD, "its owner and mode could not be read")
        differences += ownership
    unread_reason = ""
    measured, version_differs, version_unread = "", False, False
    if kind == wp.DIRECTORY:
        if item.get("acl") is True:
            differences.append("Has an extra access list")
        elif item.get("acl") is not False:
            unread_reason = "its access list could not be read"
    elif kind == wp.UNIT:
        if not item.get("text_sha256"):
            unread_reason = "its text could not be read"
        elif item["text_sha256"] != fields["text_sha256"]:
            differences.append("its text differs")
        unit = _unit_differences(item)
        if unit is None:
            unread_reason = unread_reason or "its service state could not be read"
        else:
            differences += unit
    elif kind == wp.PACKAGE:
        if item.get("present") is not False and item.get("sha256") != fields["sha256"]:
            differences.append("its apt preferences file differs")
        # The version is the row's measured value, said there once and for
        # every package (review round 2, LESSONS 6); an unread one is unread.
        measured, version = amd_smi_version(item, fields["packages"], fields["version"], kept)
        version_differs, version_unread = version == DIFFERS, version == UNREAD
        if not version_unread:
            package, unread_reason = _package_differences(item, fields)
            differences += package
    elif kind != wp.MARKER:
        if not fields.get("sha256"):
            unread_reason = fields.get("cannot_compare") or "it cannot be compared"
        elif not item.get("sha256"):
            unread_reason = "its contents could not be read"
        elif item["sha256"] != fields["sha256"]:
            differences.append("its contents differ")
    note = ""
    state = item.get("state") if isinstance(item.get("state"), dict) else {}
    if state.get("NeedDaemonReload") == "yes":
        note = "systemd has a reload pending for it"
    if differences or version_differs:
        owned = bool(entry.get("appliance_owned")) and appliance_present
        # RW-L2 (LESSONS 8): a difference decides the status, and what could
        # not be read is still said beside it, never dropped.
        parts = differences + ([unread_reason] if unread_reason else []) + ([APPLIANCE_OWNED] if owned else [])
        why = " ".join(sentence(part) for part in parts)
        return _row(entry, DIFFERS, why, note, owned, measured)
    if unread_reason or version_unread:
        return _row(entry, UNREAD, unread_reason, note, measured=measured)
    return _row(entry, MATCHES, "", note, measured=measured)


def compare_component(entry: Mapping[str, Any], reading: Mapping[str, Any],
                      appliance_present: bool, kept: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """One component's row. ``kept`` is the amd-smi record an apply wrote (P2)."""
    if entry["applies"] is False:
        return _row(entry, NOT_APPLICABLE, entry["why"])
    if entry["applies"] is None:
        return _row(entry, UNREAD, entry["why"])
    if not reading.get("ok"):
        return _row(entry, UNREAD, reading.get("error") or "the machine could not be read")
    item = (reading.get("items") or {}).get(entry["id"])
    if not isinstance(item, dict):
        return _row(entry, UNREAD, "the machine did not report it")
    if item.get("error"):
        return _row(entry, UNREAD, "It could not be read: {}".format(item["error"]))
    return _compare_present(entry, item, appliance_present, kept)


def measured_fields(kind: str, item: Mapping[str, Any],
                    kept: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The fingerprint fields (`worker_profile.FINGERPRINT_FIELDS`) as the worker holds them.

    A version an apply recorded as kept (the newest available when the pin is
    gone, owner answer 9) counts as the pin for the digest: the digest stays
    the release's, and only the record carries the difference (design §2.5).
    """
    if kind == wp.PREREQUISITE:
        return {"present": item.get("present")}
    fields = {"present": item.get("present"), "owner": item.get("owner"),
              "group": item.get("group"), "mode": item.get("mode"), "acl": item.get("acl"),
              "sha256": item.get("sha256"), "text_sha256": item.get("text_sha256")}
    state = item.get("state") if isinstance(item.get("state"), dict) else {}
    fields.update({"active": state.get("ActiveState"), "enabled": state.get("UnitFileState")})
    if kind == wp.PACKAGE:
        packages = item.get("packages") if isinstance(item.get("packages"), dict) else {}
        versions = {entry.get("version") for entry in packages.values()}
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        version = versions.pop() if len(versions) == 1 and len(packages) == len(wp.AMD_SMI_PACKAGES) else None
        if version and kept_version(kept) == version:
            version = wp.AMD_SMI_PIN
        fields.update({
            "version": version,
            "held": all(entry.get("held") is True for entry in packages.values()) and bool(packages),
            "source_sha256": source.get("text_sha256"),
        })
    return fields


def compare(expected: Mapping[str, Any], reading: Mapping[str, Any],
            appliance_present: bool, amd_smi_record: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Every component's row, and the measured digest (``None`` while anything is unread)."""
    rows = [compare_component(entry, reading, appliance_present, amd_smi_record)
            for entry in expected["components"]]
    measured: Optional[str] = None
    if not any(row["status"] == UNREAD for row in rows):
        items = reading.get("items") or {}
        measured = wp.fingerprint_digest(
            (entry["id"], entry["kind"],
             measured_fields(entry["kind"], items.get(entry["id"]) or {}, amd_smi_record))
            for entry in expected["components"] if entry["applies"]
        )
    return {"components": rows, "measured_digest": measured}


def kept_version(record: Optional[Mapping[str, Any]]) -> str:
    """The version an apply installed or kept and recorded as such, or ``""``.

    Only a record that says why it is not the pin counts: the newest available
    because the pin is no longer published, or a version already installed
    that the apply kept rather than downgrade.
    """
    record = record if isinstance(record, Mapping) else {}
    if record.get("fallback") or record.get("note"):
        return str(record.get("installed_version") or "")
    return ""


def amd_smi_version(item: Optional[Mapping[str, Any]], names=wp.AMD_SMI_PACKAGES,
                    pin: str = wp.AMD_SMI_PIN,
                    kept: Optional[Mapping[str, Any]] = None) -> Tuple[str, str]:
    """Which amd-smi this worker runs, in words, and how that compares with the pin.

    The AMD amd-smi row's measured value (design §2.5, VD-173): the one place
    the installed version is said, covering every package (review round 2,
    LESSONS 6). Returns ``(words, verdict)``, the verdict ``matches``,
    ``differs``, ``unread``, or ``""`` when no package is installed (the row's
    differences say which). A version that was not read is never printed as
    one (LESSONS 5); "the version the profile pins" says what was compared,
    not that apt holds it, which is one of the row's differences.
    """
    packages = item.get("packages") if isinstance(item, dict) else None
    if not isinstance(item, dict) or item.get("error") or item.get("package_error") or not isinstance(
            packages, dict):
        return "Which version is installed could not be read.", UNREAD
    installed = {name: packages[name] for name in names if packages.get(name)}
    if not installed:
        return "No version of AMD's amd-smi is installed.", ""
    unread = [name for name, entry in installed.items() if not entry.get("version")]
    if unread:
        return "The installed version of {} could not be read.".format(", ".join(unread)), UNREAD
    versions = {name: str(entry["version"]) for name, entry in installed.items()}
    found = set(versions.values())
    if found == {pin}:
        return "Version {} installed, the version the profile pins.".format(pin), MATCHES
    if len(found) == 1 and kept_version(kept) in found:
        return _kept_words(kept, pin), MATCHES
    if len(found) == 1:
        return "Version {} installed; the profile pins {}.".format(found.pop(), pin), DIFFERS
    return "Versions installed: {}; the profile pins {}.".format(
        ", ".join("{} {}".format(name, version) for name, version in versions.items()), pin), DIFFERS


def _kept_words(kept: Mapping[str, Any], pin: str) -> str:
    """The card's words for a kept version, naming both (design §2.5, owner answer 9)."""
    version = kept.get("installed_version")
    if kept.get("fallback"):
        # The reason the apply recorded: no longer published, or no longer
        # installable (review S2) - never a reason it did not observe.
        why = str(kept.get("reason") or "the pinned {} is no longer published by AMD".format(pin))
        return "Version {} installed, the newest available; {}.".format(version, why)
    return "Version {} installed, kept as it was found; the profile pins {}.".format(version, pin)
