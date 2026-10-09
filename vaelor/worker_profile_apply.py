"""Lay a worker's profile down: stage, swap, verify, and put back on failure (VD-194 P2).

The apply runner the ``cluster.node.profile`` job drives (`worker_profile_job`).
It takes what the read-only probe found (`worker_profile_probe`) and what this
controller expects (`worker_profile.node_expected`), compares them with the
one comparison the card uses (`worker_profile_compare`), and changes only the
parts that differ - so a worker that already matches is left untouched, and
"what needs doing" has one answer (:func:`plan`, LESSONS 6).

**Every command is an allowlisted argv through `SshTransport.run`** (LESSONS
18): ``install``, ``chown``, ``chmod``, ``tee`` (which the transport lands
beside the target, digest-checks and renames into place), ``sha256sum``,
``stat``, ``tar``, ``systemctl``, ``rm``, ``apt-get`` and ``python3 -c`` for
the few questions no allowlisted tool answers. Artifacts are staged in a
private directory (VD-172) and their digests are checked on both sides.

**Serving is never touched (guard G6).** No component names a model server,
a model download, a Ray slice or Docker, and nothing here restarts any of
them; the only services restarted are the telemetry agent and the GPU
sampler. A component that would interrupt serving is marked ``disruptive``
in the manifest and is held while a deployment names the worker (design
§3a.4); none is today.

**Each change is put back if it fails, and the putting back is checked**
(design §3, as VD-161). A file is copied to ``<path>.vaelor-prev`` first; on
failure the copy goes back and its digest is read again. Three outcomes are
told apart: put back and checked, could not be put back (with what the file
holds now and the command that fixes it), and the machine stopped answering
(counted as failed; it is read again when it answers). The marker that says
which profile the worker holds is written last, only after every other part
was read back as matching - so a partial apply is never reported as success.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import worker_profile as wp
from . import worker_profile_amd_smi as amd_smi
from .worker_profile_amd_smi import StepFailed
from . import worker_telemetry_config as telemetry
from .ssh_transport import (
    COMMAND_ENDED_WITHOUT_REASON, SshTransportError, channel_drops, machine_failures,
)
from .worker_profile_compare import DIFFERS, UNREAD, compare
from .worker_telemetry_runtime import _absent
from .worker_telemetry_bundle import (
    build_sampler_bytes, build_zipapp_bytes, bundle_digest, sampler_bundle_digest,
)

LOGGER = logging.getLogger(__name__)

#: Where a file's previous content waits while its replacement is checked.
PREVIOUS_SUFFIX = ".vaelor-prev"

#: The job's outcomes. ``applied`` and ``current`` are the two successes.
APPLIED, CURRENT, REFUSED, HELD, INCOMPLETE = "applied", "current", "refused", "held-serving", "incomplete"
FAILED_RESTORED, FAILED_NOT_RESTORED, CONNECTION_LOST = (
    "failed-restored", "failed-not-restored", "connection-lost")
SUCCESSES = frozenset({APPLIED, CURRENT})

#: The services a profile change may restart, and which components each owns.
#: Nothing else is ever restarted (guard G6).
SERVICE_GROUPS = (
    (telemetry.UNIT_NAME, ("telegraf", "emitter", "controller-ca", "telegraf-config", "telegraf-unit")),
    (telemetry.SAMPLER_UNIT_NAME, ("sampler", "sampler-unit")),
)
_GROUP_OF = {component: unit for unit, members in SERVICE_GROUPS for component in members}

#: A directory's access list, dropped or put back, and an empty directory
#: removed again: ``setfacl`` and ``rmdir`` are not on the allowlist.
DIRECTORY_PROGRAM = r'''
import base64, json, os, sys
action, path = sys.argv[1], sys.argv[2]
name = "system.posix_acl_access"
try:
    if action == "drop_acl":
        try:
            value = os.getxattr(path, name, follow_symlinks=False)
        except OSError:
            value = b""
        if value:
            os.removexattr(path, name, follow_symlinks=False)
        print(json.dumps({"acl": base64.b64encode(value).decode("ascii")}))
    elif action == "set_acl":
        os.setxattr(path, name, base64.b64decode(sys.argv[3]), follow_symlinks=False)
        print(json.dumps({"ok": True}))
    elif action == "rmdir":
        os.rmdir(path)
        print(json.dumps({"ok": True}))
except OSError as error:
    print(json.dumps({"error": type(error).__name__}))
'''


_DROPS = channel_drops()
_MACHINE = machine_failures()


def _connection_lost(error: BaseException) -> bool:
    """A channel that went away, as opposed to a machine that refused and said why."""
    return isinstance(error, _DROPS) or (
        isinstance(error, SshTransportError) and str(error).strip() == COMMAND_ENDED_WITHOUT_REASON)


def plan(expected: Mapping[str, Any], reading: Mapping[str, Any], appliance_present: bool,
         *, amd_smi_record: Optional[Mapping[str, Any]] = None,
         key_provisioned: bool = True) -> Dict[str, Any]:
    """What an apply would change on this worker, from the card's own comparison.

    ``apply``: the ids that differ and that the profile owns. ``refused``: a
    missing prerequisite, which nothing here installs. ``appliance_owned``:
    differences the full appliance's own layout makes, left alone while it is
    installed. ``unread``: parts that could not be compared, which are not
    changed. A telemetry config is rewritten when the controller holds no key
    for its agent, because the comparison cannot see a key (TELEM-1).
    """
    rows = compare(expected, reading, appliance_present, amd_smi_record=amd_smi_record)["components"]
    kinds = {entry["id"]: entry for entry in expected["components"]}
    result: Dict[str, List[str]] = {"apply": [], "refused": [], "appliance_owned": [], "unread": [],
                                     "disruptive": []}
    for row in rows:
        entry = kinds[row["id"]]
        if row["status"] == UNREAD:
            result["unread"].append(row["id"])
        elif row["status"] != DIFFERS:
            if row["id"] == "telegraf-config" and entry.get("applies") and not key_provisioned:
                result["apply"].append(row["id"])
            continue
        elif row["appliance_owned"]:
            result["appliance_owned"].append(row["id"])
        elif entry["kind"] == wp.PREREQUISITE:
            result["refused"].append(row["id"])
        else:
            result["apply"].append(row["id"])
            if entry.get("disruptive"):
                result["disruptive"].append(row["id"])
    marker = (reading.get("items") or {}).get("marker") or {}
    stale = marker.get("present") is True and marker.get("profile_digest") != expected["digest"]
    if ((result["apply"] or stale) and "marker" not in result["apply"]
            and kinds.get("marker", {}).get("applies")):
        # The marker records the digest; whatever else changes, it is rewritten.
        result["apply"].append("marker")
    result["apply"].sort(key=_ORDER.get)
    return result


_ORDER = {component.id: index for index, component in enumerate(wp.MANIFEST)}


def needs_apply(planned: Mapping[str, Any]) -> bool:
    """Whether an apply job has anything it could change."""
    return bool(planned["apply"])


@dataclass
class Undo:
    """What one change needs to be put back."""

    component: str
    path: str
    kind: str
    existed: bool
    sha256: str = ""
    owner: str = "root"
    group: str = "root"
    mode: str = "0644"
    acl: str = ""


@dataclass
class Inputs:
    """What the controller contributes to one apply. Everything here is the controller's own."""

    node_id: str
    cluster_id: str
    architecture: str
    ingest_url: str
    controller_ca_source: Optional[str]
    telegraf_artifact: Callable[[str], Any]
    telegraf_binary_sha256: str
    mint_key: Callable[[], str]
    restore_key: Callable[[], None]
    expected_digest: str
    previous_amd_smi: Optional[Mapping[str, Any]] = None
    gpu_sampler: bool = False
    facts: Dict[str, Any] = field(default_factory=dict)


class ApplyFailed(RuntimeError):
    """One part could not be changed; carries what was put back."""

    def __init__(self, component: str, sentence: str, outcome: str, undo: List[str]):
        super().__init__(sentence)
        self.component, self.outcome, self.undo = component, outcome, undo


class ProfileApplier:
    """One apply against one worker. ``transport`` is its enrolled `SshTransport`."""

    def __init__(self, transport, inputs: Inputs, reading: Mapping[str, Any]):
        self.transport = transport
        self.inputs = inputs
        self.items = dict(reading.get("items") or {})
        self.amd_smi_record: Optional[Dict[str, Any]] = None
        self._key_minted = False

    # -- the run -----------------------------------------------------------
    def apply(self, ids: List[str]) -> List[str]:
        """Change every id in ``ids`` (marker excluded), in manifest order.

        Raises :class:`ApplyFailed` at the first part that cannot be changed,
        after putting that part (or its whole service group) back. Returns the
        ids changed.
        """
        done: List[str] = []
        pending = [component for component in ids if component != "marker"]
        started = set()
        # Manifest order (review B1): the folders come first, so a file is never
        # written before the folder it lives in exists on a fresh worker. A
        # service group is applied whole, where its first member falls.
        for component in sorted(pending, key=_ORDER.get):
            unit = _GROUP_OF.get(component)
            if unit is None:
                self._apply_single(component)
                done.append(component)
            elif unit not in started:
                started.add(unit)
                group = [member for member in pending if _GROUP_OF.get(member) == unit]
                self._apply_group(unit, group)
                done += group
        return sorted(done, key=_ORDER.get)

    def write_marker(self) -> None:
        """The installer's fence (design §2.1 item 6), written after everything else was read back."""
        text = json.dumps({
            "cluster_id": self.inputs.cluster_id, "node_id": self.inputs.node_id,
            "profile_digest": self.inputs.expected_digest, "applied_at": int(time.time()),
        }, sort_keys=True) + "\n"
        undo = self._backup("marker", wp.MARKER_PATH)
        try:
            self._write_text(wp.MARKER_PATH, text, "0644")
        except Exception as error:  # noqa: BLE001 - put back, then reported by kind
            raise self._failed("marker", error, [undo]) from error
        self._drop_backups([undo])

    # -- single parts ------------------------------------------------------
    def _apply_single(self, component: str) -> None:
        entry = wp.COMPONENTS[component]
        if entry.kind == wp.DIRECTORY:
            undo = self._directory_undo(entry)
            try:
                self._apply_directory(entry, undo)
            except Exception as error:  # noqa: BLE001 - put back, then reported by kind
                raise self._failed(component, error, [undo]) from error
            return
        if entry.kind == wp.PACKAGE:
            self._apply_amd_smi()
            return
        undo = self._backup(component, entry.path)
        try:
            self._write_text(entry.path, wp.WMI_MODULE_CONF_TEXT, entry.mode)
        except Exception as error:  # noqa: BLE001 - put back, then reported by kind
            raise self._failed(component, error, [undo]) from error
        self._drop_backups([undo])

    def _apply_amd_smi(self) -> None:
        item = self.items.get("amd-smi") or {}
        undo = self._backup("amd-smi", wp.AMD_PREFERENCES_PATH)
        try:
            self.amd_smi_record = amd_smi.apply(
                self.transport, item, self.inputs.facts,
                write_text=self._write_text, previous_record=self.inputs.previous_amd_smi)
        except amd_smi.AmdSmiFailed as failure:
            restored = self._restore_all([undo])
            outcome = FAILED_RESTORED if restored["ok"] else FAILED_NOT_RESTORED
            raise ApplyFailed("amd-smi", str(failure), outcome,
                              failure.undo + restored["words"]) from failure
        except Exception as error:  # noqa: BLE001 - put back, then reported by kind
            raise self._failed("amd-smi", error, [undo]) from error
        self._drop_backups([undo])

    def _directory_undo(self, entry: wp.Component) -> Undo:
        item = self.items.get(entry.id) or {}
        return Undo(entry.id, entry.path, "directory", existed=item.get("present") is True,
                    owner=str(item.get("owner") or "root"), group=str(item.get("group") or "root"),
                    mode=str(item.get("mode") or entry.mode))

    def _apply_directory(self, entry: wp.Component, undo: Undo) -> None:
        run = self.transport.run
        if not undo.existed:
            run(["install", "-d", "-m", entry.mode, "-o", entry.owner, "-g", entry.group, entry.path], sudo=True)
        else:
            run(["chown", "{}:{}".format(entry.owner, entry.group), entry.path], sudo=True)
            run(["chmod", entry.mode, entry.path], sudo=True)
            if (self.items.get(entry.id) or {}).get("acl") is True:
                undo.acl = str(self._directory_program("drop_acl", entry.path).get("acl") or "")
        self._verify_directory(entry.path, entry.owner, entry.group, entry.mode)

    def _verify_directory(self, path: str, owner: str, group: str, mode: str) -> None:
        found = str(self.transport.run(["stat", "-c", "%U:%G:%a", path], sudo=True)).strip()
        wanted = "{}:{}:{:o}".format(owner, group, int(mode, 8))
        if found != wanted:
            raise StepFailed("{} reads {} after the change, not {}.".format(path, found, wanted))

    def _directory_program(self, action: str, path: str, *extra: str) -> Dict[str, Any]:
        raw = self.transport.run(["python3", "-c", DIRECTORY_PROGRAM, action, path, *extra], sudo=True)
        try:
            answer = json.loads(str(raw).strip().splitlines()[-1])
        except (ValueError, IndexError) as error:
            raise StepFailed("The machine's answer about {} could not be read.".format(path)) from error
        if not isinstance(answer, dict) or answer.get("error"):
            raise StepFailed("The machine could not change {}.".format(path))
        return answer

    # -- service groups ----------------------------------------------------
    def _apply_group(self, unit: str, group: List[str]) -> None:
        """Change a service's files together, restart it once, and read it back running."""
        undos: List[Undo] = []
        current = ""
        try:
            for component in group:
                current = component
                entry = wp.COMPONENTS[component]
                undos.append(self._backup(component, entry.path))
                self._write_component(entry)
            current = group[-1]
            if any(wp.COMPONENTS[component].kind == wp.UNIT for component in group):
                self.transport.run(["systemctl", "daemon-reload"], sudo=True)
            self.transport.run(["systemctl", "enable", unit], sudo=True)
            self.transport.run(["systemctl", "restart", unit], sudo=True)
            self._verify_running(unit)
        except Exception as error:  # noqa: BLE001 - put back, then reported by kind
            raise self._failed(current, error, undos, unit=unit) from error
        self._drop_backups(undos)

    def _verify_running(self, unit: str) -> None:
        raw = str(self.transport.run(["systemctl", "show", unit, "--property=ActiveState"]))
        if raw.strip() != "ActiveState=active":
            raise StepFailed("{} is not running after the change.".format(unit))

    def _write_component(self, entry: wp.Component) -> None:
        """Land one telemetry or sampler file, checked by digest."""
        if entry.id == "telegraf":
            self._ship_telegraf()
        elif entry.id == "emitter":
            self._install_bytes(build_zipapp_bytes(), "emitter.pyz", entry.path, entry.mode, bundle_digest())
        elif entry.id == "sampler":
            self._install_bytes(build_sampler_bytes(), "sampler.pyz", entry.path, entry.mode,
                                sampler_bundle_digest())
        elif entry.id == "controller-ca":
            source = self.inputs.controller_ca_source
            if not source:
                raise StepFailed("This controller has no certificate to send.")
            data = Path(source).read_bytes()
            self._install_bytes(data, "controller-ca.pem", entry.path, entry.mode, _sha(data))
        elif entry.id == "telegraf-config":
            key = self.inputs.mint_key()
            self._key_minted = True
            ca_pinned = bool(self.inputs.controller_ca_source)
            text = telemetry.render_config(
                node_id=self.inputs.node_id, ingest_url=self.inputs.ingest_url, ingest_key=key,
                tls_ca_path=telemetry.CONTROLLER_CA_PATH if ca_pinned else None,
                allow_insecure_tls=not ca_pinned)
            self._write_text(entry.path, text, entry.mode)
        elif entry.id == "telegraf-unit":
            self._write_text(entry.path, telemetry.render_unit(), entry.mode)
        elif entry.id == "sampler-unit":
            self._write_text(entry.path, telemetry.render_sampler_unit(), entry.mode)
        else:
            raise ValueError("No way to write component {!r}.".format(entry.id))

    def _ship_telegraf(self) -> None:
        """The pinned binary from the verified tarball, extracted privately and checked by digest."""
        source, pinned, member = self.inputs.telegraf_artifact(self.inputs.architecture)
        if not Path(source).is_file() or _file_sha(source) != pinned:
            raise StepFailed("This controller has no verified Telegraf tarball staged.")
        if not self.inputs.telegraf_binary_sha256:
            raise StepFailed("This controller could not work out the Telegraf binary's digest.")
        staged = self.transport.stage_private_artifact(source, "telegraf.tar.gz")
        directory = staged.rsplit("/", 1)[0]
        try:
            self._expect_sha(staged, pinned)
            self.transport.run(["tar", "-xzf", staged, "-C", directory, "--strip-components",
                                str(member.count("/")), member], sudo=True)
            self.transport.run(["install", "-m", "0755", "-o", "root", "-g", "root",
                                directory + "/telegraf", telemetry.TELEGRAF_BINARY_PATH], sudo=True)
            self._expect_sha(telemetry.TELEGRAF_BINARY_PATH, self.inputs.telegraf_binary_sha256)
        finally:
            self.transport.remove_private_directory(directory)

    def _install_bytes(self, data: bytes, filename: str, path: str, mode: str, digest: str) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(data)
            local = handle.name
        try:
            staged = self.transport.stage_private_artifact(local, filename)
        finally:
            Path(local).unlink(missing_ok=True)
        try:
            self._expect_sha(staged, digest)
            self.transport.run(["install", "-m", mode, "-o", "root", "-g", "root", staged, path], sudo=True)
        finally:
            self.transport.discard_private_artifact(staged)
        self._expect_sha(path, digest)

    # -- files: write, back up, put back ---------------------------------------
    def _write_text(self, path: str, text: str, mode: str) -> None:
        """Land a root-owned text file: staged beside it, digest-checked, renamed in.

        A file that holds a key (0600) is made private before its new text
        arrives, so the rename never lands it readable by anyone else.
        """
        run = self.transport.run
        if mode == "0600":
            if self._present(path):
                run(["chmod", "0600", path], sudo=True)
            else:
                run(["install", "-m", "0600", "/dev/null", path], sudo=True)
        run(["tee", path], sudo=True, stdin_text=text)
        run(["chown", "root:root", path], sudo=True)
        run(["chmod", mode, path], sudo=True)
        self._expect_sha(path, _sha(text.encode("utf-8")))

    def _present(self, path: str) -> bool:
        try:
            self._sha(path)
        except _Absent:
            return False
        return True

    def _sha(self, path: str) -> str:
        try:
            listing = str(self.transport.run(["sha256sum", path], sudo=True)).split()
        except SshTransportError as error:
            if _absent(error):
                raise _Absent(path) from error
            raise
        if not listing:
            raise StepFailed("The machine did not give the digest of {}.".format(path))
        return listing[0]

    def _expect_sha(self, path: str, digest: str) -> None:
        found = self._sha(path)
        if found != digest:
            raise StepFailed("{} on the machine is not what Vaelor sent.".format(path))

    def _backup(self, component: str, path: str) -> Undo:
        item = self.items.get(component) or {}
        if item.get("present") is not True or item.get("type") != "file":
            return Undo(component, path, "file", existed=False)
        sha = self._sha(path)
        self.transport.run(["install", "-m", "0600", "-o", "root", "-g", "root", path,
                            path + PREVIOUS_SUFFIX], sudo=True)
        self._expect_sha(path + PREVIOUS_SUFFIX, sha)
        return Undo(component, path, "file", existed=True, sha256=sha,
                    owner=str(item.get("owner") or "root"), group=str(item.get("group") or "root"),
                    mode=str(item.get("mode") or "0644"))

    def _drop_backups(self, undos: List[Undo]) -> None:
        for undo in undos:
            if undo.kind == "file" and undo.existed:
                try:
                    self.transport.run(["rm", "-f", undo.path + PREVIOUS_SUFFIX], sudo=True)
                except Exception as error:  # noqa: BLE001 - the change stands; the leftover is logged
                    LOGGER.warning("could not remove %s%s: %s", undo.path, PREVIOUS_SUFFIX, error)

    def _restore(self, undo: Undo) -> str:
        """Put one change back and read it back; the sentence for what happened."""
        run = self.transport.run
        if undo.kind == "directory":
            if not undo.existed:
                self._directory_program("rmdir", undo.path)
                return "the folder {} it created was removed again".format(undo.path)
            run(["chown", "{}:{}".format(undo.owner, undo.group), undo.path], sudo=True)
            run(["chmod", undo.mode, undo.path], sudo=True)
            if undo.acl:
                self._directory_program("set_acl", undo.path, undo.acl)
            self._verify_directory(undo.path, undo.owner, undo.group, undo.mode)
            return "{} was put back as it was".format(undo.path)
        if not undo.existed:
            run(["rm", "-f", undo.path], sudo=True)
            if self._present(undo.path):
                raise StepFailed("{} is still there after it was removed.".format(undo.path))
            return "{} was removed again".format(undo.path)
        if self._sha(undo.path) == undo.sha256:
            # The change never landed (an immutable file, a refused write): the
            # file is as it was, and copying it back would fail the same way.
            run(["rm", "-f", undo.path + PREVIOUS_SUFFIX], sudo=True)
            return "{} was not changed, so it stays as it was".format(undo.path)
        run(["install", "-m", undo.mode, "-o", undo.owner, "-g", undo.group,
             undo.path + PREVIOUS_SUFFIX, undo.path], sudo=True)
        self._expect_sha(undo.path, undo.sha256)
        run(["rm", "-f", undo.path + PREVIOUS_SUFFIX], sudo=True)
        return "{} was put back as it was and checked".format(undo.path)

    def _restore_all(self, undos: List[Undo], unit: str = "") -> Dict[str, Any]:
        words, ok, lost = [], True, False
        for undo in reversed(undos):
            try:
                words.append(self._restore(undo))
            except Exception as error:  # noqa: BLE001 - each part says what it holds now
                LOGGER.warning("could not put back %s: %s", undo.path, error)
                ok, lost = False, lost or _connection_lost(error)
                words.append(_not_restored(undo))
        if self._key_minted:
            try:
                self.inputs.restore_key()
            except Exception as error:  # noqa: BLE001 - said, and logged
                LOGGER.warning("could not put the reporting key back for node %s: %s", self.inputs.node_id, error)
                ok = False
                words.append("the telemetry agent's previous reporting key could not be put back, so it "
                             "reports again only after the next successful update")
        if unit:
            ok, lost = self._restart_after_restore(unit, undos, words, ok, lost)
        return {"ok": ok, "lost": lost, "words": words}

    def _restart_after_restore(self, unit, undos, words, ok, lost):
        unit_existed = any(undo.existed for undo in undos if wp.COMPONENTS[undo.component].kind == wp.UNIT)
        try:
            self.transport.run(["systemctl", "daemon-reload"], sudo=True)
            if unit_existed or self._unit_was_running(unit):
                self.transport.run(["systemctl", "restart", unit], sudo=True)
                self._verify_running(unit)
                words.append("{} was started again on its previous files and is running".format(unit))
            else:
                self.transport.run(["systemctl", "disable", "--now", unit], sudo=True)
                words.append("{}, which was not there before, was stopped".format(unit))
        except Exception as error:  # noqa: BLE001 - said, and logged
            LOGGER.warning("could not restart %s after putting it back: %s", unit, error)
            words.append("{} could not be started again after its files were put back".format(unit))
            return False, lost or _connection_lost(error)
        return ok, lost

    def _unit_was_running(self, unit: str) -> bool:
        for component in wp.MANIFEST:
            if component.unit == unit:
                state = (self.items.get(component.id) or {}).get("state") or {}
                return state.get("ActiveState") == "active"
        return False

    def _failed(self, component: str, error: BaseException, undos: List[Undo], unit: str = "") -> ApplyFailed:
        """Put back what this step changed and say which of the three outcomes it is."""
        if isinstance(error, _MACHINE):
            LOGGER.warning("the profile change of %s on node %s failed: %s", component, self.inputs.node_id, error)
        else:
            # A programming error is logged as one, with its traceback, never
            # worded as the machine failing (LESSONS 1 / 5).
            LOGGER.error("internal error changing %s on node %s", component, self.inputs.node_id,
                         exc_info=(type(error), error, error.__traceback__))
        restored = self._restore_all(undos, unit=unit)
        if _connection_lost(error) or restored["lost"]:
            outcome = CONNECTION_LOST
        else:
            outcome = FAILED_RESTORED if restored["ok"] else FAILED_NOT_RESTORED
        return ApplyFailed(component, failure_sentence(error), outcome, restored["words"])


class _Absent(SshTransportError):
    """``sha256sum`` said, in its own words, that the file is not there."""


def _not_restored(undo: Undo) -> str:
    if undo.kind == "directory":
        return ("{} could not be put back; set it by hand with: sudo chown {}:{} {} && sudo chmod {} {}"
                .format(undo.path, undo.owner, undo.group, undo.path, undo.mode, undo.path))
    if not undo.existed:
        return "{} could not be removed again; delete it by hand with: sudo rm -f {}".format(undo.path, undo.path)
    return ("{} could not be put back; what it held is kept in {}{}. Put it back by hand with: sudo install "
            "-m {} -o {} -g {} {}{} {}".format(undo.path, undo.path, PREVIOUS_SUFFIX, undo.mode, undo.owner,
                                              undo.group, undo.path, PREVIOUS_SUFFIX, undo.path))


def failure_sentence(error: BaseException) -> str:
    """The reason in Vaelor's words (review S1). The machine's raw output and any
    exception text stay in the log (LESSONS 24); only Vaelor's own step
    sentences (:class:`StepFailed`) are shown."""
    if _connection_lost(error):
        return "the machine stopped answering"
    if isinstance(error, StepFailed) and str(error).strip():
        return str(error).strip().rstrip(".")
    if isinstance(error, _MACHINE):
        return "the machine refused the change; its answer is in this controller's log"
    return "an internal error stopped the change (logged)"


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _file_sha(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def outcome_message(outcome: str, *, machine: str, labels: Mapping[str, str], applied: List[str],
                    failed: str = "", reason: str = "", undo: Optional[List[str]] = None,
                    planned: Optional[Mapping[str, Any]] = None) -> str:
    """The job's and the card's sentence for an outcome; the backend owns every word (VD-173)."""
    planned = planned or {}
    names = ", ".join(labels.get(component, component) for component in applied if component != "marker")
    tail = ""
    if planned.get("appliance_owned"):
        tail = " The full appliance's own folders on this machine were left as they are."
    if outcome == CURRENT:
        return "Worker software already matches this controller's profile; nothing was changed." + tail
    if outcome == APPLIED:
        return ("Worker software updated ({}). It now matches this controller's profile."
                .format(names or labels.get("marker", "marker")) + tail)
    if outcome == REFUSED:
        missing = ", ".join(labels.get(component, component) for component in planned.get("refused", []))
        return ("{} is missing on {}, so nothing was changed. The profile needs it; install it on the "
                "machine, then press Recheck.".format(missing, machine))
    if outcome == HELD:
        return ("A change to {} would interrupt serving, so it waits until no loaded deployment uses {}."
                .format(", ".join(labels.get(c, c) for c in planned.get("disruptive", [])), machine))
    if outcome == INCOMPLETE:
        unread = ", ".join(labels.get(component, component) for component in planned.get("unread", []))
        done = " Updated: {}.".format(names) if names else ""
        return "Not updated in full; left unchanged because they could not be read: {}.{}".format(
            unread, done)
    label = labels.get(failed, failed)
    put_back = "; ".join(undo or [])
    if outcome in (FAILED_RESTORED, FAILED_NOT_RESTORED):
        if outcome == FAILED_RESTORED:
            put_back = put_back[:1].upper() + put_back[1:] if put_back else "Nothing had changed"
        return "Update failed: {}: {}. {}. Serving was not touched.".format(label, reason, put_back)
    return ("The machine stopped answering while {} was being updated. What it holds now is read again "
            "when it answers, at the next Recheck or 15-minute check. Serving was not touched."
            .format(label))
