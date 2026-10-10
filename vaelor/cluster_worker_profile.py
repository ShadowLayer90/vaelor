"""Read and show each worker's software against its profile (VD-194 P1).

The I/O half of the slim-worker read path, mixed into `ClusterManager` beside
`WorkerTelemetryMixin`. It runs the read-only profile probe over the worker's
enrolled SSH login (`worker_profile_probe`), stores what was read on the node
(`ClusterStore.set_node_profile`), and projects it for the Fleet card and the
``/cluster/worker-software`` route (`worker_profile_state`).

**Nothing here writes to a worker.** The probe is one ``python3 -c`` program
that reads files and ``systemctl show``. When a reading of a joined worker
shows a part the profile owns differing, this queues the
``cluster.node.profile`` job (P2), which the workload executor runs; the
15-minute check (:func:`start_profile_sweep`) does the same, bounded.

**A failed check never erases a reading** (LESSONS 8). The last good reading
stays, beside an ``attempt`` that says when the newest check failed and why;
a machine that did not answer is told apart from one that answered and
refused, because the first is "not reachable" and the second is not.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import worker_profile as wp
from .cluster_node_removal import failure_words
from .cluster_store import NODE_NOT_FOUND
from .runtime_paths import env_value
from .ssh_transport import channel_drops, machine_failures
from .worker_profile_probe import read_profile
from .worker_profile_apply import needs_apply, plan
from .worker_profile_compare import sentence as as_sentence
from .worker_profile_job import (
    RECHECK_REASON, SWEEP_REASON, card_job, latest_profile_job, queue_profile_job, update_note,
)
from .worker_profile_state import (
    NODE_RECORD_EXIT, NODE_RECORD_UNREADABLE, NOT_CHECKED_INTERNAL, NOT_CHECKED_STORE, NOT_ENROLLED,
    NOT_ENROLLED_EXIT, NOT_SAVED, NOT_SHOWN_INTERNAL, NOT_SHOWN_LABEL, STORED_UNREADABLE, age_words,
    appliance_present, appliance_reading, worker_software_view,
)
from .worker_telemetry_runtime import _local_sha256, default_telegraf_resolver

LOGGER = logging.getLogger(__name__)

#: The binary's digest per (tarball, size, change time): reading it means
#: hashing a ~60 MB archive, and the summary asks every 15 seconds.
_BINARY_SHAS: Dict[tuple, str] = {}


#: P1-R4-3: only these are a failed check, worded by `failure_words`. Anything
#: else raised while asking the machine is this controller's own bug: logged
#: with its traceback and shown as an internal error, never as "the command on
#: the machine failed" (LESSONS 1 / 5). One definition, beside the transport
#: (PH-5).
MACHINE_FAILURES = machine_failures()


def _now() -> int:
    return int(time.time())


def telegraf_binary_sha256(architecture: str,
                           resolver: Callable[[str], Any] = default_telegraf_resolver) -> str:
    """The sha256 of the Telegraf binary this controller would ship, or ``""``.

    Read from the staged tarball, and only once that tarball matches its pinned
    checksum - the same check `WorkerTelemetryRuntime._ship_telegraf` makes
    before it ships one. ``""`` means "cannot say", which the comparison
    reports as unread, never as a match.
    """
    try:
        source, pinned, member = resolver(architecture)
        stat = Path(source).stat()
    except (ValueError, OSError):
        return ""
    key = (source, stat.st_size, stat.st_mtime_ns, pinned, member)
    if key not in _BINARY_SHAS:
        try:
            if _local_sha256(source) != pinned:
                return ""
            digest = hashlib.sha256()
            with tarfile.open(source, "r:gz") as archive:
                handle = archive.extractfile(member)
                if handle is None:
                    return ""
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
        except (OSError, tarfile.TarError, KeyError):
            return ""
        _BINARY_SHAS[key] = digest.hexdigest()
    return _BINARY_SHAS[key]


#: How this controller came by what its workers trust (VD-212).
#: ``bundle``: the household authority's trust bundle (the root, plus the
#: certificate still served while an existing install migrates).
#: ``served-certificate``: no bundle yet - an install whose authority has not
#: run - so the served certificate is pinned, as before VD-212, and said so.
#: ``none``: nothing to pin at all.
TRUST_BUNDLE, SERVED_CERTIFICATE, NO_TRUST = "bundle", "served-certificate", "none"

#: The card's words while the authority has produced no bundle (LESSONS 1: the
#: fallback is said, never silent). Shown on the "Household certificate
#: authority" row of every worker's software.
AUTHORITY_NOT_YET = ("This controller's household certificate authority has not produced a trust "
                     "bundle yet, so workers pin the certificate it serves today; they move to the "
                     "authority on the first check after it does")

#: The refusal when an installed controller has nothing a worker could pin: no
#: bundle and no served certificate. Telemetry is then not set up rather than
#: set up without certificate checks (VD-212 fail loud).
NOTHING_TO_PIN = ("This controller has no trust bundle from its household certificate authority "
                  "and no certificate it serves, so worker telemetry is not set up without "
                  "certificate checks; run the installer again or start vaelor-tls-authority")

#: The states already logged, so the 15-second summary does not log every time.
_WARNED: set = set()


@dataclass(frozen=True)
class ControllerTrust:
    """What workers trust: the file, how it was chosen, and the words for the card."""

    path: Optional[str]
    kind: str
    sentence: str = ""
    #: Non-empty only on an installed controller with nothing to pin: the
    #: reason worker telemetry is refused instead of left unverified.
    refusal: str = ""


def _appliance_tls_folder() -> str:
    """The installed TLS folder (``/opt/vaelor/tls``); present on every installed controller."""
    from .runtime_paths import app_path

    return app_path("tls")


def fleet_trust_record(store: Any, worker_ids: List[str], now: Optional[float] = None,
                       last_telemetry: Optional[Callable[[str], Any]] = None) -> Optional[Dict[str, Any]]:
    """Which joined workers hold the current trust bundle, from their stored readings (VD-212).

    The record itself is the household authority's
    (`tls_fleet_gate.fleet_trust_record`, schema ``vaelor-fleet-trust/1``): one
    shape, built by the reader's own module, never hand-rolled here (LESSONS
    6). The authority reads it as ADVICE on when to promote its new console
    certificate, never on what to sign - this account can write it.
    ``bundle_sha256`` is the SHA-256 of `tls_paths.WORKER_TRUST_BUNDLE`'s
    bytes. Per worker: ``matches`` is whether the worker's
    ``/etc/vaelor/controller-ca.pem``, as its last stored reading found it,
    has that digest (absent or unread is ``False``); ``last_reached`` is when
    the worker last answered a check (``None`` if it never did). A worker
    whose newest check failed keeps its older reading and time, so it ages out
    of the authority's 24-hour window instead of blocking it (LESSONS 22).
    ``last_telemetry`` is when the controller last heard from the worker's
    telemetry agent (``last_telemetry(node_id)``: the Fleet card's own rule,
    `WorkerTelemetryMixin._last_report_time`), or ``None`` - so a worker whose
    SSH check fails while its agent still posts counts as online and blocks
    promotion (review F4). ``advertise_address`` is the controller's recorded
    cluster address, the one workers connect to, or ``None`` when it has none;
    the authority refuses to promote a certificate that does not name it (F3).

    ``None`` - write nothing - while there is no bundle: the authority reads a
    missing record as "wait up to 24 hours, then promote".
    """
    from . import tls_paths
    from .tls_fleet_gate import fleet_trust_record as authority_record

    bundle = _file_sha256(tls_paths.WORKER_TRUST_BUNDLE)
    if not bundle:
        return None
    workers = []
    for node_id in worker_ids:
        try:
            record = store.node_profile(node_id) or {}
        except Exception:  # noqa: BLE001 - one unreadable record is one unknown worker
            LOGGER.exception("could not read the software record of node %s for the fleet trust record", node_id)
            record = {}
        reading = record.get("reading") if isinstance(record.get("reading"), dict) else {}
        item = (reading.get("items") or {}).get("controller-ca") if reading.get("ok") else None
        held = ""
        if isinstance(item, dict) and item.get("present") is True and not item.get("error"):
            held = str(item.get("sha256") or "")
        # Answered, even if the reading was incomplete: then it does not match
        # and counts as reached without the bundle - the cautious side.
        reached = record.get("checked_at")
        workers.append({"id": str(node_id), "matches": held == bundle,
                        "last_reached": reached if isinstance(reached, (int, float)) else None,
                        "last_telemetry": _telemetry_time(last_telemetry, node_id)})
    return authority_record(bundle, workers, now, advertise_address=_advertise_address(store))


def _telemetry_time(reader: Optional[Callable[[str], Any]], node_id: str) -> Optional[float]:
    """Epoch seconds of the worker's newest telemetry, or ``None`` when unknown or unreadable."""
    if reader is None:
        return None
    try:
        value = reader(node_id)
    except Exception:  # noqa: BLE001 - retention off or a store error: unknown, logged
        LOGGER.warning("could not read when node %s last sent telemetry", node_id, exc_info=True)
        return None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _advertise_address(store: Any) -> Optional[str]:
    """The controller's recorded cluster address, or ``None`` when it has none or cannot be read."""
    try:
        address = str((store.controller() or {}).get("advertise_address") or "").strip()
    except Exception:  # noqa: BLE001 - unreadable: no address, logged
        LOGGER.warning("could not read the controller's cluster address", exc_info=True)
        return None
    return address or None


def write_fleet_trust(path: str, record: Mapping[str, Any]) -> None:
    """Replace the record atomically: a temporary file beside it, flushed, then renamed in.

    The reader never sees half a file. 0640: no secret in it, but nothing
    beyond the control plane and the authority needs it.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    handle, temporary = tempfile.mkstemp(prefix=".fleet-trust.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True, indent=1)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, str(target))
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _log_once(state: str, level: int, message: str, *args: Any) -> None:
    """Log a change of trust state once, not on every 15-second summary."""
    if state not in _WARNED:
        _WARNED.clear()
        _WARNED.add(state)
        LOGGER.log(level, message, *args)


def controller_trust() -> ControllerTrust:
    """What a worker pins, and how it was chosen - the one answer (LESSONS 6).

    The household authority's bundle (`tls_paths.WORKER_TRUST_BUNDLE`) when it
    exists. Without one - an install whose authority has not run yet (VD-212
    migration) - the served certificate (``VAELOR_TLS_CERT``, else the
    installer's ``tls/vaelor.crt``), as before VD-212, logged once and carried
    to the card as :data:`AUTHORITY_NOT_YET` (LESSONS 1). With neither, nothing
    is pinned; on an installed controller that is :data:`NOTHING_TO_PIN`, a
    reported refusal, and only a development box keeps the explicit insecure
    opt-in.

    Five readers go through here: the expected profile's digest
    (:func:`controller_inputs`), the profile apply job, the telemetry install,
    its reconcile and the reconcile's stale-CA check (via
    :func:`controller_ca_source`), plus the custom agents' pin
    (`agent_memory.controller_certificate_pem`). The profile job runs in the
    workload executor, whose unit does not carry the control plane's TLS
    environment, so every path here is a file both processes read.
    """
    from . import tls_paths

    bundle = tls_paths.WORKER_TRUST_BUNDLE
    if bundle and Path(bundle).is_file():
        _log_once(TRUST_BUNDLE, logging.INFO, "workers trust the household authority's bundle %s", bundle)
        return ControllerTrust(bundle, TRUST_BUNDLE)
    served = tls_paths.leaf_cert()
    if served and Path(served).is_file():
        _log_once(SERVED_CERTIFICATE, logging.WARNING,
                  "no trust bundle from the household authority at %s yet; workers pin the served "
                  "certificate %s until it exists", bundle, served)
        return ControllerTrust(served, SERVED_CERTIFICATE, AUTHORITY_NOT_YET)
    installed = Path(_appliance_tls_folder()).is_dir() or Path(tls_paths.AUTHORITY_DIR).is_dir()
    refusal = NOTHING_TO_PIN if installed else ""
    _log_once(NO_TRUST, logging.ERROR if installed else logging.WARNING,
              "no trust bundle (%s) and no served certificate (%s): %s", bundle, served,
              "worker telemetry is refused" if installed else "a development box; its insecure opt-in stands")
    return ControllerTrust(None, NO_TRUST, refusal, refusal)


def controller_ca_source() -> Optional[str]:
    """The file a worker pins (:func:`controller_trust`), or ``None`` when there is none."""
    return controller_trust().path


def household_root_source() -> Optional[str]:
    """The authority's public root (`tls_paths.PUBLIC_ROOT`) for a worker's OS trust store.

    The 0644 public copy, readable by the control plane and the executor
    alike; the authority's own ``root.crt`` sits in a folder only root reads.
    ``None`` before the authority has run.
    """
    from . import tls_paths

    path = tls_paths.PUBLIC_ROOT
    return path if path and Path(path).is_file() else None


def controller_ingest_url(store: Any) -> str:
    """The keyed ingest URL a worker's agent posts to; ``ValueError`` without an address."""
    from .worker_telemetry_config import ingest_url

    controller = store.controller()
    address = str(controller.get("advertise_address") or controller.get("candidate_address") or "").strip()
    return ingest_url(address, int(env_value("VAELOR_PORT", "PM_DASHBOARD_PORT", "34001")))


def controller_inputs(store: Any, architecture: str) -> Dict[str, str]:
    """What this controller contributes to a worker's expected profile: one
    derivation for the card (control plane) and the apply job (executor).

    ``controller_ca_refusal`` is :data:`NOTHING_TO_PIN` on an installed
    controller with nothing to pin, so the profile reads the certificate and
    the telemetry settings as not comparable rather than expecting settings
    without certificate checks (VD-212 fail loud). It names no file and is
    not hashed into the digest.
    """
    try:
        url = controller_ingest_url(store)
    except ValueError:
        url = ""
    trust = controller_trust()
    root = household_root_source()
    return {"ingest_url": url, "controller_ca_sha256": _file_sha256(trust.path) if trust.path else "",
            "controller_ca_refusal": trust.refusal,
            # No root on this controller yet: the component does not apply,
            # decided by the file's presence, not by a digest helper.
            "household_root_sha256": _file_sha256(root) if root else "",
            "telegraf_binary_sha256": telegraf_binary_sha256(architecture)}


def _file_sha256(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        return _local_sha256(path)
    except OSError:
        return ""


#: The note on a row whose part the stored reading never looked at (below).
NOT_IN_OLDER_READING = ("This reading was taken before this controller's profile had this part, so the "
                        "check did not look for it and Vaelor had not placed it; the next check reads it")


def with_later_components(record: Mapping[str, Any]) -> tuple:
    """A stored reading from before a component existed, read as "not placed yet".

    The probe is shipped by the controller on every check, so a check after an
    upgrade reports every component; only a reading STORED before the upgrade
    lacks the newer ones. Unchanged, those rows read "could not be read" and
    turn an otherwise current worker Unknown until the next check. Vaelor never
    placed a part its profile did not yet have, so it reads as not on the
    machine - behind, an update due - with :data:`NOT_IN_OLDER_READING` on the
    row saying it was not looked at. Only `worker_profile.LATER_COMPONENTS`, and
    only a reading that did answer: a failed item is still unread.
    Returns ``(record, ids filled in)``.
    """
    reading = record.get("reading") if isinstance(record.get("reading"), dict) else None
    if not reading or not reading.get("ok", True) or not isinstance(reading.get("items"), dict):
        return record, []
    missing = [ident for ident in wp.LATER_COMPONENTS if ident not in reading["items"]]
    if not missing:
        return record, []
    items = dict(reading["items"])
    for ident in missing:
        items[ident] = {"present": False}
    return {**record, "reading": {**reading, "items": items}}, missing


def _note_rows(view: Dict[str, Any], ids, words: str) -> Dict[str, Any]:
    note = as_sentence(words)
    for row in view.get("components") or []:
        if row.get("id") in ids:
            row["note"] = " ".join(filter(None, (row.get("note"), note)))
            row["words"] = " ".join(filter(None, (row.get("words"), note)))
    return view


def with_trust_note(view: Dict[str, Any], trust: ControllerTrust) -> Dict[str, Any]:
    """Say on the "Household certificate authority" row what workers pin when it is not the bundle.

    LESSONS 1: before the authority has run, the row would otherwise read
    "Up to date" while the worker pins the served certificate - true of the
    file, silent about the fallback. The sentence joins the row's note and its
    ``words`` (the whole sentence the card draws, VD-173).
    """
    if not trust.sentence:
        return view
    return _note_rows(view, ("controller-ca",), trust.sentence)


def unknown_view(node_id: str, sentence: str, label: str = NOT_SHOWN_LABEL, read_now: bool = False,
                 exits: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The card's view when no reading can be shown, in fixed honest words (LESSONS 8).

    Round 4 N3: never a fabricated failed attempt - the stored record may say
    the check succeeded - and never raw exception text (LESSONS 24); the cause
    is in the log. ``read_now`` dates it when the machine was in fact read.
    ``exits`` replaces the unknown state's "Recheck." where a Recheck cannot
    leave this view (P1-R4-5, LESSONS 22).
    """
    unshown: Dict[str, Any] = {"label": label, "sentence": sentence}
    if read_now:
        unshown["at"] = _now()
    if exits is not None:
        unshown["exit_words"] = dict(exits)
    return worker_software_view(node_id=str(node_id), record={"unshown": unshown},
                                expected={"components": [], "digest": ""}, release_digest="", now=_now())


class WorkerProfileMixin:
    """The worker-software read path. Needs ``store``, ``_worker_transport``,
    ``_controller_ingest_url`` and ``wants_gpu_sampler`` from its host class.

    ``job_store`` is the job queue the control plane shares with the executor
    (set by `control_plane_runtime`); with it, a Recheck or the 15-minute check
    that finds a part the profile owns differing queues the apply job, and the
    card shows the newest job. Without it (a test, a preview), nothing is queued.
    """

    store: Any
    job_store: Any = None
    #: Where the sweep writes the fleet-trust record (VD-212); set by
    #: :func:`start_profile_sweep`, so a test or preview writes nothing.
    fleet_trust_path: Optional[str] = None

    def worker_profile_inputs(self, node: Mapping[str, Any]) -> Dict[str, str]:
        """What the controller itself contributes to a worker's expected profile.

        The same derivation the apply job uses in the executor
        (:func:`controller_inputs`), so the card and the job expect one profile.
        """
        return controller_inputs(self.store, str((node.get("inventory") or {}).get("architecture", "")))

    def _profile_attempt_failed(self, node_id: str, error: BaseException) -> Dict[str, Any]:
        """Record a check that failed beside the last reading, which is kept.

        Returns the record even when the store cannot be read or written (round
        2 B1): that failure becomes part of the reason, never a raised error.
        """
        reason = failure_words(error)
        try:
            record = dict(self.store.node_profile(node_id))
        except Exception:  # noqa: BLE001 - said in the reason, the cause logged
            # Round 3 R3-3: without the record there is nothing to write the
            # attempt beside, and writing it alone would erase the last good
            # reading. Write nothing; say so.
            LOGGER.exception("could not read back the software record of node %s", node_id)
            return {"attempt": {
                "at": _now(), "ok": False, "unreachable": isinstance(error, channel_drops()),
                "reason": reason + "; the last reading could not be read back (logged), so nothing "
                                   "was recorded",
            }}
        record.pop("unreadable", None)
        record["attempt"] = {
            "at": _now(), "ok": False,
            # A machine that did not answer, as opposed to one that answered
            # and refused: the channel itself failed (review S4).
            "unreachable": isinstance(error, channel_drops()),
            "reason": reason,
        }
        try:
            self.store.set_node_profile(node_id, record)
        except Exception:  # noqa: BLE001 - said in the reason, the cause logged
            LOGGER.exception("could not save the failed software check of node %s", node_id)
            record["attempt"]["reason"] = reason + "; and the result could not be recorded (logged)"
        return record

    def note_worker_profile_unreached(self, node_id: str, error: BaseException) -> None:
        """A Recheck whose hardware re-read already failed: record it for the profile too.

        Only a machine or transport failure is recorded (P1-R4-3). Any other
        error - a node that is gone, a bug - is not a failed check of the
        machine; the Recheck raises and reports it itself.
        """
        if not isinstance(error, MACHINE_FAILURES):
            LOGGER.warning("the Recheck of node %s stopped on %s, which is not a failure of the "
                           "machine; no failed software check is recorded", node_id, type(error).__name__)
            return
        try:
            self._profile_attempt_failed(node_id, error)
        except Exception as failure:  # noqa: BLE001 - the re-read's own error is the one raised
            LOGGER.warning("could not record the failed check of node %s: %s", node_id, failure)

    def recheck_worker_profile(self, node_id: str, reason: str = RECHECK_REASON,
                               bounded: bool = False) -> Dict[str, Any]:
        """Read one worker's software now, store it, queue an update if one is
        needed (P2), and return the card's view.

        Raises only ``ValueError(NODE_NOT_FOUND)`` for a node that is not
        enrolled; every other failure is in the view it returns. ``bounded`` is
        the 15-minute check's limit on retrying one change (`worker_profile_job`).
        """
        node = self.store.get_node(node_id, include_credential=True)
        if node is None:
            raise ValueError(NODE_NOT_FOUND)
        try:
            transport = self._worker_transport(node)  # type: ignore[attr-defined]
            reading = read_profile(transport, node_id)
        except MACHINE_FAILURES as error:
            record = self._profile_attempt_failed(node_id, error)
        except Exception:  # noqa: BLE001 - P1-R4-3: a bug here is not a failed check
            # Nothing is recorded: the machine was never asked, so the last
            # reading and its attempt stand as they were (LESSONS 5 / 8).
            LOGGER.exception("internal error asking node %s for its worker software", node_id)
            return unknown_view(node_id, NOT_CHECKED_INTERNAL)
        else:
            facts = dict(reading.get("facts") or {})
            try:
                facts["gpu_sampler"] = bool(self.wants_gpu_sampler(node))  # type: ignore[attr-defined]
            except Exception as error:  # noqa: BLE001 - unread, said as such by the predicate
                LOGGER.warning("could not decide the GPU sampler for node %s: %s", node_id, error)
                facts["gpu_sampler"] = None
            reading["facts"] = facts
            now = _now()
            record = {"checked_at": now, "reading": reading, "attempt": {"at": now, "ok": True}}
            try:
                kept = self.store.node_profile(node_id).get("amd_smi")
                if isinstance(kept, dict):
                    # What the last apply installed stays recorded (design §2.5).
                    record["amd_smi"] = kept
                self.store.set_node_profile(node_id, record)
            except Exception:  # noqa: BLE001 - an unsaved reading is not shown as current
                LOGGER.exception("could not save the software reading of node %s", node_id)
                return unknown_view(node_id, NOT_SAVED, label="Not saved", read_now=True)
            self._queue_if_needed(node, record, reason, bounded)
        # Never raises (review P1-3): Recheck has already re-read the hardware
        # and repaired telemetry by now, and a 500 here would lose both reports.
        return self.worker_software_safe(node, record)

    def _queue_if_needed(self, node: Mapping[str, Any], record: Mapping[str, Any], reason: str,
                         bounded: bool) -> None:
        """Queue the apply job when a part the profile owns differs (P2).

        The same plan the job itself makes (`worker_profile_apply.plan`), so
        "an update is needed" has one answer (LESSONS 6). Never raises: the
        reading stands whether or not a job could be queued.
        """
        if self.job_store is None or not (node.get("labels") or {}).get("swarm_node_id"):
            # Only a joined worker: the join itself queues the first apply.
            return
        try:
            reading = record["reading"]
            expected = wp.node_expected(node.get("inventory") or {}, self.worker_profile_inputs(node),
                                        reading.get("facts"))
            appliance = appliance_reading(reading.get("appliance"))
            if appliance["reading"] == "unread":
                return
            provisioned = self.telemetry_provisioned(node["id"])  # type: ignore[attr-defined]
            planned = plan(expected, reading, appliance_present(appliance),
                           amd_smi_record=record.get("amd_smi"), key_provisioned=provisioned)
        except Exception:  # noqa: BLE001 - a bug here is logged, the reading stands
            LOGGER.exception("could not decide whether node %s needs a profile update", node["id"])
            return
        if needs_apply(planned):
            queue_profile_job(self.job_store, node["id"], reason,
                              expected_digest=expected["digest"], bounded=bounded)

    def sweep_worker_profile(self, node_id: str) -> Dict[str, Any]:
        """The 15-minute check (design §3a.5): read, and queue a bounded update if one is needed.

        A worker that does not answer gets a failed, unreachable check and no
        job; the next sweep that finds it answering queues one. Each pass also
        rewrites the fleet-trust record the household authority reads (VD-212,
        :meth:`record_fleet_trust`), whatever the reading's outcome.
        """
        try:
            return self.recheck_worker_profile(node_id, reason=SWEEP_REASON, bounded=True)
        finally:
            self.record_fleet_trust()

    def record_fleet_trust(self) -> Optional[Dict[str, Any]]:
        """Write :func:`fleet_trust_record` to ``fleet_trust_path``; never raises.

        Rewritten on every sweep pass (the authority ignores a record older
        than two hours). Built from the stored readings alone
        (`ClusterStore.node_profile`), so a failed check keeps the worker's last
        good reading and its time, and the record never invents a reach
        (LESSONS 8). ``None`` when nothing was written: no path (a test, a
        preview), no bundle yet, or a failure, which is logged.
        """
        path = self.fleet_trust_path
        if not path:
            return None
        try:
            record = fleet_trust_record(self.store, self.profile_sweep_workers(),
                                        last_telemetry=getattr(self, "_last_report_time", None))
            if record is not None:
                write_fleet_trust(path, record)
        except Exception:  # noqa: BLE001 - advisory: the sweep goes on, the cause is logged
            LOGGER.exception("could not write the fleet trust record %s", path)
            return None
        return record

    def profile_sweep_pass(self) -> List[str]:
        """The start of one 15-minute pass: the fleet-trust record, then the workers to read.

        The record is written here once per pass, whatever the number of
        joined workers - with none, it says so (``workers: []``) and the
        household authority may promote at once rather than wait 24 hours for
        a record that would never come (review R18). Each worker's check then
        rewrites it with that worker's fresh reading.
        """
        self.record_fleet_trust()
        return self.profile_sweep_workers()

    def profile_sweep_workers(self) -> List[str]:
        """The joined workers the 15-minute check reads, in enrolment order."""
        # Review S7: one corrupt row is skipped (and logged by the store), never
        # every worker with it.
        nodes = self.store.list_nodes_tolerant()
        return [node["id"] for node in nodes if not node.get("record_unreadable")
                and (node.get("labels") or {}).get("swarm_node_id")]

    def _card_job(self, node_id: str) -> Optional[Dict[str, Any]]:
        if self.job_store is None:
            return None
        try:
            return card_job(latest_profile_job(self.job_store, node_id))
        except Exception:  # noqa: BLE001 - the card shows the reading without the job
            LOGGER.exception("could not read the profile jobs of node %s", node_id)
            return None

    def recheck_worker_profile_contained(self, node_id: str) -> Dict[str, Any]:
        """:meth:`recheck_worker_profile` for the hardware Recheck: never raises (round 2 B1).

        That Recheck has already re-read the machine and repaired its telemetry
        by the time it gets here, so a node removed in between, or a store that
        cannot be read, must still leave it to report those - with an unknown
        view that says why.
        """
        try:
            return self.recheck_worker_profile(node_id)
        except ValueError as error:
            if str(error) != NODE_NOT_FOUND:
                LOGGER.exception("could not check the worker software of node %s", node_id)
                return unknown_view(node_id, NOT_CHECKED_INTERNAL)
            LOGGER.warning("node %s left the fleet during its Recheck", node_id)
            return unknown_view(node_id, NOT_ENROLLED, label="Not enrolled", exits=NOT_ENROLLED_EXIT)
        except (OSError, sqlite3.Error):
            LOGGER.exception("could not read the software records of node %s", node_id)
            return unknown_view(node_id, NOT_CHECKED_STORE)
        except Exception:  # noqa: BLE001 - round 4 N3: a programming error, logged as one
            LOGGER.exception("internal error checking the worker software of node %s", node_id)
            return unknown_view(node_id, NOT_CHECKED_INTERNAL)

    def worker_software_for(self, node: Mapping[str, Any],
                            record: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """The card's "Worker software" view for one worker, from its stored record."""
        if record is None:
            record = self.store.node_profile(node["id"])
        if record.get("unreadable"):
            LOGGER.warning("the stored software record of node %s is unreadable: %s", node["id"],
                           record["unreadable"])
            return unknown_view(node["id"], STORED_UNREADABLE)
        record, filled = with_later_components(record)
        reading = record.get("reading") if isinstance(record.get("reading"), dict) else {}
        expected = wp.node_expected(node.get("inventory") or {}, self.worker_profile_inputs(node),
                                    reading.get("facts"))
        # Owner check 2026-10-04 (LESSONS 8): the release code is one labelled
        # line under the row's disclosure; when it cannot be worked out, that
        # line says so and the reading is still shown, as the list route does.
        try:
            release_digest = wp.release_profile_digest()
        except OSError:
            LOGGER.exception("the release profile code for node %s's card could not be worked out", node["id"])
            release_digest = ""
        job = self._card_job(node["id"])
        now = _now()
        view = worker_software_view(node_id=node["id"], record=record, expected=expected,
                                    release_digest=release_digest, now=now, job=job,
                                    update_note=update_note(job, lambda at: age_words(at, now)))
        applying = {entry["id"] for entry in expected["components"] if entry["applies"]}
        view = _note_rows(view, [ident for ident in filled if ident in applying], NOT_IN_OLDER_READING)
        return with_trust_note(view, controller_trust())

    def worker_software_safe(self, node: Mapping[str, Any],
                             record: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """:meth:`worker_software_for`, or an ``unknown`` view that says why it could not be shown."""
        if node.get("record_unreadable"):
            # PH-R1: listed by `ClusterStore.list_nodes_tolerant`, logged there.
            return unknown_view(str(node.get("id")), NODE_RECORD_UNREADABLE, exits=NODE_RECORD_EXIT)
        try:
            return self.worker_software_for(node, record)
        except Exception:  # noqa: BLE001 - round 4 N3: shown honestly, logged with its traceback
            LOGGER.exception("internal error showing the worker software of node %s", node.get("id"))
            return unknown_view(str(node.get("id")), NOT_SHOWN_INTERNAL)

    def worker_software_all(self) -> List[Dict[str, Any]]:
        """Every enrolled worker's view, in enrolment order.

        P1-R4-1 (LESSONS 8): a node row whose own fields will not decode fails
        the whole listing, so the list is then built a node at a time: every
        readable worker in enrolment order, then each unreadable one as an
        unknown view of its own - never a 500 for the whole fleet.
        """
        records = self.store.node_profiles()
        try:
            nodes = self.store.list_nodes()
        except ValueError:  # json's decode errors are ValueErrors
            LOGGER.exception("a node record will not decode; listing the workers' software one by one")
            return self._worker_software_one_by_one(records)
        return [self.worker_software_safe(node, records.get(node["id"], {})) for node in nodes]

    def _worker_software_one_by_one(self, records: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
        readable, unreadable = [], []
        for node_id in records:
            try:
                node = self.store.get_node(node_id)
            except ValueError:
                LOGGER.exception("the enrolment record of node %s will not decode", node_id)
                unreadable.append(unknown_view(node_id, NODE_RECORD_UNREADABLE, exits=NODE_RECORD_EXIT))
                continue
            if node is not None:
                readable.append(node)
        readable.sort(key=lambda node: node.get("created_at") or 0)
        return [self.worker_software_safe(node, records.get(node["id"], {})) for node in readable] + unreadable


def start_profile_sweep(manager: Any, job_store: Any):
    """Give the manager the shared job queue and start the 15-minute profile check.

    Its own daemon loop, on the telemetry repair's cadence and machinery
    (`WorkerTelemetryReconcileScheduler`): each joined worker is read in turn,
    one failure never stops the rest, and a worker that differs gets one
    bounded update queued (design §3a.5). Wired from `control_plane_runtime`,
    which is near its line ceiling.
    """
    from .worker_telemetry_reconcile_scheduler import WorkerTelemetryReconcileScheduler

    from .tls_paths import FLEET_TRUST_STATE

    manager.job_store = job_store
    # VD-212: the household authority promotes its new console certificate
    # only once the workers reached recently hold the bundle; this is how it knows.
    manager.fleet_trust_path = FLEET_TRUST_STATE
    scheduler = WorkerTelemetryReconcileScheduler(
        list_workers=manager.profile_sweep_pass, reconcile=manager.sweep_worker_profile)
    scheduler.start()
    return scheduler
